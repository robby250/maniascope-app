#!/usr/bin/env python3
"""Focused checks for the recommender (plan §11): pp + ledger, identity/import, session/selection, refresh.
Run one: python3 test_recommend.py test_ledger_cases"""
import json
import os
import sys
import tarfile
import tempfile
import time

import numpy as np

import pytest
import recdata
import recommend as R

CHART = os.path.join(recdata.FROZEN, "osu", "282336.osu")


def _db():
    return recdata.connect(os.path.join(tempfile.mkdtemp(), "rec.db"))


def test_start_keeps_displayed_difficulty_with_the_actual_rate(tmp_path):
    from types import SimpleNamespace
    db = recdata.connect(str(tmp_path/'rec.db'))
    sha = 'a'*64
    db.execute("INSERT INTO installed(sha256,artist,title,version,md5,keys,length) VALUES (?,?,?,?,?,?,?)",
               (sha, 'Artist', 'Chart', 'Test', 'b'*32, 7, 120000))
    f = dict(keys=7, overall=6., preunit_overall=8.)
    expected = dict(f=f, acc_mid=.94, md5='b'*32)
    worker = SimpleNamespace(db=db, predictor_id='test', target=None, playlists={m:[] for m in R.MODES},
        rec=SimpleNamespace(pub=dict(calc='test', snapshot='test'), expectation=lambda *a, **kw: expected))
    play = dict(sha=sha, path='cached.osu', rate=.87, mods=[], t=100.)
    R.Worker.t_start(worker, play)
    saved = json.loads(db.execute("SELECT info FROM events ORDER BY id DESC LIMIT 1").fetchone()[0])
    assert saved['difficulty'] == 7. and saved['expected'] == .94 and saved['rate'] == .87
    # A later recalibration cannot change the stored displayed value.
    f['preunit_overall'] = 12.
    assert json.loads(db.execute("SELECT info FROM events LIMIT 1").fetchone()[0])['difficulty'] == 7.
    expected.clear()  # An uncached changed rate has no pre-play estimate.
    R.Worker.t_start(worker, dict(play, rate=.78, t=101.))
    missing = json.loads(db.execute("SELECT info FROM events ORDER BY id DESC LIMIT 1").fetchone()[0])
    assert missing['difficulty'] is None and missing['expected'] is None and missing['rate'] == .78
    db.close()


# ---- pp + ledger ------------------------------------------------------------
def test_pp_formula_matches_rosu():
    import rosu_pp_py as rp
    if not os.path.exists(CHART):
        return print("skip: frozen chart missing")
    bm = rp.Beatmap(path=CHART)
    for mods in (0, 64, 256):
        d = rp.Difficulty(mods=mods, lazer=True).calculate(bm)
        n = d.n_objects + d.n_hold_notes
        stats = (n - 60, 30, 10, 10, 5, 5)
        acc, hits = recdata.acc320(stats)
        pp = rp.Performance(mods=mods, lazer=True, n_geki=stats[0], n300=stats[1], n_katu=stats[2], n100=stats[3],
                            n50=stats[4], misses=stats[5]).calculate(d).pp
        assert hits == n and abs(float(R.pp_at(d.stars, n, acc)) - pp) < 1e-6, (mods, pp)


def _brute(best):
    P = sorted((v[0] for v in best.values()), reverse=True)
    return sum(p * 0.95 ** i for i, p in enumerate(P)) + R.bonus(len(P))


def test_ledger_cases():
    rng = np.random.default_rng(3)
    best = {b: (float(p), "x", None) for b, p in enumerate(rng.uniform(50, 600, 300))}
    L = R.Ledger(best)
    assert abs(L.total() - _brute(best)) < 1e-6
    b0 = max(best, key=lambda b: best[b][0] if best[b][0] < 300 else 0)    # a mid-table map
    for new, what in ((best[b0][0] + 1, "refarm, one step"), (590.0, "refarm crossing many positions"),
                      (best[b0][0] - 20, "worse result"), (0.0, "failure")):
        exp = _brute({**best, b0: (max(new, best[b0][0]), "x", None)}) - _brute(best)
        got = float(L.delta(b0, [new])[0])
        assert abs(got - exp) < 1e-6, (what, got, exp)
    for new in (10.0, 250.0, 700.0):                                        # a new map, incl. bonus step
        exp = _brute({**best, 9999: (new, "x", None)}) - _brute(best)
        assert abs(float(L.delta(9999, [new])[0]) - exp) < 1e-6
    # >99 % refarm: a small acc gain near the ceiling is still a real (small) gain
    assert L.delta(b0, [best[b0][0] * 1.004])[0] > 0


def test_mod_alternatives_compete():
    """NM/HT/DT on one difficulty replace the same counted score: never added twice."""
    best = {1: (300.0, "x", None), 2: (200.0, "x", None)}
    L = R.Ledger(best)
    both = L.delta_many(np.array([300.0, 300.0]), np.array([[320.0], [330.0]]))
    assert abs(both[1, 0] - (_brute({1: (330.0, "", 0), 2: best[2]}) - _brute(best))) < 1e-9
    assert both[1, 0] < both[0, 0] + both[1, 0]        # the variants are alternatives, not a sum


def test_ranked_variants():
    assert recdata.variant([{"acronym": "DT", "settings": {"speed_change": 1.2}}]) == (1.2, False)
    assert recdata.variant([{"acronym": "DT", "settings": {"adjust_pitch": True}}]) == (1.5, True)
    assert recdata.variant([{"acronym": "HT"}, {"acronym": "NF"}]) == (0.75, True)
    assert recdata.variant([{"acronym": "CS"}]) == (1.0, False)
    assert recdata.variant([{"acronym": "WU"}]) == (None, False)


# ---- identity / import --------------------------------------------------------
def _raw(i, stats, total, online=0, user=2653437, date="2026-09-20T10:00:00Z"):
    return {"id": f"uuid-{i}", "hash": "a" * 64, "acc": 0.95, "total": total, "combo": 100, "rank": 3, "date": date,
            "mods": json.dumps([{"acronym": "HT"}]), "stats": json.dumps(dict(zip(
                ("perfect", "great", "good", "ok", "meh", "miss"), stats))), "user": "robby250", "user_id": user,
            "online_id": online, "legacy_online_id": -1, "legacy": False, "client": "2026.921.0-lazer",
            "beatmap_id": 42, "md5": "m" * 32, "status": 1}


def test_import_idempotent_and_live_then_realm():
    db = _db()
    stats = (900, 50, 20, 10, 5, 15)
    live = {"sha256": "a" * 64, "score": 777, "hits": dict(zip(("geki", "300", "katu", "100", "50", "0"), stats)),
            "mods_list": [{"acronym": "HT"}], "played": "2026-09-20T10:00:00Z"}
    k1, new = recdata.add_live_score(db, live, "/x", fresh=True)
    assert new
    rows, _m = recdata.realm_rows([_raw(1, stats, 777, online=5), _raw(2, stats, 900, user=999)], 2653437)
    assert len(rows) == 1                               # another account's score is not ours
    for _ in range(2):
        with db:
            for r in rows:
                recdata.upsert_score(db, r)
    got = db.execute("SELECT key, online_id, fresh, local_id FROM scores").fetchall()
    assert len(got) == 1 and got[0]["online_id"] == 5 and got[0]["fresh"] == 1 and got[0]["local_id"] == "uuid-1"
    # the same result polled again later is not a second attempt
    assert recdata.add_live_score(db, live, "/x", fresh=False)[1] is False


def test_old_result_is_history_not_form():
    """A viewed old result: its start is voided, so the session has no attempt and no residual."""
    t = time.time()
    ev = [{"t": t - 60, "kind": "start", "beatmap": "42", "info": {"mu": -3.0, "sd": 0.4, "keys": 7}},
          {"t": t - 50, "kind": "void", "beatmap": "42", "info": {}},
          {"t": t - 40, "kind": "finish", "beatmap": "42", "info": {"y": -5.0}}]
    s = R.Session(ev, t)
    assert not s.attempts and s.form(7, None) == 0.0


# ---- session / selection --------------------------------------------------------
def _attempts(t0, zs, keys=7, skill="stream", ranked=True, purpose="farm", bid=None, spacing=300):
    ev = []
    for i, z in enumerate(zs):
        t = t0 + i * spacing
        b = str(bid or 100 + i)
        ev.append({"t": t, "kind": "start", "beatmap": b, "info": {"mu": -3.0, "sd": 0.4, "keys": keys, "skill": skill,
                                                                    "ranked": ranked, "purpose": purpose, "length": 120}})
        ev.append({"t": t + 120, "kind": "finish", "beatmap": b, "info": {"y": -3.0 - 0.4 * z}})
    return ev


def test_session_ramp_and_recovery():
    t0 = time.time() - 4000
    assert R.Session([], t0).phase() == "warmup"
    s = R.Session(_attempts(t0, [0.2, 0.3], purpose="warmup"), t0 + 700)
    assert s.phase() == "warmup"                                 # two warmups, nothing strong yet
    s = R.Session(_attempts(t0, [0.2, 0.3, 0.8, 0.6, 0.7, 0.9]), t0 + 2000)
    assert s.phase() == "build"          # strong but no beaten PP best: the staircase pushes only on beats
    one_bad = R.Session(_attempts(t0, [0.8, 0.6, 0.7, 0.9, 0.5, -2.0]), t0 + 2000)
    assert one_bad.phase() == "recover" and one_bad.form(7, "stream") < 0       # severe underperformance acts promptly
    s = R.Session(_attempts(t0, [0.5, 0.4, 0.3, 0.2, -1.5, -1.8]), t0 + 2000)
    assert s.phase() == "recover"


def test_keymode_form_does_not_transfer_fully():
    t0 = time.time() - 3000
    s = R.Session(_attempts(t0, [1.5, 1.4, 1.6, 1.5, 1.3]), t0 + 1600)
    assert s.form(7, "stream") > s.form(4, "stream") > 0         # broad form carries a little, not 7K's peak
    s2 = R.Session(_attempts(t0, [-1.0, -1.2, -0.9, -1.1], skill="jack"), t0 + 1300)
    assert s2.form(7, "ln") > s2.form(7, "jack")                 # bad jack results do not erase LN


def test_session_boundary_and_restart():
    t0 = time.time() - 10000
    ev = _attempts(t0, [0.5, 0.6])
    assert len(R.Session(ev, t0 + 900).attempts) == 2           # an app restart rebuilds the same session
    assert R.Session(ev, t0 + 900 + recommend_gap()).attempts == []   # an hour without gameplay: new session
    ev2 = ev + [{"t": t0 + 1000, "kind": "reset", "beatmap": None, "info": {}}]
    assert R.Session(ev2, t0 + 1001).attempts == []


def recommend_gap():
    return R.SESSION_GAP + 60


def test_recent_maps_stay_back_after_restart_and_return_after_variety():
    t0 = time.time() - 1000
    ev = _attempts(t0, [-0.2], bid=7)                            # near miss on difficulty 7
    s = R.Session(ev, t0 + 200)
    assert s.freshness(7, 540) < 0.1                             # 9-minute farm: not immediately
    assert s.freshness(7, 100) < 0.1                             # Next is variety; manual retry is still available
    assert R.Session(ev,t0+200+recommend_gap()).freshness(7,100) < .15
    ev += _attempts(t0 + 200, [0.1, 0.0, 0.2], spacing=200)      # three other maps in between
    s = R.Session(ev, t0 + 900)
    assert s.freshness(7, 540) > .2


def test_offered_chart_returns_after_a_share_of_the_pool_not_a_clock():
    t = time.time()
    offer = lambda dt, b: {"t": t - dt, "kind": "offer", "beatmap": b, "info": {"mode": "nps"}}
    ev = [offer(5000, "9")] + [offer(4000 - i, str(100 + i)) for i in range(10)]
    s = R.Session(ev, t)
    assert s.freshness(9, 120, mode="nps", pool=400) < .01         # 10 of 400: still held back, hours later
    assert .2 < s.freshness(9, 120, mode="nps", pool=20) < .3      # half the pool since: a quarter back
    assert s.freshness(9, 120, mode="pp", pool=40) < 1e-3           # offers in another tab do not count


def test_skip_is_not_permanent_and_render_is_not_exposure():
    t = time.time()
    ev = [{"t": t - 30, "kind": "start", "beatmap": "1", "info": {}}, {"t": t - 20, "kind": "skip", "beatmap": "9", "info": {}}]
    s = R.Session(ev, t)
    assert s.freshness(9, 120) < 0.2 and s.freshness(10, 120) == 1.0
    later = R.Session(ev, t + 3 * 86400)
    assert later.freshness(9, 120) > 0.9


class _FakeRec(R.Recommender):
    """Pool logic on synthetic candidates: no public data needed."""

    def __init__(self, cands):
        self._c, self.rows = cands, []

    def score(self, session=None, now=None, keys=None):
        return [dict(c) for c in self._c], session

    def home_keys(self):
        return 7

    def warmups(self, *a):
        return []


def _cand(b, gain, keys=7, length=120, ch=0.0):
    return {"bid": b, "gain": gain, "ranked": True, "sd_model": 0.2, "keys": keys, "challenge": ch, "length": length,
            "best": 100.0, "p_up": 0.5, "var": "NM", "rate": 1.0, "title": f"m{b}", "md5": f"h{b}", "installed": True}


def test_smaller_pp_gains_remain_eligible_instead_of_recycling_the_top_three():
    t0 = time.time() - 5000
    cands = [_cand(b, g) for b, g in ((1, 10.0), (2, 9.0), (3, 8.0), (4, 0.3), (5, 0.2))]
    ev = []
    for i, b in enumerate((1, 2, 3)):                            # every good map was just played, some twice
        ev += _attempts(t0 + 600 * i, [0.5, 0.5, 0.5], bid=b, spacing=150)
    for event in ev:
        event["beatmap"] = f"md5:h{event['beatmap']}"    # new exact-chart identities, same freshness scenario
    rec = _FakeRec(cands)
    phase, shown, pick, note = rec.pool(R.Session(ev, t0 + 1900), rng=__import__("random").Random(1))
    assert {4,5} <= {c["bid"] for c in shown}                    # no tiny top-gain shortlist
    assert pick and pick['bid'] in {c['bid'] for c in rec.pp_choices}


def test_keymode_continuity():
    t0 = time.time() - 2000
    cands = [_cand(1, 5.0, keys=7), _cand(2, 5.5, keys=4)]
    rec = _FakeRec(cands)
    s = R.Session(_attempts(t0, [0.5, 0.6, 0.4, 0.5, 0.7]), t0 + 1600)
    _p, shown, _pick, _n = rec.pool(s)
    weights={c['bid']:c['weight'] for c in shown}
    assert weights[1] > weights[2]                               # continuity affects weight, not a fixed order
    rec = _FakeRec([_cand(1, 5.0, keys=7), _cand(2, 30.0, keys=4)])
    assert 2 in {c['bid'] for c in rec.pool(s)[1]}               # a comfortable alternate-mode probe remains eligible


# ---- refresh / frozen bundle --------------------------------------------------------
def test_refresh_interrupted_keeps_active_and_prunes_only_rolling():
    tmp = tempfile.mkdtemp()
    old = recdata.ROLLING, recdata.ACTIVE
    recdata.ROLLING, recdata.ACTIVE = os.path.join(tmp, "rolling"), os.path.join(tmp, "rolling", "active.json")
    try:
        for name in ("2026_07_01_performance_mania_top_1000", "2026_08_01_performance_mania_top_1000"):
            os.makedirs(os.path.join(recdata.ROLLING, name))
            recdata.activate(name)
        arch = os.path.join(tmp, "2026_10_01_performance_mania_top_1000.tar.bz2")
        with tarfile.open(arch, "w:bz2") as tf:               # a truncated snapshot: tables missing
            p = os.path.join(tmp, "osu_beatmaps.sql")
            open(p, "w").write("-- nothing\n")
            tf.add(p, "x/osu_beatmaps.sql")
        try:
            recdata.refresh(archive=arch)
            raise AssertionError("an incomplete archive must not switch")
        except ValueError:
            pass
        act = json.load(open(recdata.ACTIVE))
        assert act["name"].startswith("2026_08_01") and not any(d.startswith("staging-") for d in os.listdir(recdata.ROLLING))
        os.makedirs(os.path.join(recdata.ROLLING, "2026_09_01_performance_mania_top_1000"))
        keep = recdata.activate("2026_09_01_performance_mania_top_1000")
        assert sorted(os.listdir(recdata.ROLLING)) == sorted(list(keep) + ["active.json"])
        assert "2026_07_01_performance_mania_top_1000" not in keep
        try:
            recdata._assert_rolling(recdata.FROZEN)
            raise AssertionError("the frozen bundle must be untouchable")
        except RuntimeError:
            pass
    finally:
        recdata.ROLLING, recdata.ACTIVE = old


def test_frozen_manifest_unchanged():
    path = os.path.join(recdata.FROZEN, "FROZEN_MANIFEST.json")
    if not os.path.exists(path):
        return print("skip: no manifest on this machine")
    assert recdata.frozen_manifest(verify=True) == []
    import pickle
    with open(os.path.join(recdata.FROZEN, "cohort.pkl"), "rb") as fh:
        assert pickle.load(fh)["base"] == "ratings_23.pkl"


def test_selected_card_matches_list():
    """The selected-map card reproduces its list entry; a custom rate shows acc but no pp."""
    db = recdata.connect()
    if not recdata.load_public():
        return print("skip: no public evidence on this machine")
    rec = R.Recommender(db)
    root = next((p for p in (os.path.expanduser("~/.local/share/osu/files"),
                             os.path.expanduser("~/.var/app/sh.ppy.osu/data/osu/files")) if os.path.isdir(p)), None)
    for c in rec.score()[0]:
        inst = rec.installed.get(c["md5"])
        path = inst and root and f"{root}/{inst['sha256'][0]}/{inst['sha256'][:2]}/{inst['sha256']}"
        if c["gain"] > 1 and path and os.path.exists(path):
            break
    else:
        return print("skip: no installed candidate")
    k = rec.card(path, c["rate"])
    for f in ("acc_mid", "pp_lo", "pp_hi", "gain", "p_up"):
        assert abs(k[f] - c[f]) < 1e-6 * max(1, abs(c[f])), (f, k[f], c[f])
    assert not rec.card(path, 1.2)["ranked"]


def test_other_calculator_refused():
    """Public features from another skill_calc.py are never mixed with this one's."""
    try:
        R.Recommender(_db(), {"calc": "not-this", "feats": {}, "maps": {}, "pop": {}, "user": {}})
        raise AssertionError("stale public evidence accepted")
    except RuntimeError as exc:
        assert "another calculator" in str(exc)


def test_lazer_accuracy_scale():
    """The shown accuracy is lazer's (PERFECT 305): a Cherry Pop result read 95.78 there, 94.26 as acc320."""
    st = (2425, 1140, 164, 26, 25, 48)
    assert round(100 * recdata.acc_lazer(st), 2) == 95.78 and round(100 * recdata.acc320(st)[0], 2) == 94.26


def test_live_nomod_is_ranked():
    """tosu's empty mod list is NoMod (ranked, counts in the ledger at once); no list at all stays unknown."""
    db = _db()
    base = {"sha256": "s" * 64, "rate": 1.0, "hits": {"geki": 900, "300": 90, "katu": 5, "100": 3, "50": 1, "0": 1}}
    k1, _ = recdata.add_live_score(db, dict(base, score=1, mods_list=[]), None, True)
    k2, _ = recdata.add_live_score(db, dict(base, score=2), None, True)
    got = {k: r for k, r in db.execute("SELECT key, ranked FROM scores")}
    assert got[k1] == 1 and got[k2] == 0


def test_abort_ends_before_the_restart():
    """An abort logged at its own end time pairs with its own start, so the restarted play keeps its result."""
    db = _db()
    recdata.log_event(db, "start", "7", t=100.0, mu=-3.0, sd=0.4, keys=7)
    recdata.log_event(db, "abort", "7", t=160.0, progress=0.4)                 # the watcher queues the abort first,
    recdata.log_event(db, "start", "7", t=160.0, mu=-3.0, sd=0.4, keys=7)     # same second as the restart
    recdata.log_event(db, "finish", "7", t=400.0, y=-3.2)
    S = R.Session(R.load_events(db, since=0), now=500.0)
    assert [a["kind"] for a in S.attempts] == ["abort", "finish"] and S.attempts[1]["z"] > 0


def test_tracking_history_and_recovery():
    """Paused attempts never train after reimport; results retain their own pre-play expectation."""
    import datetime
    import feedback
    from contextlib import closing

    t = time.time() - 1000
    iso = lambda n: datetime.datetime.fromtimestamp(t + n, datetime.timezone.utc).isoformat()
    stats = (900, 50, 20, 10, 5, 15)
    live = {"sha256": "a" * 64, "score": 777, "hits": dict(zip(("geki", "300", "katu", "100", "50", "0"), stats)),
            "mods_list": [{"acronym": "HT"}], "rate": .75, "played": iso(20)}
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "rec.db")
        with closing(recdata.connect(path)) as db:
            k, _ = recdata.add_live_score(db, live, None, True)
            recdata.set_tracking(db, False, t=t + 40, start=t + 30)
        with closing(recdata.connect(path)) as db:
            assert not recdata.tracking_enabled(db)                 # restart preserves the pause
            paused = dict(live, played=iso(50))
            assert recdata.add_live_score(db, paused, None, True)[0] is None
            recdata.set_tracking(db, True, t=t + 60)
            assert recdata.tracking_enabled(db)
            late = dict(live, played=iso(70))
            assert recdata.add_live_score(db, late, None, True, {"t": t + 45, "end": t + 70})[0] is None
            for i, at in enumerate((20, 50, 70, 100)):
                rows, _ = recdata.realm_rows([_raw(i, stats, 777, date=iso(at))], 2653437)
                with db:
                    recdata.upsert_score(db, rows[0])
            assert db.execute("SELECT COUNT(*) FROM scores").fetchone()[0] == 2  # identical separate keyboard plays survive
            assert not feedback.add_score({"scores": []}, paused, "/unused", db)
            assert not feedback.add_score({"scores": []}, late, "/unused", db)
            sid = recdata.log_event(db, "start", "42", t=t + 80, sha=live["sha256"], rate=.75, mods=["HT"],
                                   mu=-2.1, sd=.2, expected=.99, predicted_at=t + 80, title="Played chart", length=30)
            # Polling resumes late and calls this an abort. Recovery must insert the actual earlier finish.
            recdata.log_event(db, "abort", "42", t=t + 110, progress=.7)
            recdata.log_event(db, "start", "99", t=t + 120, sha="b" * 64, rate=1, mu=-5, sd=.2)
            assert R.recover_results(db) == 1 and R.recover_results(db) == 0
            history = R.play_history(db)
            assert history[0]["expected"] == .99 and history[0]["recovered"] and history[0]["z"] > 1.5
            assert history[0]["title"] == "Played chart" and history[1]["expected"] is None
            assert len(R.play_history(db, outliers=True)) == 1
            assert history[0]["pause_positions_ms"] is None  # archive not available yet
            import replays
            db.executescript(replays.SCHEMA)
            db.execute("INSERT INTO replay_evidence(score_key,pauses) VALUES (?,?)",
                       (history[0]["key"], json.dumps([-1200, 76740])))
            db.execute("INSERT INTO replay_evidence(score_key,pauses) VALUES (?,?)", (history[1]["key"], "[]"))
            saved = R.play_history(db)
            assert saved[0]["pause_positions_ms"] == [-1200, 76740] and saved[1]["pause_positions_ms"] == []
            assert {k: v for k, v in saved[0].items() if k != "pause_positions_ms"} == {
                k: v for k, v in history[0].items() if k != "pause_positions_ms"}
            assert "2 play pauses (duration unknown)" in R.history_text(saved)
            assert "play pause" not in R.history_text(saved[1:])
            assert len(R.play_history(db, outliers=True)) == 1
            session = R.Session(R.load_events(db, since=0))
            assert len(session.attempts) == 1 and session.attempts[0]["kind"] == "finish"
            # A legacy key must retain its old event references instead of becoming a duplicate.
            old = recdata.score_key(live["sha256"], 777, stats)
            with db:
                db.execute("UPDATE scores SET key=? WHERE key=?", (old, k))
                rows, _ = recdata.realm_rows([_raw(8, stats, 777, date=iso(20))], 2653437)
                assert recdata.upsert_score(db, rows[0]) == old
            assert db.execute("SELECT COUNT(*) FROM scores").fetchone()[0] == 2
            # An unrelated last start must not supply a live play's label or expectation.
            msgs = []
            worker = R.Worker(msgs.append)
            worker.db = db
            worker.t_score(dict(live, played=iso(140)), "/missing", True, {"t": t + 80})
            assert msgs[-1]["type"] == "result" and "Played chart" in msgs[-1]["title"]
            assert "99.00%" in msgs[-1]["text"]


def test_watcher_keeps_polling_during_index_scan():
    import threading
    import skillsets
    from unittest.mock import patch
    entered, release = threading.Event(), threading.Event()

    def scan(**kw):
        entered.set()
        assert release.wait(2)
        return {"map": __file__}

    w = skillsets.LazerWatcher(lambda *_: None, lambda *_: None, lambda *_: None, lambda *_: None, lambda *_: None)
    with patch.object(skillsets.lazer_index, "load_or_build", scan):
        try:
            assert w._resolve("map") is None
            assert entered.wait(1) and w._index_job.is_alive()
            assert w._resolve("map") is None              # returns while scanning, allowing the next tosu poll
        finally:
            release.set()
            w._index_job.join(2)
        assert w._resolve("map") == __file__
    attempts = []
    w.on_attempt = lambda *a: attempts.append(a)
    w._play = {"sha": "a", "t": time.time() - 100, "progress": .7}
    w._disconnected = True
    w._track("songselect", {"files": {"beatmap": "a"}, "beatmap": {"time": {}}})
    assert attempts[0][0] == "void"               # an outage is not evidence of a failed play


if __name__ == "__main__":
    names = sys.argv[1:] or [n for n in dir() if n.startswith("test_")]
    for n in names:
        globals()[n]()
        print("ok", n)


def test_website_top_plays_become_ledger_bests(tmp_path):
    import json as _json
    from unittest.mock import patch
    db = recdata.connect(str(tmp_path / "rec.db"))
    page = b'<link rel="canonical" href="https://osu.ppy.sh/users/42">'
    best = [{"id": 7, "beatmap_id": 11, "pp": 500.5, "ended_at": "2026-05-03T17:09:43Z", "total_score": 900000,
             "mods": [{"acronym": "DT"}], "beatmap": {"checksum": "m" * 32, "ranked": 1},
             "statistics": {"perfect": 900, "great": 80, "good": 10, "ok": 5, "meh": 3, "miss": 2}},
            {"id": 8, "beatmap_id": 12, "pp": None, "mods": [], "beatmap": {}, "statistics": {}}]
    replies = iter([("https://osu.ppy.sh/users/someone", page), ("x", _json.dumps(best).encode())])
    with patch.object(recdata, "_website_json", lambda url: next(replies)):
        assert recdata.import_website_best(db, "someone") == (42, 1)
    row = db.execute("SELECT beatmap_id, rate, pp, md5, client FROM scores WHERE key='web|7'").fetchone()
    assert tuple(row) == (11, 1.5, 500.5, "m" * 32, "website")
    assert recdata.kv_get(db, "website_user")["id"] == 42


def test_release_evidence_is_stripped_and_installs_for_this_calculator(tmp_path, monkeypatch):
    import io, pickle as _pickle
    from unittest.mock import patch
    pub = {"calc": recdata.calc_id(), "snapshot": "snap_x", "user": {"id": 5, "scores": [1], "stats": {"a": 1}}}
    released = recdata.release_public(pub)
    assert released["user"]["scores"] == [] and released["user"]["id"] is None and pub["user"]["scores"] == [1]
    monkeypatch.setattr(recdata, "ROLLING", str(tmp_path))
    monkeypatch.setattr(recdata, "ACTIVE", str(tmp_path / "active.json"))
    with patch.object(recdata.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(_pickle.dumps(released))):
        assert recdata.fetch_release_public()["snapshot"] == "snap_x"
    assert recdata.load_public()["user"]["scores"] == []
    with patch.object(recdata.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(_pickle.dumps(dict(released, calc="other")))):
        with pytest.raises(ValueError):
            recdata.fetch_release_public()


def test_jack_and_ln_follow_the_players_level_not_only_the_stars():
    """Jack maps cost more once they pass the player's level; LN costs less near it (user 2026-10-01)."""
    import math
    import recdata
    import recommend
    pop = {"slope": {0: 2.}, "window": {0: 0.}, "mb": {}, "mbr": {},
           "gap": {4: {"jack": [-.4, 0., .2, .4, .2], "ln": [.2, 0., -.2, -.2, 0.]}}}
    jack = {"keys": 4, "overall": 5., "od": 8., "sk": {"jackspeed": 1.}}
    ln = {"keys": 4, "overall": 5., "od": 8., "sk": {"ln": 1.}}
    at = math.log(5.)
    shift = lambda f, level: recommend.base_of(pop, f, level=level)[0] - recommend.base_of(pop, f)[0]
    assert shift(jack, at + .3) < 0 < shift(jack, at - .15)       # easy for this player: cheap; above them: costly
    assert shift(ln, at) < 0 < shift(ln, at + .3)                 # LN at their level: cheaper; far below: still costs
    assert shift(dict(jack, keys=6), at - .15) == 0.              # 6K borrows 7K's curves: none here
    pop["gap"][7] = pop["gap"][4]
    assert shift(dict(jack, keys=6), at - .15) > 0
    assert shift(dict(jack, keys=9), at - .15) == 0.              # 8K–10K keep none (DEV worse with 7K's)
    # Centred parts describe "more than this keymode's usual mix": a chart at the mean share gets nothing.
    pop["gap"][7] = {"chords": [.3, .3, .3, .3, .3]}
    pop["share_mean"] = {7: {"chords": .5}}
    chords = lambda share: {"keys": 7, "overall": 5., "od": 8., "sk": {"chordstream": share}}
    assert shift(chords(.5), at) == 0. and shift(chords(1.), at) > 0 > shift(chords(0.), at)
    assert recommend.play_levels([{"keys": 4, "t": 0., "f": {"overall": 5.}}] * 25, now=1.)[4] == recdata.competence_level([at] * 25)


def test_packaged_install_replaces_evidence_fitted_by_older_links():
    """Friends' installs keep public.pkl; a newer POP_VERSION re-downloads it, a failed download keeps the old one."""
    from unittest.mock import patch
    import sys
    import recdata
    import recommend as R
    old = {"calc": recdata.calc_id(), "pop": {"version": 1}}
    new = {"calc": recdata.calc_id(), "pop": {"version": recdata.POP_VERSION}}
    with patch.object(sys, "frozen", True, create=True), patch.object(R.Recommender, "refit", lambda self: None), \
            patch.object(recdata, "import_public_user", lambda db, pub: None):
        with patch.object(recdata, "fetch_release_public", return_value=new):
            assert R.Recommender(None, pub=old).pub is new
            assert R.Recommender(None, pub=dict(new)).pub is not new        # current evidence: no download
        with patch.object(recdata, "fetch_release_public", side_effect=OSError("offline")):
            assert R.Recommender(None, pub=old).pub is old
