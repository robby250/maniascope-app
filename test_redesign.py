"""Focused behavioral checks; isolated data only."""
import math
from types import SimpleNamespace
import difficulty_model
import nps
import recommend as R
import datetime
import hashlib
import json
from unittest.mock import patch
import recdata
import replays


def events(zs, skill="chordjack", played_rate=1., offered_rate=1., kind="finish", seconds=100.):
    out = []
    for i, z in enumerate(zs):
        t = 1000 + i * 150
        out += [{"id": 2*i+1, "t": t, "kind": "start", "beatmap": f"md5:m{i}", "info":
                 {"keys": 7, "skill": skill, "mu": -3., "base_mu": -3., "sd": .3,
                  "length": 120., "rate": played_rate, "offered_rate": offered_rate}},
                {"id": 2*i+2, "t": t+seconds, "kind": kind, "beatmap": f"md5:m{i}", "info":
                 {"y": -3.-.3*z, "played_seconds": seconds} if kind == "finish" else {"played_seconds": seconds}}]
    return out


def test_asymmetric_skill_sensitive_session():
    s = R.Session(events([-4.]), 1101.)
    assert s.correction(7, "chordjack") > .4
    assert s.correction(7, "chordjack") > 3*s.correction(7, "ln")
    assert s.correction(4, "chordjack") < s.correction(7, "ln")
    good = R.Session(events([1.]), 1101.)
    assert abs(good.correction(7, "chordjack")) < .05
    assert not R.Session(events([0., 0., 0., 0., 0.]), 1800).push()
    # Only beaten bests climb the staircase; good form alone is correction()'s job.
    assert R.Session(events([1., 1., 1.]), 1600).push(7, "chordjack") == 0


def beaten(zs, beats):
    """events() from PP offers; the listed attempts improved their counted best."""
    out = events(zs)
    for e in out:
        if e["kind"] == "start":
            e["info"].update(mode="pp", purpose="farm")
    for i in beats:
        out.append({"id": 100+i, "t": 1000+i*150+101, "kind": "pp_result", "beatmap": f"md5:m{i}",
                    "info": {"start_id": 2*i+1, "before": 500., "after": 520.}})
    return out


def test_push_is_a_two_to_one_staircase():
    end = lambda n: 1000 + n*150
    push = lambda zs, beats, pp=True: R.Session(beaten(zs, beats) if pp else events(zs), end(len(zs))).push(7, "chordjack")
    # No threshold: every beaten PP-offered best is one step at once.
    steps = [push([0.]*3, list(range(n))) for n in range(4)]
    assert steps == [0, 1, 2, 3]
    # A PP attempt below its prediction without a beat, or any clearly bad play, is PUSH_DOWN steps down.
    assert push([0., 0., 0., -.2], [0, 1, 2]) == 3 - R.PUSH_DOWN
    assert push([-1.5], [], pp=False) == 0 and push([2.], [], pp=False) == 0
    assert push([0., 0., 0., .3], [0, 1, 2]) == 3       # missed the best but met the prediction
    assert push([0., -3.], [0]) == 0
    # Equilibrium: two beats per miss hold the position.
    assert push([0.]*4 + [0., 0., -.5]*2, [0, 1, 2, 3, 4, 5, 7, 8]) == 4
    # Readiness follows it smoothly.
    import pp_playlist
    shares = [pp_playlist.push_share(p) for p in steps]
    assert all(0 < b - a < .45 for a, b in zip(shares, shares[1:]))


def test_downrate_requires_real_play_and_does_not_invent_failure():
    browsing = R.Session(events([0], played_rate=.8, offered_rate=1., kind="abort", seconds=1), 1100)
    assert browsing.rate_penalty("m0", 1.) == 0 and browsing.correction(7, "chordjack") == 0
    quit_early = R.Session(events([0], played_rate=.8, offered_rate=1., kind="abort", seconds=80), 1100)
    assert quit_early.rate_penalty('m0',1.)==0 and quit_early.activation(7)==0
    played = R.Session(events([0], played_rate=.8, offered_rate=1., kind="finish", seconds=80), 1100)
    assert played.rate_penalty("m0", 1.) > .4
    assert played.rate_penalty("m0", .8) == 0 and played.rate_penalty("other", 1.) == 0
    assert played.correction(7, "chordjack") < .03


def test_central_prediction_is_not_risk_shifted():
    pred = R.Predictor.__new__(R.Predictor)
    pred.model = SimpleNamespace(predict=lambda *a: (-3., .4, .3), retry=0)
    pred.shown = (math.log(.75), 1.)
    session = R.Session([], 1000)
    session.warm_flag = True  # isolate central estimate from the explicit cold-session prior
    e = pred.predict({"keys": 7, "sk": {"chordjack": 1}}, "m", None, 1., session)
    assert abs(e["acc_mid"] - (1-.75*math.exp(-3))) < 1e-12
    assert e["acc_lo"] < e["acc_mid"] < e["acc_hi"]


def test_category_sampling_matches_eligible_families_not_slots():
    cs = [{"sha": str(i), "md5": f"m{i}", "keys": 7, "family": str(i), "group": "sv" if i>=98 else "rice",
           "profile": [0, 1] if i>=98 else [1, 0], "base_value": math.log(30), "length": 160.,
           "rate": 1., "nps": 30., "description": "SV" if i>=98 else "Stream", "acc_mid": .94} for i in range(100)]
    cs.append(dict(cs[99], sha="duplicate-baked-rate"))
    rows = nps.weighted_candidates(cs, R.Session([], 1000), [7])
    assert len(rows) == 100
    share = sum(c["weight"] for c in rows if c["group"] == "sv") / sum(c["weight"] for c in rows)
    assert abs(share-.02) < 1e-10
    cs += [dict(cs[0], sha="too-fast", family="x", rate=1.51), dict(cs[0], sha="too-slow", family="y", rate=.69)]
    assert all(.7 <= c["rate"] <= 1.5 for c in nps.select(cs, R.Session([], 1000), [7], n_show=None))
    assert nps.central_target(R.Session([], 1000)) == .94


def test_short_drills_penalized_without_banning_specialists():
    chart = SimpleNamespace(keys=7, notes=[(80*i, 80*i, c) for i in range(1000)
                                         for c in ((0,1,2) if i%2 else (3,4,5,6))])
    st = nps.structure(chart)
    assert st["drill"] > .7 and nps.quality_penalty(38., st) > 2.
    assert nps.quality_penalty(75., {}) > nps.quality_penalty(100., {})
    assert nps.quality_penalty(240., {"drill": 0.}) == 0.


def test_endurance_recovers_and_depends_on_margin():
    def run(values):
        fine = [v/8 for v in values for _ in range(8)]
        return [(None, None, (fine, fine, [0.]*len(fine)))]
    continuous, _ = difficulty_model.endurance(run([10.]*240))
    rested, timeline = difficulty_model.endurance(run([10.]*120 + [0.]*60 + [10.]*120))
    assert continuous["load"] > rested["load"] and timeline[179] < timeline[119] / 5
    f = {"sk": {}, "ln": 0., "stam": 1., "endurance": {"load": .8}}
    assert R.zvec(f, (), -4.5)[-1] < R.zvec(f, (), -2.5)[-1] / 3


def test_rate_guidance_does_not_claim_unknown_matches():
    import skillsets as ui
    c = {"title": "A & B [Test]", "keys": 7, "rate": .85, "sha": "a"*64, "acc_mid": .94}
    obs = {"source": "tosu", "path": "/tmp/"+c["sha"], "rate": .85000001, "mods": [{"acronym": "HT"}]}
    assert "#238747" in ui.target_markup(c, obs)
    assert "#d9384f" in ui.target_markup(c, dict(obs, rate=.9))
    assert "#238747" not in ui.target_markup(c, dict(obs, source=None))
    assert "#238747" not in ui.target_markup(c, dict(obs, path="other"))
    assert "CURRENT" not in ui.target_markup(c, obs) and "A &amp; B" in ui.target_markup(c, obs)
    nm = dict(c, rate=1.)
    assert "#d9384f" in ui.target_markup(nm, dict(obs, rate=1., mods=[{"acronym": "DT"}]))
    assert "#238747" in ui.target_markup(nm, dict(obs, rate=1., mods=[]))
    updated=ui.target_markup(dict(c,bid=1123210),dict(obs,path='/tmp/new-sha',bid=1123210))
    assert '#238747' in updated and 'Updated chart selected' in updated
    assert 'Expected ~94.0%' not in updated
    assert '#238747' not in ui.target_markup(dict(c,bid=1123210),dict(obs,path='/tmp/new-sha',bid=123))



def test_replay_archive_is_idempotent_and_survives_source_cleanup(tmp_path):
    db = recdata.connect(str(tmp_path/"rec.db")); recdata.kv_set(db, "user_id", 7)
    data = b"compressed-osr-fixture"; h = hashlib.sha256(data).hexdigest()
    source = tmp_path/"source"; source.write_bytes(data)
    date = datetime.datetime.now(datetime.timezone.utc).isoformat()
    st = [900, 50, 20, 10, 5, 15]
    score = {"sha256": "a"*64, "score": 777, "mods_list": [], "rate": 1., "played": date,
             "hits": dict(zip(("geki", "300", "katu", "100", "50", "0"), st))}
    key, _ = recdata.add_live_score(db, score, None, True)
    raw = [{"hash": "a"*64, "date": date, "id": "score-uuid", "user_id": 7, "total": 777,
            "stats": json.dumps(dict(zip(("perfect", "great", "good", "ok", "meh", "miss"), st))),
            "mods": "[]", "replay_sha256": h, "pauses": [20000]}]
    with patch.object(recdata, "local_file", return_value=str(source)):
        assert replays.archive(db, raw) == 1
        source.unlink()
        assert replays.archive(db, raw) == 0
    r = db.execute("SELECT * FROM replay_evidence WHERE score_key=?", (key,)).fetchone()
    assert r["status"] == "archived" and json.loads(r["pauses"]) == [20000]
    assert (tmp_path/r["archive_path"]).read_bytes() == data
    db.close()


def test_replay_catchup_runs_with_queued_work_but_not_during_gameplay(tmp_path):
    w = R.Worker(lambda message: None, busy=lambda: True)
    w.db = recdata.connect(str(tmp_path/'rec.db'))
    w.db.executescript(replays.SCHEMA)
    w._replay_due = 1.
    w.put('ordinary_pending_work')
    with patch.object(replays, 'recent', return_value=0) as read:
        w._catch_up_replays()
        assert read.call_count == 0 and w._replay_due == 1.
        w.busy = lambda: False
        w._catch_up_replays()
        w._replay_thread.join(timeout=2)
        assert read.call_count == 1 and w._replay_due is None and not w.q.empty()
    w.db.close()
