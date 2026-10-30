"""osu!stable support: library and scores from stable's databases, the same chart identity as lazer, HT/NM/DT only."""
import hashlib
import json
import os
import struct
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
import navigation  # noqa: E402
import nps  # noqa: E402
import paths  # noqa: E402
import recdata  # noqa: E402

OSU = b"osu file format v14\n\n[General]\nMode: 3\n\n[Difficulty]\nCircleSize:7\nOverallDifficulty:8\n"


def _s(v):
    b = v.encode()
    return b"\x0b" + bytes([len(b)]) + b


def osu_db(player, charts):
    """Minimal osu!.db (version 20250107: star ratings are Int-Float pairs)."""
    out = struct.pack("<iiBq", 20250107, 1, 1, 0) + _s(player) + struct.pack("<i", len(charts))
    for c in charts:
        out += b"".join(_s(x) for x in (c["artist"], "", c["title"], "", "mapper", c["version"], "audio.mp3",
                                       c["md5"], c["file"]))
        out += struct.pack("<Bhhhqffffd", 4, 1, 0, 0, 0, 9., 7., 8., 8., 1.4)
        out += struct.pack("<i", 1) + b"\x08" + struct.pack("<i", 0) + b"\x0c" + struct.pack("<f", 4.)  # osu! stars
        out += struct.pack("<iii", 0, 0, 0)
        out += struct.pack("<iii", 90, 90000, 0) + struct.pack("<i", 0)            # drain, total, preview, points
        out += struct.pack("<iii", c["bid"], 77, 0) + bytes(4) + struct.pack("<hf", 0, .7) + bytes([c["mode"]])
        out += _s("") + _s("") + struct.pack("<h", 0) + _s("") + struct.pack("<?q?", True, 0, False)
        out += _s(c["folder"]) + struct.pack("<q", 0) + bytes(5) + struct.pack("<iB", 0, 20)
    return out + struct.pack("<i", 0)


def scores_db(md5, player, rows):
    out = struct.pack("<ii", 20250107, 1) + _s(md5) + struct.pack("<i", len(rows))
    for stats, score, mods, unix in rows:
        geki, c300, katu, c100, c50, miss = stats
        out += struct.pack("<Bi", 3, 20250107) + _s(md5) + _s(player) + _s("r")
        out += struct.pack("<hhhhhhihBi", c300, c100, c50, geki, katu, miss, score, 500, 0, mods)
        out += _s("") + struct.pack("<qiq", 621_355_968_000_000_000 + unix * 10_000_000, -1, 0)
    return out


@pytest.fixture
def stable(tmp_path, monkeypatch):
    root = tmp_path / "osu!"
    (root / "Songs" / "123 set").mkdir(parents=True)
    (root / "Songs" / "123 set" / "a.osu").write_bytes(OSU)
    (root / "Songs" / "123 set" / "b.osu").write_bytes(OSU.replace(b"Mode: 3", b"Mode: 0"))
    md5 = hashlib.md5(OSU).hexdigest()
    charts = [dict(artist="Art", title="Song", version="Hard", md5=md5, file="a.osu", folder="123 set", bid=55, mode=3),
              dict(artist="Art", title="Std", version="Hard", md5="0" * 32, file="b.osu", folder="123 set", bid=56, mode=0)]
    (root / "osu!.db").write_bytes(osu_db("me", charts))
    (root / "scores.db").write_bytes(scores_db(md5, "me", [((900, 80, 10, 5, 2, 3), 812345, 64, 1_700_000_000),
                                                          ((900, 80, 10, 5, 2, 3), 700000, 0, 1_700_000_100)]))
    monkeypatch.setattr(paths, "DATA", str(tmp_path / "data"))
    monkeypatch.setenv("STABLE_DATA", str(root))
    recdata.set_client("stable")
    yield recdata.connect(str(tmp_path / "rec.db")), hashlib.sha256(OSU).hexdigest(), md5
    recdata.set_client("lazer")


def test_library_and_scores_import_with_lazer_identity(stable):
    db, sha, md5 = stable
    assert recdata.import_library(db) == {"user_id": None, "seen": 2, "new": 2, "installed": 1}   # mania only
    row = dict(db.execute("SELECT * FROM installed").fetchone())
    assert (row["sha256"], row["beatmap_id"], row["md5"], row["status"], row["keys"]) == (sha, 55, md5, 1, 7)
    with open(recdata.local_file(sha), "rb") as fh:      # every sha → file lookup reaches stable's chart
        assert fh.read() == OSU
    rates = {r["total"]: (r["rate"], r["client"], r["beatmap_id"]) for r in db.execute("SELECT * FROM scores")}
    assert rates == {812345: (1.5, "stable", 55), 700000: (1.0, "stable", 55)}
    assert recdata.import_library(db)["new"] == 0                  # idempotent, files not re-hashed
    # tosu's results screen of the same play lands on the same row, whatever its timestamp
    live = {"sha256": sha, "client": "stable", "hits": {"geki": 900, "300": 80, "katu": 10, "100": 5, "50": 2, "0": 3},
            "score": 812345, "played": "2026-10-31T01:00:00", "mods_list": [{"acronym": "DT"}]}
    assert recdata.add_live_score(db, live, None, False)[1] is False
    assert db.execute("SELECT COUNT(*) FROM scores").fetchone()[0] == 2


def test_rates_search_and_shown_accuracy(stable):
    assert [nps.snap(r) for r in (.7, .86, .9, 1.2, 1.23, 1.5)] == [.75, .75, 1., 1., 1.5, 1.5]
    assert nps.offerable(1.5) and not nps.offerable(1.1)
    c = {"artist": "Art", "song_title": 'Song "X"', "version": "Hard", "keys": 7}
    assert navigation.song_search(c) == "Art Song X Hard keys=7"
    st = (900, 80, 10, 5, 2, 3)
    assert recdata.acc_stable(st) > recdata.acc_lazer(st)
    assert abs(recdata.model_acc(recdata.shown_acc(.95)) - .95) < 1e-12 and recdata.shown_acc(.95) > .95
    import recommend                                # every shown accuracy in stable units, lazer plays included
    row = dict(played="2026-10-31T01:00:00", rate=1.5, expected=.95, z=.5, title="T", actual=recdata.acc_lazer(st),
               stats=list(st), recovered=True, pause_positions_ms=None)
    shown = recommend.history_text([row])
    assert f"{100 * recdata.acc_stable(st):.2f}%" in shown and f"expected {100 * recdata.shown_acc(.95):.2f}%" in shown
    assert "from osu!stable" in shown and recdata.shown_points(1.) < 1.
    recdata.set_client("lazer")
    assert nps.snap(1.23) == 1.23 and nps.offerable(1.1) and recdata.shown_acc(.95) == .95
    assert json.loads(json.dumps(navigation.song_search(c))).startswith('artist="Art"')


def test_tracked_stable_play_archives_its_replay_and_tosu_pauses(stable, tmp_path):
    db, sha, md5 = stable
    import replays
    root = os.environ["STABLE_DATA"]
    recdata.import_library(db)                  # links the chart (a live selection does the same)
    st = (900, 80, 10, 5, 2, 3)
    head = struct.pack("<Bi", 3, 20250107) + _s(md5) + _s("me") + _s("r")
    head += struct.pack("<hhhhhhihBi", 80, 5, 2, 900, 10, 3, 812345, 500, 0, 64) + _s("") + struct.pack("<q", 0)
    os.makedirs(os.path.join(root, "Data", "r"))
    with open(os.path.join(root, "Data", "r", f"{md5}-638000000000000000.osr"), "wb") as fh:
        fh.write(head + struct.pack("<i", 0))
    with open(os.path.join(root, "Data", "r", f"{md5}-638000000000000001.osr"), "wb") as fh:
        fh.write(head.replace(struct.pack("<i", 812345), struct.pack("<i", 812344)))        # another play
    live = {"sha256": sha, "client": "stable", "hits": {"geki": 900, "300": 80, "katu": 10, "100": 5, "50": 2, "0": 3},
            "score": 812345, "played": "2026-10-31T01:00:00", "mods_list": [{"acronym": "DT"}]}
    key, _new = recdata.add_live_score(db, live, None, True)
    recdata.log_event(db, "finish", None, key=key, pauses=[12000, 30500])
    untracked = dict(live, score=700000, hits=dict(live["hits"], miss=4))
    recdata.add_live_score(db, untracked, None, False)
    assert replays.archive_stable(db, root, str(tmp_path)) == 1
    row = dict(db.execute("SELECT * FROM replay_evidence").fetchone())
    assert (row["score_key"], row["status"], json.loads(row["pauses"]), row["rate"]) == (key, "archived", [12000, 30500], 1.5)
    with open(tmp_path / row["archive_path"], "rb") as fh:
        assert fh.read() == head + struct.pack("<i", 0)
    assert db.execute("SELECT COUNT(*) FROM replay_evidence").fetchone()[0] == 1       # only observed attempts
    assert replays.archive_stable(db, root, str(tmp_path)) == 0                        # idempotent


def test_watcher_records_pause_positions_during_play():
    import skillsets
    w = skillsets.LazerWatcher(*(lambda *a: None,) * 5)
    def poll(live, paused, state="play"):
        w._track(state, {"files": {"beatmap": "x"}, "beatmap": {"time": {"live": live, "firstObject": 0, "lastObject": 9e4}},
                         "play": {"mods": {"array": []}}, "game": {"paused": paused}}, "/s/" + "a" * 64)
    for live, paused in ((1000, False), (2000, False), (2100, True), (2100, True), (2200, False), (5000, True)):
        poll(live, paused)
    assert w._play["pauses"] == [2100, 5000]
    poll(5000, False, "resultscreen")
    assert w._done["pauses"] == [2100, 5000]


@pytest.mark.parametrize("own", [True, False])
def test_stable_result_counts_only_after_own_attempt_not_a_watched_replay(own):
    import skillsets
    got = []
    w = skillsets.LazerWatcher(lambda *a: None, lambda *a: None, lambda *a: None, lambda *a: got.append(a), lambda *a: None)
    sha = "a" * 64
    w._done = {"sha": sha, "t": 0, "own": own}
    d = {"client": "stable", "profile": {"name": "Guest"},
         "resultsScreen": {"playerName": "wd", "mode": {"name": "mania"}, "mods": {"array": []},   # tosu: stable's own name
                           "hits": {"geki": 9, "300": 1, "katu": 0, "100": 0, "50": 0, "0": 0}, "score": 1,
                           "createdAt": "2026-10-31T00:00:00Z"}}
    w._note_score(d, "/s/" + sha)
    assert len(got) == (1 if own else 0)
