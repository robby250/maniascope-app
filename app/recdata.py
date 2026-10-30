#!/usr/bin/env python3
"""
Recommendation data: the personal store, imports, the public evidence build and its refresh.

Four kinds of state stay apart (docs/PLAN_2026-09-24_PERSONAL_PP_RECOMMENDER.md §3):
  * personal attempts + events + session: REC_DIR/rec.db (SQLite; every writer uses transactions)
  * account ledger: rows of the same table marked eligible, reconciled against the snapshot's total
  * rolling public evidence: REC_DIR/rolling/<snapshot>/public.pkl, switched through active.json
  * the frozen calibration bundle (~/.cache/maniascope/dump): read only, hashed into FROZEN_MANIFEST.json

The public dump licence (https://data.ppy.sh/LICENCE.txt) covers statistical analysis and testing; it
does not grant production deployment. The osu! API terms forbid competitive-advantage use. So nothing
here polls the API, and `refresh --download` is a manual, personal-analysis step.

usage: recdata.py import [realm]          lazer scores + installed maps → rec.db
       recdata.py build [DUMP_DIR]        public.pkl from the frozen snapshot (heavy; ~15 min)
       recdata.py refit                   refit the active public.pkl's population links (charts kept)
       recdata.py check                   newest snapshot on data.ppy.sh vs the active one
       recdata.py refresh --archive TAR | --download   stage, validate, switch, prune
       recdata.py manifest [verify]       write / verify the frozen bundle's manifest
"""
import collections
import datetime
import hashlib
import json
import math
import os
import pickle
import re
import shutil
import sqlite3
from functools import lru_cache
import sys
import tarfile
import time
import urllib.parse
import urllib.request

import paths  # noqa: E402
REC_DIR = os.path.expanduser(os.environ.get("MANIASCOPE_REC", os.path.join(paths.DATA, "rec")))
DB_FILE = os.path.join(REC_DIR, "rec.db")
ROLLING = os.path.join(REC_DIR, "rolling")
ACTIVE = os.path.join(ROLLING, "active.json")
FROZEN = os.path.join(paths.CACHE, "dump")
FROZEN_SNAPSHOT = "2026_09_01_performance_mania_top_1000"
INDEX_URL = "https://data.ppy.sh/"
KEEP_ROLLING = 2                       # active + one previous, for rollback

RANKED_STATUS = (1, 2)                 # ranked, approved; 4 = loved (warmup only)
DEFAULT_RATE = {"HT": 0.75, "DC": 0.75, "DT": 1.5, "NC": 1.5}
RANKED_MODS = {"NF", "EZ", "HD", "FI", "FL", "MR", "SD", "PF", "HT", "DC", "DT", "NC", "CL"}
VARIANTS = ((0.75, "HT"), (1.0, "NM"), (1.5, "DT"))
SUPPORTED_KEYS = tuple(range(4, 11))


def played_keys(db, n=500, share=.1):
    """Keymodes holding at least `share` of the player's latest n lazer plays, or () before 20 plays."""
    rows = [r[0] for r in db.execute("SELECT i.keys FROM scores s JOIN installed i ON i.sha256=s.sha256 "
                                     "WHERE s.src!='public' ORDER BY s.played DESC LIMIT ?", (n,))
            if r[0] in SUPPORTED_KEYS]
    if len(rows) < 20:
        return ()
    return tuple(sorted(k for k in set(rows) if rows.count(k) >= share * len(rows)))


def normalize_keys(value):
    if value is None or value == "all":
        return SUPPORTED_KEYS
    if isinstance(value, (str, int)):
        value = [value]
    if not isinstance(value, (list, tuple, set, range)):
        return ()
    out = set()
    for k in value:
        try:
            n = int(k)
            if not isinstance(k, bool) and str(n) == str(k).strip() and n in SUPPORTED_KEYS:
                out.add(n)
        except (TypeError, ValueError):
            pass
    return tuple(sorted(out))


def keys_label(value):
    keys = normalize_keys(value)
    if keys == SUPPORTED_KEYS:
        return "All"
    if len(keys) >= 3 and keys == tuple(range(keys[0], keys[-1] + 1)):
        return f"{keys[0]}K–{keys[-1]}K"
    return ", ".join(f"{k}K" for k in keys) or "None"


def raw_nps(chart, rate):
    """Chord-inclusive normalized heads / first-to-last head seconds; LN tails are duration only."""
    if not chart.notes or rate is None or not math.isfinite(rate) or rate <= 0:
        return None
    span = chart.notes[-1][0] - chart.notes[0][0]
    return len(chart.notes) * rate * 1000 / span if span > 0 else None


CLIENTS = ("lazer", "stable")
_client, _miss_ratio = "lazer", .88


def set_client(name, miss_ratio=None):
    """The osu! client whose library and rates the app serves (tosu reports it; kv 'client' keeps it).
    miss_ratio: stable_miss_ratio, for showing accuracies as stable does."""
    global _client, _miss_ratio
    _client = name if name in CLIENTS else "lazer"
    _miss_ratio = miss_ratio or _miss_ratio


def client():
    return _client


def shown_acc(acc):
    """A lazer-unit accuracy (predictions, targets inside the model) as the active client displays it."""
    return acc if _client != "stable" or acc is None else 1 - (1 - acc) * _miss_ratio


def model_acc(acc):
    """shown_acc's inverse: a displayed accuracy (the target slider) in the model's lazer units."""
    return acc if _client != "stable" or acc is None else 1 - (1 - acc) / _miss_ratio


def stable_store():
    """stable's .osu files under their SHA-256, in lazer's files/ layout (hard links): every sha → path lookup
    works for both clients, and an edited chart keeps its old revision's identity."""
    return os.path.join(paths.DATA, "stable")


def local_origin():
    import socket
    root = stable_store() if _client == "stable" else os.path.realpath(paths.lazer_data())
    return {"host": socket.gethostname(), "root": root}


def local_file(sha, root=None):
    """root: local_origin()["root"], for callers resolving thousands of files (one realpath, not one each)."""
    if not isinstance(sha, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", sha):
        return None
    return os.path.join(root or local_origin()["root"], "files", sha[0], sha[:2], sha)


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS scores(key TEXT PRIMARY KEY, src TEXT, local_id TEXT, online_id INTEGER,
  player_id INTEGER, sha256 TEXT, md5 TEXT, beatmap_id INTEGER, status INTEGER, played TEXT, imported TEXT,
  client TEXT, mods TEXT, rate REAL, ranked INTEGER, stats TEXT, acc REAL, n INTEGER, total INTEGER,
  pp REAL, fresh INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, t REAL, kind TEXT, beatmap TEXT, info TEXT);
CREATE TABLE IF NOT EXISTS kv(k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS tracking_pauses(start REAL PRIMARY KEY, end REAL);
CREATE TABLE IF NOT EXISTS ignored_scores(key TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS feats(key TEXT PRIMARY KEY, data TEXT);
CREATE TABLE IF NOT EXISTS installed(sha256 TEXT PRIMARY KEY, beatmap_id INTEGER, set_id INTEGER, md5 TEXT,
  status INTEGER, keys INTEGER, od REAL, length REAL, title TEXT, artist TEXT, version TEXT);
CREATE TABLE IF NOT EXISTS installed_local(sha256 TEXT PRIMARY KEY, online_md5 TEXT, audio_sha TEXT,
  audio_required INTEGER, creator TEXT);
CREATE INDEX IF NOT EXISTS scores_bm ON scores(beatmap_id);
CREATE INDEX IF NOT EXISTS scores_sha ON scores(sha256);
CREATE INDEX IF NOT EXISTS scores_local ON scores(local_id);
CREATE INDEX IF NOT EXISTS installed_bid ON installed(beatmap_id);
CREATE INDEX IF NOT EXISTS events_t ON events(t);
"""


def connect(path=None):
    path = path or DB_FILE
    os.makedirs(os.path.dirname(path), exist_ok=True)
    db = sqlite3.connect(path, timeout=30, check_same_thread=False)
    # The worker, NPS builder, feature fill and import each hold a connection. With a rollback journal
    # every feature-batch commit made the selected card's reads wait (WAL: readers never wait for a writer).
    db.execute("PRAGMA journal_mode=WAL")
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA)
    return db


def kv_get(db, k, default=None):
    r = db.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
    return json.loads(r[0]) if r else default


def kv_set(db, k, v):
    with db:
        db.execute("INSERT OR REPLACE INTO kv VALUES (?, ?)", (k, json.dumps(v)))


def log_event(db, kind, beatmap=None, t=None, **info):
    with db:
        return db.execute("INSERT INTO events(t, kind, beatmap, info) VALUES (?, ?, ?, ?)",
                          (time.time() if t is None else t, kind, beatmap, json.dumps(info))).lastrowid


@lru_cache(maxsize=2048)
def event_info(raw):
    """Read-only saved payload; readers adding metadata must copy its root.

    Chart/profile values are immutable, like the feature cache. Keying by raw
    JSON also observes edits immediately; tracking/eligibility is never cached.
    """
    return json.loads(raw or '{}')


def annotate_attempt(db, start_id, reason, note="", confidence=1.):
    """Append explicit player feedback without changing the recorded attempt.

    This is evidence annotation, never an automatic guess that an abort means
    failure. A matching existing annotation is idempotent.
    """
    if reason not in ("failed", "interrupted", "experiment", "bored"):
        raise ValueError("unsupported attempt reason")
    if not isinstance(confidence,(int,float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1")
    start=db.execute("SELECT beatmap FROM events WHERE id=? AND kind='start'",(start_id,)).fetchone()
    if start is None:
        raise ValueError("start event not found")
    previous=db.execute("SELECT id,info FROM events WHERE kind='attempt_note' "
                        "AND json_extract(info,'$.start_id')=? ORDER BY id DESC LIMIT 1",(start_id,)).fetchone()
    payload=dict(start_id=start_id,reason=reason,note=str(note)[:1000],confidence=float(confidence),source="player")
    if previous and json.loads(previous['info']) == payload:
        return previous['id']
    return log_event(db,"attempt_note",start['beatmap'],**payload)


def timestamp(iso):
    try:
        return datetime.datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
    except (AttributeError, ValueError):
        return None


def tracking_enabled(db):
    return db.execute("SELECT 1 FROM tracking_pauses WHERE end IS NULL").fetchone() is None


def set_tracking(db, enabled, t=None, start=None):
    t = time.time() if t is None else t
    with db:
        if enabled:
            db.execute("UPDATE tracking_pauses SET end=? WHERE end IS NULL", (t,))
        elif tracking_enabled(db):
            # Pausing halfway through a play excludes that whole attempt.
            db.execute("INSERT INTO tracking_pauses VALUES (?, NULL)", (min(t, start) if start else t,))


def tracking_allowed(db, end, start=None):
    if end is None:
        # Undated imports cannot be proved to fall outside a controller session.
        return db.execute("SELECT 1 FROM tracking_pauses LIMIT 1").fetchone() is None
    start = end if start is None else start
    return db.execute("SELECT 1 FROM tracking_pauses WHERE start<=? AND (end IS NULL OR end>?) LIMIT 1",
                      (end, start)).fetchone() is None


def tracked_events(db, since, kinds=(), until=None, fields=None):
    """Read current pause exclusions in one query, not once per event.

    fields: keep only these info keys (SQLite JSON), sparing the decode of large start payloads."""
    info = ("json_object(" + ",".join(f"'{k}',json_extract(e.info,'$.{k}')" for k in fields) + ") AS info"
            if fields else "e.info")
    query = (f"SELECT e.id,e.t,e.kind,e.beatmap,{info} FROM events e WHERE e.t>=? AND NOT EXISTS ("
             "SELECT 1 FROM tracking_pauses p WHERE p.start<=e.t AND (p.end IS NULL OR p.end>e.t))")
    if kinds:
        query += " AND e.kind IN (" + ",".join("?" for _ in kinds) + ")"
    if until is not None:
        query += " AND e.t<?"
    return db.execute(query + " ORDER BY e.t,e.id", (since, *kinds, *(() if until is None else (until,))))


def score_allowed(db, score, play=None, exclude=False):
    """Shared by live results and feedback imports; tombstones prevent later Realm resurrection."""
    h = score.get("hits") or {}
    stats = [int(h.get(k, 0) or 0) for k in ("geki", "300", "katu", "100", "50", "0")]
    key = score_key(score["sha256"], score.get("score") or 0, stats, score.get("played"))
    allowed = not db.execute("SELECT 1 FROM ignored_scores WHERE key=?", (key,)).fetchone() \
        and tracking_allowed(db, timestamp(score.get("played")))
    if play:
        allowed = allowed and tracking_allowed(db, play.get("end", time.time()), play["t"])
    if not allowed and exclude:
        with db:
            db.execute("INSERT OR IGNORE INTO ignored_scores VALUES (?)", (key,))
    return bool(allowed)


def variant(mods):
    """lazer mod list → (rate | None, ranked: bool). Custom DT/HT speeds and any other setting are
    unranked (osu! wiki, Double Time (lazer)); adjust_pitch alone keeps a score ranked."""
    rate, ranked = 1.0, True
    for m in mods or []:
        ac = str(m.get("acronym", "")).upper()
        st = {k: v for k, v in (m.get("settings") or {}).items() if k != "adjust_pitch"}
        if ac in ("WU", "WD", "AS"):
            return None, False
        if ac in DEFAULT_RATE:
            rate = float(st.get("speed_change", DEFAULT_RATE[ac]))
            ranked &= rate == DEFAULT_RATE[ac] and set(st) <= {"speed_change"}
        else:
            ranked &= ac in RANKED_MODS and not st
    return rate, ranked


def acc320(stats):
    """(perfect, great, good, ok, meh, miss) → (mania pp accuracy, judgement count)."""
    p, g, gd, ok, meh, miss = stats
    n = p + g + gd + ok + meh + miss
    return ((320 * p + 300 * g + 200 * gd + 100 * ok + 50 * meh) / (320 * n) if n else 0.0), n


def acc_stable(stats):
    """The accuracy stable's results screen shows: MAX and 300 both count 300."""
    p, g, gd, ok, meh, miss = stats
    n = p + g + gd + ok + meh + miss
    return (300 * (p + g) + 200 * gd + 100 * ok + 50 * meh) / (300 * n) if n else 0.0


def stable_miss_ratio(db):
    """(1 − stable accuracy) / (1 − lazer accuracy) of the same judgements: predictions stay in lazer units, stable
    shows MAX as 300 (≈0.88 for a 95 % player). The player's own median when they have lazer plays."""
    ratios = sorted((1 - acc_stable(st)) / (1 - acc_lazer(st)) for st in
                    (json.loads(r[0]) for r in db.execute("SELECT stats FROM scores WHERE stats IS NOT NULL "
                                                          "ORDER BY played DESC LIMIT 2000"))
                    if sum(st) >= 100 and acc_lazer(st) < .995)
    return ratios[len(ratios) // 2] if len(ratios) >= 20 else .88


def acc_lazer(stats):
    """The accuracy lazer's results screen shows: PERFECT 305, GREAT 300 (acc320 counts PERFECT 320)."""
    p, g, gd, ok, meh, miss = stats
    n = p + g + gd + ok + meh + miss
    return (305 * p + 300 * g + 200 * gd + 100 * ok + 50 * meh) / (305 * n) if n else 0.0


def score_key(sha, total, stats, played=None):
    """One play across live/Realm sources, including identical achievements on different attempts."""
    ts = timestamp(played)
    return f"lz|{sha}|{total}|{','.join(str(int(x)) for x in stats)}" + (f"|{ts:.3f}" if ts is not None else "")


def upsert_score(db, row):
    """Insert, or fill in what a later source knows (online id, local id, status); never duplicates."""
    if not tracking_allowed(db, timestamp(row.get("played"))) or db.execute(
            "SELECT 1 FROM ignored_scores WHERE key=?", (row["key"],)).fetchone():
        return None
    if row["key"].startswith("lz|"):
        old = score_key(row["sha256"], row["total"], json.loads(row["stats"]))
        previous = db.execute("SELECT played FROM scores WHERE key=?", (old,)).fetchone()
        if previous and timestamp(previous[0]) == timestamp(row.get("played")):
            row["key"] = old           # retain existing event references; no destructive migration
    cols = list(row)
    db.execute(f"INSERT INTO scores({','.join(cols)}) VALUES ({','.join('?' * len(cols))}) "
               f"ON CONFLICT(key) DO UPDATE SET online_id=MAX(COALESCE(online_id,0), COALESCE(excluded.online_id,0)), "
               f"local_id=COALESCE(local_id, excluded.local_id), status=COALESCE(excluded.status, status), "
               f"beatmap_id=COALESCE(beatmap_id, excluded.beatmap_id), md5=COALESCE(md5, excluded.md5), "
               f"played=COALESCE(played, excluded.played), ranked=MAX(COALESCE(ranked,0), COALESCE(excluded.ranked,0))", [row[c] for c in cols])
    return row["key"]


def realm_rows(raw, user_id):
    """lazer_scores.js lines → (score rows of user_id, installed-map rows)."""
    scores, maps, now = [], [], time.strftime("%FT%T")
    for s in raw:
        if "map" in s:
            if s.get("keys") and s["map"]:
                maps.append((s["map"], s.get("beatmap_id") or 0, s.get("set_id") or 0, s.get("md5"), s.get("status"),
                             int(s["keys"]), s.get("od"), s.get("length"), s.get("title"), s.get("artist"), s.get("version")))
            continue
        if s.get("user_id") != user_id:
            continue                        # explicit account identity, not the most frequent name
        st = json.loads(s.get("stats") or "{}")
        stats = tuple(int(st.get(k, 0)) for k in ("perfect", "great", "good", "ok", "meh", "miss"))
        a, n = acc320(stats)
        mods = json.loads(s.get("mods") or "[]")
        rate, ranked = variant(mods)
        key = f"legacy|{s['legacy_online_id']}" if s.get("legacy") and (s.get("legacy_online_id") or 0) > 0 \
            else score_key(s["hash"], s["total"], stats, s.get("date"))
        scores.append({"key": key, "src": "realm", "local_id": s["id"], "online_id": max(0, s.get("online_id") or 0),
                       "player_id": user_id, "sha256": s["hash"], "md5": s.get("md5"), "beatmap_id": s.get("beatmap_id"),
                       "status": s.get("status"), "played": s.get("date"), "imported": now, "client": s.get("client"),
                       "mods": json.dumps(mods), "rate": rate, "ranked": int(ranked), "stats": json.dumps(stats),
                       "acc": a, "n": n, "total": s["total"], "pp": None, "fresh": 0})
    return scores, maps


def import_realm(db, realm=None, user_id=None):
    import lazer_scores
    try:
        raw = lazer_scores.dump(realm, maps=True)
    except Exception as exc:
        kv_set(db, "manifest_error", str(exc).splitlines()[0][:160])
        raise
    if user_id is None:
        users = collections.Counter(s.get("user_id") for s in raw if s.get("user_id") and "map" not in s)
        user_id = kv_get(db, "user_id") or (users.most_common(1)[0][0] if users else None)
        if user_id:
            kv_set(db, "user_id", user_id)
    scores, maps = realm_rows(raw, user_id)
    with db:
        before = db.execute("SELECT COUNT(*) FROM scores").fetchone()[0]
        for r in scores:
            upsert_score(db, r)
        # dump() requires its END marker: a successful empty manifest really is empty.
        db.execute("DELETE FROM installed")
        db.executemany("INSERT OR REPLACE INTO installed VALUES (?,?,?,?,?,?,?,?,?,?,?)", maps)
        db.execute("DELETE FROM installed_local")
        db.executemany("INSERT OR REPLACE INTO installed_local VALUES (?,?,?,?,?)",
                       [(s["map"], s.get("online_md5"), s.get("audio_sha"), int(bool(s.get("audio_required"))),
                         s.get("creator")) for s in raw if s.get("map")])
        after = db.execute("SELECT COUNT(*) FROM scores").fetchone()[0]
    kv_set(db, "realm_import", {"t": time.time(), "scores": len(scores), "new": after - before, "maps": len(maps)})
    origin = local_origin()
    if realm and os.path.realpath(realm) != os.path.join(origin["root"], "client.realm"):
        origin = dict(origin, root=os.path.dirname(os.path.realpath(realm)))
    kv_set(db, "manifest_origin", origin)
    kv_set(db, "manifest_error", None)
    import replays
    replays.archive(db, raw)
    return {"user_id": user_id, "seen": len(scores), "new": after - before, "installed": len(maps)}


# stable's ranked-status byte → lazer's BeatmapOnlineStatus (what `installed.status` holds)
STABLE_STATUS = {4: 1, 5: 2, 6: 3, 7: 4, 2: 0}


def stable_songs(root):
    """The Songs folder: osu!.<user>.cfg's BeatmapDirectory when it exists here, else <root>/Songs."""
    import glob
    for cfg in glob.glob(os.path.join(root, "osu!.*.cfg")):
        try:
            with open(cfg, encoding="utf-8-sig", errors="replace") as fh:
                for line in fh:
                    k, _, v = line.partition("=")
                    if k.strip() == "BeatmapDirectory" and v.strip():
                        d = os.path.join(root, v.strip())
                        if os.path.isdir(d):
                            return d
        except OSError:
            pass
    return os.path.join(root, "Songs")


def long_path(path):
    """Windows: stable's Songs paths often pass MAX_PATH (260); the long-path (\\\\?\\) form has no limit."""
    return "\\\\?\\" + os.path.abspath(path) if paths.WINDOWS and not path.startswith("\\\\?\\") else path


def stable_link(path, sha):
    """The stable_store path of a stable .osu (hard link; a copy across file systems)."""
    dest = local_file(sha, stable_store())
    if not os.path.isfile(dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        tmp = f"{dest}.{os.getpid()}.part"
        try:
            os.link(path, tmp)
        except OSError:
            shutil.copyfile(path, tmp)
        os.replace(tmp, dest)
    return dest


def stable_key(sha, total, stats):
    """One stable play from scores.db and from tosu's results screen alike (their timestamps differ)."""
    return f"st|{sha}|{total}|{','.join(str(int(x)) for x in stats)}"


def import_stable(db, root=None):
    """stable's osu!.db (mania charts) and scores.db (the account's mania scores), the import_realm of stable."""
    import native_backend
    nat = native_backend.load()
    root = root or paths.stable_data()
    try:
        player, charts = nat.stable_library(os.path.join(root, "osu!.db"))
    except Exception as exc:
        kv_set(db, "manifest_error", str(exc).splitlines()[0][:160])
        raise
    songs = stable_songs(root)
    charts = [c for c in charts if c["mode"] == 3 and c["md5"] and c["file"]]
    for c in charts:
        c["path"] = long_path(os.path.join(songs, c["folder"], c["file"]))
    db.execute("CREATE TABLE IF NOT EXISTS stable_files(path TEXT PRIMARY KEY, mtime REAL, size INTEGER, sha256 TEXT)")
    known = {r[0]: (r[1], r[2], r[3]) for r in db.execute("SELECT path, mtime, size, sha256 FROM stable_files")}
    stats, todo = {}, []
    for c in charts:
        try:
            st = os.stat(c["path"])
        except OSError:
            continue
        stats[c["path"]] = (st.st_mtime, st.st_size)
        k = known.get(c["path"])
        if not k or k[:2] != stats[c["path"]]:
            todo.append(c["path"])
    hashed = dict(zip(todo, nat.sha256_files(todo)))
    sha_of = {p: (hashed[p] if p in hashed else known[p][2]) for p in stats}
    maps, local, md5_sha, md5_bid = [], [], {}, {}
    for c in charts:
        sha = sha_of.get(c["path"])
        if not sha:
            continue
        stable_link(c["path"], sha)
        md5_sha[c["md5"]], md5_bid[c["md5"]] = sha, c["beatmap_id"] or None
        maps.append((sha, c["beatmap_id"] or 0, c["set_id"] or 0, c["md5"], STABLE_STATUS.get(c["status"], -3),
                     int(round(c["cs"])), c["od"], c["length"], c["title"], c["artist"], c["version"]))
        local.append((sha, c["md5"], None, 0, c["creator"]))
    user_id, now = kv_get(db, "user_id"), time.strftime("%FT%T")
    try:
        raw = nat.stable_scores(os.path.join(root, "scores.db"))
    except Exception:
        raw = []                                 # no plays yet: stable writes scores.db on the first one
    scores = []
    for s in raw:
        if s["mode"] != 3 or s["player"] != player or s["md5"] not in md5_sha:
            continue
        st, mods = tuple(s["stats"]), s["mods"]
        a, n = acc320(st)
        rate = 1.5 if mods & (64 | 512) else 0.75 if mods & 256 else 1.0
        ranked = not (mods & ~(1 | 2 | 8 | 64 | 256 | 512 | 1024 | 1048576 | 1073741824 | 16384 | 32))
        played = datetime.datetime.fromtimestamp(s["played"], datetime.timezone.utc).strftime("%FT%T")
        scores.append({"key": stable_key(md5_sha[s["md5"]], s["score"], st), "src": "stable", "local_id": None,
                       "online_id": max(0, s["online_id"]), "player_id": user_id, "sha256": md5_sha[s["md5"]],
                       "md5": s["md5"], "beatmap_id": md5_bid[s["md5"]], "status": None, "played": played, "imported": now,
                       "client": "stable", "mods": json.dumps(mods), "rate": rate, "ranked": int(ranked),
                       "stats": json.dumps(st), "acc": a, "n": n, "total": s["score"], "pp": None, "fresh": 0})
    with db:
        before = db.execute("SELECT COUNT(*) FROM scores").fetchone()[0]
        for r in scores:
            upsert_score(db, r)
        db.execute("DELETE FROM installed")
        db.executemany("INSERT OR REPLACE INTO installed VALUES (?,?,?,?,?,?,?,?,?,?,?)", maps)
        db.execute("DELETE FROM installed_local")
        db.executemany("INSERT OR REPLACE INTO installed_local VALUES (?,?,?,?,?)", local)
        db.execute("DELETE FROM stable_files")
        db.executemany("INSERT INTO stable_files VALUES (?,?,?,?)",
                       [(p, *stats[p], sha_of[p]) for p in stats if sha_of.get(p)])
        after = db.execute("SELECT COUNT(*) FROM scores").fetchone()[0]
    kv_set(db, "realm_import", {"t": time.time(), "scores": len(scores), "new": after - before, "maps": len(maps)})
    kv_set(db, "stable_player", player)
    ratio = stable_miss_ratio(db)
    kv_set(db, "stable_miss_ratio", ratio)
    set_client("stable", ratio)
    kv_set(db, "manifest_origin", local_origin())
    kv_set(db, "manifest_error", None)
    return {"user_id": user_id, "seen": len(scores), "new": after - before, "installed": len(maps)}


def import_library(db):
    """The active client's scores and installed maps."""
    return import_stable(db) if _client == "stable" else import_realm(db)


def import_public_user(db, pub):
    """The snapshot's stable high scores of this user (their own namespace; pp as the server had it)."""
    now = time.strftime("%FT%T")
    with db:
        for s in pub["user"]["scores"]:
            a, n = acc320(s["stats"])
            upsert_score(db, {"key": f"legacy|{s['score_id']}", "src": "public", "local_id": None, "online_id": s["score_id"],
                              "player_id": pub["user"]["id"], "sha256": None, "md5": pub["maps"].get(s["beatmap_id"], {}).get("md5"),
                              "beatmap_id": s["beatmap_id"], "status": pub["maps"].get(s["beatmap_id"], {}).get("approved"),
                              "played": s["date"], "imported": now, "client": "stable", "mods": json.dumps(s["mods"]),
                              "rate": s["rate"], "ranked": int(s["ranked"]), "stats": json.dumps(s["stats"]), "acc": a,
                              "n": n, "total": s["score"], "pp": s["pp"], "fresh": 0})


def _website_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "maniascope (personal practice tool)",
                                               "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.geturl(), resp.read()


def import_website_best(db, username):
    """The player's top 100 mania plays from their public osu! profile (the website's own JSON; no API key,
    user 2026-10-01): one lookup + one request. Bests outside the top 100 stay unknown (below #100)."""
    final, page = _website_json(f"https://osu.ppy.sh/users/{urllib.parse.quote(str(username).strip())}")
    # The profile page names its numeric id in the canonical link (a redirect may also carry it).
    m = re.search(r'rel="canonical" href="https://osu\.ppy\.sh/users/(\d+)"', page.decode("utf-8", "replace")) \
        or re.search(r"/users/(\d+)$", final)
    if not m:
        raise ValueError(f"osu! user {username!r} not found")
    user_id = int(m.group(1))
    _, body = _website_json(f"https://osu.ppy.sh/users/{user_id}/scores/best?mode=mania&limit=100")
    now = time.strftime("%FT%T")
    count = 0
    with db:
        for s in json.loads(body):
            st, b = s.get("statistics") or {}, s.get("beatmap") or {}
            stats = tuple(int(st.get(k, 0) or 0) for k in ("perfect", "great", "good", "ok", "meh", "miss"))
            rate, ranked = variant(s.get("mods"))
            if rate is None or not s.get("pp"):
                continue
            a, n = acc320(stats)
            upsert_score(db, {"key": f"web|{s['id']}", "src": "public", "local_id": None, "online_id": s["id"],
                              "player_id": user_id, "sha256": None, "md5": b.get("checksum"),
                              "beatmap_id": s.get("beatmap_id"), "status": b.get("ranked"),
                              "played": (s.get("ended_at") or now).replace("Z", "").split("+")[0], "imported": now,
                              "client": "website", "mods": json.dumps(s.get("mods") or []), "rate": rate,
                              "ranked": int(ranked), "stats": json.dumps(stats), "acc": a, "n": n,
                              "total": s.get("total_score") or 0, "pp": s["pp"], "fresh": 0})
            count += 1
    kv_set(db, "website_user", {"name": username, "id": user_id, "t": time.time(), "scores": count})
    return user_id, count


def add_live_score(db, score, path, fresh, play=None):
    """A results-screen observation from tosu. fresh=False: a past score being viewed (history only)."""
    h = score.get("hits") or {}
    stats = tuple(int(h.get(k, 0) or 0) for k in ("geki", "300", "katu", "100", "50", "0"))
    a, n = acc320(stats)
    mods = score.get("mods_list")
    rate, ranked = variant(mods) if mods is not None else (score.get("rate"), False)    # [] is NoMod: ranked
    mods = mods or []
    if not score.get("played") and fresh:
        score = dict(score, played=datetime.datetime.now(datetime.timezone.utc).isoformat())
    if not score_allowed(db, score, play if fresh else None, exclude=True):
        return None, False
    stable = score.get("client") == "stable"
    key = (stable_key(score["sha256"], score.get("score") or 0, stats) if stable
           else score_key(score["sha256"], score.get("score") or 0, stats, score.get("played")))
    with db:
        known = db.execute("SELECT fresh FROM scores WHERE key=?", (key,)).fetchone()
        old = db.execute("SELECT played, fresh FROM scores WHERE key=?", (
            score_key(score["sha256"], score.get("score") or 0, stats),)).fetchone()
        if old and timestamp(old["played"]) == timestamp(score.get("played")):
            known = [old["fresh"]]
        key = upsert_score(db, {"key": key, "src": "live", "local_id": None, "online_id": 0, "player_id": kv_get(db, "user_id"),
                          "sha256": score["sha256"], "md5": score.get("md5"), "beatmap_id": score.get("beatmap_id"),
                          "status": score.get("status"),
                          "played": score.get("played") or time.strftime("%FT%T"), "imported": time.strftime("%FT%T"),
                          "client": "stable-live" if stable else "lazer-live", "mods": json.dumps(mods), "rate": rate,
                          "ranked": int(ranked),
                          "stats": json.dumps(stats), "acc": a, "n": n, "total": score.get("score") or 0,
                          "pp": None, "fresh": int(fresh)})
        if fresh and known is not None and not known[0]:
            db.execute("UPDATE scores SET fresh=1 WHERE key=?", (key,))
    return key, known is None


# ---------------------------------------------------------------------------
# chart features (shared by the public build and personal scores)
# ---------------------------------------------------------------------------
# Exact reviewed implementation digests only. The speed-only rewrite reproduced
# all 422 full results (450,238 numeric fields) bit-for-bit against c1021be2829b.
# Keeping that semantic identity avoids invalidating compatible prediction data
# or breaking an old viewer's subprocess while it remains open during sync.
# ANY further source/parameter/skill-schema edit misses this full-hash allowlist
# and receives its own identity as before. Never alias a changed model by version.
_EXACT_IMPLEMENTATIONS = {
    # source digest → earlier calc ID, for refactors proven exact on the parity set.
    # lib.rs left the identity (it only registers modules; fit.rs added): calculator sources unchanged.
    "ee9cce9ecc7d8c5d4db94092ebd268d5d795b6c1797c53f2b244cf13956dbc84": "3c695ee7a23f",
    # Modules moved to app/: difficulty_model finds calib/difficulty.json via native_backend.ROOT, same file.
    "df6725a18a949a82f1214a8a39299c8ecdf4aec44c819a3052c234e3ee309e66": "3c695ee7a23f",
}


# The Python faces of the native calculator (native_backend.calc_id hashes them with the crate).
CALCULATOR_MODULES = ('skill_calc', 'difficulty_model', 'structural_residual', 'chart_structure', 'score_units')


def _calculator_identity(paths):
    h = hashlib.sha256()
    for path in paths:
        if os.path.isfile(path):
            with open(path, 'rb') as fh:
                h.update(fh.read())
    h.update(repr(SKILLS).encode())
    digest = h.hexdigest()
    return _EXACT_IMPLEMENTATIONS.get(digest, digest[:12])


@lru_cache(maxsize=1)
def calc_id():
    """Identity of the calculator loaded by this process.

    Dropbox/git may replace source files while the app is open. Re-hashing them
    on every score would label the already-imported old implementation as new.
    A normal new process is required to activate a new calculator/data bundle.
    """
    import native_backend
    return native_backend.calc_id()


SKILLS = ("stream", "jumpstream", "handstream", "jackspeed", "chordjack", "minijack", "chordstream", "delay",
          "bracket", "jack", "ln", "release", "inverse", "hybrid", "shield", "sv", "technical", "vibro", "stamina",
          "dump", "jumptrill", "splittrill", "longjack", "patterntech", "rhythmtech", "trill1h", "anchor", "mash",
          "quadstream", "sv_fast", "sv_slow", "sv_accel", "sv_stutter", "sv_brake")


def identity(f):
    """What a per-chart cache compares instead of the features object: a website chart's features are
    decoded afresh on each read (webmaps.Decoded), their stored bytes are not."""
    return getattr(f, "source", f)


def per_chart(cache, f, compute):
    """compute(f), once per chart features object (by identity()). Ranking the NPS library asks the
    same chart's skill group ~10 times per request (~1.5M calls, G835LX library)."""
    token = identity(f)
    hit = cache.get(id(token))
    if hit is not None and hit[0] is token:
        return hit[1]
    value = compute(f)
    if len(cache) > 200000:
        cache.clear()
    cache[id(token)] = (token, value)
    return value


def feats_from(keys, sc, levels, ln, notes, od, length_ms):
    o = sc.get("overall", 0) or 0
    return {"keys": keys, "overall": o, "od": od, "ln": ln, "notes": notes, "length": length_ms,
            "stam": ((levels or {}).get(120.0) or (levels or {}).get("120.0") or o) / o if o else 1.0,
            "sk": {k: round(sc[k] / o, 4) for k in SKILLS if o and sc.get(k)}}


def chart_feats(path, rates, anchors=None, *, analyses=None):
    """→ {rate: feats} for one chart; ranked variants (0.75 / 1 / 1.5) also get lazer stars and judgement count.
    analyses: precomputed skill_calc results by rate (no recalculation)."""
    out = feats_batch([(path, rates, anchors)], analyses=analyses)[0]
    if isinstance(out, Exception):
        raise out
    return out


def feats_batch(items, threads=0, *, analyses=None):
    """chart_feats for many charts: [(path, rates, anchors)] → per item {rate: feats} or the Exception.
    One native batch computes every missing chart-rate on `threads` niced threads without the GIL."""
    import skill_calc
    needs, datas = [], []
    for path, rates, anchors in items:
        need = [r for r in rates if not (anchors or {}).get(r) and (analyses or {}).get(r) is None]
        needs.append(need)
        if not need:
            datas.append(None)
            continue
        try:
            if not path:
                raise FileNotFoundError("no local file for this chart")   # one bad chart, not the whole batch
            with open(path, "rb") as fh:
                datas.append(fh.read())
        except OSError as exc:
            datas.append(exc)
    jobs = [(d, need) for d, need in zip(datas, needs) if isinstance(d, bytes)]
    computed = iter(skill_calc.compute_batch(jobs, threads=threads) if jobs else ())
    out = []
    for (path, rates, anchors), need, data in zip(items, needs, datas):
        if isinstance(data, OSError):
            out.append(data)
            continue
        results = dict(analyses or {})
        if isinstance(data, bytes):
            got = next(computed)
            if isinstance(got, str):
                out.append(skill_calc.ChartError(got))
                continue
            results.update(zip(need, got))
        try:
            out.append({r: _feats_at(path, r, (anchors or {}).get(r), results.get(r)) for r in rates})
        except Exception as exc:          # one broken chart is one missing candidate
            out.append(exc)
    # Lazer stars and judgement count for the ranked variants, on the same background threads.
    starred = [(i, [r for r in fs if r in (0.75, 1.0, 1.5) and "stars" not in fs[r]]) for i, fs in enumerate(out)
               if isinstance(fs, dict)]
    starred = [(i, rs) for i, rs in starred if rs]
    for (i, rs), got in zip(starred, skill_calc.N.lazer_stars_batch(
            [(datas[i] if isinstance(datas[i], bytes) else _read(items[i][0]), rs) for i, rs in starred], threads)):
        for r, (stars, hits) in zip(rs, got or ()):   # None: rosu-pp can't read it; stars stay optional
            out[i][r]["stars"], out[i][r]["hits"] = stars, hits
    return out


def _read(path):
    try:
        with open(path, "rb") as fh:
            return fh.read()
    except OSError:
        return b""


def _feats_at(path, rate, anchor, res):
    import skill_calc
    import nps
    f = dict(anchor or {})
    if not f:
        demand = res.get('preunit_result', res)
        od = res["od"] if "od" in res else __import__("feedback")._od(path)
        f = feats_from(res["keys"], demand["scores"], demand["levels"] if res["horizon"] else None,
                       res["ln_notes"] / max(1, res["notes"]), res["notes"], od, res.get("length", 0))
        # The fitted predictor's sustain coordinate predates the horizon
        # display-unit repair. Preserve that schema until a validated refit.
        if res['horizon'] and not res.get('rolled_correction'):
            f['stam'] /= rate ** skill_calc.RATE_G
        f["description"] = skill_calc.describe(res)[0] if res["horizon"] else "Sparse"
        f["endurance"] = res.get("endurance", {})
        f["raw_overall"] = res.get("raw_overall", f["overall"])
        f["calibration_features"] = res.get("calibration_features", [])
        f["execution"] = res.get("execution", {})
        if res.get('structure'):
            f['nps'] = res['structure']
        f['baseline_overall'] = demand.get('baseline_overall', f['overall'])
        f['tech_profile'] = res.get('tech_profile', {})
        f['archetype_profile'] = res.get('archetype_profile', {})
        import score_units, difficulty_model
        f = score_units.feature(f, difficulty_model.parameters())
    if 'nps' not in f:
        f["nps"] = nps.structure(skill_calc.parse_osu(path), rate)
    return f


def predictable_context(mods, rate):
    """Only known constant-rate, unmodified-chart contexts enter live residuals."""
    if not isinstance(rate, (float, int)) or not math.isfinite(rate) or rate <= 0 or not isinstance(mods, list):
        return False
    allowed = {"NF", "HD", "FI", "FL", "MR", "SD", "PF", "DT", "NC", "HT", "DC"}
    for mod in mods:
        if not isinstance(mod, dict) or mod.get("acronym", "").upper() not in allowed:
            return False
        if set(mod.get("settings") or {}) - {"speed_change", "adjust_pitch"}:
            return False
    return True


def _cand_feats(jobs, threads=0):
    """[(bid, path, md5)] → [(bid, {rate: feats} or the error text)] for the ranked variants (0.75 / 1 / 1.5)."""
    checked, out = [], []
    for bid, path, md5 in jobs:
        try:
            with open(path, "rb") as fh:
                ok = hashlib.md5(fh.read()).hexdigest() == md5
        except OSError as exc:
            out.append((bid, f"{type(exc).__name__}: {exc}"))
            continue
        if ok:
            checked.append((bid, path))
        else:
            out.append((bid, "md5 mismatch"))
    for (bid, _p), fs in zip(checked, feats_batch([(p, (0.75, 1.0, 1.5), None) for _b, p in checked], threads)):
        out.append((bid, f"{type(fs).__name__}: {fs}" if isinstance(fs, Exception) else fs))
    return out


# ---------------------------------------------------------------------------
# public build: population fit, map effects, candidate table, the user's public record
# ---------------------------------------------------------------------------
# Map difficulty relative to the player (log stars − the player's competence-zone mean log stars).
# Jacks are cheap below a player's speed and punishing at and above it; LN keeps costing far below the
# player's level and costs less than rated near it (user 2026-10-01). Production-path DEV, all plays:
# 4K RMSE .737 → .711, 7K .495 → .490. A steeper JACK_P in the stars was worse on 4K DEV: the effect
# depends on the player, so it lives in the accuracy link, not the rating. A curve shared by every map
# was rejected: fitted in the competence zone, it extrapolated badly to easy maps (DEV 4K/7K worse).
# Other keymodes borrow 7K's curves: their own fits overfit (DEV 5K/6K/10K worse), 7K's improve every
# one of them (DEV RMSE 5K .906 → .818, 6K .653 → .622, 8K .773 → .718, 9K .629 → .563, 10K .558 → .525).
# Bumped when fit_population's links change meaning; packaged installs re-download older evidence.
POP_VERSION = 2
GAP_KNOTS = (-.3, -.15, 0., .15, .3)
GAP_SHARED = 7
# Execution families (Technical describes the same work: no curve of its own), "jacks" = every jack
# kind together, "all" = 1 for every chart.
GAP_FAMILIES = ("rice", "chords", "chordjack", "jack", "ln", "trill", "sv")
GAP_SHARE_PARTS = GAP_FAMILIES + ("jacks", "all")
# Level curves per keymode, chosen by calib/pattern_scan.py on DEV (docs/REPORT_2026-10-01_PATTERN_SCAN.md):
# part → centred on the keymode's mean share. Centred, a curve means "more of this family than usual";
# raw rice/chords shares (≈ 1 on most 7K maps) copied the star slope and soaked it up (7K slope
# 1.41 → 0.25, DEV worse). "all" is a bump around the player's level: inner knots only, the ends stay
# with the slope (all five knots, or knots out to −.9, were worse on DEV).
# 4K: raw jack/LN curves plus centred SV beat all families centred (DEV RMSE .7085 vs .7183).
GAP_SPEC = {4: {"jacks": False, "ln": False, "sv": True},
            7: dict({p: True for p in GAP_FAMILIES}, all=False)}
# Keymodes that borrow 7K's curves: DEV 5K/6K slightly better, 8K–10K worse (all on 7K's link).
GAP_BORROW = (5, 6)


def gap_basis(g):
    """Piecewise-linear hats over GAP_KNOTS (clamped beyond the ends), for a float or an array."""
    if isinstance(g, (float, int)):       # every prediction: plain floats, numpy per scalar cost 30 % of a refit
        g = min(max(g, GAP_KNOTS[0]), GAP_KNOTS[-1])
        out = []
        for i, k in enumerate(GAP_KNOTS):
            lo = GAP_KNOTS[i-1] if i else k - 1.
            hi = GAP_KNOTS[i+1] if i + 1 < len(GAP_KNOTS) else k + 1.
            out.append(min(1., max(0., min((g - lo) / (k - lo), (hi - g) / (hi - k)))))
        return out
    import numpy as np
    g = np.clip(g, GAP_KNOTS[0], GAP_KNOTS[-1])
    out = []
    for i, k in enumerate(GAP_KNOTS):
        lo = GAP_KNOTS[i-1] if i else k - 1.
        hi = GAP_KNOTS[i+1] if i + 1 < len(GAP_KNOTS) else k + 1.
        out.append(np.clip(np.minimum((g - lo) / (k - lo), (hi - g) / (hi - k)), 0., 1.))
    return out


_GAP_MEMBERS = {}


def _gap_members():
    import warmup                         # warmup imports recdata: filled on first use
    _GAP_MEMBERS.update({p: tuple(warmup.MEMBERS[p]) for p in GAP_FAMILIES},
                        jacks=("chordjack", "jackspeed", "minijack", "longjack"))


def _gap_share(s, part):
    """Strongest member skill / overall of one part (sk values are already relative to overall)."""
    if part == "all":
        return 1.
    if not _GAP_MEMBERS:
        _gap_members()
    best = 0.
    for x in _GAP_MEMBERS[part]:
        v = s.get(x, 0.)
        if v > best:
            best = v
    return min(1., best)


def feats_dump(f):
    """A feature dict as rec.db `feats` stores it: the native binary form (codec.rs), half JSON's decode time."""
    import native_backend
    return native_backend.load().encode_feats(f)


def feats_load(data):
    """A rec.db `feats` row → its feature dict; rows written before the binary form are JSON text."""
    if isinstance(data, bytes):
        import native_backend
        return native_backend.load().decode_feats(data)
    return json.loads(data)


def group_ids(*cols):
    """(id per row, distinct value columns) of the rows' value tuples; np.unique(axis=0) sorts bytes, 5× slower."""
    import numpy as np
    order = np.lexsort(cols[::-1])
    new = np.zeros(len(order), bool)
    new[:1] = True
    for c in cols:
        new[1:] |= c[order][1:] != c[order][:-1]
    ids = np.empty(len(order), np.int64)
    ids[order] = np.cumsum(new) - 1
    return ids, [c[order[new]] for c in cols]


def _first_seen(ids):
    """(distinct ids in order of first appearance, each one's first row)."""
    import numpy as np
    first = np.full(ids.max() + 1 if len(ids) else 0, len(ids))
    np.minimum.at(first, ids, np.arange(len(ids)))
    order = np.argsort(first, kind="stable")
    return order, first[order]


def _group_percentile(ids, values, q):
    """np.percentile(values of each id, q) (linear method) for every id at once."""
    import numpy as np
    order = np.lexsort((values, ids))
    v, count = values[order], np.bincount(ids)
    start = np.concatenate([[0], np.cumsum(count)[:-1]])
    pos = (count - 1) * (q / 100)
    lo = np.floor(pos).astype(np.int64)
    t = pos - lo
    a, c = v[start + lo], v[start + np.minimum(lo + 1, count - 1)]
    diff = c - a
    return np.where(t >= .5, c - diff * (1 - t), a + diff * t)      # numpy's _lerp


_SCORES = [None, None]


def _score_columns(scores):
    """(player, beatmap, rate, year, yyyymmdd, judgement counts) of the dump scores as arrays, once per list."""
    import numpy as np
    if _SCORES[0] is not scores:
        _SCORES[:] = scores, (np.array([s[0] for s in scores]), np.array([s[1] for s in scores]),
                              np.array([s[3] for s in scores], float), np.array([s[4] for s in scores]),
                              np.array([s[6] for s in scores]),
                              np.array([s[7][:6] for s in scores], np.int64).reshape(len(scores), 6))
    return _SCORES[1]


def gap_shares(f):
    """Share of each GAP_SHARE_PARTS part in a chart (strongest member skill / overall), the weights of its curves."""
    s = f.get("sk", {})
    return tuple(_gap_share(s, p) for p in GAP_SHARE_PARTS)


def competence_level(log_ratings):
    """Mean log stars of the plays within 70 % of the 90th percentile: where the player plays."""
    import numpy as np
    lr = np.asarray(log_ratings, float)
    return float(lr[lr >= np.percentile(lr, 90) + math.log(.7)].mean())


def gap_term(pop, f, level):
    """Link shift from the level curves for a chart played by a player at `level` (log stars)."""
    gap, means = pop.get("gap", {}), pop.get("share_mean", {})
    k = f["keys"] if f["keys"] in gap else GAP_SHARED if f["keys"] in GAP_BORROW else None
    if k not in gap or level is None:
        return 0.
    s, mean = f.get("sk", {}), means.get(k, {})
    hats = gap_basis(math.log(max(.05, f["overall"])) - level)
    return float(sum((_gap_share(s, p) - mean.get(p, 0.)) * sum(c * h for c, h in zip(cs, hats))
                     for p, cs in gap[k].items()))


def box_cd(A, rhs, x0, lower, upper, sweeps, tol):
    """Projected Gauss-Seidel for a bounded convex ridge fit (native: the Python sweeps were seconds per fit)."""
    import native_backend
    import numpy as np
    return np.array(native_backend.load().box_cd(np.ascontiguousarray(A, float).tobytes(), [float(x) for x in rhs],
                                                 [float(x) for x in x0], [float(x) for x in lower],
                                                 [float(x) for x in upper], sweeps, tol))


def fit_population(scores, maps, feats, exclude_user, date_cut=None, *, accuracy='pp',
                   ceilings=False, specialization=False, spec=None):
    """y = log(1 − acc320): player level + monotone rating curve + hit window + map effect.
    Competence zone as in calib/eval.py (rows within 70 % of the player-keymode-year's 90th percentile).
    → dict of slopes, window coefficients, map effects (per beatmap and per beatmap@rate) and variances.

    The public snapshot contains stable judgements. 'lazer' applies 305 weights
    to those counts; it cannot reconstruct separate hold heads/tails. This is
    a transfer prior, not a native lazer hold-score dataset. Personal models
    require native hold judgements for both displayed and PP-target accuracy.
    """
    import numpy as np
    if accuracy not in ('pp', 'lazer'):
        raise ValueError('Unknown accuracy target')
    # Columns over the 3M dump scores; features, gap shares and skills are read once per chart variant.
    u, b, rate, year, ymd, stats = _score_columns(scores)
    keep = u != exclude_user
    if date_cut:
        keep &= ymd < date_cut
    idx = np.flatnonzero(keep)
    pid, (pb, pr) = group_ids(b[idx], rate[idx])
    variants = [feats.get(x, {}).get(r) for x, r in zip(pb.tolist(), pr.tolist())]
    usable = np.array([bool(f) and f["overall"] >= 1.8 and f["notes"] >= 300 for f in variants], bool)
    idx, pid = idx[usable[pid]], pid[usable[pid]]
    variant = lambda key, default=0.: np.array([f[key] if f else default for f in variants], float)[pid]
    overall, od, keys = variant("overall"), variant("od"), variant("keys").astype(np.int64)
    st = stats[idx]
    n = st.sum(axis=1)
    weights = (305, 300, 200, 100, 50) if accuracy == 'lazer' else (320, 300, 200, 100, 50)
    with np.errstate(invalid="ignore", divide="ignore"):
        acc = np.where(n > 0, (st[:, :5] @ np.array(weights)) / (weights[0] * n), 0.)
    # Competence zone per player × keymode × year (groups in order of first appearance, as the rows were).
    gid, _ = group_ids(u[idx], keys, year[idx])
    order, first = _first_seen(gid)
    rank = np.empty(len(order), np.int64); rank[order] = np.arange(len(order))
    count = np.bincount(gid)
    top = _group_percentile(gid, overall, 90.)
    ok = ((count[gid] >= 20) & (overall >= 0.7 * top[gid]) & (0.75 <= acc) & (acc <= (1. if ceilings else .998)))
    rows = np.flatnonzero(ok)
    rows = rows[np.lexsort((rows, rank[gid[rows]]))]
    K = keys[rows]
    ks = sorted(k for k, n in collections.Counter(K.tolist()).items() if n >= 1000 and k in (4, 7))
    # 5K/6K/8K–10K share 7K's link: their own fits came out far too flat (5K slope .64, 9K .35 vs 7K
    # 1.41) and 7K's link cut DEV RMSE 5K .906→.756, 8K .773→.604, 9K .629→.490 (2026-10-01).
    shared = ks.index(GAP_SHARED) if GAP_SHARED in ks else len(ks)
    kg = np.full(len(K), shared)
    for i, k in enumerate(ks):
        kg[K == k] = i
    losses=1.-acc[rows]
    if ceilings:
        counts=n[rows].astype(float)
        smallest=5./305. if accuracy=='lazer' else 20./320.
        losses=(counts*losses+.5*smallest)/(counts+1.)
    y = np.log(losses)
    lr = np.log(overall[rows])
    bend = np.maximum(0., lr - math.log(4.)) ** 2
    lw = np.log(np.maximum(10.0, 64 - 3 * od[rows]))
    ug = np.unique(rank[gid[rows]], return_inverse=True)[1].reshape(-1)
    n_u = np.bincount(ug)
    dm = lambda v: v - (np.bincount(ug, v, minlength=len(n_u)) / n_u)[ug]
    G = len(ks) + 1
    level = np.bincount(ug, lr) / n_u
    hats = gap_basis(lr - level[ug])
    spec = GAP_SPEC if spec is None else spec
    gap_keys = [k for k in spec if k in ks]
    shares = np.array([gap_shares(f) if usable[i] else (0.,) * len(GAP_SHARE_PARTS)
                       for i, f in enumerate(variants)])[pid[rows]]
    ix = {p: i for i, p in enumerate(GAP_SHARE_PARTS)}
    share_mean = {k: {p: float(shares[K == k, ix[p]].mean()) if centred else 0. for p, centred in spec[k].items()}
                  for k in gap_keys}
    inner = lambda p: range(1, len(GAP_KNOTS) - 1) if p == "all" else range(len(GAP_KNOTS))
    gap_cols = [(k, p, j) for k in gap_keys for p in spec[k] for j in inner(p)]
    Xr = np.column_stack([lr * (kg == i) for i in range(G)] + [lw * (kg == i) for i in range(G)]
                         + [bend * (kg == i) for i in range(G)]
                         + [(shares[:, ix[p]] - share_mean[k][p]) * hats[j] * (K == k) for k, p, j in gap_cols])
    X = np.column_stack([dm(c) for c in Xr.T])
    # Small convex bounded ridge fit. The rating link cannot turn down, and a
    # wider timing window cannot make an otherwise identical chart harder.
    A, rhs = X.T @ X + np.eye(X.shape[1]) * max(1., len(rows)*.0001), X.T @ dm(y)
    nuisance=None
    if specialization:
        import population_response
        z=np.array([population_response.skills(f) if usable[i] else (0.,)*5
                    for i,f in enumerate(variants)])[pid[rows]]
        gram,projected,nuisance=population_response.eliminate(X,dm(y),z,ug)
        A-=gram;rhs-=projected
    lower = np.concatenate([np.full(G, .2), np.full(G, -np.inf), np.zeros(G), np.full(len(gap_cols), -np.inf)])
    upper = np.concatenate([np.full(G, np.inf), np.zeros(G), np.full(G, np.inf), np.full(len(gap_cols), np.inf)])
    coef = box_cd(A, rhs, np.clip(np.linalg.solve(A, rhs), lower, upper), lower, upper, 1500, 1e-9)
    specialized,spread=nuisance(coef) if nuisance is not None else (np.zeros(len(rows)),[])
    a = np.bincount(ug, y - Xr @ coef-specialized, minlength=len(n_u)) / n_u
    e = y - Xr @ coef - a[ug]-specialized
    # map effects: residual mean per beatmap, then per rate; shrunk toward the map / zero
    bb = b[idx][rows]
    rr = rate[idx][rows]
    bid, (ub,) = group_ids(bb)
    nb, sb = np.bincount(bid), np.bincount(bid, e)
    sig2 = float(np.var(e))
    seen, _ = _first_seen(bid)
    tau2 = max(1e-4, float(np.mean((sb / nb)[seen] ** 2 - sig2 / nb[seen])))
    kap = sig2 / tau2
    mbv = sb / (nb + kap)
    mb = dict(zip(ub.tolist(), zip(mbv.tolist(), nb.tolist())))
    brid, (ubb, urr) = group_ids(bb, rr)
    nbr, sbr = np.bincount(brid), np.bincount(brid, e)
    mbr = dict(zip(zip(ubb.tolist(), urr.tolist()),
                   zip(((sbr + kap * mbv[np.searchsorted(ub, ubb)]) / (nbr + kap)).tolist(), nbr.tolist())))
    slope = {k: float(coef[i]) for i, k in enumerate(ks)}
    window = {k: float(coef[G + i]) for i, k in enumerate(ks)}
    curve = {k: float(coef[2*G + i]) for i, k in enumerate(ks)}
    # 0 = any other keymode
    slope[0], window[0], curve[0] = (float(coef[shared]), float(coef[G + shared]), float(coef[2*G + shared]))
    gap = {k: {p: [0.] * len(GAP_KNOTS) for p in spec[k]} for k in gap_keys}
    for (k, p, j), c in zip(gap_cols, coef[3*G:]):
        gap[k][p][j] = float(c)
    return {"version": POP_VERSION, "slope": slope, "window": window, "curve": curve, "curve_knee": 4., "gap": gap,
            "share_mean": share_mean,
            "mb": mb, "mbr": mbr, "sig2": sig2, "tau2": tau2, "kappa": kap,
            "rows": len(rows), "players": len(np.unique(u[idx][rows])),
            "accuracy_target":'lazer305' if accuracy=='lazer' else 'pp320',
            "ceilings_retained":bool(ceilings),'specialization_sd':spread}


def parse_user(D, user_id):
    """The user's stable high scores (with pp, all mods), stats row and stable play counts."""
    row = re.compile(r"\((\d+),(\d+)," + str(user_id) +
                     r",(\d+),(\d+),'(\w+)',(\d+),(\d+),(\d+),(\d+),(\d+),(\d+),\d,(\d+),'([^']+)',([\d.]+|NULL),\d+,(\d+)")
    out = []
    with open(os.path.join(D, "osu_scores_mania_high.sql"), encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if str(user_id) in line:
                out += row.findall(line)
    scores = []
    for o in out:
        mods = int(o[11])
        c50, c100, c300, miss, geki, katu = (int(o[i]) for i in (5, 6, 7, 8, 9, 10))
        rate = 1.5 if mods & (64 | 512) else 0.75 if mods & 256 else 1.0
        # stable bits: NF 1, EZ 2, HD 8, DT 64, HT 256, NC 512, FL 1024, FI 1048576, MR 1073741824 are ranked
        ranked = not (mods & ~(1 | 2 | 8 | 64 | 256 | 512 | 1024 | 1048576 | 1073741824 | 16384 | 32))
        scores.append({"score_id": int(o[0]), "beatmap_id": int(o[1]), "score": int(o[2]), "stats": (geki, c300, katu, c100, c50, miss),
                       "mods": mods, "rate": rate, "ranked": ranked, "date": o[12].replace(" ", "T"),
                       "pp": None if o[13] == "NULL" else float(o[13]), "hidden": int(o[14])})
    stats = None
    with open(os.path.join(D, "osu_user_stats_mania.sql"), encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = re.search(r"\(" + str(user_id) + r",((?:[^,()]*,){22})([\d.]+),(\d+),[\d.]+,'([^']+)'", line)
            if m:
                stats = {"rank_score": float(m.group(2)), "rank": int(m.group(3)), "last_update": m.group(4)}
                break
    plays = {}
    with open(os.path.join(D, "osu_user_beatmap_playcount.sql"), encoding="utf-8", errors="replace") as fh:
        for line in fh:
            for m in re.finditer(r"\(" + str(user_id) + r",(\d+),(\d+)\)", line):
                plays[int(m.group(1))] = int(m.group(2))
    return {"id": user_id, "scores": scores, "stats": stats, "playcount": plays}


def build_public(D, out_dir, scores, maps, user_id, charts=None, prev=None, workers=None, date_cut=None):
    """Everything the app needs from a public snapshot, in one pickle. charts(bid) → path or None.
    prev: an earlier public.pkl whose chart features (same md5 + calculator identity) are reused."""
    cid = calc_id()
    feats, jobs = {}, []
    old = (prev or {}).get("feats", {})
    for b, m in maps.items():
        o = old.get(b)
        if o and o.get("md5") == m[5] and o.get("calc") == cid:
            feats[b] = o
        elif charts and charts(b):
            jobs.append((b, charts(b), m[5]))
    print(f"features: {len(feats)} reused, {len(jobs)} charts to analyse", flush=True)
    bad = collections.Counter()
    threads = workers or max(1, (os.cpu_count() or 2) - 2)
    for i in range(0, len(jobs), 256):          # in-process native batches on niced threads
        for b, f in _cand_feats(jobs[i:i + 256], threads):
            if isinstance(f, str):
                bad[f.split(":")[0]] += 1
            else:
                feats[b] = dict(f, md5=maps[b][5], calc=cid)
        print(f"  {min(len(jobs), i + 256)}/{len(jobs)}", flush=True)
    print("failed charts:", dict(bad), flush=True)
    fr = {b: {r: v for r, v in f.items() if isinstance(r, float)} for b, f in feats.items()}
    pop = fit_population(scores, maps, fr, user_id, date_cut)
    pop_lazer = fit_population(scores, maps, fr, user_id, date_cut, accuracy="lazer",
                               ceilings=True, specialization=True)
    user = parse_user(D, user_id)
    pub = {"snapshot": os.path.basename(os.path.normpath(D)), "built": time.strftime("%FT%T"), "calc": cid,
           "pop": pop, "pop_lazer": pop_lazer, "feats": feats, "user": user,
           "maps": {b: {"keys": int(m[0]), "od": m[1], "set": m[2], "approved": m[3], "file": m[4], "md5": m[5]} for b, m in maps.items()}}
    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir, "public.pkl.part")
    with open(tmp, "wb") as fh:
        pickle.dump(pub, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, os.path.join(out_dir, "public.pkl"))
    return pub


def load_public():
    try:
        with open(ACTIVE, "r", encoding="utf-8") as fh:
            act = json.load(fh)
        # A versioned sibling may be installed while an older viewer is still
        # running. It keeps its original public.pkl and in-memory model; a new
        # launch picks the bundle matching its exact numerical implementation.
        directory=os.path.join(ROLLING,act['name'])
        cid=calc_id()
        versioned=os.path.join(directory,'calculators',cid,'public.pkl')
        if os.path.isfile(versioned):
            with open(versioned,'rb') as fh:
                candidate=pickle.load(fh)
            if candidate.get('calc')==cid and candidate.get('snapshot')==act['name']:
                return candidate
        with open(os.path.join(directory, "public.pkl"), "rb") as fh:
            return pickle.load(fh)
    except (OSError, ValueError, KeyError, EOFError, pickle.UnpicklingError):
        return None


RELEASE_REPO = os.environ.get("MANIASCOPE_RELEASE_REPO", "robby250/maniascope-app")
RELEASE_URL = os.environ.get("MANIASCOPE_RELEASE_URL", f"https://github.com/{RELEASE_REPO}/releases/latest/download")


def fetch_release_public(progress=None):
    """The published population evidence for this exact calculator (derived statistics, never the dump;
    the owner's own scores are stripped at release). progress(done_bytes, total_bytes) per chunk. → pub."""
    cid = calc_id()
    url = f"{RELEASE_URL}/public-{cid}.pkl"
    chunks, done = [], 0
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "maniascope"}), timeout=60) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        while chunk := resp.read(1 << 20):
            chunks.append(chunk)
            done += len(chunk)
            if progress:
                progress(done, total)
    data = b"".join(chunks)
    pub = pickle.loads(data)
    if pub.get("calc") != cid:
        raise ValueError("published evidence belongs to another calculator")
    # Versioned beside any personal public.pkl (load_public prefers it): a source install on the
    # builder's own machines must never have its unstripped evidence replaced by the release copy.
    directory = os.path.join(ROLLING, pub["snapshot"], "calculators", cid)
    os.makedirs(directory, exist_ok=True)
    tmp = os.path.join(directory, "public.pkl.part")
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, os.path.join(directory, "public.pkl"))
    activate(pub["snapshot"])
    return pub


def release_public(pub):
    """The distributable copy: population evidence only, without the builder's own scores."""
    return dict(pub, user={"id": None, "scores": [], "stats": {}, "playcount": {}})


def activate(name):
    """Atomic switch of the active snapshot, then bounded pruning of superseded rolling snapshots only."""
    hist = []
    try:
        with open(ACTIVE, "r", encoding="utf-8") as fh:
            hist = json.load(fh).get("history", [])
    except (OSError, ValueError):
        pass
    hist = [h for h in hist if h != name] + [name]
    tmp = ACTIVE + ".part"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"name": name, "history": hist[-KEEP_ROLLING:], "t": time.time()}, fh)
    os.replace(tmp, ACTIVE)
    keep = set(hist[-KEEP_ROLLING:])
    for d in os.listdir(ROLLING):
        p = os.path.join(ROLLING, d)
        if os.path.isdir(p) and d not in keep and (d.startswith("staging-") is False):
            _assert_rolling(p)
            shutil.rmtree(p)
    return keep


def _assert_rolling(p):
    rp_, fz = os.path.realpath(p), os.path.realpath(FROZEN)
    if not rp_.startswith(os.path.realpath(ROLLING) + os.sep) or rp_.startswith(fz + os.sep) or rp_ == fz:
        raise RuntimeError(f"refusing to touch {p}: not rolling recommendation data")


# ---------------------------------------------------------------------------
# refresh (E): check, stage, validate, switch, prune
# ---------------------------------------------------------------------------
SNAP_RE = re.compile(r"(\d{4}_\d\d_\d\d)_performance_mania_top_1000\.tar\.bz2")
NEEDED = ("osu_scores_mania_high.sql", "osu_beatmaps.sql", "osu_user_stats_mania.sql", "osu_user_beatmap_playcount.sql")


def check_index(db=None, url=INDEX_URL, timeout=20):
    """Newest mania top-1000 snapshot on data.ppy.sh (one small request). Recorded in kv when db given."""
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        names = sorted(set(SNAP_RE.findall(resp.read().decode("utf-8", "replace"))))
    newest = names[-1] + "_performance_mania_top_1000" if names else None
    act = None
    try:
        with open(ACTIVE, "r", encoding="utf-8") as fh:
            act = json.load(fh)["name"]
    except (OSError, ValueError, KeyError):
        pass
    info = {"t": time.time(), "newest": newest, "active": act, "newer": bool(newest and act and newest > act)}
    if db is not None:
        kv_set(db, "snapshot_check", info)
    return info


def _safe_extract(tar_path, dest, wanted):
    """Only the named SQL members, flattened to basenames (no traversal, no links)."""
    got = set()
    with tarfile.open(tar_path, "r:*") as tf:
        for ti in tf:
            base = os.path.basename(ti.name)
            if base in wanted and ti.isfile():
                with tf.extractfile(ti) as src, open(os.path.join(dest, base + ".part"), "wb") as out:
                    shutil.copyfileobj(src, out)
                os.replace(os.path.join(dest, base + ".part"), os.path.join(dest, base))
                got.add(base)
    return got


def refresh(archive=None, download=False, user_id=None, charts=None, build=build_public, workers=None):
    """Stage a snapshot, build its public.pkl, validate against the active one, then switch.
    Any failure leaves the active snapshot untouched; the staging directory is removed."""
    if download:
        info = check_index()
        if not info["newer"]:
            return {"status": "up to date", **info}
        name = info["newest"]
        os.makedirs(ROLLING, exist_ok=True)
        archive = os.path.join(ROLLING, f"staging-{name}.tar.bz2")
        with urllib.request.urlopen(INDEX_URL + name + ".tar.bz2", timeout=60) as resp, open(archive + ".part", "wb") as out:
            shutil.copyfileobj(resp, out, 1 << 20)
        os.replace(archive + ".part", archive)
    m = SNAP_RE.search(os.path.basename(archive or ""))
    if not m:
        raise ValueError(f"not a mania top-1000 snapshot archive: {archive}")
    name = m.group(1) + "_performance_mania_top_1000"
    stage = os.path.join(ROLLING, "staging-" + name)
    _assert_rolling(stage) if os.path.exists(stage) else None
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(stage)
    try:
        got = _safe_extract(archive, stage, NEEDED)
        if set(NEEDED) - got:
            raise ValueError(f"archive lacks {sorted(set(NEEDED) - got)}")
        import native_backend
        sys.path.insert(0, os.path.join(native_backend.ROOT, "calib"))
        import stage1
        scores, maps = stage1.parse(stage + os.sep)
        prev = load_public()
        user_id = user_id or (prev or {}).get("user", {}).get("id")
        pub = build(stage, stage, scores, maps, user_id, charts=charts, prev=prev, workers=workers)
        if prev:
            if len(pub["maps"]) < 0.9 * len(prev["maps"]) or pub["pop"]["rows"] < 0.5 * prev["pop"]["rows"]:
                raise ValueError("new snapshot is much smaller than the active one — kept the active one")
        if not pub["user"]["scores"] and prev and prev["user"]["scores"]:
            raise ValueError("user missing from the new snapshot — kept the active one")
        for f in NEEDED:                                  # derived view only; the SQL is not kept
            os.remove(os.path.join(stage, f))
        final = os.path.join(ROLLING, name)
        if os.path.exists(final):
            _assert_rolling(final)
            shutil.rmtree(final)
        os.replace(stage, final)
        kept = activate(name)
        return {"status": "switched", "name": name, "kept": sorted(kept)}
    finally:
        if os.path.exists(stage):
            shutil.rmtree(stage, ignore_errors=True)
        if download and archive and os.path.exists(archive):
            os.remove(archive)


# ---------------------------------------------------------------------------
# frozen calibration bundle
# ---------------------------------------------------------------------------
FROZEN_FILES = ("scores.pkl", "cohort.pkl", "ratings_23.pkl", "users.pkl", "feats.pkl",
                *(os.path.join(FROZEN_SNAPSHOT, f) for f in NEEDED + ("osu_counts.sql", "osu_difficulty_attribs.sql", "sample_users.sql")))


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def frozen_manifest(verify=False):
    """Hashes of the September validation inputs, the chart tree and the calculator sources in use.
    verify=True compares against the stored manifest → list of changed paths."""
    path = os.path.join(FROZEN, "FROZEN_MANIFEST.json")
    files = {f: _sha(os.path.join(FROZEN, f)) for f in FROZEN_FILES if os.path.exists(os.path.join(FROZEN, f))}
    tree = hashlib.sha256()
    for f in sorted(os.listdir(os.path.join(FROZEN, "osu"))):
        with open(os.path.join(FROZEN, "osu", f), "rb") as fh:
            tree.update(f.encode() + hashlib.sha256(fh.read()).digest())
    files["osu/ (tree)"] = tree.hexdigest()
    if verify:
        with open(path, "r", encoding="utf-8") as fh:
            old = json.load(fh)["files"]
        return [f for f in old if files.get(f) != old[f]]
    with open(os.path.join(FROZEN, "cohort.pkl"), "rb") as fh:
        base = pickle.load(fh)["base"]
    man = {"written": time.strftime("%FT%T"), "files": files, "cohort_base": base,
           "note": "September 2026 calculator-validation bundle. Frozen: cohort rows index scores.pkl in this exact order. "
                   "ratings.pkl is the moving alias of the current calculator and is NOT part of this bundle. "
                   "recdata.py refresh never writes here."}
    with open(path + ".part", "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1)
    os.replace(path + ".part", path)
    return man


def frozen_chart(bid):
    p = os.path.join(FROZEN, "osu", f"{bid}.osu")
    return p if os.path.exists(p) else None


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "import"
    if cmd == "import":
        db = connect()
        print(import_realm(db, sys.argv[2] if len(sys.argv) > 2 else None))
        pub = load_public()
        if pub:
            import_public_user(db, pub)
    elif cmd == "build":
        D = sys.argv[2] if len(sys.argv) > 2 else os.path.join(FROZEN, FROZEN_SNAPSHOT)
        with open(os.path.join(FROZEN, "scores.pkl"), "rb") as fh:
            scores, maps = pickle.load(fh)
        uid = kv_get(connect(), "user_id") or 2653437
        name = os.path.basename(os.path.normpath(D))
        pub = build_public(D, os.path.join(ROLLING, name), scores, maps, uid, charts=frozen_chart,
                           prev=load_public())
        activate(name)
        print(f"public.pkl: {len(pub['feats'])} charts, population {pub['pop']['rows']} rows / {pub['pop']['players']} players, "
              f"user {len(pub['user']['scores'])} stable scores, stats {pub['user']['stats']}")
    elif cmd == "refit":
        with open(ACTIVE, "r", encoding="utf-8") as fh:
            directory = os.path.join(ROLLING, json.load(fh)["name"])
        path = os.path.join(directory, "calculators", calc_id(), "public.pkl")
        path = path if os.path.isfile(path) else os.path.join(directory, "public.pkl")
        with open(path, "rb") as fh:
            pub = pickle.load(fh)
        with open(os.path.join(FROZEN, "scores.pkl"), "rb") as fh:
            scores, maps = pickle.load(fh)
        fr = {b: {r: v for r, v in f.items() if isinstance(r, float)} for b, f in pub["feats"].items()}
        pub["pop"] = fit_population(scores, maps, fr, pub["user"]["id"])
        pub["pop_lazer"] = fit_population(scores, maps, fr, pub["user"]["id"], accuracy="lazer",
                                          ceilings=True, specialization=True)
        with open(path + ".part", "wb") as fh:
            pickle.dump(pub, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(path + ".part", path)
        print("refit", path, {k: {p: [round(c, 2) for c in v] for p, v in g.items()} for k, g in pub["pop_lazer"]["gap"].items()})
    elif cmd == "check":
        print(check_index(connect()))
    elif cmd == "refresh":
        a = sys.argv[2:]
        print(refresh(archive=a[a.index("--archive") + 1] if "--archive" in a else None, download="--download" in a,
                      charts=frozen_chart))
    elif cmd == "manifest":
        print(frozen_manifest(verify="verify" in sys.argv) if "verify" in sys.argv else frozen_manifest()["files"].keys())


if __name__ == "__main__":
    main()
