"""Website charts for the NPS playlist: every status, analysed like installed maps.

    python3 webmaps.py crawl|check [--keys 7[,4…]] [--stars LO-HI] [--full] [--workers N] [--publish]

Runs on G533QR: osu.direct metadata → chart text (.osu only, no audio) for charts whose
density can reach the NPS range at 0.70–1.50× → features on a rate grid → WEB_DIR/web.pkl
(only the fields the playlists read, one pickled blob per rate, decoded when used).
Afterwards rsync WEB_DIR to G835LX. The app treats these charts as not installed; the
playlist's Next sends osu://b/<id> so lazer offers the download. A calculator change needs
another crawl (only analysis reruns; files and metadata are cached).
`check` only refreshes the listing, so the app can show counts, size and time before Start.
Progress goes to state.db (kv "progress"); a "stop" file in WEB_DIR pauses it, and the next
crawl resumes from state.db. crawl.lock is held while one runs.
"""
import datetime
import hashlib
import http.client
import itertools
import json
import os
import pickle
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import recdata

WEB_DIR = os.path.join(recdata.REC_DIR, "web")
SEARCH = "https://osu.direct/api/v2/search?mode=3&status={status}&amount=100&offset={offset}&sort=last_updated:desc"
# osu.direct first; ppy only for what the mirror lacks, slowly: both laptops share the home IP
# that lazer uses, so a throttled IP at ppy would hurt play (user concern 2026-09-30).
FILES = (("https://osu.direct/api/osu/{bid}", .6), ("https://osu.ppy.sh/osu/{bid}", 3.))
UA = "maniascope/0.1 (personal osu!mania practice recommender)"
STATUSES = (-2, -1, 0, 1, 2, 3, 4)       # graveyard, wip, pending, ranked, approved, qualified, loved
# HT/NM/DT only (user 2026-10-02: ~3x faster crawl, ~3x smaller web.pkl). NPS/Skills compute the exact
# rate a candidate needs on demand, as for installed maps (nps.candidates_from proposals).
RATES = (0.75, 1.0, 1.5)
PAUSE = .6                               # osu.direct allows 120 requests/minute
PAGES = 4                                # pages in flight (~3 s each server-side → ~60/min)
# Estimates before the first run on this PC (measured on G533QR 2026-10-01); runs learn their own.
SEC_FETCH = 2.4                          # one chart from osu.direct incl. pacing and ppy fallback
SEC_ANALYSE = 0.7                        # CPU seconds per chart for the 3 rates, split over workers
CHART_BYTES = 150_000                    # .osu ≈ 120 KB mean + its features in state.db and web.pkl


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return b""
            if exc.code == 429:           # rate limited (e.g. another client on this IP): wait it out
                time.sleep(60)
            error = exc
        except (OSError, http.client.HTTPException) as exc:     # IncompleteRead killed the 09-30 crawl
            error = exc
        time.sleep(5 * (attempt + 1))
    raise error


def _db():
    os.makedirs(os.path.join(WEB_DIR, "osu"), exist_ok=True)
    db = sqlite3.connect(os.path.join(WEB_DIR, "state.db"))
    db.row_factory = sqlite3.Row
    db.executescript("""
        CREATE TABLE IF NOT EXISTS diffs(bid INTEGER PRIMARY KEY, set_id INTEGER, status INTEGER, keys INTEGER,
          md5 TEXT, notes INTEGER, length REAL, playcount INTEGER, updated TEXT, artist TEXT, title TEXT,
          version TEXT, creator TEXT, sha TEXT, file_md5 TEXT, fetched TEXT, error TEXT);
        CREATE TABLE IF NOT EXISTS feats(sha TEXT, calc TEXT, data BLOB, PRIMARY KEY(sha, calc));
        CREATE TABLE IF NOT EXISTS kv(k TEXT PRIMARY KEY, v TEXT);""")
    if "stars" not in {r[1] for r in db.execute("PRAGMA table_info(diffs)")}:
        db.execute("ALTER TABLE diffs ADD COLUMN stars REAL")    # nomod star rating from the listing
    return db


def _kv(db, k, v=None):
    if v is not None:
        with db:
            db.execute("INSERT OR REPLACE INTO kv VALUES(?,?)", (k, json.dumps(v)))
    row = db.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
    return json.loads(row[0]) if row else None


class Stopped(Exception):
    """The app asked to pause (stop file); the next crawl resumes from state.db."""


def _check_stop():
    if os.path.exists(os.path.join(WEB_DIR, "stop")):
        raise Stopped


def _lock():
    """The OS releases it when the crawl exits however it ends; None while another crawl holds it."""
    lock = sqlite3.connect(os.path.join(WEB_DIR, "crawl.lock"), timeout=0, isolation_level=None)
    try:
        lock.execute("BEGIN EXCLUSIVE")
        return lock
    except sqlite3.OperationalError:
        lock.close()
        return None


def running():
    lock = _lock()
    if lock:
        lock.close()
    return lock is None


def _progress(db, **p):
    _kv(db, "progress", dict(_kv(db, "progress") or {}, t=time.time(), **p))


def _workers(workers=None):
    return workers or max(1, (os.cpu_count() or 2) - 2)


def seconds(db, fetch, analyse):
    """Time left for these counts at this PC's measured pace (defaults before its first run)."""
    return fetch * (_kv(db, "sec_fetch") or SEC_FETCH) + analyse * (_kv(db, "sec_analyse") or SEC_ANALYSE / _workers())


def disk_bytes():
    total = 0
    for d in (WEB_DIR, os.path.join(WEB_DIR, "osu")):
        with os.scandir(d) as it:
            total += sum(e.stat().st_size for e in it if e.is_file())
    return total


def estimate(db, keys, stars):
    """What a crawl for these keymodes and ★ range still has to do, as counts, bytes and seconds."""
    fetch = analyse = ready = 0
    for r in db.execute("SELECT d.*, f.sha IS NOT NULL AS done FROM diffs d LEFT JOIN feats f "
                        "ON f.sha=d.sha AND f.calc=? WHERE d.error IS NULL", (recdata.calc_id(),)):
        if r["keys"] not in keys or not reachable(r) or not in_stars(r, stars):
            continue
        if r["sha"] is None or r["fetched"] != r["md5"]:
            fetch += 1
        elif not r["done"]:
            analyse += 1
        else:
            ready += 1
    return {"fetch": fetch, "analyse": fetch + analyse, "ready": ready, "new_bytes": fetch * CHART_BYTES,
            "seconds": seconds(db, fetch, fetch + analyse), "disk": disk_bytes(), "checked": _kv(db, "checked"),
            "unlisted": sorted(set(keys) - _listed_keys(db))}


def _listed_keys(db):
    """Keymodes the listing watermarks cover (crawls before 2026-10-01 only left their rows)."""
    saved = _kv(db, "keys")
    return set(saved) if saved is not None else {r[0] for r in db.execute("SELECT DISTINCT keys FROM diffs")}


def in_stars(row, stars):
    """Charts without a listed rating stay in; the filter only drops what it can see."""
    return not stars or row["stars"] is None or stars[0] <= row["stars"] <= stars[1]


def crawl_meta(db, keys, full=False):
    """Newest-first per status; an incremental crawl stops a day past the last watermark."""
    # Rows crawled before the stars column need one full pass to learn their rating.
    backfill = not _kv(db, "stars_backfilled")
    # The watermarks only cover the keymodes listed so far; a new keymode needs its older sets too.
    full = full or backfill or not keys <= _listed_keys(db)
    sets = 0
    for status in STATUSES:
        mark = (db.execute("SELECT v FROM kv WHERE k=?", (f"mark{status}",)).fetchone() or [""])[0]
        stop = "" if full or not mark else (datetime.datetime.fromisoformat(mark.replace("Z", "+00:00"))
                                             - datetime.timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        newest, offset, done = mark, 0, False
        pool = ThreadPoolExecutor(PAGES)
        while not done:
            batch = list(pool.map(lambda o: json.loads(_get(SEARCH.format(status=status, offset=o)) or b"[]"),
                                  range(offset, offset + 100 * PAGES, 100)))
            time.sleep(PAUSE)
            pages = [p for p in batch if isinstance(p, list) and p]
            sets += sum(len(p) for p in pages)
            _progress(db, phase="check", sets=sets)
            _check_stop()
            done = len(pages) < PAGES or any(len(p) < 100 for p in pages)
            if not pages:
                break
            with db:
                for s in (s for page in pages for s in page):
                    for b in s.get("beatmaps", ()):
                        if b.get("mode_int") != 3 or b.get("convert") or int(b.get("cs") or 0) not in keys:
                            continue
                        db.execute("""INSERT INTO diffs(bid,set_id,status,keys,md5,notes,length,playcount,updated,
                                      artist,title,version,creator,stars) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                                      ON CONFLICT(bid) DO UPDATE SET status=excluded.status, md5=excluded.md5,
                                      notes=excluded.notes, length=excluded.length, playcount=excluded.playcount,
                                      updated=excluded.updated, stars=excluded.stars,
                                      error=CASE WHEN diffs.md5 IS excluded.md5 THEN diffs.error END""",
                                   (b["id"], s["id"], b.get("ranked", s.get("ranked")), int(b["cs"]), b.get("checksum"),
                                    (b.get("count_circles") or 0) + (b.get("count_sliders") or 0), b.get("hit_length") or 0,
                                    b.get("playcount") or 0, b.get("last_updated") or s.get("last_updated"),
                                    s.get("artist"), s.get("title"), b.get("version"), s.get("creator"),
                                    b.get("difficulty_rating")))
            newest = max(newest, pages[0][0].get("last_updated") or "")
            print(f"status {status} offset {offset} {pages[-1][-1].get('last_updated')}", flush=True)
            if stop and (pages[-1][-1].get("last_updated") or "") < stop:
                break
            offset += 100 * PAGES
        pool.shutdown()
        with db:
            db.execute("INSERT OR REPLACE INTO kv VALUES(?,?)", (f"mark{status}", newest))
    if backfill:
        _kv(db, "stars_backfilled", True)
    _kv(db, "keys", sorted(keys | _listed_keys(db)))
    _kv(db, "checked", time.time())


def reachable(row):
    """The metadata density can reach the NPS range at 1.5× (nps.weighted_candidates' floor ≈ 20·k/7)."""
    return row["notes"] >= 100 and row["length"] >= 30 and \
        row["notes"] / row["length"] * 1.5 >= .6 * 20 * row["keys"] / 7


def fetch(db, keys, limit=None, stars=None):
    """Chart text once per revision, most-played first; dump copies for ranked/loved; empty body = unavailable."""
    dump = os.path.join(recdata.FROZEN, "osu")
    rows = [r for r in db.execute("SELECT * FROM diffs WHERE error IS NULL AND (sha IS NULL OR fetched IS NOT md5) "
                                  "ORDER BY playcount DESC") if r["keys"] in keys and reachable(r) and in_stars(r, stars)][:limit]
    print(f"charts to fetch: {len(rows)}", flush=True)

    def text(job):
        i, r = job
        local = os.path.join(dump, f"{r['bid']}.osu")
        data = open(local, "rb").read() if os.path.isfile(local) else b""
        if hashlib.md5(data).hexdigest() == r["md5"]:
            return data
        for url, pause in FILES:
            data = _get(url.format(bid=r["bid"]))
            time.sleep(pause)
            if data:
                return data
        return b""

    t0, done = time.time(), (_kv(db, "progress") or {}).get("fetched", 0)
    pool = ThreadPoolExecutor(1)             # prefetch one ahead of the store loop, ~60/min
    try:
        results = pool.map(text, enumerate(rows))
        for i, (r, data) in enumerate(zip(rows, results)):
            _store(db, r, data)
            _progress(db, phase="download", fetched=done + i + 1)
            if i % 200 == 0:
                print(f"fetched {i}/{len(rows)}", flush=True)
            if i >= 20 and i % 20 == 0:
                _kv(db, "sec_fetch", (time.time() - t0) / (i + 1))
            _check_stop()
    finally:                                 # a pause drops the queued downloads, not just stops reading them
        pool.shutdown(wait=False, cancel_futures=True)
    return len(rows)


def _store(db, r, data):
    with db:
        if not data.strip():
            db.execute("UPDATE diffs SET error='unavailable', fetched=md5 WHERE bid=?", (r["bid"],))
            return
        sha = hashlib.sha256(data).hexdigest()
        with open(os.path.join(WEB_DIR, "osu", sha + ".osu"), "wb") as fh:
            fh.write(data)
        db.execute("UPDATE diffs SET sha=?, file_md5=?, fetched=md5 WHERE bid=?",
                   (sha, hashlib.md5(data).hexdigest(), r["bid"]))


def _analyse(sha):
    try:
        return sha, recdata.chart_feats(os.path.join(WEB_DIR, "osu", sha + ".osu"), RATES), None
    except Exception as exc:          # one broken chart is one missing candidate
        return sha, None, f"{type(exc).__name__}: {exc}"


def analyse(db, workers=None):
    calc = recdata.calc_id()
    todo = [r[0] for r in db.execute("SELECT DISTINCT sha FROM diffs WHERE sha IS NOT NULL AND sha NOT IN "
                                     "(SELECT sha FROM feats WHERE calc=?)", (calc,))]
    print(f"charts to analyse: {len(todo)}", flush=True)
    t0, done = time.time(), (_kv(db, "progress") or {}).get("analysed", 0)
    with Pool(_workers(workers)) as pool:          # leaving the block terminates the workers on a pause
        for i, (sha, fs, err) in enumerate(pool.imap_unordered(_analyse, todo, chunksize=2)):
            with db:
                db.execute("INSERT OR REPLACE INTO feats VALUES(?,?,?)",
                           (sha, calc, pickle.dumps(fs) if fs else pickle.dumps({"error": err})))
            _progress(db, phase="analyse", analysed=done + i + 1)
            if i % 500 == 0:
                print(f"analysed {i}/{len(todo)}", flush=True)
            if i >= 20 and i % 20 == 0:
                _kv(db, "sec_analyse", (time.time() - t0) / (i + 1))
            _check_stop()
    return calc


# What NPS/Skills predictions, eligibility and practice matching read from a website chart
# (key-access probe 2026-10-01). Full dicts cost ~90 KB per chart in memory; the app would
# hold every one of them for the whole session.
NPS_FIELDS = ("active", "burst", "chord", "drill", "family", "nps", "overlap", "play_span", "repeat", "rhythm",
              "version", "vibro_focused", "wide")


def compact(f):
    return {"keys": f["keys"], "overall": f["overall"], "od": f["od"], "ln": f["ln"], "notes": f["notes"],
            "length": f["length"], "stam": f["stam"], "description": f.get("description"), "sk": f["sk"],
            "stars": f.get("stars"), "endurance": {"load": f.get("endurance", {}).get("load", 0.)},
            "execution": {"residual_vector": f.get("execution", {}).get("residual_vector")},
            "calibration_features": f.get("calibration_features", []),
            "nps": {k: f["nps"][k] for k in NPS_FIELDS if k in f.get("nps", {})},
            "tech_profile": f.get("tech_profile", {})}


class Rates(dict):
    """rate → features, stored pickled and decoded on each read (~1.5 KB instead of ~10 KB resident)."""
    def __getitem__(self, rate):
        v = dict.__getitem__(self, rate)
        return pickle.loads(v) if isinstance(v, bytes) else v

    def get(self, rate, default=None):
        return self[rate] if rate in self else default

    def setdefault(self, rate, value=None):
        if rate not in self:
            dict.__setitem__(self, rate, value)
        return self[rate]

    def items(self):
        return [(r, self[r]) for r in self]

    def values(self):
        return [self[r] for r in self]


def publish(db, calc, stars=None):
    maps, feats = {}, {}
    for r in db.execute("SELECT d.*, f.data FROM diffs d JOIN feats f ON f.sha=d.sha AND f.calc=? "
                        "WHERE d.error IS NULL", (calc,)):
        if not in_stars(r, stars):
            continue
        fs = pickle.loads(r["data"])
        if "error" in fs:
            continue
        maps[r["sha"]] = {k: r[k] for k in ("bid", "set_id", "status", "keys", "file_md5", "playcount",
                                             "artist", "title", "version", "creator")}
        maps[r["sha"]]["overall"] = {rate: f["overall"] for rate, f in fs.items()}
        feats[r["sha"]] = {rate: pickle.dumps(compact(f), pickle.HIGHEST_PROTOCOL) for rate, f in fs.items()}
    tmp = os.path.join(WEB_DIR, "web.pkl.part")
    with open(tmp, "wb") as fh:
        pickle.dump({"calc": calc, "format": 3, "maps": maps, "feats": feats, "built": time.time()}, fh)
    os.replace(tmp, os.path.join(WEB_DIR, "web.pkl"))
    _progress(db, ready=len(maps))
    print(f"web.pkl: {len(maps)} charts", flush=True)


_loaded = (None, None)


def catalog(keys, installed_md5=(), reach=None):
    """→ ({sha: installed-like row}, {sha: {rate: feats}}) for the current calculator; {} if absent.
    reach {keys: (lo, hi)} keeps only rates whose rating can land near the player's target."""
    global _loaded
    path = os.path.join(WEB_DIR, "web.pkl")
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return {}, {}
    if _loaded[0] != mtime:
        with open(path, "rb") as fh:
            _loaded = (mtime, pickle.load(fh))
    web = _loaded[1]
    if web.get("format") != 3:
        return {}, {}
    # Another calculator's features stay in use until the next publish: re-analysing them on the
    # device held NPS/Skills at loading for ~37k charts (G835LX 2026-10-01), and skipping them hid
    # every download suggestion. A calculator change moves ordinary charts by well under 1 %.
    out, feats = {}, {}
    for sha, m in web["maps"].items():
        chart = os.path.join(WEB_DIR, "osu", sha + ".osu")
        if m["keys"] not in keys or m["file_md5"] in installed_md5 or not os.path.isfile(chart):
            continue
        out[sha] = dict(sha256=sha, beatmap_id=m["bid"], set_id=m["set_id"], md5=m["file_md5"],
                        online_md5=m["file_md5"], status=m["status"], keys=m["keys"], artist=m["artist"],
                        title=m["title"], version=m["version"], creator=m["creator"], audio_sha=None,
                        audio_required=0, path=chart, installed=False, playcount=m["playcount"])
        lo, hi = (reach or {}).get(m["keys"], (0., float("inf")))
        ov = m["overall"]
        rates = {r: b for r, b in web["feats"][sha].items() if lo <= ov[r] <= hi}
        if not rates:     # the window falls between two rated rates: both anchor an exact refinement
            below = [r for r in ov if ov[r] < lo]
            above = [r for r in ov if ov[r] > hi]
            if below and above:
                rates = {r: web["feats"][sha][r] for r in (max(below, key=ov.get), min(above, key=ov.get))}
        if not rates:
            del out[sha]
            continue
        feats[sha] = Rates(rates)
    return out, feats


def main(argv):
    if not argv or argv[0] not in ("crawl", "check"):
        print(__doc__)
        return 2
    keys = {int(k) for k in argv[argv.index("--keys") + 1].split(",")} if "--keys" in argv else {7}
    workers = int(argv[argv.index("--workers") + 1]) if "--workers" in argv else None
    import paths
    paths.lower_priority()          # the app may start this while the game runs
    db = _db()
    lock = _lock()
    if lock is None:
        print("another crawl is running", flush=True)
        return 1
    try:
        os.remove(os.path.join(WEB_DIR, "stop"))
    except FileNotFoundError:
        pass
    _kv(db, "want_keys", sorted(keys))       # the app's window reopens with these
    if "--stars" in argv:    # remembered: later runs and the app keep the chosen range
        _kv(db, "stars", [float(x) for x in argv[argv.index("--stars") + 1].split("-")])
    stars = _kv(db, "stars")
    try:
        crawl_meta(db, keys, full="--full" in argv)
        if argv[0] == "check":
            _progress(db, phase="checked")
            return 0
        if "--publish" in argv:  # repackage what is analysed, no network
            publish(db, analyse(db, workers), stars)
            return 0
        plan = estimate(db, keys, stars)
        _progress(db, phase="download", fetched=0, fetch_total=plan["fetch"], analysed=0,
                  analyse_total=plan["analyse"], error=None)
        # Publish every chunk; small first chunks so a new player sees website maps within minutes,
        # not after 5,000 downloads (fresh-install run 2026-10-02).
        for size in itertools.chain((300, 1500), itertools.repeat(5000)):
            more = fetch(db, keys, size, stars)
            publish(db, analyse(db, workers), stars)
            if not more:
                break
        _progress(db, phase="done")
    except Stopped:
        _progress(db, phase="paused")
        print("paused", flush=True)
    except Exception as exc:
        _progress(db, phase="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
