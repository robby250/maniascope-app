"""Website charts for the NPS playlist: every status, analysed like installed maps.

    python3 webmaps.py crawl [--keys 7[,4…]] [--stars LO-HI] [--full] [--workers N] [--publish]

Runs on G533QR: osu.direct metadata → chart text (.osu only, no audio) for charts whose
density can reach the NPS range at 0.70–1.50× → features on a rate grid → WEB_DIR/web.pkl
(only the fields the playlists read, one pickled blob per rate, decoded when used).
Afterwards rsync WEB_DIR to G835LX. The app treats these charts as not installed; the
playlist's Next sends osu://b/<id> so lazer offers the download. A calculator change needs
another crawl (only analysis reruns; files and metadata are cached).
"""
import datetime
import hashlib
import http.client
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
RATES = (0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5)
PAUSE = .6                               # osu.direct allows 120 requests/minute
PAGES = 4                                # pages in flight (~3 s each server-side → ~60/min)


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


def in_stars(row, stars):
    """Charts without a listed rating stay in; the filter only drops what it can see."""
    return not stars or row["stars"] is None or stars[0] <= row["stars"] <= stars[1]


def crawl_meta(db, keys, full=False):
    """Newest-first per status; an incremental crawl stops a day past the last watermark."""
    # Rows crawled before the stars column need one full pass to learn their rating.
    backfill = not _kv(db, "stars_backfilled")
    full = full or backfill
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


def reachable(row):
    """The metadata density can reach the NPS range at 1.5× (nps.weighted_candidates' floor ≈ 20·k/7)."""
    return row["notes"] >= 100 and row["length"] >= 30 and \
        row["notes"] / row["length"] * 1.5 >= .6 * 20 * row["keys"] / 7


def fetch(db, limit=None, stars=None):
    """Chart text once per revision, most-played first; dump copies for ranked/loved; empty body = unavailable."""
    dump = os.path.join(recdata.FROZEN, "osu")
    rows = [r for r in db.execute("SELECT * FROM diffs WHERE error IS NULL AND (sha IS NULL OR fetched IS NOT md5) "
                                  "ORDER BY playcount DESC") if reachable(r) and in_stars(r, stars)][:limit]
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

    with ThreadPoolExecutor(1) as pool:      # prefetch one ahead of the store loop, ~60/min
        results = pool.map(text, enumerate(rows))
        for i, (r, data) in enumerate(zip(rows, results)):
            _store(db, r, data)
            if i % 200 == 0:
                print(f"fetched {i}/{len(rows)}", flush=True)
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
    with Pool(workers or max(1, (os.cpu_count() or 2) - 2)) as pool:
        for i, (sha, fs, err) in enumerate(pool.imap_unordered(_analyse, todo, chunksize=8)):
            with db:
                db.execute("INSERT OR REPLACE INTO feats VALUES(?,?,?)",
                           (sha, calc, pickle.dumps(fs) if fs else pickle.dumps({"error": err})))
            if i % 500 == 0:
                print(f"analysed {i}/{len(todo)}", flush=True)
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
    fresh = web["calc"] == recdata.calc_id() and web.get("format") == 3
    out, feats = {}, {}
    for sha, m in web["maps"].items():
        chart = os.path.join(WEB_DIR, "osu", sha + ".osu")
        if m["keys"] not in keys or m["file_md5"] in installed_md5 or not os.path.isfile(chart):
            continue
        out[sha] = dict(sha256=sha, beatmap_id=m["bid"], set_id=m["set_id"], md5=m["file_md5"],
                        online_md5=m["file_md5"], status=m["status"], keys=m["keys"], artist=m["artist"],
                        title=m["title"], version=m["version"], creator=m["creator"], audio_sha=None,
                        audio_required=0, path=chart, installed=False, playcount=m["playcount"])
        if fresh:
            lo, hi = (reach or {}).get(m["keys"], (0., float("inf")))
            rates = {r: b for r, b in web["feats"][sha].items() if lo <= m["overall"][r] <= hi}
            if not rates:
                del out[sha]
                continue
            feats[sha] = Rates(rates)
    return out, feats


def main(argv):
    if not argv or argv[0] != "crawl":
        print(__doc__)
        return 2
    keys = {int(k) for k in argv[argv.index("--keys") + 1].split(",")} if "--keys" in argv else {7}
    workers = int(argv[argv.index("--workers") + 1]) if "--workers" in argv else None
    import paths
    paths.lower_priority()          # the app may start this while the game runs
    db = _db()
    if "--stars" in argv:    # remembered: later runs and the app keep the chosen range
        _kv(db, "stars", [float(x) for x in argv[argv.index("--stars") + 1].split("-")])
    stars = _kv(db, "stars")
    crawl_meta(db, keys, full="--full" in argv)
    if "--publish" in argv:  # repackage what is analysed, no network
        publish(db, analyse(db, workers), stars)
        return 0
    while True:              # publish every chunk: the popular part is usable long before the tail
        more = fetch(db, 5000, stars)
        publish(db, analyse(db, workers), stars)
        if not more:
            break
    print(f"now: rsync -a {WEB_DIR}/ g835lx:{WEB_DIR}/  (state.db optional)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
