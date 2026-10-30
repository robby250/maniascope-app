"""Rate the community dan charts with this calculator → calib/dans.json (the dan display's anchors).

    python3 calib/dan_table.py [CHART_DIR]      # default ~/.cache/maniascope/dans (calib/dan_fetch.py fills it)

calib/dan_courses.tsv names each chart by SHA-256 with its series, tier, pack (its beatmap set: the
hold-out unit) and kind (course, or a single song). Course files are also split at their song
boundaries, so each course adds its songs at its tier. A tier's anchor is the median over its packs of
each pack's value: its course, else the median of its songs, else of its split pieces. Pooling a course with
its own pieces put the course above its tier (pieces rate ~5 % lower; REFORM Theta showed "Theta+",
user 2026-10-11). The report places every pack's charts with anchors built without that pack.
Mappers' files stay out of the repository. Rerun after any calculator change: dans.py ignores a table
from another calculator.

A shared skill-mix weighting (jack, speed, stamina, tech, LN, length on top of stars) was tried and
moved held-out error 0.75 → 0.73 tiers: noise, so anchors use the rating alone (2026-10-01).
"""
import collections
import copy
import csv
import json
import math
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "app"))
import dans  # noqa: E402
import recdata  # noqa: E402
import skill_calc  # noqa: E402

SONG_REST = 4000      # ms without notes, starting a new uninherited timing point, between two course songs


def song_starts(path, chart, songs=4):
    """Start times of a course's songs after the first, or None unless exactly songs-1 boundaries."""
    red, section = [], None
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("["):
                section = line
            elif line and section == "[TimingPoints]":
                f = line.split(",")
                if len(f) < 7 or f[6] == "1":
                    red.append(float(f[0]))
    starts, cuts, busy = sorted(chart.notes), [], None
    for (s, e, _c), (nxt, _e, _c2) in zip(starts, starts[1:]):
        busy = max(busy or e, e)
        if nxt - busy >= SONG_REST and any(busy < t <= nxt + 5 for t in red):
            cuts.append(nxt)
    return cuts if len(cuts) == songs - 1 else None


def piece(chart, lo, hi):
    sub = copy.deepcopy(chart)
    sub.notes = [n for n in chart.notes if lo <= n[0] < hi]
    sub.sv = [p for p in chart.sv if lo - 5000 <= p[0] < hi] if chart.sv else chart.sv
    return sub


def rate(row, path):
    chart = skill_calc.parse_osu(path)
    charts = [(row["kind"], chart)]
    if row["kind"] == "course":
        cuts = song_starts(path, chart)
        if cuts:
            bounds = [-math.inf] + cuts + [math.inf]
            charts += [("split", piece(chart, lo, hi)) for lo, hi in zip(bounds, bounds[1:])]
    out = []
    for kind, c in charts:
        res = skill_calc.compute(c, 1.0)
        out.append(dict(row, kind=kind, rating=res["scores"]["overall"],
                        ln=res["ln_notes"] / max(1, res["notes"]), vibro=dans.vibro_runs(c.notes)))
    return out


def _job(args):
    row, path = args
    try:
        return rate(row, path)
    except (OSError, ValueError) as exc:
        print("unreadable", row["course"], exc, flush=True)
        return []


def vibro_weights(charts):
    """Least-squares tier ~ log vibro speed + log stars over the vibro dans (tier units)."""
    import numpy as np
    vib = [c for c in charts if c["series"] == "4K Vibro"]
    A = np.array([[1., math.log(max(1., c["vibro"]["speed"])), math.log(c["rating"])] for c in vib])
    return [float(w) for w in np.linalg.lstsq(A, np.array([float(c["order"]) for c in vib]), rcond=None)[0][1:]]


def anchors(charts, weights):
    by = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(
        lambda: collections.defaultdict(list))))
    names = {}
    for c in charts:
        by[c["series"]][float(c["order"])][c["pack"]][c["kind"]].append(
            dans.value(c["series"], c["rating"], c["vibro"], weights))
        names[c["series"], float(c["order"])] = c["tier"]
    out = {}
    for s, t in by.items():
        out[s] = []
        for o, packs in sorted(t.items()):
            v = [statistics.median(k["course"] or k["song"] or k["split"]) for k in packs.values()]
            out[s].append(dict(order=o, tier=names[s, o], value=round(statistics.median(v), 5), n=len(v)))
    return out


def songs(charts, paths):
    """{series: {tier: [features at 1.0×]}}: each tier's songs, else its courses. The Stats page predicts the
    player on these to place them on the ladder (dans.songs)."""
    keep = collections.defaultdict(lambda: collections.defaultdict(list))
    for c in charts:
        if c["kind"] != "split":
            keep[c["series"]][c["tier"]].append(c)
    picked = [c for tiers in keep.values() for cs in tiers.values()
              for c in ([c for c in cs if c["kind"] == "song"] or cs)]
    feats = recdata.feats_batch([(paths[c["sha256"]], [1.0], None) for c in picked])
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for c, f in zip(picked, feats):
        if not isinstance(f, Exception):
            out[c["series"]][c["tier"]].append(f[1.0])
    return out


def ladder(series_anchors):
    return dans.monotone([(a["order"], a["tier"], a["value"]) for a in series_anchors])


def detected(c):
    if c["series"] == "4K Vibro":
        return c["vibro"]["share"] >= dans.VIBRO_SHARE
    if int(c["keys"]) == 4 and dans.vibro_only(c["vibro"]):
        return False                    # shown as vibro alone at runtime
    return dans.SERIES.get((int(c["keys"]), dans.kind(c["ln"]))) == c["series"]


def report(charts):
    """Series detection, then held-out placement: each pack against anchors built from every other pack."""
    hits = collections.defaultdict(list)
    for c in charts:
        hits[c["series"]].append(detected(c))
    print("series detected:", {s: f"{sum(v)}/{len(v)}" for s, v in sorted(hits.items())})
    err = collections.defaultdict(list)
    for pack in sorted({c["pack"] for c in charts}):
        others = [c for c in charts if c["pack"] != pack]
        weights = vibro_weights(others)
        rest = anchors(others, weights)
        for c in charts:
            if c["pack"] != pack or c["series"] not in rest:
                continue
            tiers = ladder(rest[c["series"]])
            orders = sorted(a["order"] for a in rest[c["series"]])
            p = dans.position(tiers, dans.value(c["series"], c["rating"], c["vibro"], weights))
            j = min(len(orders) - 1, max(0, round(p)))
            err[c["series"], c["kind"]].append(orders[j] - float(c["order"]))
    for (series, kind), shown in sorted(err.items()):
        v = shown
        print(f"{series:12} {kind:6} n={len(v):3}  shown dan exact {sum(e == 0 for e in shown) / len(v):4.0%}"
              f"  within ±1 {sum(abs(e) <= 1 for e in shown) / len(v):4.0%}")


def main(argv):
    root = os.path.expanduser(argv[0] if argv else "~/.cache/maniascope/dans")
    files = {f: os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs}
    rows = list(csv.DictReader(open(os.path.join(HERE, "dan_courses.tsv"), encoding="utf-8"), delimiter="\t"))
    jobs = []
    for row in rows:
        path = files.get(row["sha256"]) or files.get(row["sha256"] + ".osu")
        if path:
            jobs.append((row, path))
        else:
            print("missing", row["course"])
    with ProcessPoolExecutor() as ex:
        charts = [c for cs in ex.map(_job, jobs, chunksize=2) for c in cs]
    # Vibro dans whose chart isn't vibro by the runtime rule would never be shown as vibro: not anchors.
    report(charts)
    # A dan chart the runtime would not file under its series is not an anchor for that series.
    charts = [c for c in charts if detected(c)]
    weights = vibro_weights(charts)
    table = anchors(charts, weights)
    for s, tiers in table.items():
        flips = [(a["tier"], b["tier"]) for a, b in zip(tiers, tiers[1:]) if b["value"] <= a["value"]]
        if flips:
            print(f"pooled (not monotone) {s}: {flips}")
    with open(os.path.join(HERE, "dans.json"), "w", encoding="utf-8") as fh:
        json.dump({"calc": recdata.calc_id(), "vibro_weights": weights, "series": {s: [dict(tier=t, value=round(v, 5)) for t, v in ladder(a)]
                                                         for s, a in table.items()},
                   "songs": songs(charts, {row["sha256"]: path for row, path in jobs})}, fh, indent=1, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
