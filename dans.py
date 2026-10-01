"""Dan equivalent of a chart: where it falls among the community dan charts (calib/dans.json).

Anchors are the dan charts themselves — courses, their split songs and the per-dan song packs — rated
by this calculator (calib/dan_table.py), not another calculator's scale (user 2026-10-01). Only chart
content decides the series and the place; titles never enter (user: "no cheating").

Series: LN-heavy charts are placed among LN dans, the rest among rice dans; 4K charts with sustained
same-column runs are also placed among vibro dans (REFORM Zeta/Eta songs hold real vibro sections, so
vibro is shown beside the rice dan, not instead of it). Vibro dans are placed by vibro speed together
with stars (weights fitted on the vibro dans): stars alone put Vibro 2–8 within 8.5–9.2★ out of order,
and population accuracy on ranked vibro charts gives no reason to move the stars themselves.
"""
import collections
import functools
import json
import math
import os

import recdata

TABLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib", "dans.json")
SERIES = {(4, "rice"): "4K REFORM", (4, "ln"): "4K LN",
          (6, "rice"): "6K Regular", (6, "ln"): "6K LN", (7, "rice"): "7K Regular", (7, "ln"): "7K LN",
          (10, "rice"): "10K Regular"}
SHORT = {"7K Regular": "Reg ", "7K LN": "LN ", "4K LN": "LN ", "6K Regular": "6K ", "6K LN": "6K LN ",
         "10K Regular": "10K ", "4K REFORM": "", "4K Vibro": ""}
VIBRO_GAP = 100        # ms: a column re-hit this fast keeps a vibro run going (10 hits/s)
VIBRO_RUN = 6          # hits in one column before it counts as vibro, not a jack
VIBRO_SHARE = .10      # share of notes in such runs: 117/120 vibro dan charts, 0.5% of ranked 4K (2026-10-01)
LN_SHARE = .22         # LN dans' charts hold >= 25% LN notes, rice dans' <= 19% (2026-10-01)


def vibro_runs(notes, rate=1.0):
    """{share, speed}: share of notes in same-column runs (>= VIBRO_RUN hits, <= VIBRO_GAP apart at
    this rate) and their median hits per second. Rolls count too: their columns repeat just as fast."""
    cols = collections.defaultdict(list)
    for start, _end, col in notes:
        cols[col].append(start / rate)
    inrun, gaps = 0, []
    for times in cols.values():
        times.sort()
        run = times[:1]
        for t in times[1:] + [math.inf]:
            if t - run[-1] <= VIBRO_GAP:
                run.append(t)
                continue
            if len(run) >= VIBRO_RUN:
                inrun += len(run)
                gaps += [b - a for a, b in zip(run, run[1:])]
            run = [t]
    gaps.sort()
    return {"share": inrun / max(1, len(notes)), "speed": 1000 / gaps[len(gaps) // 2] if gaps else 0.}


@functools.lru_cache(maxsize=1)
def _data():
    try:
        with open(TABLE, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    if data.get("calc") != recdata.calc_id():
        return {}                       # another calculator's ratings would misplace every chart
    return data


def _table():
    return {s: [(t["tier"], t["value"]) for t in tiers] for s, tiers in _data().get("series", {}).items()}


def monotone(points):
    """[(order, tier, value)] → [(tier, value)] in order, adjacent violators pooled; tied tiers share the
    gap up to the next block so each stays reachable."""
    blocks = []
    for _o, tier, value in sorted(points):
        blocks.append([value, 1, [tier]])
        while len(blocks) > 1 and blocks[-2][0] >= blocks[-1][0]:
            v2, n2, t2 = blocks.pop()
            v1, n1, t1 = blocks.pop()
            blocks.append([(v1*n1 + v2*n2) / (n1 + n2), n1 + n2, t1 + t2])
    out = []
    for j, (value, n, names) in enumerate(blocks):
        step = (blocks[j+1][0] - value) / n if j + 1 < len(blocks) else 0.
        out += [(names[i], value + i*step) for i in range(n)]
    return out


def kind(ln_share):
    return "ln" if ln_share >= LN_SHARE else "rice"


def value(series, rating, vibro=None, weights=None):
    """The quantity a series' anchors are in: log stars, or for vibro dans weighted log speed + log stars."""
    if series == "4K Vibro":
        speed, stars = weights or _data().get("vibro_weights", (1., 0.))
        return speed * math.log(max(1., vibro["speed"])) + stars * math.log(max(1e-3, rating))
    return math.log(max(1e-3, rating))


def position(tiers, v):
    """Continuous tier index (0 = first anchor) of value v, linear between and beyond the anchors."""
    if len(tiers) == 1:
        return 0.
    j = 0 if v <= tiers[0][1] else len(tiers) - 2 if v >= tiers[-1][1] else \
        next(i for i in range(len(tiers) - 1) if tiers[i][1] <= v <= tiers[i+1][1])
    lo, hi = tiers[j][1], tiers[j+1][1]
    return j + .5 if hi - lo < 1e-9 else j + (v - lo) / (hi - lo)


def place(series, rating, vibro=None):
    """Nearest dan, with −/+ when the chart sits more than a quarter of the way to the next one."""
    tiers = _table().get(series)
    if not tiers:
        return None
    prefix = SHORT.get(series, "")
    p = position(tiers, value(series, rating, vibro))
    if p < -.5:
        return f"below {prefix}{tiers[0][0]}"
    if p > len(tiers) - .5:
        return f"{prefix}{tiers[-1][0]}+"
    j = min(len(tiers) - 1, max(0, round(p)))
    off = p - j
    return f"{prefix}{tiers[j][0]}" + ("+" if off > .25 else "−" if off < -.25 else "")


def label(keys, rating, ln_share, vibro=None):
    # Vibro last: the rice/LN dan is the main reading (user 2026-10-01).
    parts = [place(SERIES.get((keys, kind(ln_share))), rating),
             place("4K Vibro", rating, vibro) if keys == 4 and vibro and vibro["share"] >= VIBRO_SHARE else None]
    return " · ".join(p for p in parts if p) or None
