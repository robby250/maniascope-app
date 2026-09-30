"""Dan equivalent of a rating: where it falls among the community dan courses (calib/dans.json).

Anchors are the courses themselves rated by this calculator (calib/dan_table.py), not another
calculator's scale (user 2026-10-01). The series follows the chart: vibro → 4K vibro dans,
LN-heavy → LN dans, otherwise rice dans. Keymodes/types without a course table show nothing.
"""
import functools
import json
import os

import recdata

TABLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib", "dans.json")
SERIES = {(4, "rice"): "4K REFORM", (4, "vibro"): "4K Vibro", (4, "ln"): "4K LN",
          (6, "rice"): "6K Regular", (6, "ln"): "6K LN", (7, "rice"): "7K Regular", (7, "ln"): "7K LN",
          (10, "rice"): "10K Regular"}
SHORT = {"7K Regular": "Reg ", "7K LN": "LN ", "4K LN": "LN ", "6K Regular": "6K ", "6K LN": "6K LN ",
         "10K Regular": "10K ", "4K REFORM": "", "4K Vibro": ""}


@functools.lru_cache(maxsize=1)
def _table():
    try:
        with open(TABLE, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    if data.get("calc") != recdata.calc_id():
        return {}                       # another calculator's ratings would misplace every chart
    return {s: _monotone(tiers) for s, tiers in data["series"].items()}


def _monotone(tiers):
    """Pool adjacent violators: a course rated below the one before it shares their mean."""
    blocks = []
    for t in sorted(tiers, key=lambda t: t["order"]):
        blocks.append([t["rating"], 1, [t["tier"]]])
        while len(blocks) > 1 and blocks[-2][0] >= blocks[-1][0]:
            v2, n2, t2 = blocks.pop()
            v1, n1, t1 = blocks.pop()
            blocks.append([(v1*n1 + v2*n2) / (n1 + n2), n1 + n2, t1 + t2])
    out = []
    for j, (value, n, names) in enumerate(blocks):
        # Tied tiers share the gap up to the next block, so each can still be shown.
        step = (blocks[j+1][0] - value) / n if j + 1 < len(blocks) else 0.
        out += [(names[i], value + i*step) for i in range(n)]
    return out


def kind(ln_share, card_names=()):
    if "Vibro" in list(card_names)[:3]:
        return "vibro"
    return "ln" if ln_share >= .5 else "rice"


def label(keys, rating, ln_share, card_names=()):
    series = SERIES.get((keys, kind(ln_share, card_names)))
    tiers = _table().get(series)
    if not tiers:
        return None
    prefix = SHORT.get(series, "")
    if rating < tiers[0][1]:
        return f"below {prefix}{tiers[0][0]}"
    if rating >= tiers[-1][1]:
        return f"{prefix}{tiers[-1][0]}+"
    for (name, lo), (_next, hi) in zip(tiers, tiers[1:]):
        if lo <= rating < hi:
            part = (rating - lo) / max(1e-9, hi - lo)
            return f"{prefix}{name} {'low' if part < 1/3 else 'mid' if part < 2/3 else 'high'}"
    return None
