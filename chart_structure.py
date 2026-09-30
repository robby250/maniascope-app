"""Shared chart-only geometry for difficulty and local recommendations."""
import collections
import hashlib
import math
import numpy as np

STRUCTURE_VERSION = 2

def structure(chart, rate=1.0):
    """Small chart descriptors, not a second difficulty calculator.

    Long fast same-column runs follow the calculator's vibro-gap/run convention.
    The avoidance decision additionally requires pervasive repeated row shapes;
    short anchors/minijacks inside otherwise varied stamina maps are not a ban.
    """
    notes = chart.notes
    if len(notes) < 2:
        return {"version": STRUCTURE_VERSION, "nps": None}
    heads = np.array([x[0] for x in notes], dtype=float)
    ends = np.array([x[1] for x in notes], dtype=float)
    span = float(heads[-1] - heads[0])
    play_span = float(ends.max() - heads[0])
    if span <= 0:
        return {"version": STRUCTURE_VERSION, "nps": None}
    rows = collections.defaultdict(int)
    cols = [[] for _ in range(chart.keys)]
    for t, e, col in notes:
        rows[t] |= 1 << col
        cols[col].append(t)
    masks = list(rows.values())
    row_times = np.fromiter(rows, dtype=float)
    sizes = np.array([m.bit_count() for m in masks], dtype=float)
    overlap = [((a & b).bit_count() / max(a.bit_count(), b.bit_count())) for a, b in zip(masks, masks[1:])]
    repeated = sum(a == b for a, b in zip(masks, masks[1:])) / max(1, len(masks) - 1)
    import skill_calc
    fast_count, longest = 0, 0.0
    for times in cols:
        start = 0
        for i in range(1, len(times) + 1):
            if i < len(times) and times[i] - times[i - 1] < 1000 * skill_calc.VIBRO_GAP * rate:
                continue
            if i - start >= skill_calc.VIBRO_MIN:
                fast_count += i - start
                longest = max(longest, (times[i - 1] - times[start]) / 1000 / rate)
            start = i
    # Layout plus coarse normalized phrase timing groups real baked-rate copies.
    # It is used only for variety, NEVER to merge accuracy observations.
    layout = bytes([col for _t, _e, col in notes])
    sample = np.linspace(0, len(heads) - 1, min(64, len(heads))).astype(int)
    timing = np.round((heads[sample] - heads[0]) / span * 200).astype('<i2').tobytes()
    holds = bytes([int(e > t) for t, e, _c in notes])
    family = hashlib.sha256(bytes([chart.keys]) + layout + holds + timing).hexdigest()[:24]
    bin_counts = np.bincount(((heads - heads[0]) / 5000).astype(int)) / 5
    mean = len(notes) / (span / 1000)
    cycles = max((sum(masks[i] == masks[i-lag] for i in range(lag, len(masks)))
                  / max(1, len(masks)-lag) for lag in (1, 2, 4, 8)), default=0.)
    # Repetition alone is not bad mapping: only pervasive fixed shapes, near-flat
    # density and large chords make a drill. Legitimate CJ stamina stays eligible.
    flat = max(0., 1. - float(np.std(bin_counts)) / max(1., mean) / .35)
    wide = min(1., max(0., float(sizes.mean()) - 1.5) / 1.5)
    drill = max(0., (cycles - .65) / .35) * flat * wide
    all_keys = float(np.mean(sizes >= chart.keys))
    drill = max(drill, max(0., (all_keys - .45) / .55) * flat)
    return {"version": STRUCTURE_VERSION, "nps": mean * rate,
            "head_span": span / 1000, "play_span": play_span / 1000, "family": family,
            "chord": float(sizes.mean()), "wide": float(np.mean(sizes >= 3)),
            "repeat": repeated, "overlap": float(np.mean(overlap)) if overlap else 0.0,
            "burst": float(np.percentile(bin_counts, 95) / max(mean, 1e-6)),
            "active": float(np.mean(bin_counts > .4 * mean)),
            "rhythm": float(np.std(np.diff(row_times)) / max(1, np.mean(np.diff(row_times)))) if len(row_times) > 1 else 0.,
            "vibro_share": fast_count / len(notes), "vibro_longest": longest,
            "vibro_focused": fast_count / len(notes) > .6 and repeated > .65 and longest > 3.0,
            "periodicity": cycles, "all_keys": all_keys, "drill": min(1., drill)}

