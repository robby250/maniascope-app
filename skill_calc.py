#!/usr/bin/env python3
"""
ManiaScope difficulty model — native osu!mania, 7K-first, 4K-10K supported.

The calculator is the Rust crate in native/ (maniascope_native); this module is its Python face.
Rating convention: the skill a well-rounded player needs for controlled, consistently accurate play.
Every displayed number — overall, each skill, stamina — comes from the SAME demand unit, the SAME
time aggregation and the SAME scale curve. Pipeline and knob meanings: native/src/demand.rs,
skills.rs and params.rs.

Usage: skill_calc.py <file.osu> [rate]
"""
import functools
import json
import os

import native_backend
N = native_backend.load()

MODEL_VERSION = N.MODEL_VERSION
Chart = N.Chart
ChartError = N.ChartError
SV_KINDS = ("sv_fast", "sv_slow", "sv_accel", "sv_stutter", "sv_brake")

# Knob defaults (overridable per calculation: compute(..., overrides={name: value})).
KNOBS = {k: tuple(tuple(x) if isinstance(x, list) else x for x in v) if isinstance(v, list) else v
         for k, v in json.loads(N.default_knobs()).items()}
globals().update(KNOBS)


class AnalysisCancelled(Exception):
    """The viewer selected another chart; no partial analysis is a result."""


def _check_cancelled(cancelled):
    if cancelled is not None and cancelled():
        raise AnalysisCancelled()


@functools.lru_cache(maxsize=16)
def _parse_cached(path, _mtime):
    with open(path, "rb") as fh:
        return N.parse(fh.read())


def parse_osu(path):
    """Parse by file revision, including replacements that retain their mtime."""
    s = os.stat(path)
    return _parse_cached(path, (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns))


def _calc(overrides):
    import difficulty_model
    return difficulty_model.calc(overrides=overrides)


def compute(chart, rate=1.0, sv=True, *, cancelled=None, overrides=None):
    """Full analysis of a parsed Chart (or .osu bytes) at a constant rate. sv=False: scroll changes
    ignored (Constant Speed). Cancelled selections return no result and are never cached."""
    _check_cancelled(cancelled)
    result = _calc(overrides).compute(chart, [rate], sv)[0]
    _check_cancelled(cancelled)
    return result


def compute_batch(jobs, sv=True, threads=0, overrides=None):
    """[(osu bytes, [rates])] → per job: [result per rate], or the error message string."""
    return _calc(overrides).compute_batch(jobs, sv, threads)


def trace(chart, rate=1.0, sv=True, overrides=None, plans=True):
    """Calibration trace of one compute (demand parts/phases/series per sv pass, gestures)."""
    return _calc(overrides).trace(chart, rate, sv, plans)


def demand_runs(chart, rate=1.0, sv=True, overrides=None):
    """Demand per centre-key assignment (odd keymodes: left then right thumb):
    [(parts {name: per-bin list}, totals, (fine hand 0, fine hand 1, fine both))]."""
    return [r["run"] for r in _calc(overrides).demand(chart, rate, sv)]


def sv_reading(chart, kinds=None, rate=1.0, detail=None, *, cancelled=None, overrides=None):
    """Per-note scroll-reading strain at the actual rate; Constant Speed bypasses this entirely."""
    _check_cancelled(cancelled)
    strain, shares, summary = _calc(overrides).reading(chart, rate)
    if kinds is not None:
        kinds.extend(shares)
    if detail is not None:
        detail.update(summary)
    return strain


def skills(keys):
    """Skill keys measured for a keymode, in display order for ties."""
    return tuple(N.skills_of(keys))


NAMES = {k: N.skill_name(k) for k in dict.fromkeys(skills(4) + skills(7) + ("ln", "sv", "stamina"))}


def skill_names(keys):
    """Display names of the keymode's skills."""
    return {k: NAMES[k] for k in skills(keys)}


def ranked(res):
    """The keymode's skills, strongest first (ties keep display order: patterns before Stamina)."""
    return N.ranked(res)


def card(res):
    """Up to 3 non-redundant card entries [{"name", "rating", "parts"}]."""
    return N.card(res)


def describe(res):
    """→ (dominant text, [structural tags]) from a compute() result."""
    return N.describe(res)


def main():
    import sys
    if len(sys.argv) < 2:
        print("usage: skill_calc.py <file.osu> [rate]", file=sys.stderr)
        return 2
    rate = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    res = compute(parse_osu(sys.argv[1]), rate)
    res["scores"] = {k: round(v, 2) for k, v in res["scores"].items()}
    res["described"] = describe(res) if res["horizon"] else None
    res.pop("timeline")
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
