"""Chart-only score calibration stages (calib/difficulty.json) over the native calculator.

No player/map lookup occurs here. Personal ability belongs to recommend.py.
"""
import json
import os
from collections import OrderedDict
from functools import lru_cache

import native_backend
N = native_backend.load()

PARAMETERS = os.path.join(os.path.dirname(__file__), "calib", "difficulty.json")
FEATURES = ("ln", "chordjack", "delay", "minijack", "technical", "sv",
            "endurance", "cj_endurance", "wide_jacks", "od_rice", "od_ln", "od_jack")
EXTRA_FEATURES = ("mash", "pseudo_mash", "ln_flow", "smooth_switch", "peak_pressure", "transition_pressure")


@lru_cache(maxsize=1)
def parameters():
    try:
        with open(PARAMETERS, encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return {}


_CALCS = OrderedDict()


def calc(params=None, overrides=None):
    """Native calculator for a parameters dict (default: parameters(), which calibration may patch)
    and knob overrides {name: value}. The cache holds the dict itself, so its id is never reused."""
    params = parameters() if params is None else params
    extra = json.dumps(overrides, sort_keys=True) if overrides else ""
    key = (id(params), extra)
    hit = _CALCS.get(key)
    if hit is None or hit[0] is not params:
        hit = _CALCS[key] = (params, N.Calc(json.dumps(params), extra))
        while len(_CALCS) > 8:
            _CALCS.popitem(last=False)
    _CALCS.move_to_end(key)
    return hit[1]


def endurance(runs, bin_seconds=.5, phases=8):
    """Hand-local fatigue accumulation/recovery relative to hardest 30s demand (runs: _demand-shaped
    (parts, totals, fines) tuples)."""
    (load, peak), timeline = N.endurance([run[2] for run in runs])
    return {"load": load, "peak": peak}, timeline


def adjustment(keys, raw, features):
    return calc().adjustment(keys, raw, list(features))


def _baseline_rating(keys, raw, features, execution=None):
    return calc().baseline_rating(keys, raw, list(features), execution)


def final_rating(keys, raw, features, execution=None):
    return calc().final_rating(keys, raw, list(features), execution)


def gesture_rating(keys, base, raw, execution=None):
    """Incremental, TRAIN-fitted gesture correction; absent modes are unchanged."""
    return calc().gesture_rating(keys, base, raw, execution)
