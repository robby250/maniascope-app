"""Chart-only score calibration and chronological endurance features.

No player/map lookup occurs here. Personal ability belongs to recommend.py.
"""
import json
import math
import os
from functools import lru_cache

PARAMETERS = os.path.join(os.path.dirname(__file__), "calib", "difficulty.json")
FEATURES = ("ln", "chordjack", "delay", "minijack", "technical", "sv",
            "endurance", "cj_endurance", "wide_jacks", "od_rice", "od_ln", "od_jack")
EXTRA_FEATURES = ("mash", "pseudo_mash", "ln_flow", "smooth_switch", "peak_pressure", "transition_pressure")


def endurance(runs, bin_seconds=.5, phases=8):
    """Hand-local fatigue accumulation/recovery relative to hardest 30s demand.

    This measures intrinsic continuity, not a particular player's remaining
    stamina. The predictor learns how strongly it matters near their limit.
    """
    summaries, tracks = [], []
    for run in runs:
        for fine in run[2][:2]:
            values = [sum(fine[i:i + phases]) for i in range(0, len(fine), phases)]
            n = max(1, round(30 / bin_seconds))
            rolling, peak = 0., 0.
            for i, v in enumerate(values):
                rolling += v - (values[i - n] if i >= n else 0.)
                peak = max(peak, rolling / n)
            if peak <= 0:
                tracks.append([0.] * len(values))
                summaries.append(0.)
                continue
            state, weighted, total, track = 0., 0., 0., []
            for v in values:
                load = min(3., max(0., v / peak))
                target = max(0., load - .2) ** 1.3
                tau = 45. if target > state else 7. + 20. * min(1., load)
                decay = math.exp(-bin_seconds / tau)
                state = target + (state - target) * decay
                track.append(state)
                weighted += v * state
                total += v
            tracks.append(track)
            summaries.append(weighted / total if total else 0.)
    if not summaries:
        return {"load": 0., "peak": 0.}, []
    per_assignment = [max(summaries[i:i + 2]) for i in range(0, len(summaries), 2)]
    timeline = [sum(max(tracks[j][i], tracks[j + 1][i]) for j in range(0, len(tracks), 2))
                / max(1, len(runs)) for i in range(len(tracks[0]))]
    return {"load": sum(per_assignment) / len(per_assignment), "peak": max(timeline, default=0.)}, timeline


def vector(res, od, rate):
    sc = res["scores"]
    sk = {k: max(0., min(1.5, v)) for k,v in res.get("physical_sk", {}).items()} or {
        k: max(0., min(1.5, v / max(.05, sc["overall"]))) for k, v in sc.items()}
    ln, cj = sk.get("ln", 0.), max(sk.get("chordjack", 0.), sk.get("jackspeed", 0.))
    load = res.get("endurance", {}).get("load", 0.)
    strict = math.log(40. / max(10., 64. - 3. * max(0., min(10., od))))
    return [ln, cj, sk.get("delay", sk.get("dump", 0.)), sk.get("minijack", 0.),
            sk.get("technical", 0.), sk.get("sv", 0.), load, load * cj, res.get("jack_width", 0.),
            strict * max(0., 1. - max(ln, cj)), strict * ln, strict * cj]


@lru_cache(maxsize=1)
def parameters():
    try:
        with open(PARAMETERS, encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return {}


def adjustment(keys, raw, features):
    p = parameters().get("modes", {}).get(str(keys), {})
    coeff = p.get("coeff", [0.] * len(FEATURES))
    # Invert a monotone score link exactly. Dividing by its local derivative was
    # only a first-order approximation and could reverse easy-map rate ordering.
    slope, curve = p.get("slope", 2.), max(0., p.get("curve", 0.))
    x = math.log(max(.01, raw))
    # These effects were learned in players' competence zones, not on tutorial
    # maps. Fade the correction below that range rather than extrapolate strong
    # stamina/skill coefficients into sparse easy patterns.
    reliability = min(1., (max(0., raw) / 4.) ** 2)
    delta = sum(a * b for a, b in zip(coeff, features)) * p.get("strength", 1.) * reliability
    if curve < 1e-8:
        shift = delta / max(1., slope)
    else:
        knee = (1. - slope) / (2 * curve)
        u = x - knee
        y = u * (1. + curve*u) if u >= 0 else u
        y += delta
        v = 2*y / (1. + math.sqrt(1. + 4*curve*y)) if y >= 0 else y
        shift = v - u
    return math.exp(max(-.7, min(.7, shift))) * p.get("scale", 1.)


def _baseline_rating(keys, raw, features, execution=None):
    physical = (execution or {}).get("scroll_base")
    if physical and physical["raw"] > 0:
        return raw/physical["raw"] * _baseline_rating(keys, physical["raw"], physical["features"], physical["execution"])
    base = raw * adjustment(keys, raw, features)
    p = parameters().get("execution", {}).get("modes", {}).get(str(keys))
    if not p:
        return gesture_rating(keys, base, raw, execution)
    center = p.get("center", [0.]*len(EXTRA_FEATURES))
    delta = sum(c*((execution or {}).get(name, 0.)-m) for name,c,m in zip(EXTRA_FEATURES,p["coeff"],center))
    delta += sum(c*(v-m) for c,v,m in zip(p.get('base_coeff',()), features, p.get('base_center',())))
    delta *= min(1., (raw/4.)**2)
    # Invert the fitted monotone log-rating -> log-error curve, rather than
    # applying an arbitrary percent discount to charts with a Mash label.
    slope, curve = p["slope"], max(0., p.get("curve", 0.))
    x = math.log(max(.05, base))
    if curve < 1e-8:
        shift = delta/max(1.,slope)
    else:
        # Continue the learned quadratic linearly below derivative=1. This
        # keeps the inverse defined for easy charts and negative corrections;
        # clamping a discriminant would silently solve a different equation.
        knee=(1.-slope)/(2*curve)
        u=x-knee
        y=u*(1.+curve*u) if u>=0 else u
        y+=delta
        v=2*y/(1.+math.sqrt(1.+4*curve*y)) if y>=0 else y
        shift=v-u
    return gesture_rating(keys, base*math.exp(max(-.5, min(.5, shift))), raw, execution)


def final_rating(keys, raw, features, execution=None):
    import structural_residual
    base = _baseline_rating(keys, raw, features, execution)
    values=(execution or {}).get('residual_vector'); p=parameters()
    value=structural_residual.correction(keys, base, values, p)*math.exp(structural_residual.tree_shift(keys, base, values, p))
    return value*(execution or {}).get('rolled_factor',1.)


def gesture_rating(keys, base, raw, execution=None):
    """Incremental, TRAIN-fitted gesture correction; absent modes are unchanged.

    Zero gesture evidence has zero correction. The earlier physical/execution
    coordinates remain intact, so unsupported specialist effects can be withheld
    without discarding the corrected public-facing pattern descriptions.
    """
    stage = parameters().get("gestures", {})
    p = stage.get("modes", {}).get(str(keys))
    if not p or base <= 0:
        return base
    delta = sum(c*(execution or {}).get(name, 0.)
                for name,c in zip(stage.get("features", ()),p["coeff"]))
    if not delta:
        return base
    delta *= min(1., (max(0.,raw)/4.)**2)
    slope,curve = p["slope"],max(0.,p.get("curve",0.))
    x = math.log(max(.05,base))
    if curve < 1e-8:
        shift = delta/max(1.,slope)
    else:
        knee = (1.-slope)/(2*curve)
        u = x-knee
        y = u*(1.+curve*u) if u >= 0 else u
        y += delta
        v = 2*y/(1.+math.sqrt(1.+4*curve*y)) if y >= 0 else y
        shift = v-u
    return base*math.exp(max(-.5,min(.5,shift)))


def apply(res, od, rate):
    raw = res["scores"]["overall"]
    res["raw_overall"] = raw
    res["calibration_features"] = vector(res, od, rate)
    import structural_residual
    if raw and 'structure' in res:
        feature = {'keys':res['keys'], 'overall':raw, 'rate':rate,
                   'sk':{k:round(v/raw,4) for k,v in res['scores'].items()},
                   'ln':res['ln_notes']/max(1,res['notes']), 'notes':res['notes'],
                   'length':res.get('length',0.), 'nps':res['structure'],
                   'stam':res.get('levels',{}).get(120.,raw)/raw,
                   'endurance':res.get('endurance',{}), 'execution':res.get('execution',{}),
                   'calibration_features':res['calibration_features']}
        res['execution']['residual_vector'] = structural_residual.vector(feature)
    res['baseline_overall'] = _baseline_rating(res['keys'], raw, res['calibration_features'], res.get('execution'))
    multiplier = final_rating(res["keys"], raw, res["calibration_features"], res.get("execution")) / max(.000001, raw) if raw else 1.
    for k in res["scores"]:
        res["scores"][k] *= multiplier
    for a in res.get("archetypes", []):
        a["rating"] *= multiplier
    res["timeline"] = [v * multiplier for v in res.get("timeline", [])]
    res["levels"] = {h: v * multiplier for h, v in res.get("levels", {}).items()}
    res["od"] = od
    res['structural_correction'] = res['scores']['overall'] - res['baseline_overall']
    return res
