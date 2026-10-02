"""Interpretable, chart-only correction learned from population score residuals.

No map IDs, titles, public score lookup, or personal state enter inference. The
exported additive basis is evaluated identically for published and local charts.
Training/evaluation live in calib/fit_structural_residual.py.
"""
import math

VERSION = 1
SKILLS = ('stream', 'jumpstream', 'handstream', 'delay', 'chordstream', 'bracket',
          'chordjack', 'minijack', 'longjack', 'jackspeed', 'ln', 'release',
          'inverse', 'hybrid', 'shield', 'jumptrill', 'splittrill', 'trill1h',
          'anchor', 'technical', 'patterntech', 'rhythmtech', 'vibro', 'stamina', 'sv')
EXECUTION = ('mash', 'pseudo_mash', 'ln_flow', 'peak_pressure', 'transition_pressure',
             'vibro_easy', 'vibro_control', 'jumptrill_easy', 'roll_control',
             'roll_anchor', 'jumptrill_sustain')
NAMES = tuple('skill:'+s for s in SKILLS) + (
    'hold_fraction', 'sustain_ratio', 'log_duration', 'log_density_per_key',
    'chord_width', 'overlap', 'repeat', 'burst', 'active', 'rhythm_variation',
    'all_columns', 'fatigue_load', 'fatigue_peak', 'wide_jack_pressure',
) + tuple('execution:'+s for s in EXECUTION) + (
    'chordstream:bracket', 'chordjack:overlap', 'chordjack:width',
    'chordjack:sustain', 'minijack:density', 'ln:release', 'ln:hybrid',
    'ln:flow', 'rice:anchor', 'trill:control',
)


def vector(f):
    """Stable compact descriptors; scalar calibration cancels from skill ratios.

    Round ratios exactly like recdata.feats_from so an exported cache and a fresh
    calculation have the same inputs. Rate is represented by physical density
    and duration, never by a mod/beatmap identity.
    """
    sk = {k: round(max(0., min(1.5, float(v))), 4) for k,v in f.get('sk', {}).items()}
    st, ex = f.get('nps', {}), f.get('execution', {})
    en = f.get('endurance', {})
    rate = f.get('rate', 1.)
    seconds = max(.001, float(st.get('play_span') or f.get('length', 0.)/1000)/rate)
    density = float(st.get('nps') or f.get('notes', 0.)/seconds)/max(1, f['keys'])
    width = max(0., float(st.get('chord', 1.)))/max(1, f['keys'])
    overlap = max(0., float(st.get('overlap', 0.)))
    load = max(0., float(en.get('load', 0.)))
    cf = f.get('calibration_features', ())
    ln, cj = sk.get('ln', 0.), sk.get('chordjack', 0.)
    rice = max(sk.get(k, 0.) for k in ('stream','delay','dump'))
    values = [sk.get(k, 0.) for k in SKILLS] + [
        f.get('ln', 0.), min(2., f.get('stam', 1.)), math.log1p(seconds/120),
        math.log1p(density), width, overlap, st.get('repeat', 0.),
        min(5., st.get('burst', 1.)), st.get('active', 0.),
        min(5., st.get('rhythm', 0.)), st.get('all_keys', 0.),
        load, en.get('peak', 0.), cf[8] if len(cf)>8 else 0.,
    ] + [ex.get(k, 0.) for k in EXECUTION] + [
        sk.get('chordstream', 0.)*sk.get('bracket', 0.), cj*overlap, cj*width,
        cj*load, sk.get('minijack', 0.)*math.log1p(density),
        ln*sk.get('release', 0.), ln*sk.get('hybrid', 0.), ln*ex.get('ln_flow', 0.),
        rice*sk.get('anchor', 0.), sk.get('jumptrill', 0.)*ex.get('roll_control', 0.),
    ]
    return [float(v) if math.isfinite(float(v)) else 0. for v in values]


def basis(values, model):
    # Training quantile bounds avoid unbounded extrapolation on unusual charts.
    z = [max(lo,min(hi,x)) for x,lo,hi in zip(values,model['low'],model['high'])]
    columns = [(v-m)/s for v,m,s in zip(z,model['mean'],model['scale'])]
    for index, knot, mean, scale in model.get('hinges', ()):
        columns.append((max(0., z[index]-knot)-mean)/scale)
    return columns


def tree_predict(X, model):
    """Boosted oblivious trees, rows of X = values + [log baseline] → log-star shifts (numpy).
    With 'scales', each split is a logistic ramp of that width instead of a step: hard splits made
    the stars fall 1.8 % from 1.20× to 1.22× on a plain 7K stream (2026-10-02), while the
    rest of the calculator rises smoothly with rate."""
    import numpy as np
    X = np.asarray(X, float)
    out = np.full(len(X), float(model['intercept']))
    scales = model.get('scales')
    for feats, cuts, leaves in model['trees']:
        if scales is None:
            i = np.zeros(len(X), int)
            for f, c in zip(feats, cuts):
                i = 2*i + (X[:, f] > c)
            out += np.asarray(leaves)[i]
            continue
        weight = np.ones((len(X), 1))
        for f, c in zip(feats, cuts):
            p = 1/(1 + np.exp(-np.clip((X[:, f] - c)/scales[f], -60, 60)))[:, None]
            weight = np.stack([weight*(1 - p), weight*p], axis=2).reshape(len(X), -1)   # leaf = 2*leaf + bit
        out += weight @ np.asarray(leaves)
    return out


def tree_shift(keys, baseline, values, parameters):
    """Second stage after the additive basis: what the current stars still miss, learned from
    TRAIN map residuals by calib/fit_structural_residual.py --trees (user 2026-10-01: fix single
    maps, not just the average). Chart-only inputs, so local and unranked charts get it too."""
    model = parameters.get('structural_trees', {}).get('modes', {}).get(str(keys))
    if not model or baseline <= 0 or values is None or len(values) != len(NAMES):
        return 0.
    x = list(values) + [math.log(max(.05, baseline))]
    scales = model.get('scales')
    out = model['intercept']
    for feats, cuts, leaves in model['trees']:
        if scales is None:
            i = 0
            for f, c in zip(feats, cuts):
                i = 2*i + (x[f] > c)
            out += leaves[i]
            continue
        weight = [1.]
        for f, c in zip(feats, cuts):
            z = max(-60., min(60., (x[f] - c)/scales[f]))
            p = 1/(1 + math.exp(-z))
            weight = [w*q for w in weight for q in (1 - p, p)]
        out += sum(w*v for w, v in zip(weight, leaves))
    return tree_limit(out, baseline)


def tree_limit(shift, baseline):
    """Clip, and fade below 4★ like the additive stage (few scored easy charts to learn from)."""
    return max(-.25, min(.25, shift))*min(1., (baseline/4.)**2)


def correction(keys, baseline, values, parameters, explain=False):
    """Return calibrated stars (and optional named log-rating contributions)."""
    stage = parameters.get('structural_residual', {})
    model = stage.get('modes', {}).get(str(keys))
    if not model or baseline <= 0 or values is None or len(values)!=len(NAMES):
        return (baseline, []) if explain else baseline
    contributions = [a*b for a,b in zip(basis(values,model),model['coeff'])]
    delta = sum(contributions) + model.get('intercept', 0.)
    delta = max(-stage.get('max_log_shift', .25), min(stage.get('max_log_shift', .25), delta))
    # Preserve tutorials and prevent sparse short files being extrapolated into
    # a high-skill population's competence zone.
    delta *= min(1., (baseline/4.)**2)
    corrected = baseline*math.exp(delta)
    if not explain:
        return corrected
    names = list(NAMES) + [NAMES[i]+f'>{k:.3g}' for i,k,_m,_s in model.get('hinges', ())]
    details = sorted(zip(names,contributions),key=lambda x:-abs(x[1]))[:6]
    return corrected, details
