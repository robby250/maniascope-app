"""Interpretable, chart-only correction learned from population score residuals.

No map IDs, titles, public score lookup, or personal state enter inference. The
exported additive basis is evaluated identically for published and local charts.
Training/evaluation live in calib/fit_structural_residual.py.
"""
import native_backend
N = native_backend.load()

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
    """Stable compact descriptors of a feature dict (recdata.feats_from shape); scalar calibration
    cancels from skill ratios. Rate enters only through physical density and duration."""
    return N.residual_vector(f)


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
    TRAIN map residuals by calib/fit_structural_residual.py --trees. Chart-only inputs."""
    import difficulty_model
    return difficulty_model.calc(parameters).tree_shift(keys, baseline, values)


def tree_limit(shift, baseline):
    """Clip, and fade below 4★ like the additive stage (few scored easy charts to learn from)."""
    return max(-.25, min(.25, shift))*min(1., (baseline/4.)**2)


def correction(keys, baseline, values, parameters):
    """Calibrated stars of the additive basis stage."""
    import difficulty_model
    return difficulty_model.calc(parameters).correction(keys, baseline, values)
