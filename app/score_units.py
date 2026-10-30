"""Fixed displayed difficulty, separate from the predictor's fitted coordinates."""
import math

import native_backend
N = native_backend.load()


def rating(value, od, model):
    return N.unit_rating(value, od, model)


def display_rating(value, od, model):
    """Halfway between pre-conversion demand and the compressed reference scale."""
    return (value + rating(value, od, model)) / 2


def display_factor(keys, baseline, execution, original, parameters):
    """Gameplay-reference correction using the existing chart-only basis."""
    import difficulty_model
    return difficulty_model.calc(parameters).display_factor(keys, baseline, execution, original)


def feature_display_factor(f):
    import difficulty_model
    raw = f.get('preunit_features', f)
    return display_factor(f.get('keys'), raw.get('baseline_overall', raw['overall']),
                          f.get('execution', {}), raw['overall'], difficulty_model.parameters())


def displayed_feature(f):
    factor = feature_display_factor(f)
    if factor == 1.:
        return (f['overall'] + f.get('preunit_overall', f['overall'])) / 2
    import difficulty_model
    model = difficulty_model.parameters()['score_units']['modes'][str(f['keys'])]
    # Unconverted caches (63 of G835LX's 7K charts) hold the pre-unit rating as 'overall'.
    return display_rating(f.get('preunit_overall', f['overall'])*factor, f['od'], model)


def displayed_value(value, keys, od=8., factor=1.):
    """Convert a fitted reference-coordinate estimate without refitting its model."""
    import difficulty_model
    model = difficulty_model.parameters().get('score_units', {}).get('modes', {}).get(str(keys))
    if value is None or value <= 0 or not model:
        return value
    knee = math.log(model.get('knee', 4.))
    x = math.log(value)
    origin, slope, curve = model['reference']
    target = origin + slope*x + curve*max(0., x-knee)**2
    target -= model['window']*math.log((64.-3*max(0., min(10., od))) /
                                     (64.-3*model.get('reference_od', 8.)))
    a, b = model['input']
    delta = target-a*knee
    original = math.exp(target/a if delta <= 0 or b == 0 else
                        knee+2*delta/(a+math.sqrt(a*a+4*b*delta)))
    return (value+original)/2 if factor == 1. else display_rating(original*factor, od, model)


def feature(f, parameters):
    """Convert an exact cache without changing the fitted horizon schema."""
    model = parameters.get('score_units', {}).get('modes', {}).get(str(f['keys']))
    if not model or f['overall'] <= 0 or 'preunit_features' in f:
        return f
    f['preunit_features'] = {k: f[k] for k in ('overall', 'sk', 'stam', 'baseline_overall')}
    old = f['overall']
    convert = lambda value: rating(value, f['od'], model)
    f['preunit_overall'] = old
    f['overall'] = convert(old)
    f['sk'] = {key: round(convert(value*old)/f['overall'], 4) for key,value in f['sk'].items()}
    f['stam'] = convert(f['stam']*old)/f['overall']
    f['baseline_overall'] = convert(f['baseline_overall'])
    return f


def restore_feature(f):
    """Calibration tools refit original coordinates before reapplying units."""
    if 'preunit_features' in f:
        f.update(f.pop('preunit_features'))
        f.pop('preunit_overall', None)
    return f
