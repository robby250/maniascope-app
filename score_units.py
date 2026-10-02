"""Fixed displayed difficulty, separate from the predictor's fitted coordinates."""
import math


def rating(value, od, model):
    if value <= 0:
        return value
    knee = model.get('knee', 4.)
    x = math.log(value)
    a, b = model['input']
    target = a*x + b*max(0., x-math.log(knee))**2
    target += model['window']*math.log((64.-3*max(0., min(10., od))) /
                                     (64.-3*model.get('reference_od', 8.)))
    origin, slope, curve = model['reference']
    if slope <= 0 or curve < 0:
        raise ValueError('Reference response must be monotone')
    delta = target-origin-slope*math.log(knee)
    if delta <= 0 or curve == 0:
        return math.exp((target-origin)/slope)
    return knee*math.exp(2*delta/(slope+math.sqrt(slope*slope+4*curve*delta)))


def display_rating(value, od, model):
    """Halfway between pre-conversion demand and the compressed reference scale."""
    return (value + rating(value, od, model)) / 2


def display_factor(keys, baseline, execution, original, parameters):
    """Gameplay-reference correction using the existing chart-only basis."""
    unit = parameters.get('score_units', {}).get('modes', {}).get(str(keys), {})
    coeff = unit.get('display_coeff')
    values = execution.get('residual_vector')
    if coeff is None or values is None or original <= 0:
        return 1.
    import structural_residual
    stage = parameters['structural_residual']
    model = dict(stage['modes'][str(keys)], coeff=coeff)
    display = {'structural_residual': dict(stage, modes={str(keys): model})}
    rolled = execution.get('rolled_factor', 1.)
    # The tree stage is part of the stars, not of the display basis: keep it on both sides of the ratio.
    corrected = structural_residual.correction(keys, baseline/rolled, values, display)*rolled \
        * math.exp(structural_residual.tree_shift(keys, baseline, values, parameters))
    return corrected/original


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


def apply(result, parameters):
    model = parameters.get('score_units', {}).get('modes', {}).get(str(result['keys']))
    if not model or not result.get('horizon'):
        return result
    factor = display_factor(result['keys'], result['baseline_overall'],
                            result.get('execution', {}), result['scores']['overall'], parameters)
    convert = lambda value: display_rating(value*factor, result['od'], model)
    # Retain exact inputs for feature extraction, including its historical
    # horizon schema and rounding. Predictions keep their trained coordinates;
    # changing the displayed scale must not change an accuracy forecast.
    result['preunit_result'] = {k: result[k] for k in ('scores', 'levels', 'baseline_overall')}
    result['preunit_overall'] = result['scores']['overall']
    result['scores'] = {key: convert(value) for key, value in result['scores'].items()}
    result['levels'] = {key: convert(value) for key, value in result['levels'].items()}
    result['timeline'] = [convert(value) for value in result['timeline']]
    for field in ('archetypes', 'unit_card'):
        for item in result.get(field, ()):
            item['rating'] = convert(item['rating'])
    result['baseline_overall'] = display_rating(result['baseline_overall'], result['od'], model)
    if 'no_sv_overall' in result:
        physical = result.get('execution', {}).get('scroll_base')
        no_sv_factor = factor
        if physical:
            import difficulty_model
            baseline = difficulty_model._baseline_rating(result['keys'], physical['raw'],
                                                         physical['features'], physical['execution'])
            no_sv_factor = display_factor(result['keys'], baseline, physical['execution'],
                                          result['no_sv_overall'], parameters)
        result['no_sv_overall'] = display_rating(result['no_sv_overall']*no_sv_factor, result['od'], model)
    result['structural_correction'] = result['scores']['overall']-result['baseline_overall']
    return result


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
