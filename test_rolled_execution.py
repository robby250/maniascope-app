"""Execution-equivalence controls: no song names or hard-coded chart ratings."""
from copy import deepcopy
from unittest.mock import patch
import math

import pytest

import difficulty_model as D
import skill_calc as S
from test_gestures import rolls,repeated
from test_execution import chart


def configured(c, strength=1., rate=1., sv=True):
    params=deepcopy(D.parameters())
    params['rolled_execution']={'modes':{str(c.keys):{'strength':strength}}}
    with patch.object(D,'parameters',return_value=params):
        return S.compute(c,rate,sv)


@pytest.mark.parametrize('keys',range(4,11))
def test_planning_does_not_change_existing_calibration_features(keys):
    cols=tuple(range(keys//2))+tuple(range((keys+1)//2,keys))
    c=rolls(cols,gap=12,keys=keys)
    old=S.N.gestures_analyse(c);new=S.N.gestures_analyse(c,with_plans=True)
    assert new['series']==old['series'] and new['features']==old['features']
    assert any(new['plans'].values())
    window=S.N.great_window(c.od)
    origin=c.notes[0][0]
    for plans in new['plans'].values():
        for p in plans:
            assert len(p['ids'])==len(set(c.notes[i][2] for i in p['ids']))
            for i,offset in zip(p['ids'],p['offsets']):
                assert abs(p['time']+offset-(c.notes[i][0]-origin)*.001)<=window+1e-10


def test_rolls_share_cost_but_fixed_jt_split_trill_and_slow_stream_are_unchanged():
    for c in (repeated(((0,1),(2,3)),gap=60), repeated(((0,2),(1,3)),gap=60),
              rolls((0,1,2,3),gap=70), repeated(((0,1,2,3),),gap=60)):
        before=configured(c,0.);after=configured(c)
        assert after==before
    c=rolls((0,1,2,3),gap=30,n=2400)
    before=configured(c,0.);after=configured(c)
    assert after['scores']['overall']<before['scores']['overall']
    assert after['scores']['jumptrill']>after['scores']['stream']
    assert after['execution']['rolled_factor']<1.
    assert after['notes']==before['notes']


def test_interleaved_rolling_offsets_are_not_simultaneous_hit_errors():
    c=rolls((0,2,1,3),gap=27,n=2400)
    info=S.N.gestures_analyse(c,with_plans=True)
    plans=info['plans'][False]
    assert len(plans)>100
    # Old simultaneous feasibility has narrow margins in these otherwise
    # regular hand cycles; rolling retains the internal per-finger offsets.
    assert sum(p['strength'] for p in plans)/len(plans)>.9
    assert any(p['span']>.05 for p in plans)
    after=configured(c)
    assert after['scores']['overall']<configured(c,0.)['scores']['overall']


def test_control_and_actual_repeats_limit_sharing():
    clean=rolls((0,1,2,3),gap=27,n=2400)
    turns=rolls((0,1,2,3,1,0,3,2),gap=27,n=2400)
    def support(c):
        return configured(c)['execution'].get('rolled_support',0.)
    assert support(turns)<support(clean)
    assert support(rolls((0,1,0,2,0,3),gap=27))<.1
    held=chart([(t,t+15,k) for t,e,k in clean.notes],keys=4,od=8)
    assert configured(held)==configured(held,0.)


def test_no_chart_or_metadata_mutation_and_support_is_rate_specific():
    c=rolls((0,1,3,2),gap=35,n=1600)
    original=deepcopy(c.notes)
    renamed=deepcopy(c);renamed.title='Unsubmitted completely different song';renamed.creator='Unknown'
    a=configured(c);b=configured(renamed)
    assert a==b and c.notes==original
    slow=configured(c,rate=.5)
    assert slow==configured(c,0.,rate=.5)
    assert not any(S.N.gestures_analyse(c,.5,with_plans=True)['plans'].values())


def test_locality_leaves_unrelated_phrase_timeline_untouched():
    plain=rolls((0,1,2,3),gap=90,n=500)
    quick=rolls((0,1,2,3),gap=27,n=1400)
    c=chart(plain.notes+[(t+60000,e+60000,k) for t,e,k in quick.notes],keys=4,od=8)
    before=configured(c,0.);after=configured(c)
    assert before['timeline'][:75]==pytest.approx(after['timeline'][:75],abs=1e-9)
    assert sum(after['timeline'][125:])<sum(before['timeline'][125:])


def test_sv_and_cached_feature_prediction_use_same_final_difficulty():
    c=rolls((0,1,3,2),gap=27,n=1600)
    c.sv=[(0,1.),(4000,.4),(5000,1.8),(6000,1.)]
    params=deepcopy(D.parameters());params['rolled_execution']={'modes':{'4':{'strength':1.}}}
    with patch.object(D,'parameters',return_value=params):
        full=S.compute(c);constant=S.compute(c,sv=False)
        assert full['no_sv_overall']==pytest.approx(constant['scores']['overall'],abs=1e-9)
        assert full['scores']['overall']>=constant['scores']['overall']-1e-9
        assert D.final_rating(4,full['raw_overall'],full['calibration_features'],full['execution'])==pytest.approx(full['scores']['overall'],rel=1e-12)


@pytest.mark.parametrize('keys,strength', [(4, 1.), (4, 0.), (7, 0.)])
def test_training_tree_input_matches_runtime_before_roll_and_units(keys, strength):
    import recdata
    import structural_residual as SR
    from calib.fit_structural_residual import additive_inputs, tree_inputs
    from calib.publish_structural import transform
    cols = (0,1,2,3) if keys==4 else (0,1,2,4,5,6)
    c = rolls(cols, gap=27, n=1600, keys=keys)
    rate = .83
    params = deepcopy(D.parameters())
    params['rolled_execution'] = {'modes': {str(keys): {'strength': strength}}}
    with patch.object(D, 'parameters', return_value=params):
        result = S.compute(c, rate)
        with patch.object(S, 'parse_osu', return_value=c), patch('feedback._od', return_value=c.od):
            f = recdata.chart_feats('/tree-coordinate-control.osu', (rate,), analyses={rate: result})[rate]
        # The runtime tree stage's inputs (difficulty_model.final_rating).
        baseline = D._baseline_rating(keys, result['raw_overall'], result['calibration_features'], result['execution'])
        values = result['execution']['residual_vector']
        if strength:
            assert f['execution']['rolled_factor'] < 1.
            assert f['baseline_overall'] != pytest.approx(baseline)
        if strength or keys==7:
            assert SR.vector(dict(f, rate=rate)) != values  # Former additive fitting path.
        before = deepcopy(f)
        for cached in (f, transform(f, rate)):
            assert tree_inputs(cached)[-1] == pytest.approx(math.log(max(.05, baseline)), abs=1e-12)
            assert additive_inputs(cached, rate) == values
        assert f == before


@pytest.mark.parametrize('factor', [0., -1., math.nan, math.inf])
def test_training_tree_input_rejects_unidentified_roll_coordinates(factor):
    from calib.fit_structural_residual import tree_inputs
    with pytest.raises(ValueError, match='finite positive rolled_factor'):
        tree_inputs({'overall':8., 'baseline_overall':7., 'execution':{'rolled_factor':factor}})


def test_training_tree_input_requires_a_baseline_only_for_non_neutral_rolls():
    from calib.fit_structural_residual import tree_inputs
    assert tree_inputs({'overall':8.}) == [math.log(8.)]
    with pytest.raises(ValueError, match='require baseline_overall'):
        tree_inputs({'overall':8., 'execution':{'rolled_factor':.8}})


def test_invalid_strength_and_cancelled_reference_are_explicit():
    c=rolls((0,1,2,3),gap=27)
    for strength in (-.1,1.1,math.nan,math.inf):
        with pytest.raises(ValueError):configured(c,strength)
    params=deepcopy(D.parameters());params['rolled_execution']={'modes':{'4':{'strength':1.}}}
    with patch.object(D,'parameters',return_value=params):
        with pytest.raises(S.AnalysisCancelled):S.compute(c,cancelled=lambda:True)


def test_credit_fades_before_the_packet_timing_boundary():
    c=rolls((0,2,1,3),gap=28,od=10,n=1400)
    values=[configured(c,rate=r)['scores']['overall'] for r in (.8234,.8236,.8238)]
    # Same-rate baseline drift is tiny here; no sudden grouped/un-grouped drop.
    assert max(values)-min(values)<.02
