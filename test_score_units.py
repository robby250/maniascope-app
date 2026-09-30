import copy
import math
import pytest
import score_units as U
from unittest.mock import patch

import difficulty_model as D
import recdata
import skill_calc as S
from test_execution import chart


def test_reference_conversion_is_monotone_and_shared_by_all_displayed_demands():
    model = {'input':[2., .7], 'window':-.8, 'reference':[0., 2., .7],
             'reference_od':8., 'knee':4.}
    for value in (.01, .3, 1., 3.99, 4., 4.01, 8., 15., 50.):
        assert U.rating(value, 8., model) == pytest.approx(value, rel=1e-12)
    assert U.rating(8., 0., model) < U.rating(8., 8., model) < U.rating(8., 10., model)
    values = [U.rating(.1+i*.03, 9., model) for i in range(1000)]
    assert all(a < b for a,b in zip(values, values[1:]))
    model['reference'] = [.4, 2., 1.2]
    params = {'score_units': {'modes': {'7': model}}}
    source = {'keys':7, 'od':5., 'horizon':8., 'scores':{'overall':10., 'stream':8., 'ln':0.},
              'levels':{8.:8.}, 'timeline':[0.,8.,10.], 'archetypes':[{'rating':8.}],
              'baseline_overall':9., 'no_sv_overall':8.}
    result = U.apply(copy.deepcopy(source), params)
    expected = U.display_rating(8., 5., model)
    assert result['scores']['stream'] == result['levels'][8.] == result['timeline'][1] == expected
    assert result['archetypes'][0]['rating'] == result['no_sv_overall'] == expected
    assert result['scores']['ln'] == result['timeline'][0] == 0.
    assert result['preunit_overall'] == 10.
    assert U.apply(dict(source, keys=6), params)['scores'] == source['scores']


def test_midpoint_display_keeps_fitted_features_in_their_original_coordinates():
    params=D.parameters();model=params['score_units']['modes']['7']
    for od in (0.,6.,8.,10.):
        for value in (.1,4.,7.,10.,18.,28.):
            raw={'keys':7,'overall':value,'od':od,'sk':{'ln':.8},'stam':.9,'baseline_overall':value*.95}
            f=U.feature(copy.deepcopy(raw),params)
            reference=U.rating(value,od,model)
            assert f['overall']==reference
            assert f['sk']['ln']==round(U.rating(.8*value,od,model)/reference,4)
            expected=(value+reference)/2
            assert U.display_rating(value,od,model)==expected
            assert U.displayed_feature(f)==expected
            assert U.displayed_value(reference,7,od)==pytest.approx(expected)
    assert U.displayed_value(12.,4)==12.


def test_fresh_and_cached_reference_units_preserve_schema_labels_and_other_modes():
    params=copy.deepcopy(D.parameters())
    params['score_units']={'modes':{'7':{'input':[1.6,.9], 'window':-.85,
        'reference':[.48,1.28,1.68], 'reference_od':8., 'knee':4.}}}
    disabled=copy.deepcopy(params);disabled.pop('score_units')
    c=chart([(1000+95*i,1000+95*i+(140 if i%9==0 else 0),i%7) for i in range(400)],od=7)
    c.sv=[(0,1.),(5000,.4),(8000,1.6),(12000,1.)]
    notes=list(c.notes)
    for keys in (4,7):
        c.keys=keys;c.notes=[(t,e,col%keys) for t,e,col in notes]
        for rate in (.75,1.,1.5):
            with patch.object(S,'parse_osu',return_value=c), patch('feedback._od',return_value=c.od):
                with patch.object(D,'parameters',return_value=disabled):
                    old=S.compute(c,rate)
                    cached=recdata.chart_feats('/unavailable-unit-control.osu',(rate,),analyses={rate:old})[rate]
                with patch.object(D,'parameters',return_value=params):
                    fresh=S.compute(c,rate)
                    actual=recdata.chart_feats('/unavailable-unit-control.osu',(rate,),analyses={rate:fresh})[rate]
                    assert fresh['no_sv_overall']==pytest.approx(S.compute(c,rate,False)['scores']['overall'],rel=1e-12)
            expected=U.feature(copy.deepcopy(cached),params)
            assert actual==expected
            assert S.describe(fresh)[0]==S.describe(old)[0]
            assert [e['name'] for e in S.card(fresh)]==[e['name'] for e in S.card(old)]
            assert U.feature(copy.deepcopy(expected),params)==expected
            assert U.restore_feature(copy.deepcopy(expected))==cached
            if keys==4:assert fresh==old


def test_gameplay_display_correction_preserves_forecast_inputs_and_labels():
    enabled=copy.deepcopy(D.parameters())
    disabled=copy.deepcopy(enabled)
    disabled['score_units']['modes']['7'].pop('display_coeff')
    c=chart([(1000+65*i,1000+65*i+(170 if i%5==0 else 0),i%7) for i in range(400)],od=8)
    c.sv=[(0,1.),(3000,.4),(8000,1.6),(12000,1.)]
    for rate in (.75,1.,1.5):
        with patch.object(S,'parse_osu',return_value=c), patch('feedback._od',return_value=c.od):
            with patch.object(D,'parameters',return_value=disabled):
                before=S.compute(c,rate)
                old=recdata.chart_feats('/display-reference.osu',(rate,),analyses={rate:before})[rate]
            with patch.object(D,'parameters',return_value=enabled):
                after=S.compute(c,rate)
                f=recdata.chart_feats('/display-reference.osu',(rate,),analyses={rate:after})[rate]
                assert f==old  # every predictor input, including labels and SV context
                assert after['preunit_result']==before['preunit_result']
                assert after['scores']['overall']!=before['scores']['overall']
                assert U.displayed_feature(f)==pytest.approx(after['scores']['overall'],rel=1e-12)
                assert after['no_sv_overall']==pytest.approx(S.compute(c,rate,False)['scores']['overall'],rel=1e-12)
                factor=U.feature_display_factor(f)
                for raw,shown in zip(S.card(before),S.card(after)):
                    assert raw['name']==shown['name']
                assert U.displayed_value(f['overall'],7,c.od,factor)==pytest.approx(after['scores']['overall'])
                assert all(v>=0 and math.isfinite(v) for v in after['timeline'])
