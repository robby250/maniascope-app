"""Physical execution controls; no titles or hand-picked rating targets."""
import math
from unittest.mock import patch

import pytest
import difficulty_model as D
import skill_calc as S
from test_execution import chart


def rolls(pattern, gap=20, n=1200, keys=4, od=8):
    return chart([(1000+i*gap,1000+i*gap,pattern[i % len(pattern)])
                  for i in range(n)],keys=keys,od=od)


def repeated(pattern, gap=80, n=400, keys=4):
    return chart([(1000+i*gap,1000+i*gap,c) for i in range(n)
                  for c in pattern[i % len(pattern)]], keys=keys)


def mean(series):
    return sum(series)/max(1,len(series))


@pytest.mark.parametrize('pattern',[(0,1,2,3),(0,1,3,2),(1,0,3,2),(0,3,1,2),
                                  (0,1,2,3,0,1,3,2,1,0,3,2,0,3,1,2)])
def test_fast_variable_and_interleaved_rolls_are_jumptrill_execution(pattern):
    c=rolls(pattern)
    info=S.N.gestures_analyse(c)
    assert mean(info['series']['rolled']) > .7
    r=S.compute(c)
    assert r['scores']['jumptrill'] > r['scores']['stream']


@pytest.mark.parametrize('keys',range(4,11))
def test_hand_grouping_generalizes_and_respects_center_assignment(keys):
    half=keys//2
    # Free outer fingers still roll while a spare center finger is held.
    cols=tuple(range(half)) + tuple(range((keys+1)//2,keys))
    c=rolls(cols,gap=12,keys=keys)
    mirror=chart([(t,e,keys-1-k) for t,e,k in c.notes],keys=keys,od=c.od)
    a,b=S.N.gestures_analyse(c),S.N.gestures_analyse(mirror)
    assert mean(a['series']['rolled']) > .7
    for key in S.N.GESTURE_FEATURES:
        assert abs(a['features'][key]-b['features'][key]) < 1e-8


def test_slow_rice_and_timing_infeasible_groups_do_not_become_jumptrills():
    c=rolls((0,1,2,3),gap=70)
    assert mean(S.N.gestures_analyse(c)['series']['rolled']) == 0
    assert S.compute(c)['scores']['jumptrill'] == 0
    # The same note order can fit at OD0 but not OD10. The feasible interval is
    # for Great, not a guarantee of Perfect/SS with all fingers simultaneous.
    loose=rolls((0,3,1,2),gap=30,od=0)
    strict=rolls((0,3,1,2),gap=30,od=10)
    assert mean(S.N.gestures_analyse(loose)['series']['rolled']) > mean(S.N.gestures_analyse(strict)['series']['rolled'])
    assert mean(S.N.gestures_analyse(strict)['series']['rolled']) == 0


def test_rate_is_applied_once_and_offsets_do_not_change_gestures():
    c=rolls((0,1,3,2),gap=30)
    retimed=chart([(t/1.5,e/1.5,k) for t,e,k in c.notes],keys=4,od=c.od)
    shifted=chart([(t+91234,e+91234,k) for t,e,k in c.notes],keys=4,od=c.od)
    a,b,z=S.N.gestures_analyse(c,1.5),S.N.gestures_analyse(retimed),S.N.gestures_analyse(shifted,1.5)
    for key in S.N.GESTURE_FEATURES:
        assert abs(a['features'][key]-b['features'][key]) < 1e-8
        assert abs(a['features'][key]-z['features'][key]) < 1e-8


def test_reversals_and_anchors_cost_control_instead_of_erasing_all_grouping():
    plain=S.N.gestures_analyse(rolls((0,1,2,3)))
    turns=S.N.gestures_analyse(rolls((0,1,2,3,1,0,3,2)))
    anchored=S.N.gestures_analyse(rolls((0,1,2,3,3,2,1,0)))
    assert turns['features']['roll_control'] > plain['features']['roll_control']+.1
    assert turns['features']['roll_anchor'] > plain['features']['roll_anchor']+.05
    assert mean(anchored['series']['rolled']) < mean(plain['series']['rolled'])
    persistent=S.N.gestures_analyse(rolls((0,1,0,2,0,3)))
    assert mean(persistent['series']['rolled']) < .1


def test_literal_jumptrill_split_trill_and_vibro_remain_different():
    jt=S.compute(repeated(((0,1),(2,3)),gap=60))
    split=S.compute(repeated(((0,2),(1,3)),gap=60))
    assert jt['scores']['jumptrill'] > .9*jt['scores']['overall']
    assert jt['scores']['vibro'] == 0
    assert split['scores']['splittrill'] > .9*split['scores']['overall']
    assert split['scores']['jumptrill'] == 0
    assert split['scores']['vibro'] == 0


def test_changing_chords_can_still_be_vibro_but_stability_is_not_control():
    stable=S.N.gestures_analyse(repeated(((0,1,2,3),)))
    changing=S.N.gestures_analyse(repeated(((0,1,2),(0,1,3),(0,1,2,3),(1,2,3))))
    assert stable['features']['vibro_easy'] > 2*changing['features']['vibro_easy']
    assert changing['features']['vibro_control'] > stable['features']['vibro_control']+.2
    assert mean(changing['series']['vibro']) > .05
    assert mean(S.N.gestures_analyse(repeated(((0,1,2,3),),gap=170))['series']['vibro']) == 0


def test_free_fingers_can_roll_under_a_hold_without_collapsing_the_hold():
    c=rolls((0,1,2,6,5,4),gap=18,keys=7)
    c.notes=sorted(c.notes+[(1000,22600,3)])
    r=S.compute(c)
    assert r['scores']['ln'] > 0
    assert r['scores']['jumptrill'] > r['scores']['delay']
    all_holds=chart([(t,t+15,k) for t,e,k in c.notes if k != 3],keys=7)
    assert mean(S.N.gestures_analyse(all_holds)['series']['rolled']) == 0


def test_sustain_is_not_reset_by_one_anchor_and_recovers_on_real_breaks():
    c=rolls((0,1,2,3),n=2400)
    interrupted=chart([(t+(10000 if i>=1200 else 0),e+(10000 if i>=1200 else 0),k)
                       for i,(t,e,k) in enumerate(c.notes)],keys=4,od=c.od)
    assert S.N.gestures_analyse(c)['features']['jumptrill_sustain'] > S.N.gestures_analyse(interrupted)['features']['jumptrill_sustain']
    assert S.N.gestures_analyse(repeated(((0,1,2,3),),n=800))['features']['vibro_sustain'] > \
           S.N.gestures_analyse(repeated(((0,1,2,3),),n=40))['features']['vibro_sustain']


def test_vibro_support_does_not_jump_when_a_slower_gap_crosses_the_speed_gate():
    c=chart([(1000+b*434+i*80,1000+b*434+i*80,k)
             for b in range(60) for i in range(5) for k in range(4)],keys=4)
    a,b=S.N.gestures_analyse(c,1.0259)['features'],S.N.gestures_analyse(c,1.0261)['features']
    for key in ('vibro_easy','vibro_control','vibro_sustain'):
        assert abs(a[key]-b[key]) < .01


def test_unfitted_execution_features_cannot_silently_discount_a_chart():
    with patch.object(D,'parameters',return_value={}):
        assert D.final_rating(4,9.,[0.]*len(D.FEATURES),dict.fromkeys(S.N.GESTURE_FEATURES,1.)) == 9.


def test_long_stable_repetition_is_distinct_from_short_fixed_chords_inside_cj():
    long=S.N.gestures_analyse(repeated(((0,1,2,3),),gap=80,n=500))['features']
    short=S.N.gestures_analyse(chart([(1000+b*2000+i*80,1000+b*2000+i*80,k)
                          for b in range(50) for i in range(5) for k in range(4)],keys=4))['features']
    changing=S.N.gestures_analyse(repeated(((0,1,2),(0,1,3),(0,1,2,3),(1,2,3))))['features']
    assert long['vibro_locked'] > .5
    assert short['vibro_locked'] < .01
    assert changing['vibro_locked'] == 0
    assert changing['vibro_control'] > .2


def test_gesture_calibration_is_local_to_evidence_and_inverts_the_score_link():
    parameters={'gestures':{'features':['vibro_easy','vibro_control'],
                           'modes':{'4':{'coeff':[-.4,.3],'slope':2.,'curve':.6}}}}
    with patch.object(D,'parameters',return_value=parameters):
        assert D.gesture_rating(4,9.,9.,{}) == 9.
        assert D.gesture_rating(7,9.,9.,{'vibro_easy':1.}) == 9.
        lower=D.gesture_rating(4,9.,9.,{'vibro_easy':.5})
        higher=D.gesture_rating(4,9.,9.,{'vibro_control':.5})
        link=lambda value:2.*math.log(value)+.6*math.log(value)**2
        assert abs(link(lower)-link(9.)+.2)<1e-9
        assert abs(link(higher)-link(9.)-.15)<1e-9
        assert 0 < lower < 9. < higher
        assert D.gesture_rating(4,0.,0.,{'vibro_easy':1.}) == 0.


def test_degenerate_input_and_invalid_rates_are_explicit():
    empty=S.N.gestures_analyse(chart([],keys=4))
    assert not any(empty['features'].values())
    assert S.N.gestures_analyse(chart([(1000,1000,0),(1000,1000,1)],keys=4))['features']['vibro_easy'] == 0
    for rate in (0,-1,float('inf'),float('nan')):
        with pytest.raises(ValueError):
            S.N.gestures_analyse(rolls((0,1,2,3)),rate)
