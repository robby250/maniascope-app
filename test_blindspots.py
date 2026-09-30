"""Small structural controls; no corpus scan or native game is needed."""
from copy import deepcopy
import math

import pytest

import hand_control as H
from calib.blindspots import family_groups, slice_summary
from calib import eval as E
from test_execution import chart


def pattern(masks, interval=70, keys=7, repeats=150):
    return chart([(1000+interval*i,1000+interval*i,c) for i in range(repeats)
                  for c in masks[i%len(masks)]],keys=keys)


def test_fixed_bounces_and_fixed_brackets_do_not_invent_retarget_control():
    bounce=pattern(((0,1,4,5),(0,1,4,5)))
    bracket=pattern(((0,1),(2,)))
    changing=pattern(((0,1),(2,),(0,),(1,2)))
    b=H.analyse(bounce)['summary'];s=H.analyse(bracket)['summary'];c=H.analyse(changing)['summary']
    assert b['selective_repeats']==0 and b['bracket_retargets']==0
    assert s['bracket_retargets']==0
    assert c['bracket_retargets']>0
    mixed=H.analyse(pattern(((0,1,4),(1,2,4))))['summary']
    assert mixed['selective_repeats']>0


def test_control_uses_played_time_and_not_metadata():
    a=pattern(((0,1),(2,),(0,),(1,2)))
    b=deepcopy(a);b.notes=[(t/1.3,e/1.3,c) for t,e,c in a.notes]
    assert H.analyse(a,1.3)['summary']==pytest.approx(H.analyse(b)['summary'],abs=1e-10)
    b=deepcopy(a);b.notes=[(t,e,a.keys-1-c) for t,e,c in a.notes];b.notes.sort()
    assert H.analyse(a)['summary']==pytest.approx(H.analyse(b)['summary'],abs=1e-10)
    b.title='Unseen local file';b.creator='Different mapper'
    assert H.analyse(a)['summary']==pytest.approx(H.analyse(b)['summary'],abs=1e-10)
    slower=H.analyse(a,.4)['summary']
    assert slower['bracket_retargets']<H.analyse(a)['summary']['bracket_retargets']
    for invalid in (0.,-1.,math.inf,math.nan):
        with pytest.raises(ValueError):H.analyse(a,invalid)


def test_linked_families_cannot_leak_back_into_training():
    labels={part:next(str(i) for i in range(100) if E.split(str(i))==part) for part in ('train','dev','test')}
    original={1:labels['train'],2:labels['dev'],3:labels['test'],4:'other song'}
    feats={1:{1.:{'nps':{'family':'same'}}},2:{1.:{'nps':{'family':'same'}}},
           3:{1.:{'nps':{'family':'same'}}},4:{1.:{'nps':{'family':'other'}}}}
    groups=family_groups(feats,original)
    assert groups[1]==groups[2]==groups[3]
    assert E.split(groups[1])=='test'
    assert groups[4]!=groups[1]


def test_slice_weights_families_not_number_of_duplicate_rates():
    rows=[{'group':'a','residual':1.} for _ in range(9)]+[{'group':'b','residual':-1.}]
    assert slice_summary(rows,[True]*10)['mean']==0


def test_calibration_confidence_never_falls_with_more_observations():
    from calib.fit_structural_residual import observation_weight
    counts=(0,5,10,50,100,500,2000,10000)
    weights=[observation_weight(n) for n in counts]
    assert weights[0]==0 and weights[2]==.5 and weights[-1]<1
    assert all(a<b for a,b in zip(weights,weights[1:]))
    boosted=[w*math.sqrt(min(500,n)) for n,w in zip(counts,weights)]
    assert all(a<b for a,b in zip(boosted,boosted[1:]))


def test_control_work_is_local_and_zero_gain_is_neutral():
    import control_demand as C
    import skill_calc as S
    a=pattern(((0,1),(2,),(0,),(1,2)),repeats=240)
    runs=[S._demand(a,1.,left,None) for left in (True,False)]
    out=C.factors(a,1.,runs,[(0.,0.,0.),(0.,.25,0.),(0.,1.,0.)])
    assert out[0]['factor']==1.
    assert out[2]['factor']>out[1]['factor']>1.
    raw=[sum(r[1][i] for r in runs)/2 for i in range(len(runs[0][1]))]
    assert out[0]['timeline']==pytest.approx(raw,abs=1e-10)
    # A fixed bracket is still physically difficult, but is not charged again
    # for changing its two finger groups when those groups never change.
    fixed=pattern(((0,1),(2,)),repeats=240)
    fr=[S._demand(fixed,1.,left,None) for left in (True,False)]
    assert C.factors(fixed,1.,fr,[(0.,1.,0.)])[0]['factor']==1.
    # A distant first phrase receives no local control increment just because
    # the second phrase has retargets. No full-map label multiplier is applied.
    c=chart(fixed.notes+[(t+30000,e+30000,col) for t,e,col in a.notes],keys=7)
    cr=[S._demand(c,1.,left,None) for left in (True,False)]
    before,after=C.factors(c,1.,cr,[(0.,0.,0.),(0.,1.,0.)])
    assert before['timeline'][:30]==pytest.approx(after['timeline'][:30],abs=1e-10)
    assert sum(after['timeline'][60:])>sum(before['timeline'][60:])


def test_candidate_demand_cannot_become_negative():
    import control_demand as C
    import skill_calc as S
    a=pattern(((0,1),(1,2)))
    runs=[S._demand(a,1.,left,None) for left in (True,False)]
    extra=C.prepare(a,1.,runs)
    for gains in ((-1.,0.,0.),(math.nan,0.,0.),(1.,0.)):
        with pytest.raises(ValueError):C.aggregate(runs,extra,gains)

