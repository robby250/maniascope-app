"""Prediction transfer follows measured rates, not a permanent map-wide bias."""
import math
from unittest.mock import patch

import numpy as np

import recommend as R


def model():
    m=R.Personal.__new__(R.Personal)
    m.pop={'tau2':.1,'kappa':3.}
    m.level={7:-3.};m.level_var={7:.01};m.sd_att={7:.2};m.sd_all=.2
    m.beta={};m.aff={'known':(-.6,10)};m.rate_aff={}
    m.rate_support={'known':(1.,1.)};m.rate_bandwidth={0:.18,7:.12}
    m.tau2_c=.12;m.kap_c=2.
    return m


def test_exact_rate_affinity_stays_intact_far_rate_weakens_and_uncertainty_grows():
    m=model()
    with patch.object(R,'base_of',return_value=(0.,0)):
        exact=m.predict({'keys':7},'known',rate=1.)
        fast=m.predict({'keys':7},'known',rate=1.5)
        slow=m.predict({'keys':7},'known',rate=.75)
        assert abs(exact[0]+3.6)<1e-12
        assert exact[0]<slow[0]<fast[0]<-3.
        assert exact[1]<slow[1]<fast[1]
        # A bad high-rate history must also fade at a much easier rate.
        m.aff['known']=(.6,10)
        assert m.predict({'keys':7},'known',rate=.75)[0]<m.predict({'keys':7},'known',rate=1.)[0]


def test_interpolation_inside_measured_rate_range_is_preserved():
    m=model();m.rate_support['known']=(.75,1.5)
    m.rate_aff['known']=[(.75,-.7,5),(1.5,-.3,5)]
    rate=math.sqrt(.75*1.5)
    with patch.object(R,'base_of',return_value=(0.,0)):
        assert abs(m.predict({'keys':7},'known',rate=rate)[0]+3.5)<1e-12


def test_bad_lower_rate_result_does_not_disappear_on_uprates():
    m=model();m.aff['known']=(.6,10)
    m.rate_support['known']=(.78,.78);m.rate_bandwidth[7]=.08
    with patch.object(R,'base_of',side_effect=lambda pop,f,b=None,rate=None,level=None:(2*math.log(rate/.78),0)):
        predictions=[m.predict({'keys':7},'known',rate=rate) for rate in (.78,.79,.9,1.)]
    assert all(b[0]>a[0] for a,b in zip(predictions,predictions[1:]))
    assert all(b[1]>a[1] for a,b in zip(predictions,predictions[1:]))
    assert m.chart_affinity('known',.9,7)[0]==.6
    assert 0<m.chart_affinity('known',.7,7)[0]<.6
    assert m.chart_affinity('unplayed',.9,7)==(0.,0)


def test_cold_capacity_integrates_across_measured_rate_anchors_without_a_cliff():
    import warmup
    m = model()
    m.warmup = warmup.Model(elasticities={'7': 1.2})
    m.pop.update(slope={0:2.,7:2.}, mbr={})
    m.rate_support['known'] = (.75, 1.5)
    m.rate_aff['known'] = [(.75, -.8, 5), (1.5, -.2, 5)]
    loss = -math.log(.9)
    values = []
    with patch.object(R, 'base_of', side_effect=lambda pop,f,b=None,r=None:(2*math.log(f['overall']),0)):
        for rate in np.linspace(1.1,1.6,101):
            f = {'keys':7,'overall':7*rate**1.2,'sk':{'chordstream':1.}}
            base = 2*math.log(f['overall']) + m.chart_affinity('known',rate,7)[0]
            cold = base + m.rate_loss(f,'known',None,rate,loss)
            equivalent = 2*math.log(f['overall']*math.exp(1.2*loss)) + m.chart_affinity('known',rate*math.exp(loss),7)[0]
            assert abs(cold-equivalent) < 1e-12
            values.append(cold)
    assert all(b>a for a,b in zip(values,values[1:]))


def test_rate_elasticity_has_no_winner_label_discontinuity():
    import warmup
    params = {'7':1., '7:rice':1., '7:chords':2.}
    a = {'keys':7,'sk':{'delay':.500001,'chordstream':.5}}
    b = {'keys':7,'sk':{'delay':.5,'chordstream':.500001}}
    assert abs(warmup.rate_elasticity(a,params)-warmup.rate_elasticity(b,params)) < .00001
