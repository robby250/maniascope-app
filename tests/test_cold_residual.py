"""Focused checks for chart-only corrections and completed-play readiness."""
import copy
import math
from unittest.mock import patch

import numpy as np
import pytest
import difficulty_model as D
import nps
import recommend as R
import structural_residual as S
import warmup as W
from test_execution import chart
from test_redesign import events


def cold_model():
    return W.Model({7:{'tau':6.,'amplitudes':[.5,.6,.9,.8,.3,.4,.4,.4]}})


def test_quits_do_not_change_ability_warmth_or_downrate_constraints():
    model=cold_model()
    empty=R.Session([],1200,warmup_model=model)
    ev=events([0],kind='abort',seconds=100,played_rate=.75,offered_rate=1.19)
    ev[-1]['info']['partial_stats']={'geki':1,'0':200}
    quit=R.Session(ev,1200,warmup_model=model)
    assert quit.correction(7,'chordjack')==empty.correction(7,'chordjack')==0
    assert quit.activation(7,'chordjack')==0 and quit.rate_penalty('m0',1.19)==0
    assert quit.warmup_penalty(7,'chordjack')==empty.warmup_penalty(7,'chordjack')


def test_completed_history_seeds_restart_once_and_transfers_only_partly():
    model=cold_model()
    model.history=[{'key':'a','t':1000,'end':1120,'keys':7,'skill':'chordjack',
                    'played_seconds':120,'kind':'finish','ability_evidence':True}]
    s=R.Session([],1120,warmup_model=model)
    assert s.activation(7,'chordjack')==2.
    assert 0<s.activation(7,'ln')<s.activation(7,'jack')<2
    assert s.activation(4,'chordjack')==0
    assert W.activation(model.history*2,1120,7,'chordjack')==2
    assert R.Session([],5000,warmup_model=model).activation(7,'chordjack')==0
    reset=R.Session([{'id':1,'t':1121,'kind':'reset','beatmap':None,'info':{}}],1122,warmup_model=model)
    assert reset.activation(7,'chordjack')==0


def test_cold_and_history_exposure_match_without_future_knowledge():
    f={'keys':7,'length':120000,'overall':6.,'sk':{'chordjack':1},'ln':0.,'stam':1.,'od':8.}
    rows=[dict(key=str(i),chart=str(i),ts=1120+150*i,t=300.,rate=1.,f=f,keys=7,src='realm',y=-3.,base=0.) for i in range(4)]
    history=[]
    steps=W.timeline(rows)
    assert len(steps)==4
    for r,_session,minutes,d,seconds,_family in steps:
        active=W.activation(history,r['ts']-seconds,7,'chordjack')
        assert active==pytest.approx(minutes,abs=1e-12)
        exposure=math.exp(-minutes/6.)*(-math.expm1(-d/6.))*6./d     # the fit's mean exposure over the map
        assert cold_model().penalty(7,'chordjack',active,120)==pytest.approx(.9*exposure)
        history.append({'key':r['key'],'t':r['ts']-120,'end':r['ts'],'keys':7,'skill':'chordjack','played_seconds':120,'kind':'finish'})
    assert not W.timeline([dict(rows[0],src='public')])
    assert not W.timeline([dict(rows[0],completed=False)])


def test_normal_completed_cold_loss_is_not_double_counted_as_bad_form():
    ev=events([0])
    ev[0]['info'].update(base_mu=-3.,mu=-2.5,cold_penalty=.5)
    ev[1]['info']['y']=-2.5
    s=R.Session(ev,1101,warmup_model=cold_model())
    assert abs(s.correction(7,'chordjack'))<1e-12
    assert s.warmup_penalty(7,'chordjack')>0


def test_density_incentive_waits_for_readiness_not_higher_accuracy_target():
    f={'keys':7,'sk':{'chordjack':1.},'length':120000,'nps':{'nps':20.,'play_span':120.}}
    high=dict(f,nps=dict(f['nps'],nps=60.))
    e={'acc_mid':.94,'sd_model':.15}
    s=R.Session([],1000,warmup_model=cold_model())
    assert nps.central_target(s)==nps.DEFAULT_TARGET
    assert nps.practice_value(f,e,1.19,.94,s)==nps.practice_value(high,e,1.19,.94,s)
    assert nps.practice_value(f,e,1.19,.94,s)>nps.practice_value(high,dict(e,sd_model=.5),1.19,.94,s)
    s.warm_flag=True
    assert nps.practice_value(high,e,1.19,.94,s)>nps.practice_value(f,e,1.19,.94,s)


def test_chart_metadata_does_not_enter_residual_and_unfitted_modes_are_unchanged():
    f={'keys':7,'overall':7.,'sk':{'chordstream':.95,'bracket':.9},'ln':0.,'stam':.8,'length':120000,'notes':3000,'nps':{'nps':25.}}
    changed=dict(f,title='other',beatmap_id=12345,creator='other',personal_accuracy=.99)
    assert S.vector(f)==S.vector(changed)
    assert S.correction(7,7.,S.vector(f),{})==7.
    assert D.final_rating(7,7.,[0.]*len(D.FEATURES),{'residual_vector':S.vector(f)})>0


def test_fresh_calculation_and_feature_transform_share_structural_inputs():
    import skill_calc as C
    from calib.publish_structural import transform
    c=chart([(1000+60*i,1000+60*i,col) for i in range(500)
             for col in ((0,2,4) if i%2 else (1,3,6))],keys=7)
    rate=.98;r=C.compute(c,rate)
    import recdata
    with patch.object(C,'parse_osu',return_value=c), patch('feedback._od',return_value=c.od):
        f=recdata.chart_feats('/unavailable-structural-control.osu',(rate,),analyses={rate:r})[rate]
    t=transform(f,rate)
    # Feature caches carry prediction units; the shown card (r['scores']) has its own display scale.
    assert t['preunit_overall']==pytest.approx(r['preunit_overall'],abs=1e-9)
    assert t['overall']==pytest.approx(f['overall'],abs=1e-9)
    assert t['execution']['residual_vector']==pytest.approx(r['execution']['residual_vector'],abs=1e-10)
    assert t['sk']==f['sk'] and t['stam']==f['stam']
    # Inference does not modify public scores or return a cached map-ID answer.
    c.title='renamed';c.creator='different'
    assert C.compute(c,rate)['scores']['overall']==r['scores']['overall']


def test_tracked_openings_that_play_warm_shrink_the_cold_curve():
    def history(cold_result):
        ev=events([0]*12)
        for i in range(6):          # six openings forecast .15 worse than warm
            ev[2*i]['info'].update(cold_penalty=.15)
            ev[2*i+1]['info']['y']=-3.+cold_result
        return R.Session(ev,99999,warmup_model=cold_model())
    assert history(.15).cold_scale(7)==pytest.approx(1.,abs=.01)     # they were as cold as forecast
    warm=history(0.)                                                  # they played like warm maps
    assert warm.cold_scale(7)<.5
    assert warm.warmup_penalty(7,'chordjack')==pytest.approx(
        warm._historical_cold(7,'chordjack')*warm.cold_scale(7))
    assert R.Session([],1200,warmup_model=cold_model()).cold_scale(7)==1.


def test_native_family_weights_and_dominant_match_the_python_definition():
    def weights(f):
        sk=f.get('sk',{});values=[]
        for name,members in W.MEMBERS.items():
            if name=='technical':continue
            w=max((max(0.,sk.get(x,0.)) for x in members),default=0.)**2
            if w:values.append((name,w))
        total=sum(w for _n,w in values)
        return tuple((n,w/total) for n,w in values) if total else ((dominant(f),1.),)
    def dominant(f):
        sk=f.get('sk',{});names=[k for k in sk if k not in W.DESCRIPTIVE]
        return W.family(max(names,key=lambda k:sk[k])) if names else 'rice'
    fs=[dict(sk=dict(chordstream=.9,delay=.5,technical=2.)),dict(sk=dict(jack=.95,minijack=.9,ln=-.2)),
        dict(sk=dict(stamina=.8,technical=.5,ln=0.)),dict(sk={}),{},dict(sk=dict(sv=.3,release=.3,stream=.3)),
        dict(sk=dict(stamina=.9,trill1h=.2,jumptrill=.7,vibro=.6,shield=1e-200))]
    assert [W.family_weights(f) for f in fs]==[weights(f) for f in fs]
    assert [W.dominant(f) for f in fs]==[dominant(f) for f in fs]
    assert W._families().weights_many(fs)==[weights(f) for f in fs]



def test_native_warmup_ridge_matches_the_numpy_alternation():
    import native_backend
    import recdata
    rng=np.random.default_rng(3);n,p,g=400,9,4
    X=rng.normal(size=(n,p));X[:,-g:]=np.abs(X[:,-g:]);y=rng.normal(size=n);w=rng.uniform(.2,1.,n)
    A=X.T@(w[:,None]*X)+np.eye(p)*2.
    lower=np.full(p,-np.inf);upper=np.full(p,np.inf);lower[-g:],upper[-g:]=0.,.4
    prior=np.zeros(p);prior[-g:]=.3
    groups=[(rng.integers(0,k,n),k,r) for k,r in ((12,4.),(30,5.),(50,8.))]
    effects=[np.zeros(k) for _i,k,_r in groups]
    for _ in range(16):         # the Python loop warmup._fit ran before
        rhs=X.T@(w*(y-sum(e[i] for e,(i,_k,_r) in zip(effects,groups))))+prior
        coeff=recdata.box_cd(A,rhs,np.linalg.solve(A,rhs),lower,upper,40,1e-7)
        for j,(i,k,r) in enumerate(groups):
            residual=y-X@coeff-sum(effects[h][groups[h][0]] for h in range(len(groups)) if h!=j)
            effects[j]=np.bincount(i,w*residual,k)/(np.bincount(i,w,k)+r)
    native,native_effects=native_backend.load().warmup_ridge(
        X.tobytes(),y.tobytes(),w.tobytes(),A.tobytes(),lower.tolist(),upper.tolist(),prior.tolist(),
        [(i.astype(np.int64).tobytes(),k,r) for i,k,r in groups],16)
    assert (coeff[-g:]==0).any() and (coeff[-g:]>0).any()      # both sides of the bound are exercised
    assert np.allclose(native,coeff,rtol=0,atol=1e-12)
    for a,b in zip(native_effects,effects):
        assert np.allclose(a,b,rtol=0,atol=1e-12)
