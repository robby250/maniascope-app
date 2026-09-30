"""PP and displayed accuracy remain distinct, evidence-backed prediction targets."""
import math
from types import SimpleNamespace
import numpy as np
import pytest
import accuracy_targets as A
import personal_support as P
import recommend as R
import warmup as W


def model(mean):
    warm=W.Model({7:{'tau':6.,'amplitudes':[0.]*len(W.FAMILIES),'units':'log_rate'}})
    return SimpleNamespace(warmup=warm,predict=lambda *_:(mean,.1,.2),retry=0.)


def test_lazer_models_exclude_combined_or_unknown_hold_judgements_before_fitting(monkeypatch):
    rows=[dict(key=key,client=client,f={'ln':ln}) for key,client,ln in (
        ('stable-hold','stable',.5), ('unknown-hold',None,.5),
        ('native-hold','2026.921.0-lazer',.5), ('live-hold','lazer-live',.5),
        ('legacy|42','2026.921.0-lazer',.5), ('stable-rice','stable',0.))]
    seen=[]
    def capture(values,*_):
        seen.extend(r['key'] for r in values)
        raise StopIteration
    monkeypatch.setattr(W,'fit',capture)
    for target in ('lazer305','pp320'):
        seen.clear()
        with pytest.raises(StopIteration):R.Personal({'pop':{'accuracy_target':target}},rows)
        assert seen==['native-hold','live-hold','stable-rice']
    assert len(rows)==6  # source history is never edited


def test_direct_display_head_does_not_change_pp_mean_or_spread():
    pp=model(math.log(.08));display=model(math.log(.02))
    rec=SimpleNamespace(model=pp,accuracy_model=display,pub={},_shown=(math.log(.75),1.),
                        feats=SimpleNamespace(calc='fixture',md5_bid={}))
    f={'keys':7,'overall':6.,'sk':{'chordstream':1.},'length':120000}
    result=R.Predictor(rec).predict(f,'chart',None,1.,R.Session([],1000))
    assert result['mu']==pytest.approx(math.log(.08))
    assert result['display_mu']==pytest.approx(math.log(.02))
    assert result['acc_mid']==pytest.approx(.98)
    assert result['acc_lo']<.98<result['acc_hi']
    assert result['sdm']==.1 and result['sda']==.2
    # Local playlists can skip the unused PP calculation without changing any
    # displayed field, readiness or the full forecast recorded at play start.
    from unittest.mock import Mock
    pp.predict = Mock(wraps=pp.predict)
    predictor = R.Predictor(rec)
    session = R.Session([], 1000)
    preview = predictor.predict(f,'chart',None,1.,session,accuracy_only=True)
    assert not pp.predict.called and session._warmup_model is pp.warmup
    assert preview == {k:v for k,v in result.items() if k not in A.FIELDS}
    assert predictor.predict(f,'chart',None,1.,session) == result
    assert pp.predict.call_count == 1


def test_pp_playlist_uses_native_accuracy_with_pp_target_for_gain():
    rec=R.Recommender.__new__(R.Recommender)
    pp=model(math.log(.08));display=model(math.log(.02))
    for fitted in (pp,display):
        fitted.rate_response=lambda *_:3.
        fitted.rate_loss=lambda *args:3.*args[-1]
    f={'keys':7,'overall':6.,'sk':{'chordstream':1.},'length':120000,
       'stars':6.,'notes':1000,'hits':1000}
    rec.model=pp;rec.accuracy_model=display;rec._shown=(0.,1.)
    rec.pub={'feats':{42:{1.:f}},'maps':{42:{'md5':'chart','approved':1,'set':1,'file':'fixture'}}}
    rec.installed={};rec.newer_ranked=lambda:[];rec.ledger=R.Ledger({})
    rec.typical=lambda _k:math.log(.08)
    rec._candidates()
    session=R.Session([],1000);session.warm_flag=True
    choices,_=rec.score(session,keys=[7])
    assert len(choices)==1
    choice=choices[0]
    assert choice['acc_mid']==pytest.approx(.98)
    assert choice['mu']==pytest.approx(math.log(.08))
    assert choice['pp_mid']==pytest.approx(R.pp_at(6.,1000,.92))


def test_session_heads_compare_their_own_frozen_targets():
    info=dict(keys=7,skill='chordstream',length=120,mu=-2.8,base_mu=-3.,cold_penalty=.2,sd=.2,
              shown_mu=-2.8,display_mu=-3.1,display_base_mu=-3.3,display_cold_penalty=.2,
              display_sd=.2,display_sdm=.1,display_sda=.1)
    events=[dict(id=1,t=1000,kind='start',beatmap='md5:a',info=info),
            dict(id=2,t=1120,kind='finish',beatmap='md5:a',info=dict(start_id=1,y=-2.4,y_lazer=-3.1,key='a'))]
    s=R.Session(events,1120);view=s.for_accuracy(model(-3).warmup)
    assert s.correction(7,'chordstream')>0
    assert abs(view.correction(7,'chordstream'))<1e-12
    assert view.activation(7,'chordstream')==pytest.approx(2.)
    assert events[0]['info']['mu']==-2.8 and events[1]['info']['y']==-2.4


def test_legacy_completion_keeps_warmth_without_inventing_a_native_forecast():
    events=[dict(id=1,t=1000,kind='start',beatmap='md5:a',info=dict(keys=7,skill='chordstream',
                length=120,mu=-3.,base_mu=-3.,sd=.2)),
            dict(id=2,t=1120,kind='finish',beatmap='md5:a',info=dict(start_id=1,y=-2.,y_lazer=-2.3,key='a'))]
    view=R.Session(events,1120).for_accuracy(model(-3).warmup)
    assert view.activation(7,'chordstream')==pytest.approx(2.)
    assert view.correction(7,'chordstream')==0.


def test_warmup_override_is_shared_by_both_targets_even_without_a_log_fixture():
    s=R.Session([],1000);warm=model(-3).warmup
    assert not s.for_accuracy(warm).warm_flag
    s.warm_flag=True
    assert s.for_accuracy(warm).warm_flag


def test_completed_error_updates_its_chart_without_claiming_every_chart_is_wrong():
    m=model(-3.)
    m.sd_att={7:.2};m.sd_all=.2;m.rate_bandwidth={0:.15,7:.15}
    m.warmup.form_prior={'global_variance':.01,'skill_variance':.01,'noise_variance':.04}
    f={'keys':7,'overall':6.,'length':120000,'sk':{'chordstream':1.}}
    events=[dict(id=1,t=1000,kind='start',beatmap='md5:a',info=dict(keys=7,skill='chordstream',
                length=120,mu=-3.,base_mu=-3.,sd=.3,rate=1.,features=f)),
            dict(id=2,t=1120,kind='finish',beatmap='md5:a',info=dict(start_id=1,y=-2.,key='a'))]
    s=R.Session(events,1120,warmup_model=m.warmup)
    same=s.chart_correction('a',1.,f,m,.08)
    assert same>.4
    assert s.chart_correction('different',1.,f,m,.08)==0.
    assert s.chart_correction('a',1.5,f,m,.08)<same
    reset=R.Session(events+[dict(id=3,t=1121,kind='reset',beatmap=None,info={})],1122,warmup_model=m.warmup)
    assert reset.chart_correction('a',1.,f,m,.08)==0.


def test_one_session_of_retries_is_one_affinity_block():
    f={'length':120000}
    rows=[dict(chart='a',rate=1.,t=300.,src='realm',f=f,keys=7,ts=1000+i*130,key=str(i)) for i in range(3)]
    blocks=P.chart_blocks(rows,np.array([.2,.3,.4]),[(0,7)]*3,300.)
    assert len(blocks['a'])==1
    assert blocks['a'][0][1:]==pytest.approx((.3,1.,3))
    separate=P.chart_blocks(rows,np.array([.2,.3,.4]),[(i,7) for i in range(3)],300.)
    assert len(separate['a'])==3
    old=[(1.,.2,.1,1),(1.,.3,.1,1),(1.,.4,.1,1)]
    assert P.effective_count(old)==pytest.approx(3.)


def test_one_retried_family_cannot_validate_a_shape_effect():
    beta,report=P.fit_shape(np.ones((20,3)),np.ones(20),np.ones(20),['same']*20)
    assert not beta.any() and report['ridge'] is None


def test_personal_geometry_counts_width_and_overlap_once():
    import structural_residual as S
    f={'keys':7,'sk':{},'ln':0.,'stam':1.,
       'nps':{'chord':2.3,'overlap':.37},'endurance':{'load':.8}}
    values=[0.]*len(S.NAMES)
    values[S.NAMES.index('chord_width')]=2.3/7
    values[S.NAMES.index('overlap')]=.37
    f['execution']={'residual_vector':values}
    z=R.zvec(f,())
    assert np.count_nonzero(np.isclose(z,2.3/7))==1
    assert np.count_nonzero(np.isclose(z,.37))==1
    assert len(z)==7+len(R.PERSONAL_GEOMETRY)
    assert z[-1]==pytest.approx(R.zvec(dict(f,execution={}),())[-1])
