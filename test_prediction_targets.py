"""PP and displayed accuracy remain distinct, evidence-backed prediction targets."""
import copy
import hashlib
import json
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


def paired_personal():
    pop=dict(slope={0:3.},window={0:-3.},mb={},mbr={},tau2=.02,kappa=3.,curve={},curve_knee=4.)
    pub=dict(calc='fixture',pop=pop,feats={},maps={})
    pp=R.Personal(pub,[],now=300.)
    display=R.Personal(dict(pub,pop=dict(pop,window={0:-3.1})),[],now=300.)
    for i,m in enumerate((pp,display)):
        m.warmup.modes={7:dict(tau=4.+i,units='log_rate',amplitudes=[.02,.15,.08,.04,.09,.01,.03,.07])}
    rec=SimpleNamespace(model=pp,accuracy_model=display,pub=pub,_shown=(0.,1.),
                        feats=SimpleNamespace(calc='fixture',md5_bid={}))
    f=dict(keys=7,overall=5.5,od=8.,sk=dict(chordstream=.5,delay=.4,ln=.1),
           ln=.2,stam=.1,length=120000,stars=5.5,hits=1200,notes=1000)
    return rec,f


def completed_loss(info,pp_error,display_error,i=0):
    start=dict(keys=7,skill='chordstream',rate=1.,length=120.,base_mu=-3.,mu=-2.8,
               cold_penalty=.2,sd=.3,sdm=.1)
    start.update(info)
    return [dict(id=2*i+1,t=1000.+150*i,kind='start',beatmap='md5:chart',info=start),
            dict(id=2*i+2,t=1120.+150*i,kind='finish',beatmap='md5:chart',
                 info=dict(start_id=2*i+1,y=-3.+pp_error,y_lazer=-3.1+display_error,played_seconds=120.))]


def test_paired_scale_survives_real_prediction_start_and_next_forecast(tmp_path):
    rec,f=paired_personal()
    seed=dict(cold_scale=1.,display_cold_scale=1.,display_base_mu=-3.1,
              display_mu=-2.95,display_cold_penalty=.15,display_sd=.3,display_sdm=.1,display_sda=.2)
    events=sum((completed_loss(seed,.15,.18,i) for i in range(8)),[])
    predictor=R.Predictor(rec);now=10000.
    pp=predictor._predict(f,'a'*32,None,1.,R.Session(events,now))
    display=predictor.display._predict(f,'a'*32,None,1.,R.Session(events,now).for_accuracy(rec.accuracy_model.warmup))
    expected=predictor.predict(f,'a'*32,None,1.,R.Session(events,now))
    assert pp['cold_scale']!=display['cold_scale']

    db=R.recdata.connect(str(tmp_path/'rec.db'))
    for event in events:
        R.recdata.log_event(db,event['kind'],event['beatmap'],t=event['t'],**event['info'])
    sha='b'*64
    with db:
        db.execute('INSERT INTO installed(sha256,md5,keys,length,title,artist,version) VALUES (?,?,?,?,?,?,?)',
                   (sha,'a'*32,7,120000,'Fixture','Contract','7K'))
    worker=R.Worker(lambda _message:None);worker.db=db
    # Only the chart-filesystem boundary is stubbed; forecast and persistence
    # are their real owners, with no worker thread or GUI started.
    worker.rec=SimpleNamespace(pub=rec.pub,expectation=lambda *_a,**_k:dict(expected,md5='a'*32,
        keys=7,length=120.,skill=R.top_skill(f),f=f))
    worker.t_start(dict(t=now,sha=sha,path='/unused-fixture.osu',rate=1.,mods=[]))
    row=db.execute("SELECT * FROM events WHERE kind='start' ORDER BY id DESC LIMIT 1").fetchone()
    saved=dict(row,info=json.loads(row['info']));db.close()
    assert saved['info']['cold_scale']==pp['cold_scale']
    assert saved['info']['display_cold_scale']==display['cold_scale']
    assert expected['cold_scale']==pp['cold_scale']
    assert expected['display_cold_scale']==display['cold_scale']
    assert {k:expected[k] for k in A.FIELDS}=={k:pp[k] for k in A.FIELDS}
    assert {k:expected['display_'+k] for k in A.FIELDS}=={k:display[k] for k in A.FIELDS}
    preview=predictor.predict(f,'a'*32,None,1.,R.Session(events,now),accuracy_only=True)
    assert preview['cold_scale']==preview['display_cold_scale']==display['cold_scale']
    assert 'mu' not in preview and preview['acc_mid']==expected['acc_mid']
    terminal=dict(id=row['id']+1,t=now+120.,kind='finish',beatmap=saved['beatmap'],info=dict(
        start_id=row['id'],y=pp['base_mu']+pp['cold_penalty'],
        y_lazer=display['base_mu']+display['cold_penalty'],played_seconds=120.))
    completed=events+[saved,terminal]
    session=R.Session(completed,now+120.)
    assert session.cold_scale(7)==pytest.approx(pp['cold_scale'],abs=1e-12)
    assert session.for_accuracy(rec.accuracy_model.warmup).cold_scale(7)==pytest.approx(display['cold_scale'],abs=1e-12)
    wrong=copy.deepcopy(completed)
    wrong[-2]['info']['cold_scale']=display['cold_scale']  # old mixed-target defect
    later=now+120.+R.SESSION_GAP+1
    actual=predictor._predict(f,'c'*32,None,1.,R.Session(completed,later))
    mixed=predictor._predict(f,'c'*32,None,1.,R.Session(wrong,later))
    assert abs(actual['mu']-mixed['mu'])>1e-4


def test_legacy_paired_scale_is_display_metadata_without_losing_pp_evidence():
    rec,f=paired_personal()
    events=completed_loss(dict(cold_scale=2.,features=f,display_base_mu=-3.1,
        display_mu=-2.8,display_cold_penalty=.3,display_sd=.3,display_sdm=.1,display_sda=.2),.3,.3)
    original=copy.deepcopy(events)
    session=R.Session(events,1120,warmup_model=rec.model.warmup)
    assert session.cold_scale(7)==1.  # its old PP scale is unidentified
    view=session.for_accuracy(rec.accuracy_model.warmup)
    assert view.cold_scale(7)==pytest.approx((.15*.3+R.COLD_PRIOR)/(.15**2+R.COLD_PRIOR))
    no_scale=copy.deepcopy(events);no_scale[0]['info'].pop('cold_scale')
    control=R.Session(no_scale,1120,warmup_model=rec.model.warmup)
    assert session.correction(7,'chordstream',f)==control.correction(7,'chordstream',f)>0
    assert session.chart_correction('chart',1.,f,rec.model,.04)==control.chart_correction('chart',1.,f,rec.model,.04)>0
    assert session.activation(7,'chordstream')==control.activation(7,'chordstream')>0
    assert len(session.attempts)==1 and session.attempts[0]['ability_evidence']
    assert session.attempts[0]['y']==-2.7 and events==original


@pytest.mark.parametrize('paired',[False,True])
def test_legacy_without_scale_keeps_unscaled_observations(paired):
    info=dict(display_base_mu=-3.1,display_mu=-2.8,display_cold_penalty=.3) if paired else {}
    session=R.Session(completed_loss(info,.3,.3),1120)
    assert session.cold_scale(7)==pytest.approx((.2*.3+R.COLD_PRIOR)/(.2**2+R.COLD_PRIOR))
    view=session.for_accuracy(model(-3.).warmup)
    assert view.cold_scale(7)==pytest.approx(1.) and view.activation(7,'chordstream')>0


@pytest.mark.parametrize('target',['pp','display'])
def test_zero_scale_is_not_an_observed_unscaled_warm_curve(target):
    zeros=dict(cold_scale=0. if target=='pp' else 1.,cold_penalty=0. if target=='pp' else .2,
        display_cold_scale=0. if target=='display' else 1.,display_base_mu=-3.1,
        display_mu=-3.1,display_cold_penalty=0. if target=='display' else .2)
    events=sum((completed_loss(zeros,.3,.3,i) for i in range(5)),[])
    events+=completed_loss(dict(cold_scale=1.,display_cold_scale=1.,display_base_mu=-3.1,
        display_mu=-2.9,display_cold_penalty=.2),.2,.2,5)
    session=R.Session(events,1870)
    view=session.for_accuracy(model(-3.).warmup)
    owner=session if target=='pp' else view
    assert owner.cold_scale(7)==pytest.approx(1.)
    assert len(owner.history_attempts)==6 and owner.activation(7,'chordstream')>0
    assert all(a['cold_scale']==0. for a in owner.history_attempts[:5])
    if target=='display':
        # Explicit display zero identifies the modern paired contract; it
        # neither falls back to the PP scale nor discards PP observations.
        assert session.cold_scale(7)>1.


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
    assert math.log(1-result['acc_lo'])-result['display_mu']==pytest.approx(R.BAND_Z*result['display_sd'])
    assert result['sdm']==.1 and result['sda']==.2
    # Local playlists can skip the unused PP calculation without changing any
    # displayed field, readiness or the full forecast recorded at play start.
    from unittest.mock import Mock
    pp.predict = Mock(wraps=pp.predict)
    predictor = R.Predictor(rec)
    session = R.Session([], 1000)
    preview = predictor.predict(f,'chart',None,1.,session,accuracy_only=True)
    assert not pp.predict.called and session._warmup_model is pp.warmup
    numbers = lambda values: {k:v for k,v in values.items() if k!='accuracy_forecast'}
    assert numbers(preview) == dict({k:v for k,v in numbers(result).items() if k not in A.FIELDS},cold_scale=result['display_cold_scale'])
    assert numbers(predictor.predict(f,'chart',None,1.,session)) == numbers(result)
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


@pytest.mark.parametrize('randomized,eligible',[
    ((True,True),False), ((False,True),False), ((True,False),False),
    ((False,False),True), ((None,None),True)])
def test_retry_uses_only_normal_chart_attempt_pairs(randomized,eligible):
    rows=[];residual=[]
    for i in range(20):
        for j,rd in enumerate(randomized):
            row=dict(chart=f'chart-{i}',src='realm',ts=1000+i*1000+j*120)
            if rd is not None:row['rd']=rd
            rows.append(row);residual.append(.4 if j==0 else 0.)
    expected=.4*20/(20+50) if eligible else 0.
    assert R.Personal._retry(rows,np.array(residual))==pytest.approx(expected)


def test_intervening_rd_preserves_the_previous_normal_retry_anchor():
    rows=[];residual=[]
    for i in range(20):
        for j,(rd,value) in enumerate(((False,.6),(True,3.),(False,.2))):
            rows.append(dict(chart=f'chart-{i}',src='realm',ts=1000+i*1000+j*120,rd=rd))
            residual.append(value)
    normal=[i for i,r in enumerate(rows) if not r['rd']]
    expected=R.Personal._retry([rows[i] for i in normal],np.array(residual)[normal])
    assert expected==pytest.approx(.4*20/(20+50))
    assert R.Personal._retry(rows,np.array(residual))==expected


def test_rd_rows_retain_reduced_broad_ability_weight_in_personal():
    rec,f=paired_personal()
    rows=[dict(chart='same',src='realm',client='lazer-live',key=str(i),f=f,
               keys=7,rate=1.,t=300.,ts=1000+i*130,base=0.,y=-3. if i==0 else -1.,rd=i>0)
          for i in range(4)]
    original=copy.deepcopy(rows)
    personal=R.Personal(rec.pub,rows,now=300.)
    # The actual broad fit keeps one normal row and three shuffled rows. Its
    # existing per-chart leverage and RD weight both remain in force.
    rd_weight=3*.35/math.sqrt(3)
    assert personal.n_rows==4
    assert personal.glob==pytest.approx((-3.-rd_weight)/(1.+rd_weight))
    assert personal.glob>-3.
    assert personal.aff['same'][1]==1.
    assert personal.aff['same|RD'][1]==pytest.approx(3*.35)
    assert rows==original


def retry_recommender(md5):
    seed,f=paired_personal()
    rec=R.Recommender.__new__(R.Recommender)
    rec.__dict__.update(seed.__dict__)
    rec.pub=dict(seed.pub,feats={1:{1.:f},2:{1.:dict(f,overall=5.6)}},maps={
        1:dict(md5=md5,approved=1,set=1,file='Same chart'),
        2:dict(md5='c'*32,approved=1,set=2,file='Other chart')})
    rec.installed={};rec.rows=[];rec.ledger=R.Ledger({})
    for personal,retry in ((rec.model,.1),(rec.accuracy_model,.07)):
        personal.retry=retry
        personal.warmup.modes[7]['amplitudes']=[0.]*len(W.FAMILIES)
    rec._candidates()
    return rec,f


@pytest.mark.parametrize('randomized,prior_normal',[(False,False),(True,False),(True,True)])
def test_live_rd_effort_does_not_establish_normal_chart_retry(tmp_path,monkeypatch,randomized,prior_normal):
    from unittest.mock import Mock
    path=tmp_path/'synthetic.osu'
    raw=b'osu file format v14\n[General]\nMode:3\n[Difficulty]\nCircleSize:7\n'
    path.write_bytes(raw)
    sha=hashlib.sha256(raw).hexdigest();md5=hashlib.md5(raw).hexdigest()
    rec,f=retry_recommender(md5)
    db=R.recdata.connect(str(tmp_path/'rec.db'))
    worker=R.Worker(lambda _message:None);worker.db=db
    try:
        with db:
            db.execute('INSERT INTO installed(sha256,md5,keys,length,title,artist,version) VALUES (?,?,?,?,?,?,?)',
                       (sha,md5,7,120000,'Synthetic','RD','7K'))
        for i,rd in enumerate(([False] if prior_normal else [])+[randomized]):
            begin=1000.+150*i
            monkeypatch.setattr(R.time,'time',lambda:begin)
            # Only the chart-feature lookup boundary is supplied. Both the
            # pre-play forecast and all event/score persistence are real owners.
            expectation=Mock(side_effect=lambda *_a,**_k:dict(
                R.Predictor(rec).predict(f,md5,1,1.,R.Session([],begin)),
                f=f,md5=md5,keys=7,skill=R.top_skill(f),length=120.))
            worker.rec=SimpleNamespace(pub=rec.pub,expectation=expectation)
            mods=[{'acronym':'RD'}] if rd else []
            play=dict(t=begin,sha=sha,path=str(path),rate=1.,mods=mods)
            worker.t_start(play)
            assert expectation.call_count==int(not rd)
            worker.rec=None  # disable subsequent recommendation/UI callbacks
            monkeypatch.setattr(R.time,'time',lambda:begin+120.)
            score=dict(sha256=sha,rate=1.,mods_list=mods,played=R.time.strftime(
                '%Y-%m-%dT%H:%M:%SZ',R.time.gmtime(begin+120.)),score=1000000,hits={'geki':1000})
            worker.t_score(score,str(path),True,play)
        events=R.load_events(db)
    finally:
        db.close()
    original=copy.deepcopy(events)
    session=R.Session(events,begin+121.,warmup_model=rec.model.warmup)
    played=set(session.played);attempts=copy.deepcopy(session.attempts)
    warmth=session.activation(7);freshness=session.freshness('md5:'+md5,120.)
    forecast=R.Predictor(rec).predict(f,md5,1,1.,session)
    rows,_=rec.score(session,keys=7)
    actual=next(r for r in rows if r['bid']==1)
    assert actual['mu']==forecast['mu']
    assert actual['acc_mid']==pytest.approx(forecast['acc_mid'],rel=0,abs=1e-15)
    for personal in (rec.model,rec.accuracy_model):personal.retry=0.
    rec._score_cache.clear()
    control=R.Predictor(rec).predict(f,md5,1,1.,session)
    control_rows,_=rec.score(session,keys=7)
    eligible=not randomized or prior_normal
    assert forecast['mu']==pytest.approx(control['mu']-.1*eligible,rel=0,abs=1e-12)
    assert forecast['display_mu']==pytest.approx(control['display_mu']-.07*eligible,rel=0,abs=1e-12)
    if not eligible:
        assert forecast==control and rows==control_rows
    else:
        other=next(r for r in rows if r['bid']==2)
        assert other==next(r for r in control_rows if r['bid']==2)
    assert session.retry_played==({'md5:'+md5} if eligible else set())
    view=session.for_accuracy(rec.accuracy_model.warmup)
    assert view.retry_played==session.retry_played
    assert session.played==view.played==played=={'md5:'+md5}
    assert session.attempts==attempts and events==original
    assert session.activation(7)==warmth>0.
    assert session.freshness('md5:'+md5,120.)==freshness<1.
    assert session.attempts[-1]['prediction_valid']==(not randomized)
    assert all(a['meaningful'] and a['ability_evidence'] for a in session.attempts)


@pytest.mark.parametrize('metadata,eligible',[
    ({'mods_list':[{'acronym':'rd'}]},False), ({'mods':['RD']},False),
    ({'mods':[],'mods_list':[{'acronym':'RD'}]},False),
    ({'mods':['rd'],'mods_list':[]},False),
    ({},True), ({'mods':None,'mods_list':None},True),
    ({'mods':[None,42,{},True],'mods_list':None},True),
    ({'mods':42,'mods_list':{'acronym':'malformed'}},True),
    ({'mods':['NF'],'mods_list':[{'acronym':'HD'}]},True)])
def test_retry_familiarity_reads_both_mod_formats_without_losing_legacy_effort(metadata,eligible):
    events=completed_loss(metadata,.2,.2)
    events[-1]['info']['prediction_valid']=False  # not a general forecast-validity gate
    session=R.Session(events,1121.)
    assert session.retry_played==({'md5:chart'} if eligible else set())
    assert session.for_accuracy(model(-3.).warmup).retry_played==session.retry_played
    assert session.played=={'md5:chart'} and session.activation(7)>0.
    assert session.freshness('md5:chart',120.)<1.


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


def test_tree_stage_learns_a_step_and_runtime_matches_training():
    import structural_residual as S
    from calib.fit_structural_residual import boost
    rng=np.random.default_rng(1)
    X=rng.normal(size=(2000,len(S.NAMES)+1))
    y=np.where(X[:,3]>.5,.2,-.05)+rng.normal(0,.02,2000)
    for model in boost(X,y,np.ones(2000),200):
        pass
    pred=S.tree_predict(X,model)
    assert np.sqrt(np.mean((pred-y)**2))<.04
    x=X[7]
    p={'structural_trees':{'modes':{'7':model}}}
    assert S.tree_shift(7,math.exp(x[-1]),list(x[:-1]),p)==pytest.approx(S.tree_limit(pred[7],math.exp(x[-1])))
    assert S.tree_shift(4,math.exp(x[-1]),list(x[:-1]),p)==0.
    # Soft splits: the runtime matches the numpy path, and a sweep across the learned step is
    # continuous (hard splits jumped; a rate sweep then went down, 2026-10-02).
    soft=dict(model,scales=[.05]*X.shape[1])
    ps={'structural_trees':{'modes':{'7':soft}}}
    assert S.tree_shift(7,math.exp(x[-1]),list(x[:-1]),ps)==pytest.approx(S.tree_limit(S.tree_predict(X[7:8],soft)[0],math.exp(x[-1])))
    sweep=np.tile(X[7],(201,1));sweep[:,3]=np.linspace(0,1,201)
    for m,limit in ((soft,.01),(model,.1)):
        steps=np.abs(np.diff(S.tree_predict(sweep,m)))
        assert (steps.max()<limit)==(m is soft),(steps.max(),limit)


def test_tree_stage_reaches_displayed_stars():
    import score_units, structural_residual as S
    values=[.1]*len(S.NAMES)
    model={'intercept':.1,'trees':[]}
    base={'structural_residual':{'modes':{'7':{'low':[-9]*len(values),'high':[9]*len(values),'mean':[0]*len(values),
                                               'scale':[1]*len(values),'coeff':[0]*len(values)}}},
          'score_units':{'modes':{'7':{'display_coeff':[0]*len(values)}}}}
    with_trees=dict(base,structural_trees={'modes':{'7':model}})
    ex={'residual_vector':values}
    # final_rating multiplies by exp(.1); the display ratio must keep that, not cancel it
    original=5.*math.exp(.1)
    assert score_units.display_factor(7,5.,ex,original,with_trees)==pytest.approx(1.)
    assert score_units.display_factor(7,5.,ex,5.,base)==pytest.approx(1.)
