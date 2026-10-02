"""PP breadth, session-sensitive ambition and persistent freshness controls."""
import math
import random
from types import SimpleNamespace

import pp_playlist as P
import recommend as R
import recdata
from test_recommend import _FakeRec, _cand, _attempts


def session(z, keys=7):
    ev=_attempts(1000,z,bid=200,keys=keys,skill='delay',spacing=220)
    return R.Session(ev,1000+(len(z)-1)*220+190)


def test_pp_pool_has_no_top25_or_gain_relative_floor():
    cands=[dict(_cand(i,.13+(i%10)*.01),acc_mid=.985,pp_mid=105.) for i in range(1,151)]
    cands.append(dict(_cand(1000,40.),acc_mid=.985,pp_mid=180.))
    cands.append(dict(_cand(2000,.05),acc_mid=.985,pp_mid=105.))     # under the .1pp floor
    r=_FakeRec(cands);s=R.Session([],1000)
    _,shown,_,note=r.pool(s,rng=random.Random(4))
    assert len(shown)==20 and len(r.pp_choices)==151
    assert '151 eligible' in note
    rng=random.Random(8)
    picks={r.pool(s,rng=rng)[2]['bid'] for _ in range(150)}
    assert len(picks)>60
    assert r.pool(s)[1]==r.pool(R.Session([],1000))[1]   # repaint/restart is not an exposure


def test_pp_pool_reuses_only_identical_candidates_and_session():
    from unittest.mock import patch
    candidates = [dict(_cand(i, .2), acc_mid=.985, pp_mid=105.) for i in range(5)]
    rec = _FakeRec(candidates)
    rec.score = lambda session, now=None, keys=None: (candidates, session)
    s = R.Session([], 1000)
    with patch.object(P, 'prepare', wraps=P.prepare) as prepare, patch.object(P, 'preview', wraps=P.preview) as preview:
        first = rec.pool(s, rng=random.Random(28))
        assert rec.pool(s, rng=random.Random(28)) == first
        assert prepare.call_count == 1
        assert rec.pool(s, n_show=0, rng=random.Random(28))[2] == first[2]
        assert prepare.call_count == 1 and preview.call_count == 2
        offered = dict(id=1, t=1000, kind='offer', beatmap=R.event_key(first[2]), info={'mode':'pp'})
        after = R.Session([offered], 1001)
        assert rec.pool(after)[2]['bid'] != first[2]['bid']
        assert prepare.call_count == 2
        rec.pool(R.Session([offered], 1300))
        assert prepare.call_count == 3
        candidates = [dict(candidates[0], acc_mid=.99)]
        rec.pool(s)
        assert prepare.call_count == 4
        s.warm_flag = True
        rec.pool(s)
        assert prepare.call_count == 5
        rec.pool(s, keys=[4])
        assert prepare.call_count == 6 and rec.pp_choices == []


def test_bad_form_safe_targets_strong_form_higher_pp_and_lower_accuracy():
    r=_FakeRec([])
    r.pp_reference={7:{'comfort':.98,'peak_acc':.92,'safe_pp':400.,'peak_pp':850.}}
    easy=dict(_cand(100,.7),acc_mid=.981,pp_mid=420.,best=390.,skill='delay')
    hard=dict(_cand(101,18.),acc_mid=.923,pp_mid=900.,best=760.,skill='delay')
    cold,_=P.prepare(r,[easy,hard],R.Session([],1000))
    assert [c['bid'] for c in cold]==[100]
    poor,_=P.prepare(r,[easy,hard],session([-1.,-.8,-1.2,-.6]))
    strong,_=P.prepare(r,[easy,hard],session([.9,1.,1.2,1.1]))
    ratio=lambda rows:next(c['weight'] for c in rows if c['bid']==101)/next(c['weight'] for c in rows if c['bid']==100)
    assert ratio(strong)>3*ratio(poor)


def test_7k_never_activates_10k_but_comfortable_probe_is_allowed():
    s=session([1.]*12)
    assert s.activation(7)>0 and s.activation(10)==0
    assert s.warmup_penalty(10,'delay')>.09
    r=_FakeRec([])
    hard=dict(_cand(100,50.,keys=10),acc_mid=.928,skill='delay')
    probe=dict(_cand(101,.2,keys=10),acc_mid=.985,skill='delay')
    rows,_=P.prepare(r,[hard,probe],s)
    assert [c['bid'] for c in rows]==[101] and rows[0]['purpose']=='probe'


def test_explicit_interruption_is_not_negative_ability_or_a_downrate_constraint():
    ev=[{'id':1,'t':1000,'kind':'start','beatmap':'md5:a','info':
         {'keys':7,'skill':'chordjack','length':200.,'rate':.9,'offered_rate':1.1}},
        {'id':2,'t':1110,'kind':'abort','beatmap':'md5:a','info':{'start_id':1,'played_seconds':110.}},
        {'id':3,'t':1120,'kind':'attempt_note','beatmap':'md5:a','info':{'start_id':1,'reason':'interrupted'}}]
    s=R.Session(ev,1130)
    assert s.correction(7,'chordjack')==0 and s.rate_penalty('a',1.1)==0
    assert s.activation(7)==0
    ev[-1]['info']['reason']='failed'
    failure=R.Session(ev,1130)
    # Notes remain useful human feedback, not synthetic completed-score evidence.
    assert failure.correction(7,'chordjack')==0 and failure.rate_penalty('a',1.1)==0
    assert failure.attempts[0].get('y') is None              # never invent a full score


def test_attempt_annotations_are_append_only_and_idempotent(tmp_path):
    db=recdata.connect(str(tmp_path/'rec.db'))
    start=recdata.log_event(db,'start','md5:a',t=1000,sha='a'*64,rate=1.24)
    recdata.log_event(db,'abort','md5:a',t=1040,start_id=start,played_seconds=37.)
    original=[tuple(r) for r in db.execute('SELECT * FROM events ORDER BY id')]
    note=recdata.annotate_attempt(db,start,'failed','Player thinks this failed.',.5)
    assert recdata.annotate_attempt(db,start,'failed','Player thinks this failed.',.5)==note
    assert original==[tuple(r) for r in db.execute('SELECT * FROM events WHERE id<? ORDER BY id',(note,))]
    assert db.execute("SELECT COUNT(*) FROM events WHERE kind='attempt_note'").fetchone()[0]==1
    db.close()


def test_newly_ranked_exact_revision_is_not_hidden_by_qualified_snapshot_status():
    rec=R.Recommender.__new__(R.Recommender)
    f={'stars':6.,'notes':1000,'hits':1000,'keys':7,'sk':{'stream':1.}}
    rec.pub={'feats':{42:{1.:f}},'maps':{42:{'md5':'a'*32,'approved':3,'set':1,'file':'newly ranked'}}}
    rec.installed={'a'*32:{'status':1}}
    rec.model=SimpleNamespace(predict=lambda *a:(-3.,.1,.2))
    rec.newer_ranked=lambda:[]
    rec._candidates()
    assert len(rec.cands)==1 and rec.cands[0][4]['approved']==1
    assert rec.A['ranked'].tolist()==[True]
    rec.installed['a'*32]['status']=-2
    rec._candidates()
    assert rec.cands==[]                                    # don't ignore newer unranked status either


def test_pp_cycles_eligible_maps_even_after_offer_penalty_recovers():
    candidates=[dict(_cand(i,.2 if i else 40.),acc_mid=.985,pp_mid=105.) for i in range(5)]
    rec=_FakeRec(candidates);events=[];picks=[]
    for i in range(15):
        now=1000+i*240   # old exposure discount permits repeats after a few minutes
        s=R.Session(events,now)
        rows,_=P.prepare(rec,candidates,s)
        pick=random.Random(i).choices(rows,weights=[c['weight'] for c in rows])[0]
        picks.append(pick['bid'])
        events.append(dict(id=i+1,t=now,kind='offer',beatmap=R.event_key(pick),info={'mode':'pp'}))
    # soft shuffle bag: every chart comes up, none twice within three picks, no forced full cycle
    assert set(picks)==set(range(5)) and all(len(set(picks[i:i+3]))==3 for i in range(13))


def test_pp_prefers_the_unseen_map_and_never_repeats_the_last():
    rec=_FakeRec([])
    candidates=[dict(_cand(i,1.),acc_mid=.985,pp_mid=105.) for i in range(4)]
    events=[dict(id=i+1,t=1000+i,kind='offer',beatmap=R.event_key(c),info={'mode':'pp'})
            for i,c in enumerate(candidates[:3])]
    rows,relaxed=P.prepare(rec,candidates,R.Session(events,1010))
    best=max(rows,key=lambda c:c['weight'])
    assert best['bid']==3 and 2 not in [c['bid'] for c in rows] and not relaxed


def test_positive_pp_opportunity_below_top100_is_not_discarded():
    rec=_FakeRec([]);rec.ledger=R.Ledger({i:(500.,'test',None) for i in range(100)})
    candidate=dict(_cand(101,.2),acc_mid=.985,pp_mid=300.,pp_hi=350.,best_rank=101)
    s=R.Session([],1000);s.warm_flag=True
    rows,_=P.prepare(rec,[candidate],s)
    assert len(rows)==1 and rows[0]['gain']==.2


def test_ht_is_offered_only_as_a_likely_improvement():
    f={'keys':7,'overall':6.,'sk':{'chordstream':1.},'length':120000,'stars':6.,'notes':1000,'hits':1000}
    def offers(ht_miss):
        rec=R.Recommender.__new__(R.Recommender)
        rec.model=SimpleNamespace(retry=0,rate_response=lambda *_:3.,rate_loss=lambda *a:3.*a[-1],
                                  predict=lambda f,md5,b,rate:(math.log(ht_miss) if rate<1 else -1.,.05,.1))
        rec._shown=(0.,1.)
        rec.pub={'feats':{42:{.75:f,1.:f}},'maps':{42:{'md5':'chart','approved':1,'set':1,'file':'fixture'}}}
        rec.installed={};rec.newer_ranked=lambda:[]
        rec.ledger=R.Ledger({42:(float(R.pp_at(6.,1000,.96)),'test',None)})
        rec.typical=lambda _k:math.log(.04)
        rec._candidates()
        session=R.Session([],1000);session.warm_flag=True
        return [(c['rate'],round(c['p_up'],2)) for c in rec.score(session,keys=[7])[0]]
    likely,unlikely=offers(.03),offers(.043)
    assert likely and likely[0][0]==.75 and likely[0][1]>=.5
    assert unlikely==[]           # a 12–50% HT long shot is no longer offered
    old=R.HT_MIN_GAIN;R.HT_MIN_GAIN=1e9
    try:
        assert offers(.03)==[]    # a likely HT that gains under the floor is not offered either
    finally:
        R.HT_MIN_GAIN=old
