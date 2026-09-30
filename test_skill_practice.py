"""Skill membership, objective separation and shared-builder routing checks."""
from types import SimpleNamespace
from unittest.mock import patch

import nps
import recdata
import recommend as R
import skill_practice as S


def feature(sk, nps_value=20., acc=.94, length=160., rate=1.):
    return {"keys": 7, "overall": 6., "length": length*1000., "notes": int(length*nps_value),
            "ln": .5 if sk.get("ln") else 0., "stam": .8, "sk": sk, "test_acc": acc,
            "nps": {"version": nps.STRUCTURE_VERSION, "nps": nps_value*rate,
                    "play_span": length, "family": "test", "drill": 0.}}


class Predictor:
    model = SimpleNamespace(pop={"slope": {0: 2.}})
    md5_bid = {}
    calc = "fixture"
    pub = {"feats": {}}

    def predict(self, f, *args, accuracy_only=False):
        return {"acc_mid": f["test_acc"], "acc_lo": f["test_acc"]-.025,
                "acc_hi": f["test_acc"]+.015, "sd_model": .1}


def inst(sha, keys=7):
    return {"sha256": sha, "md5": sha, "keys": keys, "beatmap_id": -1, "title": sha,
            "artist": "Test", "version": "Local", "length": 160000}


def test_broad_membership_union_not_incidental_traces():
    f = feature({"ln": 1., "release": .9, "chordjack": .85, "sv": .15})
    assert S.match(f, ["ln", "sv"])[1] == ("ln",)
    assert S.match(f, ["sv"])[0] == 0
    assert S.match(f, ["chordjack"])[0] > .8
    assert S.match(f, ())[0] == 1
    assert S.normalize(["ln", "ln", "sv", "release", "nonsense"]) == ("ln", "sv")


def test_jacks_related_but_not_same_as_chordjacks():
    for leaf in ("jack", "jackspeed", "minijack", "longjack"):
        f = feature({leaf: 1.})
        assert S.match(f, ["jack"])[0] == 1 and S.match(f, ["chordjack"])[0] == 0
    cj = feature({"chordjack": 1.})
    # The explicit Jackspeed family is no longer the old vague umbrella that
    # silently included changing chords. Chordjack has its own selection.
    assert S.match(cj, ["jackspeed"])[0] == 0
    assert S.match(cj, ["chordjack"])[0] == 1
    for leaf in ("jumptrill", "trill1h", "bracket"):
        assert S.match(feature({leaf: 1.}), ["trill"])[0] == 1
    assert S.match(dict(feature({"splittrill":1.}),keys=4), ["jumpstream/splittrill"])[0] == 1
    assert S.match(feature({"splittrill":1.}), ["jumpstream/splittrill"])[0] == 0
    assert S.match(feature({"delay": 1.}), ["technical"])[0] == 0


def test_skills_any_does_not_reward_nps_and_checks_whole_map():
    catalog = {s: inst(s) for s in ('ordinary', 'dense', 'ln-with-impossible-wall')}
    analyses = {"ordinary": {1.: feature({"stream": 1.})},
                "dense": {1.: feature({"stream": 1.}, nps_value=70.)},
                "ln-with-impossible-wall": {1.: feature({"ln": 1.}, acc=.78)}}
    session = R.Session([], 1000)
    session.warm_flag=True  # isolate normal density objective from cold readiness
    skills, _ = nps.candidates_from(catalog, analyses, Predictor(), session, mode="skills")
    highnps, _ = nps.candidates_from(catalog, analyses, Predictor(), session)
    assert {c['sha'] for c in skills} == {'ordinary', 'dense'}
    assert skills[0]['base_value'] == skills[1]['base_value']
    assert highnps[1]['base_value'] > highnps[0]['base_value']
    assert not nps.candidates_from(catalog, analyses, Predictor(), session, mode='skills', selected_skills=['sv'])[0]


def test_best_skills_rate_is_exact_and_no_unanalysed_rate_is_offered():
    f = feature({'ln': 1.})
    fs = {r: dict(f, test_acc=a) for r,a in ((.69,.94),(.7,.96),(.85,.943),(1.,.92),(1.5,.84),(1.51,.94))}
    cands, proposals = nps.candidates_from({'a': inst('a')}, {'a': fs}, Predictor(), R.Session([],1000),
                                          mode='skills', selected_skills=['ln'])
    assert len(cands) == 1 and cands[0]['rate'] == .85
    assert cands[0]['practice_skills'] == ('ln',) and cands[0]['purpose'] == 'skills'
    assert all(.7 <= r <= 1.5 for _score,_sha,r in proposals)


def test_skill_warmup_uses_related_play_not_tab_selection():
    s = R.Session([],1000)
    assert s.warmup_penalty(7,'chordjack') > 0
    s.warm_flag = True
    assert s.warmup_penalty(7,'chordjack') == 0
    from test_redesign import events
    warm = R.Session(events([0.,0.,0.,0.],seconds=120),1580)
    assert warm.warmup_penalty(7,'chordjack') < warm.warmup_penalty(7,'ln')


def test_builder_only_works_for_active_objective_and_reuses_catalogue(tmp_path, monkeypatch):
    import webmaps
    monkeypatch.setattr(webmaps, 'WEB_DIR', str(tmp_path/'web'))     # never the real website catalogue
    db = recdata.connect(str(tmp_path/'rec.db'))
    f = feature({'ln': 1., 'stream': .1})
    with db:
        db.execute('INSERT INTO feats VALUES (?,?)',('a@1.000|fixture',__import__('json').dumps(f)))
    messages=[];builder=nps.Builder(lambda kind,data:messages.append((kind,data)))
    with patch.object(nps,'installed',return_value=({'a':inst('a')},'')) as installed:
        builder._build(db,None,(Predictor(),R.Session([],1000),(7,),12,3,(7,),('ln',),'skills'))
        assert messages and messages[-1][0]=='ready'
        data=messages[-1][1]
        assert data['revision']==12 and set(data['states'])=={'skills'}
        assert data['states']['skills']['candidates'][0]['practice_skills']==('ln',)
        builder._build(db,None,(Predictor(),R.Session([],1000),(7,),13,3,(7,),('ln',),'nps'))
        assert set(messages[-1][1]['states'])=={'nps'}
        assert messages[-1][1]['states']['nps']['candidates'][0]['mode']=='nps'
        assert installed.call_count==1
        before=len(messages)
        builder._build(db,None,(Predictor(),R.Session([],1000),(7,),14,3,(7,),('ln',),'pp'))
        assert len(messages)==before and installed.call_count==1
    db.close()


def test_worker_rejects_stale_filters_and_routes_skills_next(tmp_path):
    out=[];w=R.Worker(out.append);w.db=recdata.connect(str(tmp_path/'rec.db'))
    w.rec=SimpleNamespace(local_shas={'a'})
    w.revision=5;w.practice_skills=('ln',)
    c={'sha':'a','keys':7,'mode':'skills','purpose':'skills','rate':1.,'var':'NM','weight':1.,
       'practice_skills':('ln',),'f':feature({'ln':1.})}
    w.playlists['skills']=[c]
    with patch.object(w,'_pool',return_value=('build',[c],'')), patch.object(w,'_publish'), patch.object(w,'_open') as opened:
        w.t_next(mode='skills',revision=4)
        assert not opened.called
        w.t_next(mode='skills',revision=5)
        assert opened.call_args.args[1]=='skills'
        opened.reset_mock()
        w.t_open(dict(c,practice_skills=('sv',)),'skills',5)
        assert not opened.called
        w.t_open(c,'skills',5)
        assert opened.called
    w.generation=1;w._local_request_id=4
    w.t_nps_update('ready',{'generation':1,'revision':3}) # stale builder never replaces current state
    assert w.skills_state['candidates']==[]
    w.db.close()


def test_skip_warmup_invalidates_both_local_prediction_caches(tmp_path):
    db=recdata.connect(str(tmp_path/'rec.db'));w=R.Worker(lambda msg:None);w.db=db
    recdata.log_event(db, 'selected', '1')
    calls=[]
    def predict(f,chart,bid,rate,session,*,accuracy_only=False):
        assert accuracy_only
        calls.append(session.warm_flag)
        return {'acc_mid':.95-session.warmup_penalty(7,'ln')/10, 'sd_model':.1}
    w.rec=SimpleNamespace(local_shas={'a'},predictor=SimpleNamespace(predict=predict),
                          feats=SimpleNamespace(md5_bid={}))
    f=feature({'ln':1.})
    c=dict(inst('a'),sha='a',f=f,rate=1.,nps=20.,length=160.,family='a',group='ln',
           profile=[1.],description='LN',base_value=0.,practice_skills=('ln',))
    w.practice_skills=('ln',)
    w.skills_state=dict(w.skills_state,candidates=[c]);w.nps_state=dict(w.nps_state,candidates=[c])
    with patch.object(nps, 'weighted_candidates', wraps=nps.weighted_candidates) as weighted, \
            patch.object(nps, 'taste', wraps=nps.taste) as taste:
        w._pool('skills')
        assert not taste.called
        w._pool('nps');w._pool('skills');w._pool('nps')
        assert weighted.call_count == 2 and taste.call_count == 1
    assert calls==[False]
    with patch.object(w,'_request_nps'),patch.object(w,'_publish'):
        w.t_skip_warmup()
    w._pool('skills');w._pool('nps')
    assert calls==[False,True]
    db.close()
