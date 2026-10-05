"""Skill membership, objective separation and shared-builder routing checks."""
from types import SimpleNamespace
from unittest.mock import patch
import pytest

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


def test_demanded_is_match_for_every_single_skill():
    """Stats reads group membership from one pass; it must equal match(f, [g]) per group."""
    import skill_practice as sp
    for f in ({'keys': 7, 'sk': {'ln': .9, 'stream': .3}}, {'keys': 4, 'sk': {'jackspeed': .8, 'chordjack': .7}},
              {'keys': 7, 'sk': {'chordstream': .5, 'bracket': .48, 'sv': .2}}):
        dem = sp.demanded(f)
        assert dem and all((g in sp.match(f, [g])[1]) == (g in dem) for g in sp.dimensions(f['keys']))


def test_builder_recovers_after_broken_process_pool(tmp_path, monkeypatch):
    from concurrent.futures import Future
    from concurrent.futures.process import BrokenProcessPool
    import webmaps
    monkeypatch.setattr(webmaps, 'WEB_DIR', str(tmp_path/'web'))
    db = recdata.connect(str(tmp_path/'rec.db'))
    monkeypatch.setattr(recdata, 'connect', lambda: db)
    pools, messages = [], []
    f = feature({'ln': 1.})

    class Pool:
        def __init__(self, **kwargs):
            self.broken = not pools
            self.closed = False
            pools.append(self)

        def submit(self, *args):
            future = Future()
            if self.broken:
                future.set_exception(BrokenProcessPool('worker exited'))
            else:
                future.set_result(('a', {1.: f}, None))
            return future

        def shutdown(self, **kwargs):
            self.closed = True

    monkeypatch.setattr(nps, 'ProcessPoolExecutor', Pool)
    session = R.Session([], 1000)
    session.warm_flag = True

    def emit(kind, data):
        messages.append((kind, data))
        if kind == 'error':
            if sum(k == 'error' for k, _ in messages) == 1:
                builder.configure(Predictor(), session, (7,), 2, 3, (7,), ('ln',), 'skills')
            else:
                builder.stop()
        elif data['states']['skills']['candidates']:
            builder.stop()

    builder = nps.Builder(emit)
    builder.configure(Predictor(), session, (7,), 1, 3, (7,), ('ln',), 'skills')
    with patch.object(nps, 'installed', return_value=({'a': inst('a')}, '')):
        builder.run()
    assert len(pools) == 2 and all(p.closed for p in pools)
    assert sum(k == 'error' for k, _ in messages) == 1
    ready = [data for kind, data in messages if kind == 'ready'][-1]
    assert ready['revision'] == 2 and ready['states']['skills']['candidates'][0]['sha'] == 'a'


def test_builder_retries_incomplete_catalogue(tmp_path, monkeypatch):
    from unittest.mock import Mock
    import webmaps
    monkeypatch.setattr(webmaps, 'WEB_DIR', str(tmp_path/'web'))
    db = recdata.connect(str(tmp_path/'rec.db'))
    monkeypatch.setattr(recdata, 'connect', lambda: db)
    monkeypatch.setattr(nps, 'ProcessPoolExecutor', Mock(return_value=Mock()))
    catalog = Mock(side_effect=[RuntimeError('catalogue unavailable'),
                               ({'a': inst('a')}, {'a': {1.: feature({'ln': 1.})}})])
    monkeypatch.setattr(webmaps, 'catalog', catalog)
    messages = []
    session = R.Session([], 1000)
    session.warm_flag = True

    def emit(kind, data):
        messages.append((kind, data))
        if kind == 'error':
            builder.configure(Predictor(), session, (7,), 2, 3, (7,), ('ln',), 'skills')
        else:
            builder.stop()

    builder = nps.Builder(emit)
    builder.configure(Predictor(), session, (7,), 1, 3, (7,), ('ln',), 'skills')
    with patch.object(nps, 'installed', return_value=({}, '')):
        builder.run()
    assert catalog.call_count == 2
    assert messages[-1][1]['states']['skills']['candidates'][0]['sha'] == 'a'


def test_builder_failure_is_scoped_preserves_list_and_clears_on_ready(tmp_path):
    worker = R.Worker(lambda msg: None)
    worker.db = recdata.connect(str(tmp_path/'rec.db'))
    worker.generation, worker._local_request_id, worker.mode = 3, 2, 'skills'
    candidate = dict(inst('a'), sha='a', f=feature({'ln': 1.}), rate=1., base_value=0.)
    previous = dict(worker.skills_state, candidates=[candidate], total=2, analyzed=1, restored=True)
    worker.skills_state = previous
    worker.rec = SimpleNamespace(feats=SimpleNamespace(cache={}, key=lambda s, r: s, calc='fixture'))
    saved = {'version': 1, 'choices': [['a', 1.]], 'analyzed': 1}
    recdata.kv_set(worker.db, 'playlist_cache:skills', saved)
    error = dict(generation=3, revision=2, mode='skills', error='BrokenProcessPool: worker exited',
                 keys_by_mode=worker.keys, selected_skills=())
    try:
        with patch.object(worker, '_publish') as publish, patch.object(worker, '_prepare_local', return_value=[]):
            worker.t_nps_update('error', dict(error, revision=1))
            assert worker.skills_state is previous and not publish.called
            worker.t_nps_update('error', error)
            assert worker.skills_state['candidates'] is previous['candidates']
            assert recdata.kv_get(worker.db, 'playlist_cache:skills') == saved
            _, _, note = worker._pool('skills')
            assert 'worker exited' in note and 'looking for maps' not in note and 'while it updates' not in note
            state = dict(previous, restored=False)
            worker.t_nps_update('ready', dict(error, states={'skills': state}))
            assert not worker.skills_state.get('build_failed')
            assert worker.skills_state['error'] == ''
            assert 'worker exited' not in worker._pool('skills')[2]
        # The real publication must also stop the GTK wait banner, not only
        # change its note while leaving an incomplete progress tuple active.
        worker.skills_state = dict(previous, error=error['error'], build_failed=True)
        worker.rec.rows = []
        emitted = []
        worker.emit = emitted.append
        with patch.object(worker, '_pool', return_value=('build', [], 'analysis stopped')), \
             patch.object(worker, '_prepared_playlist', return_value=[]), patch.object(R, 'summary', return_value='fixture'):
            worker._publish('skills')
        assert emitted[-1]['progress'] is None
    finally:
        worker.db.close()


@pytest.mark.parametrize('interrupt', ['stop', 'configure'])
@pytest.mark.parametrize('running', [False, True])
def test_builder_interrupts_pending_result(tmp_path, monkeypatch, interrupt, running):
    import threading
    from concurrent.futures import Future
    from unittest.mock import Mock
    import webmaps
    monkeypatch.setattr(webmaps, 'WEB_DIR', str(tmp_path/'web'))
    connect = recdata.connect
    monkeypatch.setattr(recdata, 'connect', lambda: connect(str(tmp_path/'rec.db')))
    first = Future()
    if running:
        first.set_running_or_notify_cancel()
    submitted = threading.Event()
    ready = []

    def submit(*args):
        if not submitted.is_set():
            submitted.set()
            return first
        future = Future()
        future.set_result(('a', {1.: feature({'ln': 1.})}, None))
        return future

    pool = Mock(submit=submit)
    monkeypatch.setattr(nps, 'ProcessPoolExecutor', lambda **kwargs: pool)
    session = R.Session([], 1000)
    session.warm_flag = True

    def emit(kind, data):
        if kind == 'ready':
            ready.append(data)
            if data['revision'] == 2 and data['states']['skills']['candidates']:
                builder.stop()

    builder = nps.Builder(emit)
    builder.configure(Predictor(), session, (7,), 1, 3, (7,), ('ln',), 'skills')
    with patch.object(nps, 'installed', return_value=({'a': inst('a')}, '')):
        builder.start()
        try:
            assert submitted.wait(2)
            if interrupt == 'stop':
                builder.stop()
            else:
                builder.configure(Predictor(), session, (7,), 2, 3, (7,), ('ln',), 'skills')
            builder.join(2)
            assert not builder.is_alive(), 'stop/new request stayed blocked on the old future'
            assert first.cancelled() == (not running)
            if interrupt == 'configure':
                assert ready[-1]['revision'] == 2 and ready[-1]['states']['skills']['candidates']
        finally:
            builder.stop()
            if not first.done():
                first.set_result(('a', {1.: feature({'ln': 1.})}, None))
            builder.join(2)


def _cached_builder(tmp_path, monkeypatch, count):
    import json
    import webmaps
    monkeypatch.setattr(webmaps, 'WEB_DIR', str(tmp_path/'web'))
    db = recdata.connect(str(tmp_path/'rec.db'))
    with db:
        db.executemany('INSERT INTO feats VALUES (?,?)',
                       [(f'a@{1+i*.001:.3f}|fixture', json.dumps(dict(feature({'ln': 1.}), cache_index=i)))
                        for i in range(count)])
    builder = nps.Builder(lambda *args: None)
    builder._catalog_key = ('fixture', (7,), None, None, None)
    builder._catalog, builder._catalog_error = {'a': inst('a')}, ''

    def done(*args, **kwargs):
        builder.stop()                 # no prediction/process work is needed for the cache check
        return [], []

    monkeypatch.setattr(nps, 'candidates_from', done)
    request = (Predictor(), None, (7,), 1, 3, (7,), (), 'skills')
    return db, builder, request


def test_builder_feature_decode_releases_read_lock_and_defers_new_rows(tmp_path, monkeypatch):
    import json
    db, builder, request = _cached_builder(tmp_path, monkeypatch, 270)
    writer = recdata.connect(str(tmp_path/'rec.db'))
    writer.execute('PRAGMA busy_timeout=20')
    with writer:
        writer.execute('INSERT INTO feats VALUES (?,?)', ('old@1.000|old-calculator', '{}'))
    upper = writer.execute('SELECT MAX(rowid) FROM feats').fetchone()[0]
    loads = json.loads
    decoded = []

    def decode(raw):
        value = loads(raw)
        if 'cache_index' in value:
            recdata.log_event(writer, 'start', 'a', t=1.)
            if not decoded:
                with writer:
                    writer.execute('INSERT INTO feats VALUES (?,?)',
                                   ('a@2.000|fixture', json.dumps(dict(feature({'ln': 1.}), cache_index=270))))
                    writer.execute('INSERT OR REPLACE INTO feats VALUES (?,?)',
                                   ('a@1.000|fixture', json.dumps(dict(feature({'ln': 1.}), cache_index=271))))
            decoded.append(value['cache_index'])
        return value

    monkeypatch.setattr(nps.json, 'loads', decode)
    try:
        builder._build(db, None, request)
        assert decoded == list(range(270))
        assert builder._loaded_rowid == upper   # advance past the irrelevant calculator tail
        assert 2. not in builder._analyses['a']
        assert builder._analyses['a'][1.]['cache_index'] == 0
        builder.halt.clear()
        builder._build(db, None, request)
        assert decoded == list(range(272))
        assert builder._analyses['a'][2.]['cache_index'] == 270
        assert builder._analyses['a'][1.]['cache_index'] == 271
        assert builder._loaded_rowid == writer.execute('SELECT MAX(rowid) FROM feats').fetchone()[0]
        assert writer.execute("SELECT COUNT(*) FROM events WHERE kind='start'").fetchone()[0] == 272
    finally:
        db.close()
        writer.close()


@pytest.mark.parametrize('interrupt', ['stop', 'configure'])
def test_builder_interrupts_cache_scan_and_resumes_remaining_rows(tmp_path, monkeypatch, interrupt):
    import json
    db, builder, request = _cached_builder(tmp_path, monkeypatch, 300)
    loads = json.loads
    decoded = []

    def decode(raw):
        value = loads(raw)
        if 'cache_index' in value:
            decoded.append(value['cache_index'])
            if len(decoded) == 1:
                if interrupt == 'stop':
                    builder.stop()
                else:
                    builder.configure(Predictor(), None, (7,), 2, 3, (7,), (), 'skills')
        return value

    monkeypatch.setattr(nps.json, 'loads', decode)
    try:
        builder._build(db, None, request)
        assert 0 < len(decoded) <= 256
        assert builder._loaded_rowid == len(decoded)
        builder.halt.clear()
        with builder.cv:
            continuation = builder.request or request
            builder.request = None
        builder._build(db, None, continuation)
        assert decoded == list(range(300))
        assert builder._loaded_rowid == 300
        assert len(builder._analyses['a']) == 300
    finally:
        db.close()


@pytest.mark.parametrize('action', ['resume', 'stop', 'configure'])
def test_builder_pauses_cache_decode_during_gameplay_and_preserves_shortlist(tmp_path, monkeypatch, action):
    import json
    import threading
    from unittest.mock import Mock
    db, builder, request = _cached_builder(tmp_path, monkeypatch, 300)
    playing, paused = threading.Event(), threading.Event()
    playing.set()
    decoded, errors = [], []
    loads = json.loads

    def decode(raw):
        value = loads(raw)
        if 'cache_index' in value:
            decoded.append(value['cache_index'])
        return value

    def busy():
        if playing.is_set():
            paused.set()
            return True
        return False

    worker = R.Worker(lambda message: None)
    worker.db, worker.generation, worker._local_request_id, worker.mode = db, 3, 1, 'skills'
    worker.keys['nps'] = worker.keys['skills'] = (7,)
    worker.practice_skills = ()
    worker.rec = SimpleNamespace(feats=SimpleNamespace(cache={}, calc='fixture', key=lambda s, r: (s, r)))
    previous = dict(worker.skills_state, candidates=[{'sha': 'previous'}], restored=True)
    worker.skills_state = previous
    worker.playlists['skills'] = [{'sha': 'previous'}]
    candidate = dict(inst('a'), sha='a', rate=1., f=feature({'ln': 1.}), base_value=1.)

    def ready(*args, **kwargs):
        builder.stop()
        return [candidate], []

    def build():
        try:
            builder._build(db, None, request)
        except Exception as exc:
            errors.append(exc)

    monkeypatch.setattr(nps.json, 'loads', decode)
    monkeypatch.setattr(nps, 'candidates_from', ready)
    builder.busy, builder.emit = busy, worker.t_nps_update
    thread = threading.Thread(target=build)
    try:
        with patch.object(worker, '_publish', Mock()) as published:
            thread.start()
            assert paused.wait(2), 'builder never observed active gameplay'
            assert decoded == [] and builder._loaded_rowid == 0
            assert worker.skills_state is previous and worker.playlists['skills'] == [{'sha': 'previous'}]
            published.assert_not_called()
            if action == 'resume':
                playing.clear()
            elif action == 'stop':
                builder.stop()
            else:
                worker._local_request_id = 2
                builder.configure(Predictor(), None, (7,), 2, 3, (7,), (), 'skills')
            thread.join(2)
            assert not thread.is_alive() and not errors
            if action != 'resume':
                assert decoded == [] and worker.skills_state is previous
                with builder.cv:
                    continuation = builder.request or request
                    builder.request = None
                playing.clear()
                builder.halt.clear()
                builder._build(db, None, continuation)
            assert decoded == list(range(300)) and builder._loaded_rowid == 300
            assert worker.skills_state['candidates'] == [candidate]
            assert worker.rec.feats.cache[('a', 1.)] is candidate['f']
            assert recdata.kv_get(db, 'playlist_cache:skills')['choices'] == [['a', 1.]]
            published.assert_called_with('skills')
    finally:
        builder.stop()
        playing.clear()
        thread.join(2)
        db.close()
