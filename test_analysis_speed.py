"""Focused exactness/cache/lifecycle checks; corpus benchmarks are offline only."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import threading
from unittest.mock import patch

import pytest

import analysis_cache as A
import skill_calc as S
from test_skill_calc import chart


def test_chart_prediction_work_is_reused_without_caching_session_results():
    import warmup
    import recommend as R
    import skill_practice
    import session_form
    from unittest.mock import Mock
    f = {'keys': 7, 'sk': {'stream': .4, 'ln': .8}}
    before = warmup.family_weights(f)
    assert before == warmup._family_weights.__wrapped__(tuple(f['sk'].items()))
    assert warmup.family_weights(dict(f, length=999)) is before
    f['sk']['stream'] = .8
    assert warmup.family_weights(f) == (('rice', .5), ('ln', .5))
    assert warmup.family_weights(f) != before
    vector = session_form.loading(features=f)
    assert session_form.loading(features=dict(f)) is vector
    assert not vector.flags.writeable
    assert skill_practice.match(f, ['ln'])[0] == .8
    f['sk']['ln'] = .1
    assert skill_practice.match(f, ['ln'])[0] == 0.
    assert session_form.loading(features=f) is not vector
    f['sk'] = {'technical': 1., 'patterntech': 1.}
    f['tech_profile'] = {'technical': .01, 'patterntech': .01}
    assert skill_practice.match(f, ['technical'])[0] == 0.
    f['tech_profile']['technical'] = 1.
    assert skill_practice.match(f, ['technical'])[0] == 1.
    model = SimpleNamespace(warmup=SimpleNamespace(rate_slope=Mock(return_value=3.)),
                            pop={}, chart_affinity=Mock(return_value=(0., 0)), levels={})
    with patch.object(R, 'base_of', side_effect=lambda pop, f, bid, rate, level=None: (rate, 0)) as base:
        value = R.Personal.rate_response(model, f, 'chart', 1, 1.)
        assert R.Personal.rate_response(model, f, 'chart', 1, 1.) == value
        assert base.call_count == 2
        assert R.Personal.rate_response(model, dict(f), 'chart', 1, 1.) == value
        assert base.call_count == 4
        R.Personal.rate_response(model, f, 'chart', 1, 1.1)
        assert base.call_count == 6


def fixture(keys=7):
    notes=[(1000+70*i, i%keys, 1000+70*i+(110 if i%5==0 else 0)) for i in range(320)]
    return chart(notes,keys=keys,timing=('0,500,4,1,0,100,1,0','3000,-25,4,1,0,100,0,0',
                                        '4500,-200,4,1,0,100,0,0','8000,-100,4,1,0,100,0,0'))


def test_calculator_hot_stages_honor_selection_cancellation():
    c = S.parse_osu(fixture())
    for calculate in (S.compute, S.sv_reading):
        with pytest.raises(S.AnalysisCancelled):
            calculate(c, cancelled=lambda: True)


def test_live_result_reads_only_its_chart_and_keeps_metadata_fresh(tmp_path):
    import hashlib
    import recdata
    import recommend as R
    path = tmp_path/'played.osu'
    path.write_bytes(b'played chart')
    md5 = hashlib.md5(path.read_bytes()).hexdigest()
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    db = recdata.connect(str(tmp_path/'live.db'))
    with db:
        for chart_sha, digest, bid in ((sha, md5, 123), ('other', 'other', 456)):
            db.execute('INSERT INTO installed(sha256,md5,beatmap_id,title,artist,version) VALUES (?,?,?,?,?,?)',
                       (chart_sha, digest, bid, 'Title', 'Artist', 'Difficulty'))
        db.execute('INSERT INTO installed_local(sha256,online_md5,creator) VALUES (?,?,?)',
                   (sha, md5, 'Mapper'))
    assert R.installed_by_md5(db, md5) == {md5: R.installed_by_md5(db)[md5]}
    db.execute('UPDATE installed SET title=? WHERE sha256=?', ('Renamed', sha))
    messages = []
    worker = R.Worker(messages.append)
    worker.db = db
    score = dict(sha256=sha, score=123, hits={'geki':100}, mods_list=[], rate=1.,
                 played='2026-09-28T12:00:00Z')
    with patch.object(R, 'installed_by_md5', wraps=R.installed_by_md5) as lookup:
        worker.t_score(score, str(path), True, None)
        lookup.assert_called_once_with(db, md5)
    assert db.execute('SELECT beatmap_id FROM scores').fetchone()[0] == 123
    assert any(m['type']=='result' and 'Renamed' in m['title'] for m in messages)
    assert R.installed_by_md5(db, 'missing') == {}
    db.close()


def test_event_decode_reuse_keeps_tracking_and_new_results_fresh(tmp_path):
    import recdata
    import recommend as R
    db = recdata.connect(str(tmp_path/'events.db'))
    start = recdata.log_event(db, 'start', 'md5:'+32*'a', t=100., keys=7, rate=1.)
    queries=[]
    db.set_trace_callback(queries.append)
    before = R.load_events(db, since=0)
    db.set_trace_callback(None)
    assert len(queries) == 1
    before[0]['info']['offered_rate'] = .5
    assert 'offered_rate' not in R.load_events(db, since=0)[0]['info']
    recdata.log_event(db, 'finish', 'md5:'+32*'a', t=200., start_id=start, y=-3.)
    assert len(R.load_events(db, since=0)) == 2
    recdata.set_tracking(db, False, t=150., start=100.)
    assert R.load_events(db, since=0) == []
    recdata.set_tracking(db, True, t=201.)
    recdata.log_event(db, 'start', 'md5:'+32*'b', t=202., keys=7, rate=1.1)
    assert [e['beatmap'] for e in R.load_events(db, since=0)] == ['md5:'+32*'b']
    # Editing an existing result and pause boundaries never reuse stale JSON
    # or exclusions. The batched predicate matches the single-play policy.
    db.execute("UPDATE events SET info=? WHERE t=202", (json.dumps({'keys':7,'rate':1.2}),))
    assert R.load_events(db, since=0)[0]['info']['rate'] == 1.2
    for t in (99., 100., 150., 200., 201.):
        recdata.log_event(db, 'selected', t=t)
    rows=list(db.execute('SELECT * FROM events WHERE t>=0 ORDER BY t,id'))
    assert [r['id'] for r in recdata.tracked_events(db, 0)] == [
        r['id'] for r in rows if recdata.tracking_allowed(db, r['t'])]
    assert all(r['kind']=='start' for r in recdata.tracked_events(db, 0, ('start',)))
    db.close()


def test_memory_disk_and_native_feature_inputs_are_exact(tmp_path):
    import recdata
    path=fixture()
    expected=S.compute(S.parse_osu(path))
    cache=A.AnalysisCache(tmp_path)
    with patch.object(S,'compute',wraps=S.compute) as compute:
        ch,result=cache.analyze(path,calculator='fixture')
        assert result==expected
        assert cache.analyze(path,calculator='fixture')[1] is result
        assert compute.call_count==1
        cache.clear_memory()
        # A separately created helper/viewer can reuse the complete timeline and
        # descriptors, not merely an old rounded overall rating.
        restored=A.AnalysisCache(tmp_path).analyze(path,calculator='fixture')[1]
        assert restored==expected and S.describe(restored)==S.describe(expected)
        assert S.card(restored)==S.card(expected)
        assert compute.call_count==1
    assert recdata.chart_feats(path,[1.],analyses={1.:restored})==recdata.chart_feats(path,[1.])


def test_memory_lookup_never_calculates_or_reads_a_disk_entry(tmp_path):
    import recdata
    path = tmp_path/'map.osu'
    path.write_bytes(Path(fixture()).read_bytes())
    cache = A.AnalysisCache(tmp_path/'cache')
    with patch.object(recdata, 'calc_id', return_value='fixture') as identity:
        expected = cache.analyze(path)
        with patch.object(S, 'parse_osu', side_effect=AssertionError('lookup parsed chart')), \
                patch.object(Path, 'read_bytes', side_effect=AssertionError('lookup read disk')):
            assert cache.get_cached(path) == expected
            assert cache.get_cached(path, 1.1) is None
            assert cache.get_cached(path, sv=False) is None
            assert cache.get_cached(tmp_path/'missing.osu') is None
            identity.return_value = 'changed'
            assert cache.get_cached(path) is None
            identity.return_value = 'fixture'
            stamp = path.stat()
            path.write_text('replacement')
            os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            assert cache.get_cached(path) is None
            cache.clear_memory()
            assert cache.get_cached(path) is None


def test_content_rate_sv_and_calculator_invalidate_independently(tmp_path):
    source=Path(fixture());path=tmp_path/'map.osu';path.write_bytes(source.read_bytes())
    cache=A.AnalysisCache(tmp_path/'cache')
    with patch.object(S,'compute',wraps=S.compute) as compute:
        for rate,sv,cid in ((1.,True,'a'),(.98,True,'a'),(1.,False,'a'),(1.,True,'b')):
            cache.analyze(path,rate,sv,calculator=cid)
        assert compute.call_count==4
        old=path.stat()
        path.write_text(path.read_text().replace('Title:t','Title:u'))
        os.utime(path,ns=(old.st_atime_ns,old.st_mtime_ns))
        changed,_=cache.analyze(path,calculator='a')
        assert changed.title=='u' and compute.call_count==5


def test_cancelled_result_is_not_cached_or_shown(tmp_path):
    path=fixture();cache=A.AnalysisCache(tmp_path)
    calls=[]
    def cancelled():
        calls.append(1)
        return len(calls)>3
    with pytest.raises(S.AnalysisCancelled):cache.analyze(path,calculator='a',cancelled=cancelled)
    assert not cache._memory and not list(tmp_path.glob('*.json'))
    assert cache.analyze(path,calculator='a')[1]['horizon'] is not None


def test_corrupt_and_unwritable_cache_are_misses_and_sizes_are_bounded(tmp_path):
    path=fixture();cache=A.AnalysisCache(tmp_path/'cache',memory_entries=1,disk_entries=2)
    expected=cache.analyze(path,calculator='a')[1]
    for entry in cache.directory.glob('*.json'):entry.write_text('{unfinished')
    cache.clear_memory()
    assert cache.analyze(path,calculator='a')[1]==expected
    for entry in cache.directory.glob('*.json'):
        damaged=json.loads(entry.read_text());damaged['result']['scores'].pop('overall')
        entry.write_text(json.dumps(damaged))
    cache.clear_memory()
    assert cache.analyze(path,calculator='a')[1]==expected
    for rate in (.75,.98,1.19):cache.analyze(path,rate,calculator='a')
    assert len(cache._memory)<=1 and cache._bytes<=cache.memory_budget
    assert len(list(cache.directory.glob('*.json')))<=2
    blocked=tmp_path/'not-a-directory';blocked.write_text('file')
    assert A.AnalysisCache(blocked).analyze(path,calculator='a')[1]==expected


def test_viewer_skips_cancelled_map_and_starts_latest_request():
    import skillsets as ui
    win=SimpleNamespace(_closed=False,_req=(1,'old',1.,True),_gen=1,_req_cv=threading.Condition())
    seen=[];published=[]
    def calculate(path,rate,sv,*,cancelled):
        seen.append(path)
        if path=='old':
            win._gen=2;win._req=(2,'new',1.,True)
            assert cancelled()
            raise S.AnalysisCancelled()
        assert not cancelled()
        return 'chart',{'done':True}
    def publish(callback,out):
        published.append(out);win._closed=True
    win._on_result=lambda out:None
    with patch.object(ui,'analyze',side_effect=calculate),patch.object(ui.GLib,'idle_add',side_effect=publish):
        ui.ManiaScopeWindow._worker(win)
    assert seen==['old','new'] and len(published)==1 and published[0][1]=='new'


def test_song_select_polls_promptly_without_increasing_gameplay_polling():
    import skillsets as ui
    for connected, live, path, delay in ((True, None, 'selected.osu', .01),
                                         (True, 0, 'selected.osu', .3),
                                         (True, None, None, .1), (False, None, None, 1.),
                                         (None, None, None, 1.)):
        watcher=ui.LazerWatcher(*(lambda *args: None for _ in range(5)))
        waits=[]
        def poll():
            if connected is None:
                raise OSError('endpoint disconnected')
            watcher._live=live
            return {'path':path} if connected else None
        def wait(seconds):
            waits.append(seconds)
            watcher.stop()
        with patch.object(watcher,'_poll_tosu',side_effect=poll), \
                patch.object(watcher,'_poll_log',return_value={'path':None}), \
                patch.object(ui,'log_exc'), \
                patch.object(watcher._halt,'wait',side_effect=wait):
            watcher.run()
        assert waits==[delay]


def test_lazy_control_is_retained_for_plans_without_jumptrill_membership():
    from test_gestures import rolls
    c=rolls((0,3,1,2),gap=30,n=40,od=10)
    result=S.N.gestures_analyse(c,with_plans=True)
    # Feasible rolling offsets can produce a plan even when no simultaneous
    # chord fits the windows. Lazy control must cover this separate consumer.
    assert not any(result['series']['rolled'])
    plans=result['plans'][False]
    assert len(plans)==20 and plans[0]['control']==0.15882352941176467
    assert all(isinstance(p['control'],float) for p in plans)


def test_compatible_source_identity_does_not_hide_a_changed_model(tmp_path):
    import recdata
    import shutil
    import native_backend as N
    identity=recdata.calc_id.__wrapped__()
    for path in N.identity_files():
        copy=tmp_path/path.relative_to(N.ROOT)
        copy.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,copy)
    copied=lambda:recdata._calculator_identity(N.identity_files(tmp_path))
    assert copied()==identity
    for name in ('calib/difficulty.json','native/src/demand.rs'):
        original=(tmp_path/name).read_text()
        (tmp_path/name).write_text(original+'\n')
        assert copied()!=identity
        (tmp_path/name).write_text(original)
    with patch.object(recdata,'SKILLS',recdata.SKILLS+('different-feature',)):
        assert recdata.calc_id.__wrapped__()!=identity
