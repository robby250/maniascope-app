"""Focused NPS/filter/session/navigation checks; synthetic stores, no real writes."""
import pytest
import json
import math
import os
import re
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import navigation
import nps
import recdata
import recommend as R


def db():
    return recdata.connect(os.path.join(tempfile.mkdtemp(), "rec.db"))


def test_keys_and_identity():
    assert recdata.normalize_keys(None) == tuple(range(4, 11))
    assert recdata.normalize_keys([9, 4, 6, 4]) == (4, 6, 9)
    assert recdata.normalize_keys([]) == ()
    assert recdata.normalize_keys([True, "garbage", 1, 18]) == ()
    assert recdata.keys_label([4, 5, 6]) == "4K–6K"
    a, b = {"bid": -1, "md5": "a" * 32}, {"bid": -1, "md5": "b" * 32}
    assert R.event_key(a) != R.event_key(b)
    assert R.event_key({"bid": -1}) is None
    assert R.legacy_key("-1") is None
    assert navigation.positive_id(-1) is None


def test_density_and_vibro():
    c = SimpleNamespace(keys=7, notes=[(0, 0, 0), (0, 500, 1), (1000, 3000, 2)])
    assert recdata.raw_nps(c, 1) == 3
    assert recdata.raw_nps(c, .98) == 2.94
    st = nps.structure(c)
    assert st["head_span"] == 1 and st["play_span"] == 3 and st["nps"] == 3
    assert recdata.raw_nps(c, None) is None
    c = SimpleNamespace(keys=7, notes=[(40*i, 40*i, 2) for i in range(200)])
    assert nps.structure(c)["vibro_focused"]
    masks = ((0, 1, 2), (3, 4, 5), (1, 2, 6), (0, 3, 5))
    c = SimpleNamespace(keys=7, notes=[(70*i, 70*i, k) for i in range(400) for k in masks[i % 4]])
    assert not nps.structure(c)["vibro_focused"]
    slow = SimpleNamespace(keys=7, notes=[(round(t/.9), round(e/.9), k) for t,e,k in c.notes])
    assert nps.structure(c)["family"] == nps.structure(slow)["family"]
    assert abs(recdata.raw_nps(slow, 1) / recdata.raw_nps(c, 1) - .9) < .0001


def test_local_search_and_actions():
    meta = {"Artist": "Cosmo", "Title": "Cyber Shaman", "Version": 'Nakano Yuko\'s "01110011 01100101 01111000"'}
    c = {"bid": -1, "keys": 7, "installed": True, "rate": .87, "search_meta": meta}
    q = navigation.song_search(c)
    assert 'title="Cyber Shaman"!' in q and 'diff="01110011 01100101 01111000"' in q
    assert '\\"' not in q and "0.87" not in q and "cs=7" in q
    # Match the complete key/value tokenization of the inspected upstream parser.
    pattern = re.compile(r'\b(?P<key>\w+)(?P<op>!?[:=]|[><][:=]?)(?P<value>".*?"!?|\S*)')
    matches = list(pattern.finditer(q))
    assert " ".join(m.group(0) for m in matches) == q
    with patch.object(navigation, "lazer_running", return_value=True), patch.object(navigation.subprocess, "Popen") as pop, \
         patch.object(recdata, "local_file", return_value=__file__):
        assert navigation.perform(c, "auto")["copy"] == q
        assert navigation.perform(c, "link")["copy"] == q
        assert not pop.called
        submitted = dict(c, bid=123, online_match=True)
        assert navigation.perform(submitted, "auto")["copy"] == q
        assert not pop.called
        assert navigation.perform(submitted, "link")["sent"]
        assert pop.call_count == 1
        assert navigation.perform(dict(submitted, online_match=False), "link")["copy"] == q


def test_next_accepts_successive_requests_and_reports_errors():
    from contextlib import ExitStack
    from unittest.mock import Mock
    messages = []
    worker = R.Worker(messages.append)
    worker.rec = object()
    worker.db = None
    worker.playlists['nps'] = [{'sha':'fixture', 'purpose':'nps', 'var':'NM'}]
    with patch.object(worker, '_pool', return_value=('warmup', [], '')), \
         patch.object(worker, '_open') as opened, patch.object(worker, '_publish') as publish, \
         patch.object(R.recdata, 'log_event'), patch.object(R.time, 'time', return_value=100.):
        worker.t_next(False, 'nps', 0)
        worker.t_next(False, 'nps', 0)
        assert opened.call_count == 2  # processing time is not click time
        worker.playlists['nps'] = [dict(worker.playlists['nps'][0], sha='different')]
        worker.t_next(False, 'nps', 0)
        assert opened.call_args.args[0]['sha'] == 'different'
        assert [worker.q.get_nowait() for _ in range(3)] == [('publish',)]*3
        publish.assert_not_called()
        worker.t_publish()
        publish.assert_called_once_with()
    def emit(msg):
        messages.append(msg)
        if msg['type'] == 'status' and 'task next failed' in msg['text']: worker._halt.set()
    worker.emit = emit
    worker.put('next', False, 'nps', 0)
    with ExitStack() as stack:
        stack.enter_context(patch.object(R.recdata, 'connect', return_value=None))
        stack.enter_context(patch.object(R.recdata, 'load_public', return_value=None))
        stack.enter_context(patch.object(R, 'Recommender', return_value=object()))
        stack.enter_context(patch.object(nps, 'Builder', return_value=Mock()))
        for name in ('_restore_local', '_request_nps', '_publish', '_fill', '_begin_import', '_catch_up_replays'):
            stack.enter_context(patch.object(worker, name))
        stack.enter_context(patch.object(worker, 't_next', side_effect=RuntimeError('fixture')))
        worker.run()
    assert any(m['type'] == 'status' and 'task next failed' in m['text'] for m in messages)


def test_filter_before_pp_limit_and_ledger_unchanged():
    r = R.Recommender.__new__(R.Recommender)
    fs, maps = {}, {}
    for b in range(605):
        f = {"keys": 4 if b == 604 else 7, "stars": 1 if b == 604 else 6,
             "overall": 5, "hits": 2000, "notes": 2000, "length": 100000,
             "sk": {"stream": 1}, "ln": 0, "stam": 1, "od": 8}
        fs[b+1] = {1.: f}
        maps[b+1] = {"approved": 1, "md5": f"{b:032x}", "set": b+1, "file": str(b), "keys": f["keys"]}
    r.pub = {"feats": fs, "maps": maps}
    r.model = SimpleNamespace(retry=0, predict=lambda *_a: (-3., .1, .3),
                              rate_response=lambda *_a: 3.)
    r.installed = {}
    r.newer_ranked = lambda: []
    r.ledger = R.Ledger({999: (50., "test", None)})
    r.typical = lambda _k: -3.
    r._shown = (math.log(.75), 1.)
    r._candidates()
    seen=[]
    r.model.rate_response=lambda f,*_a:seen.append(f['keys']) or 3.
    before = r.ledger.total()
    session=R.Session([]);session.warm_flag=True                # isolate filtering from the cold-keymode guard
    rows, _s = r.score(session, keys=[4])
    assert len(rows) == 1 and rows[0]["keys"] == 4
    assert seen==[4]  # excluded keymodes must not incur per-map prediction work
    assert r.ledger.total() == before
    assert r.score(session, keys=[])[0] == []


def test_marathon_and_unknown_context():
    now = 10000.
    ev = [{"id": 1, "t": now, "kind": "start", "beatmap": "md5:a", "info":
           {"mu": -3., "sd": .4, "keys": 7, "skill": "chordstream", "length": 4000, "ranked": False, "purpose": "nps"}},
          {"id": 2, "t": now+4000, "kind": "finish", "beatmap": "md5:a", "info":
           {"start_id": 1, "prediction_valid": True, "y": -3.1}}]
    s = R.Session(ev, now+4001)
    assert len(s.attempts) == 1 and s.attempts[0]["z"] > 0
    assert s.phase() != "warmup"
    assert R.Session(ev[:1], now+3500).events
    assert not R.Session(ev, now+4001+R.SESSION_GAP).attempts
    assert not recdata.predictable_context(None, 1.)
    assert not recdata.predictable_context([], None)
    assert recdata.predictable_context([{"acronym":"HT","settings":{"speed_change":.87}}], .87)
    assert not recdata.predictable_context([{"acronym":"CS"}], 1.)
    f = R.FeatureSource(db(), {"maps": {}})
    assert f({"rate": None}) == (None, None)


def test_smooth_public_rate_effect():
    pop = {"slope": {0: 3.}, "window": {0: -.9}, "mb": {1: (.1, 50)},
           "mbr": {(1,.75): (.3,10), (1,1.): (.6,20), (1,1.5): (.2,10)}}
    f = {"keys":7, "overall":5, "od":8}
    a = R.base_of(pop, f, 1, 1.)[0]
    assert abs(a - R.base_of(pop, f, 1, .9999)[0]) < .001
    assert abs(a - R.base_of(pop, f, 1, 1.0001)[0]) < .001


def test_variety_preserves_chordjack_and_family_identity():
    cands = []
    for i in range(35):
        g = 0 if i < 25 else 1 if i < 30 else 2
        v = [0., 0., 0.];v[g] = 1.
        cands.append({"sha":str(i), "md5":f"{i:032x}", "family":str(i), "profile":v,
                      "keys":7, "group":str(g), "base_value":math.log(60 if g == 0 else 40),
                      "nps":60 if g == 0 else 40, "length":240, "description":("Chordjack","Chordstream","LN")[g],
                      "acc_mid":.95, "bid":None})
    cands.append(dict(cands[30], sha="copy30"))
    got = nps.select(cands, R.Session([]), [7])
    assert len(got) == 20 and len({c["group"] for c in got}) == 3
    assert len({c["family"] for c in got}) == len(got)
    assert nps.select(cands, R.Session([]), [4]) == []


def test_taste_requires_real_completion_and_is_idempotent():
    d = db(); now = time.time()
    recdata.log_event(d, "next", "a", mode="nps", keys=7, group="chords")
    assert nps.taste(d) == {}
    for g in ("chords", "ln"):
        for i in range(6):
            offer = recdata.log_event(d, "offer", f"{g}{i}", mode="nps", keys=7, group=g)
            if g == "chords" and i < 4:
                start = recdata.log_event(d, "start", f"{g}{i}", offer_id=offer)
                recdata.log_event(d, "finish", f"{g}{i}", start_id=start, key=str(start))
                recdata.log_event(d, "finish", f"{g}{i}", start_id=start, key=str(start))
    t = nps.taste(d)
    assert 0 < t[(7,"chords")] <= .25 and -.25 <= t[(7,"ln")] < 0
    recdata.log_event(d, "taste_reset")
    assert nps.taste(d) == {}


if __name__ == "__main__":
    for name in sorted(n for n in globals() if n.startswith("test_")):
        globals()[name]()
        print("ok", name)


def test_prepared_playlist_preserves_order_and_logs_immediate_selection(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import Mock
    worker = R.Worker(lambda msg: messages.append(msg))
    messages = []
    worker.db = recdata.connect(str(tmp_path/'rec.db'))
    worker.rec = SimpleNamespace(local_shas={'a','b','c'}, feats=SimpleNamespace(calc='test'),
        predictor=SimpleNamespace(predict=Mock(return_value={'acc_mid':.96})))
    a = dict(sha='a',keys=7,rate=1.,var='NM',purpose='nps',f={},installed=True,acc_mid=.94)
    b, c = dict(a,sha='b'), dict(a,sha='c')
    worker.playlists['nps'] = [a,b]
    with patch.object(worker, '_session', return_value=SimpleNamespace(warm_flag=True)):
        shown = worker._prepared_playlist('nps', [c,b,a])
        assert [x['sha'] for x in shown] == ['a','b','c']
        assert shown[0]['acc_mid'] == .96 and a['acc_mid'] == .94
        result = dict(copy='already copied',msg='Copied',sent=False)
        with patch.object(navigation, 'perform') as perform, patch.object(worker, '_pool') as pool:
            # A later filter change cannot discard a selection already made.
            worker.revision=2
            worker.t_next(False,'nps',1,a,result,100.,1)
            perform.assert_not_called(); pool.assert_not_called()
        events = worker.db.execute('SELECT kind,t,info FROM events ORDER BY id').fetchall()
        assert [e['kind'] for e in events] == ['next','offer']
        assert all(e['t'] == 100. for e in events)
        assert json.loads(events[-1]['info'])['accuracy'] == .94
        assert messages[-1]['copy'] is None and messages[-1]['selection_id'] == 1
        assert [x['sha'] for x in worker.playlists['nps']] == ['b','c']
        shown=worker._prepared_playlist('nps',[a,c,b])
        assert [x['sha'] for x in shown] == ['b','c']
    worker.db.close()


def test_prepared_local_queue_rechecks_readiness_after_prediction_refresh():
    worker = R.Worker(lambda msg: None)
    worker.rec = SimpleNamespace(local_shas={'hard','ready','easy','cold','new'},
        predictor=SimpleNamespace(predict=lambda f,*a,**k:f['expected']))
    session=SimpleNamespace(warm_flag=False)
    rows=[dict(sha=sha,keys=7,rate=1.1,var='1.10x',purpose='warmup',installed=True,
               practice_skills=(),f=dict(keys=7,expected=dict(acc_mid=acc,opening_acc=opening,
                                                             activation_minutes=0.,warmup_tau=6.)))
          for sha,acc,opening in (('hard',.86,.86),('ready',.94,.94),('easy',.99,.99),('cold',.94,.90),('new',.94,.94))]
    with patch.object(worker,'_session',return_value=session):
        for mode in ('nps','skills'):
            worker.playlists[mode]=rows[:4]
            shown=worker._prepared_playlist(mode,[rows[-1]])
            assert [c['sha'] for c in shown]==['ready','new']
            assert shown[0]['rate']==1.1 and shown[0]['acc_mid']==.94
        # PP owns a different eligibility policy; it must not inherit the
        # local accuracy band's upper/lower limits.
        worker.playlists['pp']=rows[:4]
        assert len(worker._prepared_playlist('pp',[]))==4


def test_prepared_queue_updates_the_same_chart_rate_before_dropping_it():
    worker = R.Worker(lambda msg: None)
    worker.rec = SimpleNamespace(local_shas={'mid','high'},
        predictor=SimpleNamespace(predict=lambda f,*a,**k:f['expected']))
    def row(sha, rate, acc, density):
        return dict(sha=sha, keys=7, rate=rate, var=nps.rate_label(rate), purpose='nps',
                    installed=True, nps=density, length=180./rate, acc_mid=acc,
                    f=dict(keys=7, expected=dict(acc_mid=acc, opening_acc=acc,
                                               activation_minutes=10., warmup_tau=6.)))
    mid, high = row('mid', 1., .967, 22.), row('high', 1., .94, 34.)
    faster = row('mid', 1.2, .941, 26.4)
    worker.playlists['nps'] = [mid, high]
    # The suitable version may be outside the fresh 20-map preview.
    worker._local_pool_cache['nps'] = (None, None, [high, faster])
    with patch.object(worker, '_session', return_value=SimpleNamespace(warm_flag=True)):
        shown = worker._prepared_playlist('nps', [high])
    assert [c['sha'] for c in shown] == ['mid', 'high']
    assert shown[0]['rate'] == 1.2 and shown[0]['nps'] == 26.4 and shown[0]['acc_mid'] == .941
    assert mid['rate'] == 1. and mid['acc_mid'] == .967


def test_nps_focus_changes_selection_without_refitting_or_changing_rate_targets(tmp_path):
    import pp_playlist
    rows = [dict(keys=7, sha=str(i), md5=str(i), family=str(i), rate=1.1,
                 nps=ns, base_value=math.log(ns), length=180., profile=[1.],
                 description='Mixed', group='chords', acc_mid=.94)
            for i, ns in enumerate([22.]*80 + [34.]*20)]
    session = R.Session([], 1000)
    weighted = nps.weighted_candidates(rows, session, [7])
    high = pp_playlist.preview(weighted, 20, 9, focus=1.)
    assert all(c['nps'] == 34. for c in high)
    variety = pp_playlist.preview(weighted, 20, 9, focus=0.)
    assert any(c['nps'] == 22. for c in variety) and any(c['nps'] == 34. for c in variety)
    assert variety == pp_playlist.preview(weighted, 20, 9, focus=0.)
    assert all(c['rate'] == 1.1 and c['acc_mid'] == .94 for c in variety + high)
    assert len({c['family'] for c in variety}) == 20
    worker = R.Worker(lambda msg: None)
    worker.db = recdata.connect(str(tmp_path/'rec.db'))
    worker.mode = 'nps'
    worker.playlists['nps'] = [rows[0]]
    worker.playlists['pp'] = [rows[1]]
    with patch.object(worker, '_restore_local'), patch.object(worker, '_request_nps') as build, \
            patch.object(worker, '_publish') as publish:
        worker.t_configure('nps', {'pp':None, 'nps':[7], 'skills':[7]}, revision=1, nps_focus=1.)
        assert worker.nps_focus == 1. and worker.playlists['nps'] == []
        assert worker.playlists['pp'] == [rows[1]]
        publish.assert_called_once(); build.assert_not_called()
        worker.t_configure('nps', {'pp':None, 'nps':[7], 'skills':[7]}, revision=0, nps_focus=0.)
        assert worker.nps_focus == 1.                    # a superseded slider event cannot win
        worker.playlists['skills'] = [rows[0]]
        worker.t_configure('nps', {'pp':None, 'nps':[7], 'skills':[7]}, revision=2, web_bias={'skills': 0.})
        assert worker.web_bias == {'nps': .5, 'skills': 0.} and worker.playlists['skills'] == []
    for value in (None, 'invalid', float('nan'), float('inf')):
        assert nps.normalize_focus(value) == .5
    assert nps.normalize_focus(-1) == 0. and nps.normalize_focus(2) == 1.
    worker.db.close()


def test_website_charts_join_nps_as_downloads_with_a_mild_popularity_prior(tmp_path, monkeypatch):
    import pickle
    import webmaps
    monkeypatch.setattr(webmaps, 'WEB_DIR', str(tmp_path))
    (tmp_path/'osu').mkdir()
    maps = {}
    for sha, md5, keys in (('a', 'm-a', 7), ('b', 'm-b', 7), ('c', 'm-c', 4)):
        (tmp_path/'osu'/f'{sha}.osu').write_text('x')
        maps[sha] = dict(bid=1, set_id=1, status=-2, keys=keys, file_md5=md5, playcount=10,
                         artist='A', title='T', version='V', creator='C', overall={1.: 6., 1.2: 9.})
    calc = recdata.calc_id()
    with open(tmp_path/'web.pkl', 'wb') as fh:
        pickle.dump({'calc': calc, 'format': 3, 'maps': maps,
                     'feats': {s: {1.: pickle.dumps({'x': 1}), 1.2: pickle.dumps({'x': 2})} for s in maps}}, fh)
    web, feats = webmaps.catalog({7}, installed_md5={'m-b'})     # installed copy wins
    assert set(web) == {'a'} and not web['a']['installed']
    assert feats['a'][1.] == {'x': 1} and dict(feats['a'].items()) == {1.: {'x': 1}, 1.2: {'x': 2}}
    assert feats['a'].setdefault(1., 'other') == {'x': 1} and feats['a'].get(1.1) is None
    # Out-of-reach rates are never loaded; a chart with none left is dropped.
    assert webmaps.catalog({7}, (), {7: (5., 7.)})[1]['b'].values() == [{'x': 1}]
    assert webmaps.catalog({7}, (), {7: (10., 12.)}) == ({}, {})
    # A window between two rated rates keeps both: they anchor the exact-rate refinement.
    assert dict(webmaps.catalog({7}, (), {7: (7., 8.)})[1]['a'].items()) == {1.: {'x': 1}, 1.2: {'x': 2}}
    # Many in-window rates: only the two nearest the window's centre are predicted.
    maps['a']['overall'] = {.8: 4., 1.: 6., 1.2: 9., 1.4: 11.}
    with open(tmp_path/'web.pkl', 'wb') as fh:
        pickle.dump({'calc': calc, 'format': 3, 'maps': maps, 'feats': {s: {r: pickle.dumps({'x': r}) for r in
                     maps[s]['overall']} for s in maps}}, fh)
    os.utime(tmp_path/'web.pkl', (2, 2))
    assert sorted(webmaps.catalog({7}, (), {7: (3.5, 12.)})[1]['a']) == [1., 1.2]
    assert len(webmaps.catalog({7})[1]['a']) == 4
    # A catalogue from another calculator keeps its features until re-published (no on-device analysis).
    with open(tmp_path/'web.pkl', 'wb') as fh:
        pickle.dump({'calc': 'older', 'format': 3, 'maps': maps,
                     'feats': {s: {1.: pickle.dumps({'x': 1})} for s in maps}}, fh)
    os.utime(tmp_path/'web.pkl', (1, 1))
    web, feats = webmaps.catalog({7})
    assert set(web) == set(feats) == {'a', 'b'}
    rows = [dict(keys=7, sha=s, md5=s, family=s, rate=1., nps=30., base_value=0., length=180., profile=[1.],
                 description='Mixed', group='chords', acc_mid=.94, installed=inst, playcount=pc)
            for s, inst, pc in (('local', True, None), ('rare', False, 10), ('popular', False, 20000))]
    w = {c['sha']: c['weight'] for c in nps.weighted_candidates(rows, R.Session([], 1000), [7])}
    assert w['popular'] == pytest.approx(w['local']) and .65 < w['rare']/w['local'] < .75
    # Settings slider: left end drops maps you don't have, the right end makes them ×4 as likely.
    assert {c['sha'] for c in nps.weighted_candidates(rows, R.Session([], 1000), [7], web=nps.web_factor(0.))} == {'local'}
    more = {c['sha']: c['weight'] for c in nps.weighted_candidates(rows, R.Session([], 1000), [7], web=nps.web_factor(1.))}
    assert more['popular'] / more['local'] == pytest.approx(4.) and nps.web_factor(.5) == 1.


def test_website_star_filter_keeps_unrated_charts():
    import webmaps
    rated = lambda s: {"stars": s}
    assert webmaps.in_stars(rated(7.), [6., 12.]) and webmaps.in_stars(rated(None), [6., 12.])
    assert not webmaps.in_stars(rated(5.9), [6., 12.]) and not webmaps.in_stars(rated(12.1), [6., 12.])
    assert webmaps.in_stars(rated(2.), None)


def test_website_crawl_estimates_before_start_pauses_on_request_and_resumes(tmp_path, monkeypatch):
    import webmaps
    monkeypatch.setattr(webmaps, 'WEB_DIR', str(tmp_path))
    monkeypatch.setattr(webmaps, 'FILES', (('{bid}', 0.),))
    db = webmaps._db()
    with db:
        for bid in range(1, 5):
            db.execute("INSERT INTO diffs(bid,keys,md5,notes,length,playcount,stars) VALUES(?,?,?,?,?,?,?)",
                       (bid, 7, f'm{bid}', 2000, 120, bid, 8. if bid < 4 else 3.))
    assert webmaps.estimate(db, {7}, [6., 12.])['unlisted'] == []     # older crawls: their rows' keymodes
    webmaps._kv(db, 'keys', [7])                     # what a listing pass records
    est = webmaps.estimate(db, {4, 7}, [6., 12.])   # chart 4 is outside the ★ range
    assert (est['fetch'], est['analyse'], est['ready'], est['unlisted']) == (3, 3, 0, [4])
    assert est['new_bytes'] == 3 * webmaps.CHART_BYTES and est['seconds'] > 0
    assert webmaps.estimate(db, {4}, [6., 12.])['fetch'] == 0
    def get(url):
        (tmp_path/'stop').touch()                    # the app pressed Pause during the first download
        return b'chart ' + url.encode()
    monkeypatch.setattr(webmaps, '_get', get)
    with pytest.raises(webmaps.Stopped):
        webmaps.fetch(db, {7}, None, [6., 12.])
    assert webmaps._kv(db, 'progress')['fetched'] == 1             # stored one, then paused
    assert webmaps.estimate(db, {7}, [6., 12.])['fetch'] == 2
    (tmp_path/'stop').unlink()
    monkeypatch.setattr(webmaps, '_get', lambda url: b'chart ' + url.encode())
    assert webmaps.fetch(db, {7}, None, [6., 12.]) == 2          # resume fetches only the rest
    assert webmaps.estimate(db, {7}, [6., 12.])['fetch'] == 0
    lock = webmaps._lock()                                         # a crawl holds it while it runs
    assert webmaps.running() and webmaps._lock() is None
    lock.close()
    assert not webmaps.running()


def test_first_real_build_replaces_the_installed_only_restart_list(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import patch
    import recommend as R
    w = R.Worker(lambda m: None)
    w.db = recdata.connect(str(tmp_path / "rec.db"))
    w.rec = SimpleNamespace(feats=SimpleNamespace(cache={}, key=lambda s, r: (s, r), calc="x"))
    w.nps_state = {"candidates": [], "restored": True}
    w.playlists["nps"] = [{"sha": "old"}]
    data = {"generation": w.generation, "revision": w._local_request_id, "selected_skills": w.practice_skills,
            "keys_by_mode": {m: w.keys[m] for m in R.LOCAL_MODES},
            "states": {"nps": {"candidates": [], "total": 0, "analyzed": 0, "failed": 0, "error": ""}}}
    with patch.object(w, "_publish"):
        w.t_nps_update("ready", data)
        assert w.playlists["nps"] == []
        w.playlists["nps"] = [{"sha": "kept"}]
        w.t_nps_update("ready", data)          # later builds keep the displayed order
        assert w.playlists["nps"] == [{"sha": "kept"}]
    w.db.close()
