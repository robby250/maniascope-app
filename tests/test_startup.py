"""Restored playlists must not wait for imports or a full catalogue rebuild."""
import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import nps
import recdata
import recommend as R


def test_prior_calculator_shortlist_uses_only_current_features_and_predictions(tmp_path):
    from test_skill_practice import feature, inst, Predictor
    db=recdata.connect(str(tmp_path/'restore.db'))
    worker=R.Worker(lambda m:None, busy=lambda:True)
    worker.db=db
    source=R.FeatureSource(db,{'maps':{}})
    worker.rec=SimpleNamespace(feats=source, predictor=Predictor(),
                              installed={s:inst(s) for s in ('ready','old-only','now-too-hard')})
    with db:
        for sha,calc,acc in (('ready',source.calc,.94),('old-only','previous',.94),
                             ('now-too-hard',source.calc,.8),('now-too-hard','previous',.94)):
            db.execute('INSERT INTO feats VALUES (?,?)',
                       (f'{sha}@0.950|{calc}',json.dumps(feature({'ln':1.},acc=acc,rate=.95))))
    session=R.Session([],1000)
    with patch.object(worker,'_session',return_value=session):
        for mode in R.LOCAL_MODES:
            recdata.kv_set(db,'playlist_cache:'+mode,{'version':1,'calc':'previous',
                'choices':[['ready',.95],['old-only',.95],['now-too-hard',.95]],'analyzed':3})
            worker._restore_local(mode)
            state=getattr(worker,mode+'_state')
            assert state['restored']
            assert [(c['sha'],c['rate'],c['acc_mid']) for c in state['candidates']]==[('ready',.95,.94)]
            assert all(key.endswith('|'+source.calc) for key in source.cache)
    db.close()


def test_saved_pool_is_published_before_background_import_finishes(tmp_path):
    path=str(tmp_path/'rec.db')
    ready=threading.Event();entered=threading.Event();release=threading.Event()
    def slow_import(db):
        entered.set();release.wait(3)
        return {'new':0,'seen':0,'installed':0}
    class Builder:
        def __init__(self,*a):pass
        def start(self):pass
        def stop(self):pass
    worker=R.Worker(lambda m:None)
    with patch.object(recdata,'DB_FILE',path),patch.object(recdata,'import_realm',side_effect=slow_import), \
         patch.object(recdata,'load_public',return_value={'fixture':True}), \
         patch.object(R.gc,'collect') as collect,patch.object(R.gc,'freeze') as freeze, \
         patch.object(R,'Recommender',return_value=SimpleNamespace()),patch.object(nps,'Builder',Builder), \
         patch.object(worker,'_publish',side_effect=lambda *a:ready.set()), \
         patch.object(worker,'_request_nps'),patch.object(worker,'_restore_local'),patch.object(worker,'_fill'):
        worker.start()
        try:
            assert ready.wait(2) and entered.wait(2)
            collect.assert_called_once()
            freeze.assert_called_once()
            assert worker._import_thread.is_alive()
            # The worker still accepts a UI/history task while import is held.
            handled=threading.Event()
            with patch.object(worker,'t_history',side_effect=handled.set):
                worker.put('history')
                assert handled.wait(1)
        finally:
            release.set();worker._import_thread.join(2)
            worker.stop();worker.put('history');worker.join(2)


def test_progress_counts_unanalysable_charts_as_done():
    out = []
    worker = R.Worker(out.append)
    worker.rec = SimpleNamespace(rows=[0]*30)
    worker.mode = "nps"
    with patch.object(worker, "_pool", return_value=("", [], "")), \
         patch.object(worker, "_prepared_playlist", return_value=[]), patch.object(R, "summary", return_value={}):
        for failed, want in ((1, None), (0, (9, 10))):
            worker.nps_state = dict(worker.nps_state, total=10, analyzed=9, failed=failed)
            worker._publish()
            assert out[-1]["progress"] == want
