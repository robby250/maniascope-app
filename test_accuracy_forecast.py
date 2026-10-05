"""Saved intervals copy prepared native305 state without changing prediction work."""
import copy
import json
import math
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import accuracy_targets as A
import nps
import pp_playlist
import recommend as R
from test_prediction_targets import model
from test_skill_practice import feature


SHA, MD5 = 'a'*64, 'b'*32


def recommender(native=.98):
    rec = R.Recommender.__new__(R.Recommender)
    rec.model = model(math.log(.08))
    rec.model.now = 0.
    rec.accuracy_model = model(math.log(1-native))
    for fitted in (rec.model, rec.accuracy_model):
        fitted.rate_response = lambda *_: 3.
        fitted.rate_loss = lambda *args: 3.*args[-1]
        fitted.predict = Mock(wraps=fitted.predict)
    f = dict(feature({'ln':1.}), stars=6., hits=1200)
    inst = dict(sha256=SHA, md5=MD5, keys=7, beatmap_id=42, status=1,
                title='Fixture', artist='Contract', version='7K', length=160000)
    rec.pub = dict(calc='old-calculator', maps={42:dict(md5=MD5, approved=1, set=1, file='fixture')},
                   feats={42:{1.:f}})
    rec.installed = {MD5:inst}
    rec.local_shas = {SHA}
    rec.feats = SimpleNamespace(calc='old-calculator', md5_bid={MD5:42})
    rec._shown = (0.,1.)
    rec.ledger = R.Ledger({})
    rec.newer_ranked = lambda: []
    rec.typical = lambda _k: math.log(.08)+.2
    rec.rows = [dict(src='live', keys=7, chart=MD5, rate=1., b=42, f=f, t=0.)]
    rec.predictor = R.Predictor(rec, source_id='old-predictor')
    return rec, f, inst


@pytest.mark.parametrize('mode', ['pp','nps','skills','warmup'])
def test_prepared_candidates_persist_their_original_native_state_without_open_work(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(R.time, 'time', lambda: 1000.)
    rec, f, inst = recommender(.94 if mode in R.LOCAL_MODES else .98)
    session = R.Session([], 1000.); session.warm_flag = True
    worker = R.Worker(lambda _message: None)
    worker.rec = rec
    if mode == 'pp':
        rec._candidates()
        scored, _ = rec.score(session, keys=[7])
        candidates, _ = pp_playlist.prepare(rec, scored, session)
        candidate = candidates[0]
        assert candidate['mu'] == pytest.approx(math.log(.08))
    elif mode == 'warmup':
        candidate = rec.warmups([], session, 7)[0]
    else:
        choices, _ = nps.candidates_from({SHA:inst}, {SHA:{1.:f}}, rec.predictor, session,
                                        mode=mode, selected_skills=('ln',) if mode=='skills' else ())
        worker.practice_skills = ('ln',) if mode=='skills' else ()
        candidate = worker._prepare_local(mode, dict(candidates=choices), session)[0]
    original = copy.deepcopy(candidate)
    state = candidate['accuracy_forecast']
    assert state['target'] == 'native_displayed_lazer' and state['kind'] == 'usual_range'
    assert (state['sha256'],state['md5'],state['rate']) == (SHA,MD5,1.)
    assert state['mid'] == candidate['acc_mid']
    assert state['mid'] != .92  # The PP mean must never masquerade as native305.
    assert state['sd'] == math.hypot(state['sdm'],state['sda'])
    assert state['lo'] < state['mid'] < state['hi']
    assert state['band_z'] == R.BAND_Z and state['generated_at'] == 1000.
    assert state['calculator'] == 'old-calculator' and state['predictor'] == 'old-predictor'
    before = (rec.model.predict.call_count, rec.accuracy_model.predict.call_count)

    worker.db = R.recdata.connect(str(tmp_path/'rec.db'))
    worker.rec = SimpleNamespace(feats=SimpleNamespace(calc='current-calculator'))
    worker.predictor_id = 'current-predictor'
    worker._publish = Mock()
    monkeypatch.setattr(R.time, 'time', lambda: 2000.)
    def forbidden(*_args, **_kwargs):
        raise AssertionError('Opening a prepared choice must not predict or calculate')
    monkeypatch.setattr(R.Predictor, 'predict', forbidden)
    monkeypatch.setattr(R.Recommender, 'score', forbidden)
    monkeypatch.setattr(R.recdata, 'chart_feats', forbidden)
    monkeypatch.setattr(R, 'predictor_id', forbidden)
    # An already performed UI choice survives a newer queued revision/refit.
    actual_mode = 'pp' if mode=='warmup' else mode
    worker.t_open(candidate, actual_mode, revision=-1,
                  result=dict(sent=False,msg='fixture',copy=None), selected_at=1990.)
    row = worker.db.execute("SELECT * FROM events WHERE kind='offer'").fetchone()
    saved = json.loads(row['info'])
    assert row['t'] == 1990. and saved['accuracy_forecast_recorded_at'] == 2000.
    assert saved['accuracy_forecast'] == state
    assert saved['calculator'] == 'old-calculator' and saved['predictor'] == 'old-predictor'
    assert saved['accuracy'] == original['acc_mid'] and candidate == original
    assert before == (rec.model.predict.call_count, rec.accuracy_model.predict.call_count)
    worker.db.close()


def test_pp_fallback_retains_its_numbers_without_inventing_native_state():
    rec, f, _ = recommender()
    rec.accuracy_model = None
    rec.predictor = R.Predictor(rec, source_id='fixture')
    prediction = rec.predictor.predict(f, MD5, 42, 1., R.Session([],1000))
    assert prediction['accuracy_forecast'] is None
    assert prediction['mu'] == pytest.approx(math.log(.08)) and prediction['acc_mid'] == pytest.approx(.92)
    rec._candidates()
    session = R.Session([],1000); session.warm_flag = True
    choices, _ = rec.score(session, keys=[7])
    assert choices and all(c['accuracy_forecast'] is None for c in choices)
    assert A.forecast_state(dict(mu=-3.,sdm=.1,sda=.2,sd=.3,acc_mid=.95,acc_lo=.9,acc_hi=.98),
                            generated_at=1000.,calculator='fixture',predictor='fixture',rate=1.,band_z=R.BAND_Z) is None


def test_actual_start_copies_existing_native_snapshot_and_actual_recording_time(tmp_path, monkeypatch):
    monkeypatch.setattr(R.time, 'time', lambda: 1000.)
    rec, f, _ = recommender()
    predicted = rec.predictor.predict(f, MD5, 42, 1., R.Session([],1000))
    worker = R.Worker(lambda _message: None)
    worker.db = R.recdata.connect(str(tmp_path/'rec.db'))
    with worker.db:
        worker.db.execute('INSERT INTO installed(sha256,md5,keys,length,title) VALUES (?,?,?,?,?)',
                          (SHA,MD5,7,160000,'Fixture'))
    expectation = Mock(return_value=dict(predicted,md5=MD5,keys=7,length=160.,f=f))
    worker.rec = SimpleNamespace(pub=rec.pub,expectation=expectation)
    monkeypatch.setattr(R.time, 'time', lambda: 1001.)
    worker.t_start(dict(t=999.,sha=SHA,path='/unused-fixture.osu',rate=1.,mods=[]))
    saved = json.loads(worker.db.execute("SELECT info FROM events WHERE kind='start'").fetchone()[0])
    assert expectation.call_count == 1
    assert saved['accuracy_forecast'] == A.forecast_chart(predicted['accuracy_forecast'],SHA,MD5)
    assert saved['calculator'] == 'old-calculator' and saved['predictor'] == 'old-predictor'
    assert saved['accuracy_forecast_recorded_at'] == 1001. and saved['predicted_at'] == 1001.
    assert saved['expected'] == predicted['acc_mid'] and saved['mu'] == predicted['mu']
    assert saved['expected_lo'] == predicted['acc_lo'] and saved['expected_hi'] == predicted['acc_hi']
    from calib.recommendation_audit import saved_interval
    row = dict(forecast_status='valid',calculator=saved['calculator'],predictor=saved['predictor'],
               sha256=SHA,md5=MD5,actual_rate=1.,expected_lazer=saved['expected'],
               expected_lo=saved['expected_lo'],expected_hi=saved['expected_hi'],
               start_t=999.,finish_t=1160.,played_timestamp=1160.)
    verified = saved_interval(row,saved,1200.)
    assert verified['interval_status'] == 'verified' and verified['interval_timing'] == 'during_play'
    worker.db.close()


def test_chart_identity_only_fills_missing_values_and_rejects_conflicts():
    state = dict(sha256=SHA,md5=MD5,predictor='old',generated_at=1000.)
    original = dict(state)
    assert A.forecast_chart(state,None,None) == original
    assert A.forecast_chart(dict(state,sha256=None),SHA,MD5) == original
    assert A.forecast_chart(state,'c'*64,MD5) is None
    assert A.forecast_chart(state,SHA,'d'*32) is None
    assert state == original


def test_snapshot_metadata_handles_a_non_string_chart_without_new_owner_errors():
    rec, f, _ = recommender()
    predictor = rec.predictor
    session = R.Session([],1000)
    pp = predictor._predict(f,MD5,42,1.,session)
    display = predictor.display._predict(f,MD5,42,1.,session.for_accuracy(rec.accuracy_model.warmup))
    predictor._predict = Mock(return_value=pp)
    predictor.display._predict = Mock(return_value=display)
    result = predictor.predict(f,None,None,1.,session)
    assert result['accuracy_forecast']['sha256'] is None and result['accuracy_forecast']['md5'] is None


def test_snapshot_copies_leave_numeric_leaves_and_pp_order_exact(monkeypatch):
    rec, f, inst = recommender()
    for bid, stars in ((43,7.),(44,5.)):
        md5, sha = f'{bid:032x}', f'{bid:064x}'
        rec.pub['maps'][bid] = dict(md5=md5,approved=1,set=bid,file=str(bid))
        rec.pub['feats'][bid] = {1.:dict(f,stars=stars)}
        rec.installed[md5] = dict(inst,sha256=sha,md5=md5,beatmap_id=bid)
    rec._candidates()
    def score():
        rec._score_cache.clear()
        session = R.Session([],1000); session.warm_flag = True
        return rec.score(session,keys=[7])[0]
    with_snapshots = score()
    monkeypatch.setattr(A,'forecast_state',lambda *_args,**_kwargs: None)
    without_snapshots = score()
    assert len(with_snapshots) == len(without_snapshots) == 3
    strip = lambda rows: [{k:v for k,v in row.items() if k!='accuracy_forecast'} for row in rows]
    assert strip(with_snapshots) == strip(without_snapshots)
    assert [r['bid'] for r in with_snapshots] == [r['bid'] for r in without_snapshots]
