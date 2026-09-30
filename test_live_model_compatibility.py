"""Source/bundle sync can leave an old viewer running without mixing models."""
import json
import pickle
from unittest.mock import patch

import recdata
import selected_analysis


def test_matching_public_sibling_is_selected_without_overwriting_old_bundle(tmp_path):
    old='c1021be2829b';new='123456789abc';name='fixture'
    active=tmp_path/'active.json';active.write_text(json.dumps({'name':name}))
    folder=tmp_path/name;folder.mkdir()
    primary=folder/'public.pkl'
    primary.write_bytes(pickle.dumps({'calc':old,'snapshot':name,'sentinel':'old'}))
    version=folder/'calculators'/new;version.mkdir(parents=True)
    (version/'public.pkl').write_bytes(pickle.dumps({'calc':new,'snapshot':name,'sentinel':'new'}))
    original=primary.read_bytes()
    with patch.object(recdata,'ACTIVE',str(active)),patch.object(recdata,'ROLLING',str(tmp_path)):
        with patch.object(recdata,'calc_id',return_value=new):
            assert recdata.load_public()['sentinel']=='new'
        with patch.object(recdata,'calc_id',return_value=old):
            assert recdata.load_public()['sentinel']=='old'
    assert primary.read_bytes()==original


def test_archived_helper_requires_exact_safe_identity(tmp_path):
    cid='c1021be2829b';folder=tmp_path/cid;folder.mkdir()
    helper=folder/'selected_analysis.py';helper.write_text('# fixture')
    assert selected_analysis.previous_helper(cid,tmp_path)==helper
    for invalid in ('../'+cid,'','ffffffffffff',None,'NOT-A-CALCULATOR'):
        assert selected_analysis.previous_helper(invalid,tmp_path) is None


def test_spawned_background_workers_keep_the_old_callers_calculator():
    import nps
    import recommend
    old = 'c1021be2829b'
    with patch.object(recdata,'calc_id',return_value='123456789abc'), \
         patch.object(recdata,'local_file',return_value='/fixture/chart.osu'), \
         patch.object(recommend,'chart_path',return_value='/fixture/chart.osu'), \
         patch.object(selected_analysis,'previous_helper',return_value='/fixture/old/selected_analysis.py'), \
         patch.object(selected_analysis,'calculate',side_effect=lambda p,r,c:{r:{'calculator':c}}) as calculate, \
         patch.object(recdata,'chart_feats') as current:
        sha, features, error = nps._job({'sha256':'a'*64},[.9,1.],{},old)
        assert error is None and set(features) == {.9,1.}
        assert features[.9]['calculator'] == old and not current.called
        sha, features = recommend._feat_job(('a'*64,[1.2],old))
        assert features[1.2]['calculator'] == old and calculate.call_count == 3


def test_preserved_runtime_survives_a_source_update(tmp_path):
    import os
    import shutil
    from test_skill_calc import chart
    cid = recdata.calc_id()
    saved = selected_analysis.preserve_runtime(tmp_path/'runtime')
    assert saved.name == cid and selected_analysis.previous_helper(cid,tmp_path/'runtime')
    assert selected_analysis.preserve_runtime(tmp_path/'runtime') == saved
    current = tmp_path/'updated'
    shutil.copytree(saved,current)
    # Simulate sync after the viewer loaded its calculator.
    with (current/'skill_calc.py').open('a') as fh: fh.write('\n# later deployment\n')
    path = chart([(1000+80*i,i%4) for i in range(200)],keys=4)
    with patch.object(selected_analysis,'__file__',str(current/'selected_analysis.py')), \
         patch.dict(os.environ,MANIASCOPE_REC=str(tmp_path),MANIASCOPE_ANALYSIS_CACHE=str(tmp_path/'cache')):
        features = selected_analysis.calculate(path,1.25,cid)
    assert features[1.25]['keys'] == 4 and features[1.25]['notes'] == 200
    assert (saved/'skill_calc.py').read_bytes() != (current/'skill_calc.py').read_bytes()


def test_selected_card_distinguishes_rate_analysis_and_failure(tmp_path):
    from types import SimpleNamespace
    import hashlib
    import recommend as R
    chart = tmp_path/'chart.osu';chart.write_text('fixture')
    sha = hashlib.sha256(chart.read_bytes()).hexdigest()
    messages=[];w=R.Worker(messages.append)
    w.rec=SimpleNamespace(feats=SimpleNamespace(calc='fixture'),card=lambda *a,**k:None)
    w._sel=(str(chart),1.25,[{'acronym':'DT','settings':{'speed_change':1.25}}])
    w._card()
    assert messages[-1]['note']=='Calculating prediction at 1.25×…'
    w._feature_errors[sha,1.25,'fixture']=(999.,'helper failed')
    w._card()
    assert messages[-1]['note']=='Prediction unavailable at 1.25×: helper failed'
    w._sel=(str(chart),None,None);w._card()
    assert 'Waiting for osu!' in messages[-1]['note']
    w._sel=(str(chart),1.25,[{'acronym':'CS'}]);w._card()
    assert messages[-1]['note']=='Prediction unavailable for these mods'


def test_updated_map_keeps_offer_link_without_reusing_old_revision(tmp_path):
    from types import SimpleNamespace
    import hashlib
    import recommend as R
    import navigation
    path=tmp_path/'chart.osu'
    path.write_text('osu file format v14\n[Metadata]\nArtist:Artist\nTitle:Updated\nVersion:Hard\nBeatmapID:1123210\n')
    sha=hashlib.sha256(path.read_bytes()).hexdigest();md5=hashlib.md5(path.read_bytes()).hexdigest()
    out=[];w=R.Worker(out.append);w.db=recdata.connect(str(tmp_path/'rec.db'))
    old=dict(sha='a'*64,md5='b'*32,bid=1123210,rate=1.25,var='DT 1.25×',purpose='nps',mode='nps',offer_id=42,acc_mid=.94)
    w.target=old
    f=dict(keys=7,overall=8.)
    w.rec=SimpleNamespace(local_shas={old['sha']},pub=dict(calc='test',snapshot='test'),
                          expectation=lambda *a,**kw:dict(f=f,acc_mid=.95,md5=md5))
    mods=[{'acronym':'DT','settings':{'speed_change':1.25}}]
    with patch.object(w,'_card'),patch.object(w,'_request_feature') as requested,patch.object(w,'_begin_import') as imported:
        w.t_selected(str(path),1.25,mods)
        requested.assert_called_once_with(sha,str(path),1.25)
        imported.assert_called_once()
    assert 'updated chart selected' in out[-1]['msg']
    selected=w.db.execute("SELECT beatmap,info FROM events WHERE kind='selected'").fetchone()
    assert selected['beatmap']=='md5:'+md5
    assert json.loads(selected['info'])['offered_revision']=='md5:'+old['md5']
    w.t_start(dict(sha=sha,path=str(path),rate=1.25,mods=mods,t=100.))
    info=json.loads(w.db.execute("SELECT info FROM events WHERE kind='start'").fetchone()[0])
    assert info['offer_id']==42 and info['purpose']=='nps'
    assert info['sha']==sha and info['md5']==md5
    assert info['expected']==.95 and info['offered_accuracy']==.94
    assert old['sha'] != sha and old['md5'] != md5
    assert not navigation.same_map(old,sha='different',bid=-1)
    assert not navigation.same_map(old,sha='different',bid=123)
    w.db.close()
