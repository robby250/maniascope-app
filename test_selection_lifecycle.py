"""Delayed selection callbacks may fail, but must not destroy the GTK viewer."""
import os
import tempfile
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch


def test_late_callbacks_do_not_touch_closed_widgets_or_score_store():
    import skillsets as ui
    closed=SimpleNamespace(_closed=True)
    assert ui.ManiaScopeWindow._on_status(closed,'late') is False
    assert ui.ManiaScopeWindow._on_obs(closed,{'path':'late'}) is False
    assert ui.ManiaScopeWindow._on_rec(closed,{'type':'status','text':'late'}) is False
    assert ui.ManiaScopeWindow._on_progress(closed,12000) is False
    assert ui.ManiaScopeWindow._on_score(closed,{},'late') is False
    assert ui.ManiaScopeWindow._on_result(closed,(1,'late',1.,None,None,'error',True)) is False


def test_identical_observations_do_not_invalidate_their_pending_analysis(tmp_path):
    import skillsets as ui
    path=tmp_path/'map.osu';path.write_text('fixture')
    win=SimpleNamespace(_closed=False,_auto=False,_local=None,_shown=None,_requested=None,
                        _req=None,_gen=0,_req_cv=threading.Condition(),follow_btn=Mock(),
                        _effective=Mock(return_value=(str(path),1.,True)),_sv=lambda:True,
                        _update_status=Mock(),_render_pending=Mock())
    ui.ManiaScopeWindow._refresh(win)
    first=win._req
    for _ in range(10):ui.ManiaScopeWindow._refresh(win)
    assert win._gen==1 and win._req==first
    win._effective.return_value=(str(path),.98,True)
    ui.ManiaScopeWindow._refresh(win)
    assert win._gen==2 and win._req[2]==.98


def test_settled_selection_survives_delayed_work_and_closes_cleanly(tmp_path):
    """An isolated real Python chart analysis, no game or personal data writes."""
    import skillsets as ui
    from gi.repository import Gtk
    from test_skill_calc import chart
    import analysis_cache
    class Idle:
        def __init__(self,*a,**kw):pass
        def put(self,*a):pass
        def start(self):pass
        def stop(self):pass
    file=chart([(1000+i*60,i%7) for i in range(800)],keys=7)
    with patch.object(ui,'UI_FILE',str(tmp_path/'prefs.json')), \
         patch.object(ui.recdata,'DB_FILE',str(tmp_path/'scores.db')), \
         patch.object(analysis_cache,'selected',analysis_cache.AnalysisCache(tmp_path/'analysis')), \
         patch.object(ui.recommend,'Worker',Idle),patch.object(ui,'LazerWatcher',Idle), \
         patch.object(ui.ManiaScopeWindow,'_start_tosu',lambda *_:None):
        win=ui.ManiaScopeWindow();win.show_all()
        win._on_obs({'path':file,'rate':1.,'mods':[],'source':'tosu','note':None})
        start=time.monotonic()
        while time.monotonic()-start<12:
            while Gtk.events_pending():Gtk.main_iteration_do(False)
            if int(time.monotonic()-start)%2==0:win._refresh()
            assert not win._closed and win.get_visible()
            time.sleep(.02)
        assert win._shown and win._shown[4] is None
        result=(win._gen,file,1.,None,None,'late error',True)
        win.destroy()
        win._on_result(result);win._on_status('late status')
        while Gtk.events_pending():Gtk.main_iteration_do(False)
        assert win._closed


def test_return_to_displayed_chart_invalidates_delayed_other_chart(tmp_path):
    import skillsets as ui
    first=tmp_path/'a.osu';second=tmp_path/'b.osu'
    first.write_text('a');second.write_text('b')
    win=SimpleNamespace(_closed=False,_auto=False,_local=None,
                        _shown=(str(first),1.,None,None,None,True),
                        _requested=None,_req=None,_gen=0,_req_cv=threading.Condition(),
                        follow_btn=Mock(),_effective=Mock(return_value=(str(second),1.,True)),
                        _sv=lambda:True,_update_status=Mock(),_render=Mock(),_render_pending=Mock())
    ui.ManiaScopeWindow._refresh(win)
    generation=win._gen
    win._effective.return_value=(str(first),1.,True)
    ui.ManiaScopeWindow._refresh(win)
    assert win._gen>generation and win._req is None and win._requested is None
    ui.ManiaScopeWindow._on_result(win,(generation,str(second),1.,None,None,'late',True))
    assert win._shown[0]==str(first)


def test_cached_selection_renders_in_the_observation_and_invalidates_old_work(tmp_path):
    import skillsets as ui
    import analysis_cache
    path = tmp_path/'map.osu'
    path.write_text('fixture')
    win = SimpleNamespace(_closed=False, _auto=False, _local=None, _shown=None,
                          _requested=('old',0,1.,True), _req=(9,'old',1.,True), _gen=9,
                          _req_cv=threading.Condition(), follow_btn=Mock(),
                          _effective=Mock(return_value=(str(path),1.060000001,True)),
                          _sv=lambda:True, _update_status=Mock(), _render=Mock())
    win._on_result = lambda out: ui.ManiaScopeWindow._on_result(win, out)
    with patch.object(analysis_cache.selected, 'get_cached', return_value=('chart',{'cached':True})) as lookup:
        ui.ManiaScopeWindow._refresh(win)
        lookup.assert_called_once_with(str(path),1.06,True)
    assert win._req is None and win._requested is None and win._gen == 10
    assert win._shown[2:5] == ('chart',{'cached':True},None)
    win._render.assert_called_once()
    win._on_result((9,'old',1.,None,None,'stale',True))
    assert win._shown[0] == str(path)


def test_a_new_selection_shows_its_title_and_calculating_until_its_analysis_lands(tmp_path):
    """Long maps: the header switches at once; the previous map's numbers never stand for the new one."""
    import skillsets as ui
    from test_skill_calc import chart
    import analysis_cache
    class Idle:
        def __init__(self,*a,**kw):pass
        def put(self,*a):pass
        def start(self):pass
        def stop(self):pass
    file=chart([(1000+i*60,i%7) for i in range(200)],keys=7)
    with patch.object(ui,'UI_FILE',str(tmp_path/'prefs.json')), \
         patch.object(ui.recdata,'DB_FILE',str(tmp_path/'scores.db')), \
         patch.object(analysis_cache,'selected',analysis_cache.AnalysisCache(tmp_path/'analysis')), \
         patch.object(ui.recommend,'Worker',Idle),patch.object(ui,'LazerWatcher',Idle), \
         patch.object(ui.ManiaScopeWindow,'_start_tosu',lambda *_:None), \
         patch.object(ui.ManiaScopeWindow,'_worker',lambda *_:None):
        win=ui.ManiaScopeWindow()
        win._on_obs({'path':file,'rate':1.,'mods':[],'source':'tosu','note':None})
        assert win.dominant_lbl.get_text()=='Calculating…'
        assert win.context_lbl.get_text().startswith('7K')
        assert win._span() is None
        gen,path,rate,sv=win._req
        chart_,res=ui.analyze(path,rate,sv)
        win._on_result((gen,path,rate,chart_,res,None,sv))
        assert win.dominant_lbl.get_text()!='Calculating…' and not win._pending
        win.destroy()
