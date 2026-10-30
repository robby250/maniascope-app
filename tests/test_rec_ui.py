"""Isolated GTK smoke tests: xvfb-run -a python3 -m pytest tests/test_rec_ui.py.

No running game, real score database, real clipboard, or user preferences touched.
"""
import json
import os
import tempfile
import time
from unittest.mock import patch


def test_tabs_keys_clipboard_and_persistence():
    with tempfile.TemporaryDirectory() as tmp:
        with patch.dict(os.environ, {"MANIASCOPE_REC": tmp}):
            import skillsets as ui
            from gi.repository import Gtk, GLib, Gdk
            class Worker:
                def __init__(self, *a, **kw): self.tasks = []
                def put(self, *task): self.tasks.append(task)
                def start(self): pass
                def stop(self): pass
            class Watcher(Worker): pass
            prefs = os.path.join(tmp, "ui.json")
            with open(prefs, "w") as f:
                json.dump({"recommendation_mode":"nps", "pp_keys":[4,6,9], "nps_keys":[7],
                           "skills_keys":[4,7], "practice_skills":["ln","sv"]}, f)
            with patch.object(ui, "UI_FILE", prefs), patch.object(ui.recdata, "DB_FILE", os.path.join(tmp,"rec.db")), \
                 patch.object(ui.recommend, "Worker", Worker), \
                 patch.object(ui, "LazerWatcher", Watcher), patch.object(ui.ManiaScopeWindow, "_start_tosu", lambda *_: None), \
                 patch.object(ui.ManiaScopeWindow, "_worker", lambda *_: None):
                w = ui.ManiaScopeWindow()
                w.show_all(); w.recs_btn.set_active(True)
                while Gtk.events_pending(): Gtk.main_iteration_do(False)
                assert w._rec_mode == "nps" and w._rec_notebook.get_current_page() == 1
                assert w.gear_btn.get_parent() is w._tab_labels[1] and w.gear_btn.get_visible()   # first open too
                assert w._rec_keys["pp"] == (4,6,9) and w._rec_keys["nps"] == (7,)
                assert w._rec_notebook.get_n_pages() == 4 and w._rec_keys['skills'] == (4,7)
                assert w._practice_skills == ('ln','sv')
                w._on_rec(dict(type='card',c=None,note='Calculating prediction at 1.25×…'))
                w._on_rec(dict(type='card',c=None,note='stale rate',path='/old/chart',rate=1.))
                assert w.card_lbl.get_text() == 'Calculating prediction at 1.25×…'

                c = {"title":"Cosmo — Cyber Shaman [Local difficulty]", "keys":7, "rate":.87,
                     "var":"HT 0.87×", "nps":27.2, "acc_mid":.94, "length":484,
                     "purpose":"nps", "description":"Chordstream Stamina", "why":"Synthetic UI fixture", "installed":True, "sha":"a"*64,
                     "search_meta":{"Artist":"Cosmo", "Title":"Cyber Shaman", "Version":"Local difficulty"}}
                local = os.path.join(tmp, "chart.osu")
                open(local, 'w').close()
                def pool(rev):
                    return {"type":"pool", "mode":"nps", "revision":rev, "selection_id":w._selection_id, "phase":"warmup", "shown":[c],
                            "note":"Synthetic UI check · target ~94%", "summary":"No actual prediction"}
                # A tab without its first list says it is loading, not that nothing fits.
                assert w._rec_views['pp']['placeholder'].get_text().startswith('Loading PP suggestions')
                w._on_rec(dict(type='status', text='Fitting…'))
                assert w._rec_views['pp']['placeholder'].get_text() == 'Loading PP suggestions…\n\nFitting…'
                w._on_rec(dict(pool(w._rec_revision), mode='pp', shown=[]))
                assert w._rec_views['pp']['placeholder'].get_text().startswith('Nothing to suggest yet.')
                assert not w.result_lbl.get_visible() and not w._rec_views['nps']['target'].get_visible()  # nothing reserved
                w._on_rec(pool(w._rec_revision - 1))
                assert not w.next_btn.get_sensitive()
                w._on_rec(pool(w._rec_revision))
                assert w.next_btn.get_sensitive() and w.next_btn.get_label() == "Next · NPS"
                # One status line and bar at the window bottom: the worker's progress first, else this tab's analysis.
                w._on_rec(dict(type='status', text='Analysing 11 new or updated maps in the background · 0 done',
                               progress=(30, 120), learning=True))
                w._on_rec(dict(pool(w._rec_revision), progress=(5, 50)))
                assert w.rec_status.get_text().count('Analysing') == 1 and abs(w.rec_progress.get_fraction() - .25) < 1e-9
                assert w.stats_view.wait.get_visible() and '30 / 120' in w.stats_view.wait.text.get_text()
                w._on_rec(dict(type='status', text='', learning=False))
                assert not w.stats_view.wait.get_visible()
                assert '5 / 50' in w.rec_status.get_text() and abs(w.rec_progress.get_fraction() - .1) < 1e-9
                w._on_rec(pool(w._rec_revision))
                assert not w.rec_status.get_visible() and not w.rec_progress.get_visible()
                q = 'artist="Cosmo"! title="Cyber Shaman"! diff="Local difficulty"! cs=7'
                # Fake worker never drains its queue: Next must still navigate.
                with patch.object(ui.recdata, 'local_file', return_value=local):
                    w._next(False)
                    assert w.rec.tasks[-1][:4] == ("next",False,"nps",w._rec_revision)
                    assert w._rec_views['nps']['target_data']['acc_mid'] == .94
                    assert Gtk.Clipboard.get(Gdk.SELECTION_CLIPBOARD).wait_for_text() == q
                    assert not w.next_btn.get_sensitive()
                    queued = len(w.rec.tasks)
                    w._next(False)
                    assert len(w.rec.tasks) == queued
                    # Completed refresh wins even when hover deferred painting it.
                    fresh = dict(c, sha='b'*64, acc_mid=.951)
                    w._rec_hover('nps', True)
                    w._on_rec(dict(pool(w._rec_revision), shown=[fresh, dict(c,sha='c'*64)]))
                    w._next(False)
                    assert w._rec_views['nps']['target_data']['acc_mid'] == .951
                    assert w.next_btn.get_sensitive()
                    # Recalculation still pending: consume the remaining snapshot — finishing the pick is Next.
                    w._on_rec(dict(type='result', title='t', text='x', targets=['pp']))   # another tab's pick
                    assert len(w.rec.tasks) == queued + 1
                    w._on_rec(dict(type='result', title='t', text='x', targets=['nps']))
                    assert w._rec_views['nps']['target_data']['sha'] == 'c'*64
                    assert len(w.rec.tasks) == queued + 2
                    w._on_rec(dict(pool(w._rec_revision), selection_id=1))
                    w._on_rec(dict(type='target',mode='nps',revision=w._rec_revision,
                                   selection_id=1,c=c,copy='STALE',msg='stale'))
                    assert not w.next_btn.get_sensitive()
                    assert w._rec_views['nps']['target_data']['sha'] == 'c'*64
                    assert Gtk.Clipboard.get(Gdk.SELECTION_CLIPBOARD).wait_for_text() == q
                w._on_rec({"type":"target", "mode":"pp", "revision":w._rec_revision-1,
                           "c":c, "copy":"SHOULD NOT COPY", "msg":"stale"})
                assert Gtk.Clipboard.get(Gdk.SELECTION_CLIPBOARD).wait_for_text() == q
                w._set_rec_keys("pp", [4,5,6,8,9])
                assert w._rec_keys["nps"] == (7,)
                w._set_rec_keys("nps", [])
                assert not w.next_btn.get_sensitive()
                w._set_rec_keys("nps", [7])
                w._on_rec(pool(w._rec_revision))
                w._set_next_action("copy")
                w._on_rec(pool(w._rec_revision))
                w._rec_hover("nps", False)
                assert len(w._rec_views["nps"]["list"].get_children()) == 1
                w._set_rec_mode('skills')
                assert w._rec_notebook.get_current_page() == 2 and w.next_btn.get_label() == 'Next · Skills'
                w._set_rec_keys('skills',[7])
                assert w._rec_keys['nps'] == (7,) and w._rec_keys['pp'] == (4,5,6,8,9)
                w._set_practice_skills(['ln','chordstream/jumptrill'])
                assert w._skill_checks['ln'].get_active() and w._skill_checks['jumptrill'].get_active()
                assert not w._skill_checks['chordstream'].get_active()
                assert not w._skill_checks['chordstream'].get_inconsistent()
                cskill=dict(c,mode='skills',purpose='skills',practice_skills=w._practice_skills,
                            acc_lo=.91,acc_hi=.97,description='LN Release / Jumptrill')
                pskill=dict(pool(w._rec_revision),mode='skills',shown=[cskill])
                w._on_rec(pskill)
                assert w.next_btn.get_sensitive()
                oldrow=w._rec_views['skills']['list'].get_children()[0]
                oldtasks=len(w.rec.tasks)
                w._set_practice_skills(['sv'])
                w._activate_rec('skills',oldrow)
                assert len(w.rec.tasks) == oldtasks+1 and w.rec.tasks[-1][0]=='configure'
                w._on_rec(pskill) # stale filter result must not repopulate
                assert not w._rec_views['skills']['list'].get_children()
                w._any_skill_btn.clicked()
                assert w._practice_skills == () and not any(b.get_active() for b in w._skill_checks.values())
                w._set_practice_skills(['ln','chordstream/jumptrill'])
                w._on_rec(dict(pskill,selection_id=w._selection_id,revision=w._rec_revision))
                with patch.object(ui.recdata, 'local_file', return_value=local):
                    w._next(False)
                assert w.rec.tasks[-1][:4] == ('next',False,'skills',w._rec_revision)
                # Stats is not a fourth recommendation mode and never changes
                # the Next action just because its overview is being viewed.
                w._rec_notebook.set_current_page(3)
                assert w.rec.tasks[-1] == ('stats',True)
                assert w._rec_mode=='skills'
                row={'value':8.,'low':7.5,'high':8.5,'current':7.8,'maps':40,'plays':90,
                     'confidence':'Strong history','activation':4.}
                groups={k:dict(row) for k in ('overall',*ui.skill_practice.NAMES)}
                w._on_rec({'type':'stats','data':{'pages':{7:{'groups':groups,'maps':120,'plays':400}},'phase':'build','target':.94}})
                w._practice_from_stats(7,'ln')
                assert w._rec_notebook.get_current_page()==2 and w._practice_skills==('ln',)
                w._set_practice_skills(['ln','chordstream/jumptrill'])
                w.resize(480,370);w.recs.resize(710,850)
                until = time.monotonic() + .25
                while time.monotonic() < until:
                    while Gtk.events_pending(): Gtk.main_iteration_do(False)
                    time.sleep(.01)
                while Gtk.events_pending(): Gtk.main_iteration_do(False)
                compact = tuple(w.get_size())
                # Details is allowed to use a larger temporary geometry, but
                # collapsing it must restore the compact size rather than
                # stretching the timeline into the freed space.
                w.details_btn.set_active(True)
                w.resize(compact[0], compact[1] + 260)
                until = time.monotonic() + .15
                while time.monotonic() < until:
                    while Gtk.events_pending(): Gtk.main_iteration_do(False)
                    time.sleep(.01)
                assert w._win_size == compact and w._details_size[1] > compact[1]
                w.details_btn.set_active(False)
                until = time.monotonic() + .30
                while time.monotonic() < until:
                    while Gtk.events_pending(): Gtk.main_iteration_do(False)
                    time.sleep(.01)
                assert abs(w.get_size()[0] - compact[0]) <= 2
                assert abs(w.get_size()[1] - compact[1]) <= 2
                # Accuracy target and display choices reach the worker and persist.
                w._acc_targets['skills'] = .925; w._sync_rec()
                assert w.rec.tasks[-1][0] == 'configure' and w.rec.tasks[-1][7] == {'nps': ui.nps_playlist.DEFAULT_TARGET, 'skills': .925}
                w._set_display('rating', False); w._set_display('skills', 1)
                assert not w.number_lbl.get_visible() and w.others_lbl.get_text().strip() == ''
                def find(widget, kind):
                    if isinstance(widget, kind): return widget
                    for child in (widget.get_children() if isinstance(widget, Gtk.Container) else ()):
                        hit = find(child, kind)
                        if hit: return hit
                dlg = w._display_dialog(); find(dlg, Gtk.SpinButton).set_value(1); find(dlg, Gtk.SpinButton).set_value(4)
                assert w._display['skills'] == 4; dlg.destroy(); w._set_display('skills', 1)
                assert not ui.first_run_prompt(w.tracking_db)                # source runs never ask
                with patch.object(ui.sys, 'frozen', True, create=True):
                    assert ui.first_run_prompt(w.tracking_db) and not ui.first_run_prompt(w.tracking_db)
                assert ui.tracking_switch(w.tracking_db)                     # source runs keep the switch
                ui.recdata.set_tracking(w.tracking_db, False)
                with patch.object(ui, 'DEBUG', False):
                    assert not ui.tracking_switch(w.tracking_db) and ui.recdata.tracking_enabled(w.tracking_db)
                assert w.recs.get_child().get_visible() and w.card_lbl.get_visible()
                # ⚙ sits in the current tab's label and flips it to its settings page; another tab or Stats closes it.
                w._set_rec_mode('nps')
                w.gear_btn.set_active(True)
                assert w._rec_views['nps']['stack'].get_visible_child_name() == 'settings'
                w._set_rec_mode('pp')
                assert not w.gear_btn.get_active() and w._rec_views['nps']['stack'].get_visible_child_name() == 'list'
                w._rec_notebook.set_current_page(3)
                assert w.gear_btn.get_parent() is None
                w._set_rec_mode('nps')
                assert w.gear_btn.get_parent() is w._tab_labels[1] and w.gear_btn.get_visible()
                def named(widget, name):
                    if isinstance(widget, (Gtk.Switch, Gtk.Scale)) and widget.get_accessible().get_name() == name:
                        return widget
                    for child in (widget.get_children() if isinstance(widget, Gtk.Container) else ()):
                        hit = named(child, name)
                        if hit: return hit
                stack = w._rec_views['nps']['stack']
                mix = named(stack, 'Mix in likely PP gains')
                revealer = mix.get_parent().get_parent().get_children()[-2]
                assert isinstance(revealer, Gtk.Revealer) and not revealer.get_reveal_child()
                mix.set_active(True)
                assert revealer.get_reveal_child() and w.rec.tasks[-1][-1]['on'] is True
                named(stack, 'One PP map every').set_value(6.)
                assert w.rec.tasks[-1][-1]['every'] == 6
                named(w._rec_views['pp']['stack'], 'DT').set_active(False)
                assert w.rec.tasks[-1][-2]['dt'] is False
                # The selected map: numbers first (fixed place whatever the title), the full card on hover.
                card = dict(c, rate=1.2, acc_mid=.941, acc_lo=.905, acc_hi=.963, ranked=True, pp_lo=380., pp_hi=430.,
                            best=400., gain=3.1, p_up=.45, skill='chordstream', plays_like=(7.4, 1000))
                w._on_rec(dict(type='card', c=card))
                assert '~94.1%' in w.card_lbl.get_text().split('\n')[0] and 'Cyber Shaman' in w.card_lbl.get_text().split('\n')[1]
                assert '+3.1pp' not in w.card_lbl.get_text() and 'Top-1000' in w.card_lbl.get_tooltip_text()
                w._set_rec_mode('pp')
                assert w.card_lbl.get_text().split('\n')[2].startswith('+3.1pp · 45% to beat')
                # The streamed top never changes height, whatever the title; the last play's acc has its own line.
                def heights():
                    until = time.monotonic() + .2
                    while time.monotonic() < until:
                        while Gtk.events_pending(): Gtk.main_iteration_do(False)
                        time.sleep(.01)
                    return [b.get_allocated_height() for b in (w.card_lbl, w.result_lbl, w._rec_views['pp']['target'])]
                short = heights()
                long_title = 'Very long title ' * 14
                w._on_rec(dict(type='card', c=dict(card, title=long_title, search_meta={})))
                w._on_rec(dict(type='result', title=long_title, text='96.95% · expected ~95.57%', acc=.9695,
                               expected=.9557, z=1.12, rate=1.15, when=time.time(), news=[]))
                w._on_rec(dict(type='target', mode='pp', revision=w._rec_revision, selection_id=w._selection_id,
                               c=dict(card, title=long_title, search_meta={}), msg=''))
                tall = heights()
                assert all(0 < a <= b for a, b in zip(short, tall))
                w._on_rec(dict(type='card', c=dict(card, title=long_title + long_title, search_meta={})))
                assert heights() == tall          # capped: a longer title cannot grow it further
                assert [b.labels[-1].get_layout().get_line_count() for b in (w.card_lbl, w.result_lbl)] == [1, 2]
                assert w.card_lbl.labels[1].get_layout().get_line_count() == 2     # the title: 2 lines, then …
                assert w.result_lbl.get_text().split('\n')[1] == '96.95%  vs ~95.57%  +1.12σ'

                w._set_rec_mode('skills')
                w._save_ui()
                geometry=json.load(open(prefs))
                assert geometry['pp_mix']['on'] and geometry['pp_mix']['every'] == 6 and not geometry['pp_settings']['dt']
                assert geometry['skills_acc_target'] == .925 and geometry['display']['skills'] == 1
                assert geometry['h']>=370 and geometry['recs_h']>=850
                assert geometry['details_h'] > geometry['h'] and geometry['recs_pos'] == list(w.recs.get_position())
                assert geometry['pos'] == list(w.get_position())
                screenshot = os.environ.get("MANIASCOPE_UI_SCREENSHOT")
                if screenshot:
                    c = dict(c, sha="a"*64, acc_mid=.94, acc_lo=.915, acc_hi=.967, ranked=False, skill="chordstream")
                    w._obs = {"source":"tosu", "path":"/test/"+c["sha"], "rate":.87, "mods":[{"acronym":"HT"}]}
                    w._on_rec({"type":"card", "c":c})
                    cskill=dict(cskill,sha=c['sha'])
                    w._on_rec(dict(pskill,selection_id=w._selection_id,revision=w._rec_revision,shown=[cskill]))
                    w._on_rec({"type":"target", "mode":"skills", "revision":w._rec_revision, "selection_id":w._selection_id, "c":cskill, "msg":""})
                    until = time.monotonic() + .25
                    while time.monotonic() < until:
                        while Gtk.events_pending(): Gtk.main_iteration_do(False)
                        time.sleep(.01)
                    size = w.recs.get_size()
                    Gdk.pixbuf_get_from_window(w.recs.get_window(), 0, 0, *size).savev(screenshot, "png", [], [])
                w.move(123, 77); w.recs.move(150, 400)
                until = time.monotonic() + .3
                while time.monotonic() < until:
                    while Gtk.events_pending(): Gtk.main_iteration_do(False)
                    time.sleep(.01)
                w.destroy(); w.recs.destroy()
                saved = json.load(open(prefs))
                assert saved['pos'] == [123, 77] and saved['recs_pos'] == [150, 400]
                assert saved["recommendation_mode"] == "skills" and saved["next_action"] == "copy"
                assert saved["pp_keys"] == [4,5,6,8,9] and saved["nps_keys"] == [7]
                assert saved['skills_keys'] == [7] and set(saved['practice_skills']) == {'ln','jumptrill'}
                assert saved["mode"] == "auto"       # viewer Auto/manual remains separate
                assert saved["recs_open"]             # closed with the playlist open: it reopens on start
                again=ui.ManiaScopeWindow();again.show_all()
                until=time.monotonic()+.15
                while time.monotonic()<until:
                    while Gtk.events_pending():Gtk.main_iteration_do(False)
                    time.sleep(.01)
                assert again.get_size()[1]>=geometry['h'] and list(again.get_position()) == [123, 77] and list(again.recs.get_position()) == [150, 400]
                assert again.recs.get_size()[1]>=geometry['recs_h'] and again.recs_btn.get_active() and again.recs.get_visible()
                again.destroy();again.recs.destroy()


def test_skill_picker_native_cycle_and_separate_expansion():
    from gi.repository import Gtk
    from skill_picker import SkillPicker
    import skill_practice as skills
    changed = []
    picker = SkillPicker((4, 7), (), changed.append)
    window = Gtk.Window()
    window.add(picker); window.show_all()
    try:
        for expected in (('sv',), ('!sv',), ()):
            picker.checks['sv'].clicked()
            assert picker.selected == expected and changed[-1] == expected
        assert len(changed) == 3
        picker.checks['ln'].clicked()
        assert all(skills.state(picker.selected, c.key) == 1 for c in skills.LN.children)
        picker.checks['ln/inverse'].clicked()
        assert skills.state(picker.selected, 'ln/inverse') == -1
        assert all(c.get_inconsistent() for c in picker.all_checks['ln/inverse'])
        assert all(c.get_active() for c in picker.all_checks['ln/release'])
        before = picker.selected
        # Arrow is a sibling, not an expander wrapped around the checkbox.
        arrow = picker.checks['ln'].get_parent().get_children()[0]
        arrow.clicked()
        assert picker.selected == before and (4, 'ln') in picker.expanded
        assert len(changed) == 5
        picker.set_keys((7,))
        assert picker.selected == before and picker.checks['ln/inverse'].get_inconsistent()
        picker.checks['chordstream'].clicked()
        assert skills.state(picker.selected, 'chordstream') == 1
        assert skills.state(picker.selected, 'jumptrill') == 0
    finally:
        window.destroy()


def test_unchosen_playlist_keymodes_follow_what_the_player_plays():
    with tempfile.TemporaryDirectory() as tmp:
        with patch.dict(os.environ, {"MANIASCOPE_REC": tmp}):
            import skillsets as ui
            from gi.repository import Gtk
            class Worker:
                def __init__(self, *a, **kw): self.tasks = []
                def put(self, *task): self.tasks.append(task)
                def start(self): pass
                def stop(self): pass
            prefs = os.path.join(tmp, "ui.json")
            with open(prefs, "w") as f:
                json.dump({"recommendation_mode": "nps", "skills_keys": [7]}, f)      # NPS never chosen
            dbfile = os.path.join(tmp, "rec.db")
            db = ui.recdata.connect(dbfile)
            with db:
                for i in range(30):
                    keys = 4 if i < 24 else 7 if i < 28 else 10       # 80 % 4K, 13 % 7K, 7 % 10K
                    db.execute("INSERT INTO installed(sha256,keys) VALUES(?,?)", (f"s{i}", keys))
                    db.execute("INSERT INTO scores(key,src,sha256,played) VALUES(?,?,?,?)", (f"k{i}", "realm", f"s{i}", f"2026-09-{i+1:02d}"))
            assert ui.recdata.played_keys(db) == (4, 7)
            db.close()
            with patch.object(ui, "UI_FILE", prefs), patch.object(ui.recdata, "DB_FILE", dbfile), \
                 patch.object(ui.recommend, "Worker", Worker), patch.object(ui, "LazerWatcher", Worker), \
                 patch.object(ui.ManiaScopeWindow, "_start_tosu", lambda *_: None), \
                 patch.object(ui.ManiaScopeWindow, "_worker", lambda *_: None):
                w = ui.ManiaScopeWindow()
                assert w._rec_keys["nps"] == (4, 7) and w._rec_keys["skills"] == (7,)
                assert [k for k, b in w._key_checks["nps"].items() if b.get_active()] == [4, 7]
                w._save_ui()
                assert json.load(open(prefs))["nps_keys"] == [4, 7]
                w.destroy(); w.recs.destroy()
            # No history yet: 7K now, not saved as a choice, and the import that lands later decides.
            with open(prefs, "w") as f:
                json.dump({"recommendation_mode": "nps"}, f)
            with patch.object(ui, "UI_FILE", prefs), patch.object(ui.recdata, "DB_FILE", os.path.join(tmp, "empty.db")), \
                 patch.object(ui.recommend, "Worker", Worker), patch.object(ui, "LazerWatcher", Worker), \
                 patch.object(ui.ManiaScopeWindow, "_start_tosu", lambda *_: None), \
                 patch.object(ui.ManiaScopeWindow, "_worker", lambda *_: None):
                w = ui.ManiaScopeWindow()
                assert w._rec_keys["nps"] == (7,) and w._keys_auto == {"nps", "skills"}
                w._save_ui()
                assert json.load(open(prefs))["nps_keys"] is None
                with patch.object(ui.recdata, "played_keys", lambda db: (4,)):
                    assert not w._auto_keys()
                assert w._rec_keys["nps"] == w._rec_keys["skills"] == (4,) and not w._keys_auto
                assert [k for k, b in w._key_checks["skills"].items() if b.get_active()] == [4]
                w.destroy(); w.recs.destroy()
