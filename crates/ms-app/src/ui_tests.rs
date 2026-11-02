//! The window against scripted worker messages. Nothing here touches the real data, clipboard or game.
use super::*;
use egui_kittest::kittest::Queryable;
use egui_kittest::Harness;
use std::sync::atomic::AtomicUsize;

fn app(prefs: Value, ctx: &egui::Context) -> App {
    static N: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!("ms-ui-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
    std::fs::create_dir_all(&dir).unwrap();
    let mut a = App::bare(ctx, Settings::from_json(&prefs), &dir.join("rec.db")).unwrap();
    a.clipboard = None;
    a.s.next_action = "copy".to_string(); // never the client
    a.open.profile = None;
    a.tracking = None;
    a
}

fn prefs() -> Value {
    json!({"recommendation_mode": "nps", "pp_keys": [4, 6, 9], "nps_keys": [7], "skills_keys": [4, 7], "practice_skills": ["ln", "sv"], "next_action": "copy"})
}

fn harness(a: App) -> Harness<'static, App> {
    Harness::builder().with_size(egui::vec2(520., 760.)).allow_missing_glyphs().with_context_setup(crate::theme::apply).build_ui_state(|ui, a: &mut App| a.playlist_ui(ui), a)
}

fn pick(sha: char) -> Pick {
    Pick { sha: Some(sha.to_string().repeat(64)), md5: Some(sha.to_string().repeat(32)), title: format!("Cosmo — Cyber Shaman {sha}"), artist: Some("Cosmo".into()), song: Some("Cyber Shaman".into()),
           version: Some("Local difficulty".into()), keys: 7, rate: 0.87, var: "HT 0.87×".into(), purpose: "nps".into(), ..Pick::default() }
}

fn pool(a: &App, mode: &str, revision: u64, shown: Vec<Pick>) -> Out {
    Out::Pool { mode: mode.into(), revision, selection_id: a.selection_id, phase: "warmup".into(), shown, note: "Synthetic UI check · target ~94%".into(), summary: "No actual prediction".into(), progress: None }
}

fn tasks(a: &App) -> Vec<Task> {
    std::iter::from_fn(|| a.q.get(Duration::ZERO)).collect()
}

fn nexts(t: &[Task]) -> usize {
    t.iter().filter(|x| matches!(x, Task::Next { .. })).count()
}

#[test]
fn the_expected_accuracy_waits_for_enough_of_the_players_own_plays() {
    let ctx = egui::Context::default();
    let mut a = app(prefs(), &ctx);
    a.card = Some(Box::new(ms_rec::engine::Card { acc_mid: 0.962, sure: false, ..Default::default() }));
    let mut h = Harness::builder().with_size(egui::vec2(430., 240.)).allow_missing_glyphs().with_context_setup(crate::theme::apply).build_ui_state(|ui, a: &mut App| a.main_ui(ui), a);
    h.step();
    assert!(h.query_by_label_contains("Expected").is_none());
    h.state_mut().card.as_mut().unwrap().sure = true;
    h.step();
    h.get_by_label("Expected ~96.2%");
}

#[test]
fn a_row_keeps_its_hover_over_the_text_inside_it() {
    let ctx = egui::Context::default();
    let mut a = app(prefs(), &ctx);
    a.sync_rec();
    let revision = a.revision;
    let out = pool(&a, "nps", revision, vec![pick('a')]);
    a.on_rec(out);
    let mut h = harness(a);
    h.step();
    h.get_by_label_contains("Local difficulty").hover();
    h.step();
    h.step();
    assert_eq!(h.output().platform_output.cursor_icon, egui::CursorIcon::PointingHand);
}

#[test]
fn a_tab_without_its_first_list_says_it_is_loading_not_that_nothing_fits() {
    let ctx = egui::Context::default();
    let mut a = app(prefs(), &ctx);
    a.s.mode = "pp".into();
    let revision = a.revision;
    let mut h = harness(a);
    h.step();
    h.get_by_label_contains("Loading PP suggestions");
    let p = pool(h.state(), "pp", revision, Vec::new());
    h.state_mut().on_rec(p);
    h.step();
    h.get_by_label_contains("Nothing to suggest yet.");
}

#[test]
fn a_stale_list_is_ignored_and_the_current_one_enables_next() {
    let ctx = egui::Context::default();
    let mut a = app(prefs(), &ctx);
    a.sync_rec();
    let revision = a.revision;
    let stale = pool(&a, "nps", revision - 1, vec![pick('a')]);
    a.on_rec(stale);
    assert!(!a.next_enabled());
    let fresh = pool(&a, "nps", revision, vec![pick('a')]);
    a.on_rec(fresh);
    assert!(a.next_enabled());
    let mut h = harness(a);
    h.step();
    h.get_by_label_contains("Next · NPS");
}

#[test]
fn one_status_line_and_bar_the_workers_progress_first_else_this_tabs_analysis() {
    let ctx = egui::Context::default();
    let mut a = app(prefs(), &ctx);
    a.sync_rec();
    let revision = a.revision;
    a.on_rec(Out::Status { text: "Analysing 11 new or updated maps in the background · 0 done".into(), progress: Some((30, 120)), blocking: false, learning: Some(true) });
    let mut p = pool(&a, "nps", revision, vec![pick('a')]);
    if let Out::Pool { progress, .. } = &mut p {
        *progress = Some((5, 50));
    }
    a.on_rec(p);
    assert_eq!(a.bg_progress, Some((30, 120)));
    assert!(a.stats_wait.as_ref().is_some_and(|w| w.0.contains("30 / 120")));
    a.on_rec(Out::Status { text: String::new(), progress: None, blocking: false, learning: Some(false) });
    assert!(a.stats_wait.is_none() && a.bg_progress.is_none());
    assert!(a.views["nps"].analysing.as_ref().is_some_and(|x| x.0.contains("5 / 50")));
    let done = pool(&a, "nps", revision, vec![pick('a')]);
    a.on_rec(done);
    assert!(a.views["nps"].analysing.is_none());
}

#[test]
fn next_uses_the_finished_list_and_a_hover_only_defers_painting() {
    let ctx = egui::Context::default();
    let mut a = app(prefs(), &ctx);
    a.sync_rec();
    tasks(&a);
    let revision = a.revision;
    let first = pool(&a, "nps", revision, vec![pick('a')]);
    a.on_rec(first);
    a.next(false); // a fake worker never drains its queue: Next still navigates
    assert_eq!(nexts(&tasks(&a)), 1);
    assert_eq!(a.views["nps"].target.as_ref().map(|t| t.sha.clone()), Some(pick('a').sha));
    assert!(!a.next_enabled(), "the pick left the list");
    a.next(false);
    assert_eq!(nexts(&tasks(&a)), 0, "a second press with nothing left is no task");
    // A completed refresh wins even when hover deferred painting it.
    a.views.get_mut("nps").unwrap().hover = true;
    let mut b = pick('b');
    b.pred.acc_mid = 0.951;
    let fresh = pool(&a, "nps", revision, vec![b, pick('c')]);
    a.on_rec(fresh);
    a.next(false);
    assert_eq!(a.views["nps"].target.as_ref().map(|t| t.pred.acc_mid), Some(0.951));
    assert!(a.next_enabled());
    // A stale target can't move the selection.
    let stale = Out::Target { mode: "nps".into(), revision: revision - 1, selection_id: a.selection_id, c: Some(Box::new(pick('z'))), msg: "stale".into(), copy: Some("STALE".into()) };
    a.on_rec(stale);
    assert_eq!(a.views["nps"].target.as_ref().map(|t| t.sha.clone()), Some(pick('b').sha));
}

fn verdict(targets: &[&str]) -> Box<Verdict> {
    Box::new(Verdict { title: "t".into(), text: "x".into(), acc: 0.95, expected: None, z: None, rate: None, when: None, news: Vec::new(), targets: targets.iter().map(|m| m.to_string()).collect() })
}

#[test]
fn a_finished_play_moves_on_only_in_its_own_tab() {
    let ctx = egui::Context::default();
    let mut a = app(prefs(), &ctx);
    a.s.auto_next = true;
    a.sync_rec();
    let list = pool(&a, "nps", a.revision, vec![pick('b'), pick('c')]);
    a.on_rec(list);
    a.on_rec(Out::Result(verdict(&["pp"])));
    assert!(a.views["nps"].target.is_none());
    a.on_rec(Out::Result(verdict(&["nps"])));
    assert_eq!(a.views["nps"].target.as_ref().map(|t| t.sha.clone()), Some(pick('b').sha));
}

#[test]
fn keymodes_are_per_tab_and_none_disables_next() {
    let ctx = egui::Context::default();
    let mut a = app(prefs(), &ctx);
    a.sync_rec();
    let list = pool(&a, "nps", a.revision, vec![pick('a')]);
    a.on_rec(list);
    a.set_keys("pp", vec![4, 5, 6, 8, 9]);
    assert_eq!(a.s.keys["nps"], [7]);
    a.set_keys("nps", Vec::new());
    assert!(!a.next_enabled());
    assert_eq!(a.views["nps"].note, "Select at least one keymode");
    a.set_keys("nps", vec![7]);
    assert_eq!(a.s.keys["pp"], [4, 5, 6, 8, 9]);
    assert!(!a.s.keys_auto.contains("nps"));
}

#[test]
fn a_row_from_an_older_skill_filter_opens_nothing_and_stale_lists_stay_out() {
    let ctx = egui::Context::default();
    let mut a = app(prefs(), &ctx);
    a.set_mode("skills");
    a.sync_rec();
    let old = Pick { practice_skills: a.s.practice.clone(), purpose: "skills".into(), ..pick('a') };
    let list = pool(&a, "skills", a.revision, vec![old]);
    a.on_rec(list);
    assert!(a.next_enabled());
    a.set_skills(vec!["sv".to_string()]);
    tasks(&a);
    a.activate("skills", 0, None);
    assert!(a.views["skills"].shown.is_empty() && tasks(&a).is_empty());
    let stale = pool(&a, "skills", a.revision - 1, vec![pick('a')]);
    a.on_rec(stale);
    assert!(a.views["skills"].shown.is_empty());
    a.set_skills(Vec::new());
    assert!(a.s.practice.is_empty());
}

#[test]
fn stats_is_not_a_fourth_mode_and_its_skill_opens_practice() {
    let ctx = egui::Context::default();
    let mut a = app(prefs(), &ctx);
    a.set_mode("skills");
    tasks(&a);
    a.set_stats(true);
    assert!(matches!(tasks(&a).last(), Some(Task::Stats(true))));
    assert_eq!(a.mode(), "skills");
    a.practice_from_stats(7, "ln");
    assert!(!a.stats_open && a.mode() == "skills" && a.s.practice == ["ln"] && a.s.keys["skills"] == [7]);
}

#[test]
fn the_convert_row_needs_a_selected_map() {
    let ctx = egui::Context::default();
    let mut a = app(prefs(), &ctx);
    a.open.menu = true;
    let mut h = Harness::builder().with_size(egui::vec2(400., 620.)).allow_missing_glyphs().build_ui_state(|ui, a: &mut App| a.menu_ui(ui), a);
    h.step();
    h.get_by_label("10K").click();
    h.run();
    assert!(h.state().open.menu, "a press with no map does nothing");
    for gone in ["Next playlist", "Rate", "New session", "Follow osu!'s rate (Auto)"] {
        assert!(h.query_by_label(gone).is_none(), "{gone}");
    }
    h.get_by_label("Dan level");
}

#[test]
fn one_checked_skill_gets_its_slider_and_any_clears_it() {
    let ctx = egui::Context::default();
    let mut a = app(json!({"recommendation_mode": "skills", "skills_keys": [7], "practice_skills": ["chordjack", "!sv"]}), &ctx);
    a.views.get_mut("skills").unwrap().settings = true;
    let mut h = harness(a);
    h.step();
    assert_eq!(h.get_all_by_label("✓ Chordjack").count(), 3, "the chip, and its slider row with the slider");
    assert_eq!(h.get_all_by_label("Other maps").count(), 2, "the row and its slider");
    assert_eq!(h.get_all_by_label("✕ SV").count(), 3, "the chip, and its slider row with the slider");
    h.get_by_label("Any").click();
    h.run();
    assert!(h.state().s.practice.is_empty());
    assert!(h.query_by_label("Other maps").is_none());
}

#[test]
fn the_focus_slider_names_the_side_it_leans_to() {
    use crate::playlist::focus_label;
    assert_eq!([focus_label(0.), focus_label(0.25), focus_label(0.5), focus_label(0.75)], ["100% variety", "75% variety", "Even mix", "75% NPS"]);
}

#[test]
fn a_linked_install_is_not_asked_for_its_profile() {
    let dir = std::env::temp_dir().join(format!("ms-profile-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let db = ms_data::store::connect(&dir.join("rec.db")).unwrap();
    assert!(super::asks_profile(&db));
    ms_data::store::kv_set(&db, "user_id", &serde_json::json!(1)).unwrap();
    assert!(!super::asks_profile(&db));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn gear_flips_the_tab_to_its_settings_and_another_tab_shows_its_list() {
    let ctx = egui::Context::default();
    let mut a = app(prefs(), &ctx);
    a.views.get_mut("nps").unwrap().settings = true;
    let mut h = harness(a);
    h.step();
    h.get_by_label_contains("Mix in likely PP gains");
    assert_eq!(h.get_by_label("⚙").rect().height(), h.get_by_label("Stats").rect().height(), "the cog never makes the tabs taller");
    h.state_mut().set_mode("pp");
    h.step();
    assert!(h.query_by_label_contains("Mix in likely PP gains").is_none());
}

#[test]
fn chip_rows_wrap_inside_the_width() {
    let rows = crate::playlist::wrap_rows(&[40., 50., 60., 200.], 100., 4.);
    assert_eq!(rows, [vec![0, 1], vec![2], vec![3]]);
}

#[test]
fn every_skill_chip_stays_inside_the_settings_page() {
    for width in [430., 320.] {
        let ctx = egui::Context::default();
        let mut a = app(json!({"recommendation_mode": "skills", "skills_keys": [7]}), &ctx);
        a.views.get_mut("skills").unwrap().settings = true;
        let mut h = Harness::builder().with_size(egui::vec2(width, 760.)).allow_missing_glyphs().with_context_setup(crate::theme::apply).build_ui_state(|ui, a: &mut App| a.playlist_ui(ui), a);
        h.step();
        h.get_by_label("Any");
        for chip in ["Delay", "Chordstream", "Jumptrill", "Bracket", "Chordjack", "Jackspeed", "LN", "Technical", "Stamina", "SV"] {
            let right = h.get_by_label(chip).rect().right() as f32;
            assert!(right <= width, "{chip} ends at {right} of {width}");
        }
    }
}

#[test]
fn clicking_the_open_tab_closes_its_settings_like_the_cog() {
    let ctx = egui::Context::default();
    let mut a = app(prefs(), &ctx);
    a.views.get_mut("nps").unwrap().settings = true;
    let mut h = harness(a);
    h.step();
    h.get_by_label("NPS").click();
    h.run();
    assert!(!h.state().views["nps"].settings);
    assert!(h.query_by_label_contains("Mix in likely PP gains").is_none());
}

#[test]
fn a_pp_map_in_the_nps_list_keeps_its_pp_details_on_the_row() {
    let ctx = egui::Context::default();
    let mut a = app(prefs(), &ctx);
    a.sync_rec();
    let row = Pick { purpose: "nps_mix".into(), gain: Some(3.1), p_up: Some(0.45), best: Some(400.), ranked: true, ..pick('a') };
    let list = pool(&a, "nps", a.revision, vec![row]);
    a.on_rec(list);
    let mut h = harness(a);
    h.step();
    h.get_by_label_contains("+3.1pp");
    h.get_by_label_contains("best 400pp");
}

#[test]
fn unchosen_keymodes_follow_what_the_player_plays() {
    let ctx = egui::Context::default();
    let mut a = app(json!({"recommendation_mode": "nps", "skills_keys": [7]}), &ctx);
    assert_eq!(a.s.keys["nps"], [7]);
    assert!(a.s.keys_auto.contains("nps"));
    for i in 0..30 {
        let keys = if i < 24 { 4 } else if i < 28 { 7 } else { 10 }; // 80 % 4K, 13 % 7K, 7 % 10K
        a.db.execute("INSERT INTO installed(sha256,keys) VALUES(?,?)", rusqlite::params![format!("s{i}"), keys]).unwrap();
        a.db.execute("INSERT INTO scores(key,src,sha256,played) VALUES(?,?,?,?)", rusqlite::params![format!("k{i}"), "realm", format!("s{i}"), format!("2026-09-{:02}", i + 1)]).unwrap();
    }
    a.keys_checked = None;
    a.auto_keys(false);
    assert!(a.s.keys["nps"] == [4, 7] && a.s.keys["skills"] == [7] && a.s.keys_auto.is_empty());
    let saved = a.s.to_json();
    assert_eq!(saved["nps_keys"], json!([4, 7]));
}

fn osu_file(dir: &std::path::Path, name: &str) -> PathBuf {
    let mut text = "osu file format v14\n\n[General]\nMode: 3\n\n[Metadata]\nTitle:t\nVersion:v\n\n[Difficulty]\nCircleSize:7\n\n[TimingPoints]\n0,500,4,1,0,100,1,0\n\n[HitObjects]\n".to_string();
    for i in 0..200 {
        text.push_str(&format!("{},192,{},1,0,0:0:0:0:\n", (i % 7) * 73 + 36, 1000 + i * 60));
    }
    std::fs::create_dir_all(dir).unwrap();
    let p = dir.join(name);
    std::fs::write(&p, text).unwrap();
    p
}

fn observed(path: &std::path::Path, rate: f64) -> Obs {
    Obs { source: Some("game"), path: Some(path.to_path_buf()), rate: Some(rate), ..Obs::default() }
}

#[test]
fn identical_observations_do_not_invalidate_their_pending_analysis() {
    let ctx = egui::Context::default();
    let mut a = app(prefs(), &ctx);
    let dir = std::env::temp_dir().join(format!("ms-ui-obs-{}", std::process::id()));
    let file = osu_file(&dir, "a.osu");
    a.obs = observed(&file, 1.);
    a.refresh();
    let first = a.generation;
    for _ in 0..10 {
        a.refresh();
    }
    assert_eq!(a.generation, first);
    a.obs = observed(&file, 0.98);
    a.refresh();
    assert_eq!(a.generation, first + 1);
    assert!(a.pending.is_some(), "the header shows the new map until its analysis lands");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_late_result_for_another_chart_cannot_replace_the_settled_one() {
    let ctx = egui::Context::default();
    let mut a = app(prefs(), &ctx);
    let dir = std::env::temp_dir().join(format!("ms-ui-late-{}", std::process::id()));
    let (first, second) = (osu_file(&dir, "a.osu"), osu_file(&dir, "b.osu"));
    a.obs = observed(&first, 1.);
    a.refresh();
    let done = |generation, path: &PathBuf| {
        let result = viewer::analyse(path, 1., true, &standard_params()).map(Arc::new);
        viewer::Done { request: viewer::Request { generation, path: path.clone(), rate: 1., sv: true }, result }
    };
    a.on_analysed(done(a.generation, &first));
    assert!(a.shown.is_some() && a.pending.is_none());
    a.obs = observed(&second, 1.);
    a.refresh();
    let old = a.generation;
    a.obs = observed(&first, 1.);
    a.refresh();
    assert!(a.generation > old && a.requested.is_none() && a.pending.is_none());
    a.on_analysed(done(old, &second));
    assert_eq!(a.shown.as_ref().map(|s| s.path.clone()), Some(first));
    let _ = std::fs::remove_dir_all(dir);
}

/// Windows has no system fallback here, so a glyph the embedded fonts lack draws as a box (`☰`, user 2026-11-01).
#[test]
fn every_character_in_the_window_is_in_the_embedded_fonts() {
    let ctx = egui::Context::default();
    crate::theme::apply(&ctx);
    ctx.run_ui(Default::default(), |_| {}).textures_delta.clear();
    let sources = [include_str!("mainview.rs"), include_str!("playlist.rs"), include_str!("windows.rs"), include_str!("fmt.rs"), include_str!("app.rs")];
    let font = egui::FontId::proportional(crate::theme::BODY);
    let missing: std::collections::BTreeSet<char> = sources.iter().flat_map(|s| s.chars()).filter(|c| !c.is_ascii()).filter(|&c| !ctx.fonts_mut(|f| f.has_glyph(&font, c))).collect();
    assert!(missing.is_empty(), "{missing:?}");
}
