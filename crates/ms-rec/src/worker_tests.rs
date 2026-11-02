//! The queue, the lists and what an offer records, on a real fit of a synthetic player.
use super::*;
use crate::testkit::*;
use ms_data::store::log_event as log;
use std::sync::mpsc::{channel, Receiver};
use ms_model::choices::{Choice, ChoiceList};

fn selected(sha: &str) -> Task {
    Task::FeatureReady { sha: sha.into(), features: None, refit: false }
}

#[test]
fn progress_counts_unanalysable_charts_as_done() {
    let state = |failed| ListState { total: 10, analyzed: 9, failed, ..ListState::default() };
    assert_eq!(state(1).progress(), None);
    assert_eq!(state(0).progress(), Some((9, 10)));
    assert_eq!(ListState { build_failed: true, ..state(0) }.progress(), None, "a stopped build ends the wait");
}

#[test]
fn the_selected_card_is_not_queued_behind_post_play_rebuilds() {
    let q = Queue::default();
    q.put(Task::Publish);
    q.put(Task::Void);
    q.put(selected("a"));
    q.put(Task::Refresh);
    q.put(selected("b"));
    let order: Vec<&str> = std::iter::from_fn(|| q.get(Duration::ZERO)).map(|t| match t {
        Task::FeatureReady { .. } => "card",
        Task::Publish => "publish",
        Task::Void => "void",
        Task::Refresh => "refresh",
        _ => "other",
    }).collect();
    assert_eq!(order, ["card", "card", "publish", "void", "refresh"]);
}

struct Live {
    w: Worker,
    rx: Receiver<Out>,
    world: World,
}

fn hooks(busy: Check) -> (Hooks, Receiver<Out>) {
    let (tx, rx) = channel();
    let tx = Mutex::new(tx);
    (Hooks { emit: Arc::new(move |m| { let _ = tx.lock().unwrap().send(m); }), busy, game: Arc::new(|| false) }, rx)
}

fn busy(on: bool) -> Check {
    Arc::new(move || on)
}

/// A worker with its fit, and every chart in the NPS and Skills lists' restart caches.
fn live(tag: &str, plays: usize, check: Check) -> Live {
    sandbox();
    let world = world(tag, sloped, plays);
    let shas = install(&world);
    let refs: Vec<Value> = shas.iter().map(|s| json!([s.0, 1.])).collect();
    for mode in ["nps", "skills"] {
        kv_set(&world.db, &format!("playlist_cache:{mode}"), &json!({"version": 1, "calc": "previous", "choices": refs, "analyzed": refs.len()})).unwrap();
    }
    kv_set(&world.db, "realm_import", &json!({"t": 1})).unwrap();
    let (h, rx) = hooks(check);
    let path = world.dir.join("rec.db");
    let mut w = Worker::new(Arc::new(Queue::default()), h, Arc::default(), connect(&path).unwrap(), path);
    w.public = Some(world.public.clone());
    w.rec = Some(engine(&world));
    w.generation = 1;
    w.mode = "nps".into();
    w.restore_local("nps").unwrap();
    Live { w, rx, world }
}

fn pools(l: &Live) -> Vec<Vec<Pick>> {
    l.rx.try_iter().filter_map(|m| if let Out::Pool { shown, .. } = m { Some(shown) } else { None }).collect()
}

fn fresh_list(l: &mut Live) -> Vec<Pick> {
    l.w.pool("nps").unwrap().1
}

fn shas(rows: &[Pick]) -> Vec<String> {
    rows.iter().filter_map(|c| c.sha.clone()).collect()
}

fn events(l: &Live, kind: &str) -> Vec<(f64, Value)> {
    l.world.db.prepare("SELECT t, info FROM events WHERE kind=? ORDER BY id").unwrap().query_map([kind], |r| Ok((r.get::<_, f64>(0)?, serde_json::from_str(&r.get::<_, String>(1)?).unwrap()))).unwrap().map(std::result::Result::unwrap).collect()
}

fn done() -> Done {
    Done { copy: Some("already copied".into()), msg: "Copied".into(), sent: false, unavailable: false }
}

#[test]
fn a_prepared_queue_keeps_its_order_and_records_the_selection_at_once() {
    let mut l = live("order", 240, busy(false));
    let list = fresh_list(&mut l);
    assert!(list.len() >= 3, "{}", list.len());
    let (a, b, c) = (list[0].clone(), list[1].clone(), list[2].clone());
    l.w.playlists.insert("nps", vec![a.clone(), b.clone()]);
    let shown = l.w.prepared_playlist("nps", vec![c.clone(), b.clone(), a.clone()]).unwrap();
    assert_eq!(shas(&shown)[..3], shas(&[a.clone(), b.clone(), c.clone()]));
    // A later filter change cannot discard a selection already made.
    l.w.revision = 2;
    l.w.t_next(false, Some("nps".into()), Some(1), Some(Box::new(shown[0].clone())), Some(done()), Some(100.), 1).unwrap();
    assert_eq!(events(&l, "next")[0].0, 100.);
    let offer = events(&l, "offer").remove(0);
    assert_eq!(offer.0, 100.);
    assert_eq!(offer.1["accuracy"].as_f64(), Some(shown[0].pred.acc_mid));
    let target = l.rx.try_iter().find_map(|m| if let Out::Target { copy, selection_id, .. } = m { Some((copy, selection_id)) } else { None });
    assert_eq!(target, Some((None, 1)));
    assert_eq!(shas(&l.w.playlists["nps"])[..2], shas(&shown)[1..3]);
    let again = l.w.prepared_playlist("nps", vec![a, c, b]).unwrap();
    assert!(!shas(&again).contains(&shown[0].sha.clone().unwrap()), "a consumed pick does not come back");
}

#[test]
fn the_queue_drops_maps_that_fell_out_of_the_current_ranking() {
    let mut l = live("credible", 240, busy(false));
    let list = fresh_list(&mut l);
    let (stale, kept, new) = (list[0].clone(), list[1].clone(), list[2].clone());
    l.w.playlists.insert("nps", vec![stale, kept.clone()]);
    l.w.credible.insert("nps".into(), [kept.identity().unwrap(), new.identity().unwrap()].into_iter().collect());
    let shown = l.w.prepared_playlist("nps", vec![new.clone()]).unwrap();
    assert_eq!(shas(&shown), shas(&[kept, new]));
}

#[test]
fn a_local_queue_rechecks_readiness_after_a_prediction_refresh_and_pp_does_not() {
    let mut l = live("ready", 240, busy(false));
    let rec = l.w.rec.as_ref().unwrap();
    let n = rec.now_state(&l.world.db, DEFAULT_TARGET).unwrap();
    let store = l.w.store.read().unwrap();
    let rows: Vec<Pick> = l.world.charts.iter().map(|(sha, md5, _)| {
        Pick { sha: Some(sha.clone()), md5: Some(md5.clone()), keys: 7, rate: 1., var: "NM".into(), purpose: "nps".into(), installed: true, f: Some(Arc::new(store.feats(sha, 1.).unwrap().clone())), ..Pick::default() }
    }).collect();
    drop(store);
    let fits = |c: &Pick| {
        let rec = l.w.rec.as_ref().unwrap();
        let f = c.f.as_ref().unwrap();
        rec.model().eligible(&rec.predict(f, c.md5.as_ref().unwrap(), None, 1., &n).unwrap().shown, n.target, n.view.warm_flag)
    };
    let want: Vec<String> = rows.iter().filter(|c| fits(c)).filter_map(|c| c.sha.clone()).collect();
    assert!(!want.is_empty() && want.len() < rows.len(), "{}", want.len());
    l.w.playlists.insert("nps", rows.clone());
    let mut shown = shas(&l.w.prepared_playlist("nps", Vec::new()).unwrap());
    shown.sort();
    let mut sorted = want.clone();
    sorted.truncate(20);
    sorted.sort();
    assert_eq!(shown, sorted);
    // PP owns another eligibility policy.
    l.w.playlists.insert("pp", rows[..4].to_vec());
    assert_eq!(l.w.prepared_playlist("pp", Vec::new()).unwrap().len(), 4);
}

#[test]
fn a_stale_click_or_filter_opens_nothing() {
    let mut l = live("stale", 240, busy(false));
    l.w.revision = 5;
    l.w.mode = "skills".into();
    l.w.practice_skills = vec!["ln".to_string()];
    let c = Pick { sha: Some("a".repeat(64)), keys: 7, mode: "skills".into(), purpose: "skills".into(), rate: 1., var: "NM".into(), practice_skills: vec!["ln".into()], ..Pick::default() };
    l.w.playlists.insert("skills", vec![c.clone()]);
    l.w.t_next(false, Some("skills".into()), Some(4), None, None, None, 0).unwrap();
    l.w.t_open(Box::new(Pick { practice_skills: vec!["sv".into()], ..c.clone() }), Some("skills".into()), Some(5), None, None, 0).unwrap();
    assert!(events(&l, "offer").is_empty());
    l.w.t_open(Box::new(c), Some("skills".into()), Some(5), Some(done()), None, 0).unwrap();
    assert_eq!(events(&l, "offer").len(), 1);
}

fn published(l: &Live, mode: &str) -> Published {
    let w = &l.w;
    Published { mode: mode.into(), candidates: Arc::default(), catalog: Arc::default(), selected_skills: w.practice_skills.clone(), total: 0, analyzed: 0, failed: 0, error: String::new(), partial: false,
                revision: w.local_request, generation: w.generation, keys_by_mode: (w.keys["nps"].clone(), w.keys["skills"].clone()) }
}

#[test]
fn a_retained_pick_and_its_candidate_read_from_newer_features() {
    let mut l = live("retained", 240, busy(false));
    let list = fresh_list(&mut l);
    let mut old = list[0].clone();
    let (sha, rate) = (old.sha.clone().unwrap(), old.rate);
    let (nps, description, length) = (old.nps, old.description.clone(), old.length);
    old.description = "Old".into();
    old.skill = Some("Old".into());
    old.nps = Some(1.);
    old.length = 1.;
    old.family = Some("old family".into());
    old.why = "1.0 raw NPS · Old · expect ~95.0%".into();
    l.w.playlists.insert("nps", vec![old.clone()]);
    let stale = Choice { sha: sha.clone(), md5: old.md5.clone().unwrap(), keys: 7, rate, value: 0., installed: true, convert: false, playcount: -1., family: "old family".into(), group: "rice".into(),
        nps: 1., length: 1., profile: Vec::new(), description: "Old".into(), matched: Vec::new(), practice: String::new(), acc_mid: 0.95, pred: old.pred.clone(), generated_at: 0. };
    l.w.mode = "skills".into(); // nothing publishes this list: the test reads what accept left
    l.w.states.get_mut("nps").unwrap().restored = false;
    let mut p = published(&l, "nps");
    p.candidates = Arc::new(ChoiceList { items: vec![Arc::new(stale)], forecast: Arc::default() });
    l.w.t_nps_update(Update::Ready(p)).unwrap();
    let c = &l.w.playlists["nps"][0];
    assert_eq!((c.nps, c.length, &c.description), (nps, length, &description));
    assert_eq!(c.skill.as_ref(), Some(&description));
    assert_eq!(c.why, format!("{:.1} raw NPS · {description} · expect ~95.0%", nps.unwrap()));
    assert_ne!(c.family.as_deref(), Some("old family"));
    let candidate = &l.w.states["nps"].candidates.items[0];
    assert_eq!((Some(candidate.nps), &candidate.description), (nps, &description));
    assert_eq!(candidate.family, c.family.clone().unwrap());
}

#[test]
fn the_first_real_build_replaces_the_installed_only_restart_list() {
    let mut l = live("restart", 240, busy(false));
    l.w.mode = "skills".into();
    let old = Pick { sha: Some("o".into()), keys: 7, rate: 1., ..Pick::default() };
    l.w.playlists.insert("nps", vec![old.clone()]);
    let first = published(&l, "nps");
    l.w.t_nps_update(Update::Ready(first)).unwrap();
    assert!(l.w.playlists["nps"].is_empty());
    l.w.playlists.insert("nps", vec![old]);
    let later = published(&l, "nps");
    l.w.t_nps_update(Update::Ready(later)).unwrap();
    assert_eq!(l.w.playlists["nps"].len(), 1, "later builds keep the displayed order");
}

#[test]
fn a_builder_failure_is_scoped_keeps_the_list_and_clears_on_ready() {
    let mut l = live("failure", 240, busy(false));
    let saved = kv_get(&l.world.db, "playlist_cache:nps");
    let candidates = l.w.states["nps"].candidates.clone();
    let (revision, generation) = (l.w.local_request, l.w.generation);
    let failed = |revision| Update::Failed { error: "worker exited".into(), mode: "nps".into(), revision, generation };
    l.w.t_nps_update(failed(revision + 1)).unwrap();
    assert!(!l.w.states["nps"].build_failed, "a stale failure changes nothing");
    l.w.t_nps_update(failed(revision)).unwrap();
    assert!(Arc::ptr_eq(&l.w.states["nps"].candidates, &candidates));
    assert_eq!(kv_get(&l.world.db, "playlist_cache:nps"), saved);
    let note = l.w.pool("nps").unwrap().2;
    assert!(note.contains("worker exited") && !note.contains("looking for maps"), "{note}");
    l.w.t_nps_update(Update::Ready(published(&l, "nps"))).unwrap();
    assert!(!l.w.states["nps"].build_failed && l.w.states["nps"].error.is_empty());
    assert!(!l.w.pool("nps").unwrap().2.contains("worker exited"));
}

fn configure(mode: &str, revision: u64) -> Config {
    Config { mode: mode.into(), revision, ..Config::default() }
}

#[test]
fn the_focus_and_website_sliders_touch_only_their_list_and_a_late_event_cannot_win() {
    let mut l = live("focus", 240, busy(false));
    let row = Pick { sha: Some("a".repeat(64)), keys: 7, rate: 1., ..Pick::default() };
    l.w.playlists.insert("nps", vec![row.clone()]);
    l.w.playlists.insert("pp", vec![row.clone()]);
    l.w.t_configure(Config { nps_focus: Some(1.), ..configure("nps", 1) }).unwrap();
    assert_eq!(l.w.nps_focus, 1.);
    assert_eq!((l.w.playlists["nps"].len(), l.w.playlists["pp"].len()), (fresh_len(&l), 1));
    l.w.t_configure(Config { nps_focus: Some(0.), ..configure("nps", 0) }).unwrap();
    assert_eq!(l.w.nps_focus, 1., "a superseded slider event cannot win");
    l.w.playlists.insert("skills", vec![row]);
    l.w.t_configure(Config { web_bias: [("skills".to_string(), 0.)].into_iter().collect(), ..configure("nps", 2) }).unwrap();
    assert_eq!((l.w.web_bias["nps"], l.w.web_bias["skills"]), (0.5, 0.));
    assert!(l.w.playlists["skills"].is_empty());
    for bad in [f64::NAN, f64::INFINITY] {
        assert_eq!(builder::normalize_focus(bad), 0.5);
    }
    assert_eq!((builder::normalize_focus(-1.), builder::normalize_focus(2.)), (0., 1.));
}

fn fresh_len(l: &Live) -> usize {
    l.w.playlists["nps"].len()
}

#[test]
fn rapid_filter_changes_skip_superseded_catalogue_searches() {
    let mut l = live("rapid", 240, busy(false));
    let skills = |s: &str, revision| Config { practice_skills: Some(vec![s.to_string()]), keys: [("skills".to_string(), vec![7])].into_iter().collect(), ..configure("skills", revision) };
    l.w.q.put(Task::Configure(Box::new(skills("sv", 1))));
    l.w.q.put(Task::Configure(Box::new(skills("!sv", 2))));
    let requested = l.w.local_request;
    let (first, second) = (l.w.q.get(Duration::ZERO).unwrap(), l.w.q.get(Duration::ZERO).unwrap());
    let (Task::Configure(first), Task::Configure(second)) = (first, second) else { panic!() };
    l.w.t_configure(*first).unwrap();
    assert_eq!((l.w.local_request, pools(&l).len()), (requested, 0));
    l.w.t_configure(*second).unwrap();
    assert_eq!(l.w.local_request, requested + 1);
    assert_eq!(pools(&l).len(), 1);
    assert_eq!((l.w.practice_skills.clone(), l.w.revision), (vec!["!sv".to_string()], 2));
}

#[test]
fn pp_maps_mix_into_the_nps_list_only_when_on_and_stay_pp_rows() {
    let mut l = live("mix", 240, busy(false));
    let last = |l: &Live| pools(l).pop().unwrap();
    l.w.publish(None).unwrap();
    assert!(last(&l).iter().all(|c| c.purpose != "nps_mix"));
    log(&l.world.db, "offer", Some("md5:x"), Some(now() - 300.), &json!({"mode": "pp"})).unwrap();
    log(&l.world.db, "offer", Some("sha:n9"), Some(now() - 200.), &json!({"mode": "nps"})).unwrap();
    l.w.session_cache = None;
    let mix = PpMix { on: true, every: 4, chance: 0., ..PpMix::default() };
    l.w.t_configure(Config { pp_mix: Some(mix), ..configure("nps", 1) }).unwrap();
    let shown = last(&l);
    let places: Vec<usize> = shown.iter().enumerate().filter(|(_, c)| c.purpose == "nps_mix").map(|(i, _)| i).collect();
    assert_eq!(places[0], 2, "one NPS pick since the last PP offer: after two more");
    let mixed = shown[2].clone();
    assert!(mixed.gain.is_some());
    // A refresh keeps it a PP row with its details.
    l.w.publish(None).unwrap();
    let again = last(&l);
    assert!(again.iter().any(|c| c.purpose == "nps_mix" && c.gain.is_some()));
    l.w.open(mixed, "nps", "list", Some(done()), Some(now()), 0).unwrap();
    let info = events(&l, "offer").pop().unwrap().1;
    assert_eq!((info["mode"].as_str(), info["purpose"].as_str()), (Some("pp"), Some("nps_mix")));
}

#[test]
fn pp_rows_come_every_nth_pick() {
    let row = |s: &str| Pick { title: s.into(), ..Pick::default() };
    let titled = |n: &[&str]| n.iter().map(|s| row(s)).collect::<Vec<_>>();
    let out = place_every(titled(&["n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7"]), titled(&["pp4", "pp1"]), 4, 1);
    assert_eq!(out.iter().map(|c| c.title.as_str()).collect::<Vec<_>>(), ["n0", "n1", "pp4", "n2", "n3", "n4", "pp1", "n5", "n6", "n7"]);
}

#[test]
fn the_replay_catch_up_runs_with_queued_work_but_not_during_gameplay() {
    let playing = Arc::new(AtomicBool::new(true));
    let flag = playing.clone();
    let mut l = live("replay", 30, Arc::new(move || flag.load(Ordering::Relaxed)));
    let past = Instant::now() - Duration::from_secs(1);
    l.w.replay_due = Some(past);
    l.w.q.put(Task::Tracking);
    l.w.catch_up_replays();
    assert_eq!(l.w.replay_due, Some(past));
    playing.store(false, Ordering::Relaxed);
    l.w.catch_up_replays();
    assert_eq!(l.w.replay_due, None);
    let end = Instant::now() + Duration::from_secs(10);
    let mut seen = false;
    while !seen && Instant::now() < end {
        seen = matches!(l.w.q.get(Duration::from_millis(50)), Some(Task::ReplayDone { .. }));
    }
    assert!(seen);
}

#[test]
fn a_post_play_rebuild_waits_for_a_selection_and_yields_to_it() {
    let mut l = live("yield", 240, busy(false));
    l.w.refresh_later();
    l.w.refresh_later(); // coalesced
    l.w.q.put(Task::Tracking);
    l.w.q.put(selected(""));
    let names: Vec<&str> = std::iter::from_fn(|| l.w.q.get(Duration::ZERO)).map(|t| match t {
        Task::FeatureReady { .. } => "selected",
        Task::Refresh => "refresh",
        _ => "other",
    }).collect();
    assert_eq!(names, ["selected", "other", "refresh"]);
    l.w.q.put(Task::Tracking);
    l.w.q.put(selected(""));
    l.w.yield_urgent(); // inside a rebuild: only the selection runs
    assert!(matches!(l.w.q.get(Duration::ZERO), Some(Task::Tracking)));
    assert!(l.w.q.get(Duration::ZERO).is_none());
}

#[test]
fn the_first_run_learns_the_level_before_suggesting() {
    let mut l = live("first", 0, busy(false));
    l.w.first_import = true;
    l.w.publish(None).unwrap();
    *l.w.learning.lock().unwrap() = Some("Analysing…".into());
    l.w.first_import = false;
    l.w.publish(None).unwrap();
    assert!(pools(&l).is_empty(), "the reason, not 4K Beginner maps");
    *l.w.learning.lock().unwrap() = None;
    l.w.first_import = true;
    l.w.t_imported(Some(ms_data::imports::Imported { user_id: None, seen: 0, new: 11049, installed: 52988, miss_ratio: None }), 0, true, None).unwrap();
    assert!(!l.w.first_import);
    assert!(l.w.refit_running, "the import restarts the fit");
    assert!(l.w.jobs.filling.load(Ordering::SeqCst), "and the analysis of the played charts");
}

fn file(l: &Live, i: usize) -> String {
    let p = l.world.dir.join(format!("chart{i}.osu"));
    std::fs::write(&p, chart_bytes(i)).unwrap();
    p.to_string_lossy().into_owned()
}

#[test]
fn a_start_keeps_its_displayed_difficulty_forecast_and_actual_rate() {
    let mut l = live("start", 240, busy(false));
    let (path, sha) = (file(&l, 5), l.world.charts[5].0.clone());
    l.w.t_start(&json!({"sha": sha, "path": path, "rate": 1., "mods": [], "t": 100.})).unwrap();
    let saved = events(&l, "start").remove(0).1;
    let forecast = &saved["accuracy_forecast"];
    assert!(saved["difficulty"].as_f64().unwrap() > 0. && saved["rate"] == 1. && saved["expected"].as_f64().unwrap() > 0.);
    assert_eq!((forecast["target"].as_str(), forecast["kind"].as_str()), (Some("native_displayed_lazer"), Some("usual_range")));
    assert_eq!((forecast["sha256"].as_str(), forecast["md5"].as_str()), (Some(sha.as_str()), Some(l.world.charts[5].1.as_str())));
    assert_eq!(forecast["mid"], saved["expected"]);
    assert_eq!((saved["calculator"].as_str(), saved["predictor"].as_str()), (Some(CALC_ID), Some(PREDICTOR_ID)));
    assert_eq!(saved["expected_lo"], forecast["lo"]);
    assert!(saved["display_mu"].as_f64().is_some() && saved["mu"] != saved["display_mu"]);
    // A rate with no analysis has no pre-play estimate.
    l.w.t_start(&json!({"sha": sha, "path": path, "rate": 0.78, "mods": [{"acronym": "HT", "settings": {"speed_change": 0.78}}], "t": 101.})).unwrap();
    let missing = events(&l, "start").remove(1).1;
    assert!(missing["difficulty"].is_null() && missing["expected"].is_null());
    assert_eq!(missing["rate"], 0.78);
}

#[test]
fn an_updated_map_keeps_its_offer_link_without_reusing_the_old_revision() {
    let mut l = live("updated", 240, busy(false));
    let list = fresh_list(&mut l);
    let offered = Pick { bid: Some(1123210), rate: 1.25, var: "DT 1.25×".into(), purpose: "nps".into(), ..list[0].clone() };
    l.w.open(offered.clone(), "nps", "list", Some(done()), None, 0).unwrap();
    let path = l.world.dir.join("updated.osu");
    let text = "osu file format v14\n[Metadata]\nArtist:Artist\nTitle:Updated\nVersion:Hard\nBeatmapID:1123210\n";
    std::fs::write(&path, text).unwrap();
    let (sha, md5) = hashes(text.as_bytes());
    let dt = vec![json!({"acronym": "DT", "settings": {"speed_change": 1.25}})];
    l.w.t_selected(Some(path.clone()), Some(1.25), Some(dt.clone())).unwrap();
    let message = l.rx.try_iter().find_map(|m| if let Out::Target { msg, .. } = m { (msg.contains("chart selected")).then_some(msg) } else { None }).unwrap();
    assert!(message.contains("updated chart selected"), "{message}");
    let (beatmap, info): (String, String) = l.world.db.query_row("SELECT beatmap, info FROM events WHERE kind='selected'", [], |r| Ok((r.get(0)?, r.get(1)?))).unwrap();
    assert_eq!(beatmap, format!("md5:{md5}"));
    assert_eq!(serde_json::from_str::<Value>(&info).unwrap()["offered_revision"].as_str(), offered.identity().as_deref());
    l.w.t_start(&json!({"sha": sha, "path": path, "rate": 1.25, "mods": dt, "t": now()})).unwrap();
    let start = events(&l, "start").remove(0).1;
    assert!(start["offer_id"].as_i64().is_some());
    assert_eq!(start["purpose"], "nps");
    assert_eq!((start["sha"].as_str(), start["md5"].as_str()), (Some(sha.as_str()), Some(md5.as_str())));
    assert_eq!(start["offered_accuracy"].as_f64(), Some(offered.pred.acc_mid));
    assert!(!same_map(&offered, Some("different"), None, Some(-1)) && !same_map(&offered, Some("different"), None, Some(123)));
}

#[test]
fn an_offer_keeps_the_prepared_forecast_as_it_was_generated() {
    let mut l = live("forecast", 240, busy(false));
    let list = fresh_list(&mut l);
    l.w.t_open(Box::new(list[0].clone()), Some("nps".into()), Some(l.w.revision.wrapping_sub(1)), Some(done()), Some(1990.), 0).unwrap();
    let (t, saved) = events(&l, "offer").remove(0);
    assert_eq!(t, 1990.);
    assert_eq!(saved["accuracy"].as_f64(), Some(list[0].pred.acc_mid));
    assert_eq!(saved["accuracy_forecast"]["mid"].as_f64(), Some(list[0].pred.acc_mid));
    assert!(saved["accuracy_forecast_recorded_at"].as_f64().unwrap() > 1990.);
    assert_eq!(saved["accuracy_forecast"]["rate"], 1.);
    assert_eq!(saved["calculator"].as_str(), Some(CALC_ID));
}

#[test]
fn a_live_result_reads_its_chart_and_keeps_its_metadata_fresh() {
    let mut l = live("score", 240, busy(false));
    let (path, sha) = (file(&l, 2), l.world.charts[2].0.clone());
    l.world.db.execute("UPDATE installed SET title='Renamed' WHERE sha256=?", [&sha]).unwrap();
    let score = json!({"sha256": sha, "score": 123, "hits": {"geki": 100}, "mods_list": [], "rate": 1., "played": "2026-09-28T12:00:00Z"});
    l.w.t_score(&score, Path::new(&path), true, None).unwrap();
    let bid: i64 = l.world.db.query_row("SELECT beatmap_id FROM scores WHERE sha256=? AND src='live'", [&sha], |r| r.get(0)).unwrap();
    assert_eq!(bid, 1002);
    assert!(l.rx.try_iter().any(|m| matches!(m, Out::Result(v) if v.title.contains("Renamed"))));
}

/// One builder run to its first whole publication (or the end of the request).
fn built(l: &mut Live, mode: &str, setup: impl FnOnce(&mut Builder)) -> (Builder, Vec<Update>) {
    let r = l.w.request_of(mode).unwrap();
    let mut b = Builder::new(l.w.store.clone(), l.w.rec.as_ref().unwrap().prm.clone());
    let (out, stop) = (Arc::new(Mutex::new(Vec::new())), Arc::new(AtomicBool::new(false)));
    let (o, s) = (out.clone(), stop.clone());
    b.emit = Arc::new(move |u: Update| {
        s.store(matches!(&u, Update::Ready(p) if !p.partial && p.analyzed + p.failed >= p.total), Ordering::Relaxed);
        o.lock().unwrap().push(u);
    });
    let s = stop;
    b.wanted = Arc::new(move || s.load(Ordering::Relaxed));
    setup(&mut b);
    b.build(&l.world.db, &r).unwrap();
    let updates = std::mem::take(&mut *out.lock().unwrap());
    (b, updates)
}

fn last_ready(updates: &[Update]) -> &Published {
    updates.iter().rev().find_map(|u| if let Update::Ready(p) = u { Some(p) } else { None }).expect("a publication")
}

#[test]
fn the_builder_works_only_for_the_active_list() {
    let mut l = live("active", 240, busy(false));
    let (_, none) = built(&mut l, "pp", |_| {});
    assert!(none.is_empty(), "PP never waits for an invisible tab");
    let (_, skills) = built(&mut l, "skills", |_| {});
    assert_eq!(last_ready(&skills).mode, "skills");
    let (_, nps) = built(&mut l, "nps", |_| {});
    assert_eq!(last_ready(&nps).mode, "nps");
}

#[test]
fn only_analysed_rates_are_offered_and_a_broken_chart_counts_as_done() {
    let mut l = live("rates", 240, busy(false));
    let broken = "b".repeat(64);
    let path = ms_data::paths::local_file(&broken, &sandbox()).unwrap();
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, "not a chart").unwrap();
    l.world.db.execute("INSERT OR REPLACE INTO installed VALUES (?,5000,5000,'broken',1,7,8,180,'broken','a','v')", [&broken]).unwrap();
    let (_, updates) = built(&mut l, "nps", |_| {});
    let p = last_ready(&updates);
    assert_eq!((p.total, p.failed), (25, 1));
    let store = l.w.store.read().unwrap();
    let refs = p.candidates.restart_refs();
    assert!(!refs.is_empty());
    for (sha, rate) in refs {
        assert!(store.has_rate(&sha, rate), "{sha} {rate}");
        assert!((ms_model::predict::MIN_RATE..=ms_model::predict::MAX_RATE).contains(&rate));
    }
}

#[test]
fn a_restart_keeps_the_saved_lists_converts() {
    let mut l = live("restart-converts", 240, busy(false));
    let saved = l.w.states["nps"].catalog.keys().next().unwrap().clone();
    let rec = l.w.rec.as_mut().unwrap();
    rec.r.installed = Arc::new(rec.r.installed.iter().filter(|(_, i)| i.sha256 != saved).map(|(k, i)| (k.clone(), i.clone())).collect()); // a convert has no library row
    l.world.db.execute("INSERT INTO converts VALUES ('src',7,?,4,?,'m','A','T','V','C',5,NULL,8,90,1,0)", rusqlite::params![ms_convert::CONVERTER_ID, saved]).unwrap();
    l.w.converts.get_mut("nps").unwrap().insert(7, Setting { on: true, sources: vec![4], often: 0.5, weighted: true });
    l.w.restored.clear();
    l.w.restore_local("nps").unwrap();
    assert_eq!(l.w.states["nps"].catalog[&saved].convert_of.as_deref(), Some("src"));
}

#[test]
fn a_filter_nothing_matches_offers_nothing() {
    let mut l = live("filter", 240, busy(false));
    l.w.practice_skills = vec!["sv".into()];
    let (_, updates) = built(&mut l, "skills", |_| {});
    assert!(last_ready(&updates).candidates.restart_refs().is_empty());
}

#[test]
fn a_newer_request_ends_the_running_build_and_gameplay_pauses_it() {
    let mut l = live("interrupt", 240, busy(false));
    let (_, none) = built(&mut l, "nps", |b| b.wanted = Arc::new(|| true));
    assert!(none.iter().all(|u| matches!(u, Update::Ready(p) if p.partial)), "superseded at once: no whole publication");
    let (_, paused) = built(&mut l, "nps", |b| {
        b.busy = Arc::new(|| true);
        let start = Instant::now();
        b.wanted = Arc::new(move || start.elapsed() > Duration::from_millis(100));
    });
    assert!(paused.iter().all(|u| matches!(u, Update::Ready(p) if p.partial)), "gameplay holds the analysis");
    let (_, whole) = built(&mut l, "nps", |_| {});
    assert!(!last_ready(&whole).candidates.restart_refs().is_empty(), "the next request still completes");
}
