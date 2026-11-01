//! The real fit, Stats and worker on a synthetic player.
#[path = "../src/testkit.rs"]
#[allow(dead_code)]
mod testkit;

use ms_data::store::kv_set;
use ms_model::predict::Consts;
use ms_rec::engine::{Card, Engine};
use ms_rec::worker::{spawn, Config, Hooks, Out, Queue, Task};
use serde_json::json;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use testkit::*;

struct Running {
    q: Arc<Queue>,
    rx: Receiver<Out>,
    halt: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Drop for Running {
    fn drop(&mut self) {
        self.halt.store(true, Ordering::Relaxed);
        self.q.put(Task::Publish);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl Running {
    /// The first message `pick` accepts, within 20 s.
    fn wait<T>(&self, what: &str, mut pick: impl FnMut(Out) -> Option<T>) -> T {
        let end = Instant::now() + Duration::from_secs(20);
        loop {
            match self.rx.recv_timeout(end.saturating_duration_since(Instant::now())) {
                Ok(m) => {
                    if let Some(v) = pick(m) {
                        return v;
                    }
                }
                Err(e) => panic!("{what}: {e}"),
            }
        }
    }

    fn card(&self, what: &str, want: impl Fn(&Option<Box<Card>>, &Option<String>) -> bool) -> (Option<Box<Card>>, Option<String>) {
        self.wait(what, |m| match m {
            Out::Card { c, note, .. } if want(&c, &note) => Some((c, note)),
            _ => None,
        })
    }
}

fn run(w: &World, busy: bool) -> Running {
    sandbox();
    let (tx, rx) = channel();
    let tx = Mutex::new(tx);
    let hooks = Hooks { emit: Arc::new(move |m| { let _ = tx.lock().unwrap().send(m); }), busy: Arc::new(move || busy), game: Arc::new(|| false) };
    let (q, halt) = (Arc::new(Queue::default()), Arc::new(AtomicBool::new(false)));
    let public = w.public.clone();
    let thread = spawn(move |_| Ok(public), q.clone(), hooks, halt.clone(), w.dir.join("rec.db"));
    Running { q, rx, halt, thread: Some(thread) }
}

#[test]
fn the_ability_number_inverts_the_shared_model_and_the_session_overlay_is_separate() {
    let w = world("ability", sloped, 240);
    let e = engine(&w);
    let profile = ms_rec::stats::build(&e, &w.db).unwrap();
    let group = |p: &ms_rec::stats::Profile| p.pages[&7].groups["overall"].clone();
    let base = group(&profile).value.unwrap();
    assert!((base - 2.).abs() < 0.02, "{base}");
    let now = ms_data::store::now();
    let cold = ms_rec::stats::current(&profile, &e.state_at(&w.db, 0.94, now).unwrap(), &e);
    let (current, shown) = (cold.pages[&7].groups["overall"].current.unwrap(), cold.pages[&7].groups["overall"].value.unwrap());
    assert!(current < shown, "{current} {shown}");
    assert_eq!(group(&profile).value, Some(base));
    ms_data::store::log_event(&w.db, "skip_warmup", None, None, &serde_json::json!({})).unwrap();
    let warm = ms_rec::stats::current(&profile, &e.state_at(&w.db, 0.94, ms_data::store::now()).unwrap(), &e);
    assert!((warm.pages[&7].groups["overall"].current.unwrap() - shown).abs() < 1e-6);
}

#[test]
fn sparse_and_unmeasured_limits_are_not_numeric_ratings() {
    let w = world("sparse", |o| 1. - 0.002 * (o / 2.).powi(2), 240);
    let profile = ms_rec::stats::build(&engine(&w), &w.db).unwrap();
    let g = &profile.pages[&7].groups["overall"];
    assert!(g.value.is_none() && g.extrapolated, "{:?}", g.value);
    assert!(profile.pages[&10].groups["overall"].value.is_none());
    let few = world("few", sloped, 2);
    assert!(ms_rec::stats::build(&engine(&few), &few.db).unwrap().pages[&7].groups["overall"].value.is_none());
}

#[test]
fn history_follows_the_monthly_level_and_marks_the_peak() {
    // Seven months of plays, the oldest clearly better than the rest.
    let w = dated("history", 2400, 210., |o, age| { let v = sloped(o); if age > 170. { 1. - (1. - v) * 0.5 } else { v } });
    let e = engine(&w);
    let profile = ms_rec::stats::build(&e, &w.db).unwrap();
    let page = &profile.pages[&7];
    let now = page.groups["overall"].value.unwrap();
    assert!(page.history.len() >= 2, "{:?}", page.history);
    assert!((page.history.last().unwrap().1 - now).abs() < 1e-9);
    let peak = page.peak.unwrap();
    assert_eq!(peak, page.history[0]);
    assert!(peak.1 > now);
    let shown = ms_rec::stats::current(&profile, &e.state_at(&w.db, 0.94, ms_data::store::now()).unwrap(), &e);
    assert!(!shown.pages[&7].session && shown.pages[&7].peak.unwrap().1 > shown.pages[&7].groups["overall"].value.unwrap());
}

#[test]
fn the_saved_list_is_published_before_the_background_import_can_finish() {
    let w = world("saved", sloped, 240);
    let shas = install(&w);
    kv_set(&w.db, "realm_import", &json!({"t": 1})).unwrap();
    kv_set(&w.db, "playlist_cache:nps", &json!({"version": 1, "calc": "previous", "choices": [[shas[3].0, 1.0], ["f".repeat(64), 1.0]], "analyzed": 2})).unwrap();
    // busy: the import thread is held for as long as the test runs.
    let r = run(&w, true);
    r.q.put(Task::Configure(Box::new(Config { mode: "nps".into(), revision: 1, ..Config::default() })));
    let saved = r.wait("first list", |m| match m {
        Out::Pool { mode, shown, .. } if mode == "nps" && !shown.is_empty() => Some(shown),
        _ => None,
    });
    assert_eq!(saved.iter().map(|p| p.sha.clone().unwrap()).collect::<Vec<_>>(), [shas[3].0.clone()], "the unknown chart stays out");
    // Still answers the UI while the import waits.
    r.q.put(Task::History(false));
    r.wait("history", |m| matches!(m, Out::History(_)).then_some(()));
}

#[test]
fn the_card_names_a_pending_rate_a_failure_and_unusable_mods() {
    let w = world("card", sloped, 240);
    let r = run(&w, false);
    let file = w.dir.join("chart.osu");
    std::fs::write(&file, "fixture").unwrap();
    let dt = vec![json!({"acronym": "DT", "settings": {"speed_change": 1.25}})];
    r.wait("started", |m| matches!(m, Out::Status { text, .. } if text.is_empty()).then_some(()));
    let valid = w.dir.join("valid.osu");
    std::fs::write(&valid, chart_bytes(40)).unwrap();
    r.q.put(Task::Selected { path: Some(valid), rate: Some(1.25), mods: Some(dt.clone()) });
    r.card("pending", |_, n| n.as_deref() == Some("Calculating prediction at 1.25×…"));
    r.q.put(Task::Selected { path: Some(file.clone()), rate: Some(1.25), mods: Some(dt.clone()) });
    r.card("failed", |_, n| n.as_deref().is_some_and(|n| n.starts_with("Prediction unavailable at 1.25×: ")));
    r.q.put(Task::Selected { path: Some(file.clone()), rate: None, mods: None });
    r.card("unknown", |_, n| n.as_deref().is_some_and(|n| n.contains("Waiting for osu!")));
    r.q.put(Task::Selected { path: Some(file), rate: Some(1.25), mods: Some(vec![json!({"acronym": "CS"})]) });
    r.card("mods", |_, n| n.as_deref() == Some("Prediction unavailable for these mods"));
}

#[test]
fn the_selected_charts_prediction_is_calculated_during_play() {
    let w = world("during", sloped, 240);
    let r = run(&w, true);
    r.wait("started", |m| matches!(m, Out::Status { text, .. } if text.is_empty()).then_some(()));
    let file = w.dir.join("chart.osu");
    std::fs::write(&file, chart_bytes(30)).unwrap();
    r.q.put(Task::Selected { path: Some(file), rate: Some(1.25), mods: Some(vec![json!({"acronym": "DT", "settings": {"speed_change": 1.25}})]) });
    let (card, _) = r.card("card", |c, _| c.is_some());
    assert_eq!(card.unwrap().keys, 7);
}

#[test]
fn the_library_keeps_only_charts_that_can_be_played() {
    let w = world("library", sloped, 0);
    let shas = install(&w);
    let (present, missing_audio, no_chart) = (&shas[0].0, &shas[1].0, &shas[2].0);
    std::fs::remove_file(ms_data::paths::local_file(no_chart, &sandbox()).unwrap()).unwrap();
    w.db.execute("INSERT INTO installed_local(sha256,audio_sha,audio_required) VALUES (?,'nope',1)", [missing_audio]).unwrap();
    w.db.execute("UPDATE installed SET keys=3 WHERE sha256=?", [&shas[3].0]).unwrap();
    let (found, error) = ms_rec::catalog::library(&w.db, "lazer").unwrap();
    assert!(error.is_empty(), "{error}");
    let found: Vec<&str> = found.iter().map(|i| i.sha256.as_str()).collect();
    assert!(found.contains(&present.as_str()));
    for gone in [missing_audio, no_chart, &shas[3].0] {
        assert!(!found.contains(&gone.as_str()));
    }
}

#[test]
fn a_new_ranked_revision_is_not_hidden_by_its_snapshot_status() {
    use ms_rec::catalog::installed_by_md5;
    use ms_rec::pp::candidates;
    let w = world("ranked", sloped, 0);
    install(&w);
    let c = Consts::standard();
    let qualified = w.charts.len() - 1;
    let count = |db: &ms_data::store::Db| {
        let installed = installed_by_md5(db).unwrap();
        candidates(&w.public, &installed, &[7], &c, Vec::new()).iter().filter(|v| v.md5 == w.charts[qualified].1).count()
    };
    assert_eq!(count(&w.db), 1, "the installed revision says ranked");
    w.db.execute("UPDATE installed SET status=-2 WHERE md5=?", [&w.charts[qualified].1]).unwrap();
    assert_eq!(count(&w.db), 0, "nor is a newer unranked status ignored");
}

#[test]
fn the_card_matches_its_list_entry_and_a_custom_rate_shows_no_pp() {
    let w = world("cardlist", sloped, 240);
    let shas = install(&w);
    let mut e = engine(&w);
    let n = e.state_at(&w.db, 0.94, ms_data::store::now()).unwrap();
    let rows = e.scored(&w.db, &[7], &n).unwrap();
    let c = rows.iter().find(|c| c.rate == 1. && c.gain.unwrap_or(0.) > 0. && shas.iter().any(|s| Some(&s.1) == c.md5.as_ref())).expect("an improvable chart");
    let path = ms_data::paths::local_file(c.sha.as_ref().unwrap(), &sandbox()).unwrap();
    let card = e.card(&w.db, &path, c.rate, &n).unwrap().unwrap();
    let close = |a: f64, b: f64| (a - b).abs() < 1e-6 * b.abs().max(1.);
    assert!(close(card.acc_mid, c.pred.acc_mid), "{} {}", card.acc_mid, c.pred.acc_mid);
    let (lo, hi, _, gain, chance) = card.pp.unwrap();
    assert!(close(gain, c.gain.unwrap()) && close(chance, c.p_up.unwrap()) && close(lo, c.pp_lo.unwrap()) && close(hi, c.pp_hi.unwrap()));
    // The PP gain comes from the PP target, the shown accuracy from the display target.
    assert!(c.pp_pred.as_ref().unwrap().mu != c.pred.mu);
    let fs = Engine::analyse(&path, 1.2, &e.prm).unwrap();
    e.store(&w.db, c.sha.as_ref().unwrap(), &fs).unwrap();
    assert!(!e.card(&w.db, &path, 1.2, &n).unwrap().unwrap().ranked);
}
