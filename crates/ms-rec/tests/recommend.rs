//! Ledger, identity and import, session and selection.
use ms_data::imports::{add_live_score, realm_rows};
use ms_data::store::{acc320, acc_lazer, connect, log_event, score_key, set_tracking, tracking_enabled, upsert_score, variant, Db, Event};
use ms_model::personal::WarmModels;
use ms_model::predict::{Consts, HashMap, Slopes};
use ms_model::session::{Session, SESSION_GAP};
use ms_model::warmup::{FormPrior, Model};
use ms_rec::evidence::{stable_evidence, Evidence};
use ms_rec::history::{history_text, play_history, recover_results, Shown};
use ms_rec::pp::Ledger;
use ms_rec::recommender::family_transfer;
use serde_json::{json, Value};
use std::sync::Arc;

fn db() -> (tempdir::Dir, Db) {
    let d = tempdir::Dir::new();
    let c = connect(&d.0.join("rec.db")).unwrap();
    (d, c)
}

mod tempdir {
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    pub struct Dir(pub PathBuf);

    impl Dir {
        pub fn new() -> Dir {
            static N: AtomicUsize = AtomicUsize::new(0);
            let p = std::env::temp_dir().join(format!("ms-rec-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
            std::fs::create_dir_all(&p).unwrap();
            Dir(p)
        }
    }

    impl Drop for Dir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

fn event(t: f64, kind: &str, beatmap: Option<&str>, info: Value) -> Arc<Event> {
    static N: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(1);
    Arc::new(Event { id: N.fetch_add(1, std::sync::atomic::Ordering::Relaxed), t, kind: kind.to_string(), beatmap: beatmap.map(String::from), info })
}

fn session(events: Vec<Arc<Event>>, now: f64) -> Session {
    let c = Arc::new(Consts::standard());
    Session::new(events, now, None, c.clone(), Arc::new(family_transfer(&c)))
}

/// A session whose form posterior runs, as every fitted one does.
fn form_session(events: Vec<Arc<Event>>, now: f64) -> Session {
    let c = Arc::new(Consts::standard());
    let prior = FormPrior { global_variance: 0.05, skill_variance: 0.12, by_key: HashMap::default(), noise_variance: 0.04 };
    let model = Model::new(&WarmModels::default(), Slopes::default(), prior, c.families.len());
    Session::new(events, now, Some(Arc::new(model)), c.clone(), Arc::new(family_transfer(&c)))
}

/// start + finish per z, as Session reads them.
fn attempts(t0: f64, zs: &[f64], keys: i64, skill: &str, purpose: &str, bid: Option<i64>, spacing: f64) -> Vec<Arc<Event>> {
    zs.iter().enumerate().flat_map(|(i, z)| {
        let t = t0 + i as f64 * spacing;
        let b = bid.unwrap_or(100 + i as i64).to_string();
        [event(t, "start", Some(&b), json!({"mu": -3., "sd": 0.4, "keys": keys, "skill": skill, "ranked": true, "purpose": purpose, "length": 120})),
         event(t + 120., "finish", Some(&b), json!({"y": -3. - 0.4 * z}))]
    }).collect()
}

fn farm(t0: f64, zs: &[f64]) -> Vec<Arc<Event>> {
    attempts(t0, zs, 7, "stream", "farm", None, 300.)
}

/// Positive means good form.
fn form(s: &Session, keys: i64, skill: Option<&str>) -> f64 {
    -s.correction(keys, skill) / 0.3
}

fn now() -> f64 {
    ms_data::store::now()
}

fn brute(best: &[f64]) -> f64 {
    let mut p = best.to_vec();
    p.sort_by(|a, b| b.total_cmp(a));
    let n = p.len();
    p.iter().enumerate().map(|(i, x)| x * 0.95f64.powi(i as i32)).sum::<f64>() + 416.6667 * (1. - 0.995f64.powi(n.min(1000) as i32))
}

fn ledger(values: &[f64]) -> Ledger {
    let best: HashMap<i64, (f64, bool, Option<String>)> = values.iter().enumerate().map(|(i, p)| (i as i64, (*p, false, None))).collect();
    Ledger::new(best)
}

#[test]
fn ledger_cases() {
    let best: Vec<f64> = (0..300).map(|i| 50. + (i as f64 * 37.7) % 550.).collect();
    let l = ledger(&best);
    assert!((l.total() - brute(&best)).abs() < 1e-6);
    let at = best.iter().enumerate().filter(|(_, p)| **p < 300.).max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    let old = best[at];
    for new in [old + 1., 590., old - 20., 0.] {
        let mut changed = best.clone();
        changed[at] = old.max(new);
        assert!((l.delta(old, new) - (brute(&changed) - brute(&best))).abs() < 1e-6, "{new}");
    }
    for new in [10., 250., 700.] {
        let mut changed = best.clone();
        changed.push(new);
        assert!((l.delta(0., new) - (brute(&changed) - brute(&best))).abs() < 1e-6, "{new}");
    }
    // >99 % refarm: a small acc gain near the ceiling is still a real (small) gain
    assert!(l.delta(old, old * 1.004) > 0.);
}

#[test]
fn mod_alternatives_compete() {
    // NM/HT/DT on one difficulty replace the same counted score: never added twice.
    let l = ledger(&[300., 200.]);
    let dt = l.delta(300., 330.);
    assert!((dt - (brute(&[330., 200.]) - brute(&[300., 200.]))).abs() < 1e-9);
    assert!(dt < l.delta(300., 320.) + dt);
}

#[test]
fn ranked_variants() {
    let v = |m: Value| variant(m.as_array().unwrap());
    assert_eq!(v(json!([{"acronym": "DT", "settings": {"speed_change": 1.2}}])), (Some(1.2), false));
    assert_eq!(v(json!([{"acronym": "DT", "settings": {"adjust_pitch": true}}])), (Some(1.5), true));
    assert_eq!(v(json!([{"acronym": "HT"}, {"acronym": "NF"}])), (Some(0.75), true));
    assert_eq!(v(json!([{"acronym": "CS"}])), (Some(1.), false));
    assert_eq!(v(json!([{"acronym": "WU"}])), (None, false));
}

const STATS: [i64; 6] = [900, 50, 20, 10, 5, 15];

fn hits() -> Value {
    json!({"geki": 900, "300": 50, "katu": 20, "100": 10, "50": 5, "0": 15})
}

/// A lazer_scores line.
fn raw(i: i64, total: i64, online: i64, user: i64, date: &str) -> Value {
    json!({"id": format!("uuid-{i}"), "hash": "a".repeat(64), "acc": 0.95, "total": total, "combo": 100, "rank": 3, "date": date,
           "mods": "[{\"acronym\": \"HT\"}]", "stats": "{\"perfect\":900,\"great\":50,\"good\":20,\"ok\":10,\"meh\":5,\"miss\":15}", "user": "robby250", "user_id": user,
           "online_id": online, "legacy_online_id": -1, "legacy": false, "client": "2026.921.0-lazer", "beatmap_id": 42, "md5": "m".repeat(32), "status": 1})
}

#[test]
fn import_idempotent_and_live_then_realm() {
    let (_d, db) = db();
    let live = json!({"sha256": "a".repeat(64), "score": 777, "hits": hits(), "mods_list": [{"acronym": "HT"}], "played": "2026-09-20T10:00:00Z"});
    assert!(add_live_score(&db, &live, true, None).unwrap().1);
    let rows = realm_rows(&[raw(1, 777, 5, 2653437, "2026-09-20T10:00:00Z"), raw(2, 900, 0, 999, "2026-09-20T10:00:00Z")], Some(2653437));
    assert_eq!(rows.len(), 1, "another account's score is not ours");
    for _ in 0..2 {
        for r in rows.clone() {
            upsert_score(&db, r).unwrap();
        }
    }
    let got: Vec<(i64, i64, String)> = db.prepare("SELECT online_id, fresh, local_id FROM scores").unwrap().query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?))).unwrap().map(Result::unwrap).collect();
    assert_eq!(got, [(5, 1, "uuid-1".to_string())]);
    // the same result polled again later is not a second attempt
    assert!(!add_live_score(&db, &live, false, None).unwrap().1);
}

#[test]
fn old_result_is_history_not_form() {
    // A viewed old result: its start is voided, so the session has no attempt and no residual.
    let t = now();
    let s = session(vec![event(t - 60., "start", Some("42"), json!({"mu": -3., "sd": 0.4, "keys": 7})), event(t - 50., "void", Some("42"), json!({})),
                         event(t - 40., "finish", Some("42"), json!({"y": -5.}))], t);
    assert!(s.attempts.is_empty() && form(&s, 7, None) == 0.);
}

#[test]
fn session_ramp_and_recovery() {
    let t0 = now() - 4000.;
    assert_eq!(session(Vec::new(), t0).phase(), "warmup");
    assert_eq!(session(attempts(t0, &[0.2, 0.3], 7, "stream", "warmup", None, 300.), t0 + 700.).phase(), "warmup");
    assert_eq!(session(farm(t0, &[0.2, 0.3, 0.8, 0.6, 0.7, 0.9]), t0 + 2000.).phase(), "build", "strong but no beaten PP best: the staircase pushes only on beats");
    let one_bad = session(farm(t0, &[0.8, 0.6, 0.7, 0.9, 0.5, -2.]), t0 + 2000.);
    assert_eq!(one_bad.phase(), "recover");
    assert_eq!(session(farm(t0, &[0.5, 0.4, 0.3, 0.2, -1.5, -1.8]), t0 + 2000.).phase(), "recover");
}

#[test]
fn keymode_form_does_not_transfer_fully() {
    let t0 = now() - 3000.;
    let s = form_session(farm(t0, &[1.5, 1.4, 1.6, 1.5, 1.3]), t0 + 1600.);
    assert!(form(&s, 7, Some("stream")) > 0.);
    assert_eq!(form(&s, 4, Some("stream")), 0., "another keymode is not evidence");
    let s2 = form_session(attempts(t0, &[-1., -1.2, -0.9, -1.1], 7, "jack", "farm", None, 300.), t0 + 1300.);
    assert!(form(&s2, 7, Some("ln")) > form(&s2, 7, Some("jack")), "bad jack results do not erase LN");
}

#[test]
fn session_boundary_and_restart() {
    let t0 = now() - 10000.;
    let ev = farm(t0, &[0.5, 0.6]);
    assert_eq!(session(ev.clone(), t0 + 900.).attempts.len(), 2, "an app restart rebuilds the same session");
    assert!(session(ev.clone(), t0 + 900. + SESSION_GAP + 60.).attempts.is_empty(), "an hour without gameplay: new session");
    let mut reset = ev;
    reset.push(event(t0 + 1000., "reset", None, json!({})));
    assert!(session(reset, t0 + 1001.).attempts.is_empty());
}

fn fresh(s: &Session, map: &str, length: f64) -> f64 {
    s.freshness(map, length, None, None, ms_model::session::EXPOSURE_S)
}

fn fresh_in(s: &Session, map: &str, mode: &str, pool: f64) -> f64 {
    s.freshness(map, 120., Some(mode), Some(pool), ms_model::session::EXPOSURE_S)
}

#[test]
fn recent_maps_stay_back_after_restart_and_return_after_variety() {
    let t0 = now() - 1000.;
    let mut ev = attempts(t0, &[-0.2], 7, "stream", "farm", Some(7), 300.); // near miss on difficulty 7
    let s = session(ev.clone(), t0 + 200.);
    assert!(fresh(&s, "7", 540.) < 0.1, "9-minute farm: not immediately");
    assert!(fresh(&s, "7", 100.) < 0.1, "Next is variety; manual retry is still available");
    assert!(fresh(&session(ev.clone(), t0 + 200. + SESSION_GAP + 60.), "7", 100.) < 0.15);
    ev.extend(attempts(t0 + 200., &[0.1, 0., 0.2], 7, "stream", "farm", None, 200.)); // three other maps in between
    assert!(fresh(&session(ev, t0 + 900.), "7", 540.) > 0.2);
}

#[test]
fn offered_chart_returns_after_a_share_of_the_pool_not_a_clock() {
    let t = now();
    let offer = |dt: f64, b: String| event(t - dt, "offer", Some(&b), json!({"mode": "nps"}));
    let mut ev = vec![offer(5000., "9".to_string())];
    ev.extend((0..10).map(|i| offer(4000. - i as f64, (100 + i).to_string())));
    let s = session(ev, t);
    assert!(fresh_in(&s, "9", "nps", 400.) < 0.01, "10 of 400: still held back, hours later");
    let half = fresh_in(&s, "9", "nps", 20.);
    assert!(0.2 < half && half < 0.3, "half the pool since: a quarter back");
    assert!(fresh_in(&s, "9", "pp", 40.) < 1e-3, "offers in another tab do not count");
}

#[test]
fn played_and_offered_maps_stay_back_for_days_until_the_pool_cycles() {
    let (_d, db) = db();
    let t = now();
    let map = format!("md5:{}", "a".repeat(32));
    let day = 86400.;
    log_event(&db, "offer", Some(&map), Some(t - 5. * day), &json!({"mode": "pp", "md5": "a".repeat(32), "forecast": "x".repeat(5000)})).unwrap();
    log_event(&db, "start", Some(&map), Some(t - 5. * day + 10.), &json!({"md5": "a".repeat(32), "length": 120, "mode": "pp"})).unwrap();
    log_event(&db, "finish", Some(&map), Some(t - 5. * day + 130.), &json!({"md5": "a".repeat(32)})).unwrap();
    let loaded = |at: f64| session(ms_data::events::load_events(&db, at - 30. * day).unwrap(), at);
    let s = loaded(t);
    assert!(s.all.is_empty(), "form and predictions still see 3 days only");
    assert!(fresh_in(&s, &map, "pp", 40.) < 1e-3, "5 days later, nothing else offered");
    assert_eq!(fresh_in(&loaded(t + 10. * day), &map, "pp", 40.), 1.);
    for i in 0..40 {
        let m5 = format!("{i:032}");
        let b = format!("md5:{m5}");
        let at = t - day + 300. * i as f64;
        log_event(&db, "start", Some(&b), Some(at), &json!({"md5": m5, "length": 120})).unwrap();
        log_event(&db, "finish", Some(&b), Some(at + 130.), &json!({"md5": m5})).unwrap();
        log_event(&db, "offer", Some(&b), Some(at), &json!({"md5": m5, "mode": "pp"})).unwrap();
    }
    let back = fresh_in(&loaded(t), &map, "pp", 40.);
    assert!(0.4 < back && back < 0.5, "{back}");
}

#[test]
fn skip_is_not_permanent_and_render_is_not_exposure() {
    let t = now();
    let ev = vec![event(t - 30., "start", Some("1"), json!({})), event(t - 20., "skip", Some("9"), json!({}))];
    let s = session(ev.clone(), t);
    assert!(fresh(&s, "9", 120.) < 0.2 && fresh(&s, "10", 120.) == 1.);
    assert!(fresh(&session(ev, t + 3. * 86400.), "9", 120.) > 0.9);
}

#[test]
fn lazer_accuracy_scale() {
    // The shown accuracy is lazer's (PERFECT 305): a Cherry Pop result read 95.78 there, 94.26 as acc320.
    let s = [2425, 1140, 164, 26, 25, 48];
    assert_eq!(((acc_lazer(&s) * 1e4).round(), (acc320(&s).0 * 1e4).round()), (9578., 9426.));
}

#[test]
fn live_nomod_is_ranked() {
    // tosu's empty mod list is NoMod (ranked, counts in the ledger at once); no list at all stays unknown.
    let (_d, db) = db();
    let base = json!({"sha256": "s".repeat(64), "rate": 1., "hits": {"geki": 900, "300": 90, "katu": 5, "100": 3, "50": 1, "0": 1}});
    let with = |score: i64, mods: Option<Value>| {
        let mut v = base.clone();
        v["score"] = json!(score);
        if let Some(m) = mods {
            v["mods_list"] = m;
        }
        add_live_score(&db, &v, true, None).unwrap().0.unwrap()
    };
    let (k1, k2) = (with(1, Some(json!([]))), with(2, None));
    let ranked = |k: &str| db.query_row("SELECT ranked FROM scores WHERE key=?", [k], |r| r.get::<_, i64>(0)).unwrap();
    assert_eq!((ranked(&k1), ranked(&k2)), (1, 0));
}

#[test]
fn abort_ends_before_the_restart() {
    // An abort logged at its own end time pairs with its own start, so the restarted play keeps its result.
    let (_d, db) = db();
    let start = json!({"mu": -3., "sd": 0.4, "keys": 7});
    log_event(&db, "start", Some("7"), Some(100.), &start).unwrap();
    log_event(&db, "abort", Some("7"), Some(160.), &json!({"progress": 0.4})).unwrap();
    log_event(&db, "start", Some("7"), Some(160.), &start).unwrap(); // the watcher queues the abort first, same second
    log_event(&db, "finish", Some("7"), Some(400.), &json!({"y": -3.2})).unwrap();
    let s = session(ms_data::events::load_events(&db, 0.).unwrap(), 500.);
    assert_eq!(s.attempts.iter().map(|a| a.finished).collect::<Vec<_>>(), [false, true]);
    assert!(s.attempts[1].z.unwrap() > 0.);
}

fn iso(t: f64) -> String {
    chrono::DateTime::from_timestamp_millis((t * 1000.) as i64).unwrap().to_rfc3339()
}

#[test]
fn tracking_history_and_recovery() {
    // Paused attempts never train after reimport; results retain their own pre-play expectation.
    let (d, first) = db();
    let t = now() - 1000.;
    let live = |at: f64| json!({"sha256": "a".repeat(64), "score": 777, "hits": hits(), "mods_list": [{"acronym": "HT"}], "rate": 0.75, "played": iso(t + at)});
    let k = add_live_score(&first, &live(20.), true, None).unwrap().0.unwrap();
    set_tracking(&first, false, Some(t + 40.), Some(t + 30.)).unwrap();
    drop(first);
    let db = connect(&d.0.join("rec.db")).unwrap();
    assert!(!tracking_enabled(&db).unwrap(), "restart preserves the pause");
    assert!(add_live_score(&db, &live(50.), true, None).unwrap().0.is_none());
    set_tracking(&db, true, Some(t + 60.), None).unwrap();
    assert!(tracking_enabled(&db).unwrap());
    assert!(add_live_score(&db, &live(70.), true, Some(&json!({"t": t + 45., "end": t + 70.}))).unwrap().0.is_none());
    for (i, at) in [20., 50., 70., 100.].into_iter().enumerate() {
        let rows = realm_rows(&[raw(i as i64, 777, 0, 2653437, &iso(t + at))], Some(2653437));
        upsert_score(&db, rows[0].clone()).unwrap();
    }
    let count = |db: &Db| db.query_row("SELECT COUNT(*) FROM scores", [], |r| r.get::<_, i64>(0)).unwrap();
    assert_eq!(count(&db), 2, "identical separate keyboard plays survive");
    log_event(&db, "start", Some("42"), Some(t + 80.), &json!({"sha": "a".repeat(64), "rate": 0.75, "mods": ["HT"], "mu": -2.1, "sd": 0.2, "expected": 0.99, "predicted_at": t + 80., "title": "Played chart", "length": 30})).unwrap();
    // Polling resumes late and calls this an abort. Recovery must insert the actual earlier finish.
    log_event(&db, "abort", Some("42"), Some(t + 110.), &json!({"progress": 0.7})).unwrap();
    log_event(&db, "start", Some("99"), Some(t + 120.), &json!({"sha": "b".repeat(64), "rate": 1, "mu": -5, "sd": 0.2})).unwrap();
    assert_eq!((recover_results(&db).unwrap(), recover_results(&db).unwrap()), (1, 0));
    let history = play_history(&db, 500, false, None).unwrap();
    assert!(history[0].expected == Some(0.99) && history[0].recovered && history[0].z.unwrap() > 1.5);
    assert!(history[0].title == "Played chart" && history[1].expected.is_none());
    assert_eq!(play_history(&db, 500, true, None).unwrap().len(), 1);
    assert!(history[0].pauses.is_none(), "archive not available yet");
    db.execute_batch(ms_data::replays::SCHEMA).unwrap();
    db.execute("INSERT INTO replay_evidence(score_key,pauses) VALUES (?,?)", rusqlite::params![history[0].key, "[-1200, 76740]"]).unwrap();
    db.execute("INSERT INTO replay_evidence(score_key,pauses) VALUES (?,?)", rusqlite::params![history[1].key, "[]"]).unwrap();
    let saved = play_history(&db, 500, false, None).unwrap();
    assert_eq!(saved[0].pauses, Some(json!([-1200, 76740])));
    assert_eq!(saved[1].pauses, Some(json!([])));
    let shown = Shown { client: "lazer".into(), miss_ratio: 0. };
    assert!(history_text(&saved, &shown).contains("2 play pauses (duration unknown)"));
    assert!(!history_text(&saved[1..], &shown).contains("play pause"));
    let s = session(ms_data::events::load_events(&db, 0.).unwrap(), now());
    assert_eq!(s.attempts.iter().map(|a| a.finished).collect::<Vec<_>>(), [true]);
    // A legacy key must retain its old event references instead of becoming a duplicate.
    let old = score_key(&"a".repeat(64), 777, &STATS, None);
    db.execute("UPDATE scores SET key=? WHERE key=?", [&old, &k]).unwrap();
    let rows = realm_rows(&[raw(8, 777, 0, 2653437, &iso(t + 20.))], Some(2653437));
    assert_eq!(upsert_score(&db, rows[0].clone()).unwrap(), Some(old));
    assert_eq!(count(&db), 2);
}

fn evidence(key: String, client: &str, ln: f64) -> Evidence {
    Evidence { y: 0., y_lazer: 0., keys: 7, t: 0., chart: String::new(), sha: None, f: 0, src: String::new(), client: Some(client.to_string()), ts: None, b: None, rate: 1.,
               key, rd: false, completed: true, ln, overall: 5., family: String::new() }
}

#[test]
fn stable_player_keeps_mostly_tap_charts_with_some_ln() {
    // Less 2026-11-01: 85 of 100 stable bests were dropped because their charts held any LN.
    let rows: Vec<Evidence> = [0., 0.1, 0.35, 0.6].iter().enumerate().map(|(i, ln)| evidence(format!("st|{i}"), "stable", *ln)).collect();
    assert_eq!(stable_evidence(rows.iter()).iter().map(|r| r.ln).collect::<Vec<_>>(), [0., 0.1, 0.35]);
    let mut both = rows;
    both.extend((0..ms_rec::evidence::MIN_LAZER_LN).map(|i| evidence(format!("l|{i}"), "osu-lazer", 0.5)));
    assert_eq!(stable_evidence(both.iter()).iter().filter(|r| r.client.as_deref() == Some("stable")).count(), 1);
}

// ---- session form ----

fn played(t: f64, error: f64, skill: &str, keys: i64, chart: &str) -> Vec<Arc<Event>> {
    vec![event(t - 120., "start", Some(chart), json!({"mu": -3., "sd": 0.25, "keys": keys, "skill": skill, "length": 120})), event(t, "finish", Some(chart), json!({"y": -3. + error}))]
}

fn plays(rows: &[(f64, &str, i64, &str)]) -> Vec<Arc<Event>> {
    rows.iter().enumerate().flat_map(|(i, r)| played(1000. + 300. * i as f64, r.0, r.1, r.2, r.3)).collect()
}

#[test]
fn repeated_bad_skill_evidence_is_not_dismissed_as_unrelated_outliers() {
    let s = form_session(plays(&[(0.8, "sv", 7, "0"), (0.8, "sv", 7, "1"), (0.8, "sv", 7, "2"), (0.8, "sv", 7, "3")]), 2000.);
    assert!(s.correction(7, Some("sv")) > 0.5);
    assert!(s.correction(7, Some("chordstream")) < s.correction(7, Some("sv")));
}

#[test]
fn one_bad_chart_does_not_override_several_good_different_charts() {
    let mut rows: Vec<(f64, &str, i64, &str)> = ["0", "1", "2", "3", "4"].map(|c| (-0.15, "chordstream", 7, c)).to_vec();
    rows.push((2., "ln", 7, "5"));
    let s = form_session(plays(&rows), 2800.);
    assert!(s.correction(7, Some("chordstream")).abs() < 0.15);
    assert!(s.correction(7, Some("ln")) > s.correction(7, Some("chordstream")));
}

#[test]
fn other_keymode_and_repeated_chart_do_not_fake_independent_evidence() {
    let at = 1500.;
    let a = form_session(plays(&[(0.5, "sv", 7, "0")]), at).posterior(7);
    let mut both = plays(&[(0.5, "sv", 7, "0")]);
    both.extend(played(1200., 0.5, "sv", 4, "other"));
    assert_eq!(a, form_session(both, at).posterior(7));
    let repeated: Vec<Arc<Event>> = (0..6).flat_map(|i| played(350. + i as f64 * 130., 0.5, "sv", 7, "same")).collect();
    let c = form_session(repeated, at).posterior(7);
    assert!(a.iter().zip(&c).all(|(x, y)| (x - y).abs() < 1e-9), "{a:?} {c:?}");
}
