//! The session: form, warm-up and freshness from events alone. Python's correction without a fitted form prior is gone.
use ms_data::store::{annotate_attempt, connect, log_event, Event};
use ms_model::personal::{Completion, WarmModels};
use ms_model::predict::{Consts, Mode, Slopes};
use ms_model::session::Session;
use ms_model::warmup::{activation, session_minutes, Done, FormPrior, Model};
use ms_rec::recommender::family_transfer;
use serde_json::{json, Map, Value};
use std::sync::Arc;

fn consts() -> Arc<Consts> {
    Arc::new(Consts::standard())
}

fn model(modes: Vec<(i64, Mode)>, history: Vec<Completion>) -> Arc<Model> {
    let c = consts();
    let prior = FormPrior { global_variance: 0.05, skill_variance: 0.12, by_key: Default::default(), noise_variance: 0.04 };
    Arc::new(Model::new(&WarmModels { modes: modes.into_iter().collect(), history, ..WarmModels::default() }, Slopes::default(), prior, c.families.len()))
}

fn cold(history: Vec<Completion>) -> Arc<Model> {
    model(vec![(7, Mode { tau: 6., amplitudes: vec![0.5, 0.6, 0.9, 0.8, 0.3, 0.4, 0.4, 0.4], ..Mode::default() })], history)
}

fn session(events: Vec<Arc<Event>>, now: f64, m: Option<Arc<Model>>) -> Session {
    let c = consts();
    let transfer = Arc::new(family_transfer(&c));
    Session::new(events, now, m, c, transfer)
}

fn event(id: i64, t: f64, kind: &str, beatmap: &str, info: Value) -> Arc<Event> {
    Arc::new(Event { id, t, kind: kind.to_string(), beatmap: (!beatmap.is_empty()).then(|| beatmap.to_string()), info })
}

fn merge(mut a: Value, b: &Value) -> Value {
    for (k, v) in b.as_object().unwrap_or(&Map::new()) {
        a[k] = v.clone();
    }
    a
}

/// A start and its end per z (test_redesign.events).
fn events(zs: &[f64], start: Value, kind: &str, seconds: f64) -> Vec<Arc<Event>> {
    zs.iter().enumerate().flat_map(|(i, z)| {
        let (t, map) = (1000. + i as f64 * 150., format!("md5:m{i}"));
        let info = merge(json!({"keys": 7, "skill": "chordjack", "mu": -3., "base_mu": -3., "sd": 0.3, "length": 120., "rate": 1., "offered_rate": 1.}), &start);
        let end = if kind == "finish" { json!({"y": -3. - 0.3 * z, "played_seconds": seconds}) } else { json!({"played_seconds": seconds}) };
        [event(2 * i as i64 + 1, t, "start", &map, info), event(2 * i as i64 + 2, t + seconds, kind, &map, end)]
    }).collect()
}

fn finish(zs: &[f64]) -> Vec<Arc<Event>> {
    events(zs, json!({}), "finish", 100.)
}

/// Attempts of a PP list; the listed ones improved their counted best.
fn beaten(zs: &[f64], beats: &[usize]) -> Vec<Arc<Event>> {
    let mut out = events(zs, json!({"mode": "pp", "purpose": "farm"}), "finish", 100.);
    for &i in beats {
        out.push(event(100 + i as i64, 1000. + i as f64 * 150. + 101., "pp_result", &format!("md5:m{i}"), json!({"start_id": 2 * i + 1, "before": 500., "after": 520.})));
    }
    out
}

fn downrates(s: &Session, chart: &str) -> usize {
    s.tables(&[7]).charts.get(chart).map_or(0, |c| c.downrates.len())
}

#[test]
fn only_beaten_bests_climb_the_staircase() {
    let end = |n: usize| 1000. + n as f64 * 150.;
    let push = |zs: &[f64], beats: &[usize]| session(beaten(zs, beats), end(zs.len()), None).push(Some(7));
    let steps: Vec<f64> = (0..4).map(|n| push(&[0.; 3], &(0..n).collect::<Vec<_>>())).collect();
    assert_eq!(steps, [0., 1., 2., 3.]);
    assert_eq!(push(&[0., 0., 0., -0.2], &[0, 1, 2]), 1.);
    assert_eq!(push(&[0., 0., 0., 0.3], &[0, 1, 2]), 3., "missed the best but met the prediction");
    assert_eq!(push(&[0., -3.], &[0]), 0.);
    let mixed = [0., 0., 0., 0., 0., 0., -0.5, 0., 0., -0.5];
    assert_eq!(push(&mixed, &[0, 1, 2, 3, 4, 5, 7, 8]), 4., "two beats per miss hold the position");
    for (zs, n) in [(&[-1.5][..], 1), (&[2.], 1), (&[1., 1., 1.], 3), (&[0.; 5], 5)] {
        assert_eq!(session(finish(zs), end(n), None).push(Some(7)), 0., "good form alone is the posterior's job");
    }
}

#[test]
fn a_downrate_needs_real_play_and_invents_no_failure() {
    let rate = json!({"rate": 0.8, "offered_rate": 1.});
    let m = || Some(cold(Vec::new()));
    let browsing = session(events(&[0.], rate.clone(), "abort", 1.), 1100., m());
    assert_eq!((downrates(&browsing, "m0"), browsing.correction(7, Some("chordjack"))), (0, 0.));
    let quit = session(events(&[0.], rate.clone(), "abort", 80.), 1100., m());
    assert_eq!((downrates(&quit, "m0"), quit.activation(7, None)), (0, 0.));
    let played = session(events(&[0.], rate, "finish", 80.), 1100., m());
    assert_eq!(downrates(&played, "m0"), 1);
    assert!(played.correction(7, Some("chordjack")).abs() < 0.03);
}

#[test]
fn quits_change_neither_ability_nor_warmth() {
    let (m, c) = (cold(Vec::new()), consts());
    let chordjack = Some(c.family("chordjack"));
    let empty = session(Vec::new(), 1200., Some(m.clone()));
    let mut ev = events(&[0.], json!({"rate": 0.75, "offered_rate": 1.19}), "abort", 100.);
    ev[1] = event(2, 1100., "abort", "md5:m0", json!({"played_seconds": 100., "partial_stats": {"geki": 1, "0": 200}}));
    let quit = session(ev, 1200., Some(m));
    assert_eq!(quit.correction(7, Some("chordjack")), 0.);
    assert_eq!(empty.correction(7, Some("chordjack")), 0.);
    assert_eq!(quit.activation(7, chordjack), 0.);
    assert_eq!(downrates(&quit, "m0"), 0);
}

#[test]
fn an_interruption_is_not_negative_ability_or_a_downrate() {
    let ev = |reason: &str| vec![
        event(1, 1000., "start", "md5:a", json!({"keys": 7, "skill": "chordjack", "length": 200., "rate": 0.9, "offered_rate": 1.1})),
        event(2, 1110., "abort", "md5:a", json!({"start_id": 1, "played_seconds": 110.})),
        event(3, 1120., "attempt_note", "md5:a", json!({"start_id": 1, "reason": reason})),
    ];
    let interrupted = session(ev("interrupted"), 1130., Some(cold(Vec::new())));
    assert_eq!(interrupted.correction(7, Some("chordjack")), 0.);
    assert_eq!((downrates(&interrupted, "a"), interrupted.activation(7, None)), (0, 0.));
    let failed = session(ev("failed"), 1130., Some(cold(Vec::new())));
    assert_eq!((failed.correction(7, Some("chordjack")), downrates(&failed, "a")), (0., 0));
    assert!(failed.attempts[0].num("y").is_none(), "never a full score invented");
}

#[test]
fn attempt_notes_are_append_only_and_idempotent() {
    let dir = std::env::temp_dir().join(format!("ms-notes-{}", std::process::id()));
    let db = connect(&dir.join("rec.db")).unwrap();
    let start = log_event(&db, "start", Some("md5:a"), Some(1000.), &json!({"sha": "a".repeat(64), "rate": 1.24})).unwrap();
    log_event(&db, "abort", Some("md5:a"), Some(1040.), &json!({"start_id": start, "played_seconds": 37.})).unwrap();
    let rows = |q: &str| -> Vec<(i64, String)> { db.prepare(q).unwrap().query_map([], |r| Ok((r.get(0)?, r.get(4)?))).unwrap().map(Result::unwrap).collect() };
    let original = rows("SELECT * FROM events ORDER BY id");
    let note = annotate_attempt(&db, start, "failed", "Player thinks this failed.", 0.5).unwrap();
    assert_eq!(annotate_attempt(&db, start, "failed", "Player thinks this failed.", 0.5).unwrap(), note);
    assert_eq!(original, rows(&format!("SELECT * FROM events WHERE id<{note} ORDER BY id")));
    assert_eq!(rows("SELECT * FROM events WHERE kind='attempt_note'").len(), 1);
    drop(db);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn completed_history_seeds_a_restart_once_and_transfers_only_partly() {
    let c = consts();
    let family = |s: &str| c.family(s);
    let done = Completion { end: 1120., t: 1000., keys: 7, family: family("chordjack"), played_seconds: 120., key: Some("a".into()), chart: "a".into() };
    let s = session(Vec::new(), 1120., Some(cold(vec![done.clone()])));
    assert_eq!(s.activation(7, Some(family("chordjack"))), 2.);
    let (ln, jack) = (s.activation(7, Some(family("ln"))), s.activation(7, Some(family("jack"))));
    assert!(0. < ln && ln < jack && jack < 2., "{ln} {jack}");
    assert_eq!(s.activation(4, Some(family("chordjack"))), 0.);
    let twice = cold(vec![done.clone(), done.clone()]);
    assert_eq!(session(Vec::new(), 1120., Some(twice)).activation(7, Some(family("chordjack"))), 2.);
    assert_eq!(session(Vec::new(), 5000., Some(cold(vec![done.clone()]))).activation(7, Some(family("chordjack"))), 0.);
    let reset = session(vec![event(1, 1121., "reset", "", json!({}))], 1122., Some(cold(vec![done])));
    assert_eq!(reset.activation(7, Some(family("chordjack"))), 0.);
}

#[test]
fn a_normal_cold_loss_is_not_counted_again_as_bad_form() {
    let m = cold(Vec::new());
    let mut ev = finish(&[0.]);
    ev[0] = event(1, 1000., "start", "md5:m0", json!({"keys": 7, "skill": "chordjack", "mu": -2.5, "base_mu": -3., "cold_penalty": 0.5, "sd": 0.3, "length": 120., "rate": 1.}));
    ev[1] = event(2, 1100., "finish", "md5:m0", json!({"y": -2.5, "played_seconds": 100.}));
    let s = session(ev, 1101., Some(m.clone()));
    assert!(s.correction(7, Some("chordjack")).abs() < 1e-12);
    assert!(m.penalty(&consts(), 7, Some("chordjack"), s.activation(7, None), 0., None, s.session_clock()) > 0.);
}

fn paired(pp_y: f64, y_lazer: f64) -> Vec<Arc<Event>> {
    vec![event(1, 1000., "start", "md5:a", json!({"keys": 7, "skill": "chordstream", "length": 120, "mu": -2.8, "base_mu": -3., "cold_penalty": 0.2, "sd": 0.2, "shown_mu": -2.8,
                                                  "display_mu": -3.1, "display_base_mu": -3.3, "display_cold_penalty": 0.2, "display_sd": 0.2, "display_sdm": 0.1, "display_sda": 0.1})),
         event(2, 1120., "finish", "md5:a", json!({"start_id": 1, "y": pp_y, "y_lazer": y_lazer, "key": "a"}))]
}

#[test]
fn each_session_head_compares_its_own_frozen_target() {
    let m = cold(Vec::new());
    let s = session(paired(-2.4, -3.1), 1120., Some(m.clone()));
    let view = s.display_view(m.clone());
    let family = s.consts.family("chordstream");
    assert!(s.correction(7, Some("chordstream")) > 0.);
    assert!(view.correction(7, Some("chordstream")).abs() < 1e-12);
    assert!((view.activation(7, Some(family)) - 2.).abs() < 1e-9);
}

#[test]
fn a_legacy_completion_keeps_warmth_without_a_native_forecast() {
    let ev = vec![event(1, 1000., "start", "md5:a", json!({"keys": 7, "skill": "chordstream", "length": 120, "mu": -3., "base_mu": -3., "sd": 0.2})),
                  event(2, 1120., "finish", "md5:a", json!({"start_id": 1, "y": -2., "y_lazer": -2.3, "key": "a"}))];
    let m = cold(Vec::new());
    let view = session(ev, 1120., Some(m.clone())).display_view(m);
    assert!((view.activation(7, Some(view.consts.family("chordstream"))) - 2.).abs() < 1e-9);
    assert_eq!(view.correction(7, Some("chordstream")), 0.);
}

#[test]
fn the_warm_up_override_reaches_both_targets() {
    let m = cold(Vec::new());
    assert!(!session(Vec::new(), 1000., Some(m.clone())).display_view(m.clone()).warm_flag);
    let skip = vec![event(1, 999., "skip_warmup", "", json!({}))];
    assert!(session(skip, 1000., Some(m.clone())).display_view(m).warm_flag);
}

#[test]
fn retry_familiarity_reads_both_mod_formats() {
    for (metadata, eligible) in [
        (json!({"mods_list": [{"acronym": "rd"}]}), false), (json!({"mods": ["RD"]}), false), (json!({"mods": [], "mods_list": [{"acronym": "RD"}]}), false),
        (json!({"mods": ["rd"], "mods_list": []}), false), (json!({}), true), (json!({"mods": null, "mods_list": null}), true),
        (json!({"mods": [null, 42, {}, true], "mods_list": null}), true), (json!({"mods": 42, "mods_list": {"acronym": "malformed"}}), true),
        (json!({"mods": ["NF"], "mods_list": [{"acronym": "HD"}]}), true),
    ] {
        let m = cold(Vec::new());
        let mut ev = paired(-2.8, -3.);
        ev[0] = event(1, 1000., "start", "md5:chart", merge(ev[0].info.clone(), &metadata));
        ev[1] = event(2, 1120., "finish", "md5:chart", merge(ev[1].info.clone(), &json!({"prediction_valid": false, "played_seconds": 120.})));
        let s = session(ev, 1121., Some(m.clone()));
        assert_eq!(s.retry_played.contains("md5:chart"), eligible, "{metadata}");
        assert_eq!(s.display_view(m).retry_played, s.retry_played);
        assert!(s.played.contains("md5:chart"), "shuffled effort still counts as played");
    }
}

#[test]
fn the_session_clock_counts_every_keymode_and_restarts_after_a_gap() {
    let done = |end: f64, keys: i64| Done { end, begun: end - 120., seconds: 120., keys: Some(keys), family: 0, identity: end.to_string() };
    let history = vec![done(1120., 7), done(1300., 4), done(1500., 7)];
    assert!((session_minutes(&history, 1600.) - 10.).abs() < 1e-9);
    let gap = ms_model::personal::GAP;
    assert_eq!(session_minutes(&history, 1500. + gap + 1.), 0.);
    let mut later = history.clone();
    later.push(done(1500. + gap + 200., 7));
    assert!((session_minutes(&later, 1500. + gap + 200.) - 2.).abs() < 1e-9);
    assert_eq!(activation(&history, 1600., 7, None, &[]) > 0., true);
}

#[test]
fn session_warm_up_and_fatigue_reach_the_live_penalty() {
    use ms_model::personal::{FATIGUE_ONSET, SESSION_TAU};
    let c = consts();
    let flat = |session, fatigue| Mode { tau: 6., log_rate: true, amplitudes: vec![0.; c.families.len()], session, fatigue };
    let m = model(vec![(7, flat(0.2, 0.1))], Vec::new());
    let penalty = |m: &Model, minutes, clock| m.penalty(&c, 7, Some("chordjack"), minutes, 0., Some(1.), clock);
    assert!((penalty(&m, 0., 0.) - 0.2).abs() < 1e-12);
    assert!((penalty(&m, 0., SESSION_TAU) - 0.2 / std::f64::consts::E).abs() < 1e-12);
    assert!((penalty(&m, 30., FATIGUE_ONSET + 60.) - 0.1).abs() < 1e-3);
    assert_eq!(penalty(&model(vec![(7, flat(0., 0.))], Vec::new()), 0., 0.), 0.);
}

#[test]
fn the_form_readout_names_the_reason() {
    let c = consts();
    let plays: Vec<Completion> = (0..16).map(|i| Completion { end: 1000. + 600. * i as f64, t: 700. + 600. * i as f64, keys: 7, family: c.family("chordjack"), played_seconds: 300., key: Some(i.to_string()), chart: i.to_string() }).collect();
    let m = model(vec![(7, Mode { tau: 6., amplitudes: vec![0.; c.families.len()], session: 0.2, fatigue: 0.2, ..Mode::default() })], plays);
    assert_eq!(session(Vec::new(), 1100., Some(m.clone())).form_state(7).0, Some("warming up"));
    assert_eq!(session(Vec::new(), 1000. + 600. * 15. + 60., Some(m.clone())).form_state(7).0, Some("fading"));
    let calm = session(Vec::new(), 1000. + 600. * 5., Some(m.clone())).form_state(7);
    assert!(calm.0.is_none() && calm.1 < 0.08, "{calm:?}");
    let skip = vec![event(1, 1050., "skip_warmup", "", json!({}))];
    assert_eq!(session(skip, 1100., Some(m)).form_state(7), (None, 0.));
}

#[test]
fn the_prior_changes_rate_demand_by_readiness() {
    let (c, m) = (consts(), model(Vec::new(), Vec::new()));
    let capacity = |minutes| m.capacity(&c, 7, Some("chordstream"), minutes);
    assert!((capacity(0.) - 0.9).abs() < 1e-12);
    assert!(0.9 < capacity(4.) && capacity(4.) < capacity(12.) && capacity(12.) < 1.);
    let penalty = |seconds, slope| m.penalty(&c, 7, Some("chordstream"), 0., seconds, Some(slope), 0.);
    assert!(penalty(120., 5.) > penalty(120., 2.));
    assert!(penalty(240., 3.) < penalty(0., 3.));
}

#[test]
fn a_warmed_up_player_pays_no_session_warm_up_on_a_chart() {
    use ms_model::personal::SESSION_TAU;
    // v0.5.8: the per-family table held session_cold at clock 0, every chart kept it all session.
    let c = consts();
    let m = model(vec![(7, Mode { tau: 4., log_rate: true, amplitudes: vec![0.1; c.families.len()], session: 0.2, fatigue: 0. })], (0..400).map(|i| {
        Completion { end: 1000. + 120. * i as f64, t: 880. + 120. * i as f64, keys: 7, family: 0, played_seconds: 120., key: Some(i.to_string()), chart: i.to_string() }
    }).collect());
    let early = session(Vec::new(), 1001., Some(m.clone())).tables(&[7]);
    assert!((early.by_keys[&7].session - 0.2 * (-2. / SESSION_TAU).exp()).abs() < 1e-3, "{}", early.by_keys[&7].session);
    let late = session(Vec::new(), 1000. + 120. * 399. + 1., Some(m)).tables(&[7]);
    assert!(late.by_keys[&7].session.abs() < 1e-6, "{}", late.by_keys[&7].session);
}

#[test]
fn a_marathon_stays_one_attempt_until_the_gap() {
    use ms_model::session::SESSION_GAP;
    let now = 10000.;
    let ev = vec![event(1, now, "start", "md5:a", json!({"mu": -3., "sd": 0.4, "keys": 7, "skill": "chordstream", "length": 4000, "ranked": false, "purpose": "nps"})),
                  event(2, now + 4000., "finish", "md5:a", json!({"start_id": 1, "prediction_valid": true, "y": -3.1}))];
    let s = session(ev.clone(), now + 4001., None);
    assert_eq!(s.attempts.len(), 1);
    assert!(s.attempts[0].z.unwrap() > 0.);
    assert_ne!(s.phase(), "warmup");
    assert!(!session(ev[..1].to_vec(), now + 3500., None).events.is_empty());
    assert!(session(ev, now + 4001. + SESSION_GAP, None).attempts.is_empty());
}

#[test]
fn taste_needs_a_real_completion_and_counts_it_once() {
    let dir = std::env::temp_dir().join(format!("ms-taste-{}", std::process::id()));
    let db = connect(&dir.join("rec.db")).unwrap();
    let now = ms_data::store::now();
    log_event(&db, "next", Some("a"), Some(now - 900.), &json!({"mode": "nps", "keys": 7, "group": "chords"})).unwrap();
    let taste = |db| ms_rec::builder::taste(db, ms_data::store::now()).unwrap();
    assert!(taste(&db).is_empty());
    for g in ["chords", "ln"] {
        for i in 0..6 {
            let map = format!("{g}{i}");
            let offer = log_event(&db, "offer", Some(&map), Some(now - 800. + i as f64), &json!({"mode": "nps", "keys": 7, "group": g})).unwrap();
            if g == "chords" && i < 4 {
                let start = log_event(&db, "start", Some(&map), Some(now - 700. + i as f64), &json!({"offer_id": offer})).unwrap();
                for _ in 0..2 {
                    log_event(&db, "finish", Some(&map), Some(now - 600. + i as f64), &json!({"start_id": start, "key": start.to_string()})).unwrap();
                }
            }
        }
    }
    let t = taste(&db);
    let (chords, ln) = (t[&(7, "chords".to_string())], t[&(7, "ln".to_string())]);
    assert!(0. < chords && chords <= 0.25 && -0.25 <= ln && ln < 0., "{chords} {ln}");
    log_event(&db, "taste_reset", None, Some(ms_data::store::now()), &json!({})).unwrap();
    assert!(taste(&db).is_empty());
    drop(db);
    let _ = std::fs::remove_dir_all(dir);
}
