//! Physical execution controls; no titles or hand-picked rating targets.
mod common;
use common::*;
use ms_calc::chart::Chart;
use ms_calc::compute::compute;
use ms_calc::features::standard_params;
use ms_calc::gestures::{analyse, great_window, s, Analysis, FEATURES};
use ms_calc::model::{final_rating, gesture_rating};
use ms_calc::params::Params;
use ms_calc::val::{Dict, Val};
use serde_json::{json, Value};

const BIN: f64 = 0.5;

fn rolls(pattern: &[u8], gap: f64, n: usize, keys: usize, od: f64) -> Chart {
    rice((0..n).map(|i| (1000. + i as f64 * gap, pattern[i % pattern.len()])), keys, od)
}

fn roll4(pattern: &[u8], gap: f64) -> Chart {
    rolls(pattern, gap, 1200, 4, 8.)
}

fn repeated(pattern: &[&[u8]], gap: f64, n: usize) -> Chart {
    rice((0..n).flat_map(|i| pattern[i % pattern.len()].iter().map(move |&c| (1000. + i as f64 * gap, c))), 4, 5.)
}

fn g(c: &Chart) -> Analysis {
    analyse(c, 1., BIN, false)
}

fn mean(a: &[f64]) -> f64 {
    a.iter().sum::<f64>() / a.len().max(1) as f64
}

fn rolled(c: &Chart) -> f64 {
    mean(&g(c).series[s::ROLLED])
}

fn feature(a: &Analysis, name: &str) -> f64 {
    a.features[FEATURES.iter().position(|f| *f == name).unwrap()]
}

fn notes(c: &Chart) -> Vec<(f64, f64, u8)> {
    (0..c.len()).map(|i| (c.t[i], c.end[i], c.col[i])).collect()
}

const FULL_ROLL: [&[u8]; 5] = [&[0, 1, 2, 3], &[0, 1, 3, 2], &[1, 0, 3, 2], &[0, 3, 1, 2], &[0, 1, 2, 3, 0, 1, 3, 2, 1, 0, 3, 2, 0, 3, 1, 2]];

#[test]
fn fast_variable_and_interleaved_rolls_are_jumptrill_execution() {
    for pattern in FULL_ROLL {
        let c = roll4(pattern, 20.);
        assert!(rolled(&c) > 0.7);
        let r = run(&c, 1.);
        assert!(score(&r, "jumptrill") > score(&r, "stream"));
    }
}

#[test]
fn hand_grouping_generalizes_and_respects_center_assignment() {
    for keys in 4..=10usize {
        // Free outer fingers still roll while a spare center finger is held.
        let cols: Vec<u8> = (0..keys / 2).chain((keys + 1) / 2..keys).map(|c| c as u8).collect();
        let c = rolls(&cols, 12., 1200, keys, 8.);
        let mirror = chart(notes(&c).into_iter().map(|(t, e, k)| (t, e, keys as u8 - 1 - k)).collect(), keys, c.od);
        let (a, b) = (g(&c), g(&mirror));
        assert!(mean(&a.series[s::ROLLED]) > 0.7);
        for (x, y) in a.features.iter().zip(&b.features) {
            assert!((x - y).abs() < 1e-8);
        }
    }
}

#[test]
fn slow_rice_and_timing_infeasible_groups_do_not_become_jumptrills() {
    let c = roll4(&[0, 1, 2, 3], 70.);
    assert_eq!(rolled(&c), 0.);
    assert_eq!(score(&run(&c, 1.), "jumptrill"), 0.);
    // The same note order can fit at OD0 but not OD10. The feasible interval is for Great, not Perfect/SS with all
    // fingers simultaneous.
    let (loose, strict) = (rolls(&[0, 3, 1, 2], 30., 1200, 4, 0.), rolls(&[0, 3, 1, 2], 30., 1200, 4, 10.));
    assert!(rolled(&loose) > rolled(&strict));
    assert_eq!(rolled(&strict), 0.);
}

#[test]
fn rate_is_applied_once_and_offsets_do_not_change_gestures() {
    let c = roll4(&[0, 1, 3, 2], 30.);
    let moved = |f: &dyn Fn(f64) -> f64| chart(notes(&c).into_iter().map(|(t, e, k)| (f(t), f(e), k)).collect(), 4, c.od);
    let (a, b, z) = (analyse(&c, 1.5, BIN, false), g(&moved(&|x| x / 1.5)), analyse(&moved(&|x| x + 91234.), 1.5, BIN, false));
    for i in 0..FEATURES.len() {
        assert!((a.features[i] - b.features[i]).abs() < 1e-8 && (a.features[i] - z.features[i]).abs() < 1e-8);
    }
}

#[test]
fn reversals_and_anchors_cost_control_instead_of_erasing_all_grouping() {
    let (plain, turns) = (g(&roll4(&[0, 1, 2, 3], 20.)), g(&roll4(&[0, 1, 2, 3, 1, 0, 3, 2], 20.)));
    assert!(feature(&turns, "roll_control") > feature(&plain, "roll_control") + 0.1);
    assert!(feature(&turns, "roll_anchor") > feature(&plain, "roll_anchor") + 0.05);
    assert!(rolled(&roll4(&[0, 1, 2, 3, 3, 2, 1, 0], 20.)) < mean(&plain.series[s::ROLLED]));
    assert!(rolled(&roll4(&[0, 1, 0, 2, 0, 3], 20.)) < 0.1);
}

#[test]
fn literal_jumptrill_split_trill_and_vibro_remain_different() {
    let (jt, split) = (run(&repeated(&[&[0, 1], &[2, 3]], 60., 400), 1.), run(&repeated(&[&[0, 2], &[1, 3]], 60., 400), 1.));
    assert!(score(&jt, "jumptrill") > 0.9 * score(&jt, "overall") && score(&jt, "vibro") == 0.);
    assert!(score(&split, "splittrill") > 0.9 * score(&split, "overall"));
    assert!(score(&split, "jumptrill") == 0. && score(&split, "vibro") == 0.);
}

const CHANGING: [&[u8]; 4] = [&[0, 1, 2], &[0, 1, 3], &[0, 1, 2, 3], &[1, 2, 3]];

#[test]
fn changing_chords_can_still_be_vibro_but_stability_is_not_control() {
    let (stable, changing) = (g(&repeated(&[&[0, 1, 2, 3]], 80., 400)), g(&repeated(&CHANGING, 80., 400)));
    assert!(feature(&stable, "vibro_easy") > 2. * feature(&changing, "vibro_easy"));
    assert!(feature(&changing, "vibro_control") > feature(&stable, "vibro_control") + 0.2);
    assert!(mean(&changing.series[s::VIBRO]) > 0.05);
    assert_eq!(mean(&g(&repeated(&[&[0, 1, 2, 3]], 170., 400)).series[s::VIBRO]), 0.);
}

#[test]
fn free_fingers_can_roll_under_a_hold_without_collapsing_the_hold() {
    let c = rolls(&[0, 1, 2, 6, 5, 4], 18., 1200, 7, 8.);
    let mut held = notes(&c);
    held.push((1000., 22600., 3));
    let r = run(&chart(held, 7, 8.), 1.);
    assert!(score(&r, "ln") > 0. && score(&r, "jumptrill") > score(&r, "delay"));
    let all_holds = chart(notes(&c).into_iter().map(|(t, _, k)| (t, t + 15., k)).collect(), 7, 8.);
    assert_eq!(rolled(&all_holds), 0.);
}

#[test]
fn sustain_is_not_reset_by_one_anchor_and_recovers_on_real_breaks() {
    let c = rolls(&[0, 1, 2, 3], 20., 2400, 4, 8.);
    let interrupted = chart(notes(&c).into_iter().enumerate().map(|(i, (t, e, k))| {
        let pause = if i >= 1200 { 10000. } else { 0. };
        (t + pause, e + pause, k)
    }).collect(), 4, 8.);
    assert!(feature(&g(&c), "jumptrill_sustain") > feature(&g(&interrupted), "jumptrill_sustain"));
    assert!(feature(&g(&repeated(&[&[0, 1, 2, 3]], 80., 800)), "vibro_sustain") > feature(&g(&repeated(&[&[0, 1, 2, 3]], 80., 40)), "vibro_sustain"));
}

fn bursts(gap_between: f64) -> Chart {
    rice((0..60).flat_map(move |b| (0..5).flat_map(move |i| (0..4).map(move |k| (1000. + b as f64 * gap_between + i as f64 * 80., k)))), 4, 5.)
}

#[test]
fn vibro_support_does_not_jump_when_a_slower_gap_crosses_the_speed_gate() {
    let c = bursts(434.);
    let (a, b) = (analyse(&c, 1.0259, BIN, false), analyse(&c, 1.0261, BIN, false));
    for name in ["vibro_easy", "vibro_control", "vibro_sustain"] {
        assert!((feature(&a, name) - feature(&b, name)).abs() < 0.01);
    }
}

#[test]
fn unfitted_execution_features_cannot_silently_discount_a_chart() {
    let mut ex = Dict::new();
    for f in FEATURES {
        ex.set(f, 1.);
    }
    assert_eq!(final_rating(&Params::new(json!({}), &Value::Null).unwrap(), 4, 9., &[0.; 12], Some(&ex)), 9.);
}

#[test]
fn long_stable_repetition_is_distinct_from_short_fixed_chords_inside_cj() {
    let long = g(&repeated(&[&[0, 1, 2, 3]], 80., 500));
    let short = g(&rice((0..50).flat_map(|b| (0..5).flat_map(move |i| (0..4).map(move |k| (1000. + b as f64 * 2000. + i as f64 * 80., k)))), 4, 5.));
    let changing = g(&repeated(&CHANGING, 80., 400));
    assert!(feature(&long, "vibro_locked") > 0.5 && feature(&short, "vibro_locked") < 0.01);
    assert!(feature(&changing, "vibro_locked") == 0. && feature(&changing, "vibro_control") > 0.2);
}

#[test]
fn gesture_calibration_is_local_to_evidence_and_inverts_the_score_link() {
    let prm = Params::new(json!({"gestures": {"features": ["vibro_easy", "vibro_control"], "modes": {"4": {"coeff": [-0.4, 0.3], "slope": 2., "curve": 0.6}}}}), &Value::Null).unwrap();
    let ex = |k: &str, v: f64| {
        let mut d = Dict::new();
        d.set(k, v);
        d
    };
    assert_eq!(gesture_rating(&prm, 4, 9., 9., Some(&Dict::new())), 9.);
    assert_eq!(gesture_rating(&prm, 7, 9., 9., Some(&ex("vibro_easy", 1.))), 9.);
    let (lower, higher) = (gesture_rating(&prm, 4, 9., 9., Some(&ex("vibro_easy", 0.5))), gesture_rating(&prm, 4, 9., 9., Some(&ex("vibro_control", 0.5))));
    let link = |v: f64| 2. * v.ln() + 0.6 * v.ln().powi(2);
    assert!((link(lower) - link(9.) + 0.2).abs() < 1e-9 && (link(higher) - link(9.) - 0.15).abs() < 1e-9);
    assert!(0. < lower && lower < 9. && 9. < higher);
    assert_eq!(gesture_rating(&prm, 4, 0., 0., Some(&ex("vibro_easy", 1.))), 0.);
}

#[test]
fn degenerate_input_is_explicit() {
    assert!(g(&chart(Vec::new(), 4, 5.)).features.iter().all(|f| *f == 0.));
    assert_eq!(feature(&g(&rice([(1000., 0), (1000., 1)], 4, 5.)), "vibro_easy"), 0.);
}

// ---- rolled execution ----

fn configured(c: &Chart, strength: f64, rate: f64) -> Dict {
    let mut model = standard_params().model.clone();
    model["rolled_execution"] = json!({"modes": {c.keys.to_string(): {"strength": strength}}});
    compute(c, rate, true, &Params::new(model, &Value::Null).unwrap()).unwrap()
}

#[test]
fn planning_does_not_change_existing_calibration_features() {
    for keys in 4..=10usize {
        let cols: Vec<u8> = (0..keys / 2).chain((keys + 1) / 2..keys).map(|c| c as u8).collect();
        let c = rolls(&cols, 12., 1200, keys, 8.);
        let (old, new) = (g(&c), analyse(&c, 1., BIN, true));
        assert!(new.series == old.series && new.features == old.features);
        let plans = new.plans.unwrap();
        assert!(plans.iter().any(|(_, p)| !p.is_empty()));
        let window = great_window(c.od);
        for p in plans.iter().flat_map(|(_, p)| p) {
            let mut columns: Vec<u8> = p.ids.iter().map(|&i| c.col[i]).collect();
            columns.sort();
            columns.dedup();
            assert_eq!(p.ids.len(), columns.len());
            for (&i, offset) in p.ids.iter().zip(&p.offsets) {
                assert!((p.time + offset - (c.t[i] - c.t[0]) * 0.001).abs() <= window + 1e-10);
            }
        }
    }
}

#[test]
fn rolls_share_cost_but_fixed_jt_split_trill_and_slow_stream_are_unchanged() {
    for c in [repeated(&[&[0, 1], &[2, 3]], 60., 400), repeated(&[&[0, 2], &[1, 3]], 60., 400), roll4(&[0, 1, 2, 3], 70.), repeated(&[&[0, 1, 2, 3]], 60., 400)] {
        assert_eq!(configured(&c, 1., 1.), configured(&c, 0., 1.));
    }
    let c = rolls(&[0, 1, 2, 3], 30., 2400, 4, 8.);
    let (before, after) = (configured(&c, 0., 1.), configured(&c, 1., 1.));
    assert!(score(&after, "overall") < score(&before, "overall") && score(&after, "jumptrill") > score(&after, "stream"));
    assert!(execution(&after, "rolled_factor") < 1. && after.f("notes") == before.f("notes"));
}

#[test]
fn interleaved_rolling_offsets_are_not_simultaneous_hit_errors() {
    let c = rolls(&[0, 2, 1, 3], 27., 2400, 4, 8.);
    let plans = analyse(&c, 1., BIN, true).plans.unwrap().into_iter().find(|(left, _)| !left).unwrap().1;
    assert!(plans.len() > 100);
    // Simultaneous feasibility has narrow margins in these regular hand cycles; rolling keeps the per-finger offsets.
    assert!(plans.iter().map(|p| p.strength).sum::<f64>() / plans.len() as f64 > 0.9);
    assert!(plans.iter().any(|p| p.span > 0.05));
    assert!(score(&configured(&c, 1., 1.), "overall") < score(&configured(&c, 0., 1.), "overall"));
}

#[test]
fn control_and_actual_repeats_limit_sharing() {
    let support = |c: &Chart| configured(c, 1., 1.).dict("execution").unwrap().fget("rolled_support", 0.);
    let (clean, turns) = (rolls(&[0, 1, 2, 3], 27., 2400, 4, 8.), rolls(&[0, 1, 2, 3, 1, 0, 3, 2], 27., 2400, 4, 8.));
    assert!(support(&turns) < support(&clean));
    assert!(support(&roll4(&[0, 1, 0, 2, 0, 3], 27.)) < 0.1);
    let held = chart(notes(&clean).into_iter().map(|(t, _, k)| (t, t + 15., k)).collect(), 4, 8.);
    assert_eq!(configured(&held, 1., 1.), configured(&held, 0., 1.));
}

#[test]
fn no_metadata_effect_and_support_is_rate_specific() {
    let c = rolls(&[0, 1, 3, 2], 35., 1600, 4, 8.);
    let mut renamed = c.clone();
    renamed.title = "Unsubmitted completely different song".into();
    renamed.creator = "Unknown".into();
    assert_eq!(configured(&c, 1., 1.), configured(&renamed, 1., 1.));
    assert_eq!(configured(&c, 1., 0.5), configured(&c, 0., 0.5));
    assert!(analyse(&c, 0.5, BIN, true).plans.unwrap().iter().all(|(_, p)| p.is_empty()));
}

#[test]
fn locality_leaves_unrelated_phrase_timeline_untouched() {
    let (plain, quick) = (rolls(&[0, 1, 2, 3], 90., 500, 4, 8.), rolls(&[0, 1, 2, 3], 27., 1400, 4, 8.));
    let mut both = notes(&plain);
    both.extend(notes(&quick).into_iter().map(|(t, e, k)| (t + 60000., e + 60000., k)));
    let c = chart(both, 4, 8.);
    let timeline = |d: &Dict| d.get("timeline").map(Val::list_f).unwrap();
    let (before, after) = (timeline(&configured(&c, 0., 1.)), timeline(&configured(&c, 1., 1.)));
    assert!(before[..75].iter().zip(&after[..75]).all(|(a, b)| (a - b).abs() < 1e-9));
    assert!(after[125..].iter().sum::<f64>() < before[125..].iter().sum::<f64>());
}

#[test]
fn sv_and_cached_feature_prediction_use_same_final_difficulty() {
    let mut c = rolls(&[0, 1, 3, 2], 27., 1600, 4, 8.);
    c.sv = vec![(0., 1.), (4000., 0.4), (5000., 1.8), (6000., 1.)];
    let mut model = standard_params().model.clone();
    model["rolled_execution"] = json!({"modes": {"4": {"strength": 1.}}});
    let prm = Params::new(model, &Value::Null).unwrap();
    let (full, constant) = (compute(&c, 1., true, &prm).unwrap(), compute(&c, 1., false, &prm).unwrap());
    assert!((full.f("no_sv_overall") - score(&constant, "overall")).abs() < 1e-9);
    assert!(score(&full, "overall") >= score(&constant, "overall") - 1e-9);
    let features = full.get("calibration_features").unwrap().list_f();
    let rating = final_rating(&prm, 4, full.f("raw_overall"), &features, full.dict("execution"));
    assert!((rating / score(&full, "overall") - 1.).abs() < 1e-12);
}

#[test]
fn credit_fades_before_the_packet_timing_boundary() {
    let c = rolls(&[0, 2, 1, 3], 28., 1400, 4, 10.);
    let values: Vec<f64> = [0.8234, 0.8236, 0.8238].iter().map(|r| score(&configured(&c, 1., *r), "overall")).collect();
    // Same-rate baseline drift is tiny here; no sudden grouped/un-grouped drop.
    assert!(values.iter().cloned().fold(f64::MIN, f64::max) - values.iter().cloned().fold(f64::MAX, f64::min) < 0.02);
}

#[test]
fn invalid_strength_is_explicit() {
    let c = roll4(&[0, 1, 2, 3], 27.);
    for strength in [-0.1, 1.1] {
        let mut model = standard_params().model.clone();
        model["rolled_execution"] = json!({"modes": {"4": {"strength": strength}}});
        assert!(compute(&c, 1., true, &Params::new(model, &Value::Null).unwrap()).is_err());
    }
}

#[test]
fn feasible_rolling_offsets_can_plan_without_any_simultaneous_chord() {
    let c = rolls(&[0, 3, 1, 2], 30., 40, 4, 10.);
    assert_eq!(rolled(&c), 0.);
    let plans = analyse(&c, 1., BIN, true).plans.unwrap().into_iter().find(|p| !p.0).unwrap().1;
    assert_eq!(plans.len(), 20);
    assert!((plans[0].control - 0.15882352941176467).abs() < 1e-15);
}
