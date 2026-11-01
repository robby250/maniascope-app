//! Display units: the reference conversion, cached features and the gameplay correction.
mod common;
use common::*;
use ms_calc::chart::Chart;
use ms_calc::compute::compute;
use ms_calc::features::{convert_units, displayed_feature, displayed_value, feature_display_factor, features_of, standard_params};
use ms_calc::model::{display_rating, unit_rating, units_apply};
use ms_calc::params::Params;
use ms_calc::val::{Dict, Key, Val};
use serde_json::{json, Value};

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * a.abs().max(b.abs())
}

fn with(model: Value) -> Params {
    Params::new(model, &Value::Null).unwrap()
}

#[test]
fn reference_conversion_is_monotone_and_shared_by_all_displayed_demands() {
    let mut model = json!({"input": [2., 0.7], "window": -0.8, "reference": [0., 2., 0.7], "reference_od": 8., "knee": 4.});
    for value in [0.01, 0.3, 1., 3.99, 4., 4.01, 8., 15., 50.] {
        assert!(close(unit_rating(value, 8., &model), value));
    }
    assert!(unit_rating(8., 0., &model) < unit_rating(8., 8., &model) && unit_rating(8., 8., &model) < unit_rating(8., 10., &model));
    let values: Vec<f64> = (0..1000).map(|i| unit_rating(0.1 + i as f64 * 0.03, 9., &model)).collect();
    assert!(values.windows(2).all(|w| w[0] < w[1]));
    model["reference"] = json!([0.4, 2., 1.2]);
    let prm = with(json!({"score_units": {"modes": {"7": model}}}));
    let source = |keys: i64| {
        let mut sc = Dict::new();
        for (k, v) in [("overall", 10.), ("stream", 8.), ("ln", 0.)] {
            sc.set(k, v);
        }
        let mut levels = Dict::new();
        levels.setf(8., 8.);
        let mut archetype = Dict::new();
        archetype.set("rating", 8.);
        let mut d = Dict::new();
        d.set("keys", keys);
        d.set("od", 5.);
        d.set("horizon", 8.);
        d.set("scores", sc);
        d.set("levels", levels);
        d.set("timeline", Val::floats(&[0., 8., 10.]));
        d.set("archetypes", Val::List(vec![Val::Dict(archetype)]));
        d.set("baseline_overall", 9.);
        d.set("no_sv_overall", 8.);
        d
    };
    let mut result = source(7);
    units_apply(&prm, &mut result);
    let expected = display_rating(8., 5., &model);
    let sc = result.dict("scores").unwrap();
    let timeline = result.get("timeline").unwrap().list_f();
    let level = result.dict("levels").unwrap().0.iter().find(|(k, _)| *k == Key::F(8.)).unwrap().1.f();
    assert!(sc.f("stream") == expected && level == expected && timeline[1] == expected && result.f("no_sv_overall") == expected);
    let Some(Val::List(archetypes)) = result.get("archetypes") else { panic!() };
    let Val::Dict(first) = &archetypes[0] else { panic!() };
    assert_eq!(first.f("rating"), expected);
    assert!(sc.f("ln") == 0. && timeline[0] == 0. && result.f("preunit_overall") == 10.);
    let mut other = source(6);
    units_apply(&prm, &mut other);
    assert_eq!(other.dict("scores"), source(6).dict("scores"));
}

#[test]
fn midpoint_display_keeps_fitted_features_in_their_original_coordinates() {
    let prm = standard_params();
    let model = prm.model["score_units"]["modes"]["7"].clone();
    for od in [0., 6., 8., 10.] {
        for value in [0.1, 4., 7., 10., 18., 28.] {
            let mut sk = Dict::new();
            sk.set("ln", 0.8);
            let mut f = Dict::new();
            f.set("keys", 7i64);
            f.set("overall", value);
            f.set("od", od);
            f.set("sk", sk);
            f.set("stam", 0.9);
            f.set("baseline_overall", value * 0.95);
            convert_units(&mut f, &prm);
            let reference = unit_rating(value, od, &model);
            assert_eq!(f.f("overall"), reference);
            assert_eq!(f.dict("sk").unwrap().f("ln"), ms_calc::py::round_digits(unit_rating(0.8 * value, od, &model) / reference, 4));
            let expected = (value + reference) / 2.;
            assert_eq!(display_rating(value, od, &model), expected);
            assert_eq!(displayed_feature(&f, &prm), expected);
            assert!((displayed_value(reference, 7, od, 1., &prm) - expected).abs() < 1e-9 * expected.max(1.));
        }
    }
    assert_eq!(displayed_value(12., 4, 8., 1., &prm), 12.);
}

fn sv_chart(gap: f64, hold_every: usize, hold: f64, od: f64) -> Chart {
    let mut c = chart((0..400).map(|i| {
        let t = 1000. + gap * i as f64;
        (t, t + if i % hold_every == 0 { hold } else { 0. }, (i % 7) as u8)
    }).collect(), 7, od);
    c.sv = vec![(0., 1.), (5000., 0.4), (8000., 1.6), (12000., 1.)];
    c
}

#[test]
fn fresh_and_cached_reference_units_preserve_labels_and_other_modes() {
    let mut model = standard_params().model.clone();
    model["score_units"] = json!({"modes": {"7": {"input": [1.6, 0.9], "window": -0.85, "reference": [0.48, 1.28, 1.68], "reference_od": 8., "knee": 4.}}});
    let enabled = with(model.clone());
    model.as_object_mut().unwrap().remove("score_units");
    let disabled = with(model);
    let base = sv_chart(95., 9, 140., 7.);
    for keys in [4usize, 7] {
        let mut c = base.clone();
        c.keys = keys;
        c.col.iter_mut().for_each(|k| *k %= keys as u8);
        for rate in [0.75, 1., 1.5] {
            let (old, fresh) = (run_with(&c, rate, &disabled), run_with(&c, rate, &enabled));
            let (cached, actual) = (features_of(&old, rate, &disabled), features_of(&fresh, rate, &enabled));
            assert!(close(fresh.f("no_sv_overall"), score(&compute(&c, rate, false, &enabled).unwrap(), "overall")));
            let mut expected = cached.clone();
            convert_units(&mut expected, &enabled);
            assert_eq!(actual, expected);
            assert_eq!(ms_calc::card::describe(&fresh, &enabled.k).0, ms_calc::card::describe(&old, &disabled.k).0);
            let names = |r: &Dict| ms_calc::card::card(r).iter().map(|e| if let Val::Dict(d) = e { format!("{:?}", d.get("name")) } else { String::new() }).collect::<Vec<_>>();
            assert_eq!(names(&fresh), names(&old));
            let mut again = expected.clone();
            convert_units(&mut again, &enabled);
            assert_eq!(again, expected, "converting twice changes nothing");
            if keys == 4 {
                assert_eq!(fresh, old);
            }
        }
    }
}

#[test]
fn gameplay_display_correction_preserves_forecast_inputs_and_labels() {
    let enabled = standard_params();
    let mut model = enabled.model.clone();
    model["score_units"]["modes"]["7"].as_object_mut().unwrap().remove("display_coeff");
    let disabled = with(model);
    let c = sv_chart(65., 5, 170., 8.);
    for rate in [0.75, 1., 1.5] {
        let (before, after) = (run_with(&c, rate, &disabled), run_with(&c, rate, &enabled));
        let f = features_of(&after, rate, &enabled);
        assert_eq!(features_of(&before, rate, &disabled), f, "every predictor input, including labels and SV context");
        assert_eq!(after.get("preunit_result"), before.get("preunit_result"));
        assert_ne!(score(&after, "overall"), score(&before, "overall"));
        assert!(close(displayed_feature(&f, &enabled), score(&after, "overall")));
        assert!(close(after.f("no_sv_overall"), score(&compute(&c, rate, false, &enabled).unwrap(), "overall")));
        let factor = feature_display_factor(&f, &enabled);
        assert!((displayed_value(f.f("overall"), 7, c.od, factor, &enabled) - score(&after, "overall")).abs() < 1e-9);
        assert!(after.get("timeline").unwrap().list_f().iter().all(|v| *v >= 0. && v.is_finite()));
    }
}

#[test]
fn unconverted_cache_still_displays() {
    let prm = standard_params();
    let mut ex = Dict::new();
    ex.set("residual_vector", Val::floats(&[0.2; ms_calc::model::SR_NAMES_LEN]));
    let mut f = Dict::new();
    f.set("keys", 7i64);
    f.set("overall", 5.);
    f.set("od", 8.);
    f.set("sk", Dict::new());
    f.set("execution", ex);
    assert!(displayed_feature(&f, &prm) > 0., "no preunit_overall: Stats failed on G835LX 2026-10-01");
}

#[test]
fn density_is_rate_aware_and_vibro_needs_one_repeated_hand() {
    use ms_calc::structure::structure;
    let k = &standard_params().k;
    let s = |c: &Chart| structure(c, 1., k);
    let plain = chart(vec![(0., 0., 0), (0., 500., 1), (1000., 3000., 2)], 7, 8.);
    let st = s(&plain);
    assert_eq!((st.f("head_span"), st.f("play_span"), st.f("nps")), (1., 3., 3.));
    let vibro = rice((0..200).map(|i| (40. * i as f64, 2)), 7, 8.);
    assert!(s(&vibro).get("vibro_focused").is_some_and(Val::truthy));
    let masks: [&[u8]; 4] = [&[0, 1, 2], &[3, 4, 5], &[1, 2, 6], &[0, 3, 5]];
    let notes = |gap: f64| (0..400).flat_map(|i| masks[i % 4].iter().map(move |&c| (gap * i as f64, c))).collect::<Vec<_>>();
    let (fast, slow) = (rice(notes(70.), 7, 8.), rice(notes(70. / 0.9).iter().map(|&(t, c)| (t.round(), c)), 7, 8.));
    assert!(!s(&fast).get("vibro_focused").is_some_and(Val::truthy));
    assert_eq!(s(&fast).get("family"), s(&slow).get("family"), "a rate of a chart is the same family");
}

#[test]
fn short_repeated_drills_are_marked_without_banning_specialists() {
    use ms_calc::structure::structure;
    let k = &standard_params().k;
    let chords: [&[u8]; 2] = [&[3, 4, 5, 6], &[0, 1, 2]];
    let c = rice((0..1000).flat_map(|i: usize| chords[i % 2].iter().map(move |&col| (80. * i as f64, col))), 7, 8.);
    assert!(structure(&c, 1., k).f("drill") > 0.7);
}

#[test]
fn endurance_recovers_and_depends_on_the_rest() {
    use ms_calc::model::endurance;
    let run = |values: Vec<f64>| {
        let fine: Vec<f64> = values.iter().flat_map(|v| std::iter::repeat_n(v / 8., 8)).collect();
        let n = fine.len();
        [fine.clone(), fine, vec![0.; n]]
    };
    let ((continuous, _), _) = endurance(&[&run(vec![10.; 240])], 1., 8);
    let ((rested, _), timeline) = endurance(&[&run([vec![10.; 120], vec![0.; 60], vec![10.; 120]].concat())], 1., 8);
    assert!(continuous > rested);
    assert!(timeline[179] < timeline[119] / 5.);
}

#[test]
fn whole_map_technical_presence_tells_regular_from_irregular_hands() {
    use ms_calc::pattern::technical_title;
    let regular = rice((0..400).map(|i| (80. * i as f64, (i % 4) as u8)), 4, 8.);
    let mut t = 0.;
    let irregular = rice((0..400).map(|i: usize| { t += [60., 93., 47., 111., 75.][i % 5]; (t, (i % 4) as u8) }), 4, 8.);
    let profile = |c: &Chart| -> [f64; 3] {
        let r = run(c, 1.);
        let p = r.dict("tech_profile").unwrap();
        [p.f("technical"), p.f("patterntech"), p.f("rhythmtech")]
    };
    let (a, b) = (profile(&regular), profile(&irregular));
    assert!(a[0] < 0.1, "{a:?}");
    assert!(b[2] > 0.5 && b[0] > 0.5, "{b:?}");
    assert_eq!(technical_title("Dump", 0.72, Some(&b)), "Tech Dump");
    assert_eq!(technical_title("Dump", 0.72, Some(&a)), "Dump");
}
