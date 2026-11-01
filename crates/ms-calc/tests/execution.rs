//! Compact Mash, honest Tech/SV attribution, archetype qualification.
mod common;
use common::*;
use ms_calc::card::{card, profile};
use ms_calc::features::standard_params;
use ms_calc::model::final_rating;
use ms_calc::params::Params;
use ms_calc::pattern::technical_title;
use ms_calc::scroll::reading;
use ms_calc::skills::skills;
use ms_calc::val::{Dict, Val};
use serde_json::{json, Value};

#[test]
fn compact_pseudo_chords_are_not_continuous_delay_or_jumptrills() {
    let pseudo = rice((0..160).flat_map(|i| (0..7).map(move |j| (1000. + 180. * i as f64 + 8. * j as f64, j as u8))), 7, 5.);
    let precise = rice((0..1000).map(|i| (1000. + 35. * i as f64, (i % 7) as u8)), 7, 5.);
    let alternating = rice((0..400).flat_map(|i| {
        let cols: &[u8] = if i % 2 == 1 { &[0, 1, 2] } else { &[3, 4, 5, 6] };
        cols.iter().map(move |&c| (1000. + 85. * i as f64, c))
    }), 7, 5.);
    let p = run(&pseudo, 1.);
    assert!(execution(&p, "pseudo_mash") > 0.5);
    assert!(score(&p, "mash") > score(&p, "technical"));
    assert!(execution(&run(&precise, 1.), "mash") < 0.02);
    assert!(execution(&run(&alternating, 1.), "mash") < 0.02);
}

#[test]
fn changing_columns_and_independent_holds_still_require_control() {
    let changing = rice((0..300).flat_map(|i| {
        let cols: &[u8] = if i % 2 == 1 { &[0, 2, 4, 6] } else { &[0, 3, 4, 5] };
        cols.iter().map(move |&c| (1000. + 160. * i as f64, c))
    }), 7, 5.);
    let fixed = chart((0..200).flat_map(|i| (0..7).map(move |c| (1000. + 160. * i as f64, 1080. + 160. * i as f64, c as u8))).collect(), 7, 5.);
    let (a, b) = (run(&changing, 1.), run(&fixed, 1.));
    assert!(execution(&a, "mash") < 0.03);
    assert!(score(&a, "chordjack") > score(&a, "mash"));
    assert!(execution(&b, "mash") > 0.7);
    // Tail synchronization is evidence separate from head width, not a rule erasing independent LN mechanics.
    let unsynced = chart((0..50).flat_map(|i| (0..7).map(move |c| (1000. + 400. * i as f64, 1070. + 400. * i as f64 + 45. * c as f64, c as u8))).collect(), 7, 5.);
    assert!(execution(&run(&unsynced, 1.), "mash") < execution(&b, "mash"));
}

#[test]
fn execution_effect_is_fitted_not_an_unconditional_mash_discount() {
    let f = [0.; 12];
    let ex = |name: &str| {
        let mut d = Dict::new();
        d.set(name, 1.);
        d
    };
    let none = Params::new(json!({}), &Value::Null).unwrap();
    assert_eq!(final_rating(&none, 7, 8., &f, Some(&ex("pseudo_mash"))), 8.);
    let fitted = Params::new(json!({"execution": {"modes": {"7": {"slope": 2., "curve": 0., "coeff": [0., -0.4, 0., 0., 0.3, 0.]}}}}), &Value::Null).unwrap();
    assert!(final_rating(&fitted, 7, 8., &f, Some(&ex("pseudo_mash"))) < 8.);
    assert!(final_rating(&fitted, 7, 8., &f, Some(&ex("peak_pressure"))) > 8.);
    assert_eq!(final_rating(&fitted, 9, 8., &f, Some(&ex("pseudo_mash"))), 8., "an unvalidated mode stays unchanged");
}

#[test]
fn monotone_brake_is_not_stutter_and_total_sv_strain_is_retained() {
    let mut c = rice((0..80).map(|i| (1000. + i as f64 * 400., (i % 7) as u8)), 7, 5.);
    c.sv = (0..80).flat_map(|i| {
        let t = i as f64 * 400.;
        [(650. + t, 1.), (800. + t, 0.7), (920. + t, 0.4)]
    }).collect();
    let (strain, kinds, _) = reading(&c, 1., &standard_params()).unwrap();
    let total: f64 = strain.iter().sum();
    assert!(total > 0.);
    assert!(strain.iter().zip(&kinds).map(|(s, k)| s * k[3]).sum::<f64>() < 0.1 * total);
    assert!(strain.iter().zip(&kinds).all(|(s, k)| *s <= 0. || (k.iter().sum::<f64>() - 1.).abs() < 1e-8));
}

#[test]
fn zero_length_two_note_chart_does_not_crash_the_catalogue_builder() {
    let r = run(&rice([(1000., 0), (1000., 6)], 7, 5.), 1.);
    assert_eq!((score(&r, "overall"), score(&r, "stamina")), (0., 0.));
}

fn scores(values: &[(&str, f64)]) -> Dict {
    let mut d = Dict::new();
    for s in skills(7) {
        d.set(s.name(), 0.);
    }
    d.set("overall", 0.);
    for (k, v) in values {
        d.set(k, *v);
    }
    d
}

#[test]
fn rice_has_no_ln_chordstream_profile() {
    let c = rice((0..100).flat_map(|i| {
        let cols: &[u8] = if i % 2 == 1 { &[0, 1] } else { &[2, 3] };
        cols.iter().map(move |&c| (i as f64 * 80., c))
    }), 4, 5.);
    assert_eq!(profile(&c, 1.), 0.);
}

#[test]
fn ln_release_refinement_precedes_generic_tech_ln() {
    let mut result = Dict::new();
    result.set("keys", 7i64);
    result.set("scores", scores(&[("overall", 10.), ("ln", 10.), ("release", 9.3), ("delay", 9.5), ("technical", 8.), ("hybrid", 6.3), ("chordstream", 7.2)]));
    let mut archetype = Dict::new();
    archetype.set("ln_chordstream", 0.4);
    result.set("archetype_profile", archetype);
    let entry = |name: &str, rating: f64, parts: &[&str]| {
        let mut d = Dict::new();
        d.set("name", name);
        d.set("rating", rating);
        d.set("parts", Val::strs(parts));
        Val::Dict(d)
    };
    let mut tech = entry("Tech LN", 10., &["ln", "technical"]);
    if let Val::Dict(d) = &mut tech {
        d.set("of", "ln");
    }
    result.set("archetypes", Val::List(vec![tech, entry("LN Release", 9.3, &["release", "ln"])]));
    let Val::Dict(first) = &card(&result)[0] else { panic!() };
    assert_eq!(first.get("name").and_then(|n| if let Val::Str(s) = n { Some(s.as_str()) } else { None }), Some("LN Release"));
}

#[test]
fn local_holds_and_chords_qualify_but_separate_sections_do_not() {
    let chords = |start: f64, gap: f64, from: i32| (from..150).flat_map(move |i| [2u8, 3].map(|c| (start + i as f64 * gap, start + i as f64 * gap, c)));
    let mut local = vec![(0., 12000., 0u8)];
    local.extend(chords(0., 80., 1));
    assert!(profile(&chart(local, 7, 5.), 1.) > 0.9);
    let mut separate = vec![(0., 1000., 0u8)];
    separate.extend(chords(2000., 80., 0));
    assert_eq!(profile(&chart(separate, 7, 5.), 1.), 0.);
}

#[test]
fn ln_tech_requires_the_original_prevalence_and_demand_without_promotion() {
    assert_eq!(technical_title("LN", 0.75, Some(&[0.43, 0., 0.])), "LN");
    assert_eq!(technical_title("LN", 0.55, Some(&[0.8, 0., 0.])), "LN");
    assert_eq!(technical_title("LN", 0.75, Some(&[0.8, 0., 0.])), "Tech LN");
}
