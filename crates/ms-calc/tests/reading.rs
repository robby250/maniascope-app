//! Observable reading geometry and Quadstream recognition.
mod common;
use common::*;
use ms_calc::card::describe;
use ms_calc::chart::Chart;
use ms_calc::compute::compute;
use ms_calc::features::standard_params;
use ms_calc::params::Params;
use ms_calc::pattern::quadstream;
use ms_calc::scroll::{geometry, reading};
use serde_json::{json, Value};

const STUTTER: usize = 3;

fn roll(gap: f64, n: usize) -> Chart {
    rice((0..n).map(|i| (1000. + i as f64 * gap, (i % 4) as u8)), 4, 5.)
}

fn title(r: &ms_calc::val::Dict) -> String {
    describe(r, &standard_params().k).0
}

#[test]
fn quadstream_recognition_has_no_new_physical_charge_and_no_wide_name() {
    let make = |pattern: &[&[u8]], keys: usize| rice((0..800).flat_map(|i| pattern[i % pattern.len()].iter().map(move |&c| (1000. + 80. * i as f64, c)).collect::<Vec<_>>()), keys, 5.);
    let four: &[u8] = &[0, 1, 2, 3];
    let c = make(&[&[0], four, &[2], four], 4);
    let a = run(&c, 1.);
    assert!(title(&a).contains("Quadstream"));
    let b = run_with(&c, 1., &Params::new(standard_params().model.clone(), &json!({"off": ["quadstream"]})).unwrap());
    assert!((score(&a, "overall") - score(&b, "overall")).abs() < 1e-9 && !title(&b).contains("Quadstream"));
    assert_eq!(score(&run(&make(&[&[0, 1], four, &[2, 3], four], 4), 1.), "quadstream"), 0.);
    let seven: &[u8] = &[0, 1, 2, 3, 4, 5, 6];
    let wide = make(&[&[0], seven, &[6], seven], 7);
    assert!(quadstream(&wide, 1., 0.5).1 > 0.8 && !run(&wide, 1.).dict("scores").unwrap().has("quadstream"));
    let mut notes: Vec<(f64, u8)> = (0..1600).map(|i| (1000. + i as f64 * 90., (i % 4) as u8)).collect();
    notes.extend([0, 2, 3].map(|c| (46090., c)));
    assert!(!title(&run(&rice(notes, 4, 5.), 1.)).contains("Quadstream"));
}

#[test]
fn constant_scroll_setting_and_uniform_multiplier_do_not_create_sv() {
    let mut c = roll(100., 100);
    c.sv = vec![(0., 2.)];
    let prm = standard_params();
    assert!(reading(&c, 1., &prm).unwrap().0.iter().all(|v| *v == 0.));
    assert_eq!(score(&run(&c, 1.), "overall"), score(&compute(&c, 1., false, &prm).unwrap(), "overall"));
    c.sv = vec![(0., 1.), (3000., 0.3), (3400., 1.), (6000., 2.), (6600., 1.)];
    let a = geometry(&c, 1.);
    c.sv.iter_mut().for_each(|s| s.1 *= 3.);
    let b = geometry(&c, 1.);
    assert!(a.summary.iter().zip(&b.summary).all(|(x, y)| (x - y).abs() < 1e-9));
}

#[test]
fn sv_rate_compensation_matches_retimed_chart_and_preserves_no_sv_mode() {
    let mut c = roll(80., 200);
    c.sv = vec![(0., 1.), (3000., 0.25), (3400., 2.5), (3500., 1.), (7000., 0.1), (7700., 1.)];
    let mut retimed = rice((0..c.len()).map(|i| (c.t[i] / 1.23, c.col[i])), 4, 5.);
    retimed.sv = c.sv.iter().map(|&(t, m)| (t / 1.23, m)).collect();
    assert!(geometry(&c, 1.23).summary.iter().zip(&geometry(&retimed, 1.).summary).all(|(x, y)| (x - y).abs() < 1e-9));
    let prm = standard_params();
    let plain = compute(&c, 1.23, false, &prm).unwrap();
    assert_eq!(score(&plain, "sv"), 0.);
    assert!(plain.dict("execution").unwrap().get("scroll_base").is_none_or(|v| !v.truthy()));
    let included = run(&c, 1.23);
    assert!(score(&included, "overall") >= score(&plain, "overall"));
    assert_eq!(included.f("no_sv_overall"), score(&plain, "overall"));
}

#[test]
fn fine_stutter_is_not_averaged_into_constant_speed_or_infinite_slow_cost() {
    let mut c = roll(80., 200);
    c.sv = vec![(0., 1.)];
    c.sv.extend((0..300).flat_map(|i| [(2000. + 10. * i as f64, 0.02), (2005. + 10. * i as f64, 1.98)]));
    c.sv.push((5000., 1.));
    assert!(geometry(&c, 1.).summary[STUTTER] > 0.);
    c.sv = vec![(0., 1.), (2000., 0.001), (3000., 1.)];
    assert!(geometry(&c, 1.).vectors.iter().map(|v| v[1]).fold(0., f64::max) <= 1.);
    let (strain, kinds, _) = reading(&c, 1., &standard_params()).unwrap();
    assert!(strain.iter().all(|v| v.is_finite()));
    assert!(strain.iter().zip(&kinds).all(|(v, k)| *v <= 0. || (k.iter().sum::<f64>() - 1.).abs() < 1e-9));
}

#[test]
fn invisible_velocity_flicker_costs_less_than_visible_stutter() {
    let flicker = |period: usize| {
        let mut c = roll(80., 160);
        c.sv = vec![(0., 1.)];
        c.sv.extend((1000..11000).step_by(period).flat_map(|base| [(base as f64, 0.02), (base as f64 + period as f64 / 2., 1.98)]));
        c.sv.push((11000., 1.));
        geometry(&c, 1.).summary[STUTTER]
    };
    assert!(flicker(4) < 0.3 * flicker(80));
}

#[test]
fn redundant_sv_control_points_do_not_change_reading() {
    let mut c = roll(80., 120);
    c.sv = vec![(0., 1.), (3000., 0.2), (4000., 2.), (4300., 1.)];
    let first = geometry(&c, 1.).summary;
    c.sv = vec![(0., 1.), (1500., 1.), (2000., 1.), (3000., 0.2), (3500., 0.2), (4000., 2.), (4100., 2.), (4300., 1.)];
    assert!(geometry(&c, 1.).summary.iter().zip(&first).all(|(x, y)| (x - y).abs() < 1e-12));
}

#[test]
fn straight_approaches_keep_preview_but_have_no_motion() {
    let mut c = roll(80., 200);
    c.sv = vec![(0., 1.), (6000., 1.3), (9000., 1.)];
    let g = geometry(&c, 1.);
    let flat: Vec<_> = c.t.iter().zip(&g.vectors).filter(|(t, _)| 7000. < **t && **t < 8500.).collect();
    assert!(!flat.is_empty() && flat.iter().all(|(_, v)| v[0] > 0. && v[2] == 0. && v[3] == 0.));
    assert!(c.t.iter().zip(&g.vectors).any(|(t, v)| 6000. < *t && *t < 6500. && v[2] > 0.));
}

#[test]
fn scroll_weights_select_only_calibrated_keymode() {
    let prm = Params::new(json!({"scroll_reading": {"weights": [1., 2., 3., 4.], "modes": {"4": {"weights": [2., 3., 4., 5.]}}}}), &Value::Null).unwrap();
    for (keys, weights) in [(4usize, [2., 3., 4., 5.]), (7, [1., 2., 3., 4.]), (10, [1., 2., 3., 4.])] {
        let mut c = rice((0..200).map(|i| (1000. + i as f64 * 80., (i % keys) as u8)), keys, 5.);
        c.sv = vec![(0., 1.), (3000., 0.25), (3400., 2.5), (3500., 1.)];
        let vectors = geometry(&c, 1.).vectors;
        assert!(vectors.iter().any(|v| v.iter().any(|x| *x != 0.)));
        for (s, v) in reading(&c, 1., &prm).unwrap().0.iter().zip(&vectors) {
            let want: f64 = v.iter().zip(&weights).map(|(a, b)| a * b).sum();
            assert!((s - want).abs() <= 1e-12 * want.abs().max(1e-300));
        }
    }
}
