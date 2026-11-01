//! Synthetic charts for the calculator tests.
#![allow(dead_code)]
use ms_calc::chart::Chart;
use ms_calc::features::standard_params;
use ms_calc::params::Params;
use ms_calc::val::Dict;

/// (start ms, end ms, column).
pub fn chart(mut notes: Vec<(f64, f64, u8)>, keys: usize, od: f64) -> Chart {
    notes.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.2.cmp(&b.2)).then(a.1.total_cmp(&b.1)));
    Chart { keys, t: notes.iter().map(|n| n.0).collect(), end: notes.iter().map(|n| n.1).collect(), col: notes.iter().map(|n| n.2).collect(), sv: Vec::new(), od,
            title: "synthetic control".into(), artist: "synthetic control".into(), version: "synthetic control".into(), creator: "synthetic control".into() }
}

/// Rice: start = end.
pub fn rice(notes: impl IntoIterator<Item = (f64, u8)>, keys: usize, od: f64) -> Chart {
    chart(notes.into_iter().map(|(t, c)| (t, t, c)).collect(), keys, od)
}

pub fn run(c: &Chart, rate: f64) -> Dict {
    ms_calc::compute::compute(c, rate, true, &standard_params()).unwrap()
}

pub fn run_with(c: &Chart, rate: f64, prm: &Params) -> Dict {
    ms_calc::compute::compute(c, rate, true, prm).unwrap()
}

pub fn execution(r: &Dict, name: &str) -> f64 {
    r.dict("execution").unwrap().f(name)
}

pub fn score(r: &Dict, name: &str) -> f64 {
    r.dict("scores").unwrap().f(name)
}
