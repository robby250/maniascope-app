//! recdata.feats_batch: the stored feature dict of a chart at a rate, from the calculator's result.
use crate::background::background_map;
use crate::chart;
use crate::compute;
use crate::model::{self, display_factor, unit_rating};
use crate::params::Params;
use crate::py::round_digits;
use crate::val::{Dict, Key, Val};
use serde_json::Value;
use std::collections::VecDeque;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, OnceLock};

type Slot<T> = Arc<OnceLock<T>>;

/// The last few calculations by key. A caller that finds one running waits for it instead of repeating it.
struct Recent<T> {
    slots: Mutex<VecDeque<(u64, Slot<T>)>>,
    size: usize,
}

impl<T: Clone> Recent<T> {
    const fn new(size: usize) -> Self {
        Recent { slots: Mutex::new(VecDeque::new()), size }
    }

    fn get(&self, key: u64, make: impl FnOnce() -> T) -> T {
        let slot = {
            let mut slots = self.slots.lock().unwrap();
            match slots.iter().find(|s| s.0 == key) {
                Some(s) => s.1.clone(),
                None => {
                    let slot: Slot<T> = Arc::default();
                    slots.push_back((key, slot.clone()));
                    if slots.len() > self.size {
                        slots.pop_front();
                    }
                    slot
                }
            }
        };
        slot.get_or_init(make).clone()
    }
}

static SELECTED: Recent<Result<Arc<Dict>, String>> = Recent::new(4);

/// compute() for the selected chart. The card and the prediction ask for the same analysis at the same moment, from two
/// threads: this is one calculation of it. Batches stay on compute(): they would only churn these slots.
pub fn compute_selected(data: &[u8], parsed: &chart::Chart, rate: f64, sv: bool, prm: &Params) -> Result<Arc<Dict>, String> {
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    (data, rate.to_bits(), sv, prm.id()).hash(&mut hash);
    SELECTED.get(hash.finish(), || compute::compute(parsed, rate, sv, prm).map(Arc::new))
}

pub const SKILLS: [&str; 34] = ["stream", "jumpstream", "handstream", "jackspeed", "chordjack", "minijack", "chordstream", "delay",
    "bracket", "jack", "ln", "release", "inverse", "hybrid", "shield", "sv", "technical", "vibro", "stamina",
    "dump", "jumptrill", "splittrill", "longjack", "patterntech", "rhythmtech", "trill1h", "anchor", "mash",
    "quadstream", "sv_fast", "sv_slow", "sv_accel", "sv_stutter", "sv_brake"];

/// calib/difficulty.json: the frozen chart-only calibration.
pub fn standard_params() -> Params {
    let model: Value = serde_json::from_str(include_str!("../../../calib/difficulty.json")).expect("difficulty.json");
    Params::new(model, &Value::Null).expect("difficulty.json parameters")
}

/// The calibration a past calculator used: calib/carry.json holds what differs from today's, null for a key it lacked.
pub fn params_of(calc: &str) -> Result<Params, String> {
    fn merge(to: &mut Value, patch: &Value) {
        let (Some(to), Some(patch)) = (to.as_object_mut(), patch.as_object()) else { return };
        for (k, v) in patch {
            match (v.is_object(), to.get_mut(k)) {
                (_, _) if v.is_null() => { to.remove(k); }
                (true, Some(at)) => merge(at, v),
                _ => { to.insert(k.clone(), v.clone()); }
            }
        }
    }
    let patches: Value = serde_json::from_str(include_str!("../../../calib/carry.json")).map_err(|e| e.to_string())?;
    let mut model = standard_params().model;
    merge(&mut model, patches.get(calc).ok_or_else(|| format!("no calibration kept for {calc}"))?);
    Params::new(model, &Value::Null)
}

/// The parameters of a keymode's display conversion.
pub fn unit_model(prm: &Params, keys: i64) -> Option<&Value> {
    prm.model.get("score_units")?.get("modes")?.get(keys.to_string())
}

fn stam(levels: Option<&Dict>, overall: f64) -> f64 {
    if overall == 0. {
        return 1.;
    }
    let level = levels.and_then(|l| l.0.iter().find(|(k, _)| matches!(k, Key::F(x) if *x == 120.)).map(|x| x.1.f()).filter(|&x| x != 0.));
    level.unwrap_or(overall) / overall
}

/// score_units.feature: the exact cache in display units, the pre-unit values kept.
pub fn convert_units(f: &mut Dict, prm: &Params) {
    let keys = f.f("keys") as i64;
    let Some(model) = unit_model(prm, keys) else { return };
    let (old, od) = (f.f("overall"), f.f("od"));
    if old <= 0. || f.has("preunit_features") {
        return;
    }
    let mut pre = Dict::new();
    for k in ["overall", "sk", "stam", "baseline_overall"] {
        pre.set(k, f.get(k).cloned().unwrap_or(Val::None));
    }
    f.set("preunit_features", Val::Dict(pre));
    let convert = |v: f64| unit_rating(v, od, model);
    f.set("preunit_overall", old);
    let overall = convert(old);
    f.set("overall", overall);
    if let Some(Val::Dict(sk)) = f.get_mut("sk") {
        for (_, v) in sk.0.iter_mut() {
            *v = Val::F(round_digits(convert(v.f() * old) / overall, 4));
        }
    }
    let (s, baseline) = (f.f("stam"), f.f("baseline_overall"));
    f.set("stam", convert(s * old) / overall);
    f.set("baseline_overall", convert(baseline));
}

/// score_units.restore_feature: calibration tools refit original coordinates before reapplying units.
pub fn restore_feature(f: &mut Dict) {
    if let Some(Val::Dict(pre)) = f.pop("preunit_features") {
        for (k, v) in pre.0 {
            if let Key::S(k) = k {
                f.set(&k, v);
            }
        }
        f.pop("preunit_overall");
    }
}

/// A chart's stored features moved from `before`'s calibration to `after`'s, without its file; None if it doesn't move.
/// Ratios keep what the stages leave out (rate uniformity), and `sk`/`stam` are shares of the overall.
pub fn recalibrated(f: &Dict, before: &Params, after: &Params) -> Option<Dict> {
    let mut f = f.clone();
    restore_feature(&mut f);
    let keys = f.f("keys") as usize;
    let raw = f.fget("raw_overall", f.f("overall"));
    let features = f.get("calibration_features").map_or_else(Vec::new, Val::list_f);
    let moved = |rating: &dyn Fn(&Params) -> f64| rating(after) / rating(before);
    let overall = moved(&|p| model::final_rating(p, keys, raw, &features, f.dict("execution")));
    if !overall.is_finite() || overall == 1. {
        return None;
    }
    let baseline = moved(&|p| model::baseline_rating(p, keys, raw, &features, f.dict("execution")));
    f.set("overall", f.f("overall") * overall);
    f.set("baseline_overall", f.f("baseline_overall") * baseline);
    convert_units(&mut f, after);
    Some(f)
}

/// recdata._feats_at for a calculated result.
pub fn features_of(res: &Dict, rate: f64, prm: &Params) -> Dict {
    let demand = res.dict("preunit_result").unwrap_or(res);
    let scores = demand.dict("scores").cloned().unwrap_or_default();
    let horizon = res.get("horizon").is_some_and(Val::truthy);
    let o = scores.get("overall").filter(|v| v.truthy()).cloned().unwrap_or(Val::Int(0));
    let overall = o.f();
    let notes = res.get("notes").cloned().unwrap_or(Val::Int(0));
    let mut sk = Dict::new();
    if overall != 0. {
        for k in SKILLS {
            if let Some(v) = scores.get(k).filter(|v| v.truthy()) {
                sk.set(k, round_digits(v.f() / overall, 4));
            }
        }
    }
    let mut f = Dict::new();
    f.set("keys", res.get("keys").cloned().unwrap_or(Val::Int(0)));
    f.set("overall", o);
    f.set("od", res.get("od").cloned().unwrap_or(Val::F(0.)));
    f.set("ln", res.f("ln_notes") / notes.f().max(1.));
    f.set("notes", notes);
    f.set("length", res.get("length").cloned().unwrap_or(Val::Int(0)));
    let mut stamina = stam(if horizon { demand.dict("levels") } else { None }, overall);
    // The fitted predictor's sustain coordinate predates the horizon display-unit repair: the schema stays until a
    // validated refit.
    if horizon && !res.get("rolled_correction").is_some_and(Val::truthy) {
        stamina /= rate.powf(prm.k.RATE_G);
    }
    f.set("stam", stamina);
    f.set("sk", Val::Dict(sk));
    f.set("description", if horizon { crate::card::describe(res, &prm.k).0 } else { "Sparse".to_string() });
    let or_empty = |k: &str| res.get(k).cloned().unwrap_or_else(|| Val::Dict(Dict::new()));
    f.set("endurance", or_empty("endurance"));
    f.set("raw_overall", res.get("raw_overall").cloned().unwrap_or_else(|| Val::F(overall)));
    f.set("calibration_features", res.get("calibration_features").cloned().unwrap_or(Val::List(Vec::new())));
    f.set("execution", or_empty("execution"));
    if let Some(s) = res.get("structure").filter(|v| v.truthy()) {
        f.set("nps", s.clone());
    }
    f.set("baseline_overall", demand.get("baseline_overall").cloned().unwrap_or_else(|| Val::F(overall)));
    f.set("tech_profile", or_empty("tech_profile"));
    f.set("archetype_profile", or_empty("archetype_profile"));
    convert_units(&mut f, prm);
    f
}

/// (lazer stars, judgement count) at a ranked rate; None for a chart rosu-pp cannot read.
fn lazer_stars(map: &rosu_pp::Beatmap, rate: f64) -> Option<(f64, u32)> {
    let mods = if rate == 0.75 { 256 } else if rate == 1.5 { 64 } else if rate == 1. { 0 } else { return None };
    match rosu_pp::Difficulty::new().mods(mods).lazer(true).calculate(map) {
        rosu_pp::any::DifficultyAttributes::Mania(a) => Some((a.stars, a.n_objects + a.n_hold_notes)),
        _ => None,
    }
}

/// {rate: features} of one chart's bytes.
pub fn chart_features(data: &[u8], rates: &[f64], prm: &Params) -> Result<Vec<(f64, Dict)>, String> {
    features_with(data, rates, prm, &|parsed, rate| compute::compute(parsed, rate, true, prm).map(Arc::new))
}

/// chart_features of the selected chart, on the card's calculation.
pub fn selected_features(data: &[u8], rate: f64, prm: &Params) -> Result<Vec<(f64, Dict)>, String> {
    features_with(data, &[rate], prm, &|parsed, rate| compute_selected(data, parsed, rate, true, prm))
}

fn features_with(data: &[u8], rates: &[f64], prm: &Params, calculate: &dyn Fn(&chart::Chart, f64) -> Result<Arc<Dict>, String>) -> Result<Vec<(f64, Dict)>, String> {
    let parsed = chart::parse(data).map_err(|e| match e {
        chart::ChartError::Chart(m) | chart::ChartError::Value(m) => m,
    })?;
    let mut out = Vec::with_capacity(rates.len());
    for &rate in rates {
        let res = calculate(&parsed, rate)?;
        out.push((rate, features_of(&res, rate, prm)));
    }
    if out.iter().any(|x| matches!(x.0, r if r == 0.75 || r == 1. || r == 1.5)) {
        if let Ok(map) = rosu_pp::Beatmap::from_bytes(data) {
            for (rate, f) in &mut out {
                if let Some((stars, hits)) = lazer_stars(&map, *rate) {
                    f.set("stars", stars);
                    f.set("hits", Val::Int(i64::from(hits)));
                }
            }
        }
    }
    Ok(out)
}

/// chart_features for many charts on `threads` background threads (0 = default); a broken chart is its own error.
pub fn features_batch(jobs: &[(Vec<u8>, Vec<f64>)], threads: usize, prm: &Params) -> Vec<Result<Vec<(f64, Dict)>, String>> {
    background_map(jobs, threads, |(data, rates)| chart_features(data, rates, prm))
}

/// score_units.feature_display_factor
pub fn feature_display_factor(f: &Dict, prm: &Params) -> f64 {
    let raw = f.dict("preunit_features").unwrap_or(f);
    let overall = raw.f("overall");
    display_factor(prm, f.f("keys") as usize, raw.fget("baseline_overall", overall), f.dict("execution"), overall)
}

fn display_rating(value: f64, od: f64, model: &Value) -> f64 {
    (value + unit_rating(value, od, model)) / 2.
}

/// score_units.displayed_feature: the rating a chart shows.
pub fn displayed_feature(f: &Dict, prm: &Params) -> f64 {
    let (factor, overall) = (feature_display_factor(f, prm), f.f("overall"));
    let pre = f.fget("preunit_overall", overall);
    if factor == 1. {
        return (overall + pre) / 2.;
    }
    // Unconverted caches (63 of G835LX's 7K charts) hold the pre-unit rating as 'overall'.
    unit_model(prm, f.f("keys") as i64).map_or((overall + pre) / 2., |model| display_rating(pre * factor, f.f("od"), model))
}

/// score_units.displayed_value: a fitted reference-coordinate estimate in display units, without refitting its model.
pub fn displayed_value(value: f64, keys: i64, od: f64, factor: f64, prm: &Params) -> f64 {
    let Some(model) = unit_model(prm, keys) else { return value };
    if value <= 0. {
        return value;
    }
    let knee = model.get("knee").and_then(Value::as_f64).unwrap_or(4.).ln();
    let x = value.ln();
    let at = |k: &str, i: usize| model[k][i].as_f64().unwrap_or(0.);
    let mut target = at("reference", 0) + at("reference", 1) * x + at("reference", 2) * (x - knee).max(0.).powi(2);
    let reference_od = model.get("reference_od").and_then(Value::as_f64).unwrap_or(8.);
    target -= model["window"].as_f64().unwrap_or(0.) * ((64. - 3. * od.clamp(0., 10.)) / (64. - 3. * reference_od)).ln();
    let (a, b) = (at("input", 0), at("input", 1));
    let delta = target - a * knee;
    let original = (if delta <= 0. || b == 0. { target / a } else { knee + 2. * delta / (a + (a * a + 4. * b * delta).sqrt()) }).exp();
    if factor == 1. { (value + original) / 2. } else { display_rating(original * factor, od, model) }
}

#[cfg(test)]
mod shared_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn callers_at_the_same_moment_share_one_calculation() {
        static RECENT: Recent<u64> = Recent::new(2);
        static RUNS: AtomicUsize = AtomicUsize::new(0);
        let threads: Vec<_> = (0..8).map(|_| std::thread::spawn(|| RECENT.get(1, || {
            RUNS.fetch_add(1, Ordering::SeqCst);
            std::thread::sleep(std::time::Duration::from_millis(50));
            7
        }))).collect();
        assert!(threads.into_iter().all(|t| t.join().unwrap() == 7));
        assert_eq!(RUNS.load(Ordering::SeqCst), 1);
        // Another key leaves it, the slots stay bounded.
        [2, 3].iter().for_each(|&k| { RECENT.get(k, || k); });
        assert_eq!(RECENT.slots.lock().unwrap().len(), 2);
        assert_eq!(RECENT.get(1, || 9), 9);
    }

    #[test]
    fn the_selected_chart_keeps_the_batch_numbers() {
        let mut text = "osu file format v14\n[General]\nMode:3\n[Difficulty]\nCircleSize:4\nOverallDifficulty:8\n[HitObjects]\n".to_string();
        (0..400).for_each(|i| text += &format!("{},192,{},1,0,0:0:0:0:\n", 64 + 128 * (i % 4), 1000 + i * 90));
        let prm = standard_params();
        let c = chart::parse(text.as_bytes()).unwrap();
        assert_eq!(*compute_selected(text.as_bytes(), &c, 1.1, true, &prm).unwrap(), compute::compute(&c, 1.1, true, &prm).unwrap());
        assert_eq!(selected_features(text.as_bytes(), 1.1, &prm).unwrap(), chart_features(text.as_bytes(), &[1.1], &prm).unwrap());
        // A different calculator is a different key.
        let mut model = prm.model.clone();
        model["rolled_execution"] = serde_json::json!({"modes": {"4": {"strength": 0.5}}});
        assert_ne!(Params::new(model, &Value::Null).unwrap().id(), prm.id());
        assert_eq!(standard_params().id(), prm.id());
    }
}
