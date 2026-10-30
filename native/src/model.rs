//! Chart-only fitted stages: difficulty_model.py, structural_residual.py, score_units.py.
use crate::py;
use crate::val::{Dict, Val};
use crate::params::Params;
use serde_json::Value;

pub const FEATURES: [&str; 12] = [
    "ln", "chordjack", "delay", "minijack", "technical", "sv", "endurance", "cj_endurance", "wide_jacks", "od_rice",
    "od_ln", "od_jack",
];
pub const EXTRA_FEATURES: [&str; 6] = ["mash", "pseudo_mash", "ln_flow", "smooth_switch", "peak_pressure", "transition_pressure"];

fn num(v: &Value, default: f64) -> f64 {
    v.as_f64().unwrap_or(default)
}
fn nums(v: &Value) -> Vec<f64> {
    v.as_array().map(|a| a.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect()).unwrap_or_default()
}
fn mode<'a>(m: &'a Value, stage: Option<&str>, keys: usize) -> &'a Value {
    let s = match stage {
        Some(st) => &m[st],
        None => m,
    };
    &s["modes"][keys.to_string()]
}
fn present(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Object(o) => !o.is_empty(),
        _ => true,
    }
}

/// Hand-local fatigue (difficulty_model.endurance): ({load, peak}, timeline).
pub fn endurance(fines: &[&[Vec<f64>; 3]], bin_seconds: f64, phases: usize) -> ((f64, f64), Vec<f64>) {
    let mut summaries: Vec<f64> = Vec::new();
    let mut tracks: Vec<Vec<f64>> = Vec::new();
    for fine in fines {
        for side in &fine[..2] {
            let values: Vec<f64> = side.chunks(phases).map(|c| py::sum(c.iter().copied())).collect();
            let n = 1i64.max(py::round(30.0 / bin_seconds) as i64) as usize;
            let mut rolling = 0.0f64;
            let mut peak = 0.0f64;
            for (i, &v) in values.iter().enumerate() {
                rolling += v - if i >= n { values[i - n] } else { 0.0 };
                peak = peak.max(rolling / n as f64);
            }
            if peak <= 0.0 {
                tracks.push(vec![0.0; values.len()]);
                summaries.push(0.0);
                continue;
            }
            let (mut state, mut weighted, mut total) = (0.0f64, 0.0f64, 0.0f64);
            let mut track = Vec::with_capacity(values.len());
            for &v in &values {
                let load = 3f64.min(0f64.max(v / peak));
                let target = 0f64.max(load - 0.2).powf(1.3);
                let tau = if target > state { 45.0 } else { 7.0 + 20.0 * 1f64.min(load) };
                let decay = (-bin_seconds / tau).exp();
                state = target + (state - target) * decay;
                track.push(state);
                weighted += v * state;
                total += v;
            }
            tracks.push(track);
            summaries.push(if total != 0.0 { weighted / total } else { 0.0 });
        }
    }
    if summaries.is_empty() {
        return ((0.0, 0.0), Vec::new());
    }
    let per: Vec<f64> = summaries.chunks(2).map(|c| c.iter().cloned().fold(f64::NEG_INFINITY, |m, x| if x > m { x } else { m })).collect();
    let nruns = fines.len();
    let timeline: Vec<f64> = (0..tracks[0].len())
        .map(|i| py::sum((0..tracks.len()).step_by(2).map(|j| tracks[j][i].max(tracks[j + 1][i]))) / 1usize.max(nruns) as f64)
        .collect();
    let peak = timeline.iter().cloned().fold(None, |m: Option<f64>, x| Some(match m { Some(m) if m >= x => m, _ => x })).unwrap_or(0.0);
    ((py::sum(per.iter().copied()) / per.len() as f64, peak), timeline)
}

/// difficulty_model.vector.
pub fn vector(res: &Dict, od: f64) -> [f64; 12] {
    let mut sk: Vec<(String, f64)> = Vec::new();
    if let Some(p) = res.dict("physical_sk").filter(|d| !d.0.is_empty()) {
        for (k, v) in &p.0 {
            if let crate::val::Key::S(k) = k {
                sk.push((k.clone(), 0f64.max(1.5f64.min(v.f()))));
            }
        }
    } else {
        let sc = res.dict("scores").unwrap();
        let o = sc.f("overall");
        for (k, v) in &sc.0 {
            if let crate::val::Key::S(k) = k {
                sk.push((k.clone(), 0f64.max(1.5f64.min(v.f() / 0.05f64.max(o)))));
            }
        }
    }
    let get = |k: &str| sk.iter().find(|(kk, _)| kk == k).map(|x| x.1);
    let ln = get("ln").unwrap_or(0.0);
    let cj = get("chordjack").unwrap_or(0.0).max(get("jackspeed").unwrap_or(0.0));
    let load = res.dict("endurance").map_or(0.0, |e| e.f("load"));
    let strict = (40.0 / 10f64.max(64.0 - 3.0 * 0f64.max(10f64.min(od)))).ln();
    [
        ln,
        cj,
        get("delay").unwrap_or_else(|| get("dump").unwrap_or(0.0)),
        get("minijack").unwrap_or(0.0),
        get("technical").unwrap_or(0.0),
        get("sv").unwrap_or(0.0),
        load,
        load * cj,
        res.fget("jack_width", 0.0),
        strict * 0f64.max(1.0 - ln.max(cj)),
        strict * ln,
        strict * cj,
    ]
}

/// Inverse of the fitted monotone log-rating curve (shared by the three stages).
fn invert(x: f64, slope: f64, curve: f64, delta: f64) -> f64 {
    if curve < 1e-8 {
        return delta / 1f64.max(slope);
    }
    let knee = (1.0 - slope) / (2.0 * curve);
    let u = x - knee;
    let mut y = if u >= 0.0 { u * (1.0 + curve * u) } else { u };
    y += delta;
    let v = if y >= 0.0 { 2.0 * y / (1.0 + (1.0 + 4.0 * curve * y).sqrt()) } else { y };
    v - u
}

pub fn adjustment(prm: &Params, keys: usize, raw: f64, features: &[f64]) -> f64 {
    let m = &prm.model;
    let p = mode(m, None, keys);
    let coeff = if p["coeff"].is_array() { nums(&p["coeff"]) } else { vec![0.0; FEATURES.len()] };
    let slope = num(&p["slope"], 2.0);
    let curve = 0f64.max(num(&p["curve"], 0.0));
    let x = 0.01f64.max(raw).ln();
    let reliability = 1f64.min((0f64.max(raw) / 4.0).powf(2.0));
    let delta = py::sum(coeff.iter().zip(features).map(|(a, b)| a * b)) * num(&p["strength"], 1.0) * reliability;
    let shift = invert(x, slope, curve, delta);
    (-0.7f64).max(0.7f64.min(shift)).exp() * num(&p["scale"], 1.0)
}

fn exec_f(execution: Option<&Dict>, name: &str) -> f64 {
    execution.map_or(0.0, |e| e.fget(name, 0.0))
}

pub fn baseline_rating(prm: &Params, keys: usize, raw: f64, features: &[f64], execution: Option<&Dict>) -> f64 {
    let m = &prm.model;
    if let Some(Val::Dict(physical)) = execution.and_then(|e| e.get("scroll_base")) {
        let praw = physical.f("raw");
        if praw > 0.0 {
            let pf = physical.get("features").unwrap().list_f();
            return raw / praw * baseline_rating(prm, keys, praw, &pf, physical.dict("execution"));
        }
    }
    let base = raw * adjustment(prm, keys, raw, features);
    let p = mode(m, Some("execution"), keys);
    if !present(p) {
        return gesture_rating(prm, keys, base, raw, execution);
    }
    let center = if p["center"].is_array() { nums(&p["center"]) } else { vec![0.0; EXTRA_FEATURES.len()] };
    let coeff = nums(&p["coeff"]);
    let mut delta = py::sum(EXTRA_FEATURES.iter().zip(&coeff).zip(&center).map(|((name, c), mm)| c * (exec_f(execution, name) - mm)));
    let bc = nums(&p["base_coeff"]);
    let bm = nums(&p["base_center"]);
    delta += py::sum(bc.iter().zip(features).zip(&bm).map(|((c, v), mm)| c * (v - mm)));
    delta *= 1f64.min((raw / 4.0).powf(2.0));
    let slope = num(&p["slope"], f64::NAN);
    let curve = 0f64.max(num(&p["curve"], 0.0));
    let x = 0.05f64.max(base).ln();
    let shift = invert(x, slope, curve, delta);
    gesture_rating(prm, keys, base * (-0.5f64).max(0.5f64.min(shift)).exp(), raw, execution)
}

pub fn gesture_rating(prm: &Params, keys: usize, base: f64, raw: f64, execution: Option<&Dict>) -> f64 {
    let m = &prm.model;
    let stage = &m["gestures"];
    let p = &stage["modes"][keys.to_string()];
    if !present(p) || base <= 0.0 {
        return base;
    }
    let names: Vec<&str> = stage["features"].as_array().map(|a| a.iter().filter_map(|x| x.as_str()).collect()).unwrap_or_default();
    let coeff = nums(&p["coeff"]);
    let mut delta = py::sum(names.iter().zip(&coeff).map(|(name, c)| c * exec_f(execution, name)));
    if delta == 0.0 {
        return base;
    }
    delta *= 1f64.min((0f64.max(raw) / 4.0).powf(2.0));
    let slope = num(&p["slope"], f64::NAN);
    let curve = 0f64.max(num(&p["curve"], 0.0));
    let x = 0.05f64.max(base).ln();
    let shift = invert(x, slope, curve, delta);
    base * (-0.5f64).max(0.5f64.min(shift)).exp()
}

pub fn final_rating(prm: &Params, keys: usize, raw: f64, features: &[f64], execution: Option<&Dict>) -> f64 {
    let base = baseline_rating(prm, keys, raw, features, execution);
    final_from_base(prm, keys, base, execution)
}

fn final_from_base(prm: &Params, keys: usize, base: f64, execution: Option<&Dict>) -> f64 {
    let values = execution.and_then(|e| e.get("residual_vector")).map(|v| v.list_f());
    let value = correction(prm, keys, base, values.as_deref()) * tree_shift(prm, keys, base, values.as_deref()).exp();
    value * execution.map_or(1.0, |e| e.fget("rolled_factor", 1.0))
}

// ---- structural_residual ----

pub const SR_SKILLS: [&str; 25] = [
    "stream", "jumpstream", "handstream", "delay", "chordstream", "bracket", "chordjack", "minijack", "longjack",
    "jackspeed", "ln", "release", "inverse", "hybrid", "shield", "jumptrill", "splittrill", "trill1h", "anchor",
    "technical", "patterntech", "rhythmtech", "vibro", "stamina", "sv",
];
pub const SR_EXECUTION: [&str; 11] = [
    "mash", "pseudo_mash", "ln_flow", "peak_pressure", "transition_pressure", "vibro_easy", "vibro_control",
    "jumptrill_easy", "roll_control", "roll_anchor", "jumptrill_sustain",
];
pub const SR_NAMES_LEN: usize = 25 + 14 + 11 + 10;

/// structural_residual.vector from the apply() feature view.
pub fn sr_vector(sk_raw: &[(String, f64)], st: &Dict, ex: &Dict, en: &Dict, rate: f64, length: f64, notes: f64, keys: usize, ln_frac: f64, stam: f64, cf: &[f64]) -> Vec<f64> {
    let sk: Vec<(&str, f64)> = sk_raw.iter().map(|(k, v)| (k.as_str(), py::round_digits(0f64.max(1.5f64.min(*v)), 4))).collect();
    let g = |k: &str| sk.iter().find(|(kk, _)| *kk == k).map_or(0.0, |x| x.1);
    let truthy = |k: &str| st.get(k).filter(|v| v.truthy()).map(|v| v.f());
    let seconds = 0.001f64.max(truthy("play_span").unwrap_or(length / 1000.0) / rate);
    let density = truthy("nps").unwrap_or(notes / seconds) / 1usize.max(keys) as f64;
    let width = 0f64.max(st.fget("chord", 1.0)) / 1usize.max(keys) as f64;
    let overlap = 0f64.max(st.fget("overlap", 0.0));
    let load = 0f64.max(en.fget("load", 0.0));
    let (ln, cj) = (g("ln"), g("chordjack"));
    let rice = g("stream").max(g("delay")).max(g("dump"));
    let mut values: Vec<f64> = SR_SKILLS.iter().map(|k| g(k)).collect();
    values.extend([
        ln_frac,
        2f64.min(stam),
        (seconds / 120.0).ln_1p(),
        density.ln_1p(),
        width,
        overlap,
        st.fget("repeat", 0.0),
        5f64.min(st.fget("burst", 1.0)),
        st.fget("active", 0.0),
        5f64.min(st.fget("rhythm", 0.0)),
        st.fget("all_keys", 0.0),
        load,
        en.fget("peak", 0.0),
        if cf.len() > 8 { cf[8] } else { 0.0 },
    ]);
    values.extend(SR_EXECUTION.iter().map(|k| ex.fget(k, 0.0)));
    values.extend([
        g("chordstream") * g("bracket"),
        cj * overlap,
        cj * width,
        cj * load,
        g("minijack") * density.ln_1p(),
        ln * g("release"),
        ln * g("hybrid"),
        ln * ex.fget("ln_flow", 0.0),
        rice * g("anchor"),
        g("jumptrill") * ex.fget("roll_control", 0.0),
    ]);
    values.into_iter().map(|v| if v.is_finite() { v } else { 0.0 }).collect()
}

fn basis(values: &[f64], model: &Value) -> Vec<f64> {
    let (lo, hi, mean, scale) = (nums(&model["low"]), nums(&model["high"]), nums(&model["mean"]), nums(&model["scale"]));
    let z: Vec<f64> = values.iter().zip(&lo).zip(&hi).map(|((&x, &l), &h)| l.max(h.min(x))).collect();
    let mut cols: Vec<f64> = z.iter().zip(&mean).zip(&scale).map(|((v, m), s)| (v - m) / s).collect();
    if let Some(hs) = model["hinges"].as_array() {
        for h in hs {
            let a = h.as_array().unwrap();
            let (index, knot, m, s) = (a[0].as_u64().unwrap() as usize, num(&a[1], 0.0), num(&a[2], 0.0), num(&a[3], 1.0));
            cols.push((0f64.max(z[index] - knot) - m) / s);
        }
    }
    cols
}

pub fn correction(prm: &Params, keys: usize, baseline: f64, values: Option<&[f64]>) -> f64 {
    let m = &prm.model;
    correction_with(prm, &m["structural_residual"]["modes"][keys.to_string()], keys, baseline, values, None)
}

fn correction_with(prm: &Params, model: &Value, _keys: usize, baseline: f64, values: Option<&[f64]>, coeff: Option<&[f64]>) -> f64 {
    let m = &prm.model;
    let stage = &m["structural_residual"];
    let Some(values) = values else { return baseline };
    if !present(model) || baseline <= 0.0 || values.len() != SR_NAMES_LEN {
        return baseline;
    }
    let own = nums(&model["coeff"]);
    let coeff = coeff.unwrap_or(&own);
    let mut delta = py::sum(basis(values, model).iter().zip(coeff).map(|(a, b)| a * b)) + num(&model["intercept"], 0.0);
    let mx = num(&stage["max_log_shift"], 0.25);
    delta = (-mx).max(mx.min(delta));
    delta *= 1f64.min((baseline / 4.0).powf(2.0));
    baseline * delta.exp()
}

/// Typed tree ensemble (structure-of-arrays) for one keymode.
pub struct Trees {
    pub intercept: f64,
    pub scales: Option<Vec<f64>>,
    pub depth3: bool,
    pub feats: Vec<[usize; 3]>,
    pub cuts: Vec<[f64; 3]>,
    pub leaves: Vec<[f64; 8]>,
    /// Generic trees (any depth): (feats, cuts, leaves).
    pub generic: Vec<(Vec<usize>, Vec<f64>, Vec<f64>)>,
}

pub fn trees(model: &Value) -> Option<Trees> {
    if !present(model) {
        return None;
    }
    let scales = model["scales"].as_array().map(|a| a.iter().map(|x| x.as_f64().unwrap()).collect::<Vec<f64>>());
    let mut t = Trees { intercept: num(&model["intercept"], 0.0), scales, depth3: true, feats: vec![], cuts: vec![], leaves: vec![], generic: vec![] };
    for tree in model["trees"].as_array().unwrap() {
        let a = tree.as_array().unwrap();
        let f: Vec<usize> = a[0].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
        let c = nums(&a[1]);
        let l = nums(&a[2]);
        t.generic.push((f, c, l));
    }
    t.depth3 = t.scales.is_some() && t.generic.iter().all(|(f, _, l)| f.len() == 3 && l.len() == 8);
    if t.depth3 {
        for (f, c, l) in &t.generic {
            t.feats.push([f[0], f[1], f[2]]);
            t.cuts.push([c[0], c[1], c[2]]);
            t.leaves.push(std::array::from_fn(|i| l[i]));
        }
    }
    Some(t)
}

pub fn tree_shift(prm: &Params, keys: usize, baseline: f64, values: Option<&[f64]>) -> f64 {
    let Some(values) = values else { return 0.0 };
    let Some(t) = prm.trees(keys) else { return 0.0 };
    if baseline <= 0.0 || values.len() != SR_NAMES_LEN {
        return 0.0;
    }
    tree_eval(t, baseline, values)
}

pub fn tree_eval(t: &Trees, baseline: f64, values: &[f64]) -> f64 {
    let mut x = values.to_vec();
    x.push(0.05f64.max(baseline).ln());
    let mut out = t.intercept;
    let p = |xv: f64, c: f64, s: f64| 1.0 / (1.0 + (-(-60f64).max(60f64.min((xv - c) / s))).exp());
    if t.depth3 {
        let sc = t.scales.as_ref().unwrap();
        for i in 0..t.feats.len() {
            let [f, g, h] = t.feats[i];
            let c = t.cuts[i];
            let l = &t.leaves[i];
            let p0 = p(x[f], c[0], sc[f]);
            let p1 = p(x[g], c[1], sc[g]);
            let p2 = p(x[h], c[2], sc[h]);
            let (a0, a1) = (1.0 - p0, p0);
            let (b0, b1, b2, b3) = (a0 * (1.0 - p1), a0 * p1, a1 * (1.0 - p1), a1 * p1);
            out += py::sum([
                b0 * (1.0 - p2) * l[0], b0 * p2 * l[1], b1 * (1.0 - p2) * l[2], b1 * p2 * l[3],
                b2 * (1.0 - p2) * l[4], b2 * p2 * l[5], b3 * (1.0 - p2) * l[6], b3 * p2 * l[7],
            ]);
        }
    } else {
        for (feats, cuts, leaves) in &t.generic {
            match &t.scales {
                None => {
                    let mut i = 0usize;
                    for (&f, &c) in feats.iter().zip(cuts) {
                        i = 2 * i + (x[f] > c) as usize;
                    }
                    out += leaves[i];
                }
                Some(sc) => {
                    let mut weight = vec![1.0f64];
                    for (&f, &c) in feats.iter().zip(cuts) {
                        let q = p(x[f], c, sc[f]);
                        weight = weight.iter().flat_map(|&w| [w * (1.0 - q), w * q]).collect();
                    }
                    out += py::sum(weight.iter().zip(leaves).map(|(w, v)| w * v));
                }
            }
        }
    }
    tree_limit(out, baseline)
}

pub fn tree_limit(shift: f64, baseline: f64) -> f64 {
    (-0.25f64).max(0.25f64.min(shift)) * 1f64.min((baseline / 4.0).powf(2.0))
}

/// difficulty_model.apply on the result dict.
pub fn apply(prm: &Params, res: &mut Dict, od: f64, rate: f64) {
    let keys = res.f("keys") as usize;
    let raw = res.dict("scores").unwrap().f("overall");
    res.set("raw_overall", raw);
    let cf = vector(res, od);
    res.set("calibration_features", Val::floats(&cf));
    if raw != 0.0 && res.has("structure") {
        let sk: Vec<(String, f64)> = res.dict("scores").unwrap().0.iter().map(|(k, v)| {
            let crate::val::Key::S(k) = k else { unreachable!() };
            (k.clone(), py::round_digits(v.f() / raw, 4))
        }).collect();
        let notes = res.f("notes");
        let empty = Dict::new();
        let stam = res.dict("levels").and_then(|l| l.0.iter().find(|(k, _)| matches!(k, crate::val::Key::F(x) if *x == 120.0)).map(|(_, v)| v.f())).unwrap_or(raw) / raw;
        let rv = sr_vector(
            &sk,
            res.dict("structure").unwrap_or(&empty),
            res.dict("execution").unwrap_or(&empty),
            res.dict("endurance").unwrap_or(&empty),
            rate,
            res.fget("length", 0.0),
            notes,
            keys,
            res.f("ln_notes") / 1f64.max(notes),
            stam,
            &cf,
        );
        res.dict_mut("execution").unwrap().set("residual_vector", Val::floats(&rv));
    }
    let base = baseline_rating(prm, keys, raw, &cf, res.dict("execution"));
    res.set("baseline_overall", base);
    let multiplier = if raw != 0.0 { final_from_base(prm, keys, base, res.dict("execution")) / 0.000001f64.max(raw) } else { 1.0 };
    res.dict_mut("scores").unwrap().map_values(|v| v * multiplier);
    scale_archetypes(res, multiplier);
    if let Some(Val::List(tl)) = res.get_mut("timeline") {
        for v in tl.iter_mut() {
            *v = Val::F(v.f() * multiplier);
        }
    }
    if let Some(l) = res.dict_mut("levels") {
        l.map_values(|v| v * multiplier);
    } else {
        res.set("levels", Dict::new());
    }
    res.set("od", od);
    let sc = res.dict("scores").unwrap().f("overall");
    res.set("structural_correction", sc - res.f("baseline_overall"));
}

pub fn scale_archetypes(res: &mut Dict, k: f64) {
    if let Some(Val::List(items)) = res.get_mut("archetypes") {
        for a in items.iter_mut() {
            if let Val::Dict(d) = a {
                let r = d.f("rating");
                d.set("rating", r * k);
            }
        }
    }
}

// ---- score_units ----

pub fn unit_rating(value: f64, od: f64, model: &Value) -> f64 {
    if value <= 0.0 {
        return value;
    }
    let knee = num(&model["knee"], 4.0);
    let x = value.ln();
    let inp = nums(&model["input"]);
    let (a, b) = (inp[0], inp[1]);
    let mut target = a * x + b * 0f64.max(x - knee.ln()).powf(2.0);
    target += num(&model["window"], 0.0) * ((64.0 - 3.0 * 0f64.max(10f64.min(od))) / (64.0 - 3.0 * num(&model["reference_od"], 8.0))).ln();
    let r = nums(&model["reference"]);
    let (origin, slope, curve) = (r[0], r[1], r[2]);
    let delta = target - origin - slope * knee.ln();
    if delta <= 0.0 || curve == 0.0 {
        return ((target - origin) / slope).exp();
    }
    knee * (2.0 * delta / (slope + (slope * slope + 4.0 * curve * delta).sqrt())).exp()
}

pub fn display_rating(value: f64, od: f64, model: &Value) -> f64 {
    (value + unit_rating(value, od, model)) / 2.0
}

pub fn display_factor(prm: &Params, keys: usize, baseline: f64, execution: Option<&Dict>, original: f64) -> f64 {
    let m = &prm.model;
    let unit = &m["score_units"]["modes"][keys.to_string()];
    let coeff = &unit["display_coeff"];
    let values = execution.and_then(|e| e.get("residual_vector")).map(|v| v.list_f());
    if coeff.is_null() || values.is_none() || original <= 0.0 {
        return 1.0;
    }
    let coeff = nums(coeff);
    let model = &m["structural_residual"]["modes"][keys.to_string()];
    let rolled = execution.map_or(1.0, |e| e.fget("rolled_factor", 1.0));
    let corrected = correction_with(prm, model, keys, baseline / rolled, values.as_deref(), Some(&coeff))
        * rolled
        * tree_shift(prm, keys, baseline, values.as_deref()).exp();
    corrected / original
}

/// score_units.apply (after card/describe were taken from the pre-unit result).
pub fn units_apply(prm: &Params, res: &mut Dict) {
    let m = &prm.model;
    let keys = res.f("keys") as usize;
    let model = &m["score_units"]["modes"][keys.to_string()];
    if !present(model) || !res.get("horizon").is_some_and(|h| h.truthy()) {
        return;
    }
    let od = res.f("od");
    let factor = display_factor(prm, keys, res.f("baseline_overall"), res.dict("execution"), res.dict("scores").unwrap().f("overall"));
    let convert = |v: f64| display_rating(v * factor, od, model);
    let mut pre = Dict::new();
    for k in ["scores", "levels", "baseline_overall"] {
        pre.set(k, res.get(k).unwrap().clone());
    }
    res.set("preunit_result", pre);
    let o = res.dict("scores").unwrap().f("overall");
    res.set("preunit_overall", o);
    res.dict_mut("scores").unwrap().map_values(convert);
    res.dict_mut("levels").unwrap().map_values(convert);
    if let Some(Val::List(tl)) = res.get_mut("timeline") {
        for v in tl.iter_mut() {
            *v = Val::F(convert(v.f()));
        }
    }
    for field in ["archetypes", "unit_card"] {
        if let Some(Val::List(items)) = res.get_mut(field) {
            for a in items.iter_mut() {
                if let Val::Dict(d) = a {
                    let r = d.f("rating");
                    d.set("rating", convert(r));
                }
            }
        }
    }
    let b = res.f("baseline_overall");
    res.set("baseline_overall", display_rating(b, od, model));
    if res.has("no_sv_overall") {
        let mut no_sv_factor = factor;
        if let Some(physical) = res.dict("execution").and_then(|e| e.dict("scroll_base")) {
            let pf = physical.get("features").unwrap().list_f();
            let baseline = baseline_rating(prm, keys, physical.f("raw"), &pf, physical.dict("execution"));
            no_sv_factor = display_factor(prm, keys, baseline, physical.dict("execution"), res.f("no_sv_overall"));
        }
        let v = res.f("no_sv_overall");
        res.set("no_sv_overall", display_rating(v * no_sv_factor, od, model));
    }
    let sc = res.dict("scores").unwrap().f("overall");
    res.set("structural_correction", sc - res.f("baseline_overall"));
}
