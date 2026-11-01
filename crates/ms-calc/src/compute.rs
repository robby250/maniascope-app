//! Full analysis of a chart at a rate (skill_calc.compute/_compute, rolled_execution).
use crate::card;
use crate::chart::Chart;
use crate::demand::{self, Action, Geometry, Parts, Run, P};
use crate::execution::{self, Info};
use crate::gestures;
use crate::kern::smooth;
use crate::model;
use crate::params::Params;
use crate::pattern;
use crate::py;
use crate::scroll;
use crate::skills::{self, combine, horizon_levels, phase_total, stamina, star, window_level, Series, Sk};
use crate::structure;
use crate::val::{Dict, Key, Val};

pub const MODEL_VERSION: &str = "3.5-technical-presence";
const SV_NAMES: [&str; 5] = ["Fast SV", "Slowjam SV", "Accel SV", "Stutter SV", "SV"];

pub struct Trace {
    pub phases: Vec<Vec<f64>>,
    pub total: Vec<f64>,
    pub series: Series,
    pub switch: Vec<f64>,
    pub parts: Parts,
    pub result: Dict,
    /// centre_left → (motor actions, fine tracks of that run)
    pub action_runs: Option<Vec<(bool, Vec<Action>, [Vec<f64>; 3])>>,
}

#[derive(Default)]
pub struct Shared {
    pub trace: Option<Vec<(bool, Trace)>>,
    pub execution_plans: bool,
    pub geometry: Option<Geometry>,
    pub motor_actions: Vec<(bool, Vec<Action>)>,
    pub plain_runs: Option<Vec<Run>>,
    pub execution: Option<Info>,
    pub gestures: Option<gestures::Analysis>,
    pub tech_profile: Option<[f64; 3]>,
    pub quadstream: Option<(Vec<f64>, f64)>,
    pub endurance: Option<((f64, f64), Vec<f64>)>,
    pub structure: Option<Dict>,
    /// rolled_execution reference charts: only raw/baseline overall and the phase totals are read.
    pub light: bool,
}
impl Shared {
    fn trace(&self, sv: bool) -> Option<&Trace> {
        self.trace.as_ref()?.iter().find(|(k, _)| *k == sv).map(|(_, t)| t)
    }
}

fn levels_dict(levels: &[f64; 4], prm: &Params) -> Dict {
    let mut d = Dict::new();
    for (h, &v) in prm.k.HORIZONS.iter().zip(levels) {
        d.setf(h.0, v);
    }
    d
}

fn argmax(v: &[f64]) -> usize {
    let mut b = 0;
    for i in 1..v.len() {
        if v[i] > v[b] {
            b = i;
        }
    }
    b
}

fn series_dict_scores(sc: &mut Dict, series: &Series, notes: usize, prm: &Params) {
    for &key in &series.keys {
        if key == Sk::switch {
            continue;
        }
        sc.set(key.name(), combine(&horizon_levels(series.get(key), &prm.k), notes, &prm.k));
    }
}

pub fn compute_inner(chart: &Chart, rate: f64, sv: bool, shared: &mut Shared, prm: &Params) -> Result<Dict, String> {
    let k = &prm.k;
    if !(rate > 0.0) || !rate.is_finite() {
        return Err(format!("invalid rate: {rate}"));
    }
    let n = chart.len();
    let keys = chart.keys;
    let mut res = Dict::new();
    res.set("model", MODEL_VERSION);
    res.set("sv_name", "SV");
    res.set("keys", keys);
    res.set("rate", rate);
    res.set("notes", n);
    res.set("ln_notes", chart.t.iter().zip(&chart.end).filter(|(t, e)| e > t).count());
    let mut sc = Dict::new();
    sc.set("overall", 0.0);
    for s in skills::skills(keys) {
        sc.set(s.name(), 0.0);
    }
    res.set("scores", sc.clone());
    res.set("horizon", Val::None);
    res.set("timeline", Val::List(Vec::new()));
    res.set("sv", if chart.sv.is_empty() { "constant" } else if sv { "included" } else { "ignored" });
    if n < 2 {
        return Ok(res);
    }
    let read = if sv && !prm.off("sv_reading") { Some(scroll::reading(chart, rate, prm)?) } else { None };
    let any_strain = read.as_ref().is_some_and(|r| r.0.iter().any(|&x| x != 0.0));
    let runs: Vec<Run> = if !sv && shared.plain_runs.is_some() {
        shared.plain_runs.clone().unwrap()
    } else {
        let want_plain = sv && !chart.sv.is_empty() && any_strain;
        let g = demand::geometry(chart, rate, k);
        let mut rs = Vec::new();
        let mut plains = Vec::new();
        for cl in if keys % 2 == 1 { vec![true, false] } else { vec![true] } {
            let mut actions = if shared.execution_plans { Some(vec![Action::default(); n]) } else { None };
            let (run, plain) = demand::demand(
                chart, &g, cl, read.as_ref().map(|r| &r.0[..]), read.as_ref().map(|r| &r.1[..]), want_plain, actions.as_mut(), k,
            );
            rs.push(run);
            if let Some(p) = plain {
                plains.push(p);
            }
            if let Some(a) = actions {
                shared.motor_actions.retain(|(c, _)| *c != cl);
                shared.motor_actions.push((cl, a));
            }
        }
        shared.geometry = Some(g);
        if want_plain {
            shared.plain_runs = Some(plains);
        }
        rs
    };
    let m = runs.len();
    let (total, mut parts): (Vec<f64>, Parts) = if m == 2 {
        let t: Vec<f64> = runs[0].totals.iter().zip(&runs[1].totals).map(|(l, r)| (l + r) / 2.0).collect();
        let mut p = runs[0].parts.clone();
        for (x, y) in p.data.iter_mut().zip(&runs[1].parts.data) {
            *x = (*x + y) / 2.0;
        }
        (t, p)
    } else {
        (runs[0].totals.clone(), runs[0].parts.clone())
    };
    let mut levels = horizon_levels(&total, k);
    let span = (chart.last_end() - chart.t[0]) / 1000.0 / rate;
    let long_chart = span > 120.0;
    let mut uniform = if long_chart { window_level(&total, k.UNIFORM_WIN, k) } else { 0.0 };
    let tracing = shared.trace.is_some();
    let mut phase_totals: Vec<Vec<f64>> = if tracing { vec![total.clone()] } else { Vec::new() };
    for p in 1..k.PHASES {
        let tp = if m == 1 {
            phase_total(&runs[0].fine, p, k)
        } else {
            let a = phase_total(&runs[0].fine, p, k);
            let b = phase_total(&runs[1].fine, p, k);
            a.iter().zip(&b).map(|(x, y)| py::sum([*x, *y]) / m as f64).collect()
        };
        let lv = horizon_levels(&tp, k);
        for (a, b) in levels.iter_mut().zip(lv) {
            *a = a.max(b);
        }
        if long_chart {
            uniform = uniform.max(window_level(&tp, k.UNIFORM_WIN, k));
        }
        if tracing {
            phase_totals.push(tp);
        }
    }
    let uniformity = if long_chart && levels[3] > 0.0 {
        (uniform / levels[3] / k.UNIFORM_R0).powf(k.UNIFORM_W * (span / 120.0).log2())
    } else {
        1.0
    };
    let overall = combine(&levels, n, k);
    sc.set("overall", overall);
    res.set("horizon", k.HORIZONS[argmax(&levels)].0);
    res.set("levels", levels_dict(&levels, prm));
    let mx = levels.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    sc.set("stamina", if mx > 0.0 { stamina(&total, k) * overall / mx } else { 0.0 });
    let mut series = skills::attribute(&parts, &total, keys, false, k);
    series_dict_scores(&mut sc, &series, n, prm);
    let mut psk = Dict::new();
    for (key, v) in &sc.0 {
        if let Key::S(s) = key {
            psk.set(s, v.f() / 0.05f64.max(overall));
        }
    }
    res.set("physical_sk", psk);
    if shared.execution.is_none() {
        shared.execution = Some(execution::analyse(chart, shared.geometry.as_ref().unwrap(), k.BIN));
    }
    if shared.gestures.is_none() {
        // Off: empty series (every reader is length-guarded), zero features, no roll plans.
        shared.gestures = Some(if prm.off("gestures") {
            gestures::Analysis::default()
        } else {
            gestures::analyse(chart, rate, k.BIN, shared.execution_plans)
        });
    }
    if shared.quadstream.is_none() {
        shared.quadstream =
            Some(if prm.off("quadstream") { (vec![0.0; total.len()], 0.0) } else { pattern::quadstream(chart, rate, k.BIN) });
    }
    let (changed, extra) = if prm.off("execution") {
        (parts.clone(), Vec::new())
    } else {
        execution::attribute(
            &parts, &total, &mut series, chart, rate, shared.execution.as_ref().unwrap(), shared.gestures.as_ref().unwrap(),
            shared.quadstream.as_ref().unwrap(), k,
        )
    };
    parts = changed;
    let mut ex = Dict::new();
    for (name, v) in extra {
        ex.set(name, v);
    }
    res.set("execution", ex);
    let light = shared.light;
    if shared.tech_profile.is_none() && !light {
        shared.tech_profile = Some(pattern::tech_profile(chart, rate, shared.execution.as_ref().unwrap(), k));
    }
    let tp = shared.tech_profile.unwrap_or([0.0; 3]);
    let mut tpd = Dict::new();
    tpd.set("version", pattern::TECH_PROFILE_VERSION);
    tpd.set("technical", tp[0]);
    tpd.set("patterntech", tp[1]);
    tpd.set("rhythmtech", tp[2]);
    if !prm.off("tech_profile") {
        res.set("tech_profile", tpd);
    }
    let switch = series.remove(Sk::switch);
    if !light {
        series_dict_scores(&mut sc, &series, n, prm);
        res.set("archetypes", Val::List(card::archetypes(&series, &sc, n, &switch, k)));
        res.set("tech_where", Val::List(card::tech_where(&series, &parts, chart.t[0] / 1000.0, rate, k)));
    }
    let sv_tot = py::sum(parts.get(P::sv).iter().copied());
    if sv_tot != 0.0 {
        let shares: Vec<f64> = demand::SV_KINDS.iter().map(|&p| py::sum(parts.get(p).iter().copied()) / sv_tot).collect();
        if let Some(i) = shares.iter().position(|&f| f >= k.SV_KIND_SHARE) {
            res.set("sv_name", SV_NAMES[i]);
        }
    }
    let w = 1usize.max((k.HORIZONS[0].0 / k.BIN) as usize);
    let mut acc = 0.0f64;
    let mut tl = Vec::with_capacity(total.len());
    for (i, &d) in total.iter().enumerate() {
        acc = 0f64.max(acc + d - if i >= w { total[i - w] } else { 0.0 });
        tl.push(star(k.SCALE * (acc / w.min(i + 1) as f64).sqrt() / k.HORIZONS[0].1, k));
    }
    let rk = rate.powf(k.RATE_G);
    sc.map_values(|v| v * rk);
    res.set("scores", sc);
    model::scale_archetypes(&mut res, rk);
    res.set("timeline", Val::floats(&tl.iter().map(|x| x * rk).collect::<Vec<f64>>()));
    if shared.endurance.is_none() {
        let fines: Vec<&[Vec<f64>; 3]> = runs.iter().map(|r| &r.fine).collect();
        shared.endurance = Some(if prm.off("endurance") {
            ((0.0, 0.0), vec![0.0; total.len()])
        } else {
            model::endurance(&fines, k.BIN, k.PHASES)
        });
    }
    let ((load, peak), ftl) = shared.endurance.clone().unwrap();
    let mut en = Dict::new();
    en.set("load", load);
    en.set("peak", peak);
    res.set("endurance", en);
    res.set("fatigue_timeline", Val::floats(&ftl));
    let mut rs: Vec<(f64, u32)> = Vec::new();
    for i in 0..n {
        let t = chart.t[i];
        match rs.last_mut() {
            Some(x) if x.0 == t => x.1 |= 1 << chart.col[i],
            _ => rs.push((t, 1 << chart.col[i])),
        }
    }
    let mut pressure = 0.0f64;
    let mut pairs = 0usize;
    for w2 in rs.windows(2) {
        let gap = (w2[1].0 - w2[0].0) / 1000.0 / rate;
        if 0.0 < gap && gap < 0.25 {
            pressure += ((w2[0].1 & w2[1].1).count_ones() as f64 / keys as f64).powf(2.0) * 2f64.min(0.125 / gap);
            pairs += 1;
        }
    }
    res.set("jack_width", pressure / 1usize.max(pairs) as f64);
    if light {
        // Same values as the full path: raw = rate-scaled overall, baseline × uniformity.
        let raw = res.dict("scores").unwrap().f("overall");
        res.set("raw_overall", raw);
        let cf = model::vector(&res, chart.od);
        let base = model::baseline_rating(prm, keys, raw, &cf, res.dict("execution")) * uniformity;
        res.set("baseline_overall", base);
        if tracing {
            let tr = Trace { phases: phase_totals, total, series, switch, parts, result: res.clone(), action_runs: None };
            let t = shared.trace.as_mut().unwrap();
            t.retain(|(key, _)| *key != sv);
            t.push((sv, tr));
        }
        return Ok(res);
    }
    let has_bracket = res.dict("scores").unwrap().has("bracket");
    let mut bracket_gate = 1.0f64;
    if has_bracket {
        let half = keys / 2;
        let lo = ((1u64 << half) - 1) as u32;
        let hm = [lo, lo << (keys - half)];
        let (mut moves, mut interleaved) = (0usize, 0usize);
        for w2 in rs.windows(2) {
            for h in hm {
                let (pa, pb) = (w2[0].1 & h, w2[1].1 & h);
                if pa != 0 && pb != 0 && pa & pb == 0 && pa.count_ones().max(pb.count_ones()) >= 2 {
                    moves += 1;
                    interleaved += (crate::kern::inner(pa, keys) & pb != 0 || crate::kern::inner(pb, keys) & pa != 0) as usize;
                }
            }
        }
        bracket_gate = if moves > 0 { 1f64.min(interleaved as f64 / (k.BRACKET_IL_FULL * moves as f64)) } else { 1.0 };
    }
    let sv_physical = sv && !chart.sv.is_empty() && any_strain;
    if sv_physical {
        let physical = compute_inner(chart, rate, false, shared, prm)?;
        res.set("no_sv_overall", physical.dict("scores").unwrap().f("overall"));
        let summary = read.as_ref().unwrap().2;
        let mut rd = Dict::new();
        for (name, v) in scroll::FEATURES.iter().zip(summary) {
            rd.set(name, v);
        }
        res.set("reading", rd);
        let mut base = Dict::new();
        base.set("raw", physical.f("raw_overall"));
        base.set("features", physical.get("calibration_features").unwrap().clone());
        base.set("execution", physical.get("execution").unwrap().clone());
        res.dict_mut("execution").unwrap().set("scroll_base", base);
    }
    if shared.structure.is_none() {
        shared.structure = Some(structure::structure(chart, rate, k));
    }
    res.set("structure", shared.structure.clone().unwrap());
    let length = chart.last_end() - chart.t[0];
    res.set("length", if length.fract() == 0.0 && length.abs() < 9e15 { Val::Int(length as i64) } else { Val::F(length) });
    if prm.model["joint"]["modes"][chart.keys.to_string()].is_object() {
        if let Some(e) = res.dict_mut("execution") {
            crate::flow::flow(chart, rate).store(e);
        }
    }
    model::apply(prm, &mut res, chart.od, rate);
    res.dict_mut("scores").unwrap().map_values(|v| v * uniformity);
    model::scale_archetypes(&mut res, uniformity);
    let b = res.f("baseline_overall");
    res.set("baseline_overall", b * uniformity);
    res.set("uniformity", uniformity);
    if has_bracket && bracket_gate < 1.0 {
        let scd = res.dict_mut("scores").unwrap();
        let bv = scd.f("bracket") * bracket_gate;
        scd.set("bracket", bv);
        let cs = scd.f("chordstream");
        if let Some(Val::List(items)) = res.get_mut("archetypes") {
            for a in items.iter_mut() {
                if let Val::Dict(d) = a {
                    if matches!(d.get("parts"), Some(Val::Tuple(p)) if p.iter().any(|x| matches!(x, Val::Str(s) if s == "bracket"))) {
                        let r = d.f("rating");
                        d.set("rating", r * bracket_gate);
                    }
                }
            }
            items.retain(|a| match a {
                Val::Dict(d) => !matches!(d.get("name"), Some(Val::Str(s)) if s == "Bracket") || bv >= k.REFINE_Q * cs,
                _ => true,
            });
        }
    }
    res.dict_mut("levels").unwrap().map_values(|v| v * rk);
    if tracing {
        let action_runs = if shared.execution_plans {
            let cls: Vec<bool> = if keys % 2 == 1 { vec![true, false] } else { vec![true] };
            Some(
                cls.iter()
                    .zip(runs.iter())
                    .map(|(&cl, run)| {
                        let acts = shared.motor_actions.iter().find(|(c, _)| *c == cl).map(|(_, a)| a.clone()).unwrap_or_default();
                        (cl, acts, run.fine.clone())
                    })
                    .collect(),
            )
        } else {
            None
        };
        let tr = Trace { phases: phase_totals, total, series, switch, parts, result: res.clone(), action_runs };
        let t = shared.trace.as_mut().unwrap();
        t.retain(|(key, _)| *key != sv);
        t.push((sv, tr));
    }
    Ok(res)
}

/// Full analysis of a parsed chart at a constant rate (skill_calc.compute).
pub fn compute(chart: &Chart, rate: f64, sv: bool, prm: &Params) -> Result<Dict, String> {
    let mut strength = prm.model["rolled_execution"]["modes"][chart.keys.to_string()]["strength"].as_f64().unwrap_or(0.0);
    if !(0.0..=1.0).contains(&strength) {
        return Err(format!("rolled_execution strength {strength} is outside 0–1"));
    }
    if prm.off("rolled_execution") {
        strength = 0.0;
    }
    if strength != 0.0 && !possible(chart, rate) {
        strength = 0.0;
    }
    let mut shared = Shared::default();
    if strength != 0.0 {
        shared.trace = Some(Vec::new());
        shared.execution_plans = true;
    }
    let mut result = compute_inner(chart, rate, sv, &mut shared, prm)?;
    if strength != 0.0 && result.get("horizon").is_some_and(|h| h.truthy()) {
        rolled_apply(chart, rate, sv, &mut result, &mut shared, strength, prm)?;
    }
    let mut ap = Dict::new();
    ap.set("version", 3i64);
    ap.set("ln_chordstream", if prm.off("archetype_profile") { 0.0 } else { card::profile(chart, rate) });
    result.set("archetype_profile", ap);
    let unit = &prm.model["score_units"]["modes"][chart.keys.to_string()];
    if unit.as_object().is_some_and(|o| !o.is_empty()) && result.get("horizon").is_some_and(|h| h.truthy()) {
        let uc = card::card(&result);
        result.set("unit_card", Val::List(uc));
        let description = card::describe(&result, &prm.k).0;
        model::units_apply(prm, &mut result);
        result.set("preunit_description", description);
    }
    Ok(result)
}

// ---- rolled_execution ----

pub fn possible(chart: &Chart, rate: f64) -> bool {
    let maximum = 45.0 * rate;
    chart.t.windows(2).any(|w| 0.0 < w[1] - w[0] && w[1] - w[0] < maximum)
}

fn reference(chart: &Chart, rate: f64, plans: &[gestures::Plan]) -> Option<(Chart, Vec<f64>)> {
    if plans.is_empty() {
        return None;
    }
    let n = chart.len();
    let origin = chart.t[0];
    let times = chart.t.clone();
    let mut targets = times.clone();
    let mut weights = vec![0.0f64; n];
    let mut bycol: Vec<Vec<usize>> = vec![Vec::new(); chart.keys];
    for i in 0..n {
        bycol[chart.col[i] as usize].push(i);
    }
    let mut neighbours: Vec<(Option<usize>, Option<usize>)> = vec![(None, None); n];
    for ids in &bycol {
        for (j, &i) in ids.iter().enumerate() {
            neighbours[i] = (if j > 0 { Some(ids[j - 1]) } else { None }, ids.get(j + 1).copied());
        }
    }
    let gw = gestures::great_window(chart.od);
    for p in plans {
        let target = origin + p.time * rate * 1000.0;
        if p.ids.contains(&0) || p.ids.contains(&(n - 1)) {
            continue;
        }
        let margin = 1.0 - p.span / (2.0 * gw);
        let w = p.strength * (1.0 - p.control).powf(2.0) * smooth(margin, 0.0, 0.25);
        if w <= 0.0 || p.anchor >= 0.8 {
            continue;
        }
        let safe = p.ids.iter().all(|&i| {
            let (l, r) = neighbours[i];
            !(l.is_some_and(|l| target <= chart.end[l] + 3.0 * rate) || r.is_some_and(|r| target >= chart.t[r] - 3.0 * rate))
        });
        if safe {
            for &i in &p.ids {
                targets[i] = target;
                weights[i] = w;
            }
        }
    }
    if !weights.iter().any(|&w| w != 0.0) {
        return None;
    }
    let mut bad = vec![false; n];
    let mut any_bad = false;
    for ids in &bycol {
        for w in ids.windows(2) {
            let (a, b) = (w[0], w[1]);
            if targets[b] - targets[a] <= 3.0 * rate && (weights[a] != 0.0 || weights[b] != 0.0) {
                bad[a] = true;
                bad[b] = true;
                any_bad = true;
            }
        }
    }
    if any_bad {
        for p in plans {
            if p.ids.iter().any(|&i| bad[i]) {
                for &i in &p.ids {
                    targets[i] = times[i];
                    weights[i] = 0.0;
                }
            }
        }
    }
    if !weights.iter().any(|&w| w != 0.0) {
        return None;
    }
    let mut ordered: Vec<usize> = (0..n).collect();
    ordered.sort_by(|&a, &b| (targets[a], chart.col[a], a).partial_cmp(&(targets[b], chart.col[b], b)).unwrap());
    let mut r = Chart { keys: chart.keys, od: chart.od, title: chart.title.clone(), artist: chart.artist.clone(),
                        version: chart.version.clone(), creator: chart.creator.clone(), ..Default::default() };
    for &i in &ordered {
        r.t.push(targets[i]);
        r.end.push(if weights[i] != 0.0 { targets[i] } else { chart.end[i] });
        r.col.push(chart.col[i]);
    }
    Some((r, weights))
}

struct Basis {
    phases: Vec<Vec<f64>>,
    removable: Vec<Vec<f64>>,
    support: Vec<f64>,
    mean_support: f64,
}

fn basis(chart: &Chart, rate: f64, sv: bool, shared: &Shared, prm: &Params) -> Result<Option<Basis>, String> {
    let k = &prm.k;
    let original = shared.trace(sv).unwrap();
    let physical = shared.trace(false).unwrap_or(original);
    let mut reference_phases: Vec<Vec<Vec<f64>>> = Vec::new();
    let mut confidences: Vec<(bool, Vec<f64>)> = Vec::new();
    let plans = shared.gestures.as_ref().and_then(|g| g.plans.as_ref());
    for (center, plans) in plans.into_iter().flatten() {
        match reference(chart, rate, plans) {
            None => {
                reference_phases.push(physical.phases.clone());
                confidences.push((*center, vec![0.0; chart.len()]));
            }
            Some((r, weights)) => {
                let mut rs = Shared { trace: Some(Vec::new()), light: true, ..Default::default() };
                let rr = compute_inner(&r, rate, false, &mut rs, prm)?;
                let p = &physical.result;
                let conversion = ((rr.f("baseline_overall") / rr.f("raw_overall")) / (p.f("baseline_overall") / p.f("raw_overall")))
                    .powf(2.0 / k.STAR_B);
                let rp = &rs.trace(false).unwrap().phases;
                reference_phases.push(rp.iter().map(|row| row.iter().map(|v| v * conversion).collect()).collect());
                confidences.push((*center, weights));
            }
        }
    }
    if confidences.is_empty() || !confidences.iter().any(|(_, c)| c.iter().any(|&x| x != 0.0)) {
        return Ok(None);
    }
    let origin = chart.t[0] * 0.001 / rate;
    let fine: Vec<i64> = chart.t.iter().map(|&t| ((t * 0.001 / rate - origin) * k.PHASES as f64 / k.BIN) as i64).collect();
    let nc = confidences.len() as f64;
    let mut deltas: Vec<Vec<f64>> = Vec::new();
    let mut support_series = Vec::new();
    for (phase, full) in original.phases.iter().enumerate() {
        let plain = &physical.phases[phase];
        let nb = full.len();
        let mut delta = vec![0.0f64; nb];
        let mut support = vec![0.0f64; nb];
        for (refp, (_, weights)) in reference_phases.iter().zip(&confidences) {
            let mut totals = vec![0.0f64; nb];
            let mut count = vec![0usize; nb];
            for (i, &index) in fine.iter().enumerate() {
                let b = (nb as i64 - 1).min(0i64.max((index + phase as i64).div_euclid(k.PHASES as i64))) as usize;
                totals[b] += weights[i];
                count[b] += 1;
            }
            for b in 0..nb {
                let target = if b < refp[phase].len() { refp[phase][b] } else { plain[b] };
                delta[b] += 0f64.max(plain[b] - target) * totals[b] / 1usize.max(count[b]) as f64 / nc;
                support[b] += totals[b] / 1usize.max(count[b]) as f64 / nc;
            }
        }
        deltas.push(delta);
        if phase == 0 {
            support_series = support;
        }
    }
    let mut floors: Vec<Vec<Vec<f64>>> = Vec::new();
    for (center, actions, tracks) in original.action_runs.as_ref().unwrap() {
        let weights = confidences.iter().find(|(c, _)| c == center).map(|x| &x.1).unwrap_or(&confidences[0].1);
        let mut minimum = tracks.clone();
        for (w, a) in weights.iter().zip(actions) {
            if a.protected {
                continue;
            }
            let h = a.hand as usize;
            minimum[h][a.fb] = 0f64.max(minimum[h][a.fb] - w * a.rice);
            minimum[2][a.fb] = 0f64.max(minimum[2][a.fb] - w * a.cross);
        }
        floors.push((0..k.PHASES).map(|p| phase_total(&minimum, p, k)).collect());
    }
    for (p, row) in deltas.iter_mut().enumerate() {
        let nf = floors.len() as f64;
        let len = floors.iter().map(|f| f[p].len()).min().unwrap();
        let floor: Vec<f64> = (0..len).map(|i| py::sum(floors.iter().map(|f| f[p][i])) / nf).collect();
        let new: Vec<f64> = row.iter().zip(&original.phases[p]).zip(&floor).map(|((d, a), b)| d.min(0f64.max(a - b))).collect();
        *row = new;
    }
    let mean_support = py::sum(confidences.iter().map(|(_, c)| py::sum(c.iter().copied()))) / nc / 1usize.max(chart.len()) as f64;
    Ok(Some(Basis { phases: original.phases.clone(), removable: deltas, support: support_series, mean_support }))
}

fn evaluate(phases: &[Vec<f64>], removable: &[Vec<f64>], strength: f64, notes: usize, prm: &Params) -> (f64, [f64; 4], Vec<f64>) {
    let k = &prm.k;
    let mut levels = [0.0f64; 4];
    let mut first: Option<Vec<f64>> = None;
    for (row, credit) in phases.iter().zip(removable) {
        let mixed: Vec<f64> = row.iter().zip(credit).map(|(v, d)| 0f64.max(v - strength * d)).collect();
        let lv = horizon_levels(&mixed, k);
        for (a, b) in levels.iter_mut().zip(lv) {
            *a = a.max(b);
        }
        if first.is_none() {
            first = Some(mixed);
        }
    }
    let mut old = [0.0f64; 4];
    for row in phases {
        let lv = horizon_levels(row, k);
        for (a, b) in old.iter_mut().zip(lv) {
            *a = a.max(b);
        }
    }
    let denominator = combine(&old, notes, k);
    let factor = if denominator != 0.0 { combine(&levels, notes, k) / denominator } else { 1.0 };
    (factor, levels, first.unwrap_or_default())
}

fn rolled_apply(chart: &Chart, rate: f64, sv: bool, result: &mut Dict, shared: &mut Shared, strength: f64, prm: &Params) -> Result<(), String> {
    let k = &prm.k;
    if !(strength.is_finite() && (0.0..=1.0).contains(&strength)) {
        return Err("Rolled execution strength must be in [0,1]".into());
    }
    let Some(cand) = basis(chart, rate, sv, shared, prm)? else { return Ok(()) };
    let n = chart.len();
    let (factor, levels, total) = evaluate(&cand.phases, &cand.removable, strength, n, prm);
    if (factor - 1.0).abs() <= 1e-12 {
        return Ok(());
    }
    let source = shared.trace(sv).unwrap();
    let ratio: Vec<f64> = total.iter().zip(&source.total).map(|(v, old)| if *old != 0.0 { v / old } else { 1.0 }).collect();
    let mut series = source.series.clone();
    for &key in &source.series.keys {
        let row: Vec<f64> = source.series.get(key).iter().zip(&ratio).map(|(v, r)| v * r).collect();
        series.set(key, &row);
    }
    let rice: Vec<Sk> = [Sk::stream, Sk::delay, Sk::dump, Sk::jumpstream, Sk::handstream, Sk::chordstream].into_iter().filter(|&s| series.has(s)).collect();
    for (b, &support) in cand.support.iter().enumerate() {
        let weight = strength * support;
        let moved = rice.iter().map(|&s| series.get(s)[b]).fold(None, |m: Option<f64>, x| Some(match m { Some(m) if m >= x => m, _ => x })).unwrap_or(0.0) * weight;
        let jt = &mut series.get_mut(Sk::jumptrill)[b];
        *jt = jt.max(moved);
        for &s in &rice {
            series.get_mut(s)[b] *= 1.0 - weight;
        }
    }
    let switch: Vec<f64> = source.switch.iter().zip(&ratio).map(|(v, r)| v * r).collect();
    let calibration = result.dict("scores").unwrap().f("overall") / result.f("raw_overall");
    let scale = calibration * rate.powf(k.RATE_G);
    let raw = combine(&levels, n, k);
    let mut scores = Dict::new();
    series_dict_scores(&mut scores, &series, n, prm);
    scores.set("overall", raw);
    let mx = levels.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    scores.set("stamina", if mx > 0.0 { stamina(&total, k) * raw / mx } else { 0.0 });
    let mut archs = card::archetypes(&series, &scores, n, &switch, k);
    let tw = card::tech_where(&series, &source.parts, chart.t[0] / 1000.0, rate, k);
    for a in &mut archs {
        if let Val::Dict(d) = a {
            let r = d.f("rating");
            d.set("rating", r * scale);
        }
    }
    result.set("archetypes", Val::List(archs));
    result.set("tech_where", Val::List(tw));
    scores.map_values(|v| v * scale);
    result.set("scores", scores);
    result.set("levels", levels_dict(&levels.map(|v| v * scale), prm));
    result.set("horizon", k.HORIZONS[argmax(&levels)].0);
    let width = 1usize.max((k.HORIZONS[0].0 / k.BIN) as usize);
    let mut acc = 0.0f64;
    let mut timeline = Vec::with_capacity(total.len());
    for (i, &d) in total.iter().enumerate() {
        acc = 0f64.max(acc + d - if i >= width { total[i - width] } else { 0.0 });
        timeline.push(star(k.SCALE * (acc / width.min(i + 1) as f64).sqrt() / k.HORIZONS[0].1, k) * scale);
    }
    result.set("timeline", Val::floats(&timeline));
    let ex = result.dict_mut("execution").unwrap();
    ex.set("rolled_factor", factor);
    ex.set("rolled_support", cand.mean_support);
    if sv {
        if let Some(physical) = shared.trace(false) {
            let (pf, _, _) = evaluate(&physical.phases, &cand.removable, strength, n, prm);
            result.set("no_sv_overall", physical.result.dict("scores").unwrap().f("overall") * pf);
        }
    }
    let o = result.dict("scores").unwrap().f("overall");
    result.set("rolled_correction", o * (1.0 - 1.0 / factor));
    let b = result.f("baseline_overall") * factor;
    result.set("baseline_overall", b);
    result.set("structural_correction", o - b);
    Ok(())
}
