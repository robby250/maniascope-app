//! warmup.Model: live cold-to-warm parameters from the fit, readiness and the session clock.
use crate::personal::{Completion, WarmModels, WarmReport, CAPACITY_PRIOR, COOLING, FATIGUE_ONSET, GAP, SESSION_TAU};
use crate::predict::{Consts, HashMap, Mode, Slopes};
use std::collections::HashSet;

/// session_form's fitted prior (Personal.__init__).
#[derive(Clone, Default)]
pub struct FormPrior {
    pub global_variance: f64,
    pub skill_variance: f64,
    pub by_key: HashMap<i64, (f64, Vec<f64>)>, // keys → (global, per family)
    pub noise_variance: f64,
}

pub struct Model {
    pub modes: HashMap<i64, Mode>,
    pub fallback: Mode,
    pub history: Vec<Completion>,
    pub report: HashMap<i64, WarmReport>,
    pub slopes: Slopes,
    pub prior: FormPrior,
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2. }
}

/// The pooled curve of a keymode too sparse to identify its own, from the constant-response control (warmup.models).
fn pooled(modes: &HashMap<i64, Mode>, families: usize) -> Mode {
    if modes.is_empty() {
        return Mode { tau: 8., log_rate: true, amplitudes: vec![CAPACITY_PRIOR; families], session: 0., fatigue: 0. };
    }
    let all: Vec<&Mode> = modes.values().collect();
    Mode {
        tau: median(all.iter().map(|m| m.tau).collect()),
        log_rate: all[0].log_rate,
        amplitudes: (0..families).map(|i| median(all.iter().map(|m| m.amplitudes[i]).collect())).collect(),
        session: median(all.iter().map(|m| m.session).collect()),
        fatigue: median(all.iter().map(|m| m.fatigue).collect()),
    }
}

impl Model {
    pub fn new(fit: &WarmModels, slopes: Slopes, prior: FormPrior, families: usize) -> Model {
        Model { modes: fit.modes.clone(), fallback: pooled(&fit.control, families), history: fit.history.clone(), report: fit.report.clone(), slopes, prior }
    }

    /// No cross-keymode activation: only a mode too sparse to identify itself borrows the pooled curve.
    pub fn parameters(&self, keys: i64) -> &Mode {
        self.modes.get(&keys).unwrap_or(&self.fallback)
    }

    fn amplitude(&self, c: &Consts, keys: i64, skill: Option<&str>) -> f64 {
        let a = &self.parameters(keys).amplitudes;
        skill.map_or_else(|| a.iter().sum::<f64>() / a.len() as f64, |s| a[c.family(s)])
    }

    /// Fraction of capacity a keymode has `minutes` into readiness.
    pub fn capacity(&self, c: &Consts, keys: i64, skill: Option<&str>, minutes: f64) -> f64 {
        let p = self.parameters(keys);
        let amplitude = self.amplitude(c, keys, skill) / if p.log_rate { 1. } else { 3. }; // legacy diagnostic models
        (-amplitude * (-minutes.max(0.) / p.tau).exp()).exp()
    }

    /// The session-long warm-up over a map starting `clock` minutes into the session, before any rate slope.
    pub fn session_cold(&self, keys: i64, clock: f64, seconds: f64) -> f64 {
        let d = (seconds / 60.).clamp(0., 10.);
        let mean = over(d, SESSION_TAU);
        self.parameters(keys).session * (-clock.max(0.) / SESSION_TAU).exp() * mean + self.fatigue(keys, clock)
    }

    pub fn fatigue(&self, keys: i64, clock: f64) -> f64 {
        self.parameters(keys).fatigue * (clock - FATIGUE_ONSET).max(0.) / 60.
    }

    /// A family's readiness loss alone: session_cold depends on the clock, so a table of this can't hold it.
    pub fn family_cold(&self, keys: i64, family: usize, minutes: f64) -> f64 {
        let p = self.parameters(keys);
        p.amplitudes[family] * (-minutes.max(0.) / p.tau).exp()
    }

    pub fn penalty(&self, c: &Consts, keys: i64, skill: Option<&str>, minutes: f64, seconds: f64, slope: Option<f64>, clock: f64) -> f64 {
        let p = self.parameters(keys);
        let d = (seconds / 60.).clamp(0., 10.);
        let loss = self.amplitude(c, keys, skill) * (-minutes.max(0.) / p.tau).exp() * over(d, p.tau) + self.session_cold(keys, clock, seconds);
        if p.log_rate { loss * slope.unwrap_or(3.) } else { loss }
    }
}

/// Mean of exp(−x/τ) over a map of `d` minutes.
pub fn over(d: f64, tau: f64) -> f64 {
    if d > 0.001 { -(-d / tau).exp_m1() * tau / d } else { 1. }
}

/// One finished map of warmup.activation's history: a fit completion or a live attempt.
#[derive(Clone)]
pub struct Done {
    pub end: f64,
    pub begun: f64,
    pub seconds: f64,
    pub keys: Option<i64>,
    pub family: usize,                  // dominant(features), else family(skill)
    pub identity: String,               // key, else (beatmap, round(end, 1))
}

fn distinct<'a>(history: &'a [Done], now: f64) -> Vec<&'a Done> {
    let mut sorted: Vec<&Done> = history.iter().filter(|a| a.end <= now).collect();
    sorted.sort_by(|a, b| a.end.total_cmp(&b.end));
    let mut seen = HashSet::new();
    sorted.retain(|a| seen.insert(a.identity.as_str()));
    sorted
}

/// Same elapsed-time accounting as the fit's timeline; quits never enter. `transfer`: [target][played] between
/// families; None = any family counts in full.
pub fn activation(history: &[Done], now: f64, keys: i64, target: Option<usize>, transfer: &[Vec<f64>]) -> f64 {
    let (mut state, mut last_end) = (0., None::<f64>);
    for a in distinct(history, now) {
        match last_end {
            Some(last) if a.begun - last <= GAP => state *= (-(a.end - last).max(0.) / COOLING).exp(),
            _ => state = 0.,
        }
        if a.keys == Some(keys) {
            state += (a.seconds / 60.).min(10.) * target.map_or(1., |t| transfer[t][a.family]);
        }
        last_end = Some(a.end);
    }
    match last_end {
        Some(last) if now - last <= GAP => state * (-(now - last).max(0.) / COOLING).exp(),
        _ => 0.,
    }
}

/// Minutes since the current session's first completed map began, any keymode; 0 when no session is running.
pub fn session_minutes(history: &[Done], now: f64) -> f64 {
    let (mut start, mut last_end) = (0., None::<f64>);
    for a in distinct(history, now) {
        if last_end.is_none_or(|last| a.begun - last > GAP) {
            start = a.begun;
        }
        last_end = Some(a.end);
    }
    match last_end {
        Some(last) if now - last <= GAP => (now - start).max(0.) / 60.,
        _ => 0.,
    }
}
