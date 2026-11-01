//! recommend.Session: transient state rebuilt from the event log (attempts since the last gap or reset, their
//! residuals, exposures), session_form's posterior, and the frozen tables the predictor reads.
use crate::fit::solve;
use crate::personal::{Completion, SESSION_TAU};
use crate::predict::{ChartSession, Consts, Feats, HashMap, KeyState, PredSession};
use crate::warmup::{activation, session_minutes, Done, Model};
use ms_data::store::Event;
use serde_json::{Map, Value};
use std::collections::HashSet;
use std::sync::Arc;

pub const SESSION_GAP: f64 = 3000.;
pub const HISTORY_S: f64 = 3. * 86400.;
pub const EXPOSURE_S: f64 = 14. * 86400.;
pub const FORM_NOTICE: f64 = 0.08; // log error: about .4 point at 95 %, under it the readout says nothing
const PESSIMISM: f64 = 0.25;
const PUSH_DOWN: f64 = 2.;
const WARMUP_MAX: usize = 3;
const FAMILIES_DIM: usize = 8;

fn truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64() != Some(0.),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
    }
}

/// A merged start + finish/abort record.
pub struct Attempt {
    pub a: Map<String, Value>,
    pub t: f64,
    pub end: f64,
    pub finished: bool,
    pub beatmap: Option<String>,
    pub start_id: Option<i64>,
    pub meaningful: bool,
    pub ability_evidence: bool,
    pub beat: bool,
    pub z: Option<f64>,
    pub weights: Vec<f64>, // session_form.loading per family, sums to 1
    pub family: usize,     // dominant(features) else family(skill)
}

impl Attempt {
    pub fn num(&self, k: &str) -> Option<f64> {
        self.a.get(k).and_then(Value::as_f64)
    }

    pub fn text(&self, k: &str) -> Option<&str> {
        self.a.get(k).and_then(Value::as_str)
    }

    pub fn keys(&self) -> Option<i64> {
        self.num("keys").map(|k| k as i64)
    }

    fn evidence(&self) -> bool {
        self.meaningful && self.ability_evidence
    }

    /// Residual against the frozen pre-session base; None without a forecast or a measured result.
    fn error(&self) -> Option<f64> {
        let (y, mu) = (self.num("y")?, self.num("mu")?);
        Some(y - self.num("base_mu").unwrap_or(mu) - self.num("cold_penalty").unwrap_or(0.))
    }
}

pub struct Session {
    pub now: f64,
    pub events: Vec<Arc<Event>>,    // current session
    pub all: Vec<Arc<Event>>,       // HISTORY_S
    pub attempts: Vec<Arc<Attempt>>, // current-session form only
    pub seen_attempts: Vec<Arc<Attempt>>, // every attempt; Session.history_attempts is the last HISTORY_S of these
    pub warm_flag: bool,
    pub warmup: Option<Arc<Model>>,
    pub consts: Arc<Consts>,
    transfer: Arc<Vec<Vec<f64>>>,
    events_by_map: HashMap<String, Vec<Arc<Event>>>,
    attempts_by_map: HashMap<String, Vec<Arc<Attempt>>>,
    offer_times: HashMap<Option<String>, Vec<f64>>,
    pub played: HashSet<String>,
    pub retry_played: HashSet<String>,
    current: HashSet<i64>, // ids of `events`
}

/// Offers and starts of a session's window in time order (Session.__init__'s scan): the index the session begins at.
fn session_start(ev: &[Arc<Event>], now: f64) -> usize {
    let mut start = 0;
    let (mut last_play, mut active_until, mut active_key, mut active_id) = (None::<f64>, None::<f64>, None::<Option<String>>, None::<i64>);
    for (i, e) in ev.iter().enumerate() {
        match e.kind.as_str() {
            "reset" => {
                start = i + 1;
                (last_play, active_until, active_key, active_id) = (None, None, None, None);
            }
            "start" | "finish" | "abort" => {
                let id_of = e.info.get("start_id").and_then(Value::as_i64).or(active_id);
                let terminal = e.kind != "start" && active_key.as_ref() == Some(&e.beatmap) && id_of == active_id;
                let continuing = active_until.is_some_and(|until| terminal || e.t <= until);
                if last_play.is_some_and(|last| e.t - last > SESSION_GAP) && !continuing {
                    start = i;
                }
                last_play = Some(e.t);
                if e.kind == "start" {
                    let length = e.info.get("length").filter(|v| truthy(Some(v))).and_then(Value::as_f64).unwrap_or(900.);
                    active_until = Some(e.t + length + SESSION_GAP);
                    active_key = Some(e.beatmap.clone());
                    active_id = Some(e.id);
                } else {
                    active_until = None;
                }
            }
            "void" => active_until = None,
            _ => {}
        }
    }
    if last_play.is_some_and(|last| now - last > SESSION_GAP) && active_until.is_none_or(|until| now > until) {
        start = ev.len();
    }
    start
}

impl Session {
    pub fn new(events: Vec<Arc<Event>>, now: f64, warmup: Option<Arc<Model>>, consts: Arc<Consts>, transfer: Arc<Vec<Vec<f64>>>) -> Session {
        let mut seen: Vec<Arc<Event>> = events.into_iter().filter(|e| now - EXPOSURE_S <= e.t && e.t <= now).collect();
        seen.sort_by(|a, b| a.t.total_cmp(&b.t));
        let ev: Vec<Arc<Event>> = seen.iter().filter(|e| e.t >= now - HISTORY_S).cloned().collect();
        let start = session_start(&ev, now);
        let current_events = ev[start..].to_vec();
        let by_start = |kind: &str| -> HashMap<i64, Value> {
            ev.iter().filter(|e| e.kind == kind).filter_map(|e| Some((e.info.get("start_id")?.as_i64()?, e.info.clone()))).collect()
        };
        let notes = by_start("attempt_note");
        let results = by_start("pp_result");
        let session_begin = current_events.first().map_or(f64::INFINITY, |e| e.t);
        let (mut attempts, mut seen_attempts) = (Vec::new(), Vec::new());
        let mut pending: Option<(Map<String, Value>, f64, Option<String>, i64)> = None;
        for e in &seen {
            let empty = Map::new();
            let info = e.info.as_object().unwrap_or(&empty);
            match e.kind.as_str() {
                "start" => pending = Some((info.clone(), e.t, e.beatmap.clone(), e.id)),
                "void" | "reset" => pending = None,
                "finish" | "abort" => {
                    let matched = pending.as_ref().is_some_and(|p| e.beatmap == p.2 && info.get("start_id").is_none_or(|s| s.as_i64() == Some(p.3)));
                    if !matched {
                        continue;
                    }
                    let (start_info, t, beatmap, start_id) = pending.take().unwrap();
                    let finish = e.kind == "finish";
                    let mut a = start_info.clone();
                    a.extend(info.iter().map(|(k, v)| (k.clone(), v.clone())));
                    let length = a.get("length").and_then(Value::as_f64).unwrap_or(0.);
                    let seconds = info.get("played_seconds").map_or_else(
                        || info.get("progress").and_then(Value::as_f64).unwrap_or(0.) * length,
                        |v| v.as_f64().unwrap_or(0.));
                    let meaningful = finish || seconds >= 20f64.min(0.3 * if length != 0. { length } else { 120. });
                    let note = notes.get(&start_id);
                    let reason = note.and_then(|n| n.get("reason").cloned()).or_else(|| info.get("reason").cloned()).unwrap_or(Value::Null);
                    let ability_evidence = finish && !matches!(reason.as_str(), Some("interrupted" | "experiment" | "bored"));
                    let beat = results.get(&start_id).is_some_and(|r| {
                        a.get("mode").and_then(Value::as_str) == Some("pp") && a.get("purpose").and_then(Value::as_str) != Some("warmup")
                            && r["after"].as_f64().unwrap_or(0.) > r["before"].as_f64().unwrap_or(0.) + 0.05
                    });
                    let start_num = |k: &str| start_info.get(k).and_then(Value::as_f64);
                    let z = (|| {
                        let y = info.get("y").and_then(Value::as_f64)?;
                        let (mu, sd) = (start_num("mu")?, start_num("sd").filter(|&s| s != 0.)?);
                        (finish && info.get("prediction_valid").is_none_or(|v| truthy(Some(v)))).then(|| {
                            (start_num("shown_mu").unwrap_or(mu + PESSIMISM * start_num("sdm").unwrap_or(0.)) - y) / sd
                        })
                    })();
                    let mut mix = None;
                    if let Some(sk) = a.get("features").filter(|f| !f.is_null()).and_then(|f| f.get("sk")) {
                        mix = Some(Feats::skills(sk, &consts));
                    }
                    let family_of_skill = consts.family(a.get("skill").and_then(Value::as_str).unwrap_or(""));
                    let (family, weights) = mix.map_or_else(|| {
                        let mut w = vec![0.; FAMILIES_DIM];
                        w[family_of_skill] = 1.;
                        (family_of_skill, w)
                    }, |f| (consts.dominant(&f), consts.family_weights(&f)));
                    a.remove("features"); // the full analysis of every start: only sk is read, above
                    a.insert("reason".into(), reason);
                    let end = e.t;
                    let attempt = Arc::new(Attempt { a, t, end, finished: finish, beatmap: beatmap.clone(), start_id: Some(start_id), meaningful,
                                                     ability_evidence, beat, z, weights, family });
                    seen_attempts.push(attempt.clone());
                    if t >= session_begin {
                        attempts.push(attempt);
                    }
                }
                _ => {}
            }
        }
        let warm_flag = current_events.iter().any(|e| e.kind == "skip_warmup");
        let mut events_by_map: HashMap<String, Vec<Arc<Event>>> = HashMap::default();
        let mut offer_times: HashMap<Option<String>, Vec<f64>> = HashMap::default();
        let name = |b: &Option<String>| b.clone().unwrap_or_else(|| "None".to_string());
        for e in &seen {
            events_by_map.entry(name(&e.beatmap)).or_default().push(e.clone());
            if e.kind == "offer" {
                offer_times.entry(e.info.get("mode").and_then(Value::as_str).map(str::to_string)).or_default().push(e.t);
            }
        }
        let mut attempts_by_map: HashMap<String, Vec<Arc<Attempt>>> = HashMap::default();
        for a in &seen_attempts {
            attempts_by_map.entry(name(&a.beatmap)).or_default().push(a.clone());
        }
        let evidence = || attempts.iter().filter(|a: &&Arc<Attempt>| a.evidence());
        let played = evidence().filter_map(|a| a.beatmap.clone()).collect();
        // Shuffled effort still counts as played, but cannot establish normal chart familiarity.
        let shuffled = |a: &Attempt| ["mods_list", "mods"].iter().any(|name| a.a.get(*name).and_then(Value::as_array).is_some_and(|mods| {
            mods.iter().any(|m| m.get("acronym").or(Some(m)).and_then(Value::as_str).is_some_and(|x| x.eq_ignore_ascii_case("RD")))
        }));
        let retry_played = evidence().filter(|a| !shuffled(a)).filter_map(|a| a.beatmap.clone()).collect();
        let current = current_events.iter().map(|e| e.id).collect();
        Session { now, events: current_events, all: ev, attempts, seen_attempts, warm_flag, warmup, consts, transfer, events_by_map,
                  attempts_by_map, offer_times, played, retry_played, current }
    }

    /// accuracy_targets.session_view: shared completed effort, each target consuming its own frozen residuals. Legacy
    /// predictions still establish warmth; they are not read as directly fitted display forecasts.
    pub fn display_view(&self, warmup: Arc<Model>) -> Session {
        let events = self.all.iter().map(|e| {
            if e.kind != "start" && e.kind != "finish" {
                return e.clone();
            }
            let mut info = e.info.clone();
            if let Some(o) = info.as_object_mut() {
                if e.kind == "start" {
                    for field in ["mu", "base_mu", "cold_penalty", "sdm", "sda", "sd"] {
                        o.insert(field.to_string(), o.get(&format!("display_{field}")).cloned().unwrap_or(Value::Null));
                    }
                    let mu = o.get("mu").cloned().unwrap_or(Value::Null);
                    o.insert("shown_mu".to_string(), mu);
                } else if e.kind == "finish" {
                    let y = o.get("y_lazer").cloned().unwrap_or(Value::Null);
                    o.insert("y".to_string(), y);
                }
            }
            Arc::new(Event { id: e.id, t: e.t, kind: e.kind.clone(), beatmap: e.beatmap.clone(), info })
        }).collect();
        let mut view = Session::new(events, self.now, Some(warmup), self.consts.clone(), self.transfer.clone());
        view.warm_flag = self.warm_flag;
        view
    }

    /// A click changes freshness, not ability. Clock decay gets a cheap 30 s bucket.
    pub fn model_key(&self) -> (i64, i64, bool, bool) {
        let last = self.events.iter().filter(|e| matches!(e.kind.as_str(), "finish" | "abort" | "reset" | "skip_warmup" | "attempt_note" | "pp_result"))
            .map(|e| e.id).max().unwrap_or(0);
        (last, (self.now / 30.) as i64, !self.events.is_empty(), self.warm_flag)
    }

    /// The fit's completions and this session's, an explicit reset overriding restored prior gameplay.
    fn history(&self, model: &Model) -> Vec<Done> {
        let of_completion = |a: &Completion| Done { end: a.end, begun: a.t, seconds: a.played_seconds.clamp(0., 1800.), keys: Some(a.keys), family: a.family,
                                                    identity: a.key.clone().unwrap_or_else(|| format!("md5:{}|{:.1}", a.chart, a.end)) };
        let of_attempt = |a: &Arc<Attempt>| -> Option<Done> {
            if !a.finished || !a.ability_evidence {
                return None;
            }
            let seconds = a.num("played_seconds").filter(|&s| s != 0.).or_else(|| a.num("length")).unwrap_or(0.).clamp(0., 1800.);
            let identity = a.text("key").map(str::to_string).filter(|k| !k.is_empty())
                .unwrap_or_else(|| format!("{}|{:.1}", a.beatmap.as_deref().unwrap_or(""), a.end));
            Some(Done { end: a.end, begun: a.t, seconds, keys: a.keys(), family: a.family, identity })
        };
        let reset = self.all.iter().filter(|e| e.kind == "reset").map(|e| e.t).fold(f64::NEG_INFINITY, f64::max);
        model.history.iter().map(of_completion).chain(self.attempts.iter().filter_map(of_attempt)).filter(|d| d.end > reset).collect()
    }

    /// session_form's posterior read for a skill (None: rice, as its loading): shared form + the family's.
    pub fn correction(&self, keys: i64, skill: Option<&str>) -> f64 {
        let p = self.posterior(keys);
        p[0] + p[1 + self.consts.family(skill.unwrap_or("rice"))]
    }

    /// Relevant minutes in this keymode. Other keymodes cannot erase cold readiness.
    pub fn activation(&self, keys: i64, family: Option<usize>) -> f64 {
        if self.warm_flag {
            return 10.;
        }
        self.warmup.as_ref().map_or(0., |m| activation(&self.history(m), self.now, keys, family, &self.transfer))
    }

    /// Minutes into the session, any keymode.
    pub fn session_clock(&self) -> f64 {
        self.warmup.as_ref().map_or(0., |m| session_minutes(&self.history(m), self.now))
    }

    /// session_form.posterior: robust shared and family-local session condition. One unfamiliar chart can be a model
    /// error; several independently underperformed charts can establish poor form.
    pub fn posterior(&self, keys: i64) -> Vec<f64> {
        let dimension = FAMILIES_DIM + 1;
        let Some(model) = &self.warmup else { return vec![0.; dimension] };
        let prior = &model.prior;
        let mut variances = vec![prior.global_variance];
        match prior.by_key.get(&keys) {
            Some((g, skills)) => {
                variances[0] = *g;
                variances.extend_from_slice(skills);
            }
            None => variances.extend(std::iter::repeat_n(prior.skill_variance, dimension - 1)),
        }
        let mut order: Vec<String> = Vec::new();
        let mut blocks: HashMap<String, Vec<(Vec<f64>, f64, f64, f64)>> = HashMap::default();
        for a in &self.attempts {
            if a.keys() != Some(keys) || !a.evidence() {
                continue;
            }
            let Some(error) = a.error() else { continue };
            let sd = a.num("sd").filter(|&s| s != 0.).unwrap_or(prior.noise_variance.sqrt());
            let age = (-(self.now - a.end).max(0.) / 1800.).exp();
            let mut loading = vec![1.];
            let total: f64 = a.weights.iter().sum();
            loading.extend(a.weights.iter().map(|w| if total > 0. { w / total } else { 1. / FAMILIES_DIM as f64 }));
            let id = a.beatmap.clone().or_else(|| a.text("key").map(str::to_string)).unwrap_or_default();
            if !blocks.contains_key(&id) {
                order.push(id.clone());
            }
            blocks.entry(id).or_default().push((loading, error, (sd * sd).max(1e-4), age));
        }
        let (mut xs, mut ys, mut vs, mut ages) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for id in &order {
            let observations = &blocks[id];
            let mass: f64 = observations.iter().map(|o| o.3).sum();
            if mass <= 0. {
                continue;
            }
            let mut x = vec![0.; dimension];
            observations.iter().for_each(|o| x.iter_mut().zip(&o.0).for_each(|(s, v)| *s += v * o.3 / mass));
            xs.push(x);
            ys.push(observations.iter().map(|o| o.1 * o.3).sum::<f64>() / mass);
            // Retries reduce timing noise, not uncertainty about that chart itself.
            vs.push(observations.iter().map(|o| o.2 * o.3).sum::<f64>() / mass);
            ages.push(observations.iter().map(|o| o.3).fold(f64::NEG_INFINITY, f64::max));
        }
        let mut mean = vec![0.; dimension];
        if xs.is_empty() {
            return mean;
        }
        let base: Vec<f64> = ages.iter().zip(&vs).map(|(a, v)| a / v).collect();
        let mut robust = vec![1.; ys.len()];
        // Unknown session/skill state can legitimately explain a large departure: outliers are judged against
        // the full innovation, or all the evidence of an unmeasured weak skill would read as "unlikely" mistakes.
        let innovation: Vec<f64> = xs.iter().zip(&vs).map(|(x, v)| v + x.iter().zip(&variances).map(|(a, s)| a * a * s).sum::<f64>()).collect();
        for _ in 0..5 {
            let mut gram = vec![0.; dimension * dimension];
            let mut right = vec![0.; dimension];
            (0..dimension).for_each(|i| gram[i * dimension + i] = 1. / variances[i].max(1e-6));
            for (r, x) in xs.iter().enumerate() {
                let w = base[r] * robust[r];
                for i in 0..dimension {
                    right[i] += x[i] * w * ys[r];
                    for j in 0..dimension {
                        gram[i * dimension + j] += x[i] * w * x[j];
                    }
                }
            }
            mean = solve(&gram, &right);
            for (r, x) in xs.iter().enumerate() {
                let fitted: f64 = x.iter().zip(&mean).map(|(a, b)| a * b).sum();
                robust[r] = (5. / (4. + (ys[r] - fitted).powi(2) / innovation[r])).min(1.);
            }
        }
        mean
    }

    /// The tables the predictor reads, per keymode, plus everything played charts add.
    pub fn tables(&self, keys: &[i64]) -> PredSession {
        let by_keys = keys.iter().map(|&k| (k, self.key_state(k))).collect();
        let mut charts: HashMap<String, ChartSession> = HashMap::default();
        let bare = |b: &str| b.strip_prefix("md5:").or_else(|| b.strip_prefix("sha:")).map(str::to_string);
        let active: HashSet<i64> = self.attempts.iter().filter_map(|a| a.start_id).collect();
        for (map, attempts) in &self.attempts_by_map {
            let Some(chart) = bare(map) else { continue };
            let entry = charts.entry(chart).or_default();
            let mut finished: Vec<&Arc<Attempt>> = attempts.iter().filter(|a| a.finished).collect();
            finished.sort_by(|a, b| a.end.total_cmp(&b.end));
            for a in finished {
                let y = if a.a.contains_key("y_lazer") { a.num("y_lazer") } else { a.num("y") };
                if let (Some(rate), Some(y)) = (a.num("rate").filter(|&r| r != 0.), y) {
                    let rate = ms_calc::py::round_digits(rate, 2);
                    let acc = 1. - y.exp();
                    match entry.last.iter_mut().find(|x| x.0 == rate) {
                        Some(slot) => slot.1 = acc,
                        None => entry.last.push((rate, acc)),
                    }
                }
            }
            for a in attempts {
                let rate = a.num("rate").filter(|&r| r != 0.);
                if a.start_id.is_some_and(|s| active.contains(&s)) && a.ability_evidence {
                    if let (Some(y), Some(base), Some(rate)) = (a.num("y"), a.num("base_mu"), rate) {
                        let age = (-(self.now - a.end).max(0.) / 3600.).exp();
                        entry.attempts.push((rate, age, y - base - a.num("cold_penalty").unwrap_or(0.), a.weights.clone()));
                    }
                }
            }
        }
        // A substantially played downrate constrains faster proposals of that exact chart: censored preference and
        // ability evidence, not a fabricated score at the unplayed rate.
        for a in self.attempts.iter().filter(|a| a.evidence()) {
            let (Some(actual), Some(offered), Some(map)) = (a.num("rate").filter(|&r| r != 0.), a.num("offered_rate").filter(|&r| r != 0.), a.beatmap.as_deref()) else { continue };
            if actual < offered - 0.025 {
                if let Some(chart) = bare(map) {
                    charts.entry(chart).or_default().downrates.push((actual, (-(self.now - a.end).max(0.) / 3600.).exp()));
                }
            }
        }
        for map in &self.retry_played {
            if let Some(chart) = bare(map) {
                charts.entry(chart).or_default().retried = true;
            }
        }
        PredSession { warm: self.warm_flag, by_keys, charts }
    }

    fn key_state(&self, k: i64) -> KeyState {
        let n = self.consts.families.len();
        let activation: Vec<f64> = (0..n).map(|f| self.activation(k, Some(f))).collect();
        let clock = self.session_clock();
        let (mut penalty, mut session, mut fatigue) = (vec![0.; n], 0., 0.);
        if let Some(m) = &self.warmup {
            penalty = (0..n).map(|f| m.family_cold(k, f, activation[f])).collect();
            fatigue = m.fatigue(k, clock);
            session = m.session_cold(k, clock, 0.) - fatigue;
        }
        KeyState { form: self.posterior(k), penalty, session, session_tau: SESSION_TAU, fatigue, activation }
    }

    /// What moves results right now: (reason, log-error shift, positive = worse).
    pub fn form_state(&self, keys: i64) -> (Option<&'static str>, f64) {
        let Some(m) = self.warmup.as_ref().filter(|_| !self.warm_flag) else { return (None, 0.) };
        let p = m.parameters(keys);
        let clock = self.session_clock();
        let unit = if p.log_rate { 3. } else { 1. };
        let fatigue = m.fatigue(keys, clock) * unit;
        // warmup_penalty(keys, None): the mean family amplitude at the keymode's whole readiness.
        let cold = m.penalty(&self.consts, keys, None, self.activation(keys, None), 0., Some(1.), clock) * if p.log_rate { 3. } else { 1. } - fatigue;
        let form = self.correction(keys, None);
        let reason = if cold > FORM_NOTICE {
            Some("warming up")
        } else if fatigue > FORM_NOTICE / 2. {
            Some("fading")
        } else if form < -FORM_NOTICE {
            Some("on form")
        } else if form > FORM_NOTICE {
            Some("off form")
        } else {
            None
        };
        (reason, cold + fatigue + form)
    }

    fn zs(&self) -> Vec<(&Arc<Attempt>, f64)> {
        self.attempts.iter().filter(|a| a.ability_evidence).filter_map(|a| Some((a, a.z?))).collect()
    }

    /// Staircase position this session in steps, ≥ 0: target only; ability itself is tracked by the posterior.
    pub fn push(&self, keys: Option<i64>) -> f64 {
        let mut push: f64 = 0.;
        for a in &self.attempts {
            if !a.evidence() || (keys.is_some() && a.keys() != keys) {
                continue;
            }
            let z = a.z.unwrap_or(0.);
            let offered = a.text("mode") == Some("pp") && a.text("purpose") != Some("warmup");
            if a.beat {
                push += 1.;
            } else if z <= -1. || (offered && z < 0.) {
                push = (push - PUSH_DOWN).max(0.);
            }
        }
        push
    }

    pub fn current_keys(&self) -> Option<i64> {
        self.attempts.iter().rev().find_map(|a| a.keys().filter(|&k| k != 0))
    }

    pub fn phase(&self) -> &'static str {
        let zs = self.zs();
        let n = self.attempts.iter().filter(|a| a.finished).count();
        let warmups = self.attempts.iter().filter(|a| a.text("purpose") == Some("warmup") && a.finished).count();
        let minutes: f64 = zs.iter().filter(|x| x.1 > -1.)
            .map(|(a, _)| (a.end - a.t).max(0.).min(a.num("length").filter(|&l| l != 0.).unwrap_or(900.)) / 60.).sum();
        let strong = zs.iter().any(|(a, z)| *z >= 0.5 && a.text("purpose") != Some("warmup"));
        if !(self.warm_flag || strong || n >= 3 || warmups >= WARMUP_MAX || minutes >= 8.) {
            return "warmup";
        }
        if self.current_keys().is_some_and(|k| self.form_state(k).0 == Some("fading")) {
            return "fading";
        }
        let recent: Vec<f64> = zs[zs.len().saturating_sub(2)..].iter().map(|x| x.1).collect();
        if zs.len() >= 2 && recent.iter().sum::<f64>() / (recent.len() as f64) < -1. {
            return "recover";
        }
        if recent.last().is_some_and(|&z| z < -1.8) {
            return "recover";
        }
        if self.push(None) > 0. { "push" } else { "build" }
    }

    /// 1 = fresh. A restart or reset affects form, never recent-play avoidance.
    ///
    /// pool: the playlist's effective size. An offered or played chart then recovers by how many other maps came
    /// since, (n/(n + pool/2))²: a soft shuffle bag — a quarter back after half the pool, 44 % after all of it, so a
    /// favourite can return before the pool is exhausted but never right away. Not a hard bag ("not guaranteed … but
    /// it shouldn't come back very soon", user 2026-10-01), and not a clock — a 40-minute recovery brought charts back
    /// within one session, an 18-hour one brought PP maps back every day (G835LX 2026-10-08: Psychoboost offered 7×,
    /// repeats 1/2/3/4/5 days apart). window: exposures older than this are forgotten (the PP tab's repeat cooldown).
    pub fn freshness(&self, map: &str, length_s: f64, mode: Option<&str>, pool: Option<f64>, window: f64) -> f64 {
        let (mut f, now) = (1f64, self.now);
        let att: Vec<&Arc<Attempt>> = self.attempts_by_map.get(map).map_or(&[][..], Vec::as_slice).iter()
            .filter(|a| a.meaningful && a.t >= now - window).collect();
        let mut last_offer = None;
        for e in self.events_by_map.get(map).map_or(&[][..], Vec::as_slice) {
            let dt = now - e.t;
            if dt > window {
                continue;
            }
            match e.kind.as_str() {
                // One offer may also log next/open: its latest timestamp, not three accidental votes.
                "offer" | "next" | "open" => last_offer = Some(e.t),
                "skip" => {
                    if mode.is_some_and(|m| e.info.get("mode").and_then(Value::as_str).unwrap_or(m) != m) {
                        continue;
                    }
                    f *= if self.current.contains(&e.id) { 0.1 } else { 1. - 0.9 * (-dt / 43200.).exp() };
                }
                _ => {}
            }
        }
        let last = att.last();
        let since = last.map(|last| {
            self.seen_attempts.iter().filter(|a| a.t > last.t && a.meaningful && a.beatmap.as_deref() != Some(map))
                .filter_map(|a| a.beatmap.as_deref()).collect::<HashSet<&str>>().len() as f64
        });
        let pool = pool.filter(|&p| p != 0.);
        if let (Some(pool), true) = (pool, last_offer.is_some() || !att.is_empty()) {
            // One bag over the latest exposure: the fewer of offers since its offer, maps played since its play.
            let times = self.offer_times.get(&mode.map(str::to_string));
            let mut n = f64::INFINITY;
            if let Some(offer) = last_offer {
                n = n.min(times.map_or(0, |t| t.len() - t.partition_point(|x| *x <= offer)) as f64);
            }
            if let Some(s) = since {
                n = n.min(s);
            }
            f *= (n / (n + 0.5 * pool)).powi(2);
        } else if let Some(offer) = last_offer {
            f *= 1. - 0.98 * (-(now - offer).max(0.) / 2400.).exp();
        }
        if let Some(last) = last {
            if pool.is_none() {
                let need = 10f64.min(5. + (length_s / 180.).ceil());
                let recovery = 1. - 0.97 * (-(now - last.end).max(0.) / (18. * 3600.)).exp();
                f *= recovery.max((since.unwrap_or(0.) / need).min(1.) * 0.6);
            }
            if att.len() >= 3 {
                f *= 0.5f64.powi(att.len() as i32 - 2); // boredom, even for short maps
            }
        }
        f.max(1e-6)
    }

    /// Identities with session events (the others are fresh).
    pub fn eventful(&self, map: &str) -> bool {
        self.events_by_map.contains_key(map)
    }

    /// Worker._recent_web's offer events in time order.
    pub fn offers(&self, mode: &str) -> impl Iterator<Item = &Arc<Event>> {
        let mode = mode.to_string();
        self.all.iter().filter(move |e| e.kind == "offer" && e.info.get("mode").and_then(Value::as_str) == Some(mode.as_str()))
    }
}
