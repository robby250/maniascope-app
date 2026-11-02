//! The accuracy predictor: base_of, chart affinity, shape, warm-up and session form for one or many (chart, rate).
use crate::choices::{Choice, Choices, Proposal};
use crate::personal::Fit;
use crate::warmup::Model;
use ms_calc::background::background_map;
use ms_calc::val::{Dict, Val};
use serde_json::Value;
use std::collections::HashMap as StdMap;
use std::hash::BuildHasherDefault;
use std::sync::Arc;

/// FxHash (rustc's): SipHash was a fifth of a prediction; these maps are built from our own export.
#[derive(Default, Clone, Copy)]
pub struct Fx(u64);

impl std::hash::Hasher for Fx {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for c in bytes.chunks(8) {
            let mut w = [0u8; 8];
            w[..c.len()].copy_from_slice(c);
            self.write_u64(u64::from_le_bytes(w));
        }
    }
    fn write_u64(&mut self, i: u64) {
        self.0 = (self.0.rotate_left(5) ^ i).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
    fn write_i64(&mut self, i: i64) {
        self.write_u64(i as u64)
    }
    fn write_usize(&mut self, i: usize) {
        self.write_u64(i as u64)
    }
    fn write_u8(&mut self, i: u8) {
        self.write_u64(i as u64)
    }
}

pub type HashMap<K, V> = StdMap<K, V, BuildHasherDefault<Fx>>;

pub type Res<T> = Result<T, String>;

pub(crate) fn err(m: &str) -> String {
    m.to_string()
}

/// {int keys: value} with the Python `.get(k, d[0])` fallback kept apart.
#[derive(Clone, Default)]
pub struct ByKeys<T> {
    pub map: HashMap<i64, T>,
}

impl<T> ByKeys<T> {
    pub fn get(&self, k: i64) -> Option<&T> {
        self.map.get(&k)
    }
}

impl<T> FromIterator<(i64, T)> for ByKeys<T> {
    fn from_iter<I: IntoIterator<Item = (i64, T)>>(i: I) -> Self {
        ByKeys { map: i.into_iter().collect() }
    }
}

/// recommend._interp: np.interp for one scalar on a short sorted list.
pub fn interp(x: f64, xs: &[f64], ys: &[f64]) -> f64 {
    let i = xs.partition_point(|&v| v <= x);
    if i == 0 {
        return ys[0];
    }
    if i == xs.len() {
        return ys[ys.len() - 1];
    }
    let (x0, x1) = (xs[i - 1], xs[i]);
    if x1 > x0 {
        ys[i - 1] + (ys[i] - ys[i - 1]) * (x - x0) / (x1 - x0)
    } else {
        ys[i]
    }
}

#[derive(Clone, Default)]
pub struct Pop {
    pub slope: ByKeys<f64>,
    pub window: ByKeys<f64>,
    pub curve: ByKeys<f64>,
    pub knee: f64,
    pub mb: HashMap<i64, (f64, f64)>,
    pub mbr: HashMap<i64, Vec<(f64, f64, f64)>>, // b → [(log rate, effect, n)] in VARIANTS order
    pub tau2: f64,
    pub kappa: f64,
    pub gap: ByKeys<Vec<(usize, Vec<f64>)>>, // keys → [(GAP_SHARE_PARTS index, coefficients)]
    pub share_mean: ByKeys<Vec<f64>>,         // keys → mean share per part
}

impl Pop {
    pub fn slope(&self, k: i64) -> f64 {
        *self.slope.get(k).or(self.slope.get(0)).unwrap_or(&0.)
    }
    pub fn curve(&self, k: i64) -> f64 {
        *self.curve.get(k).or(self.curve.get(0)).unwrap_or(&0.)
    }
}

/// warmup.Model parameters of one keymode.
#[derive(Clone, Default)]
pub struct Mode {
    pub tau: f64,
    pub log_rate: bool,
    pub amplitudes: Vec<f64>,
    pub session: f64,
    pub fatigue: f64,
}

/// warmup.score_rate_slope's inputs: the population's slope/curve/knee and the learned rate elasticities.
#[derive(Clone, Default)]
pub struct Slopes {
    pub slope: ByKeys<f64>,
    pub curve: ByKeys<f64>,
    pub knee: f64,
    pub elasticities: HashMap<i64, (f64, Vec<Option<f64>>)>, // keys → (keymode response, per family)
}

impl Slopes {
    /// The population's slope, curve and knee and warmup.rate_elasticities' {"<keys>": v, "<keys>:<family>": v}.
    pub fn new(slope: ByKeys<f64>, curve: ByKeys<f64>, knee: f64, flat: &StdMap<String, f64>, fams: &[String]) -> Slopes {
        let mut elasticities: HashMap<i64, (f64, Vec<Option<f64>>)> = HashMap::default();
        for (key, &v) in flat {
            let (k, fam) = key.split_once(':').map_or((key.as_str(), None), |(k, f)| (k, Some(f)));
            let Ok(k) = k.parse::<i64>() else { continue };
            let e = elasticities.entry(k).or_insert((f64::NAN, vec![None; fams.len()]));
            match fam {
                None => e.0 = v,
                Some(f) => if let Some(i) = fams.iter().position(|x| x == f) { e.1[i] = Some(v) },
            }
        }
        Slopes { slope, curve, knee, elasticities }
    }

    /// warmup.rate_elasticity: the family-weight blend of measured chart-rate responses.
    pub fn elasticity(&self, f: &Feats, weights: &[f64]) -> f64 {
        let (default, by_family) = match self.elasticities.get(&f.keys) {
            Some((d, fams)) => (if d.is_nan() { 1. } else { *d }, Some(fams)),
            None => (1., None),
        };
        let mut total = 0.;
        for (i, &w) in weights.iter().enumerate() {
            if w != 0. {
                total += w * by_family.and_then(|f| f[i]).unwrap_or(default);
            }
        }
        total
    }

    /// warmup.score_rate_slope
    pub fn physical(&self, f: &Feats, weights: &[f64]) -> f64 {
        let k = f.keys;
        let mut slope = *self.slope.get(k).or(self.slope.get(0)).unwrap_or(&2.);
        let curve = *self.curve.get(k).or(self.curve.get(0)).unwrap_or(&0.);
        slope += 2. * curve * (f.overall.max(0.05) / self.knee).ln().max(0.);
        (slope * self.elasticity(f, weights)).clamp(1., 8.)
    }
}

/// Skill names, warmup families, gap parts and knots (consts.json).
pub struct Consts {
    pub gap_part_names: Vec<String>,
    pub names: Vec<String>,
    pub families: Vec<String>,    // warmup.FAMILIES
    pub members: Vec<(String, Vec<String>)>, // warmup.MEMBERS in order
    gap_parts: Vec<Vec<String>>,         // recdata.GAP_SHARE_PARTS members; empty = "all"
    gap_knots: Vec<f64>,
    gap_shared: i64,
    gap_borrow: Vec<i64>,
    recent_families: Vec<String>,
    geometry: Vec<usize>,
    dominant_skip: Vec<String>,
    // derived: skill name → index into names, and per-name lookups
    pub idx: HashMap<String, usize>,
    fam_of: Vec<usize>,
    skip: Vec<bool>,
    member_idx: Vec<(usize, Vec<usize>)>, // family index → member skill indices (family_weights; no technical)
    gap_idx: Vec<Vec<usize>>,
    recent_idx: Vec<usize>,
}

impl Consts {
    pub fn standard() -> Consts {
        let v: Value = serde_json::from_str(include_str!("consts.json")).expect("consts.json");
        let strings = |x: &Value| -> Vec<String> { x.as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect() };
        Consts {
            gap_part_names: strings(&v["gap_part_names"]),
            names: strings(&v["names"]),
            families: strings(&v["families"]),
            members: v["members"].as_array().unwrap().iter().map(|m| (m[0].as_str().unwrap().to_string(), strings(&m[1]))).collect(),
            gap_parts: v["gap_parts"].as_array().unwrap().iter().map(strings).collect(),
            gap_knots: v["gap_knots"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect(),
            gap_shared: v["gap_shared"].as_i64().unwrap(),
            gap_borrow: v["gap_borrow"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect(),
            recent_families: strings(&v["recent_families"]),
            geometry: v["geometry"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect(),
            dominant_skip: strings(&v["dominant_skip"]),
            idx: HashMap::default(),
            fam_of: Vec::new(),
            skip: Vec::new(),
            member_idx: Vec::new(),
            gap_idx: Vec::new(),
            recent_idx: Vec::new(),
        }
        .derive()
        .expect("consts.json names")
    }

    /// warmup.family(skill) as a FAMILIES index.
    pub fn family(&self, skill: &str) -> usize {
        let name = self.members.iter().find(|(_, m)| m.iter().any(|x| x == skill)).map(|(k, _)| k.as_str())
            .unwrap_or(if self.families.iter().any(|x| x == skill) { skill } else { "rice" });
        self.families.iter().position(|x| x == name).unwrap()
    }

    /// warmup.dominant(f)
    pub fn dominant(&self, f: &Feats) -> usize {
        f.dom
    }

    /// Strongest non-descriptive skill's family, first of equals in dict order ('rice' without one).
    fn pick_dominant(&self, order: impl Iterator<Item = (f64, usize, bool)>) -> usize {
        let mut best: Option<(usize, f64)> = None;
        for (v, fam, descriptive) in order {
            if !descriptive && best.is_none_or(|b| v > b.1) {
                best = Some((fam, v));
            }
        }
        best.map_or_else(|| self.family("rice"), |b| b.0)
    }

    /// recommend.zvec
    pub fn zvec(&self, f: &Feats, mu: f64) -> Vec<f64> {
        let load = f.z_tail[2];
        let near_limit = 1. / (1. + (-20f64).max(20f64.min(-2. * (mu + 3.2))).exp());
        let mut z = f.sk.clone();
        z.extend_from_slice(&f.z_tail);
        z.push(f.width.max(0.).sqrt() * load);
        match &f.geometry {
            Some(g) => z.extend_from_slice(g),
            None => z.extend(std::iter::repeat_n(0., self.geometry.len())),
        }
        z.push(load * near_limit);
        z
    }

    /// warmup.family_weights(f) as a weight per FAMILIES index.
    pub fn family_weights(&self, f: &Feats) -> Vec<f64> {
        let n = self.families.len();
        let mut w = vec![0.; n];
        let mut total = 0.;
        for (fam, members) in &self.member_idx {
            let m = members.iter().map(|&x| f.sk[x].max(0.)).fold(0., f64::max);
            w[*fam] = m * m;
            total += m * m;
        }
        if total != 0. {
            w.iter_mut().for_each(|x| *x /= total);
        } else {
            w = vec![0.; n];
            w[self.dominant(f)] = 1.;
        }
        w
    }

    /// recommend.recent_vec
    pub fn recent_vec(&self, f: &Feats) -> Vec<f64> {
        let mut v: Vec<f64> = self.recent_idx.iter().map(|&i| f.sk[i]).collect();
        v.push((v[v.len() - 1] - 0.7).max(0.));
        v
    }

    fn derive(mut self) -> Res<Consts> {
        self.idx = self.names.iter().enumerate().map(|(i, n)| (n.clone(), i)).collect();
        let at = |c: &Consts, n: &str| c.idx.get(n).copied().ok_or_else(|| err(&format!("{n} is not a skill name")));
        self.fam_of = self.names.iter().map(|n| self.family(n)).collect();
        self.skip = self.names.iter().map(|n| self.dominant_skip.contains(n)).collect();
        let mut member_idx = Vec::new();
        for (name, members) in &self.members {
            if name != "technical" {
                let fam = self.families.iter().position(|x| x == name).ok_or_else(|| err(name))?;
                member_idx.push((fam, members.iter().map(|m| at(&self, m)).collect::<Res<Vec<_>>>()?));
            }
        }
        self.member_idx = member_idx;
        self.gap_idx = self.gap_parts.iter().map(|m| m.iter().map(|x| at(&self, x)).collect()).collect::<Res<_>>()?;
        self.recent_idx = self.recent_families.iter().map(|x| at(&self, x)).collect::<Res<_>>()?;
        Ok(self)
    }
}

pub struct PredModel {
    pub c: Arc<Consts>,
    pub pop: Arc<Pop>,
    pub levels: HashMap<i64, f64>,
    pub level: HashMap<i64, (f64, f64, f64)>, // keys → (level, level var, attempt sd)
    pub glob: f64,
    pub sd_all: f64,
    pub shape: HashMap<i64, (Vec<f64>, Vec<f64>, Vec<f64>)>, // keys → (zmean, zscale, beta)
    pub recent: HashMap<i64, (Vec<f64>, Vec<f64>, Vec<f64>)>,
    pub aff: HashMap<String, (f64, f64)>,
    pub rate_aff: HashMap<String, (Vec<f64>, Vec<f64>, Vec<f64>)>, // chart → (log rates, values, counts)
    pub rate_support: HashMap<String, (f64, f64)>,
    pub rate_bandwidth: ByKeys<f64>,
    pub tau2_c: f64,
    pub kap_c: f64,
    pub shown: (f64, f64),
    pub retry: f64,
    pub sd_att: HashMap<i64, f64>,
    // warmup.Model
    pub slopes: Arc<Slopes>,
    pub modes: ByKeys<Mode>,
    pub fallback: Mode,
}

/// One chart of a builder request.
pub struct ScoreJob {
    pub sha: String,
    pub key: String,          // chart identity: md5, else sha256
    pub bid: Option<i64>,
    pub md5: String,
    pub keys: i64,
    pub installed: bool,
    pub playcount: f64,
    pub convert: bool,
}

impl PredModel {
    /// A population-only model: `base_of` for many plays.
    pub fn population(c: Arc<Consts>, pop: Arc<Pop>) -> PredModel {
        PredModel { c, pop, levels: HashMap::default(), level: HashMap::default(), glob: 0., sd_all: 0., shape: HashMap::default(),
                    recent: HashMap::default(), aff: HashMap::default(), rate_aff: HashMap::default(), rate_support: HashMap::default(),
                    rate_bandwidth: [(0, 1.)].into_iter().collect(), tau2_c: 0., kap_c: 1., shown: (0., 1.), retry: 0.,
                    sd_att: HashMap::default(), slopes: Arc::new(Slopes::default()), modes: ByKeys::default(),
                    fallback: Mode { tau: 8., log_rate: false, amplitudes: Vec::new(), session: 0., fatigue: 0. } }
    }

    /// A fitted Personal and its warm-up model.
    pub fn fitted(c: Arc<Consts>, pop: Arc<Pop>, levels: HashMap<i64, f64>, fit: &Fit, warmup: &Model, shown: (f64, f64)) -> PredModel {
        let level = fit.level.iter().map(|(&k, &l)| (k, (l, fit.level_var[&k], fit.sd_att.get(&k).copied().unwrap_or(fit.sd_all)))).collect();
        let shape = fit.beta.iter().map(|(&k, b)| (k, (fit.zmean[&k].clone(), fit.zscale[&k].clone(), b.clone()))).collect();
        let rate_aff = fit.rate_aff.iter().map(|(c, a)| {
            (c.clone(), (a.iter().map(|x| x.0.ln()).collect(), a.iter().map(|x| x.1).collect(), a.iter().map(|x| x.2).collect()))
        }).collect();
        PredModel { c, pop, levels, level, glob: fit.glob, sd_all: fit.sd_all, shape, recent: fit.recent.clone(), aff: fit.aff.clone(), rate_aff,
                    rate_support: fit.rate_support.clone(), rate_bandwidth: fit.rate_bandwidth.iter().map(|(&k, &v)| (k, v)).collect(), tau2_c: fit.tau2_c,
                    kap_c: fit.kap_c, shown, retry: fit.retry, sd_att: fit.sd_att.clone(), slopes: Arc::new(warmup.slopes.clone()),
                    modes: warmup.modes.iter().map(|(&k, v)| (k, v.clone())).collect(), fallback: warmup.fallback.clone() }
    }

    /// nps.candidates_from's scoring of `jobs` into the request's Choices. `store` keeps the charts' features read
    /// across calls; its owner drops a chart whenever the chart's analyses change.
    #[allow(clippy::too_many_arguments)]
    pub fn score_into(&self, choices: &mut Choices, session: &PredSession, store: &FeatStore, jobs: &[ScoreJob],
                      selected: &[String], tuning: &crate::skills::Tuning, target: f64, skills: bool, wants_vibro: bool, preference: bool, now: f64) -> Res<()> {
        let ctx = Ctx { target, warm: session.warm, skills, wants_vibro, preference, version: store.version };
        let sel: Vec<Selection> = store.trees.iter().map(|t| t.select(selected, tuning)).collect();
        let included: Vec<&String> = selected.iter().filter(|k| !k.starts_with('!')).collect();
        static EMPTY: Vec<Entry> = Vec::new();
        let entries = |sha: &str| store.charts.get(sha).unwrap_or(&EMPTY);
        // Every core but two (~0.4 s on one thread for 25k charts).
        let threads = std::thread::available_parallelism().map_or(1, |x| x.get()).saturating_sub(2).max(1);
        let scored = background_map(jobs, threads, |j| self.score_chart(&ctx, &sel, entries(&j.sha), &j.key, j.bid, session));
        for (r, j) in scored.into_iter().zip(jobs) {
            let slot = match r? {
                None => (None, None),
                Some((best, prop, e)) => {
                    let choice = match (best, e) {
                        (Some((rate, value)), Some(p)) => {
                            let entry = entries(&j.sha).iter().find(|x| x.offer.rate == rate).unwrap();
                            let feats = entry.feats.as_ref().unwrap();
                            let matched = entry.skill.as_ref().map_or_else(Vec::new, |s| {
                                s.matched(&sel[usize::from(feats.keys != 4)]).into_iter().map(|i| included[i].clone()).collect()
                            });
                            let practice = if skills {
                                let tree = &store.trees[usize::from(feats.keys != 4)];
                                entry.skill.as_ref().map_or_else(String::new, |s| s.describe(&sel[usize::from(feats.keys != 4)], tree))
                            } else {
                                String::new()
                            };
                            Some(Arc::new(Choice {
                                practice, group: GROUPS[self.group(feats)].to_string(), nps: entry.offer.nps, length: entry.offer.play_span.unwrap_or(0.) / rate,
                                family: entry.offer.family.clone(), description: entry.offer.description.clone(), profile: entry.profile.clone(), acc_mid: p.acc_mid, matched,
                                sha: j.sha.clone(), md5: j.md5.clone(), keys: j.keys, rate, value, installed: j.installed, convert: j.convert,
                                playcount: j.playcount, pred: p, generated_at: now,
                            }))
                        }
                        _ => None,
                    };
                    (choice, prop.map(|(quality, rate, group)| Proposal { quality, rate, keys: j.keys, group: group.to_string() }))
                }
            };
            let at = choices.slot(&j.sha).ok_or_else(|| err(&format!("{} is not in this catalogue", j.sha)))?;
            choices.slots[at] = Some(slot);
        }
        Ok(())
    }

    /// nps.central_target.
    pub fn central_target(&self, target: f64, warm: bool) -> f64 {
        let ctx = Ctx { target, warm, skills: false, wants_vibro: false, preference: false, version: 0 };
        ctx.central()
    }

    /// nps.eligible: a prediction inside the tab's accuracy band.
    pub fn eligible(&self, p: &Prediction, target: f64, warm: bool) -> bool {
        let ctx = Ctx { target, warm, skills: false, wants_vibro: false, preference: false, version: 0 };
        ctx.eligible(p)
    }

    /// nps.Builder.initial_value: a known chart's density at the rate that meets the target, for analysis order.
    pub fn initial_value(&self, store: &FeatStore, job: &ScoreJob, rate: f64, session: &PredSession, target: f64) -> Res<f64> {
        let (Some(f), Some((notes, length))) = (store.feats(&job.sha, rate), store.size(&job.sha, rate)) else { return Ok(0.) };
        let ctx = Ctx { target, warm: session.warm, skills: false, wants_vibro: false, preference: false, version: store.version };
        let p = self.predict(f, &job.key, job.bid, rate, session)?;
        let need = rate * (ctx.rate_shift(&p, ctx.central()) / p.slope).exp();
        Ok(notes / length.max(1000.) * need.min(MAX_RATE) * if need < MIN_RATE * 0.85 { 0.05 } else { 1. })
    }

    /// Session.correction + Session.warmup_penalty, as Recommender.score adds them.
    pub fn session_shift(&self, f: &Feats, chart: &str, bid: Option<i64>, rate: f64, session: &PredSession) -> Res<f64> {
        Ok(self.predict(f, chart, bid, rate, session)?.shift)
    }
}

/// What nps.candidates_from reads of one analysed chart-rate besides the prediction.
struct Offer {
    rate: f64,
    notes: f64,
    length: f64,
    version: Option<i64>,
    nps: f64,
    play_span: Option<f64>,
    drill: f64,
    vibro: bool,
    family: String, // nps family (the chart's shape hash)
    description: String,
}

impl Offer {
    fn of(rate: f64, f: &Dict) -> Offer {
        // None reads as missing: fields of rates the filter skips first can be None.
        let num = |d: Option<&Dict>, k: &str| d.and_then(|d| d.get(k)).filter(|v| !v.is_none()).map(Val::f);
        let st = f.dict("nps");
        Offer {
            rate,
            notes: num(Some(f), "notes").unwrap_or(0.),
            length: num(Some(f), "length").unwrap_or(0.),
            version: num(st, "version").map(|v| v as i64),
            nps: num(st, "nps").unwrap_or(0.),
            play_span: num(st, "play_span"),
            drill: num(st, "drill").unwrap_or(0.),
            vibro: st.and_then(|d| d.get("vibro_focused")).is_some_and(Val::truthy),
            family: st.and_then(|d| d.get("family")).filter(|v| !v.is_none()).map_or_else(String::new, text),
            description: f.get("description").filter(|v| v.truthy()).map_or_else(String::new, text),
        }
    }

    /// nps._rateable without the Skills match: playable rate, long enough, current structure.
    fn static_ok(&self, version: i64) -> bool {
        (MIN_RATE..=MAX_RATE).contains(&self.rate) && self.notes >= 100. && self.length >= 10000.
            && self.version == Some(version) && self.nps != 0.
    }
}

/// One analysed chart-rate as the builder reads it.
pub struct Entry {
    offer: Offer,
    feats: Option<Feats>,        // None: not rateable in any request (Offer::static_ok)
    skill: Option<SkillValues>,  // with feats
    profile: Vec<f64>,           // nps.profile, with feats
}

/// skill_practice.dimensions/tree of one keymode class (4K or wider).
pub struct Tree {
    dims: Vec<String>,
    members: Vec<Vec<String>>,
    tech: Vec<(usize, String)>,    // practice_strengths: dims index → tech_profile leaf
}

/// One request's skill_practice.normalize selection against a Tree (None: not a dimension, strength 0).
struct Selection {
    excluded: Vec<(Option<usize>, f64)>, // with how much of it a map may hold, as a share of the map's strongest skill
    positive: Vec<Option<usize>>,
    other: bool,                         // charts with no included skill stay in the list
}

impl Tree {
    fn select(&self, selected: &[String], tuning: &crate::skills::Tuning) -> Selection {
        let at = |k: &str| self.dims.iter().position(|d| d == k);
        Selection {
            excluded: selected.iter().filter_map(|k| k.strip_prefix('!')).map(|k| (at(k), tuning.hold(k))).collect(),
            positive: selected.iter().filter(|k| !k.starts_with('!')).map(|k| at(k)).collect(),
            other: tuning.other > 0.,
        }
    }
}

/// skill_practice.strengths and practice_strengths of one chart-rate, per Tree dimension.
struct SkillValues {
    values: Vec<f64>,
    practice: Vec<(usize, f64)>, // where it differs from `values`: a full copy per chart was 24 MB
}

impl SkillValues {
    fn of(f: &Dict, tree: &Tree) -> SkillValues {
        let sk = f.dict("sk");
        let values: Vec<f64> = tree.members.iter().map(|members| {
            members.iter().filter_map(|k| sk.and_then(|d| d.get(k))).fold(0f64, |m, v| m.max(0f64.max(v.f())))
        }).collect();
        let mut practice = Vec::new();
        // Technical practice needs actual technical character (tech_profile), not an awkward local peak.
        if let Some(profile) = f.dict("tech_profile").filter(|p| !p.0.is_empty()) {
            for (i, leaf) in &tree.tech {
                if let Some(capped) = profile.get(leaf).map(|v| 0f64.max(v.f()).sqrt()).filter(|c| *c < values[*i]) {
                    practice.push((*i, capped));
                }
            }
        }
        SkillValues { values, practice }
    }

    fn practice(&self, i: usize) -> f64 {
        self.practice.iter().find(|p| p.0 == i).map_or(self.values[i], |p| p.1)
    }

    /// An avoided skill reaches what the map may hold. The strictest step reaches below the 0.45 a skill needs to count as
    /// demanded: "maps with any of it" has to catch what the other steps call too little to matter.
    fn avoided(&self, s: &Selection) -> bool {
        let maximum = self.values.iter().copied().fold(0., f64::max);
        s.excluded.iter().any(|&(k, hold)| k.is_some_and(|i| self.values[i] >= (if hold < 0.6 { 0.2f64 } else { 0.45 }).max(hold * maximum)))
    }

    /// skill_practice.match(f, selected)[0]
    fn strength(&self, s: &Selection) -> f64 {
        let maximum = self.values.iter().copied().fold(0., f64::max);
        let floor = 0.45f64.max(0.60 * maximum);
        if self.avoided(s) {
            return 0.;
        }
        if s.positive.is_empty() {
            return 1.;
        }
        let matched = s.positive.iter().map(|&k| k.map_or(0., |i| self.practice(i))).filter(|&v| v >= floor).map(|v| v.min(1.)).fold(0., f64::max);
        if matched == 0. && s.other { 1. } else { matched }
    }

    /// skill_practice.describe_match: the two strongest matches, by raw strength.
    fn describe(&self, s: &Selection, tree: &Tree) -> String {
        let maximum = self.values.iter().copied().fold(0., f64::max);
        let floor = 0.45f64.max(0.60 * maximum);
        if self.avoided(s) {
            return "Mixed skills".to_string();
        }
        let requested: Vec<usize> = if s.positive.is_empty() {
            (0..tree.dims.len()).filter(|&i| !tree.dims[i].contains('/')).collect()
        } else {
            s.positive.iter().flatten().copied().collect()
        };
        let mut matched: Vec<usize> = requested.into_iter().filter(|&i| self.practice(i) >= floor).collect();
        matched.sort_by(|&a, &b| self.values[b].total_cmp(&self.values[a]));
        let names: Vec<&str> = matched.iter().take(2).map(|&i| crate::skills::name(&tree.dims[i])).collect();
        if names.is_empty() { "Mixed skills".to_string() } else { names.join(" / ") }
    }

    /// skill_practice.match(f, selected)[1], as indices into the included skills.
    fn matched(&self, s: &Selection) -> Vec<usize> {
        let floor = 0.45f64.max(0.60 * self.values.iter().copied().fold(0., f64::max));
        s.positive.iter().enumerate().filter(|(_, k)| k.is_some_and(|i| self.practice(i) >= floor)).map(|(j, _)| j).collect()
    }
}

/// nps.Builder's analysed library: sha → its rates in analysis order, read once. The owner drops a chart whenever
/// its analyses change.
pub struct FeatStore {
    charts: HashMap<String, Vec<Entry>>,
    trees: Vec<Tree>, // 4K, wider (skill_practice.native_tree)
    profile: Vec<String>, // nps.PROFILE
    c: Arc<Consts>,
    pub version: i64,
}

impl FeatStore {
    /// Per class (4K, wider) [(dimension, members)]; nps.PROFILE; the structure version.
    pub fn new(trees: Vec<Vec<(String, Vec<String>)>>, profile: Vec<String>, c: Arc<Consts>, version: i64) -> Res<Self> {
        let trees: Vec<Tree> = trees.into_iter().map(|dims| Tree {
            tech: ["technical", "technical/patterntech", "technical/rhythmtech"].iter()
                .filter_map(|k| dims.iter().position(|d| d.0 == *k).map(|i| (i, k.rsplit('/').next().unwrap().to_string())))
                .collect(),
            members: dims.iter().map(|d| d.1.clone()).collect(),
            dims: dims.into_iter().map(|d| d.0).collect(),
        }).collect();
        if trees.len() != 2 {
            return Err(err("FeatStore needs the 4K and wider skill trees"));
        }
        Ok(FeatStore { charts: HashMap::default(), trees, profile, c, version })
    }

    /// One analysed rate of a chart; an analysed rate is replaced where it stands, a new one comes last.
    pub fn put(&mut self, sha: &str, rate: f64, f: &Dict) {
        let offer = Offer::of(rate, f);
        let (feats, skill, profile) = match Feats::of(f, &self.c) {
            Ok(feats) => {
                let skill = SkillValues::of(f, &self.trees[usize::from(feats.keys != 4)]);
                let profile = if offer.static_ok(self.version) { profile_of(f, &self.profile) } else { Vec::new() };
                (Some(feats), Some(skill), profile)
            }
            Err(_) => (None, None, Vec::new()),
        };
        let entry = Entry { offer, feats, skill, profile };
        let entries = self.charts.entry(sha.to_string()).or_default();
        match entries.iter_mut().find(|e| e.offer.rate == rate) {
            Some(slot) => *slot = entry,
            None => entries.push(entry),
        }
    }

    /// Rates analysed, in analysis order.
    pub fn rates(&self, sha: &str) -> Vec<f64> {
        self.charts.get(sha).map_or_else(Vec::new, |v| v.iter().map(|e| e.offer.rate).collect())
    }

    pub fn has_rate(&self, sha: &str, rate: f64) -> bool {
        self.charts.get(sha).is_some_and(|v| v.iter().any(|e| e.offer.rate == rate))
    }

    /// The chart has an analysis of the current structure version.
    pub fn current(&self, sha: &str) -> bool {
        self.charts.get(sha).is_some_and(|v| v.iter().any(|e| e.offer.version == Some(self.version)))
    }

    pub fn feats(&self, sha: &str, rate: f64) -> Option<&Feats> {
        self.charts.get(sha)?.iter().find(|e| e.offer.rate == rate)?.feats.as_ref()
    }

    /// nps.group of an analysed rate.
    pub fn group(&self, sha: &str, rate: f64, model: &PredModel) -> Option<&'static str> {
        self.feats(sha, rate).map(|f| GROUPS[model.group(f)])
    }

    /// What a list shows of an analysed rate; `practice` for the Skills list.
    pub fn metadata(&self, sha: &str, rate: f64, selected: &[String], tuning: &crate::skills::Tuning, practice: bool, model: &PredModel) -> Option<crate::choices::Metadata> {
        let e = self.charts.get(sha)?.iter().find(|e| e.offer.rate == rate)?;
        let feats = e.feats.as_ref()?;
        let tree = usize::from(feats.keys != 4);
        let sel = self.trees[tree].select(selected, tuning);
        let included: Vec<&String> = selected.iter().filter(|k| !k.starts_with('!')).collect();
        let skill = e.skill.as_ref().filter(|_| practice);
        Some(crate::choices::Metadata {
            family: e.offer.family.clone(), group: GROUPS[model.group(feats)].to_string(), nps: e.offer.nps, length: e.offer.play_span.unwrap_or(e.offer.length / 1000.) / rate,
            profile: e.profile.clone(), description: e.offer.description.clone(), matched: skill.map_or_else(Vec::new, |s| s.matched(&sel).into_iter().map(|i| included[i].clone()).collect()),
            practice: skill.map_or_else(String::new, |s| s.describe(&sel, &self.trees[tree])),
        })
    }

    /// (notes, length) of an analysed rate.
    pub fn size(&self, sha: &str, rate: f64) -> Option<(f64, f64)> {
        let e = self.charts.get(sha)?.iter().find(|e| e.offer.rate == rate)?;
        Some((e.offer.notes, e.offer.length))
    }

    pub fn contains(&self, sha: &str) -> bool {
        self.charts.contains_key(sha)
    }

    pub fn drop_chart(&mut self, sha: &str) {
        self.charts.remove(sha);
    }

    pub fn len(&self) -> usize {
        self.charts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.charts.is_empty()
    }
}

/// Python's round(x, 2): correctly rounded on the exact binary value, as Rust's formatting.
fn round2(x: f64) -> f64 {
    format!("{x:.2}").parse().unwrap()
}

const MISS_TOL: f64 = 0.005;
pub const MIN_RATE: f64 = 0.70; // nps.MIN_RATE / MAX_RATE / MAX_TARGET / MAX_HIGH
pub const MAX_RATE: f64 = 1.50;
const MAX_TARGET: f64 = 0.985;
const MAX_HIGH: f64 = 0.99;

/// The request context of nps.candidates_from.
struct Ctx {
    target: f64, // the tab's accuracy target (session.acc_target)
    warm: bool,
    skills: bool,
    wants_vibro: bool,
    preference: bool,
    version: i64,
}

impl Ctx {
    fn err(&self) -> f64 {
        1. - self.target.min(MAX_TARGET)
    }
    /// nps.central_target. Jack-led charts used to aim a third closer to 100 %: a workaround for chordjacks rated too easy, gone
    /// with the chord rules of calculator 65c0a4cb63d6 (user 2026-11-02).
    fn central(&self) -> f64 {
        self.target.min(MAX_TARGET)
    }
    fn band(&self) -> (f64, f64) {
        let t = self.central();
        (t - 0.25 * self.err(), MAX_HIGH.min(t + 0.25 * self.err()))
    }
    fn opening_floor(&self, p: &Prediction) -> f64 {
        let low = self.band().0 - 0.1 * self.err();
        if self.warm {
            return low;
        }
        let cold = (-p.activation / p.tau).exp();
        low + cold * (self.central() - 0.05 * self.err() - low)
    }
    fn eligible(&self, p: &Prediction) -> bool {
        let (low, high) = self.band();
        let slack = 0.1 * self.err();
        if !(low - slack <= p.acc_mid && p.acc_mid <= MAX_HIGH.min(high + slack)) {
            return false;
        }
        self.warm || p.opening_acc >= self.opening_floor(p)
    }
    fn rate_shift(&self, p: &Prediction, target: f64) -> f64 {
        let mut gap = (1. - target).ln() - (1. - p.acc_mid).max(1e-6).ln();
        if !self.warm {
            gap = gap.min((1. - self.opening_floor(p)).ln() - (1. - p.opening_acc).max(1e-6).ln());
        }
        gap
    }
    fn ready(&self, p: &Prediction) -> f64 {
        if self.warm { 1. } else { -(-p.activation / p.tau).exp_m1() }
    }
    /// nps.practice_value (NPS) or skill_practice.value (Skills) at `rate`, with acc_mid → `acc`.
    fn value(&self, o: &Offer, strength: Option<f64>, nps: f64, p: &Prediction, acc: f64, rate: f64, target: f64) -> Option<f64> {
        let length = o.play_span.unwrap_or(o.length / 1000.) / rate;
        let quality = 1.5 * (90. / length.max(10.)).ln().max(0.) + 2.8 * o.drill + 0.18 * rate.ln().abs();
        if self.skills {
            let strength = strength.filter(|s| *s != 0.)?;
            let preference = if self.preference { 2.2 * strength.min(1.).max(0.01).ln() } else { 0. };
            return Some(1.1 * strength.max(0.01).ln() + preference - 14. * (acc - target).abs() - 0.40 * p.sdm - quality
                        - 0.15 * rate.ln().abs());
        }
        let ready = self.ready(p);
        Some(ready * nps.max(1.).ln() - 12. * (acc - target).abs() - (0.35 + 1. - ready) * p.sdm - quality)
    }
}

const GROUPS: [&str; 7] = ["rice", "chords", "chordjack", "jack", "ln", "trill", "sv"];

/// What the predictor reads of one feature dict.
#[derive(Clone)]
pub struct Feats {
    pub keys: i64,
    pub overall: f64,
    pub od: f64,
    pub length: f64,
    pub sk: Vec<f64>,                  // by Consts.names, missing = 0
    pub dom: usize,                    // warmup.dominant
    pub z_tail: [f64; 5],              // ln, stam, load, chord/keys, overlap
    pub width: f64,
    pub geometry: Option<Vec<f64>>,
}

impl Feats {
    /// Only the skills: what dominant and family_weights read of an attempt's saved features.
    pub fn skills(sk: &Value, c: &Consts) -> Feats {
        let mut out = Feats::bare(0, 0., 0., c);
        let mut order = Vec::new();
        for (k, v) in sk.as_object().into_iter().flatten() {
            let v = v.as_f64().unwrap_or(0.);
            match c.idx.get(k) {
                Some(&i) => {
                    out.sk[i] = v;
                    order.push((v, c.fam_of[i], c.skip[i]));
                }
                None => order.push((v, c.family(k), c.dominant_skip.iter().any(|x| x == k))),
            }
        }
        out.dom = c.pick_dominant(order.into_iter());
        out
    }

    /// A chart known by its rating alone (nps.reach's search).
    pub fn bare(keys: i64, overall: f64, od: f64, c: &Consts) -> Feats {
        Feats { keys, overall, od, length: 0., sk: vec![0.; c.names.len()], dom: c.family("rice"), z_tail: [0.; 5], width: 0., geometry: None }
    }

    pub fn of(f: &Dict, c: &Consts) -> Res<Feats> {
        let need = |k: &str| f.get(k).filter(|v| !v.is_none()).ok_or_else(|| err(&format!("features lack {k}")));
        let keys = need("keys")?.f() as i64;
        let Val::Dict(sk_d) = need("sk")? else { return Err(err("features' sk is not a dict")) };
        let mut order = Vec::with_capacity(sk_d.0.len());
        let mut sk = vec![0.; c.names.len()];
        for (k, v) in &sk_d.0 {
            let ms_calc::val::Key::S(k) = k else { continue };
            let v = v.f();
            match c.idx.get(k) {
                Some(&i) => {
                    sk[i] = v;
                    order.push((v, c.fam_of[i], c.skip[i]));
                }
                None => order.push((v, c.family(k), c.dominant_skip.iter().any(|x| x == k))),
            }
        }
        let sub = |name: &str, key: &str, default: f64| f.dict(name).and_then(|d| d.get(key)).map_or(default, Val::f);
        let load = sub("endurance", "load", 0.);
        let chord = sub("nps", "chord", 1.);
        let overlap = sub("nps", "overlap", 0.);
        let list = |v: &Val| -> Option<Vec<Val>> {
            match v {
                Val::List(l) | Val::Tuple(l) => Some(l.clone()),
                _ => None,
            }
        };
        let width = f.get("calibration_features").and_then(list).and_then(|l| l.get(8).map(Val::f)).unwrap_or(0.);
        let geometry = f.dict("execution").and_then(|e| e.get("residual_vector")).filter(|v| v.truthy()).and_then(list)
            .map(|l| c.geometry.iter().map(|&i| l[i].f()).collect::<Vec<f64>>());
        Ok(Feats {
            keys,
            overall: need("overall")?.f(),
            od: need("od")?.f(),
            length: f.get("length").map_or(0., Val::f),
            sk,
            dom: c.pick_dominant(order.into_iter()),
            z_tail: [need("ln")?.f(), need("stam")?.f(), load, chord / (keys.max(1) as f64), overlap],
            width,
            geometry,
        })
    }
}

/// Frozen Session tables of one request (the display target's session view).
pub struct PredSession {
    pub warm: bool,
    pub by_keys: HashMap<i64, KeyState>,
    pub charts: HashMap<String, ChartSession>, // charts with plays in the session window
}

pub struct KeyState {
    pub form: Vec<f64>,       // session_form posterior mean: shared + per family
    pub penalty: Vec<f64>,    // warmup penalty(keys, family, activation, 0, 1) per family
    pub session: f64,         // warmup session_cold(keys, session_clock)
    pub session_tau: f64,
    pub fatigue: f64,         // warmup fatigue(keys, session_clock)
    pub activation: Vec<f64>, // activation(keys, family) per family
}

/// What the session knows of one played chart: Session.rate_penalty, chart_correction and retry read it.
#[derive(Clone, Default)]
pub struct ChartSession {
    pub last: Vec<(f64, f64)>,                       // (round(rate, 2), displayed accuracy) of each rate's last finish
    pub retried: bool,
    pub downrates: Vec<(f64, f64)>,                  // (played rate, age factor)
    pub attempts: Vec<(f64, f64, f64, Vec<f64>)>,    // (rate, age factor, error, family loading)
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Prediction {
    pub mu: f64,
    pub base_mu: f64,
    pub shift: f64, // session correction + cold penalty (mu − base_mu)
    pub cold: f64,
    pub sdm: f64,
    pub sda: f64,
    pub sd: f64,
    pub acc_mid: f64,
    pub acc_lo: f64,
    pub acc_hi: f64,
    pub opening_acc: f64,
    pub activation: f64,
    pub slope: f64,
    pub tau: f64,
}

pub const BAND_Z: f64 = 1.2816 * 0.75; // recommend.BAND_Z

impl PredModel {

    /// score_into for one chart: None (nothing rateable) or (best (rate, value), proposal (quality, rate, group),
    /// the best rate's prediction).
    #[allow(clippy::type_complexity)]
    fn score_chart(&self, ctx: &Ctx, sel: &[Selection], entries: &[Entry], key: &str, bid: Option<i64>, session: &PredSession)
                   -> Res<Option<(Option<(f64, f64)>, Option<(f64, f64, &'static str)>, Option<Prediction>)>> {
        let mut rated: Vec<(&Offer, &Feats, Prediction, Option<f64>)> = Vec::new();
        for e in entries {
            let (Some(feats), Some(skill)) = (&e.feats, &e.skill) else { continue };
            if !e.offer.static_ok(ctx.version) {
                continue;
            }
            let s = if ctx.skills { Some(skill.strength(&sel[usize::from(feats.keys != 4)])) } else { None };
            if ctx.skills && s == Some(0.) {
                continue;
            }
            let p = self.predict(feats, key, bid, e.offer.rate, session)?;
            rated.push((&e.offer, feats, p, s));
        }
        if rated.is_empty() {
            return Ok(None);
        }
        let target = ctx.central();
        // Rates whose last finish (14 days) missed the target by more than MISS_TOL: not offered again, the next rate
        // comes from that result (user 2026-10-13: "not get picked exactly the same rate even though it's not accurate").
        let missed = |rate: f64| session.charts.get(key).and_then(|c| c.last.iter().find(|x| x.0 == round2(rate))).map(|x| x.1).filter(|acc| (acc - target).abs() > MISS_TOL);
        let shift_of = |i: usize| ctx.rate_shift(&rated[i].2, target);
        let mut a = 0;
        for i in 1..rated.len() {
            if shift_of(i).abs() < shift_of(a).abs() {
                a = i;
            }
        }
        let (o, f, e, sa) = (rated[a].0, rated[a].1, &rated[a].2, rated[a].3);
        let r = o.rate;
        let mut slope = e.slope;
        if rated.len() > 1 {
            let mut nearest: Option<usize> = None;
            for (i, v) in rated.iter().enumerate() {
                if (v.0.rate - r).abs() > 0.015 && nearest.is_none_or(|n| (v.0.rate - r).abs() < (rated[n].0.rate - r).abs()) {
                    nearest = Some(i);
                }
            }
            if let Some(n) = nearest {
                let (rr, ee) = (rated[n].0.rate, &rated[n].2);
                let empirical = ((1. - ee.acc_mid).max(1e-6).ln() - (1. - e.acc_mid).max(1e-6).ln()) / (rr / r).ln();
                if empirical > 0.2 {
                    slope = 8f64.min(0.5f64.max(empirical));
                }
            }
        }
        let shift = ctx.rate_shift(e, target) / slope;
        let mut proposal = round2(MAX_RATE.min(MIN_RATE.max(r * shift.exp())));
        let has = |x: f64| entries.iter().any(|e| e.offer.rate == x);
        let eligible_a = ctx.eligible(e);
        if !eligible_a && has(proposal) && shift != 0. {
            let direction: i64 = if shift < 0. { -1 } else { 1 };
            let mut step = if direction < 0 { (r * 100. - 1e-8).floor() as i64 } else { (r * 100. + 1e-8).ceil() as i64 };
            while (MIN_RATE..=MAX_RATE).contains(&(step as f64 / 100.)) && has(step as f64 / 100.) {
                step += direction;
            }
            if (MIN_RATE..=MAX_RATE).contains(&(step as f64 / 100.)) {
                proposal = step as f64 / 100.;
            }
        }
        let anchor_missed = missed(r);
        if let Some(acc) = anchor_missed {
            // The anchor is a rate you just missed: move from your result, not the forecast.
            let shift = ((1. - target).ln() - (1. - acc).max(1e-6).ln()) / slope;
            proposal = round2(MAX_RATE.min(MIN_RATE.max(r * shift.exp())));
            if proposal == round2(r) {
                proposal = round2(MAX_RATE.min(MIN_RATE.max(r + if shift > 0. { 0.01 } else { -0.01 })));
            }
        }
        let mut prop = None;
        if !has(proposal) && (!eligible_a || (e.acc_mid - target).abs() > 0.003 || anchor_missed.is_some()) {
            let nps = if ctx.skills { o.nps } else { o.nps * proposal / r };
            if let Some(q) = ctx.value(o, sa, nps, e, target, proposal, target) {
                prop = Some((q, proposal, GROUPS[self.group(f)]));
            }
        }
        let mut best: Option<(usize, f64)> = None;
        for (i, &(o, _, ref e, s)) in rated.iter().enumerate() {
            if missed(o.rate).is_some() || !ctx.eligible(e) || (o.vibro && !(ctx.skills && ctx.wants_vibro)) {
                continue;
            }
            if let Some(v) = ctx.value(o, s, o.nps, e, e.acc_mid, o.rate, target) {
                if best.is_none_or(|b| v > b.1) {
                    best = Some((i, v));
                }
            }
        }
        // Picked with the first rate's target (as the Builder did); valued with its own, as the worker's re-score did.
        let best_out = best.and_then(|(i, _)| {
            let (o, e, s) = (rated[i].0, &rated[i].2, rated[i].3);
            ctx.value(o, s, o.nps, e, e.acc_mid, o.rate, ctx.central()).map(|v| (o.rate, v))
        });
        Ok(Some((best_out, prop, best.map(|(i, _)| rated.swap_remove(i).2))))
    }
    /// nps._group: the chart's leading pattern family (GROUPS index).
    pub fn group(&self, f: &Feats) -> usize {
        let g = |names: &[&str]| names.iter().map(|n| self.c.idx.get(*n).map_or(0., |&i| f.sk[i])).fold(f64::NEG_INFINITY, f64::max);
        let values = [
            g(&["stream", "delay", "dump", "jumpstream", "handstream"]), g(&["chordstream"]), g(&["chordjack"]),
            g(&["jackspeed", "jack", "minijack", "longjack"]), g(&["ln", "release", "hybrid", "inverse", "shield"]),
            g(&["jumptrill", "splittrill", "bracket"]), g(&["sv"]),
        ];
        let mut best = 0;
        for i in 1..values.len() {
            if values[i] > values[best] {
                best = i;
            }
        }
        best
    }

    fn dominant(&self, f: &Feats) -> usize {
        self.c.dominant(f)
    }

    fn family_weights(&self, f: &Feats) -> Vec<f64> {
        self.c.family_weights(f)
    }

    fn mode(&self, k: i64) -> &Mode {
        self.modes.get(k).unwrap_or(&self.fallback)
    }

    fn gap_term(&self, f: &Feats, level: Option<f64>) -> f64 {
        let k = if self.pop.gap.get(f.keys).is_some() {
            f.keys
        } else if self.c.gap_borrow.contains(&f.keys) {
            self.c.gap_shared
        } else {
            return 0.;
        };
        let (Some(parts), Some(level)) = (self.pop.gap.get(k), level) else { return 0. };
        let means = self.pop.share_mean.get(k);
        let knots = &self.c.gap_knots;
        let g = ((f.overall.max(0.05)).ln() - level).clamp(knots[0], knots[knots.len() - 1]);
        let hats: Vec<f64> = (0..knots.len()).map(|i| {
            let kk = knots[i];
            let lo = if i > 0 { knots[i - 1] } else { kk - 1. };
            let hi = if i + 1 < knots.len() { knots[i + 1] } else { kk + 1. };
            1f64.min(0f64.max(((g - lo) / (kk - lo)).min((hi - g) / (hi - kk))))
        }).collect();
        let mut total = 0.;
        for (p, cs) in parts {
            let members = &self.c.gap_idx[*p];
            let share = if members.is_empty() { 1. } else { members.iter().map(|&x| f.sk[x]).fold(0., f64::max).min(1.) };
            let mean = means.map_or(0., |m| m[*p]);
            let mut s = 0.;
            for (c, h) in cs.iter().zip(&hats) {
                s += c * h;
            }
            total += (share - mean) * s;
        }
        total
    }

    /// recommend.base_of → (value, n public rows)
    pub fn base_of(&self, f: &Feats, b: Option<i64>, rate: f64, level: Option<f64>) -> (f64, f64) {
        let pop = &self.pop;
        let s = pop.slope(f.keys);
        let c = *pop.window.get(f.keys).or(pop.window.get(0)).unwrap_or(&0.);
        let (mut mb, mut n_mb) = b.and_then(|b| pop.mb.get(&b).copied()).unwrap_or((0., 0.));
        if let Some(anchors) = b.and_then(|b| pop.mbr.get(&b)) {
            if rate > 0. && !anchors.is_empty() {
                let xs: Vec<f64> = anchors.iter().map(|a| a.0).collect();
                mb = interp(rate.ln(), &xs, &anchors.iter().map(|a| a.1).collect::<Vec<_>>());
                n_mb = interp(rate.ln(), &xs, &anchors.iter().map(|a| a.2).collect::<Vec<_>>());
            }
        }
        let lr = f.overall.max(0.05).ln();
        let curve = pop.curve(f.keys);
        let bend = (lr - pop.knee.ln()).max(0.).powi(2);
        (s * lr + curve * bend + c * (64. - 3. * f.od).max(10.).ln() + mb + self.gap_term(f, level), n_mb)
    }

    pub fn chart_affinity(&self, chart: &str, rate: f64, k: i64) -> (f64, f64) {
        let (mut a, mut n_c) = self.aff.get(chart).copied().unwrap_or((0., 0.));
        if let Some((xs, vs, ns)) = self.rate_aff.get(chart) {
            if !xs.is_empty() && rate != 0. {
                a = interp(rate.ln(), xs, vs);
                n_c = interp(rate.ln(), xs, ns);
            }
        }
        if let Some(&(lo, hi)) = self.rate_support.get(chart) {
            if rate != 0. {
                let distance = 0f64.max((lo / rate).ln()).max((rate / hi).ln());
                let bw = *self.rate_bandwidth.get(k).or(self.rate_bandwidth.get(0)).unwrap();
                let transfer = (-0.5 * (distance / bw).powi(2)).exp();
                if a <= 0. || rate < lo {
                    a *= transfer;
                }
                n_c *= transfer * transfer;
            }
        }
        (a, n_c)
    }

    pub fn zvec(&self, f: &Feats, mu: f64) -> Vec<f64> {
        self.c.zvec(f, mu)
    }

    /// Personal.level_of, `recent_offset`: the terms of `personal` around the shape.
    pub fn level_of(&self, k: i64) -> f64 {
        self.level.get(&k).map_or(self.glob + 0.3, |l| l.0)
    }

    pub fn recent_offset(&self, f: &Feats) -> f64 {
        let Some((rm, rs, beta)) = self.recent.get(&f.keys) else { return 0. };
        let shares: Vec<f64> = self.c.recent_idx.iter().map(|&i| f.sk[i]).collect();
        let hinge = (shares[shares.len() - 1] - 0.7).max(0.);
        shares.iter().chain(std::iter::once(&hinge)).enumerate().map(|(i, v)| (v - rm[i]) / rs[i] * beta[i]).sum()
    }

    /// Personal.predict → (mu, model sd, attempt sd)
    pub fn personal(&self, f: &Feats, chart: &str, b: Option<i64>, rate: f64) -> (f64, f64, f64) {
        let k = f.keys;
        let (base, n_mb) = self.base_of(f, b, rate, self.levels.get(&k).copied());
        let (l, lv, sd) = self.level.get(&k).copied().unwrap_or((self.glob + 0.3, 0.25, self.sd_all));
        let mut mu = l + base;
        if let Some((zm, zs, beta)) = self.shape.get(&k) {
            let z = self.zvec(f, mu);
            let mut dot = 0.;
            for i in 0..z.len() {
                dot += (z[i] - zm[i]) / zs[i] * beta[i];
            }
            mu += dot;
        }
        if let Some((rm, rs, beta)) = self.recent.get(&k) {
            let shares: Vec<f64> = self.c.recent_idx.iter().map(|&i| f.sk[i]).collect();
            let hinge = (shares[shares.len() - 1] - 0.7).max(0.);
            let mut dot = 0.;
            for (i, v) in shares.iter().chain(std::iter::once(&hinge)).enumerate() {
                dot += (v - rm[i]) / rs[i] * beta[i];
            }
            mu += dot;
        }
        let (a, n_c) = self.chart_affinity(chart, rate, k);
        mu += a;
        let var = lv + self.pop.tau2 * self.pop.kappa / (n_mb + self.pop.kappa) + self.tau2_c * self.kap_c / (n_c + self.kap_c);
        (mu, var.sqrt(), sd)
    }


    /// Personal.rate_response: warmup.score_rate_slope + public/personal rate evidence around this rate.
    pub fn rate_response(&self, f: &Feats, chart: &str, b: Option<i64>, rate: f64, weights: &[f64]) -> f64 {
        let k = f.keys;
        let physical = self.slopes.physical(f, weights);
        let (lo, hi) = (rate / 1.01, rate * 1.01);
        let level = self.levels.get(&k).copied();
        let left = self.base_of(f, b, lo, level).0 + self.chart_affinity(chart, lo, k).0;
        let right = self.base_of(f, b, hi, level).0 + self.chart_affinity(chart, hi, k).0;
        (physical + (right - left) / (hi / lo).ln()).clamp(1., 8.)
    }

    /// Personal.rate_loss
    pub fn rate_loss(&self, f: &Feats, chart: &str, b: Option<i64>, rate: f64, loss: f64, weights: &[f64]) -> f64 {
        if loss <= 0. {
            return 0.;
        }
        let k = f.keys;
        let slope = self.pop.slope(k);
        let curve = self.pop.curve(k);
        let knee = self.pop.knee.ln();
        let elasticity = self.slopes.elasticity(f, weights);
        let old = f.overall.max(0.05).ln();
        let new = (f.overall * (elasticity * loss).exp()).max(0.05).ln();
        let mut shift = slope * (new - old) + curve * ((new - knee).max(0.).powi(2) - (old - knee).max(0.).powi(2));
        if let Some(anchors) = b.and_then(|b| self.pop.mbr.get(&b)).filter(|a| !a.is_empty()) {
            let xs: Vec<f64> = anchors.iter().map(|a| a.0).collect();
            let ys: Vec<f64> = anchors.iter().map(|a| a.1).collect();
            shift += interp(rate.ln() + loss, &xs, &ys) - interp(rate.ln(), &xs, &ys);
        }
        if self.aff.contains_key(chart) {
            shift += self.chart_affinity(chart, rate * loss.exp(), k).0 - self.chart_affinity(chart, rate, k).0;
        }
        shift.max(0.)
    }

    /// Session.rate_penalty + chart_correction − retry of a chart with plays in the window. The shared form posterior
    /// marginalizes chart error; conditional on it, repeated measurements update the chart's own residual, so a bad
    /// forecast stays evidence about THIS chart even when downweighted as evidence about every other map.
    fn played(&self, f: &Feats, chart: &str, rate: f64, st: &KeyState, s: &PredSession, variance: f64) -> f64 {
        let Some(cs) = s.charts.get(chart) else { return 0. };
        let mut penalty: f64 = 0.;
        for &(actual, age) in cs.downrates.iter().filter(|d| rate > d.0) {
            penalty = penalty.max((2.8 * (rate / actual).ln()).min(0.8) * age);
        }
        let mut correction = 0.;
        if variance > 0. {
            let bandwidth = *self.rate_bandwidth.get(f.keys).or(self.rate_bandwidth.get(0)).unwrap_or(&1.);
            let noise = self.sd_att.get(&f.keys).copied().unwrap_or(self.sd_all).powi(2).max(1e-4);
            let (mut precision, mut numerator) = (1. / variance.max(1e-6), 0.);
            for (at, age, error, loading) in &cs.attempts {
                let transfer = (-0.5 * ((rate / at).ln() / bandwidth).powi(2)).exp();
                let weight = transfer * age / noise;
                numerator += weight * (error - st.form[0] - loading.iter().zip(&st.form[1..]).map(|(a, b)| a * b).sum::<f64>());
                precision += weight;
            }
            correction = numerator / precision;
        }
        penalty + correction - if cs.retried { self.retry } else { 0. }
    }

    pub fn predict(&self, f: &Feats, chart: &str, b: Option<i64>, rate: f64, s: &PredSession) -> Res<Prediction> {
        let k = f.keys;
        let weights = self.family_weights(f);
        let slope = self.rate_response(f, chart, b, rate, &weights);
        let (mut mu, sdm, sda) = self.personal(f, chart, b, rate);
        let base_mu = mu;
        let mode = self.mode(k);
        let st = s.by_keys.get(&k).ok_or_else(|| err(&format!("predictor session lacks {k}K")))?;
        let dominant = || self.dominant(f);
        // Session._historical_cold at `seconds`
        let historical = |seconds: f64| -> f64 {
            if s.warm {
                return 0.;
            }
            let d = 0f64.max(10f64.min(seconds / 60.));
            let over = |tau: f64| if d > 0.001 { -(-d / tau).exp_m1() * tau / d } else { 1. };
            let mean = over(mode.tau);
            let session = st.session * over(st.session_tau) + st.fatigue;
            if mode.log_rate {
                let mut loss = 0.;
                for (i, &w) in weights.iter().enumerate() {
                    if w != 0. {
                        loss += w * st.penalty[i];
                    }
                }
                self.rate_loss(f, chart, b, rate, loss * mean + session, &weights)
            } else {
                st.penalty[dominant()] * mean + session
            }
        };
        let seconds = f.length / 1000. / rate.max(0.01);
        let cold = historical(seconds);
        let mut correction = st.form[0];
        let total: f64 = weights.iter().sum();
        for (i, &w) in weights.iter().enumerate() {
            correction += if total > 0. { w / total } else { 1. / weights.len() as f64 } * st.form[i + 1];
        }
        let shift = correction + cold + self.played(f, chart, rate, st, s, sdm * sdm);
        mu += shift;
        let (a, bb) = self.shown;
        let shown = |y: f64| 0f64.max(1f64.min(1. - (20f64.min(a + bb * y)).exp()));
        let spread = sda.hypot(sdm);
        let opening = historical(0.);
        let activation = if s.warm {
            10.
        } else if !mode.log_rate {
            st.activation[dominant()]
        } else {
            weights.iter().zip(&st.activation).filter(|(w, _)| **w != 0.).map(|(w, v)| w * v).sum()
        };
        Ok(Prediction {
            mu,
            base_mu,
            shift,
            cold,
            sdm,
            sda,
            sd: (sda * sda + sdm * sdm).sqrt(),
            acc_mid: shown(mu),
            acc_lo: shown(mu + BAND_Z * spread),
            acc_hi: shown(mu - BAND_Z * spread),
            opening_acc: shown(mu + 0f64.max(opening - cold)),
            activation,
            slope,
            tau: mode.tau,
        })
    }
}

/// nps.profile: skill shares and structure, unit length.
pub fn profile_of(f: &Dict, names: &[String]) -> Vec<f64> {
    let num = |d: Option<&Dict>, k: &str, default: f64| d.and_then(|d| d.get(k)).map_or(default, Val::f);
    let (sk, st) = (f.dict("sk"), f.dict("nps"));
    let mut v: Vec<f64> = names.iter().map(|k| num(sk, k, 0.)).collect();
    v.extend([0.65 * 2f64.min(num(st, "chord", 1.) - 1.), num(st, "wide", 0.), num(st, "repeat", 0.), num(st, "overlap", 0.),
              0.5 * 2f64.min(num(st, "burst", 1.)), 0.6 * num(st, "active", 0.5), 0.35 * 2f64.min(num(st, "rhythm", 0.))]);
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(0.001);
    v.into_iter().map(|x| x / norm).collect()
}

/// str(value) of a stored identity.
fn text(v: &Val) -> String {
    match v {
        Val::Str(s) => s.clone(),
        Val::Int(i) => i.to_string(),
        Val::F(x) => x.to_string(),
        other => format!("{other:?}"),
    }
}

/// Measured rates, public map effects and the player's level.
#[cfg(test)]
mod model_tests {
    use super::*;

    fn feats(keys: i64, overall: f64, skills: serde_json::Value, c: &Consts) -> Feats {
        let mut f = Feats::skills(&skills, c);
        (f.keys, f.overall, f.od) = (keys, overall, 8.);
        f
    }

    fn model(pop: Pop) -> PredModel {
        let c = Arc::new(Consts::standard());
        PredModel { c, pop: Arc::new(Pop { tau2: 0.1, kappa: 3., knee: 4., ..pop }), levels: HashMap::default(), level: [(7, (-3., 0.01, 0.2))].into_iter().collect(), glob: -3., sd_all: 0.2,
                    shape: HashMap::default(), recent: HashMap::default(), aff: [("known".to_string(), (-0.6, 10.))].into_iter().collect(), rate_aff: HashMap::default(),
                    rate_support: [("known".to_string(), (1., 1.))].into_iter().collect(), rate_bandwidth: ByKeys { map: [(0, 0.18), (7, 0.12)].into_iter().collect() }, tau2_c: 0.12, kap_c: 2.,
                    shown: (0.75f64.ln(), 1.), retry: 0., sd_att: [(7, 0.2)].into_iter().collect(), slopes: Arc::new(Slopes::default()), modes: ByKeys::default(), fallback: Mode::default() }
    }

    fn plain(m: &PredModel) -> Feats {
        Feats::bare(7, 5., 8., &m.c)
    }

    #[test]
    fn exact_rate_affinity_stays_intact_far_rate_weakens_and_uncertainty_grows() {
        let mut m = model(Pop::default());
        let f = plain(&m);
        let (exact, fast, slow) = (m.personal(&f, "known", None, 1.), m.personal(&f, "known", None, 1.5), m.personal(&f, "known", None, 0.75));
        assert!((exact.0 + 3.6).abs() < 1e-12);
        assert!(exact.0 < slow.0 && slow.0 < fast.0 && fast.0 < -3.);
        assert!(exact.1 < slow.1 && slow.1 < fast.1);
        // A bad high-rate history must also fade at a much easier rate.
        m.aff.insert("known".into(), (0.6, 10.));
        assert!(m.personal(&f, "known", None, 0.75).0 < m.personal(&f, "known", None, 1.).0);
    }

    #[test]
    fn interpolation_inside_measured_rate_range_is_preserved() {
        let mut m = model(Pop::default());
        m.rate_support.insert("known".into(), (0.75, 1.5));
        m.rate_aff.insert("known".into(), (vec![0.75f64.ln(), 1.5f64.ln()], vec![-0.7, -0.3], vec![5., 5.]));
        let f = plain(&m);
        assert!((m.personal(&f, "known", None, (0.75f64 * 1.5).sqrt()).0 + 3.5).abs() < 1e-12);
    }

    #[test]
    fn bad_lower_rate_result_does_not_disappear_on_uprates() {
        // the map's base rises 2·log(rate/.78)
        let mut pop = Pop::default();
        pop.mbr.insert(1, vec![(0.78f64.ln(), 0., 300.), (2f64.ln(), 2. * (2. / 0.78f64).ln(), 300.)]);
        let mut m = model(pop);
        m.aff.insert("known".into(), (0.6, 10.));
        m.rate_support.insert("known".into(), (0.78, 0.78));
        m.rate_bandwidth.map.insert(7, 0.08);
        let f = plain(&m);
        let p: Vec<(f64, f64, f64)> = [0.78, 0.79, 0.9, 1.].iter().map(|r| m.personal(&f, "known", Some(1), *r)).collect();
        assert!(p.windows(2).all(|w| w[1].0 > w[0].0 && w[1].1 > w[0].1));
        assert_eq!(m.chart_affinity("known", 0.9, 7).0, 0.6);
        let below = m.chart_affinity("known", 0.7, 7).0;
        assert!(0. < below && below < 0.6);
        assert_eq!(m.chart_affinity("unplayed", 0.9, 7), (0., 0.));
    }

    /// Session tables: warm, one chart played with `error` against its forecast.
    fn played_session(error: f64, reset: bool) -> PredSession {
        use ms_data::store::Event;
        let c = Arc::new(Consts::standard());
        let start = serde_json::json!({"keys": 7, "skill": "chordstream", "length": 120, "mu": -3., "base_mu": -3., "sd": 0.3, "rate": 1.});
        let event = |id, t, kind: &str, info| Arc::new(Event { id, t, kind: kind.to_string(), beatmap: (kind != "reset").then(|| "md5:a".to_string()), info });
        let mut ev = vec![event(1, 1000., "start", start), event(2, 1120., "finish", serde_json::json!({"start_id": 1, "y": -3. + error, "key": "a"}))];
        if reset {
            ev.push(event(3, 1121., "reset", serde_json::json!({})));
        }
        let prior = crate::warmup::FormPrior { global_variance: 0.01, skill_variance: 0.01, by_key: HashMap::default(), noise_variance: 0.04 };
        let model = crate::warmup::Model::new(&crate::personal::WarmModels::default(), Slopes::default(), prior, c.families.len());
        let n = c.families.len();
        crate::session::Session::new(ev, 1122., Some(Arc::new(model)), c, Arc::new(vec![vec![1.; n]; n])).tables(&[7])
    }

    #[test]
    fn a_completed_error_updates_its_chart_without_claiming_every_chart_is_wrong() {
        let mut m = model(Pop::default());
        m.rate_bandwidth.map.insert(0, 0.15);
        m.rate_bandwidth.map.insert(7, 0.15);
        let f = plain(&m);
        let s = played_session(1., false);
        let shift = |chart: &str, rate: f64| m.predict(&f, chart, None, rate, &s).unwrap().shift;
        let (same, other) = (shift("a", 1.), shift("different", 1.));
        assert!(same - other > 0.4, "{same} {other}");
        assert!(shift("a", 1.5) - other < same - other);
        let reset = played_session(1., true);
        assert_eq!(m.predict(&f, "a", None, 1., &reset).unwrap().shift, m.predict(&f, "different", None, 1., &reset).unwrap().shift);
    }

    #[test]
    fn the_central_prediction_is_not_risk_shifted() {
        let mut m = model(Pop::default());
        m.aff.clear();
        let f = plain(&m);
        let n = m.c.families.len();
        let state = KeyState { form: vec![0.; n + 1], penalty: vec![0.; n], session: 0., session_tau: 8., fatigue: 0., activation: vec![0.; n] };
        let s = PredSession { warm: true, by_keys: [(7, state)].into_iter().collect(), charts: HashMap::default() };
        let p = m.predict(&f, "m", None, 1., &s).unwrap();
        assert!((p.acc_mid - (1. - 0.75 * p.mu.exp())).abs() < 1e-12);
        assert!(p.acc_lo < p.acc_mid && p.acc_mid < p.acc_hi);
    }

    #[test]
    fn the_recent_family_offset_adds_a_chordjack_led_hinge_only_above_point_seven() {
        let mut m = model(Pop::default());
        let n = m.c.recent_idx.len() + 1;
        let mut beta = vec![0.; n];
        beta[n - 1] = 0.3;
        m.recent.insert(7, (vec![0.; n], vec![1.; n], beta));
        let f = |keys, share: f64| feats(keys, 5., serde_json::json!({"chordjack": share}), &m.c);
        let base = m.personal(&plain(&m), "x", None, 1.).0;
        assert!((m.personal(&f(7, 0.9), "x", None, 1.).0 - base - 0.3 * 0.2).abs() < 1e-12);
        assert_eq!(m.personal(&f(7, 0.6), "x", None, 1.).0, base);
    }

    #[test]
    fn personal_geometry_counts_width_and_overlap_once() {
        let m = model(Pop::default());
        let mut f = plain(&m);
        f.z_tail = [0., 1., 0.8, 2.3 / 7., 0.37];
        let g = m.c.geometry.len();
        f.geometry = Some(vec![0.; g]);
        let z = m.c.zvec(&f, -3.);
        let count = |x: f64| z.iter().filter(|v| (*v - x).abs() < 1e-12).count();
        assert_eq!((count(2.3 / 7.), count(0.37)), (1, 1));
        assert_eq!(z.len(), m.c.names.len() + 5 + 1 + g + 1);
        let bare = Feats { geometry: None, ..f.clone() };
        assert_eq!(z[z.len() - 1], m.c.zvec(&bare, -3.)[m.c.zvec(&bare, -3.).len() - 1]);
    }

    #[test]
    fn readiness_uses_the_same_execution_family_as_historical_fitting() {
        let c = Consts::standard();
        let f = Feats::skills(&serde_json::json!({"technical": 1., "mash": 0.95, "chordjack": 0.8}), &c);
        assert_eq!(c.dominant(&f), c.family("chordjack"));
    }

    #[test]
    fn smooth_public_rate_effect() {
        let mut pop = Pop { slope: ByKeys { map: [(0, 3.)].into_iter().collect() }, window: ByKeys { map: [(0, -0.9)].into_iter().collect() }, ..Pop::default() };
        pop.mb.insert(1, (0.1, 50.));
        pop.mbr.insert(1, vec![(0.75f64.ln(), 0.3, 10.), (0., 0.6, 20.), (1.5f64.ln(), 0.2, 10.)]);
        let m = model(pop);
        let f = plain(&m);
        let a = m.base_of(&f, Some(1), 1., None).0;
        assert!((a - m.base_of(&f, Some(1), 0.9999, None).0).abs() < 0.001 && (a - m.base_of(&f, Some(1), 1.0001, None).0).abs() < 0.001);
    }

    #[test]
    fn jack_and_ln_follow_the_players_level_not_only_the_stars() {
        // Jack maps cost more once they pass the player's level; LN costs less near it (user 2026-10-01).
        let part = |c: &Consts, name: &str| c.gap_part_names.iter().position(|p| p == name).unwrap();
        let c = Consts::standard();
        let (jack_part, ln_part) = (part(&c, "jack"), part(&c, "ln"));
        let curves = vec![(jack_part, vec![-0.4, 0., 0.2, 0.4, 0.2]), (ln_part, vec![0.2, 0., -0.2, -0.2, 0.])];
        let with = |gap: Vec<(i64, Vec<(usize, Vec<f64>)>)>, means: Vec<(i64, Vec<f64>)>| model(Pop {
            slope: ByKeys { map: [(0, 2.)].into_iter().collect() }, window: ByKeys { map: [(0, 0.)].into_iter().collect() },
            gap: ByKeys { map: gap.into_iter().collect() }, share_mean: ByKeys { map: means.into_iter().collect() }, ..Pop::default() });
        let m = with(vec![(4, curves.clone())], Vec::new());
        let shift = |m: &PredModel, f: &Feats, level: f64| m.base_of(f, None, 1., Some(level)).0 - m.base_of(f, None, 1., None).0;
        let at = 5f64.ln();
        let jack = feats(4, 5., serde_json::json!({"jackspeed": 1.}), &m.c);
        let ln = feats(4, 5., serde_json::json!({"ln": 1.}), &m.c);
        assert!(shift(&m, &jack, at + 0.3) < 0. && 0. < shift(&m, &jack, at - 0.15), "easy for this player: cheap; above them: costly");
        assert!(shift(&m, &ln, at) < 0. && 0. < shift(&m, &ln, at + 0.3), "LN at their level: cheaper; far below: still costs");
        let six = feats(6, 5., serde_json::json!({"jackspeed": 1.}), &m.c);
        assert_eq!(shift(&m, &six, at - 0.15), 0., "6K borrows 7K's curves: none here");
        let borrowed = with(vec![(4, curves.clone()), (7, curves.clone())], Vec::new());
        assert!(shift(&borrowed, &six, at - 0.15) > 0.);
        let nine = feats(9, 5., serde_json::json!({"jackspeed": 1.}), &m.c);
        assert_eq!(shift(&borrowed, &nine, at - 0.15), 0., "8K–10K keep none (DEV worse with 7K's)");
        // Centred parts describe "more than this keymode's usual mix": a chart at the mean share gets nothing.
        let chords = part(&c, "chords");
        let mut means = vec![0.; c.gap_part_names.len()];
        means[chords] = 0.5;
        let centred = with(vec![(7, vec![(chords, vec![0.3; 5])])], vec![(7, means)]);
        let of = |share: f64| shift(&centred, &feats(7, 5., serde_json::json!({"chordstream": share}), &centred.c), at);
        assert!(of(0.5) == 0. && of(1.) > 0. && 0. > of(0.));
    }
}

/// Skill membership.
#[cfg(test)]
mod skill_tests {
    use super::*;
    use crate::skills::{demanded, dimensions, feat_trees, normalize, shares, tree, wants_vibro};

    fn feature(keys: i64, sk: &[(&str, f64)]) -> Dict {
        let mut d = Dict::new();
        sk.iter().for_each(|(k, v)| d.set(k, *v));
        let mut f = Dict::new();
        f.set("keys", keys);
        f.set("sk", d);
        f
    }

    fn strength(f: &Dict, selected: &[&str]) -> f64 {
        let dims = &feat_trees()[usize::from(f.f("keys") as i64 != 4)];
        let t = Tree { dims: dims.iter().map(|d| d.0.clone()).collect(), members: dims.iter().map(|d| d.1.clone()).collect(),
                       tech: ["technical", "technical/patterntech", "technical/rhythmtech"].iter().filter_map(|k| dims.iter().position(|d| d.0 == *k).map(|i| (i, k.rsplit('/').next().unwrap().to_string()))).collect() };
        let selected: Vec<String> = normalize(&selected.iter().map(|s| s.to_string()).collect::<Vec<_>>());
        SkillValues::of(f, &t).strength(&t.select(&selected, &crate::skills::Tuning::default()))
    }

    fn strings(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    fn tuned(f: &Dict, selected: &[&str], tuning: &crate::skills::Tuning) -> f64 {
        let dims = &feat_trees()[usize::from(f.f("keys") as i64 != 4)];
        let t = Tree { dims: dims.iter().map(|d| d.0.clone()).collect(), members: dims.iter().map(|d| d.1.clone()).collect(), tech: Vec::new() };
        SkillValues::of(f, &t).strength(&t.select(&normalize(&strings(selected)), tuning))
    }

    #[test]
    fn avoid_steps_nest_and_the_middle_is_the_old_filter() {
        let (lead, some, little) = (feature(7, &[("chordstream", 1.), ("sv", 0.95)]), feature(7, &[("chordstream", 1.), ("sv", 0.7)]), feature(7, &[("chordstream", 1.), ("sv", 0.35)]));
        let hold = |h: f64| crate::skills::Tuning { avoid: [("sv".to_string(), h)].into(), other: 0. };
        let kept = |h: f64| [&lead, &some, &little].map(|f| tuned(f, &["!sv"], &hold(h)) > 0.);
        assert_eq!(kept(0.9), [false, true, true]);
        assert_eq!(kept(0.6), [false, false, true]);
        assert_eq!(kept(0.3), [false, false, false]);
        for f in [&lead, &some, &little] {
            assert_eq!(tuned(f, &["!sv"], &crate::skills::Tuning::default()), strength(f, &["!sv"]));
            assert_eq!(tuned(f, &["!sv"], &hold(0.6)), strength(f, &["!sv"]));
        }
    }

    #[test]
    fn other_maps_stay_only_when_the_slider_is_on() {
        let f = feature(7, &[("chordstream", 1.), ("sv", 0.7)]);
        let other = crate::skills::Tuning { other: 1., ..Default::default() };
        assert_eq!(tuned(&f, &["chordjack"], &crate::skills::Tuning::default()), 0.);
        assert_eq!(tuned(&f, &["chordjack"], &other), 1.);
        assert_eq!(tuned(&f, &["chordjack", "!sv"], &other), 0., "avoided maps stay out");
    }

    #[test]
    fn a_blacklist_is_material_and_works_without_positive_choices() {
        let f = feature(7, &[("chordstream", 1.), ("sv", 0.05)]);
        assert_eq!(strength(&f, &["!sv"]), 1.);
        assert!(strength(&f, &["chordstream", "!sv"]) > 0.);
        let hard = feature(7, &[("chordstream", 1.), ("sv", 0.8)]);
        assert_eq!(strength(&hard, &["!sv"]), 0.);
        assert_eq!(strength(&hard, &["chordstream", "!sv"]), 0.);
        assert_eq!(strength(&feature(7, &[("vibro", 1.)]), &["!jackspeed"]), 0.);
        assert!(!wants_vibro(&strings(&["jackspeed"])));
    }

    #[test]
    fn restored_parent_exceptions_agree_with_the_visible_children() {
        let selected = normalize(&strings(&["!ln", "ln/inverse"]));
        assert!(!selected.contains(&"!ln".to_string()));
        assert_eq!(crate::skills::state(&selected, "ln/inverse"), 1);
        assert_eq!(crate::skills::state(&selected, "ln/release"), -1);
        let choice: Vec<&str> = selected.iter().map(String::as_str).collect();
        assert!(strength(&feature(7, &[("ln", 1.), ("inverse", 0.9)]), &choice) > 0.);
        assert_eq!(normalize(&selected), selected);
    }

    #[test]
    fn trills_are_independent_and_old_choices_migrate() {
        let keys = |k: i64| tree(k).iter().map(|s| s.key.clone()).collect::<Vec<_>>();
        assert!(keys(4).contains(&"jumptrill".to_string()) && keys(4).contains(&"splittrill".to_string()) && keys(7).contains(&"jumptrill".to_string()));
        assert_eq!(normalize(&strings(&["!chordstream/jumptrill"])), strings(&["!jumptrill"]));
        assert_eq!(normalize(&strings(&["jumpstream/splittrill"])), strings(&["splittrill"]));
        let f = feature(7, &[("jumptrill", 1.)]);
        assert_eq!(strength(&f, &["jumptrill"]), 1.);
        assert_eq!((strength(&f, &["chordstream"]), strength(&f, &["splittrill"])), (0., 0.));
    }

    #[test]
    fn technical_practice_requires_presence_not_only_a_local_peak() {
        let with = |technical: f64, pattern: f64| {
            let mut f = feature(7, &[("chordstream", 1.), ("technical", 0.85), ("patterntech", 0.8)]);
            let mut profile = Dict::new();
            profile.set("technical", technical);
            profile.set("patterntech", pattern);
            profile.set("rhythmtech", 0.02);
            f.set("tech_profile", profile);
            f
        };
        let weak = with(0.1, 0.08);
        assert_eq!(strength(&weak, &["technical"]), 0.);
        assert_eq!(strength(&weak, &["technical/patterntech"]), 0.);
        assert_eq!(strength(&weak, &["!technical"]), 0., "a material forbidden passage stays forbidden");
        let present = with(0.85, 0.72);
        assert!(strength(&present, &["technical"]) > 0.8);
        assert!(strength(&present, &["technical/patterntech"]) >= 0.8);
    }

    #[test]
    fn keymode_taxonomy() {
        let names = |k: i64| tree(k).iter().map(|s| s.name).collect::<Vec<_>>();
        let (four, wide) = (names(4), names(7));
        for n in ["Stream", "Jumpstream", "Handstream", "Jackspeed", "Chordjack"] {
            assert!(four.contains(&n), "{n}");
        }
        for n in ["Delay", "Chordstream", "Bracket", "Jackspeed", "Chordjack"] {
            assert!(wide.contains(&n), "{n}");
        }
        assert!(!["Chords", "Jacks", "Trills"].iter().any(|n| four.contains(n) || wide.contains(n)));
        assert!(wants_vibro(&strings(&["jackspeed/vibro"])) && !wants_vibro(&strings(&["jackspeed"])));
        assert!(!normalize(&strings(&["trill"])).is_empty() && !normalize(&strings(&["trill"])).contains(&"trill".to_string()));
    }

    #[test]
    fn broad_membership_union_not_incidental_traces() {
        let f = feature(7, &[("ln", 1.), ("release", 0.9), ("chordjack", 0.85), ("sv", 0.15)]);
        assert!(strength(&f, &["ln", "sv"]) > 0.);
        assert_eq!(strength(&f, &["sv"]), 0.);
        assert!(strength(&f, &["chordjack"]) > 0.8);
        assert_eq!(strength(&f, &[]), 1.);
        assert_eq!(normalize(&strings(&["ln", "ln", "sv", "release", "nonsense"])), ["ln", "sv"]);
    }

    #[test]
    fn jacks_related_but_not_same_as_chordjacks() {
        for leaf in ["jack", "jackspeed", "minijack", "longjack"] {
            let f = feature(7, &[(leaf, 1.)]);
            assert!(strength(&f, &["jack"]) == 1. && strength(&f, &["chordjack"]) == 0., "{leaf}");
        }
        let cj = feature(7, &[("chordjack", 1.)]);
        // The explicit Jackspeed family is no longer the vague umbrella that silently included changing chords.
        assert!(strength(&cj, &["jackspeed"]) == 0. && strength(&cj, &["chordjack"]) == 1.);
        for leaf in ["jumptrill", "trill1h", "bracket"] {
            assert_eq!(strength(&feature(7, &[(leaf, 1.)]), &["trill"]), 1., "{leaf}");
        }
        assert_eq!(strength(&feature(4, &[("splittrill", 1.)]), &["jumpstream/splittrill"]), 1.);
        assert_eq!(strength(&feature(7, &[("splittrill", 1.)]), &["jumpstream/splittrill"]), 0.);
        assert_eq!(strength(&feature(7, &[("delay", 1.)]), &["technical"]), 0.);
    }

    #[test]
    fn demanded_is_match_for_every_single_skill() {
        // Stats reads group membership from one pass; it must equal match(f, [g]) per group.
        for f in [feature(7, &[("ln", 0.9), ("stream", 0.3)]), feature(4, &[("jackspeed", 0.8), ("chordjack", 0.7)]), feature(7, &[("chordstream", 0.5), ("bracket", 0.48), ("sv", 0.2)])] {
            let dem = demanded(&f);
            assert!(!dem.is_empty());
            for g in dimensions(f.f("keys") as i64) {
                assert_eq!(strength(&f, &[g.key.as_str()]) > 0., dem.contains(&g.key.as_str()), "{}", g.key);
            }
        }
    }

    #[test]
    fn skill_shares() {
        let w = |v: &[(&str, f64)]| v.iter().map(|(k, x)| (k.to_string(), *x)).collect::<std::collections::HashMap<_, _>>();
        assert_eq!(shares(&strings(&["delay", "jackspeed", "!sv"]), &w(&[("delay", 2.)]), 0.), Some(w(&[("delay", 2.), ("jackspeed", 1.)])));
        assert_eq!(shares(&strings(&["delay", "jackspeed"]), &w(&[]), 0.), None);
        assert_eq!(shares(&strings(&["delay"]), &w(&[("delay", 3.)]), 0.), None);
    }
}

#[cfg(test)]
mod target_tests {
    use super::*;

    fn ctx(target: f64, warm: bool) -> Ctx {
        Ctx { target, warm, skills: false, wants_vibro: false, preference: false, version: 1 }
    }

    fn prediction(acc: f64, opening: f64) -> Prediction {
        Prediction { acc_mid: acc, opening_acc: opening, tau: 6., ..Prediction::default() }
    }

    #[test]
    fn jacks_get_no_separate_target_and_the_band_narrows_near_100() {
        assert!((ctx(0.94, false).central() - 0.94).abs() < 1e-9, "jacks get no separate target");
        let c = ctx(0.97, false);
        assert!((c.central() - 0.97).abs() < 1e-9);
        assert!(c.band().1 - c.band().0 < 0.02);
        let top = ctx(0.99, false);
        assert_eq!(top.central(), MAX_TARGET);
        assert!(top.band().1 <= MAX_HIGH);
    }

    #[test]
    fn a_cold_opening_cannot_borrow_warmth_from_the_same_maps_ending() {
        let c = ctx(0.94, false);
        assert!(!c.eligible(&prediction(0.944, 0.91)));
        assert!(c.eligible(&prediction(0.946, 0.94)));
        assert!(ctx(0.94, true).eligible(&prediction(0.944, 0.91)));
    }

    #[test]
    fn the_rate_search_moves_toward_an_eligible_cold_opening() {
        let p = prediction(0.94, 0.91);
        assert!(ctx(0.94, false).rate_shift(&p, 0.94) < 0.);
        assert_eq!(ctx(0.94, true).rate_shift(&p, 0.94), 0.);
    }
}
