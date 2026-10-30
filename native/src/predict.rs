//! Bulk accuracy predictor: recommend.Predictor._predict for many (chart, rate) at once.
//!
//! Not part of the calculator identity (native_backend.identity_files leaves this file out): a predictor
//! change must not re-analyse every library. Python owns the fit and every constant; it exports the fitted
//! Personal model once per fit (recommend.predictor_model) and the frozen session tables once per builder
//! request (recommend.predictor_session). Charts with plays in the session window keep the Python path
//! (rate_penalty, chart_correction, retry read their attempts).
use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};
use crate::background::background_map;
use crate::choices::{Choice, Choices, Pred, Proposal};
use std::sync::Arc;
use std::collections::HashMap as StdMap;
use std::hash::BuildHasherDefault;

/// FxHash (rustc's): SipHash was a fifth of a prediction; these maps are built from our own export.
#[derive(Default, Clone, Copy)]
pub(crate) struct Fx(u64);

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

pub(crate) type HashMap<K, V> = StdMap<K, V, BuildHasherDefault<Fx>>;

pub(crate) fn err(m: &str) -> PyErr {
    PyValueError::new_err(m.to_string())
}

pub(crate) fn get<'py>(d: &Bound<'py, PyDict>, k: &str) -> PyResult<Bound<'py, PyAny>> {
    d.get_item(k)?.ok_or_else(|| err(&format!("predictor export lacks {k}")))
}

/// A feature dict's entry by interned key (a fresh key string per lookup cost more than the prediction).
fn at<'py>(d: &Bound<'py, PyDict>, k: &Bound<'py, PyString>) -> PyResult<Bound<'py, PyAny>> {
    d.get_item(k)?.ok_or_else(|| err(&format!("features lack {k}")))
}

fn fl(d: &Bound<'_, PyDict>, k: &str) -> PyResult<f64> {
    get(d, k)?.extract()
}

/// {int keys: float} with the Python `.get(k, d[0])` fallback kept apart.
struct ByKeys<T> {
    map: HashMap<i64, T>,
}

impl<T: Clone> ByKeys<T> {
    fn new(o: &Bound<'_, PyAny>, f: impl Fn(&Bound<'_, PyAny>) -> PyResult<T>) -> PyResult<Self> {
        let mut map = HashMap::default();
        for (k, v) in o.cast::<PyDict>()?.iter() {
            map.insert(k.extract::<i64>()?, f(&v)?);
        }
        Ok(ByKeys { map })
    }
    fn get(&self, k: i64) -> Option<&T> {
        self.map.get(&k)
    }
}

fn floats(o: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    o.extract()
}

fn num(o: &Bound<'_, PyAny>) -> PyResult<f64> {
    o.extract()
}

/// recommend._interp: np.interp for one scalar on a short sorted list.
pub(crate) fn interp(x: f64, xs: &[f64], ys: &[f64]) -> f64 {
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

struct Pop {
    slope: ByKeys<f64>,
    window: ByKeys<f64>,
    curve: ByKeys<f64>,
    knee: f64,
    mb: HashMap<i64, (f64, f64)>,
    mbr: HashMap<i64, Vec<(f64, f64, f64)>>, // b → [(log rate, effect, n)] in VARIANTS order
    tau2: f64,
    kappa: f64,
    gap: ByKeys<Vec<(usize, Vec<f64>)>>, // keys → [(GAP_SHARE_PARTS index, coefficients)]
    share_mean: ByKeys<Vec<f64>>,         // keys → mean share per part
}

impl Pop {
    fn new(d: &Bound<'_, PyDict>) -> PyResult<Pop> {
        let mut mb = HashMap::default();
        for (k, v) in get(d, "mb")?.cast::<PyDict>()?.iter() {
            mb.insert(k.extract()?, v.extract()?);
        }
        let mut mbr = HashMap::default();
        for (k, v) in get(d, "mbr")?.cast::<PyDict>()?.iter() {
            mbr.insert(k.extract()?, v.extract()?);
        }
        Ok(Pop {
            slope: ByKeys::new(&get(d, "slope")?, num)?,
            window: ByKeys::new(&get(d, "window")?, num)?,
            curve: ByKeys::new(&get(d, "curve")?, num)?,
            knee: fl(d, "curve_knee")?,
            mb,
            mbr,
            tau2: fl(d, "tau2")?,
            kappa: fl(d, "kappa")?,
            gap: ByKeys::new(&get(d, "gap")?, |v| v.extract())?,
            share_mean: ByKeys::new(&get(d, "share_mean")?, floats)?,
        })
    }
    fn slope(&self, k: i64) -> f64 {
        *self.slope.get(k).or(self.slope.get(0)).unwrap_or(&0.)
    }
    fn curve(&self, k: i64) -> f64 {
        *self.curve.get(k).or(self.curve.get(0)).unwrap_or(&0.)
    }
}

/// warmup.Model parameters of one keymode.
#[derive(Clone)]
struct Mode {
    tau: f64,
    log_rate: bool,
}

fn mode(o: &Bound<'_, PyAny>) -> PyResult<Mode> {
    let d = o.cast::<PyDict>()?;
    Ok(Mode {
        tau: fl(d, "tau")?,
        log_rate: d.get_item("units")?.map(|u| u.extract::<String>().map(|s| s == "log_rate")).transpose()?.unwrap_or(false),
    })
}

/// warmup.score_rate_slope's inputs: the population's slope/curve/knee and the learned rate elasticities.
pub(crate) struct Slopes {
    slope: ByKeys<f64>,
    curve: ByKeys<f64>,
    knee: f64,
    elasticities: HashMap<i64, (f64, Vec<Option<f64>>)>, // keys → (keymode response, per family)
}

impl Slopes {
    /// {"slope", "curve", "curve_knee", "elasticities"}; `fams` = warmup.FAMILIES.
    pub(crate) fn of(w: &Bound<'_, PyDict>, fams: &[String]) -> PyResult<Slopes> {
        // warmup.rate_elasticities keys: "<keys>" and "<keys>:<family>"
        let flat: HashMap<String, f64> = get(w, "elasticities")?.extract()?;
        let mut elasticities: HashMap<i64, (f64, Vec<Option<f64>>)> = HashMap::default();
        for (key, &v) in &flat {
            let (k, fam) = key.split_once(':').map_or((key.as_str(), None), |(k, f)| (k, Some(f)));
            let Ok(k) = k.parse::<i64>() else { continue };
            let e = elasticities.entry(k).or_insert((f64::NAN, vec![None; fams.len()]));
            match fam {
                None => e.0 = v,
                Some(f) => if let Some(i) = fams.iter().position(|x| x == f) { e.1[i] = Some(v) },
            }
        }
        Ok(Slopes { slope: ByKeys::new(&get(w, "slope")?, num)?, curve: ByKeys::new(&get(w, "curve")?, num)?,
                    knee: fl(w, "curve_knee")?, elasticities })
    }

    /// warmup.rate_elasticity: the family-weight blend of measured chart-rate responses.
    pub(crate) fn elasticity(&self, f: &Feats, weights: &[f64]) -> f64 {
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
    pub(crate) fn physical(&self, f: &Feats, weights: &[f64]) -> f64 {
        let k = f.keys;
        let mut slope = *self.slope.get(k).or(self.slope.get(0)).unwrap_or(&2.);
        let curve = *self.curve.get(k).or(self.curve.get(0)).unwrap_or(&0.);
        slope += 2. * curve * (f.overall.max(0.05) / self.knee).ln().max(0.);
        (slope * self.elasticity(f, weights)).clamp(1., 8.)
    }
}

/// Constants Python owns: skill names, warmup families, gap parts and knots.
pub(crate) struct Consts {
    names: Vec<String>,
    pub(crate) families: Vec<String>,    // warmup.FAMILIES
    pub(crate) members: Vec<(String, Vec<String>)>, // warmup.MEMBERS in order
    gap_parts: Vec<Vec<String>>,         // recdata.GAP_SHARE_PARTS members; empty = "all"
    gap_knots: Vec<f64>,
    gap_shared: i64,
    gap_borrow: Vec<i64>,
    recent_families: Vec<String>,
    geometry: Vec<usize>,
    dominant_skip: Vec<String>,
    // derived: skill name → index into names, and per-name lookups
    pub(crate) idx: HashMap<String, usize>,
    fam_of: Vec<usize>,
    skip: Vec<bool>,
    member_idx: Vec<(usize, Vec<usize>)>, // family index → member skill indices (family_weights; no technical)
    gap_idx: Vec<Vec<usize>>,
    recent_idx: Vec<usize>,
}

impl Consts {
    pub(crate) fn of(cs: &Bound<'_, PyDict>) -> PyResult<Consts> {
        Consts {
            names: get(cs, "names")?.extract()?,
            families: get(cs, "families")?.extract()?,
            members: get(cs, "members")?.extract()?,
            gap_parts: get(cs, "gap_parts")?.extract()?,
            gap_knots: get(cs, "gap_knots")?.extract()?,
            gap_shared: get(cs, "gap_shared")?.extract()?,
            gap_borrow: get(cs, "gap_borrow")?.extract()?,
            recent_families: get(cs, "recent_families")?.extract()?,
            geometry: get(cs, "geometry")?.extract()?,
            dominant_skip: get(cs, "dominant_skip")?.extract()?,
            idx: HashMap::default(),
            fam_of: Vec::new(),
            skip: Vec::new(),
            member_idx: Vec::new(),
            gap_idx: Vec::new(),
            recent_idx: Vec::new(),
        }
        .derive()
    }

    /// warmup.family(skill) as a FAMILIES index.
    fn family(&self, skill: &str) -> usize {
        let name = self.members.iter().find(|(_, m)| m.iter().any(|x| x == skill)).map(|(k, _)| k.as_str())
            .unwrap_or(if self.families.iter().any(|x| x == skill) { skill } else { "rice" });
        self.families.iter().position(|x| x == name).unwrap()
    }

    /// warmup.dominant(f)
    pub(crate) fn dominant(&self, f: &Feats) -> usize {
        let mut best: Option<(usize, f64)> = None;
        for &(v, fam, descriptive) in &f.order {
            if !descriptive && best.is_none_or(|b| v > b.1) {
                best = Some((fam, v));
            }
        }
        best.map_or_else(|| self.family("rice"), |b| b.0)
    }

    /// recommend.zvec
    pub(crate) fn zvec(&self, f: &Feats, mu: f64) -> Vec<f64> {
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
    pub(crate) fn family_weights(&self, f: &Feats) -> Vec<f64> {
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
    pub(crate) fn recent_vec(&self, f: &Feats) -> Vec<f64> {
        let mut v: Vec<f64> = self.recent_idx.iter().map(|&i| f.sk[i]).collect();
        v.push((v[v.len() - 1] - 0.7).max(0.));
        v
    }

    fn derive(mut self) -> PyResult<Consts> {
        self.idx = self.names.iter().enumerate().map(|(i, n)| (n.clone(), i)).collect();
        let at = |c: &Consts, n: &str| c.idx.get(n).copied().ok_or_else(|| err(&format!("{n} is not a skill name")));
        self.fam_of = self.names.iter().map(|n| self.family(n)).collect();
        self.skip = self.names.iter().map(|n| self.dominant_skip.contains(n)).collect();
        let mut member_idx = Vec::new();
        for (name, members) in &self.members {
            if name != "technical" {
                let fam = self.families.iter().position(|x| x == name).ok_or_else(|| err(name))?;
                member_idx.push((fam, members.iter().map(|m| at(&self, m)).collect::<PyResult<Vec<_>>>()?));
            }
        }
        self.member_idx = member_idx;
        self.gap_idx = self.gap_parts.iter().map(|m| m.iter().map(|x| at(&self, x)).collect()).collect::<PyResult<_>>()?;
        self.recent_idx = self.recent_families.iter().map(|x| at(&self, x)).collect::<PyResult<_>>()?;
        Ok(self)
    }
}

#[pyclass(module = "maniascope_native")]
pub struct PredModel {
    c: Consts,
    consts_key: String, // repr of the consts export: a FeatStore's Feats were read with these
    pop: Pop,
    levels: HashMap<i64, f64>,
    level: HashMap<i64, (f64, f64, f64)>, // keys → (level, level var, attempt sd)
    glob: f64,
    sd_all: f64,
    shape: HashMap<i64, (Vec<f64>, Vec<f64>, Vec<f64>)>, // keys → (zmean, zscale, beta)
    recent: HashMap<i64, (Vec<f64>, Vec<f64>, Vec<f64>)>,
    aff: HashMap<String, (f64, f64)>,
    rate_aff: HashMap<String, (Vec<f64>, Vec<f64>, Vec<f64>)>, // chart → (log rates, values, counts)
    rate_support: HashMap<String, (f64, f64)>,
    rate_bandwidth: ByKeys<f64>,
    tau2_c: f64,
    kap_c: f64,
    shown: (f64, f64),
    // warmup.Model
    slopes: Slopes,
    modes: ByKeys<Mode>,
    fallback: Mode,
}

#[pymethods]
impl PredModel {
    #[new]
    fn new(d: &Bound<'_, PyDict>) -> PyResult<Self> {
        let cs = get(d, "consts")?;
        let cs = cs.cast::<PyDict>()?;
        let consts_key = cs.repr()?.to_string();
        let c = Consts::of(cs)?;
        let pop = Pop::new(get(d, "pop")?.cast::<PyDict>()?)?;
        let mut shape = HashMap::default();
        for (k, v) in get(d, "shape")?.cast::<PyDict>()?.iter() {
            shape.insert(k.extract()?, v.extract()?);
        }
        let mut recent = HashMap::default();
        for (k, v) in get(d, "recent")?.cast::<PyDict>()?.iter() {
            recent.insert(k.extract()?, v.extract()?);
        }
        let mut rate_aff = HashMap::default();
        for (k, v) in get(d, "rate_aff")?.cast::<PyDict>()?.iter() {
            let a: Vec<(f64, f64, f64)> = v.extract()?;
            rate_aff.insert(k.extract()?, (a.iter().map(|x| x.0.ln()).collect(), a.iter().map(|x| x.1).collect(), a.iter().map(|x| x.2).collect()));
        }
        let w = get(d, "warmup")?;
        let w = w.cast::<PyDict>()?;
        let fams = c.families.clone();
        Ok(PredModel {
            c,
            consts_key,
            pop,
            levels: get(d, "levels")?.extract()?,
            level: get(d, "level")?.extract()?,
            glob: fl(d, "glob")?,
            sd_all: fl(d, "sd_all")?,
            shape,
            recent,
            aff: get(d, "aff")?.extract()?,
            rate_aff,
            rate_support: get(d, "rate_support")?.extract()?,
            rate_bandwidth: ByKeys::new(&get(d, "rate_bandwidth")?, num)?,
            tau2_c: fl(d, "tau2_c")?,
            kap_c: fl(d, "kap_c")?,
            shown: get(d, "shown")?.extract()?,
            slopes: Slopes::of(w, &fams)?,
            modes: ByKeys::new(&get(w, "modes")?, mode)?,
            fallback: mode(&get(w, "fallback")?)?,
        })
    }

    /// nps.candidates_from's scoring of `shas` into the request's Choices (charts without plays in the session
    /// window; the same rules as its Python scoring) → the shas left to the Python path. `store` (FeatStore) keeps
    /// the charts' features read across calls; its owner drops a chart whenever analyses[sha] changes
    /// (Builder._note). `now`: the accuracy_forecasts' generated_at.
    #[allow(clippy::too_many_arguments)]
    fn score_into<'py>(&self, py: Python<'py>, choices: &Bound<'py, Choices>, session: &PredSession, store: &Bound<'py, FeatStore>,
                       analyses: &Bound<'py, PyAny>, catalog: &Bound<'py, PyAny>, shas: &Bound<'py, PyList>, played: &Bound<'py, PyAny>,
                       md5_bid: &Bound<'py, PyDict>, selected: Vec<String>, target: f64, skills: bool, wants_vibro: bool,
                       preference: bool, version: i64, now: f64) -> PyResult<Vec<Bound<'py, PyAny>>> {
        let ctx = Ctx { target, warm: session.warm, skills, wants_vibro, preference };
        let mut st = store.borrow_mut();
        if st.consts != self.consts_key || st.version != version {
            st.charts.clear();
            st.consts = self.consts_key.clone();
            st.version = version;
        }
        let sel: Vec<Selection> = st.trees.iter().map(|t| t.select(&selected)).collect();
        struct Job {
            sha: String,
            key: String,
            bid: Option<i64>,
            md5: String,
            keys: i64,
            installed: bool,
            playcount: f64,
        }
        let mut jobs: Vec<Job> = Vec::with_capacity(shas.len());
        let mut left = Vec::new();
        for name in shas.iter() {
            // Chart identity md5 (else sha256); charts played in the session window keep the Python path.
            let inst = catalog.get_item(&name)?;
            let md5 = inst.get_item(intern!(py, "md5"))?;
            let chart = if md5.is_truthy()? { md5.clone() } else { inst.get_item(intern!(py, "sha256"))? };
            if played.contains(&chart)? {
                left.push(name);
                continue;
            }
            let bid: Option<i64> = match md5_bid.get_item(&md5)? {
                Some(b) => b.extract()?,
                None => None,
            };
            let inst = inst.cast::<PyDict>()?;
            let sha: String = name.extract()?;
            if !st.charts.contains_key(&sha) {
                let mut entries = Vec::new();
                let fs = analyses.get_item(&sha)?;
                for rate in fs.try_iter()? {
                    let rate: f64 = rate?.extract()?;
                    let f = fs.get_item(rate)?;
                    let f = f.cast::<PyDict>()?;
                    let offer = Offer::of(rate, f)?;
                    let (feats, skill, profile) = if offer.static_ok(version) {
                        let feats = Feats::of(f, &self.c)?;
                        let skill = SkillValues::of(f, &st.trees[usize::from(feats.keys != 4)])?;
                        (Some(feats), Some(skill), profile_of(f, &st.profile)?)
                    } else {
                        (None, None, Vec::new())
                    };
                    entries.push(Entry { offer, feats, skill, profile });
                }
                st.charts.insert(sha.clone(), entries);
            }
            let opt = |k: &Bound<'py, PyString>| -> PyResult<Option<Bound<'py, PyAny>>> { Ok(inst.get_item(k)?.filter(|v| !v.is_none())) };
            jobs.push(Job {
                sha,
                key: chart.extract()?,
                bid,
                md5: if md5.is_truthy()? { md5.extract()? } else { String::new() },
                keys: inst.get_item(intern!(py, "keys"))?.ok_or_else(|| err("catalogue entry lacks keys"))?.extract()?,
                installed: opt(intern!(py, "installed"))?.map(|v| v.is_truthy()).transpose()?.unwrap_or(true),
                playcount: opt(intern!(py, "playcount"))?.map(|v| v.extract()).transpose()?.unwrap_or(0.),
            });
        }
        // Scoring reads only Rust data: every core but two, without the GIL (~0.4 s on one thread for 25k charts).
        let threads = std::thread::available_parallelism().map_or(1, |x| x.get()).saturating_sub(2).max(1);
        let st: &FeatStore = &st;
        let scored = py.detach(|| background_map(&jobs, threads, |j| {
            self.score_chart(&ctx, &sel, &st.charts[&j.sha], &j.key, j.bid, session).map_err(|e| e.to_string())
        }));
        let mut out = choices.borrow_mut();
        for (r, j) in scored.into_iter().zip(jobs) {
            let slot = match r.map_err(PyValueError::new_err)? {
                None => (None, None),
                Some((best, prop, e)) => {
                    let choice = match (best, e) {
                        (Some((rate, value)), Some(p)) => {
                            let entry = st.charts[&j.sha].iter().find(|x| x.offer.rate == rate).unwrap();
                            let feats = entry.feats.as_ref().unwrap();
                            Some(Arc::new(Choice {
                                group: GROUPS[self.group(feats)].to_string(), nps: entry.offer.nps, length: entry.offer.play_span.unwrap_or(0.) / rate,
                                family: entry.offer.family.clone(), profile: entry.profile.clone(), acc_mid: p.acc_mid,
                                sha: j.sha.clone(), md5: j.md5, keys: j.keys, rate, value, installed: j.installed,
                                playcount: j.playcount, pred: Pred::Native { p, generated_at: now }, patch: None,
                            }))
                        }
                        _ => None,
                    };
                    (choice, prop.map(|(quality, rate, group)| Proposal { quality, rate, keys: j.keys, group: group.to_string() }))
                }
            };
            let at = out.slot(&j.sha).ok_or_else(|| err(&format!("{} is not in this catalogue", j.sha)))?;
            out.slots[at] = Some(slot);
        }
        Ok(left)
    }

    /// Session.correction + Session.warmup_penalty of [(f, chart, bid, rate)], as Recommender.score adds them.
    fn session_shift_many(&self, session: &PredSession, items: &Bound<'_, PyList>) -> PyResult<Vec<f64>> {
        let mut out = Vec::with_capacity(items.len());
        for item in items.iter() {
            let t = item.cast::<PyTuple>()?;
            let f = t.get_item(0)?;
            let f = Feats::of(f.cast::<PyDict>()?, &self.c)?;
            let chart: Option<String> = t.get_item(1)?.extract()?;
            out.push(self.predict(&f, chart.as_deref().unwrap_or(""), t.get_item(2)?.extract()?, t.get_item(3)?.extract()?, session)?.shift);
        }
        Ok(out)
    }

    /// recommend.base_of(pop, f, b, rate, level) for [(f, b, rate, level)] → [(base, map-effect count)].
    fn base_many(&self, items: &Bound<'_, PyList>) -> PyResult<Vec<(f64, f64)>> {
        let mut out = Vec::with_capacity(items.len());
        for item in items.iter() {
            let t = item.cast::<PyTuple>()?;
            let f = t.get_item(0)?;
            let f = Feats::of(f.cast::<PyDict>()?, &self.c)?;
            out.push(self.base_of(&f, t.get_item(1)?.extract()?, t.get_item(2)?.extract()?, t.get_item(3)?.extract()?));
        }
        Ok(out)
    }

    /// Personal.predict (before session form) for [(f, chart, bid, rate)] → [(mu, model sd, attempt sd)].
    fn personal_many(&self, items: &Bound<'_, PyList>) -> PyResult<Vec<(f64, f64, f64)>> {
        let mut out = Vec::with_capacity(items.len());
        for item in items.iter() {
            let t = item.cast::<PyTuple>()?;
            let f = t.get_item(0)?;
            let f = Feats::of(f.cast::<PyDict>()?, &self.c)?;
            // A chart without an md5 has no affinity (Python looks up None).
            let chart: Option<String> = t.get_item(1)?.extract()?;
            out.push(self.personal(&f, chart.as_deref().unwrap_or(""), t.get_item(2)?.extract()?, t.get_item(3)?.extract()?));
        }
        Ok(out)
    }

    /// Predictor.predict(accuracy_only=True) of the display model for [(f, chart, bid, rate)] → [dict].
    fn predict_accuracy<'py>(&self, py: Python<'py>, session: &PredSession, items: &Bound<'py, PyList>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let mut out = Vec::with_capacity(items.len());
        for item in items.iter() {
            let t = item.cast::<PyTuple>()?;
            let f = t.get_item(0)?;
            let f = Feats::of(f.cast::<PyDict>()?, &self.c)?;
            let chart: String = t.get_item(1)?.extract()?;
            let bid: Option<i64> = t.get_item(2)?.extract()?;
            let rate: f64 = t.get_item(3)?.extract()?;
            let p = self.predict(&f, &chart, bid, rate, session)?;
            out.push(p.dict(py)?);
        }
        Ok(out)
    }
}


impl Prediction {
    /// The dict Predictor.predict(accuracy_only=True) returns (accuracy_forecast added in Python).
    fn dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (k, v) in [
            (intern!(py, "display_mu"), self.mu), (intern!(py, "display_base_mu"), self.base_mu),
            (intern!(py, "display_cold_penalty"), self.cold), (intern!(py, "display_cold_scale"), self.cold_scale),
            (intern!(py, "display_sdm"), self.sdm), (intern!(py, "display_sda"), self.sda), (intern!(py, "display_sd"), self.sd),
            (intern!(py, "acc_mid"), self.acc_mid), (intern!(py, "acc_lo"), self.acc_lo), (intern!(py, "acc_hi"), self.acc_hi),
            (intern!(py, "opening_acc"), self.opening_acc), (intern!(py, "activation_minutes"), self.activation),
            (intern!(py, "rate_slope"), self.slope), (intern!(py, "sd_model"), self.sdm), (intern!(py, "warmup_tau"), self.tau),
            (intern!(py, "cold_scale"), self.cold_scale),
        ] {
            d.set_item(k, v)?;
        }
        Ok(d)
    }

    /// dict() with accuracy_forecast (accuracy_targets.forecast_state) for `chart` (sha256 or md5) at `rate`.
    pub(crate) fn forecast_dict<'py>(&self, py: Python<'py>, chart: &str, rate: f64,
                          (generated_at, calculator, predictor, band_z): &(f64, Bound<'py, PyAny>, Bound<'py, PyAny>, f64))
                          -> PyResult<Bound<'py, PyDict>> {
        let d = self.dict(py)?;
        let f = PyDict::new(py);
        let identity = |n: usize| if chart.len() == n { Some(chart) } else { None };
        f.set_item(intern!(py, "schema"), 1)?;
        f.set_item(intern!(py, "target"), intern!(py, "native_displayed_lazer"))?;
        f.set_item(intern!(py, "kind"), intern!(py, "usual_range"))?;
        f.set_item(intern!(py, "generated_at"), generated_at)?;
        f.set_item(intern!(py, "calculator"), calculator)?;
        f.set_item(intern!(py, "predictor"), predictor)?;
        f.set_item(intern!(py, "sha256"), identity(64))?;
        f.set_item(intern!(py, "md5"), identity(32))?;
        f.set_item(intern!(py, "rate"), rate)?;
        f.set_item(intern!(py, "band_z"), band_z)?;
        for (k, v) in [
            (intern!(py, "mid"), self.acc_mid), (intern!(py, "lo"), self.acc_lo), (intern!(py, "hi"), self.acc_hi),
            (intern!(py, "mu"), self.mu), (intern!(py, "sdm"), self.sdm), (intern!(py, "sda"), self.sda), (intern!(py, "sd"), self.sd),
        ] {
            f.set_item(k, v)?;
        }
        d.set_item(intern!(py, "accuracy_forecast"), f)?;
        Ok(d)
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
}

impl Offer {
    fn of<'py>(rate: f64, f: &Bound<'py, PyDict>) -> PyResult<Offer> {
        let py = f.py();
        // None reads as missing: fields of rates the Python filter skips first can be None.
        let opt = |k: &Bound<'py, PyString>| -> PyResult<Option<f64>> {
            f.get_item(k)?.filter(|v| !v.is_none()).map(|v| v.extract()).transpose()
        };
        let st = f.get_item(intern!(py, "nps"))?;
        let st = st.as_ref().filter(|s| !s.is_none()).map(|s| s.cast::<PyDict>()).transpose()?;
        let sub = |k: &Bound<'py, PyString>| -> PyResult<Option<Bound<'py, PyAny>>> {
            Ok(st.map_or(Ok(None), |d| d.get_item(k))?.filter(|v| !v.is_none()))
        };
        Ok(Offer {
            rate,
            notes: opt(intern!(py, "notes"))?.unwrap_or(0.),
            length: opt(intern!(py, "length"))?.unwrap_or(0.),
            version: sub(intern!(py, "version"))?.map(|v| v.extract()).transpose().unwrap_or(None),
            nps: sub(intern!(py, "nps"))?.map(|v| v.extract()).transpose()?.unwrap_or(0.),
            play_span: sub(intern!(py, "play_span"))?.map(|v| v.extract()).transpose()?,
            drill: sub(intern!(py, "drill"))?.map(|v| v.extract()).transpose()?.unwrap_or(0.),
            vibro: sub(intern!(py, "vibro_focused"))?.map(|v| v.is_truthy()).transpose()?.unwrap_or(false),
            family: sub(intern!(py, "family"))?.map(|v| v.str().map(|s| s.to_string())).transpose()?.unwrap_or_default(),
        })
    }

    /// nps._rateable without the Skills match: playable rate, long enough, current structure.
    fn static_ok(&self, version: i64) -> bool {
        (MIN_RATE..=MAX_RATE).contains(&self.rate) && self.notes >= 100. && self.length >= 10000.
            && self.version == Some(version) && self.nps != 0.
    }
}

/// One analysed chart-rate as the builder reads it.
struct Entry {
    offer: Offer,
    feats: Option<Feats>,        // None: not rateable in any request (Offer::static_ok)
    skill: Option<SkillValues>,  // with feats
    profile: Vec<f64>,           // nps.profile, with feats
}

/// skill_practice.dimensions/tree of one keymode class (4K or wider).
struct Tree {
    dims: Vec<String>,
    members: Vec<Vec<String>>,
    tech: Vec<(usize, String)>,    // practice_strengths: dims index → tech_profile leaf
}

/// One request's skill_practice.normalize selection against a Tree (None: not a dimension, strength 0).
struct Selection {
    excluded: Vec<Option<usize>>,
    positive: Vec<Option<usize>>,
}

impl Tree {
    fn select(&self, selected: &[String]) -> Selection {
        let at = |k: &str| self.dims.iter().position(|d| d == k);
        Selection {
            excluded: selected.iter().filter_map(|k| k.strip_prefix('!')).map(at).collect(),
            positive: selected.iter().filter(|k| !k.starts_with('!')).map(|k| at(k)).collect(),
        }
    }
}

/// skill_practice.strengths and practice_strengths of one chart-rate, per Tree dimension.
struct SkillValues {
    values: Vec<f64>,
    practice: Vec<f64>,
}

impl SkillValues {
    fn of(f: &Bound<'_, PyDict>, tree: &Tree) -> PyResult<SkillValues> {
        let py = f.py();
        let sk = f.get_item(intern!(py, "sk"))?;
        let sk = sk.as_ref().map(|d| d.cast::<PyDict>()).transpose()?;
        let mut values = Vec::with_capacity(tree.dims.len());
        for members in &tree.members {
            let mut m = 0f64;
            for k in members {
                if let Some(v) = sk.map(|d| d.get_item(k)).transpose()?.flatten() {
                    m = m.max(0f64.max(v.extract()?));
                }
            }
            values.push(m);
        }
        let mut practice = values.clone();
        // Technical practice needs actual technical character (tech_profile), not an awkward local peak.
        if let Some(profile) = f.get_item(intern!(py, "tech_profile"))?.filter(|p| p.is_truthy().unwrap_or(false)) {
            let profile = profile.cast::<PyDict>()?;
            for (i, leaf) in &tree.tech {
                if let Some(v) = profile.get_item(leaf)? {
                    practice[*i] = practice[*i].min(0f64.max(v.extract()?).sqrt());
                }
            }
        }
        Ok(SkillValues { values, practice })
    }

    /// skill_practice.match(f, selected)[0]
    fn strength(&self, s: &Selection) -> f64 {
        let maximum = self.values.iter().copied().fold(0., f64::max);
        let floor = 0.45f64.max(0.60 * maximum);
        let at = |v: &[f64], k: Option<usize>| k.map_or(0., |i| v[i]);
        if s.excluded.iter().any(|&k| at(&self.values, k) >= floor) {
            return 0.;
        }
        if s.positive.is_empty() {
            return 1.;
        }
        s.positive.iter().map(|&k| at(&self.practice, k)).filter(|&v| v >= floor).map(|v| v.min(1.)).fold(0., f64::max)
    }
}

/// nps.Builder's analysed library in native form: sha → its rates in analysis order, read once. Python drops a
/// chart whenever its analyses change (Builder._note), so a request after a play reads no feature dict and
/// decodes no website chart (3.5 s → see docs/REPORT_2026-10-30_RUST.md).
#[pyclass(module = "maniascope_native")]
#[derive(Default)]
pub struct FeatStore {
    charts: HashMap<String, Vec<Entry>>,
    consts: String,
    version: i64,
    trees: Vec<Tree>, // 4K, wider (skill_practice.native_tree)
    profile: Vec<String>, // nps.PROFILE
}

#[pymethods]
impl FeatStore {
    /// skill_practice.native_tree(): per class (4K, wider) [(dimension, members)]; nps.PROFILE
    #[new]
    fn new(trees: Vec<Vec<(String, Vec<String>)>>, profile: Vec<String>) -> PyResult<Self> {
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
        Ok(FeatStore { trees, profile, ..Self::default() })
    }

    fn drop(&mut self, sha: &str) {
        self.charts.remove(sha);
    }

    fn __len__(&self) -> usize {
        self.charts.len()
    }
}

/// Python's round(x, 2): correctly rounded on the exact binary value, as Rust's formatting.
fn round2(x: f64) -> f64 {
    format!("{x:.2}").parse().unwrap()
}

pub(crate) const MIN_RATE: f64 = 0.70; // nps.MIN_RATE / MAX_RATE / MAX_TARGET / MAX_HIGH
pub(crate) const MAX_RATE: f64 = 1.50;
const MAX_TARGET: f64 = 0.985;
const MAX_HIGH: f64 = 0.99;

/// The request context of nps.candidates_from.
struct Ctx {
    target: f64, // the tab's accuracy target (session.acc_target)
    warm: bool,
    skills: bool,
    wants_vibro: bool,
    preference: bool,
}

impl Ctx {
    fn err(&self) -> f64 {
        1. - self.target.min(MAX_TARGET)
    }
    /// nps.central_target: jack-led charts aim a third closer to 100 %.
    fn central(&self, jack: bool) -> f64 {
        let t = if jack { 1. - (1. - self.target) * 2. / 3. } else { self.target };
        t.min(MAX_TARGET)
    }
    fn band(&self, jack: bool) -> (f64, f64) {
        let t = self.central(jack);
        (t - 0.25 * self.err(), MAX_HIGH.min(t + 0.25 * self.err()))
    }
    fn opening_floor(&self, jack: bool, p: &Prediction) -> f64 {
        let low = self.band(jack).0 - 0.1 * self.err();
        if self.warm {
            return low;
        }
        let cold = (-p.activation / p.tau).exp();
        low + cold * (self.central(jack) - 0.05 * self.err() - low)
    }
    fn eligible(&self, jack: bool, p: &Prediction) -> bool {
        let (low, high) = self.band(jack);
        let slack = 0.1 * self.err();
        if !(low - slack <= p.acc_mid && p.acc_mid <= MAX_HIGH.min(high + slack)) {
            return false;
        }
        self.warm || p.opening_acc >= self.opening_floor(jack, p)
    }
    fn rate_shift(&self, jack: bool, p: &Prediction, target: f64) -> f64 {
        let mut gap = (1. - target).ln() - (1. - p.acc_mid).max(1e-6).ln();
        if !self.warm {
            gap = gap.min((1. - self.opening_floor(jack, p)).ln() - (1. - p.opening_acc).max(1e-6).ln());
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
pub(crate) struct Feats {
    pub(crate) keys: i64,
    pub(crate) overall: f64,
    pub(crate) od: f64,
    pub(crate) length: f64,
    pub(crate) sk: Vec<f64>,                  // by Consts.names, missing = 0
    pub(crate) order: Vec<(f64, usize, bool)>, // sk in dict order (dominant's tie order): value, family, descriptive
    pub(crate) z_tail: [f64; 5],              // ln, stam, load, chord/keys, overlap
    pub(crate) width: f64,
    pub(crate) geometry: Option<Vec<f64>>,
}

impl Feats {
    pub(crate) fn of(f: &Bound<'_, PyDict>, c: &Consts) -> PyResult<Feats> {
        let py = f.py();
        let keys: i64 = at(f, intern!(py, "keys"))?.extract()?;
        let sk_o = at(f, intern!(py, "sk"))?;
        let sk_d = sk_o.cast::<PyDict>()?;
        let mut order = Vec::with_capacity(sk_d.len());
        let mut sk = vec![0.; c.names.len()];
        for (k, v) in sk_d.iter() {
            let k = k.cast::<PyString>()?;
            let k = k.to_str()?;
            let v: f64 = v.extract()?;
            match c.idx.get(k) {
                Some(&i) => {
                    sk[i] = v;
                    order.push((v, c.fam_of[i], c.skip[i]));
                }
                None => order.push((v, c.family(k), c.dominant_skip.iter().any(|x| x == k))),
            }
        }
        let sub = |name: &Bound<'_, PyString>, key: &Bound<'_, PyString>, default: f64| -> PyResult<f64> {
            match f.get_item(name)? {
                Some(d) => match d.cast::<PyDict>()?.get_item(key)? {
                    Some(v) => v.extract(),
                    None => Ok(default),
                },
                None => Ok(default),
            }
        };
        let load = sub(intern!(py, "endurance"), intern!(py, "load"), 0.)?;
        let chord = sub(intern!(py, "nps"), intern!(py, "chord"), 1.)?;
        let overlap = sub(intern!(py, "nps"), intern!(py, "overlap"), 0.)?;
        // Index the lists in place: calibration features and the residual vector are long.
        let width = match f.get_item(intern!(py, "calibration_features"))? {
            Some(v) => match v.cast::<PyList>() {
                Ok(l) if l.len() > 8 => l.get_item(8)?.extract()?,
                Ok(_) => 0.,
                Err(_) if v.len()? > 8 => v.get_item(8)?.extract()?,
                Err(_) => 0.,
            },
            None => 0.,
        };
        let geometry = match f.get_item(intern!(py, "execution"))? {
            Some(e) => match e.cast::<PyDict>()?.get_item(intern!(py, "residual_vector"))? {
                Some(v) if v.is_truthy()? => Some(match v.cast::<PyList>() {
                    Ok(l) => c.geometry.iter().map(|&i| l.get_item(i)?.extract()).collect::<PyResult<Vec<f64>>>()?,
                    Err(_) => c.geometry.iter().map(|&i| v.get_item(i)?.extract()).collect::<PyResult<Vec<f64>>>()?,
                }),
                _ => None,
            },
            None => None,
        };
        let opt = |k: &Bound<'_, PyString>, d: f64| -> PyResult<f64> { f.get_item(k)?.map(|v| v.extract()).transpose().map(|v| v.unwrap_or(d)) };
        Ok(Feats {
            keys,
            overall: at(f, intern!(py, "overall"))?.extract()?,
            od: at(f, intern!(py, "od"))?.extract()?,
            length: opt(intern!(py, "length"), 0.)?,
            sk,
            order,
            z_tail: [at(f, intern!(py, "ln"))?.extract()?, at(f, intern!(py, "stam"))?.extract()?, load, chord / (keys.max(1) as f64), overlap],
            width,
            geometry,
        })
    }
}

/// Frozen Session tables of one builder request (the display target's session view).
#[pyclass(module = "maniascope_native")]
pub struct PredSession {
    warm: bool,
    by_keys: HashMap<i64, KeyState>,
}

struct KeyState {
    cold_scale: f64,
    form: Vec<f64>,       // session_form posterior mean: shared + per family
    penalty: Vec<f64>,    // warmup penalty(keys, family, activation, 0, 1) per family
    activation: Vec<f64>, // activation(keys, family) per family
}

#[pymethods]
impl PredSession {
    #[new]
    fn new(d: &Bound<'_, PyDict>) -> PyResult<Self> {
        let mut by_keys = HashMap::default();
        for (k, v) in get(d, "keys")?.cast::<PyDict>()?.iter() {
            let v = v.cast::<PyDict>()?;
            by_keys.insert(k.extract()?, KeyState {
                cold_scale: fl(v, "cold_scale")?,
                form: get(v, "form")?.extract()?,
                penalty: get(v, "penalty")?.extract()?,
                activation: get(v, "activation")?.extract()?,
            });
        }
        Ok(PredSession { warm: get(d, "warm")?.extract()?, by_keys })
    }
}

#[derive(Clone)]
pub(crate) struct Prediction {
    mu: f64,
    base_mu: f64,
    shift: f64, // session correction + cold penalty (mu − base_mu)
    cold: f64,
    cold_scale: f64,
    sdm: f64,
    sda: f64,
    sd: f64,
    acc_mid: f64,
    acc_lo: f64,
    acc_hi: f64,
    opening_acc: f64,
    activation: f64,
    slope: f64,
    tau: f64,
}

const BAND_Z: f64 = 1.2816 * 0.75; // recommend.BAND_Z

impl PredModel {

    /// score_into for one chart: None (nothing rateable) or (best (rate, value), proposal (quality, rate, group),
    /// the best rate's prediction).
    #[allow(clippy::type_complexity)]
    fn score_chart(&self, ctx: &Ctx, sel: &[Selection], entries: &[Entry], key: &str, bid: Option<i64>, session: &PredSession)
                   -> PyResult<Option<(Option<(f64, f64)>, Option<(f64, f64, &'static str)>, Option<Prediction>)>> {
        let mut rated: Vec<(&Offer, &Feats, Prediction, Option<f64>)> = Vec::new();
        for e in entries {
            let (Some(feats), Some(skill)) = (&e.feats, &e.skill) else { continue };
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
        let jack = |f: &Feats| -> bool {
            let g = self.group(f);
            g == 2 || g == 3
        };
        let target = ctx.central(jack(&rated[0].1));
        let shift_of = |i: usize| ctx.rate_shift(jack(&rated[i].1), &rated[i].2, target);
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
        let jack_a = jack(f);
        let shift = ctx.rate_shift(jack_a, e, target) / slope;
        let mut proposal = round2(MAX_RATE.min(MIN_RATE.max(r * shift.exp())));
        let has = |x: f64| entries.iter().any(|e| e.offer.rate == x);
        let eligible_a = ctx.eligible(jack_a, e);
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
        let mut prop = None;
        if !has(proposal) && (!eligible_a || (e.acc_mid - target).abs() > 0.003) {
            let nps = if ctx.skills { o.nps } else { o.nps * proposal / r };
            if let Some(q) = ctx.value(o, sa, nps, e, target, proposal, target) {
                prop = Some((q, proposal, GROUPS[self.group(f)]));
            }
        }
        let mut best: Option<(usize, f64)> = None;
        for (i, &(o, f, ref e, s)) in rated.iter().enumerate() {
            if !ctx.eligible(jack(f), e) || (o.vibro && !(ctx.skills && ctx.wants_vibro)) {
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
            let (o, f, e, s) = (rated[i].0, rated[i].1, &rated[i].2, rated[i].3);
            ctx.value(o, s, o.nps, e, e.acc_mid, o.rate, ctx.central(jack(f))).map(|v| (o.rate, v))
        });
        Ok(Some((best_out, prop, best.map(|(i, _)| rated.swap_remove(i).2))))
    }
    /// nps._group: the chart's leading pattern family (GROUPS index).
    fn group(&self, f: &Feats) -> usize {
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
    fn base_of(&self, f: &Feats, b: Option<i64>, rate: f64, level: Option<f64>) -> (f64, f64) {
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

    fn chart_affinity(&self, chart: &str, rate: f64, k: i64) -> (f64, f64) {
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

    fn zvec(&self, f: &Feats, mu: f64) -> Vec<f64> {
        self.c.zvec(f, mu)
    }

    /// Personal.predict → (mu, model sd, attempt sd)
    fn personal(&self, f: &Feats, chart: &str, b: Option<i64>, rate: f64) -> (f64, f64, f64) {
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
    fn rate_response(&self, f: &Feats, chart: &str, b: Option<i64>, rate: f64, weights: &[f64]) -> f64 {
        let k = f.keys;
        let physical = self.slopes.physical(f, weights);
        let (lo, hi) = (rate / 1.01, rate * 1.01);
        let level = self.levels.get(&k).copied();
        let left = self.base_of(f, b, lo, level).0 + self.chart_affinity(chart, lo, k).0;
        let right = self.base_of(f, b, hi, level).0 + self.chart_affinity(chart, hi, k).0;
        (physical + (right - left) / (hi / lo).ln()).clamp(1., 8.)
    }

    /// Personal.rate_loss
    fn rate_loss(&self, f: &Feats, chart: &str, b: Option<i64>, rate: f64, loss: f64, weights: &[f64]) -> f64 {
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

    fn predict(&self, f: &Feats, chart: &str, b: Option<i64>, rate: f64, s: &PredSession) -> PyResult<Prediction> {
        let k = f.keys;
        let weights = self.family_weights(f);
        let slope = self.rate_response(f, chart, b, rate, &weights);
        let (mut mu, sdm, sda) = self.personal(f, chart, b, rate);
        let base_mu = mu;
        let mode = self.mode(k);
        let st = s.by_keys.get(&k).ok_or_else(|| err(&format!("predictor session lacks {k}K")))?;
        let dominant = || self.dominant(f);
        // Session._historical_cold at `seconds` (× cold_scale outside)
        let historical = |seconds: f64| -> f64 {
            if s.warm {
                return 0.;
            }
            let d = 0f64.max(10f64.min(seconds / 60.));
            let mean = if d > 0.001 { -(-d / mode.tau).exp_m1() * mode.tau / d } else { 1. };
            if mode.log_rate {
                let mut loss = 0.;
                for (i, &w) in weights.iter().enumerate() {
                    if w != 0. {
                        loss += w * st.penalty[i];
                    }
                }
                self.rate_loss(f, chart, b, rate, loss * mean, &weights)
            } else {
                st.penalty[dominant()] * mean
            }
        };
        let seconds = f.length / 1000. / rate.max(0.01);
        let cold = historical(seconds) * st.cold_scale;
        let mut correction = st.form[0];
        let total: f64 = weights.iter().sum();
        for (i, &w) in weights.iter().enumerate() {
            correction += if total > 0. { w / total } else { 1. / weights.len() as f64 } * st.form[i + 1];
        }
        let shift = correction + cold;
        mu += shift;
        let (a, bb) = self.shown;
        let shown = |y: f64| 0f64.max(1f64.min(1. - (20f64.min(a + bb * y)).exp()));
        let spread = sda.hypot(sdm);
        let opening = historical(0.) * st.cold_scale;
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
            cold_scale: st.cold_scale,
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

/// warmup.family_weights and warmup.dominant (called per chart by every fit and scoring path).
#[pyclass(module = "maniascope_native")]
pub struct Families {
    c: Consts,
    names: Vec<Py<PyString>>,
}

#[pymethods]
impl Families {
    /// (skill names, FAMILIES, MEMBERS items, DESCRIPTIVE)
    #[new]
    fn new(py: Python<'_>, names: Vec<String>, families: Vec<String>, members: Vec<(String, Vec<String>)>, descriptive: Vec<String>) -> PyResult<Self> {
        let c = Consts {
            names,
            families,
            members,
            gap_parts: Vec::new(),
            gap_knots: Vec::new(),
            gap_shared: 0,
            gap_borrow: Vec::new(),
            recent_families: Vec::new(),
            geometry: Vec::new(),
            dominant_skip: descriptive,
            idx: HashMap::default(),
            fam_of: Vec::new(),
            skip: Vec::new(),
            member_idx: Vec::new(),
            gap_idx: Vec::new(),
            recent_idx: Vec::new(),
        }
        .derive()?;
        let names = c.families.iter().map(|n| PyString::new(py, n).unbind()).collect();
        Ok(Families { c, names })
    }

    /// warmup.family_weights(f): ((family, weight), ...) in MEMBERS order.
    fn weights<'py>(&self, py: Python<'py>, f: &Bound<'py, PyDict>) -> PyResult<Bound<'py, PyTuple>> {
        let sk = vec_of_sk(f, &self.c)?;
        let mut values = Vec::new();
        let mut total = 0.;
        for (fam, members) in &self.c.member_idx {
            let m = members.iter().map(|&x| sk[x].max(0.)).fold(0., f64::max);
            if m * m != 0. {
                values.push((*fam, m * m));
                total += m * m;
            }
        }
        let pairs: Vec<(Bound<'py, PyString>, f64)> = if total != 0. {
            values.iter().map(|&(i, w)| (self.names[i].bind(py).clone(), w / total)).collect()
        } else {
            vec![(self.names[dominant_of(f, &self.c)?].bind(py).clone(), 1.)]
        };
        PyTuple::new(py, pairs)
    }

    fn weights_many<'py>(&self, py: Python<'py>, fs: &Bound<'py, PyList>) -> PyResult<Vec<Bound<'py, PyTuple>>> {
        fs.iter().map(|f| self.weights(py, f.cast::<PyDict>()?)).collect()
    }

    /// warmup.dominant(f)
    fn dominant<'py>(&self, py: Python<'py>, f: &Bound<'py, PyDict>) -> PyResult<Bound<'py, PyString>> {
        Ok(self.names[dominant_of(f, &self.c)?].bind(py).clone())
    }
}

/// nps.profile: skill shares and structure, unit length.
fn profile_of(f: &Bound<'_, PyDict>, names: &[String]) -> PyResult<Vec<f64>> {
    let py = f.py();
    let sk = f.get_item(intern!(py, "sk"))?.ok_or_else(|| err("features lack sk"))?;
    let sk = sk.cast::<PyDict>()?;
    let st = f.get_item(intern!(py, "nps"))?;
    let st = st.as_ref().map(|s| s.cast::<PyDict>()).transpose()?;
    let num = |d: Option<&Bound<'_, PyDict>>, k: &str, default: f64| -> PyResult<f64> {
        d.map_or(Ok(None), |d| d.get_item(k))?.map_or(Ok(default), |v| v.extract())
    };
    let mut v = names.iter().map(|k| num(Some(sk), k, 0.)).collect::<PyResult<Vec<f64>>>()?;
    v.extend([0.65 * 2f64.min(num(st, "chord", 1.)? - 1.), num(st, "wide", 0.)?, num(st, "repeat", 0.)?, num(st, "overlap", 0.)?,
              0.5 * 2f64.min(num(st, "burst", 1.)?), 0.6 * num(st, "active", 0.5)?, 0.35 * 2f64.min(num(st, "rhythm", 0.)?)]);
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(0.001);
    Ok(v.into_iter().map(|x| x / norm).collect())
}

/// f["sk"] by Consts.names (missing = 0).
fn vec_of_sk(f: &Bound<'_, PyDict>, c: &Consts) -> PyResult<Vec<f64>> {
    let mut sk = vec![0.; c.names.len()];
    if let Some(d) = f.get_item(intern!(f.py(), "sk"))? {
        for (k, v) in d.cast::<PyDict>()?.iter() {
            if let Some(&i) = c.idx.get(k.cast::<PyString>()?.to_str()?) {
                sk[i] = v.extract()?;
            }
        }
    }
    Ok(sk)
}

/// warmup.dominant(f) as a FAMILIES index.
fn dominant_of(f: &Bound<'_, PyDict>, c: &Consts) -> PyResult<usize> {
    let mut best: Option<(usize, f64)> = None;
    if let Some(d) = f.get_item(intern!(f.py(), "sk"))? {
        for (k, v) in d.cast::<PyDict>()?.iter() {
            let k = k.cast::<PyString>()?;
            let k = k.to_str()?;
            let v: f64 = v.extract()?;
            if !c.dominant_skip.iter().any(|x| x == k) && best.is_none_or(|b| v > b.1) {
                best = Some((c.family(k), v));
            }
        }
    }
    Ok(best.map_or_else(|| c.family("rice"), |b| b.0))
}
