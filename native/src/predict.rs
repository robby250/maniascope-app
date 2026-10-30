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
use std::collections::HashMap as StdMap;
use std::hash::BuildHasherDefault;

/// FxHash (rustc's): SipHash was a fifth of a prediction; these maps are built from our own export.
#[derive(Default, Clone, Copy)]
struct Fx(u64);

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

type HashMap<K, V> = StdMap<K, V, BuildHasherDefault<Fx>>;

fn err(m: &str) -> PyErr {
    PyValueError::new_err(m.to_string())
}

fn get<'py>(d: &Bound<'py, PyDict>, k: &str) -> PyResult<Bound<'py, PyAny>> {
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
fn interp(x: f64, xs: &[f64], ys: &[f64]) -> f64 {
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

/// Constants Python owns: skill names, warmup families, gap parts and knots.
struct Consts {
    names: Vec<String>,
    families: Vec<String>,               // warmup.FAMILIES
    members: Vec<(String, Vec<String>)>, // warmup.MEMBERS in order
    gap_parts: Vec<Vec<String>>,         // recdata.GAP_SHARE_PARTS members; empty = "all"
    gap_knots: Vec<f64>,
    gap_shared: i64,
    gap_borrow: Vec<i64>,
    recent_families: Vec<String>,
    geometry: Vec<usize>,
    dominant_skip: Vec<String>,
    // derived: skill name → index into names, and per-name lookups
    idx: HashMap<String, usize>,
    fam_of: Vec<usize>,
    skip: Vec<bool>,
    member_idx: Vec<(usize, Vec<usize>)>, // family index → member skill indices (family_weights; no technical)
    gap_idx: Vec<Vec<usize>>,
    recent_idx: Vec<usize>,
}

impl Consts {
    /// warmup.family(skill) as a FAMILIES index.
    fn family(&self, skill: &str) -> usize {
        let name = self.members.iter().find(|(_, m)| m.iter().any(|x| x == skill)).map(|(k, _)| k.as_str())
            .unwrap_or(if self.families.iter().any(|x| x == skill) { skill } else { "rice" });
        self.families.iter().position(|x| x == name).unwrap()
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
    w_slope: ByKeys<f64>,
    w_curve: ByKeys<f64>,
    w_knee: f64,
    elasticities: HashMap<i64, (f64, Vec<Option<f64>>)>, // keys → (keymode response, per family)
    modes: ByKeys<Mode>,
    fallback: Mode,
}

#[pymethods]
impl PredModel {
    #[new]
    fn new(d: &Bound<'_, PyDict>) -> PyResult<Self> {
        let cs = get(d, "consts")?;
        let cs = cs.cast::<PyDict>()?;
        let c = Consts {
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
        .derive()?;
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
            w_slope: ByKeys::new(&get(w, "slope")?, num)?,
            w_curve: ByKeys::new(&get(w, "curve")?, num)?,
            w_knee: fl(w, "curve_knee")?,
            elasticities: {
                // warmup.rate_elasticities keys: "<keys>" and "<keys>:<family>"
                let flat: HashMap<String, f64> = get(w, "elasticities")?.extract()?;
                let mut out: HashMap<i64, (f64, Vec<Option<f64>>)> = HashMap::default();
                for (key, &v) in &flat {
                    let (k, fam) = key.split_once(':').map_or((key.as_str(), None), |(k, f)| (k, Some(f)));
                    let Ok(k) = k.parse::<i64>() else { continue };
                    let e = out.entry(k).or_insert((f64::NAN, vec![None; fams.len()]));
                    match fam {
                        None => e.0 = v,
                        Some(f) => if let Some(i) = fams.iter().position(|x| x == f) { e.1[i] = Some(v) },
                    }
                }
                out
            },
            modes: ByKeys::new(&get(w, "modes")?, mode)?,
            fallback: mode(&get(w, "fallback")?)?,
        })
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
            let d = PyDict::new(py);
            for (k, v) in [
                (intern!(py, "display_mu"), p.mu), (intern!(py, "display_base_mu"), p.base_mu),
                (intern!(py, "display_cold_penalty"), p.cold), (intern!(py, "display_cold_scale"), p.cold_scale),
                (intern!(py, "display_sdm"), p.sdm), (intern!(py, "display_sda"), p.sda), (intern!(py, "display_sd"), p.sd),
                (intern!(py, "acc_mid"), p.acc_mid), (intern!(py, "acc_lo"), p.acc_lo), (intern!(py, "acc_hi"), p.acc_hi),
                (intern!(py, "opening_acc"), p.opening_acc), (intern!(py, "activation_minutes"), p.activation),
                (intern!(py, "rate_slope"), p.slope), (intern!(py, "sd_model"), p.sdm), (intern!(py, "warmup_tau"), p.tau),
                (intern!(py, "cold_scale"), p.cold_scale),
            ] {
                d.set_item(k, v)?;
            }
            out.push(d);
        }
        Ok(out)
    }
}

/// What the predictor reads of one feature dict.
struct Feats {
    keys: i64,
    overall: f64,
    od: f64,
    length: f64,
    sk: Vec<f64>,                  // by Consts.names, missing = 0
    order: Vec<(f64, usize, bool)>, // sk in dict order (dominant's tie order): value, family, descriptive
    z_tail: [f64; 5],              // ln, stam, load, chord/keys, overlap
    width: f64,
    geometry: Option<Vec<f64>>,
}

impl Feats {
    fn of(f: &Bound<'_, PyDict>, c: &Consts) -> PyResult<Feats> {
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

struct Prediction {
    mu: f64,
    base_mu: f64,
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
    /// warmup.dominant(f)
    fn dominant(&self, f: &Feats) -> usize {
        let mut best: Option<(usize, f64)> = None;
        for &(v, fam, descriptive) in &f.order {
            if !descriptive && best.is_none_or(|b| v > b.1) {
                best = Some((fam, v));
            }
        }
        best.map_or_else(|| self.c.family("rice"), |b| b.0)
    }

    /// warmup.family_weights(f) as a weight per FAMILIES index.
    fn family_weights(&self, f: &Feats) -> Vec<f64> {
        let n = self.c.families.len();
        let mut w = vec![0.; n];
        let mut total = 0.;
        for (fam, members) in &self.c.member_idx {
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
        let load = f.z_tail[2];
        let near_limit = 1. / (1. + (-20f64).max(20f64.min(-2. * (mu + 3.2))).exp());
        let mut z = f.sk.clone();
        z.extend_from_slice(&f.z_tail);
        z.push(f.width.max(0.).sqrt() * load);
        match &f.geometry {
            Some(g) => z.extend_from_slice(g),
            None => z.extend(std::iter::repeat_n(0., self.c.geometry.len())),
        }
        z.push(load * near_limit);
        z
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

    fn rate_elasticity(&self, f: &Feats, weights: &[f64]) -> f64 {
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

    /// Personal.rate_response: warmup.score_rate_slope + public/personal rate evidence around this rate.
    fn rate_response(&self, f: &Feats, chart: &str, b: Option<i64>, rate: f64, weights: &[f64]) -> f64 {
        let k = f.keys;
        let mut slope = *self.w_slope.get(k).or(self.w_slope.get(0)).unwrap_or(&2.);
        let curve = *self.w_curve.get(k).or(self.w_curve.get(0)).unwrap_or(&0.);
        slope += 2. * curve * (f.overall.max(0.05) / self.w_knee).ln().max(0.);
        let physical = (slope * self.rate_elasticity(f, weights)).clamp(1., 8.);
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
        let elasticity = self.rate_elasticity(f, weights);
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
        mu += correction + cold;
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
