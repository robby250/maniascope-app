//! Python bindings: Chart, Calc (parameters fixed at construction) and the stage/trace functions
//! calibration reads.
use crate::background::background_map;
use crate::chart;
use crate::compute;
use crate::demand;
use crate::gestures;
use crate::model;
use crate::params::{Knobs, Params};
use crate::pattern;
use crate::scroll;
use crate::skills;
use crate::val::{self, from_py, to_py, Dict, Val};
use pyo3::create_exception;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList, PyTuple};
use std::sync::Arc;

create_exception!(maniascope_native, ChartError, PyValueError, "Input is not a usable native osu!mania chart.");

fn verr(e: String) -> PyErr {
    PyValueError::new_err(e)
}

pub fn parse_bytes(data: &[u8]) -> PyResult<chart::Chart> {
    chart::parse(data).map_err(|e| match e {
        chart::ChartError::Chart(m) => ChartError::new_err(m),
        chart::ChartError::Value(m) => PyValueError::new_err(m),
    })
}

fn num<'py>(py: Python<'py>, x: f64) -> PyResult<Bound<'py, PyAny>> {
    // Parsed charts carry integral milliseconds: keep them Python ints.
    if x.fract() == 0.0 && x.abs() < 9e15 {
        Ok((x as i64).into_pyobject(py)?.into_any())
    } else {
        Ok(x.into_pyobject(py)?.into_any())
    }
}

/// A parsed (or hand-built) chart: keys, notes [(start ms, end ms, column)], sv [(ms, multiplier)], od, metadata.
#[pyclass(name = "Chart", module = "maniascope_native")]
pub struct PyChart {
    pub c: chart::Chart,
    notes_cache: Option<Py<PyList>>,
}

impl PyChart {
    pub fn new_from(c: chart::Chart) -> PyChart {
        PyChart { c, notes_cache: None }
    }
}

#[pymethods]
impl PyChart {
    #[new]
    #[pyo3(signature = (keys=0, notes=None, sv=None, od=5.0, title=String::new(), version=String::new(), artist=String::new(), creator=String::new()))]
    #[allow(clippy::too_many_arguments)]
    fn new(keys: usize, notes: Option<Vec<(f64, f64, u8)>>, sv: Option<Vec<(f64, f64)>>, od: f64, title: String, version: String, artist: String, creator: String) -> PyChart {
        let mut c = chart::Chart { keys, od, title, version, artist, creator, sv: sv.unwrap_or_default(), ..Default::default() };
        for (t, e, col) in notes.unwrap_or_default() {
            c.t.push(t);
            c.end.push(e);
            c.col.push(col);
        }
        PyChart::new_from(c)
    }
    /// Last note end − first head, ms (0 without notes).
    #[getter]
    fn length(&self) -> f64 {
        if self.c.len() == 0 { 0.0 } else { self.c.last_end() - self.c.t.iter().copied().fold(f64::INFINITY, f64::min) }
    }
    #[getter]
    fn keys(&self) -> usize {
        self.c.keys
    }
    #[setter]
    fn set_keys(&mut self, v: usize) {
        self.c.keys = v;
    }
    #[getter]
    fn od(&self) -> f64 {
        self.c.od
    }
    #[setter]
    fn set_od(&mut self, v: f64) {
        self.c.od = v;
    }
    #[getter]
    fn title(&self) -> String {
        self.c.title.clone()
    }
    #[setter]
    fn set_title(&mut self, v: String) {
        self.c.title = v;
    }
    #[getter]
    fn version(&self) -> String {
        self.c.version.clone()
    }
    #[setter]
    fn set_version(&mut self, v: String) {
        self.c.version = v;
    }
    #[getter]
    fn artist(&self) -> String {
        self.c.artist.clone()
    }
    #[setter]
    fn set_artist(&mut self, v: String) {
        self.c.artist = v;
    }
    #[getter]
    fn creator(&self) -> String {
        self.c.creator.clone()
    }
    #[setter]
    fn set_creator(&mut self, v: String) {
        self.c.creator = v;
    }
    #[getter]
    fn sv<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let items: Vec<Bound<'py, PyAny>> = self
            .c
            .sv
            .iter()
            .map(|&(t, m)| Ok((num(py, t)?, m).into_pyobject(py)?.into_any()))
            .collect::<PyResult<_>>()?;
        PyList::new(py, items)
    }
    #[setter]
    fn set_sv(&mut self, v: Vec<(f64, f64)>) {
        self.c.sv = v;
    }
    #[getter]
    fn notes<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        if let Some(l) = &self.notes_cache {
            return Ok(l.bind(py).clone());
        }
        let items: Vec<Bound<'py, PyAny>> = (0..self.c.len())
            .map(|i| Ok((num(py, self.c.t[i])?, num(py, self.c.end[i])?, self.c.col[i]).into_pyobject(py)?.into_any()))
            .collect::<PyResult<_>>()?;
        let l = PyList::new(py, items)?;
        self.notes_cache = Some(l.clone().unbind());
        Ok(l)
    }
    /// pickle / copy.deepcopy (multiprocessing workers, tests): rebuilt from the constructor arguments.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
        let py = slf.py();
        let notes = slf.borrow_mut().notes(py)?;
        let me = slf.borrow();
        let c = &me.c;
        let args = (c.keys, notes, c.sv.clone(), c.od, c.title.clone(), c.version.clone(), c.artist.clone(), c.creator.clone());
        Ok((py.get_type::<PyChart>().into_any(), args.into_pyobject(py)?))
    }
    #[setter]
    fn set_notes(&mut self, v: Vec<(f64, f64, u8)>) {
        self.c.t = v.iter().map(|x| x.0).collect();
        self.c.end = v.iter().map(|x| x.1).collect();
        self.c.col = v.iter().map(|x| x.2).collect();
        self.notes_cache = None;
    }
}

/// .osu bytes → Chart.
#[pyfunction]
pub fn parse(data: &[u8]) -> PyResult<PyChart> {
    Ok(PyChart::new_from(parse_bytes(data)?))
}

fn chart_of(o: &Bound<'_, PyAny>) -> PyResult<chart::Chart> {
    if let Ok(c) = o.cast::<PyChart>() {
        return Ok(c.borrow().c.clone());
    }
    if let Ok(b) = o.cast::<PyBytes>() {
        return parse_bytes(b.as_bytes());
    }
    // Hand-built charts (tests, SimpleNamespace): keys, notes, optional sv and od.
    let opt = |name: &str| o.getattr(name).ok().filter(|v| !v.is_none());
    let keys: usize = o.getattr("keys").map_err(|_| PyValueError::new_err("expected a Chart or .osu bytes"))?.extract()?;
    let notes: Vec<(f64, f64, u8)> = o.getattr("notes")?.extract()?;
    let sv: Option<Vec<(f64, f64)>> = opt("sv").map(|v| v.extract()).transpose()?;
    let od: f64 = opt("od").map(|v| v.extract()).transpose()?.unwrap_or(5.0);
    Ok(PyChart::new(keys, Some(notes), sv, od, String::new(), String::new(), String::new(), String::new()).c)
}

fn dict_of(o: &Bound<'_, PyAny>) -> PyResult<Dict> {
    match from_py(o)? {
        Val::Dict(d) => Ok(d),
        _ => Err(PyValueError::new_err("expected a dict")),
    }
}

fn floats(v: &[f64]) -> Val {
    Val::floats(v)
}

fn run_val(r: &demand::Run) -> Val {
    let mut parts = Dict::new();
    for &p in &r.parts.keys {
        parts.set(demand::P_NAMES[p as usize], Val::floats(r.parts.get(p)));
    }
    Val::Tuple(vec![Val::Dict(parts), floats(&r.totals), Val::Tuple(r.fine.iter().map(|f| floats(f)).collect())])
}

fn series_val(s: &skills::Series) -> Val {
    let mut d = Dict::new();
    for &k in &s.keys {
        d.set(k.name(), Val::floats(s.get(k)));
    }
    Val::Dict(d)
}

/// A calculator with fixed parameters: difficulty.json (as JSON text) + knob overrides (JSON text).
#[pyclass(frozen, module = "maniascope_native")]
pub struct Calc {
    pub prm: Arc<Params>,
}

#[pymethods]
impl Calc {
    #[new]
    #[pyo3(signature = (model, overrides=""))]
    fn new(model: &str, overrides: &str) -> PyResult<Self> {
        let m: serde_json::Value = serde_json::from_str(model).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let o: serde_json::Value = if overrides.is_empty() {
            serde_json::Value::Null
        } else {
            serde_json::from_str(overrides).map_err(|e| PyValueError::new_err(e.to_string()))?
        };
        Ok(Calc { prm: Arc::new(Params::new(m, &o).map_err(verr)?) })
    }

    /// Full analyses of a Chart (or .osu bytes), one result dict per rate.
    #[pyo3(signature = (chart, rates, sv=true))]
    fn compute<'py>(&self, py: Python<'py>, chart: &Bound<'py, PyAny>, rates: Vec<f64>, sv: bool) -> PyResult<Vec<Bound<'py, PyAny>>> {
        let c = chart_of(chart)?;
        let prm = self.prm.clone();
        let res: Result<Vec<Dict>, String> = py.detach(|| rates.iter().map(|&r| compute::compute(&c, r, sv, &prm)).collect());
        res.map_err(verr)?.into_iter().map(|d| to_py(py, &Val::Dict(d))).collect()
    }

    /// Many (chart bytes, rates) jobs on `threads` OS threads without the GIL. Each item is a list of result
    /// dicts, or the error message string.
    #[pyo3(signature = (jobs, sv=true, threads=0))]
    fn compute_batch<'py>(&self, py: Python<'py>, jobs: Vec<(Vec<u8>, Vec<f64>)>, sv: bool, threads: usize) -> PyResult<Vec<Bound<'py, PyAny>>> {
        let prm = self.prm.clone();
        let out: Vec<Result<Vec<Dict>, String>> = py.detach(|| background_map(&jobs, threads, |(data, rates)| {
            chart::parse(data)
                .map_err(|e| match e {
                    chart::ChartError::Chart(m) | chart::ChartError::Value(m) => m,
                })
                .and_then(|c| rates.iter().map(|&r| compute::compute(&c, r, sv, &prm)).collect())
        }));
        out.into_iter()
            .map(|r| match r {
                Ok(ds) => Ok(PyList::new(py, ds.iter().map(|d| to_py(py, &Val::Dict(d.clone()))).collect::<PyResult<Vec<_>>>()?)?.into_any()),
                Err(e) => Ok(e.into_pyobject(py)?.into_any()),
            })
            .collect()
    }

    /// Calibration trace of one compute: phases, total, series, switch, parts, result (+ action runs when
    /// rolled execution applies) for sv in (True, False) as computed, and the rolled basis if any.
    #[pyo3(signature = (chart, rate, sv=true, plans=true))]
    fn trace<'py>(&self, py: Python<'py>, chart: &Bound<'py, PyAny>, rate: f64, sv: bool, plans: bool) -> PyResult<Bound<'py, PyAny>> {
        let c = chart_of(chart)?;
        let prm = self.prm.clone();
        let (res, shared) = py.detach(|| {
            let mut shared = compute::Shared { trace: Some(Vec::new()), execution_plans: plans, ..Default::default() };
            compute::compute_inner(&c, rate, sv, &mut shared, &prm).map(|r| (r, shared))
        }).map_err(verr)?;
        let mut out = Dict::new();
        out.set("result", res);
        let mut tr = Dict::new();
        for (key, t) in shared.trace.as_ref().unwrap() {
            let mut d = Dict::new();
            d.set("phases", Val::List(t.phases.iter().map(|p| floats(p)).collect()));
            d.set("total", floats(&t.total));
            d.set("series", series_val(&t.series));
            d.set("switch", floats(&t.switch));
            let mut parts = Dict::new();
            for &p in &t.parts.keys {
                parts.set(demand::P_NAMES[p as usize], Val::floats(t.parts.get(p)));
            }
            d.set("parts", parts);
            d.set("result", t.result.clone());
            if let Some(ar) = &t.action_runs {
                let mut runs = Dict::new();
                for (cl, acts, fine) in ar {
                    let acts = Val::List(acts.iter().map(|a| Val::Tuple(vec![Val::Int(a.hand as i64), Val::F(a.rice), Val::F(a.cross), Val::Bool(a.protected), Val::Int(a.fb as i64)])).collect());
                    runs.setb(*cl, Val::Tuple(vec![acts, Val::Tuple(fine.iter().map(|f| floats(f)).collect())]));
                }
                d.set("action_runs", runs);
            }
            tr.setb(*key, d);
        }
        out.set("trace", tr);
        if let Some(g) = &shared.gestures {
            out.set("gestures", gestures_val(g));
        }
        to_py(py, &Val::Dict(out))
    }

    /// `_demand` runs (+ plain run on SV charts) per centre assignment: [(parts, totals, (fine0, fine1, fine2))].
    #[pyo3(signature = (chart, rate, sv=true))]
    fn demand<'py>(&self, py: Python<'py>, chart: &Bound<'py, PyAny>, rate: f64, sv: bool) -> PyResult<Bound<'py, PyAny>> {
        let c = chart_of(chart)?;
        let k = &self.prm.k;
        let g = demand::geometry(&c, rate, k);
        let read = if sv { Some(scroll::reading(&c, rate, &self.prm).map_err(verr)?) } else { None };
        let want_plain = sv && !c.sv.is_empty() && read.as_ref().is_some_and(|r| r.0.iter().any(|&x| x != 0.0));
        let mut runs = Vec::new();
        for cl in if c.keys % 2 == 1 { vec![true, false] } else { vec![true] } {
            let (run, plain) = demand::demand(&c, &g, cl, read.as_ref().map(|r| &r.0[..]), read.as_ref().map(|r| &r.1[..]), want_plain, None, k);
            let mut d = Dict::new();
            d.set("centre_left", cl);
            d.set("run", run_val(&run));
            if let Some(p) = plain {
                d.set("plain", run_val(&p));
            }
            runs.push(Val::Dict(d));
        }
        to_py(py, &Val::List(runs))
    }

    /// Scroll reading: (per-note strain, per-note SV kind shares, summary dict).
    fn reading<'py>(&self, py: Python<'py>, chart: &Bound<'py, PyAny>, rate: f64) -> PyResult<Bound<'py, PyAny>> {
        let c = chart_of(chart)?;
        let (strain, kinds, summary) = scroll::reading(&c, rate, &self.prm).map_err(verr)?;
        let mut s = Dict::new();
        for (n, v) in scroll::FEATURES.iter().zip(summary) {
            s.set(n, v);
        }
        to_py(py, &Val::Tuple(vec![floats(&strain), Val::List(kinds.iter().map(|k| Val::Tuple(k.iter().map(|&x| Val::F(x)).collect())).collect()), Val::Dict(s)]))
    }

    /// Named knob values (defaults + overrides).
    fn knob(&self, name: &str) -> PyResult<f64> {
        knob_value(&self.prm.k, name)
    }

    /// difficulty_model._baseline_rating / final_rating / adjustment / gesture_rating on stored values.
    #[pyo3(signature = (keys, raw, features, execution=None))]
    fn baseline_rating(&self, keys: usize, raw: f64, features: Vec<f64>, execution: Option<&Bound<'_, PyAny>>) -> PyResult<f64> {
        let ex = execution.map(dict_of).transpose()?;
        Ok(model::baseline_rating(&self.prm, keys, raw, &features, ex.as_ref()))
    }
    #[pyo3(signature = (keys, raw, features, execution=None))]
    fn final_rating(&self, keys: usize, raw: f64, features: Vec<f64>, execution: Option<&Bound<'_, PyAny>>) -> PyResult<f64> {
        let ex = execution.map(dict_of).transpose()?;
        Ok(model::final_rating(&self.prm, keys, raw, &features, ex.as_ref()))
    }
    fn adjustment(&self, keys: usize, raw: f64, features: Vec<f64>) -> f64 {
        model::adjustment(&self.prm, keys, raw, &features)
    }
    #[pyo3(signature = (keys, base, raw, execution=None))]
    fn gesture_rating(&self, keys: usize, base: f64, raw: f64, execution: Option<&Bound<'_, PyAny>>) -> PyResult<f64> {
        let ex = execution.map(dict_of).transpose()?;
        Ok(model::gesture_rating(&self.prm, keys, base, raw, ex.as_ref()))
    }
    /// structural_residual.correction / tree_shift.
    #[pyo3(signature = (keys, baseline, values))]
    fn correction(&self, keys: usize, baseline: f64, values: Option<Vec<f64>>) -> f64 {
        model::correction(&self.prm, keys, baseline, values.as_deref())
    }
    #[pyo3(signature = (keys, baseline, values))]
    fn tree_shift(&self, keys: usize, baseline: f64, values: Option<Vec<f64>>) -> f64 {
        model::tree_shift(&self.prm, keys, baseline, values.as_deref())
    }
    /// score_units.display_factor.
    #[pyo3(signature = (keys, baseline, execution, original))]
    fn display_factor(&self, keys: usize, baseline: f64, execution: &Bound<'_, PyAny>, original: f64) -> PyResult<f64> {
        let ex = dict_of(execution)?;
        Ok(model::display_factor(&self.prm, keys, baseline, Some(&ex), original))
    }
    /// score_units.apply on a result dict (returns the converted copy).
    fn units_apply<'py>(&self, py: Python<'py>, result: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let mut d = dict_of(result)?;
        model::units_apply(&self.prm, &mut d);
        to_py(py, &Val::Dict(d))
    }
}

pub fn knob_value(k: &Knobs, name: &str) -> PyResult<f64> {
    k.get(name).ok_or_else(|| PyValueError::new_err(format!("unknown or non-scalar knob: {name}")))
}

fn gestures_val(g: &gestures::Analysis) -> Val {
    let mut series = Dict::new();
    for (n, s) in gestures::SERIES.iter().zip(&g.series) {
        series.set(n, Val::floats(s));
    }
    let mut feats = Dict::new();
    for (n, v) in gestures::FEATURES.iter().zip(g.features) {
        feats.set(n, v);
    }
    let mut d = Dict::new();
    d.set("series", series);
    d.set("features", feats);
    if let Some(plans) = &g.plans {
        let mut pd = Dict::new();
        for (cl, ps) in plans {
            let items = ps
                .iter()
                .map(|p| {
                    let mut x = Dict::new();
                    x.set("ids", Val::Tuple(p.ids.iter().map(|&i| Val::Int(i as i64)).collect()));
                    x.set("time", p.time);
                    x.set("strength", p.strength);
                    x.set("control", p.control);
                    x.set("anchor", p.anchor);
                    x.set("span", p.span);
                    x.set("offsets", Val::Tuple(p.offsets.iter().map(|&o| Val::F(o)).collect()));
                    Val::Dict(x)
                })
                .collect();
            pd.setb(*cl, Val::List(items));
        }
        d.set("plans", pd);
    }
    Val::Dict(d)
}

/// gestures.analyse (series, features, plans).
#[pyfunction]
#[pyo3(signature = (chart, rate=1.0, bin_seconds=0.5, with_plans=false))]
pub fn gestures_analyse<'py>(py: Python<'py>, chart: &Bound<'py, PyAny>, rate: f64, bin_seconds: f64, with_plans: bool) -> PyResult<Bound<'py, PyAny>> {
    let c = chart_of(chart)?;
    if !rate.is_finite() || rate <= 0.0 || !bin_seconds.is_finite() || bin_seconds <= 0.0 {
        return Err(PyValueError::new_err("rate and bin_seconds must be positive and finite"));
    }
    let g = gestures::analyse(&c, rate, bin_seconds, with_plans);
    to_py(py, &gestures_val(&g))
}

/// gestures.great_window.
#[pyfunction]
pub fn great_window(od: f64) -> f64 {
    gestures::great_window(od)
}

/// execution.analyse (per-bin averages + release_sync + flow).
#[pyfunction]
#[pyo3(signature = (chart, rate=1.0))]
pub fn execution_analyse<'py>(py: Python<'py>, chart: &Bound<'py, PyAny>, rate: f64) -> PyResult<Bound<'py, PyAny>> {
    let c = chart_of(chart)?;
    let k = Knobs::default();
    let g = demand::geometry(&c, rate, &k);
    let i = crate::execution::analyse(&c, &g, k.BIN);
    let mut d = Dict::new();
    for (n, v) in [("mash", &i.mash), ("pseudo", &i.pseudo), ("regular", &i.regular), ("switch", &i.switch), ("regular_cal", &i.regular_cal), ("switch_cal", &i.switch_cal), ("count", &i.count)] {
        d.set(n, Val::floats(v));
    }
    d.set("release_sync", i.release_sync);
    d.set("flow", i.flow);
    to_py(py, &Val::Dict(d))
}

/// pattern_control.tech_profile (standalone: runs the execution pass).
#[pyfunction]
#[pyo3(signature = (chart, rate=1.0))]
pub fn tech_profile<'py>(py: Python<'py>, chart: &Bound<'py, PyAny>, rate: f64) -> PyResult<Bound<'py, PyAny>> {
    let c = chart_of(chart)?;
    let k = Knobs::default();
    let g = demand::geometry(&c, rate, &k);
    let i = crate::execution::analyse(&c, &g, k.BIN);
    let t = pattern::tech_profile(&c, rate, &i, &k);
    let mut d = Dict::new();
    d.set("version", pattern::TECH_PROFILE_VERSION);
    d.set("technical", t[0]);
    d.set("patterntech", t[1]);
    d.set("rhythmtech", t[2]);
    to_py(py, &Val::Dict(d))
}

/// pattern_control.quadstream → (per-bin series, share).
#[pyfunction]
#[pyo3(signature = (chart, rate=1.0, bin_seconds=0.5))]
pub fn quadstream(chart: &Bound<'_, PyAny>, rate: f64, bin_seconds: f64) -> PyResult<(Vec<f64>, f64)> {
    Ok(pattern::quadstream(&chart_of(chart)?, rate, bin_seconds))
}

/// pattern_control.technical_title(name, skills, profile).
#[pyfunction]
#[pyo3(signature = (name, skills, profile=None))]
pub fn technical_title(name: &str, skills: &Bound<'_, PyAny>, profile: Option<&Bound<'_, PyAny>>) -> PyResult<String> {
    let sk = dict_of(skills)?;
    let p = match profile {
        Some(p) if p.is_truthy()? => {
            let d = dict_of(p)?;
            Some([d.fget("technical", 0.0), d.fget("patterntech", 0.0), d.fget("rhythmtech", 0.0)])
        }
        _ => None,
    };
    Ok(pattern::technical_title(name, sk.fget("technical", 0.0), p.as_ref()))
}

/// archetype_names.profile → ln_chordstream.
#[pyfunction]
#[pyo3(signature = (chart, rate=1.0))]
pub fn archetype_profile(chart: &Bound<'_, PyAny>, rate: f64) -> PyResult<f64> {
    if !rate.is_finite() || rate <= 0.0 {
        return Err(PyValueError::new_err("invalid rate"));
    }
    Ok(crate::card::profile(&chart_of(chart)?, rate))
}

/// chart_structure.structure.
#[pyfunction]
#[pyo3(signature = (chart, rate=1.0))]
pub fn structure<'py>(py: Python<'py>, chart: &Bound<'py, PyAny>, rate: f64) -> PyResult<Bound<'py, PyAny>> {
    to_py(py, &Val::Dict(crate::structure::structure(&chart_of(chart)?, rate, &Knobs::default())))
}

/// skill_calc.card / describe on a result dict.
#[pyfunction]
pub fn card<'py>(py: Python<'py>, result: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    to_py(py, &Val::List(crate::card::card(&dict_of(result)?)))
}
#[pyfunction]
pub fn describe(result: &Bound<'_, PyAny>) -> PyResult<(String, Vec<String>)> {
    Ok(crate::card::describe(&dict_of(result)?, &Knobs::default()))
}
#[pyfunction]
pub fn ranked(result: &Bound<'_, PyAny>) -> PyResult<Vec<&'static str>> {
    Ok(crate::card::ranked(&dict_of(result)?).into_iter().map(|s| s.name()).collect())
}
#[pyfunction]
pub fn skills_of(keys: usize) -> Vec<&'static str> {
    skills::skills(keys).into_iter().map(|s| s.name()).collect()
}
/// Display name of a skill key.
#[pyfunction]
pub fn skill_name(key: &str) -> PyResult<&'static str> {
    skills::Sk::from_name(key).map(crate::card::display_name).ok_or_else(|| PyValueError::new_err(format!("unknown skill: {key}")))
}

/// structural_residual.vector on a feature dict (recdata feats layout).
#[pyfunction]
pub fn residual_vector(f: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    let f = dict_of(f)?;
    let empty = Dict::new();
    let sk: Vec<(String, f64)> = f.dict("sk").map(|d| d.0.iter().filter_map(|(k, v)| match k {
        val::Key::S(k) => Some((k.clone(), v.f())),
        _ => None,
    }).collect()).unwrap_or_default();
    let cf = f.get("calibration_features").map(|v| v.list_f()).unwrap_or_default();
    Ok(model::sr_vector(
        &sk,
        f.dict("nps").unwrap_or(&empty),
        f.dict("execution").unwrap_or(&empty),
        f.dict("endurance").unwrap_or(&empty),
        f.fget("rate", 1.0),
        f.fget("length", 0.0),
        f.fget("notes", 0.0),
        f.f("keys") as usize,
        f.fget("ln", 0.0),
        f.fget("stam", 1.0),
        &cf,
    ))
}

/// Default knob values: {name: number | [numbers] | [[a, b], ...]} as JSON text.
#[pyfunction]
pub fn default_knobs() -> String {
    let m: serde_json::Map<String, serde_json::Value> = Knobs::default().dump().into_iter().map(|(k, v)| (k.to_string(), v)).collect();
    serde_json::Value::Object(m).to_string()
}

/// score_units.rating(value, od, model) on a score_units mode dict.
#[pyfunction]
pub fn unit_rating(value: f64, od: f64, model: &Bound<'_, PyAny>) -> PyResult<f64> {
    Ok(model::unit_rating(value, od, &json_of(&from_py(model)?)))
}

fn json_of(v: &Val) -> serde_json::Value {
    match v {
        Val::None => serde_json::Value::Null,
        Val::Bool(b) => (*b).into(),
        Val::Int(i) => (*i).into(),
        Val::F(x) => (*x).into(),
        Val::Str(s) => s.clone().into(),
        Val::List(xs) | Val::Tuple(xs) => xs.iter().map(json_of).collect(),
        Val::Dict(d) => serde_json::Value::Object(d.0.iter().filter_map(|(k, x)| match k {
            val::Key::S(k) => Some((k.clone(), json_of(x))),
            _ => None,
        }).collect()),
    }
}

/// Default value of a scalar knob.
#[pyfunction]
pub fn default_knob(name: &str) -> PyResult<f64> {
    knob_value(&Knobs::default(), name)
}

/// skill_calc._combine / _star / _horizon_levels / _phase_total / _stamina with default knobs.
#[pyfunction]
pub fn combine(levels: [f64; 4], notes: usize) -> f64 {
    skills::combine(&levels, notes, &Knobs::default())
}
#[pyfunction]
pub fn star(level: f64) -> f64 {
    skills::star(level, &Knobs::default())
}
#[pyfunction]
pub fn horizon_levels(series: Vec<f64>) -> [f64; 4] {
    skills::horizon_levels(&series, &Knobs::default())
}
#[pyfunction]
pub fn phase_total(fine: [Vec<f64>; 3], p: usize) -> Vec<f64> {
    skills::phase_total(&fine, p, &Knobs::default())
}
#[pyfunction]
pub fn endurance(fines: Vec<[Vec<f64>; 3]>) -> ((f64, f64), Vec<f64>) {
    let k = Knobs::default();
    let refs: Vec<&[Vec<f64>; 3]> = fines.iter().collect();
    model::endurance(&refs, k.BIN, k.PHASES)
}
/// Rolled execution: is a rolled packet possible at all.
#[pyfunction]
pub fn rolled_possible(chart: &Bound<'_, PyAny>, rate: f64) -> PyResult<bool> {
    Ok(compute::possible(&chart_of(chart)?, rate))
}

/// scroll_reading geometry: per-note 4-vectors (preview, crowding, motion, stutter), kind flags, summary.
#[pyfunction]
#[pyo3(signature = (chart, rate=1.0))]
pub fn scroll_geometry<'py>(py: Python<'py>, chart: &Bound<'py, PyAny>, rate: f64) -> PyResult<Bound<'py, PyAny>> {
    if !rate.is_finite() || rate <= 0.0 {
        return Err(PyValueError::new_err("rate must be finite and positive"));
    }
    let g = scroll::geometry(&chart_of(chart)?, rate);
    let mut s = Dict::new();
    for (n, v) in scroll::FEATURES.iter().zip(g.summary) {
        s.set(n, v);
    }
    let rows = |v: &[f64]| Val::List(v.iter().map(|&x| Val::F(x)).collect());
    let mut d = Dict::new();
    d.set("vectors", Val::List(g.vectors.iter().map(|x| rows(x)).collect()));
    d.set("kinds", Val::List(g.kinds.iter().map(|x| rows(x)).collect()));
    d.set("summary", s);
    to_py(py, &Val::Dict(d))
}

/// The speed kernel of the demand model (skill_calc._v): knee/taper default to FLAM and 2.
#[pyfunction]
#[pyo3(signature = (d, knee=None, taper=2.0))]
pub fn v(d: f64, knee: Option<f64>, taper: f64) -> f64 {
    crate::kern::v(d, knee.unwrap_or(Knobs::default().FLAM), taper)
}

/// osu!lazer star rating and judgement count (objects + hold notes) per (osu bytes, [rates]) job, rates
/// limited to the ranked 0.75 (HT) / 1 / 1.5 (DT); None for a chart rosu-pp cannot read.
#[pyfunction]
#[pyo3(signature = (jobs, threads=0))]
pub fn lazer_stars_batch(py: Python<'_>, jobs: Vec<(Vec<u8>, Vec<f64>)>, threads: usize) -> Vec<Option<Vec<(f64, u32)>>> {
    py.detach(|| background_map(&jobs, threads, |(data, rates)| {
        let map = rosu_pp::Beatmap::from_bytes(data).map_err(|e| e.to_string())?;
        rates.iter().map(|&r| {
            let mods = if r == 0.75 { 256 } else if r == 1.5 { 64 } else if r == 1.0 { 0 } else { return Err(format!("unranked rate {r}")) };
            match rosu_pp::Difficulty::new().mods(mods).lazer(true).calculate(&map) {
                rosu_pp::any::DifficultyAttributes::Mania(a) => Ok((a.stars, a.n_objects + a.n_hold_notes)),
                _ => Err("not mania".to_string()),
            }
        }).collect()
    }))
    .into_iter()
    .map(Result::ok)
    .collect()
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("ChartError", m.py().get_type::<ChartError>())?;
    m.add_class::<PyChart>()?;
    m.add_class::<Calc>()?;
    for f in [
        wrap_pyfunction!(parse, m)?, wrap_pyfunction!(gestures_analyse, m)?, wrap_pyfunction!(great_window, m)?,
        wrap_pyfunction!(execution_analyse, m)?, wrap_pyfunction!(tech_profile, m)?, wrap_pyfunction!(quadstream, m)?,
        wrap_pyfunction!(technical_title, m)?, wrap_pyfunction!(archetype_profile, m)?, wrap_pyfunction!(structure, m)?,
        wrap_pyfunction!(card, m)?, wrap_pyfunction!(describe, m)?, wrap_pyfunction!(ranked, m)?,
        wrap_pyfunction!(skills_of, m)?, wrap_pyfunction!(skill_name, m)?, wrap_pyfunction!(residual_vector, m)?,
        wrap_pyfunction!(default_knob, m)?, wrap_pyfunction!(default_knobs, m)?, wrap_pyfunction!(unit_rating, m)?, wrap_pyfunction!(combine, m)?, wrap_pyfunction!(star, m)?,
        wrap_pyfunction!(horizon_levels, m)?, wrap_pyfunction!(phase_total, m)?, wrap_pyfunction!(endurance, m)?,
        wrap_pyfunction!(rolled_possible, m)?, wrap_pyfunction!(scroll_geometry, m)?, wrap_pyfunction!(v, m)?, wrap_pyfunction!(lazer_stars_batch, m)?,
    ] {
        m.add_function(f)?;
    }
    m.add("MODEL_VERSION", compute::MODEL_VERSION)?;
    m.add("GESTURE_FEATURES", gestures::FEATURES.to_vec())?;
    m.add("GESTURE_SERIES", gestures::SERIES.to_vec())?;
    m.add("READING_FEATURES", scroll::FEATURES.to_vec())?;
    m.add("READING_WEIGHTS", scroll::DEFAULT_WEIGHTS.to_vec())?;
    Ok(())
}
