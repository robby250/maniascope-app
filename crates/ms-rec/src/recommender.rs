//! recommend.Recommender's fit: both accuracy targets on the historical rows.
use crate::evidence::{evidence_rows, now_months, personal_row, play_levels, stable_evidence, Evidence, FeatureSource, Rows};
use ms_data::events::load_events;
use ms_data::public::Public;
use ms_data::store::{acc320, acc_lazer, now, parse_stats, Db, Result};
use ms_model::personal::{personal_fit, transfer, Fit, Knobs, WarmupArgs};
use ms_model::predict::{Consts, HashMap, PredModel};
use ms_model::public::{population, slopes};
use ms_model::session::{Session, HISTORY_S, SESSION_GAP};
use ms_model::warmup::{FormPrior, Model};
use std::sync::Arc;

pub const KNOBS: Knobs = Knobs { half_life: 12., level_bw: 4., history_zone: 0.15, recent_months: 2., att_sd: 1., session_gap: SESSION_GAP };

pub struct Target {
    pub fit: Fit,
    pub warmup: Arc<Model>,
    pub model: Arc<PredModel>,
}

pub struct Fitted {
    pub rows: Rows,
    pub levels: HashMap<i64, f64>,
    pub pp: Target,
    pub display: Option<Target>,
    pub shown: (f64, f64),
}

/// {target: {played: share}} between warm-up families.
pub fn family_transfer(c: &Consts) -> Vec<Vec<f64>> {
    c.families.iter().map(|a| c.families.iter().map(|b| transfer(a, b)).collect()).collect()
}

/// log(1 − shown) = a + b·log(1 − acc320), fit on the player's own judgement mix: the model works in acc320, the
/// player reads lazer's accuracy.
pub fn fit_shown(db: &Db, until: f64) -> Result<(f64, f64)> {
    let mut query = "SELECT stats FROM scores WHERE src IN ('realm', 'live') AND stats IS NOT NULL".to_string();
    if until.is_finite() {
        query += &format!(" AND julianday(played)<julianday({until:?},'unixepoch')");
    }
    let mut statement = db.prepare(&query)?;
    let (mut xs, mut zs) = (Vec::new(), Vec::new());
    for s in statement.query_map([], |r| r.get::<_, String>(0))? {
        let Some(s) = parse_stats(&s?) else { continue };
        let ((a, n), al) = (acc320(&s), acc_lazer(&s));
        if n != 0 && 0. < a && a < 1. && 0. < al && al < 1. {
            xs.push((1. - a).ln());
            zs.push((1. - al).ln());
        }
    }
    if xs.len() < 20 {
        return Ok((0.75f64.ln(), 1.));
    }
    let n = xs.len() as f64;
    let (mx, mz) = (xs.iter().sum::<f64>() / n, zs.iter().sum::<f64>() / n);
    let sxx: f64 = xs.iter().map(|x| (x - mx).powi(2)).sum();
    let sxz: f64 = xs.iter().zip(&zs).map(|(x, z)| (x - mx) * (z - mz)).sum();
    let b = sxz / sxx;
    Ok((mz - b * mx, b))
}

pub fn fit_target(c: &Arc<Consts>, public: &Public, pop: &ms_data::public::Population, rows: &[&Evidence], feats: &[ms_model::predict::Feats],
              levels: &HashMap<i64, f64>, display: bool, month: f64, shown: (f64, f64)) -> Target {
    let native = Arc::new(population(pop, c));
    let base_model = PredModel::population(c.clone(), native.clone());
    let kept = stable_evidence(rows.iter().copied());
    let personal: Vec<_> = kept.iter().map(|r| {
        let (base, _) = base_model.base_of(&feats[r.f], r.b, ms_calc::py::round_digits(r.rate, 3), levels.get(&r.keys).copied());
        personal_row(r, if display { r.y_lazer } else { r.y }, base)
    }).collect();
    let args = WarmupArgs { transfer: family_transfer(c), slopes: slopes(pop, &public.meta, c) };
    let fit = personal_fit(c, personal, feats, month, levels, KNOBS, &args);
    let context = &fit.session_validation;
    let prior = FormPrior {
        global_variance: context.condition_variance.max(1e-4),
        skill_variance: context.skill_condition_variance.max(1e-4),
        by_key: context.by_key.iter().map(|(k, b)| (*k, (b.global_variance, b.skill_variances.clone()))).collect(),
        noise_variance: (fit.sd_all * fit.sd_all).max(1e-4),
    };
    let warmup = Arc::new(Model::new(&fit.warmup, args.slopes, prior, c.families.len()));
    let model = Arc::new(PredModel::fitted(c.clone(), native, levels.clone(), &fit, &warmup, shown));
    Target { fit, warmup, model }
}

/// Recommender._fit's models. The current session's plays are left out: form reads them live.
pub fn fit(db: &Db, public: &Arc<Public>, calc: &str, c: &Arc<Consts>) -> Result<Fitted> {
    let mut source = FeatureSource { public: public.clone(), calc: calc.to_string(), cache: HashMap::default() };
    let rows = evidence_rows(db, &mut source, c)?;
    let current = now();
    let session = Session::new(load_events(db, ((current - HISTORY_S) / 60.).floor() * 60.)?, current, None, c.clone(), Arc::new(family_transfer(c)));
    let begin = session.events.iter().filter(|e| e.kind == "start").map(|e| e.t).fold(f64::INFINITY, f64::min);
    let month = now_months();
    let historical: Vec<&Evidence> = rows.rows.iter().filter(|r| r.ts.is_none_or(|t| t == 0. || t < begin)).collect();
    let levels = play_levels(&historical, month);
    let shown = fit_shown(db, begin)?;
    let pp = fit_target(c, public, &public.meta.pop, &historical, &rows.feats, &levels, false, month, shown);
    let display = public.meta.pop_lazer.as_ref().map(|pop| fit_target(c, public, pop, &historical, &rows.feats, &levels, true, month, (0., 1.)));
    Ok(Fitted { rows, levels, pp, display, shown })
}
