//! Player ability in chart-difficulty units, using the existing personal model. A skill estimate is the difficulty giving
//! 94% displayed accuracy on a diverse set of that player's representative chart shapes. It is not a raw regression
//! coefficient, an average of personal bests, or a claimed population percentile. No chart analyses or model fits are
//! started by this page.
use crate::dans;
use crate::engine::{Engine, Now};
use crate::evidence::{personal_row, Evidence, FeatureSource};
use crate::history::play_history;
use crate::catalog::SUPPORTED_KEYS;
use ms_calc::features::{displayed_feature, displayed_value, feature_display_factor};
use ms_calc::py::round_digits;
use ms_calc::val::Dict;
use ms_data::store::{Db, Result};
use ms_model::personal::{self, Fit, FATIGUE_ONSET};
use ms_model::predict::{Feats, HashMap, PredModel};
use ms_model::skills;
use ms_model::warmup::Model;
use std::collections::BTreeMap;
use std::sync::Arc;

pub const TARGET: f64 = 0.94;
const FORM_SESSIONS: usize = 12;
const CALIBRATION_HALF_LIFE: f64 = 6.; // months before the keymode's last play
const CALIBRATION_MIN: f64 = 20.;      // effective recent plays for a skill's own calibration; fewer use the keymode's
const PEAK_SHARE: f64 = 0.5;           // of a player's plays, measured against expectation; a raw accuracy cut would keep only the easy charts
const GRIND: f64 = -0.15;              // log(1 − acc) a ground course gains on those plays: the user's 7K history lands on Gamma, their stated level
const THIN_LINE: f64 = 5.;              // effective recent plays that can still carry the main keymode's slope
const MIN_RECENT: f64 = 3.;             // charts, halved per year: below it and without a line of its own a keymode has no Level
const MIN_CHARTS: usize = 10;          // charts before a skill gets a number (4K LN from 3 charts read "Yume+", user 2026-10-30)
const HISTORY_BW: f64 = 1.;            // months: Gaussian smoothing of the monthly history (a peak month stays visible)
const UNSEEN: &str = "profile:unseen";

#[derive(Clone, Default)]
pub struct Group {
    pub value: Option<f64>,
    pub low: f64,
    pub high: f64,
    pub current: Option<f64>,
    pub slope: f64,
    pub display_factor: f64,
    pub maps: usize,
    pub plays: usize,
    pub newest: f64,
    pub confidence: &'static str,
    pub references: usize,
    pub examples: Vec<(String, f64, f64)>, // title, rate, difficulty
    pub tested_to: f64,
    pub extrapolated: bool,
    pub activation: f64,
}

#[derive(Clone, Default)]
pub struct FormSession {
    pub t: f64,
    pub hours: f64,
    pub plays: Vec<(f64, f64, &'static str)>, // clock, value, mark
}

#[derive(Clone, Default)]
pub struct FormHistory {
    pub sessions: Vec<FormSession>,
    pub warmup: f64,
    pub fatigue: f64,
}

#[derive(Clone, Default)]
pub struct Page {
    pub keys: i64,
    pub groups: BTreeMap<String, Group>,
    pub order: Vec<String>,
    pub history: Vec<(i64, f64)>,
    pub now: f64,
    pub dan_at: HashMap<String, (String, Option<f64>)>,
    pub dan_scale: std::collections::HashMap<String, f64>,
    pub form: Option<FormHistory>,
    pub evidence: (usize, usize, f64),
    pub peak: Option<(i64, f64)>,
    pub maps: usize,
    pub plays: usize,
    pub mash_maps: usize,
    pub diagnostics: Vec<(String, f64, usize)>, // skill, delta, maps
    pub session: bool,
}

pub struct Profile {
    pub pages: BTreeMap<i64, Page>,
}

pub struct Current {
    pub target: f64,
    pub pages: BTreeMap<i64, Page>,
    pub phase: String,
}

/// Weighted quantile of (value, weight), as the first pair past the share.
fn quantile(values: &[f64], weights: &[f64], q: f64) -> f64 {
    let mut pairs: Vec<(f64, f64)> = values.iter().copied().zip(weights.iter().copied()).collect();
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    let total: f64 = pairs.iter().map(|p| p.1).sum();
    let mut acc = 0.;
    for (value, weight) in &pairs {
        acc += weight;
        if acc >= total * q {
            return *value;
        }
    }
    pairs.last().map_or(0., |p| p.0)
}

fn unseen(model: &PredModel, f: &Feats, overall: f64) -> f64 {
    let mut c = f.clone();
    c.overall = overall;
    model.personal(&c, UNSEEN, None, 1.).0
}

/// (the rating at target_y, the slope there). Shape, timing-window and sustained-demand context stay fixed; only the
/// calibrated difficulty coordinate is inverted. Unseen-chart predictions omit a map's learned affinity.
fn capacity(model: &PredModel, f: &Feats, target_y: f64) -> (f64, f64) {
    let (mut lo, mut hi) = (0.1f64.ln(), 60f64.ln());
    for _ in 0..18 {
        let mid = (lo + hi) / 2.; // 6.4 / 2^19: 0.001 % of the rating
        if unseen(model, f, mid.exp()) < target_y { lo = mid } else { hi = mid }
    }
    let value = ((lo + hi) / 2.).exp();
    let (lower, upper) = (unseen(model, f, value * 0.99), unseen(model, f, value * 1.01));
    (value, 0.25f64.max((upper - lower) / (1.01f64 / 0.99).ln()))
}

/// (c, d) of the weighted least-squares line y = c + d·x over (x, y, w), or None.
fn fit_line(points: &[(f64, f64, f64)]) -> Option<(f64, f64)> {
    let total: f64 = points.iter().map(|p| p.2).sum();
    if total < CALIBRATION_MIN {
        return None;
    }
    line(points)
}

/// The keymode whose slope `keys` may take: the most played above it, if played more than `keys` itself. Skill carries down the
/// keymodes, never up (user 2026-11-01: "7k skill translates to 4k but 4k skill means nothing in 7k", "same for 7k->10k").
fn slope_source(keys: i64, plays: &[(i64, usize)]) -> Option<i64> {
    let own = plays.iter().find(|p| p.0 == keys).map_or(0, |p| p.1);
    plays.iter().filter(|p| p.0 > keys && p.1 > own).max_by_key(|p| p.1).map(|p| p.0)
}

/// `fit` no flatter than `slope`, a more played higher keymode's: a side keymode's few plays sit near one difficulty, so its own
/// slope reads flat and the rating runs off (the user's 6K at d 0.39: Level over 7K, the dan past the top of the ladder).
/// The line stays through the plays. Too few plays for a fit of their own still get one, down to THIN_LINE.
fn steepened(points: &[(f64, f64, f64)], fit: Option<(f64, f64)>, slope: Option<f64>) -> Option<(f64, f64)> {
    let total: f64 = points.iter().map(|p| p.2).sum();
    let Some(slope) = slope.filter(|s| fit.map_or(total >= THIN_LINE, |f| f.1 < *s)) else { return fit };
    let mean = |i: fn(&(f64, f64, f64)) -> f64| points.iter().map(|p| i(p) * p.2).sum::<f64>() / total;
    Some((mean(|p| p.1) - slope * mean(|p| p.0), slope))
}

/// The line through the better half of a player's plays, moved by GRIND. A dan is passed after grinding it, not on a first
/// try, so a line through every play read a 7K Gamma player as 9th (user 2026-11-01: "the upper 50%" and "shift the values").
fn peak_line(points: &[(f64, f64, f64)], slope: Option<f64>) -> Option<(f64, f64)> {
    let (c, d) = steepened(points, fit_line(points), slope)?;
    let left: Vec<f64> = points.iter().map(|p| p.1 - c - d * p.0).collect();
    let cut = quantile(&left, &points.iter().map(|p| p.2).collect::<Vec<_>>(), PEAK_SHARE);
    let best: Vec<(f64, f64)> = points.iter().zip(&left).filter(|x| *x.1 <= cut).map(|x| (x.0.2, *x.1)).collect();
    let kept: Vec<(f64, f64, f64)> = points.iter().zip(&left).filter(|x| *x.1 <= cut).map(|x| *x.0).collect();
    // Too few plays to set a slope (iess's 4K read 15.9 on five): keep the line's, moved up to them.
    let (c, d) = steepened(&kept, fit_line(&kept), slope).unwrap_or((c + best.iter().map(|b| b.0 * b.1).sum::<f64>() / best.iter().map(|b| b.0).sum::<f64>(), d));
    Some((c + GRIND, d))
}

fn line(points: &[(f64, f64, f64)]) -> Option<(f64, f64)> {
    let total: f64 = points.iter().map(|p| p.2).sum();
    let mean = |g: &dyn Fn(&(f64, f64, f64)) -> f64| points.iter().map(|p| g(p) * p.2).sum::<f64>() / total;
    let (mx, my) = (mean(&|p| p.0), mean(&|p| p.1));
    let var = mean(&|p| (p.0 - mx).powi(2));
    if var < 1e-6 {
        return None;
    }
    let d = mean(&|p| (p.0 - mx) * (p.1 - my)) / var;
    (d > 0.2).then_some((my - d * mx, d))
}

/// The fractional tier where the tiers' log(1 − acc) crosses `goal`: a tier below the easiest or above the hardest when
/// every known one is on the same side.
fn crossing(y: &[Option<f64>], goal: f64) -> Option<f64> {
    let known: Vec<(usize, f64)> = y.iter().enumerate().filter_map(|(j, v)| v.map(|v| (j, v))).collect();
    let Some(j) = known.iter().filter(|x| x.1 <= goal).map(|x| x.0).max() else { // none at or above the goal
        return known.first().map(|k| k.0 as f64 - 1.);
    };
    let Some(&(i, v)) = known.iter().find(|k| k.0 > j) else { return Some(j as f64 + 1.) };
    let at = y[j].unwrap();
    Some(j as f64 + (i - j) as f64 * (goal - at) / (v - at))
}

struct Ctx<'a> {
    model: &'a PredModel,
    fit: &'a Fit,
    warmup: &'a Arc<Model>,
    rec: &'a Engine,
    dicts: HashMap<(String, u64), Arc<Dict>>,
}

impl Ctx<'_> {
    fn feats(&self, r: &Evidence) -> &Feats {
        &self.rec.r.fitted.rows.feats[r.f]
    }

    fn dict(&self, r: &Evidence) -> Option<&Arc<Dict>> {
        self.dicts.get(&(r.chart.clone(), r.rate.to_bits()))
    }

    /// {group: (c, d)}: the player's recent completed lazer results against the unseen-chart baseline, displayed
    /// y ≈ c + d·baseline. The baseline alone put 4K at 9.1 (≈ Delta) where the player's plays reached 94 % around 7.5
    /// (user 2026-10-30): it leaves out warm-up, fatigue and chart choice, and its difficulty slope was twice the player's.
    /// Groups without enough plays use the keymode's line.
    fn calibration(&self, rows: &[&Evidence], groups: &[String], line: &dyn Fn(&[(f64, f64, f64)]) -> Option<(f64, f64)>) -> HashMap<String, (f64, f64)> {
        let own: Vec<&&Evidence> = rows.iter().filter(|r| r.src != "public" && r.completed).collect();
        let Some(last) = own.iter().map(|r| r.t).max_by(f64::total_cmp) else { return HashMap::default() };
        let mut points: HashMap<String, Vec<(f64, f64, f64)>> = HashMap::default();
        for r in own {
            let w = 0.5f64.powf((last - r.t) / CALIBRATION_HALF_LIFE);
            if w < 0.02 {
                continue;
            }
            let f = self.feats(r);
            let point = (self.model.personal(f, UNSEEN, None, 1.).0, r.y_lazer, w);
            points.entry("overall".to_string()).or_default().push(point);
            for g in self.dict(r).map_or_else(Vec::new, |d| skills::demanded(d)) {
                if groups.iter().any(|x| x == g) {
                    points.entry(g.to_string()).or_default().push(point);
                }
            }
        }
        let Some(base) = points.get("overall").and_then(|p| line(p)) else { return HashMap::default() };
        groups.iter().map(|g| (g.clone(), points.get(g).and_then(|p| line(p)).unwrap_or(base))).collect()
    }

    /// Rating on the series' dan ladder where the player's peak predicted accuracy on its dan songs crosses dans::PEAK.
    /// The level's own rating sat ~2 tiers high on the dans: it is a median over the player's chart shapes, and 4K ones
    /// ranged from ≈ Beta (stable bests) to ≈ Delta (2026 lazer plays).
    fn dan_rating(&self, series: &str, line: Option<(f64, f64)>) -> Option<f64> {
        let (tiers, songs, (c, d)) = (dans::tiers(series)?, dans::songs(series, &self.rec.c), line?);
        let y: Vec<Option<f64>> = tiers.iter().map(|(tier, _)| songs.get(tier).filter(|f| !f.is_empty()).map(|fs| {
            let mut v: Vec<f64> = fs.iter().map(|f| c + d * self.model.personal(f, UNSEEN, None, 1.).0).collect();
            v.sort_by(f64::total_cmp);
            let n = v.len();
            if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2. }
        })).collect();
        dans::rating_at(series, crossing(&y, (1f64 - dans::PEAK).ln())?)
    }

    /// [(month, ability)]: today's overall ability moved by how much better or worse each month's recorded plays went
    /// against their predictions. Lazer keeps every play; website/stable top-100s keep only bests and are left out, so a
    /// new install shows little or no history. Months with less than 20 (smoothed) plays are left out.
    fn history(&self, keys: i64, overall: Option<&Group>) -> Vec<(i64, f64)> {
        const SUPPORT: f64 = 20.;
        let (Some(g), Some((start, se, sw))) = (overall.filter(|g| g.value.is_some()), self.fit.monthly.get(&keys)) else { return Vec::new() };
        let n = sw.len();
        let smooth = |i: usize, v: &[f64]| (0..n).map(|j| (-0.5 * ((i as f64 - j as f64) / HISTORY_BW).powi(2)).exp() * v[j]).sum::<f64>();
        let months: Vec<usize> = (0..n).filter(|&i| smooth(i, sw) >= SUPPORT && sw[i] > 0.).collect();
        let Some(&last) = months.last() else { return Vec::new() };
        let level = |i: usize| smooth(i, se) / smooth(i, sw);
        let now = level(last); // the latest played month is today's ability
        months.into_iter().map(|i| (start + i as i64, g.value.unwrap() * ((now - level(i)) / g.slope).exp())).collect()
    }

    /// The last sessions play by play: accuracy points against the model's rested expectation (at 95 %), each play marked
    /// warm-up, peak, bad or fading; the fitted warm-up cost and fatigue per hour beside them.
    fn form_history(&self, keys: i64, local: &[&Evidence]) -> Option<FormHistory> {
        if !self.fit.keys.contains(&keys) {
            return None;
        }
        let c = &self.rec.c;
        let rows: Vec<personal::Row> = local.iter().map(|r| personal_row(r, r.y, 0.)).collect();
        let transfer = crate::recommender::family_transfer(c);
        let steps = personal::steps(c, &rows, &self.rec.r.fitted.rows.feats, &transfer, personal::GAP, personal::COOLING);
        let mut ids: Vec<i64> = steps.iter().map(|s| s.session).collect();
        ids.sort_unstable();
        ids.dedup();
        let keep: std::collections::HashSet<i64> = ids[ids.len().saturating_sub(FORM_SESSIONS)..].iter().copied().collect();
        let points = |shift: f64| 100. * 0.05 * (1. - shift.clamp(-3., 3.).exp());
        let p = self.warmup.parameters(keys);
        let unit = if p.log_rate { 3. } else { 1. };
        let mut sessions: Vec<(i64, Vec<&personal::Step>)> = Vec::new();
        for s in steps.iter().filter(|s| keep.contains(&s.session)) {
            match sessions.iter_mut().find(|x| x.0 == s.session) {
                Some(x) => x.1.push(s),
                None => sessions.push((s.session, vec![s])),
            }
        }
        if sessions.is_empty() {
            return None;
        }
        let out = sessions.into_iter().map(|(_, plays)| {
            let at = |s: &personal::Step| local[s.row];
            let first = at(plays[0]).ts.unwrap_or(0.) - plays[0].seconds;
            let values: Vec<(f64, f64, f64)> = plays.iter().map(|s| {
                let r = at(s);
                let m = self.model.personal(self.feats(r), &r.chart, r.b, round_digits(r.rate, 3)).0;
                let clock = (r.ts.unwrap_or(0.) - s.seconds - first) / 60.;
                let cold = self.warmup.penalty(c, keys, Some(&c.families[s.family]), s.minutes, s.seconds, None, clock) - self.warmup.fatigue(keys, clock) * unit;
                (clock, points(r.y_lazer - m), cold)
            }).collect();
            let n = values.len() as f64;
            let centre = values.iter().map(|v| v.1).sum::<f64>() / n;
            let spread = 0.3f64.max((values.iter().map(|v| (v.1 - centre).powi(2)).sum::<f64>() / n).sqrt());
            let marked: Vec<(f64, f64, &'static str)> = values.iter().map(|&(clock, value, cold)| {
                let mark = if cold > 0.08 { "warm-up" } else if value < centre - spread { "bad" } else if clock >= FATIGUE_ONSET && value < centre { "fading" } else if value > centre + spread { "peak" } else { "" };
                (round_digits(clock, 1), round_digits(value, 2), mark)
            }).collect();
            FormSession { t: first, hours: marked[marked.len() - 1].0 / 60., plays: marked }
        }).collect();
        let mean = p.amplitudes.iter().sum::<f64>() / p.amplitudes.len() as f64;
        Some(FormHistory { sessions: out, warmup: -points(unit * (p.session + mean)), fatigue: -points(unit * p.fatigue) })
    }
}

pub fn build(rec: &Engine, db: &Db) -> Result<Profile> {
    let f = &rec.r.fitted;
    let (a, b) = f.shown;
    let target = f.display.as_ref().unwrap_or(&f.pp);
    let target_y = if f.display.is_some() { (1f64 - TARGET).ln() } else { ((1f64 - TARGET).ln() - a) / b.max(0.1) };
    let usable: Vec<&Evidence> = f.rows.rows.iter().filter(|r| SUPPORTED_KEYS.contains(&r.keys) && !r.rd && f.rows.feats[r.f].overall > 0.1).collect();
    // The representative charts' features, once: Feats keep no display fields.
    let mut source = FeatureSource { public: rec.public.clone(), calc: rec.source.calc.clone(), cache: HashMap::default() };
    let mut dicts: HashMap<(String, u64), Arc<Dict>> = HashMap::default();
    source.prefetch(db, usable.iter().filter_map(|r| Some(source_key(r.sha.as_deref()?, r.rate))))?;
    for r in &usable {
        let id = (r.chart.clone(), r.rate.to_bits());
        if !dicts.contains_key(&id) {
            let md5 = (r.chart.len() == 32).then_some(r.chart.as_str());
            if let (Some(d), _) = source.of(db, r.sha.as_deref(), md5, r.b, Some(r.rate))? {
                dicts.insert(id, d);
            }
        }
    }
    let ctx = Ctx { model: &target.model, fit: &target.fit, warmup: &target.warmup, rec, dicts };
    let month = crate::evidence::now_months();
    let mut pages = BTreeMap::new();
    let of_keys = |keys: i64| usable.iter().filter(|r| r.keys == keys).copied().collect::<Vec<&Evidence>>();
    let plays: Vec<(i64, usize)> = SUPPORTED_KEYS.filter(|k| ctx.fit.keys.contains(k)).map(|k| (k, of_keys(k).iter().filter(|r| r.src != "public" && r.completed).count())).collect();
    let overall_only = ["overall".to_string()];
    let slope_of = |k: i64, fit: &dyn Fn(&[(f64, f64, f64)]) -> Option<(f64, f64)>| ctx.calibration(&of_keys(k), &overall_only, fit).get("overall").map(|l| l.1);
    for keys in SUPPORTED_KEYS {
        let rows = of_keys(keys);
        let source = slope_source(keys, &plays);
        let slope = source.and_then(|k| slope_of(k, &fit_line));
        let peak_slope = source.and_then(|k| slope_of(k, &|p| peak_line(p, None)));
        // Many retries/rates of one chart are not independent coverage. Choose the highest played difficulty as its
        // representative, not its best accuracy.
        let mut charts: HashMap<&str, &Evidence> = HashMap::default();
        let overall = |r: &Evidence| ctx.feats(r).overall;
        for &row in &rows {
            let hit = charts.get(row.chart.as_str()).is_none_or(|old| overall(row) > overall(old));
            if hit {
                charts.insert(&row.chart, row);
            }
        }
        let mut ratings: Vec<f64> = charts.values().map(|r| overall(r)).collect();
        ratings.sort_by(f64::total_cmp);
        let floor = if ratings.is_empty() { 0. } else {
            let at = 0.65 * (ratings.len() - 1) as f64;
            let (i, x) = (at.floor() as usize, at.fract());
            (ratings[i] + x * (ratings[(i + 1).min(ratings.len() - 1)] - ratings[i])) * 0.6
        };
        let representatives: Vec<&Evidence> = charts.values().filter(|r| overall(r) >= floor).copied().collect();
        // Each chart's display numbers once, not once per skill group it belongs to.
        let shown = |r: &Evidence| ctx.dict(r).map_or(0., |d| displayed_feature(d, &rec.prm));
        let factor = |r: &Evidence| ctx.dict(r).map_or(1., |d| feature_display_factor(d, &rec.prm));
        let mut names = vec!["overall".to_string()];
        names.extend(skills::dimensions(keys).iter().map(|n| n.key.clone()));
        let known = ctx.fit.keys.contains(&keys);
        let calibration = if known { ctx.calibration(&rows, &names, &|p| steepened(p, fit_line(p), slope)) } else { HashMap::default() };
        let mut groups: BTreeMap<String, Group> = BTreeMap::new();
        let mut capacities: std::collections::HashMap<(String, u64, u64, u64), (f64, f64)> = Default::default();
        for group in &names {
            let mut relevant: Vec<&Evidence> = representatives.iter().filter(|r| group == "overall" || ctx.dict(r).is_some_and(|d| skills::demanded(d).contains(&group.as_str()))).copied().collect();
            let newest = relevant.iter().map(|r| r.ts.unwrap_or(0.)).fold(0., f64::max);
            let count = relevant.len();
            if count < MIN_CHARTS || !known {
                groups.insert(group.clone(), Group { maps: count, newest, confidence: "Not enough evidence", display_factor: 1., ..Group::default() });
                continue;
            }
            // Decade-old website bests on easy charts extrapolate: the user's 5K read 10.2 from 99 % scores on 3-stars.
            if !calibration.contains_key(group) && relevant.iter().map(|r| 0.5f64.powf((month - r.t).max(0.) / 12.)).sum::<f64>() < MIN_RECENT {
                groups.insert(group.clone(), Group { maps: count, newest, confidence: "No recent plays", display_factor: 1., ..Group::default() });
                continue;
            }
            // Quantile-spaced shapes, not the most retried/favourite charts.
            relevant.sort_by(|x, y| overall(x).total_cmp(&overall(y)).then_with(|| x.chart.cmp(&y.chart)));
            let n = count.min(32);
            let chosen: Vec<&Evidence> = (0..n).map(|i| relevant[((count - 1) as f64 * i as f64 / (n.max(2) - 1) as f64) as usize]).collect();
            let (c, d) = calibration.get(group).copied().unwrap_or((0., 1.));
            let (mut values, mut slopes, mut weights, mut display) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for row in &chosen {
                let id = (row.chart.clone(), row.rate.to_bits(), c.to_bits(), d.to_bits());
                let (value, slope) = *capacities.entry(id).or_insert_with(|| capacity(ctx.model, ctx.feats(row), (target_y - c) / d));
                values.push(value);
                slopes.push(d * slope); // per unit of the player's displayed y
                // Ability has no single chart shape; the sample's median display correction, its fitted value intact.
                display.push(factor(row));
                weights.push(0.5f64.powf((month - row.t).max(0.) / 12.));
            }
            let (value, slope) = (quantile(&values, &weights, 0.5), quantile(&slopes, &weights, 0.5));
            let shape_spread = (quantile(&values, &weights, 0.85).ln() - quantile(&values, &weights, 0.15).ln()) / 2.;
            let level_var = ctx.fit.level_var.get(&keys).copied().unwrap_or(0.25);
            // Conservative spread: shape ambiguity and sparse coverage. Not a calibrated 95% confidence interval.
            let spread = (level_var / 0.25f64.max(slope * slope) + shape_spread.powi(2) + 0.12f64.powi(2) / count.min(25) as f64).sqrt();
            let eff: f64 = relevant.iter().map(|r| 0.5f64.powf((month - r.t).max(0.) / 12.)).sum();
            let confidence = if count >= 40 && eff >= 20. { "Strong history" } else if count >= 10 { "Moderate history" } else { "Limited history" };
            let members: std::collections::HashSet<&str> = relevant.iter().map(|r| r.chart.as_str()).collect();
            let highest = relevant.iter().map(|r| overall(r)).fold(0., f64::max);
            let examples = relevant[count.saturating_sub(3)..].iter().map(|r| {
                let title = rec.r.installed.get(&r.chart).map(|i| i.name()).or_else(|| r.b.and_then(|b| rec.public.map_info(b)).map(|m| m.file.clone())).unwrap_or_else(|| "Unavailable chart title".to_string());
                (title, r.rate, shown(r))
            }).collect();
            let mut g = Group { value: Some(value), low: value * (-spread).exp(), high: value * spread.exp(), slope, display_factor: quantile(&display, &weights, 0.5), maps: count,
                                plays: rows.iter().filter(|r| members.contains(r.chart.as_str())).count(), newest, confidence, references: chosen.len(), examples,
                                tested_to: relevant.iter().map(|r| shown(r)).fold(0., f64::max), ..Group::default() };
            if value > highest * 1.35 {
                // Easy/near-perfect historical scores do not identify the player's limit. Do not present a huge
                // extrapolated rating as a measured strength merely because the solver found it.
                g.value = None;
                g.confidence = "Limit not measured";
                g.extrapolated = true;
            }
            groups.insert(group.clone(), g);
        }
        let peak = if known { ctx.calibration(&rows, &["overall".to_string(), "ln".to_string()], &|p| peak_line(p, peak_slope)) } else { HashMap::default() };
        let mut dan_at = HashMap::default();
        for (kind, anchor) in [("rice", "overall"), ("ln", "ln")] {
            if let Some(series) = dans::series(keys, kind) {
                if groups.get(anchor).is_some_and(|g| g.value.is_some()) {
                    dan_at.insert(kind.to_string(), (anchor.to_string(), ctx.dan_rating(series, peak.get(anchor).copied())));
                }
            }
        }
        let history = ctx.history(keys, groups.get("overall"));
        let local: Vec<&Evidence> = rows.iter().filter(|r| r.src != "public").copied().collect();
        let form = ctx.form_history(keys, &local);
        pages.insert(keys, Page { keys, order: names, history: history.clone(), now: month, dan_at, form, evidence: (local.len(), rows.len() - local.len(), rows.iter().map(|r| r.ts.unwrap_or(0.)).fold(0., f64::max)),
                                  peak: history.iter().copied().max_by(|a, b| a.1.total_cmp(&b.1)), maps: charts.len(), plays: rows.len(),
                                  mash_maps: charts.values().filter(|r| ctx.dict(r).and_then(|d| d.dict("sk")).is_some_and(|sk| sk.fget("mash", 0.) >= 0.3)).count(), groups, ..Page::default() });
    }
    // Only frozen pre-play predictions can support a model-error diagnosis. Collapse retries by exact chart, and require
    // several charts before a label.
    let mut residuals: HashMap<(i64, String), HashMap<String, Vec<f64>>> = HashMap::default();
    for row in play_history(db, 300, false, None)? {
        let (Some(expected), Some(keys), Some(features)) = (row.expected, row.keys, row.prediction.get("features").filter(|f| f.is_object())) else { continue };
        if !pages.contains_key(&keys) {
            continue;
        }
        let ms_calc::val::Val::Dict(f) = ms_data::codec::from_json(features) else { continue };
        for group in skills::matched_families(&f) {
            residuals.entry((keys, group.to_string())).or_default().entry(row.sha256.clone().unwrap_or_default()).or_default().push(100. * (row.actual - expected));
        }
    }
    for (keys, page) in &mut pages {
        let mut found: Vec<(String, f64, usize)> = Vec::new();
        for ((k, group), charts) in residuals.iter().filter(|x| x.0 .0 == *keys && x.1.len() >= 4) {
            let _ = k;
            let errors: Vec<f64> = charts.values().map(|e| e.iter().sum::<f64>() / e.len() as f64).collect();
            let n = errors.len() as f64;
            let delta = errors.iter().sum::<f64>() / n;
            let se = (errors.iter().map(|e| (e - delta).powi(2)).sum::<f64>() / n).sqrt() / n.sqrt();
            if delta.abs() > 0.75f64.max(1.5 * se) {
                found.push((group.clone(), delta, charts.len()));
            }
        }
        found.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
        found.truncate(3);
        page.diagnostics = found;
    }
    Ok(Profile { pages })
}

fn source_key(sha: &str, rate: f64) -> String {
    format!("{sha}@{:.3}|{}", round_digits(rate, 3), crate::engine::CALC_ID)
}

/// Cheap session overlay. This never alters the persistent skill estimate.
pub fn current(profile: &Profile, n: &Now, rec: &Engine) -> Current {
    let f = &rec.r.fitted;
    let warmup = &f.display.as_ref().unwrap_or(&f.pp).warmup;
    let view = &n.view;
    let mut pages = BTreeMap::new();
    for (keys, old) in &profile.pages {
        let keys = *keys;
        let mut page = old.clone();
        for (group, row) in &mut page.groups {
            let skill = if group == "overall" { None } else { skills::session_skill(group, keys) };
            let family = skill.map(|s| rec.c.family(s));
            row.activation = view.activation(keys, family);
            let od = 8.;
            if let Some(value) = row.value {
                let cold = if view.warm_flag { 0. } else { warmup.penalty(&rec.c, keys, skill, row.activation, 0., None, view.session_clock()) };
                let correction = view.correction(keys, skill) + cold;
                row.current = Some(displayed_value(value * (-correction / row.slope).exp(), keys, od, row.display_factor, &rec.prm));
                row.value = Some(displayed_value(value, keys, od, row.display_factor, &rec.prm));
                row.low = displayed_value(row.low, keys, od, row.display_factor, &rec.prm);
                row.high = displayed_value(row.high, keys, od, row.display_factor, &rec.prm);
            }
        }
        let factor = old.groups.get("overall").map_or(1., |g| g.display_factor);
        let shown = |v: f64| displayed_value(v, keys, 8., factor, &rec.prm);
        // A displayed level × scale = the dan rating it matches on the player's predicted dan-song accuracy.
        page.dan_scale = old.dan_at.iter().filter_map(|(kind, (anchor, rating))| Some((kind.clone(), (*rating)? / page.groups.get(anchor)?.value?))).collect();
        page.history = old.history.iter().map(|&(m, v)| (m, shown(v))).collect();
        page.peak = old.peak.map(|(m, v)| (m, shown(v)));
        // Before a session is warm the session marker only shows the cold start, not the player's form.
        page.session = view.activation(keys, None) >= 1.5;
        pages.insert(keys, page);
    }
    Current { target: TARGET, pages, phase: n.session.phase().to_string() }
}

#[cfg(test)]
mod tests {
    use super::{fit_line, peak_line, slope_source, steepened};
    use super::crossing;

    #[test]
    fn the_dan_line_follows_the_best_plays_at_every_difficulty() {
        // Per difficulty: one play 0.6 worse than expected, one 0.6 better.
        let points = |n: usize| -> Vec<(f64, f64, f64)> {
            (0..n).flat_map(|i| {
                let x = -4. + i as f64 * 0.05;
                [(x, x + 0.6, 1.), (x, x - 0.6, 1.)]
            }).collect()
        };
        let (all, _) = fit_line(&points(40)).unwrap();
        let (c, d) = peak_line(&points(40), None).unwrap();
        assert!((d - 1.).abs() < 1e-9 && (c - (all - 0.6 + super::GRIND)).abs() < 1e-9, "{c} {d}");
        // Too few better plays for a slope: the whole line's, moved up by them.
        let (full, slope) = fit_line(&points(15)).unwrap();
        let (c, d) = peak_line(&points(15), None).unwrap();
        assert!(d == slope && (c - (full - 0.6 + super::GRIND)).abs() < 1e-9, "{c} {d}");
    }

    #[test]
    fn a_slope_only_comes_from_a_higher_keymode_played_more() {
        let plays = [(4, 300), (6, 50), (7, 9000), (8, 20), (10, 4000)];
        assert_eq!(slope_source(4, &plays), Some(7));
        assert_eq!(slope_source(6, &plays), Some(7));
        assert_eq!(slope_source(7, &plays), None, "the most played keymode keeps its own");
        assert_eq!(slope_source(8, &plays), Some(10), "7K is below it");
        assert_eq!(slope_source(10, &plays), None, "nothing above it");
        assert_eq!(slope_source(7, &[(4, 900), (7, 30)]), None, "4K skill means nothing in 7K");
    }

    #[test]
    fn a_side_keymode_takes_the_main_slope_through_its_own_plays() {
        let flat: Vec<(f64, f64, f64)> = (0..40).map(|i| (-3. + i as f64 * 0.01, -3. + i as f64 * 0.003, 1.)).collect();
        let (c, d) = fit_line(&flat).map_or((0., 0.), |l| l);
        assert!(d < 0.31, "{d}");
        let (c2, d2) = steepened(&flat, Some((c, d)), Some(0.8)).unwrap();
        let (x, y) = (flat.iter().map(|p| p.0).sum::<f64>() / 40., flat.iter().map(|p| p.1).sum::<f64>() / 40.);
        assert!(d2 == 0.8 && (c2 + d2 * x - y).abs() < 1e-9, "{c2} {d2}");
        assert_eq!(steepened(&flat, Some((c, d)), Some(0.1)), Some((c, d)), "a steeper keymode keeps its own");
        assert_eq!(steepened(&flat[..3], None, Some(0.8)), None, "three plays are no line");
        assert!(steepened(&flat[..6], None, Some(0.8)).is_some());
    }

    #[test]
    fn the_dan_is_where_predicted_accuracy_crosses_the_target() {
        let y = |x: f64| x.ln();
        // 94 % at log(1 − acc) = ln .06: tier 1 is easier, tier 2 harder, linear between them.
        let goal = y(0.06);
        let at = crossing(&[Some(y(0.02)), Some(y(0.04)), Some(y(0.08))], goal).unwrap();
        assert!((at - (1. + (goal - y(0.04)) / (y(0.08) - y(0.04)))).abs() < 1e-12);
        assert_eq!(crossing(&[Some(y(0.01)), Some(y(0.02)), Some(y(0.03))], goal), Some(3.)); // beyond the top
        assert_eq!(crossing(&[Some(y(0.1)), Some(y(0.2))], goal), Some(-1.)); // below the first
        assert_eq!(crossing(&[None, Some(y(0.02)), None, Some(y(0.08))], goal), Some(1. + 2. * (goal - y(0.02)) / (y(0.08) - y(0.02))));
        assert_eq!(crossing(&[None, None], goal), None);
    }
}
