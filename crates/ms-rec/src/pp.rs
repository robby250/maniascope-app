//! PP: lazer mania pp, the account ledger, per-chart gain of the next attempt, and the broad PP playlist. Selection
//! only: ratings, central accuracy and the exact weighted-account gain come from the shared model.
use crate::catalog::{rate_label_pp, Inst, Pick, SUPPORTED_KEYS};
use crate::evidence::{Evidence, Rows};
use crate::recommender::Fitted;
use ms_calc::background::background_map;
use ms_calc::py::round_digits;
use ms_calc::val::Val;
use ms_data::paths;
use ms_data::public::Public;
use ms_data::store::{acc_lazer, parse_stats, Db, Result, RANKED_STATUS};
use ms_model::predict::{Consts, Feats, HashMap, PredModel, PredSession, Prediction, BAND_Z};
use ms_model::session::{Session, EXPOSURE_S};
use rusqlite::OptionalExtension;
use std::collections::{HashSet};
use std::path::PathBuf;
use std::sync::Arc;

pub const PESSIMISM: f64 = 0.25;   // model uncertainty shifts the mean (in its sd), never widens the upside
// One step is half an attempt sd of challenge: small enough that noise can't swing the target, large enough to cross
// the user's safe→peak range within a session.
const PUSH_STEP: f64 = 0.5;
const Z_Q: [f64; 21] = [-1.9807523966472786, -1.4652337926855223, -1.1797611176118603, -0.9674215661017014, -0.7916386077433746, -0.6374841609623769,
    -0.497200570681554, -0.3661063568005697, -0.24104039388602683, -0.11964811303984202, 0.0, 0.11964811303984216, 0.24104039388602683, 0.3661063568005697,
    0.497200570681554, 0.637484160962377, 0.7916386077433746, 0.9674215661017014, 1.1797611176118603, 1.4652337926855223, 1.9807523966472784];
pub const VARIANTS: [(f64, &str); 3] = [(0.75, "HT"), (1., "NM"), (1.5, "DT")];

/// PP tab settings (user 2026-10-30); the defaults are the former fixed bars.
#[derive(Clone, Debug, PartialEq)]
pub struct PpSettings {
    pub min_gain: f64,   // below this expected gain a PP map is not worth offering (user 2026-10-09: .1 left 0–75 maps; was .001)
    pub min_chance: f64,
    pub ht: bool,
    pub ht_chance: f64,  // HT only for a real gain, at least an even chance (user 2026-10-03)
    pub ht_gain: f64,
    pub dt: bool,
    pub push: f64,       // added to readiness (−.5 safe … +.5 push)
    pub cooldown_days: f64, // offers and plays older than this no longer hold a map back
    pub top100: bool,
}

impl Default for PpSettings {
    fn default() -> Self {
        PpSettings { min_gain: 0.01, min_chance: 0.12, ht: true, ht_chance: 0.5, ht_gain: 5., dt: true, push: 0., cooldown_days: EXPOSURE_S / 86400., top100: false }
    }
}

pub fn pp_scale(stars: f64, hits: f64) -> f64 {
    8. * (stars - 0.15).max(0.05).powf(2.2) * (1. + 0.1 * (hits / 1500.).min(1.))
}

fn bonus(n: usize) -> f64 {
    416.6667 * (1. - 0.995f64.powi(n.min(1000) as i32))
}

/// P(acc320 > acc_needed) for y = log(1 − acc) ~ N(mu, sd).
fn p_beat(mu: f64, sd: f64, needed: f64) -> f64 {
    if needed >= 1. {
        return 0.;
    }
    let z = ((1. - needed).max(1e-9).ln() - mu) / sd;
    0.5 * (1. + libm::erf(z / std::f64::consts::SQRT_2))
}

/// Best counted pp per beatmap → total, and the exact change of replacing or adding one map's best.
#[derive(Clone, Default)]
pub struct Ledger {
    pub best: HashMap<i64, (f64, bool, Option<String>)>, // bid → (pp, lazer, when)
    pub p: Vec<f64>,
    w: Vec<f64>,
    cum: Vec<f64>,
}

impl Ledger {
    pub fn new(best: HashMap<i64, (f64, bool, Option<String>)>) -> Ledger {
        let mut p: Vec<f64> = best.values().map(|v| v.0).collect();
        p.sort_by(|a, b| b.total_cmp(a));
        let w: Vec<f64> = (0..=p.len()).map(|i| 0.95f64.powi(i as i32)).collect();
        let mut cum = vec![0.];
        for (x, y) in p.iter().zip(&w) {
            cum.push(cum[cum.len() - 1] + x * y);
        }
        Ledger { best, p, w, cum }
    }

    pub fn total(&self) -> f64 {
        self.cum[self.cum.len() - 1] + bonus(self.p.len())
    }

    pub fn best_of(&self, bid: i64) -> f64 {
        self.best.get(&bid).map_or(0., |v| v.0)
    }

    /// Account change of one outcome. Replacing a with b > a shifts the scores between them down one place (×0.95); a
    /// new map also shifts everything below it and adds its bonus step.
    pub fn delta(&self, a: f64, new: f64) -> f64 {
        if new <= a {
            return 0.;
        }
        let n = self.p.len();
        let j = self.p.partition_point(|&x| x > new); // scores strictly above the new result
        if a > 0. {
            let i = self.p.partition_point(|&x| x > a).min(n); // first position holding a
            let jj = j.min(i);
            new * self.w[jj] - a * self.w[i] - 0.05 * (self.cum[i] - self.cum[jj])
        } else {
            new * self.w[j] - 0.05 * (self.cum[n] - self.cum[j]) + bonus(n + 1) - bonus(n)
        }
    }

    /// bid → place, best first.
    pub fn ranks(&self) -> HashMap<i64, i64> {
        let mut order: Vec<(&i64, f64)> = self.best.iter().map(|(b, v)| (b, v.0)).collect();
        order.sort_by(|a, b| b.1.total_cmp(&a.1));
        order.into_iter().enumerate().map(|(i, x)| (*x.0, i as i64 + 1)).collect()
    }
}

pub fn status_of(status: Option<i64>, md5: Option<&str>, bid: Option<i64>, public: &Public, installed: &HashMap<String, Arc<Inst>>) -> Option<i64> {
    status.or_else(|| md5.and_then(|m| installed.get(m)).and_then(|i| i.status)).or_else(|| bid.and_then(|b| public.map_info(b)).map(|m| m.approved))
}

/// pp of a score via rosu-pp; a panic on one strange chart loses that score, not the app.
pub fn score_pp(path: &std::path::Path, stats: &[i64; 6], mods: &serde_json::Value, lazer: bool) -> Option<f64> {
    let data = std::fs::read(path).ok()?;
    let mods = mods.clone();
    let [geki, n300, katu, n100, n50, misses] = stats.map(|x| x.max(0) as u32);
    std::panic::catch_unwind(move || {
        let map = rosu_pp::Beatmap::from_bytes(&data).ok()?;
        let mods = rosu_mods(&mods);
        let attrs = rosu_pp::Performance::new(&map).mods(mods).lazer(lazer).n_geki(geki).n300(n300).n_katu(katu).n100(n100).n50(n50).misses(misses).calculate();
        Some(attrs.pp())
    }).ok().flatten().filter(|p: &f64| p.is_finite() && *p >= 0.)
}

/// Lazer mod list or stable bitmask → rosu's mods.
fn rosu_mods(mods: &serde_json::Value) -> u32 {
    if let Some(bits) = mods.as_u64() {
        return bits as u32;
    }
    const BITS: [(&str, u32); 13] = [("NF", 1), ("EZ", 2), ("HD", 8), ("HR", 16), ("SD", 32), ("DT", 64), ("HT", 256), ("NC", 576), ("FL", 1024), ("SO", 4096), ("PF", 16416), ("FI", 1048576), ("MR", 1073741824)];
    mods.as_array().map_or(0, |list| list.iter().filter_map(|m| m.get("acronym").or(Some(m)).and_then(|a| a.as_str()))
        .filter_map(|a| BITS.iter().find(|b| b.0.eq_ignore_ascii_case(a)).map(|b| b.1)).fold(0, |x, b| x | b))
}

/// A local copy of the chart: lazer's store, then the stable-linked one, by sha256.
pub fn chart_path(sha: Option<&str>) -> Option<PathBuf> {
    let sha = sha?;
    [paths::lazer_data(), paths::stable_store()].iter().find_map(|root| paths::local_file(sha, root).filter(|p| p.is_file()))
}

/// chart_path, else the frozen dump by id, at this revision only.
pub fn scored_chart(sha: Option<&str>, bid: Option<i64>, md5: Option<&str>) -> Option<PathBuf> {
    chart_path(sha).or_else(|| paths::frozen_chart(bid?, md5))
}

/// Counted bests: the snapshot's stable pp, plus lazer scores that were submitted (online id) or seen live, on ranked or
/// approved maps with ranked mods at default settings. Missing pp comes from rosu-pp and is saved.
pub fn build_ledger(db: &Db, public: &Public, installed: &HashMap<String, Arc<Inst>>) -> Result<(Ledger, usize)> {
    struct R { key: String, src: String, client: Option<String>, sha: Option<String>, md5: Option<String>, bid: i64, status: Option<i64>, online: Option<i64>, pp: Option<f64>,
               played: Option<String>, stats: Option<String>, mods: Option<String> }
    let mut q = db.prepare("SELECT key,src,client,sha256,md5,beatmap_id,status,online_id,pp,played,stats,mods FROM scores WHERE ranked=1 AND beatmap_id>0")?;
    let rows: Vec<R> = q.query_map([], |r| Ok(R { key: r.get(0)?, src: r.get(1)?, client: r.get(2)?, sha: r.get(3)?, md5: r.get(4)?, bid: r.get(5)?, status: r.get(6)?, online: r.get(7)?,
                                                  pp: r.get(8)?, played: r.get(9)?, stats: r.get(10)?, mods: r.get(11)? }))?.collect::<Result<_>>()?;
    let live = |c: &Option<String>| matches!(c.as_deref(), Some("lazer-live" | "stable-live"));
    let submitted = |r: &R| !matches!(r.src.as_str(), "realm" | "stable") || r.online.is_some_and(|o| o != 0) || live(&r.client);
    let (mut best, mut missing): (HashMap<i64, (f64, bool, Option<String>)>, usize) = (HashMap::default(), 0);
    for r in &rows {
        if !status_of(r.status, r.md5.as_deref(), Some(r.bid), public, installed).is_some_and(|s| RANKED_STATUS.contains(&s)) || !submitted(r) {
            continue;
        }
        let mut pp = r.pp;
        if pp.is_none() && r.src != "public" {
            let stats = r.stats.as_deref().and_then(parse_stats);
            let path = scored_chart(r.sha.as_deref(), Some(r.bid), r.md5.as_deref());
            let mods: serde_json::Value = r.mods.as_deref().and_then(|m| serde_json::from_str(m).ok()).unwrap_or(serde_json::json!([]));
            let lazer = !r.client.as_deref().unwrap_or("").starts_with("stable"); // stable judges a hold note once
            pp = match (path, stats) {
                (Some(p), Some(s)) => score_pp(&p, &s, &mods, lazer),
                _ => None,
            };
            match pp {
                None => {
                    missing += 1;
                    continue;
                }
                Some(v) => {
                    db.execute("UPDATE scores SET pp=? WHERE key=?", rusqlite::params![v, r.key])?;
                }
            }
        }
        let Some(pp) = pp else { continue };
        if pp > best.get(&r.bid).map_or(0., |v| v.0) {
            best.insert(r.bid, (pp, r.src != "public", r.played.clone()));
        }
    }
    Ok((Ledger::new(best), missing))
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Reference {
    pub safe_pp: f64,
    pub peak_pp: f64,
    pub comfort: f64,
    pub peak_acc: f64,
    pub maps: usize,
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2. }
}

fn percentile(mut v: Vec<f64>, q: f64) -> f64 {
    v.sort_by(f64::total_cmp);
    let at = q / 100. * (v.len() - 1) as f64;
    let (lo, hi) = (at.floor() as usize, at.ceil() as usize);
    v[lo] + (v[hi] - v[lo]) * (at - lo as f64)
}

/// Empirical accuracy and PP endpoints, one counted best per map (not retry count).
pub fn reference(db: &Db, ledger: &Ledger) -> Result<(HashMap<i64, i64>, HashMap<i64, Reference>)> {
    let ranks = ledger.ranks();
    let mut q = db.prepare("SELECT beatmap_id,keys FROM installed WHERE beatmap_id>0")?;
    let by_bid: HashMap<i64, Option<i64>> = q.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?.collect::<Result<_>>()?;
    let mut q = db.prepare("SELECT s.beatmap_id,s.pp,s.stats,i.keys FROM scores s LEFT JOIN installed i ON i.sha256=s.sha256 WHERE s.pp>0 AND s.ranked=1 ORDER BY s.pp DESC")?;
    let rows: Vec<(i64, f64, Option<String>, Option<i64>)> = q.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))?.collect::<Result<_>>()?;
    let mut by: HashMap<i64, Vec<(f64, f64)>> = HashMap::default();
    let mut seen = HashSet::new();
    for (bid, pp, stats, keys) in rows {
        if seen.contains(&bid) || !ranks.contains_key(&bid) || (pp - ledger.best_of(bid)).abs() > 0.01 {
            continue;
        }
        seen.insert(bid);
        let Some(acc) = stats.as_deref().and_then(parse_stats).map(|s| acc_lazer(&s)) else { continue };
        let keys = keys.or_else(|| by_bid.get(&bid).copied().flatten());
        if let Some(k) = keys.filter(|k| SUPPORTED_KEYS.contains(k)) {
            if 0. < acc && acc <= 1. {
                by.entry(k).or_default().push((pp, acc));
            }
        }
    }
    let result = by.into_iter().map(|(k, mut values)| {
        values.sort_by(|a, b| b.0.total_cmp(&a.0).then(b.1.total_cmp(&a.1)));
        values.truncate(100);
        let half = values.len() / 2;
        let safe = if half < values.len() { &values[half..] } else { &values[..] };
        let peak = &values[..(values.len() / 5).max(1)];
        (k, Reference {
            safe_pp: median(safe.iter().map(|x| x.0).collect()),
            peak_pp: median(peak.iter().map(|x| x.0).collect()),
            comfort: median(safe.iter().map(|x| x.1).collect()).clamp(0.965, 0.99),
            // The peak tier contains both precise LN plays and lower-accuracy speed plays. Its lower quartile describes a
            // demonstrated push, not an invented universal 94% floor.
            peak_acc: percentile(peak.iter().map(|x| x.1).collect(), 25.).clamp(0.88, 0.98),
            maps: values.len(),
        })
    }).collect();
    Ok((ranks, result))
}

/// The ranked or loved variants with features.
pub struct Variant {
    pub bid: i64,
    pub rate: f64,
    pub var: &'static str,
    pub f: Arc<Feats>,
    pub md5: String,
    pub file: String,
    pub set: i64,
    pub approved: i64,
    pub stars: f64,
    pub length: f64,
    pub scale: f64,
    pub skill: Option<String>,
    pub identity: String,
}

fn top_skill(f: &Feats, c: &Consts) -> Option<String> {
    let mut best: Option<(usize, f64)> = None;
    for (i, &v) in f.sk.iter().enumerate() {
        if v != 0. && c.names[i] != "stamina" && best.is_none_or(|b| v > b.1) {
            best = Some((i, v));
        }
    }
    best.map(|b| c.names[b.0].clone())
}

/// Every ranked or loved difficulty × NM/HT/DT with features.
pub fn candidates(public: &Public, installed: &HashMap<String, Arc<Inst>>, keys: &[i64], c: &Consts, newer: Vec<Variant>) -> Vec<Variant> {
    let mut out = Vec::new();
    for chart in &public.meta.charts {
        let bid = chart.0;
        let Some(meta) = public.map_info(bid) else { continue };
        // The archive may still call a newly ranked map Qualified: an installed exact revision carries the newer status.
        let approved = installed.get(&meta.md5).and_then(|i| i.status).unwrap_or(meta.approved);
        if !RANKED_STATUS.contains(&approved) && approved != 4 {
            continue;
        }
        for (rate, var) in VARIANTS {
            let Some(Val::Dict(d)) = public.feature(bid, rate) else { continue };
            let Some(stars) = d.get("stars").filter(|v| !v.is_none()).map(Val::f) else { continue };
            if !keys.contains(&(d.f("keys") as i64)) || d.f("notes") < 100. {
                continue;
            }
            let Ok(f) = Feats::of(&d, c) else { continue };
            out.push(Variant { bid, rate, var, md5: meta.md5.clone(), file: meta.file.clone(), set: meta.set, approved, stars, length: f.length / 1000. / rate,
                               scale: pp_scale(stars, d.f("hits")), skill: top_skill(&f, c), identity: format!("md5:{}", meta.md5), f: Arc::new(f) });
        }
    }
    out.extend(newer);
    out
}

/// Installed ranked or approved maps without snapshot features: ranked since, or their chart was not in the dump's
/// archive (the client keeps the status current). Chart features only.
/// Installed ranked/approved maps without snapshot features: ranked since, or their chart was not in the dump's archive
/// (lazer keeps the status current).
pub fn newer_installed<'a>(public: &Public, installed: &'a HashMap<String, Arc<Inst>>) -> Vec<&'a Arc<Inst>> {
    let analysed = |bid: i64| public.meta.charts.binary_search_by(|x| x.0.cmp(&bid)).is_ok();
    installed.iter().filter(|(md5, i)| {
        i.status.is_some_and(|s| RANKED_STATUS.contains(&s)) && i.beatmap_id.is_some_and(|b| b > 0) && (4..=10).contains(&i.keys)
            && !public.md5_bid.get(*md5).is_some_and(|&b| analysed(b))
    }).map(|x| x.1).collect()
}

pub fn newer_ranked(db: &Db, source: &mut crate::evidence::FeatureSource, installed: &HashMap<String, Arc<Inst>>, keys: &[i64], c: &Consts) -> Result<Vec<Variant>> {
    let mut out = Vec::new();
    let public = source.public.clone();
    for inst in newer_installed(&public, installed) {
        for (rate, var) in VARIANTS {
            let key = source.key(&inst.sha256, rate);
            let Some(d) = source.lookup(db, &key)? else { continue };
            let Some(stars) = d.get("stars").filter(|v| !v.is_none()).map(Val::f) else { continue };
            if !keys.contains(&(d.f("keys") as i64)) || d.f("notes") < 100. {
                continue;
            }
            let Ok(f) = Feats::of(&d, c) else { continue };
            let md5 = inst.md5.clone().unwrap_or_default();
            out.push(Variant { bid: inst.beatmap_id.unwrap_or(0), rate, var, identity: format!("md5:{md5}"), md5, file: inst.name(), set: inst.set_id.unwrap_or(0), approved: inst.status.unwrap_or(0),
                               stars, length: f.length / 1000. / rate, scale: pp_scale(stars, d.f("hits")), skill: top_skill(&f, c), f: Arc::new(f) });
        }
    }
    Ok(out)
}

/// One scored request's inputs that do not change between charts.
pub struct Scoring<'a> {
    pub pp_model: &'a PredModel,
    pub pp_tables: &'a PredSession,
    pub display: Option<(&'a PredModel, &'a PredSession)>,
    pub shown: (f64, f64),
    pub session: &'a Session,
    pub ledger: &'a Ledger,
    pub ranks: &'a HashMap<i64, i64>,
    pub references: &'a HashMap<i64, Reference>,
    pub settings: &'a PpSettings,
    pub typical: &'a HashMap<i64, f64>,
    pub installed: &'a HashMap<String, Arc<Inst>>,
    pub public: &'a Public,
    pub now: f64,
}

/// {keys: recent typical y}: the median of the last ~60 days of attempts, else all.
pub fn typical(rows: &Rows, month: f64) -> HashMap<i64, f64> {
    let mut recent: HashMap<i64, Vec<f64>> = HashMap::default();
    let mut all: HashMap<i64, Vec<f64>> = HashMap::default();
    for r in &rows.rows {
        all.entry(r.keys).or_default().push(r.y);
        if r.src != "public" && r.t >= month - 2. {
            recent.entry(r.keys).or_default().push(r.y);
        }
    }
    all.into_iter().map(|(k, v)| (k, median(recent.remove(&k).unwrap_or(v)))).collect()
}

fn push_share(steps: f64) -> f64 {
    1. - (-PUSH_STEP * steps).exp()
}

/// push: the PP tab's Safe ↔ Push setting, added after the automatic readiness.
pub fn readiness(session: &Session, keys: i64, skill: Option<&str>, push: f64) -> f64 {
    let family = skill.map(|s| session.consts.family(s));
    let active = (session.activation(keys, family) / 6.).min(1.);
    // Warming up unlocks ordinary PP work; each beaten best moves further toward the player's high-PP, lower-accuracy
    // edge. A bad result pulls back at once.
    let auto = active * (0.45 + 0.5 * push_share(session.push(Some(keys)))) * (-3. * session.correction(keys, skill).max(0.)).exp();
    if push != 0. { (auto + push).clamp(0., 1.) } else { auto }
}

fn comfort(refs: &HashMap<i64, Reference>, keys: i64) -> f64 {
    refs.get(&keys).map_or(0.975, |r| r.comfort)
}

fn shown_of(y: f64) -> f64 {
    (1. - y.min(20.).exp()).clamp(0., 1.)
}

/// The user dislikes HT: offer it only as a likely improvement (2026-09-30).
fn rate_allowed(rate: f64, p_up: f64, gain: f64, opt: &PpSettings) -> bool {
    (rate >= 1. || (opt.ht && p_up >= opt.ht_chance && gain >= opt.ht_gain)) && (opt.dt || rate <= 1.)
}

/// Recommender.score: one entry per difficulty, its best variant, every credible difficulty eligible.
pub fn score(variants: &[Variant], s: &Scoring<'_>, threads: usize) -> Vec<Pick> {
    let opt = s.settings;
    // Readiness per (keys, family name): cheap per pair, a few dozen pairs.
    let mut states: HashMap<(i64, Option<String>), (f64, f64)> = HashMap::default();
    for v in variants {
        let key = (v.f.keys, v.skill.clone());
        states.entry(key).or_insert_with(|| (s.session.push(Some(v.f.keys)), readiness(s.session, v.f.keys, v.skill.as_deref(), opt.push)));
    }
    let cold: HashMap<i64, bool> = SUPPORTED_KEYS.map(|k| (k, s.session.activation(k, None) < 1.5)).collect();
    struct Row {
        p: Prediction,
        shown: f64,
        lo: f64,
        hi: f64,
        pp: [f64; 21],
        gain: f64,
        p_up: f64,
        best: f64,
        chal: f64,
        normal: bool,
        credible: bool,
        value: f64,
        central: f64,
        display: Option<Prediction>,
    }
    let scored = background_map(variants, threads, |v| -> std::result::Result<Row, String> {
        let f = &*v.f;
        let (k, ranked) = (f.keys, RANKED_STATUS.contains(&v.approved));
        let p = s.pp_model.predict(f, &v.md5, Some(v.bid), v.rate, s.pp_tables)?;
        let (mu, sdm, sda) = (p.mu, p.sdm, p.sda);
        let mut pp = [0.; 21];
        for (out, z) in pp.iter_mut().zip(Z_Q) {
            let y = mu + PESSIMISM * sdm + sda * z;
            *out = if ranked { v.scale * (5. * (1. - y.exp()) - 4.).max(0.) } else { 0. };
        }
        let best = s.ledger.best_of(v.bid);
        let gain = pp.iter().map(|&x| s.ledger.delta(best, x)).sum::<f64>() / 21.;
        let need = (best / v.scale.max(1e-9) + 4.) / 5.; // acc320 that would beat the counted best
        let p_up = if ranked { p_beat(mu + PESSIMISM * sdm, sda, need) } else { 0. };
        let chal: f64 = s.typical.get(&k).map_or(0., |t| { let c: f64 = (mu - t) / sda; if c.is_nan() { 0. } else { c } });
        let normal = chal <= 1.25;
        let (ready, state) = states[&(k, v.skill.clone())];
        // Challenge is in attempt sds; each staircase step raises the ceiling by one step.
        let mut credible = chal <= 1.25 + PUSH_STEP * ready.min(4.) && p_up >= if normal { opt.min_chance } else { opt.min_chance.max(0.25) };
        credible &= rate_allowed(v.rate, p_up, gain, opt);
        let display = s.display.and_then(|(m, t)| m.predict(f, &v.md5, Some(v.bid), v.rate, t).ok());
        let (dmu, dsd) = display.as_ref().map_or((s.shown.0 + s.shown.1 * mu, s.shown.1 * sdm.hypot(sda)), |d| (d.mu, d.sdm.hypot(d.sda)));
        let shown = shown_of(dmu);
        // The first play in an unactivated keymode is a comfortable probe: a long 7K session never authorises cold 10K DT.
        credible &= !cold.get(&k).copied().unwrap_or(true) || shown >= comfort(s.references, k);
        // Choose the variant for this readiness BEFORE reducing a chart to one entry: otherwise a high-gain DT variant
        // hides its safer NM/HT option.
        let reference = s.references.get(&k);
        let acc_target: f64 = (1. - state) * reference.map_or(0.975, |r| r.comfort) + state * reference.map_or(0.93, |r| r.peak_acc);
        let central = v.scale * (1. - 5. * mu.min(20.).exp()).max(0.);
        let pp_target = reference.map_or(central, |r| { let t = (1. - state) * r.safe_pp + state * r.peak_pp; if t != 0. { t } else { central } });
        let accuracy_fit = (-0.5 * ((acc_target - shown).max(0.) / 0.025).powi(2)).exp();
        let tier_fit = (-0.5 * (central.max(1.).ln() - pp_target.max(1.).ln()).powi(2) / (0.35f64 * 0.35)).exp();
        let utility = 0.3 + (gain.max(0.) / 0.05).ln_1p().powf(0.65);
        Ok(Row { shown, lo: (1. - (dmu + BAND_Z * dsd).min(20.).exp()).max(0.), hi: (1. - (dmu - BAND_Z * dsd).min(20.).exp()).max(0.), pp, gain, p_up, best, chal, normal, credible,
                   value: utility * (0.2 + 0.8 * tier_fit) * accuracy_fit * p_up, central, p, display })
    });
    let rows: Vec<Option<Row>> = scored.into_iter().map(std::result::Result::ok).collect();
    let mut order: Vec<usize> = (0..variants.len()).filter(|&i| rows[i].as_ref().is_some_and(|r| r.credible)).collect();
    // A comfortable zero-gain variant must not hide another variant which passes readiness and offers a real gain.
    let at = |i: usize| rows[i].as_ref().unwrap();
    order.sort_by(|&a, &b| (at(b).gain >= opt.min_gain).cmp(&(at(a).gain >= opt.min_gain)).then(at(b).value.total_cmp(&at(a).value))
        .then((variants[a].rate != 1.).cmp(&(variants[b].rate != 1.))));
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for i in order {
        let (v, r) = (&variants[i], at(i));
        if !seen.insert(v.bid) {
            continue;
        }
        let inst = s.installed.get(&v.md5);
        out.push(Pick {
            mode: "pp".into(), bid: Some(v.bid), rate: v.rate, var: v.var.to_string(), keys: v.f.keys, title: v.file.clone(), sha: inst.map(|i| i.sha256.clone()),
            online_match: inst.is_none_or(|i| i.online_md5.as_deref() == Some(&v.md5) || s.public.map_info(v.bid).is_some_and(|m| m.md5 == v.md5)),
            artist: inst.map(|i| i.artist.clone()), song: inst.map(|i| i.title.clone()), version: inst.map(|i| i.version.clone()), creator: inst.and_then(|i| i.creator.clone()),
            md5: Some(v.md5.clone()), ranked: RANKED_STATUS.contains(&v.approved), gain: Some(r.gain), p_up: Some(r.p_up), best: (r.best != 0.).then_some(r.best),
            pp_lo: Some(r.pp[16]), pp_hi: Some(r.pp[4]), // outcomes run best → worst
            best_rank: s.ranks.get(&v.bid).copied().unwrap_or(100000), pp_mid: Some(r.central),
            pred: Prediction { acc_mid: r.shown, acc_lo: r.lo, acc_hi: r.hi, ..r.display.clone().unwrap_or_else(|| r.p.clone()) }, pp_pred: Some(r.p.clone()),
            push_only: !r.normal, challenge: r.chal, skill: v.skill.clone(), length: v.length, installed: inst.is_some(), stars: Some(v.stars), scale: Some(v.scale), f: Some(v.f.clone()),
            generated_at: s.now, ..Pick::default()
        });
    }
    out
}

/// pp_playlist.prepare → (rows with weights, all cycled).
pub fn prepare(candidates: &[Pick], session: &Session, ledger: &Ledger, references: &HashMap<i64, Reference>, opt: &PpSettings, home_keys: i64) -> (Vec<Pick>, bool) {
    let k0 = session.current_keys().unwrap_or(home_keys);
    let floor = ledger.p.get(99).copied().unwrap_or(0.);
    let mut rows = Vec::new();
    for source in candidates {
        let (gain, p_up) = (source.gain.unwrap_or(0.), source.p_up.unwrap_or(0.));
        let sd_model = source.pp_pred.as_ref().map_or(source.pred.sdm, |p| p.sdm);
        if !source.ranked || gain < opt.min_gain || sd_model >= 1. || p_up < opt.min_chance {
            continue;
        }
        let (keys, acc) = (source.keys, source.pred.acc_mid);
        let cold = session.activation(keys, None) < 1.5;
        if cold && acc < comfort(references, keys) {
            continue;
        }
        let contender = source.best_rank <= 100 || source.pp_hi.or(source.pp_mid).unwrap_or(f64::INFINITY) >= floor;
        if opt.top100 && !contender {
            continue;
        }
        let state = readiness(session, keys, source.skill.as_deref(), opt.push);
        let reference = references.get(&keys);
        let target_acc = (1. - state) * reference.map_or(0.975, |r| r.comfort) + state * reference.map_or(0.93, |r| r.peak_acc);
        // Not a hard accuracy floor for ordinary PP play: difficulty can increase as readiness improves, including the
        // user's low-acc top-play style.
        let accuracy_fit = (-0.5 * ((target_acc - acc).max(0.) / 0.025).powi(2)).exp();
        let pp = source.pp_mid.or(source.best).unwrap_or(1.);
        let target_pp = reference.map_or(pp, |r| (1. - state) * r.safe_pp + state * r.peak_pp);
        let tier_fit = (-0.5 * ((pp.max(1.) / target_pp.max(1.)).ln() / 0.35).powi(2)).exp();
        // Compress, do not erase, utility: a 46pp outlier must not remove a hundred credible 0.1–3pp improvements before
        // freshness is considered.
        let utility = 0.3 + (gain / 0.05).ln_1p().powf(0.65);
        let headroom = (pp - source.best.unwrap_or(pp)).max(0.) / (pp * 0.1).max(20.);
        let continuity = if keys == k0 { 1. } else { 0.45 };
        let mut c = source.clone();
        c.purpose = (if cold { "probe" } else if source.push_only || state > 0.8 { "push" } else { "farm" }).to_string();
        // Likely improvements come far more often than long shots (user 2026-09-30).
        c.weight = utility * (0.2 + 0.8 * tier_fit) * accuracy_fit * continuity * (1. + 0.3 * headroom.min(2.)) * p_up;
        let refarm = source.best.map_or_else(|| format!("new map · {:.0}% to count", 100. * p_up), |b| format!("refarm · best {b:.0}pp · {:.0}% to beat", 100. * p_up));
        c.why = format!("{refarm} · expected {:.1}%{}{}", 100. * acc,
            if cold { " · comfortable keymode probe" } else if state > 0.8 { " · peak opportunity" } else { " · current-form fit" },
            if source.best_rank <= 100 { format!(" · current top #{}", source.best_rank) } else if contender { " · new/top-100 contender".to_string() } else { " · smaller PP improvement".to_string() });
        rows.push(c);
    }
    let total: f64 = rows.iter().map(|c| c.weight).sum();
    let pool = (total > 0.).then(|| total * total / rows.iter().map(|c| c.weight * c.weight).sum::<f64>().max(1e-300));
    for c in &mut rows {
        c.fresh = session.freshness(&c.identity().unwrap_or_default(), c.length, Some("pp"), pool, opt.cooldown_days * 86400.);
        c.weight *= c.fresh;
    }
    // Offer or skip never promises a repeat. Only relax when *all* credible choices have been cycled.
    let fresh: Vec<Pick> = rows.iter().filter(|c| c.fresh >= 0.08).cloned().collect();
    let relaxed = !rows.is_empty() && fresh.is_empty();
    if relaxed {
        for c in &mut rows {
            c.weight /= c.fresh.max(1e-6).sqrt();
        }
    } else {
        rows = fresh;
    }
    // No hard bag: freshness's offer count is the soft one.
    let last = session.all.iter().rev().find(|e| e.kind == "offer").map(|e| e.beatmap.clone());
    if rows.len() > 1 {
        rows.retain(|c| last.as_ref() != Some(&c.identity()));
    }
    (rows, relaxed)
}

/// (lo, hi, best, gain, chance) of a chart at a predicted result: the card's PP line.
pub fn estimates(p: &Prediction, stars: f64, hits: f64, bid: Option<i64>, ledger: &Ledger) -> (f64, f64, Option<f64>, f64, f64) {
    estimates_at(p, pp_scale(stars, hits), bid, ledger)
}

pub fn estimates_at(p: &Prediction, scale: f64, bid: Option<i64>, ledger: &Ledger) -> (f64, f64, Option<f64>, f64, f64) {
    let pp: Vec<f64> = Z_Q.iter().map(|z| scale * (5. * (1. - (p.mu + PESSIMISM * p.sdm + p.sda * z).exp()) - 4.).max(0.)).collect();
    let best = bid.map_or(0., |b| ledger.best_of(b));
    (pp[16], pp[4], (best != 0.).then_some(best), pp.iter().map(|&x| ledger.delta(best, x)).sum::<f64>() / 21., p_beat(p.mu + PESSIMISM * p.sdm, p.sda, (best / scale.max(1e-9) + 4.) / 5.))
}

/// The keymode contributing most account pp, else the most played lately.
pub fn home_keys(ledger: &Ledger, public: &Public, rows: &Rows, month: f64) -> i64 {
    let mut contribution: HashMap<i64, f64> = HashMap::default();
    let mut order: Vec<(&i64, f64)> = ledger.best.iter().map(|(b, v)| (b, v.0)).collect();
    order.sort_by(|a, b| b.1.total_cmp(&a.1));
    for (i, (bid, pp)) in order.iter().enumerate() {
        if let Some(k) = public.map_info(**bid).map(|m| m.keys).filter(|k| SUPPORTED_KEYS.contains(k)) {
            *contribution.entry(k).or_insert(0.) += pp * 0.95f64.powi(i as i32);
        }
    }
    if let Some((k, _)) = contribution.iter().max_by(|a, b| a.1.total_cmp(b.1)) {
        return *k;
    }
    let mut recent: HashMap<i64, usize> = HashMap::default();
    rows.rows.iter().filter(|r| r.src != "public" && r.t >= month - 1.).for_each(|r| *recent.entry(r.keys).or_insert(0) += 1);
    recent.into_iter().max_by_key(|x| x.1).map_or(7, |x| x.0)
}

/// Comfortable maps the player already knows (installed, any status, their own rate) when warming up.
pub fn warmups(fitted: &Fitted, rows: &[Pick], session: &Session, k0: i64, tables: &PredSession, display: &PredSession, installed_by_chart: &dyn Fn(&str) -> Option<Arc<Inst>>, typical: &HashMap<i64, f64>) -> Vec<Pick> {
    let _ = rows;
    let mut known: Vec<((String, u64), Vec<&Evidence>)> = Vec::new();
    let mut at: std::collections::HashMap<(String, u64), usize> = std::collections::HashMap::new();
    for r in fitted.rows.rows.iter().filter(|r| r.src != "public" && r.keys == k0) {
        let key = (r.chart.clone(), round_digits(r.rate, 3).to_bits());
        let i = *at.entry(key.clone()).or_insert_with(|| { known.push((key, Vec::new())); known.len() - 1 });
        known[i].1.push(r);
    }
    let mut out = Vec::new();
    for ((chart, rate), rs) in &known {
        let (rate, r) = (f64::from_bits(*rate), *rs.last().unwrap());
        let Some(inst) = installed_by_chart(chart) else { continue };
        let f = &fitted.rows.feats[r.f];
        let Ok(pp) = fitted.pp.model.predict(f, chart, r.b, rate, tables) else { continue };
        let target = fitted.display.as_ref().map_or(&fitted.pp, |d| d);
        let Ok(p) = target.model.predict(f, chart, r.b, rate, display) else { continue };
        if p.opening_acc < 0.937 {
            continue;
        }
        let ch: f64 = typical.get(&r.keys).map_or(0., |t| (pp.mu - t) / pp.sda);
        if !(-2.2..=-0.3).contains(&ch) {
            continue;
        }
        let length = f.length / 1000. / rate;
        let identity = inst.identity();
        let fresh = session.freshness(&identity, length, Some("pp"), None, EXPOSURE_S);
        let n = rs.len();
        out.push(Pick {
            mode: "pp".into(), bid: inst.beatmap_id.filter(|&b| b > 0), sha: Some(inst.sha256.clone()), rate, var: rate_label_pp(rate), purpose: "warmup".into(), keys: r.keys,
            online_match: inst.online_md5.is_some() && inst.online_md5 == inst.md5, artist: Some(inst.artist.clone()), song: Some(inst.title.clone()), version: Some(inst.version.clone()),
            creator: inst.creator.clone(), title: inst.name(), ranked: inst.status.is_some_and(|s| RANKED_STATUS.contains(&s)), gain: Some(0.), installed: true, challenge: ch, length,
            weight: (-(ch + 1f64).powi(2)).exp() * fresh * (1. + n.min(5) as f64 / 5.) * (-(length - 240.).max(0.) / 120.).exp(), // short ones warm up
            why: format!("warmup · played {n}×"), md5: Some(chart.clone()), pred: p, pp_pred: Some(pp), f: Some(Arc::new(f.clone())), ..Pick::default()
        });
    }
    out.sort_by(|a, b| b.weight.total_cmp(&a.weight));
    out
}

pub fn exists(db: &Db, sql: &str, key: &str) -> Result<bool> {
    Ok(db.query_row(sql, [key], |_| Ok(())).optional()?.is_some())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ms_data::store::Event;
    use ms_model::personal::WarmModels;
    use ms_model::predict::{Consts, Prediction, Slopes};
    use ms_model::warmup::{FormPrior, Model};
    use serde_json::{json, Value};

    fn cand(b: i64, gain: f64) -> Pick {
        Pick { bid: Some(b), gain: Some(gain), ranked: true, keys: 7, length: 120., best: Some(100.), p_up: Some(0.5), var: "NM".into(), rate: 1., title: format!("m{b}"), md5: Some(format!("{b:032x}")),
               installed: true, pp_mid: Some(105.), best_rank: 100000, pred: Prediction { sdm: 0.2, acc_mid: 0.985, ..Prediction::default() }, ..Pick::default() }
    }

    fn event(id: i64, t: f64, kind: &str, beatmap: Option<String>, info: Value) -> Arc<Event> {
        Arc::new(Event { id, t, kind: kind.to_string(), beatmap, info })
    }

    fn session(events: Vec<Arc<Event>>, now: f64) -> Session {
        let c = Arc::new(Consts::standard());
        let prior = FormPrior { global_variance: 0.05, skill_variance: 0.12, by_key: Default::default(), noise_variance: 0.04 };
        let model = Model::new(&WarmModels::default(), Slopes::default(), prior, c.families.len());
        Session::new(events, now, Some(Arc::new(model)), c.clone(), Arc::new(crate::recommender::family_transfer(&c)))
    }

    /// A session of finished plays with these z scores.
    fn played(zs: &[f64], keys: i64) -> Session {
        let mut out = Vec::new();
        for (i, z) in zs.iter().enumerate() {
            let t = 1000. + i as f64 * 220.;
            let b = Some(format!("{}", 200 + i));
            out.push(event(2 * i as i64 + 1, t, "start", b.clone(), json!({"mu": -3., "sd": 0.4, "keys": keys, "skill": "delay", "ranked": true, "purpose": "farm", "length": 120})));
            out.push(event(2 * i as i64 + 2, t + 120., "finish", b, json!({"y": -3. - 0.4 * z})));
        }
        session(out, 1000. + (zs.len() - 1) as f64 * 220. + 190.)
    }

    fn warm() -> Session {
        session(vec![event(1, 999., "skip_warmup", None, json!({}))], 1000.)
    }

    fn offer(id: i64, t: f64, c: &Pick) -> Arc<Event> {
        event(id, t, "offer", c.identity(), json!({"mode": "pp"}))
    }

    fn prep(rows: &[Pick], s: &Session, ledger: &Ledger, refs: &HashMap<i64, Reference>, o: &PpSettings) -> (Vec<Pick>, bool) {
        prepare(rows, s, ledger, refs, o, 7)
    }

    fn plain(rows: &[Pick], s: &Session) -> (Vec<Pick>, bool) {
        prep(rows, s, &Ledger::default(), &HashMap::default(), &PpSettings::default())
    }

    fn bids(rows: &[Pick]) -> Vec<i64> {
        let mut v: Vec<i64> = rows.iter().filter_map(|c| c.bid).collect();
        v.sort_unstable();
        v
    }

    fn full_ledger() -> Ledger {
        Ledger::new((0..100).map(|i| (i, (500., true, None))).collect())
    }

    #[test]
    fn the_pool_has_no_top_25_or_gain_relative_floor() {
        let mut rows: Vec<Pick> = (1..=150).map(|i| cand(i, 0.13 + (i % 10) as f64 * 0.01)).collect();
        rows.push(Pick { pp_mid: Some(180.), ..cand(1000, 40.) });
        rows.push(cand(2000, 0.005)); // under the .01pp floor
        let (kept, _) = plain(&rows, &session(Vec::new(), 1000.));
        assert_eq!(kept.len(), 151);
        assert!(!bids(&kept).contains(&2000));
    }

    #[test]
    fn smaller_gains_stay_eligible_when_the_good_maps_were_just_played() {
        let rows = [cand(1, 10.), cand(2, 9.), cand(3, 8.), cand(4, 0.3), cand(5, 0.2)];
        let mut events = Vec::new();
        for (i, c) in rows[..3].iter().enumerate() {
            for j in 0..3 {
                let t = 1000. + 600. * i as f64 + 150. * j as f64;
                events.push(event(events.len() as i64 + 1, t, "start", c.identity(), json!({"mu": -3., "sd": 0.4, "keys": 7, "skill": "delay", "ranked": true, "purpose": "farm", "length": 120})));
                events.push(event(events.len() as i64 + 1, t + 120., "finish", c.identity(), json!({"y": -3.2})));
            }
        }
        let (kept, _) = plain(&rows, &session(events, 2900.));
        assert!(bids(&kept).contains(&4) && bids(&kept).contains(&5));
    }

    #[test]
    fn keymode_continuity_is_a_weight_not_an_order() {
        let s = played(&[0.5, 0.6, 0.4, 0.5, 0.7], 7);
        let (seven, four) = (cand(1, 5.), Pick { keys: 4, ..cand(2, 5.5) });
        let (kept, _) = plain(&[seven.clone(), four], &s);
        let weight = |b| kept.iter().find(|c| c.bid == Some(b)).unwrap().weight;
        assert!(weight(1) > weight(2));
        let probe = Pick { keys: 4, ..cand(2, 30.) };
        assert!(bids(&plain(&[seven, probe], &s).0).contains(&2), "a comfortable alternate-mode probe stays eligible");
    }

    #[test]
    fn bad_form_stays_safe_and_strong_form_goes_for_more_pp() {
        let refs: HashMap<i64, Reference> = [(7, Reference { comfort: 0.98, peak_acc: 0.92, safe_pp: 400., peak_pp: 850., maps: 50 })].into_iter().collect();
        let easy = Pick { pp_mid: Some(420.), best: Some(390.), skill: Some("delay".into()), pred: Prediction { sdm: 0.2, acc_mid: 0.981, ..Prediction::default() }, ..cand(100, 0.7) };
        let hard = Pick { pp_mid: Some(900.), best: Some(760.), skill: Some("delay".into()), pred: Prediction { sdm: 0.2, acc_mid: 0.923, ..Prediction::default() }, ..cand(101, 18.) };
        let rows = [easy, hard];
        let run = |s: &Session| prep(&rows, s, &Ledger::default(), &refs, &PpSettings::default()).0;
        assert_eq!(bids(&run(&session(Vec::new(), 1000.))), [100]);
        let ratio = |r: &[Pick]| r.iter().find(|c| c.bid == Some(101)).unwrap().weight / r.iter().find(|c| c.bid == Some(100)).unwrap().weight;
        let (poor, strong) = (run(&played(&[-1., -0.8, -1.2, -0.6], 7)), run(&played(&[0.9, 1., 1.2, 1.1], 7)));
        assert!(ratio(&strong) > 3. * ratio(&poor));
    }

    #[test]
    fn seven_keys_never_activate_ten_keys_but_a_comfortable_probe_is_allowed() {
        let s = played(&[1.; 12], 7);
        assert!(s.activation(7, None) > 0. && s.activation(10, None) == 0.);
        let hard = Pick { keys: 10, skill: Some("delay".into()), pred: Prediction { sdm: 0.2, acc_mid: 0.928, ..Prediction::default() }, ..cand(100, 50.) };
        let probe = Pick { keys: 10, skill: Some("delay".into()), ..cand(101, 0.2) };
        let (kept, _) = plain(&[hard, probe], &s);
        assert_eq!(bids(&kept), [101]);
        assert_eq!(kept[0].purpose, "probe");
    }

    #[test]
    fn a_soft_bag_cycles_every_map_with_no_repeat_within_three_picks() {
        let rows: Vec<Pick> = (0..5).map(|i| cand(i, if i == 0 { 40. } else { 0.2 })).collect();
        let (mut events, mut picks) = (Vec::new(), Vec::new());
        for i in 0..15 {
            let now = 1000. + i as f64 * 240.; // the old exposure discount allowed repeats after a few minutes
            let (kept, _) = plain(&rows, &session(events.clone(), now));
            let total: f64 = kept.iter().map(|c| c.weight).sum();
            let mut at = total * ((i * 7919 + 13) % 101) as f64 / 101.;
            let chosen = kept.iter().find(|c| { at -= c.weight; at < 0. }).unwrap_or(&kept[0]).clone();
            picks.push(chosen.bid.unwrap());
            events.push(offer(i + 1, now, &chosen));
        }
        let mut distinct = picks.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(distinct, [0, 1, 2, 3, 4], "{picks:?}");
        for w in picks.windows(3) {
            assert!(w[0] != w[1] && w[1] != w[2] && w[0] != w[2], "{picks:?}");
        }
    }

    #[test]
    fn the_unseen_map_wins_and_the_last_offer_never_repeats() {
        let rows: Vec<Pick> = (0..4).map(|i| cand(i, 1.)).collect();
        let events: Vec<_> = rows[..3].iter().enumerate().map(|(i, c)| offer(i as i64 + 1, 1000. + i as f64, c)).collect();
        let (kept, relaxed) = plain(&rows, &session(events, 1010.));
        let best = kept.iter().max_by(|a, b| a.weight.total_cmp(&b.weight)).unwrap();
        assert!(best.bid == Some(3) && !bids(&kept).contains(&2) && !relaxed);
    }

    #[test]
    fn a_positive_gain_below_the_top_100_is_not_discarded() {
        let candidate = Pick { pp_mid: Some(300.), pp_hi: Some(350.), best_rank: 101, ..cand(101, 0.2) };
        let (kept, _) = prep(&[candidate], &warm(), &full_ledger(), &HashMap::default(), &PpSettings::default());
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].gain, Some(0.2));
    }

    #[test]
    fn settings_change_the_list() {
        let small = cand(1, 0.05);
        let rows = [small.clone(), Pick { p_up: Some(0.15), ..cand(2, 3.) }, Pick { pp_mid: Some(520.), pp_hi: Some(560.), best_rank: 40, ..cand(3, 1.) }];
        let (ledger, s) = (full_ledger(), warm());
        let run = |o: PpSettings| bids(&prep(&rows, &s, &ledger, &HashMap::default(), &o).0);
        let d = PpSettings::default;
        assert_eq!(run(d()), [1, 2, 3]);
        assert_eq!(run(PpSettings { min_gain: 0.5, ..d() }), [2, 3]);
        assert_eq!(run(PpSettings { min_chance: 0.3, ..d() }), [1, 3]);
        assert_eq!(run(PpSettings { top100: true, ..d() }), [3]);
        // Push raises readiness, Safe lowers it, clamped.
        let hot = played(&[0.9, 1., 1.2, 1.1], 7);
        let (up, auto, down) = (readiness(&hot, 7, Some("delay"), 0.5), readiness(&hot, 7, Some("delay"), 0.), readiness(&hot, 7, Some("delay"), -0.5));
        assert!(up > auto && auto > down && down >= 0.);
        assert_eq!(readiness(&hot, 7, Some("delay"), 5.), 1.);
        // An offer older than the cooldown no longer holds the map back.
        let later = session(vec![offer(1, 1000., &small)], 1000. + 4. * 86400.);
        let id = small.identity().unwrap();
        assert!(later.freshness(&id, 120., Some("pp"), Some(40.), 7. * 86400.) < 1e-3);
        assert_eq!(later.freshness(&id, 120., Some("pp"), Some(40.), 3. * 86400.), 1.);
    }

    #[test]
    fn ht_is_offered_only_as_a_likely_improvement() {
        let d = PpSettings::default;
        assert!(rate_allowed(0.75, 0.6, 6., &d()));
        assert!(!rate_allowed(0.75, 0.3, 6., &d()), "a 12–50 % HT long shot is no longer offered");
        assert!(!rate_allowed(0.75, 0.6, 1., &d()), "nor a likely HT under the gain floor");
        assert!(!rate_allowed(0.75, 0.6, 6., &PpSettings { ht: false, ..d() }));
        assert!(rate_allowed(0.75, 0.2, 1., &PpSettings { ht_chance: 0.15, ht_gain: 0.5, ..d() }));
        assert!(rate_allowed(1.5, 0.2, 1., &d()) && !rate_allowed(1.5, 0.2, 1., &PpSettings { dt: false, ..d() }));
    }

    #[test]
    fn the_estimate_matches_rosu_pp() {
        let mut text = "osu file format v14\n\n[General]\nMode: 3\n\n[Difficulty]\nCircleSize:4\nOverallDifficulty:8\n\n[TimingPoints]\n0,500,4,1,0,100,1,0\n\n[HitObjects]\n".to_string();
        for i in 0..1000 {
            text.push_str(&format!("{},192,{},1,0,0:0:0:0:\n", (i % 4) * 128 + 64, 1000 + i * 80));
        }
        let map = rosu_pp::Beatmap::from_bytes(text.as_bytes()).unwrap();
        for (bits, rate) in [(0u32, 1.), (64, 1.5), (256, 0.75)] {
            let attrs = rosu_pp::Difficulty::new().mods(bits).lazer(true).calculate(&map);
            let stars = attrs.stars();
            let stats = [940i64, 30, 10, 10, 5, 5];
            let (acc, hits) = ms_data::store::acc320(&stats);
            let pp = rosu_pp::Performance::new(attrs).mods(bits).lazer(true).n_geki(940).n300(30).n_katu(10).n100(10).n50(5).misses(5).calculate().pp();
            let ours = pp_scale(stars, hits as f64) * (5. * acc - 4.).max(0.);
            assert!((ours - pp).abs() < 1e-6, "{rate}: {ours} {pp}");
        }
    }
}
