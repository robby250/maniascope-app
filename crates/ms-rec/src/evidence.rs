//! recommend.py's evidence side: chart features for a score, the personal rows, keymode levels.
use chrono::Datelike;
use ms_calc::py::round_digits;
use ms_calc::val::{Dict, Val};
use ms_data::codec;
use ms_data::public::Public;
use ms_data::store::{acc_lazer, parse_stats, timestamp, Db, Result};
use ms_model::personal::{Best, Row};
use ms_model::predict::{Consts, Feats, HashMap};
use rusqlite::types::ValueRef;
use serde_json::Value;
use std::collections::HashSet;
use std::sync::Arc;

const EVIDENCE_MODS: [&str; 13] = ["NF", "HD", "FI", "FL", "MR", "SD", "PF", "DT", "NC", "HT", "DC", "CL", "RD"];
const STABLE_MODS: i64 = 1 | 8 | 64 | 256 | 512 | 1024 | 1048576 | 1073741824 | 16384 | 32; // stage1 rules drop the rest
pub const MIN_LAZER_LN: usize = 30;   // lazer LN plays of a keymode from which stable ones stop being evidence
const STABLE_LN: f64 = 0.35;      // LN share up to which a stable result stands in for a lazer one

pub fn months(iso: &str) -> Option<f64> {
    let part = |r: std::ops::Range<usize>| iso.get(r)?.parse::<f64>().ok();
    Some((part(0..4)? - 2000.) * 12. + part(5..7)? - 1. + part(8..10)? / 31.)
}

pub fn now_months() -> f64 {
    let d = chrono::Local::now().date_naive();
    (f64::from(d.year()) - 2000.) * 12. + f64::from(d.month()) - 1. + f64::from(d.day()) / 31.
}

pub fn y_of(acc: f64) -> f64 {
    (1. - acc.min(0.999)).max(1e-3).ln()
}

/// A score row with its features.
pub struct Evidence {
    pub y: f64,
    pub y_lazer: f64,
    pub keys: i64,
    pub t: f64,
    pub chart: String,
    pub sha: Option<String>,
    pub f: usize,           // into Rows.feats
    pub src: String,
    pub client: Option<String>,
    pub ts: Option<f64>,
    pub b: Option<i64>,
    pub rate: f64,
    pub key: String,
    pub rd: bool,
    pub completed: bool,
    pub ln: f64,
    pub overall: f64,
    pub family: String,     // nps.family, else the chart
}

#[derive(Default)]
pub struct Rows {
    pub rows: Vec<Evidence>,
    pub feats: Vec<Feats>,
    pub skipped: std::collections::BTreeMap<&'static str, usize>,
}

/// Chart features for a (chart, rate): the local table first, else the public one when the md5 matches.
pub struct FeatureSource {
    pub public: Arc<Public>,
    pub calc: String,
    pub cache: HashMap<String, Arc<Dict>>,
}

impl FeatureSource {
    pub fn key(&self, sha: &str, rate: f64) -> String {
        format!("{sha}@{rate:.3}|{}", self.calc)
    }

    /// One query per 900 keys: 6.8k single reads took 0.4 s on a cold disk cache, batched 0.05 s.
    pub fn prefetch(&mut self, db: &Db, keys: impl Iterator<Item = String>) -> Result<()> {
        let mut seen = HashSet::new();
        let need: Vec<String> = keys.filter(|k| !self.cache.contains_key(k) && seen.insert(k.clone())).collect();
        for chunk in need.chunks(900) {
            let query = format!("SELECT key, data FROM feats WHERE key IN ({})", vec!["?"; chunk.len()].join(","));
            let mut statement = db.prepare(&query)?;
            let rows = statement.query_map(rusqlite::params_from_iter(chunk), |r| {
                Ok((r.get::<_, String>(0)?, match r.get_ref(1)? {
                    ValueRef::Blob(b) | ValueRef::Text(b) => b.to_vec(),
                    _ => Vec::new(),
                }))
            })?;
            for row in rows {
                let (key, data) = row?;
                if let Ok(Val::Dict(d)) = codec::load(&data) {
                    self.cache.insert(key, Arc::new(d));
                }
            }
        }
        Ok(())
    }

    pub fn lookup(&mut self, db: &Db, key: &str) -> Result<Option<Arc<Dict>>> {
        if let Some(f) = self.cache.get(key) {
            return Ok(Some(f.clone()));
        }
        self.prefetch(db, std::iter::once(key.to_string()))?;
        Ok(self.cache.get(key).cloned())
    }

    /// (features, the snapshot's beatmap id) of a score. `stars`/`hits` of the public row are not carried over: only
    /// the PP path reads them, from the public table.
    pub fn of(&mut self, db: &Db, sha: Option<&str>, md5: Option<&str>, bid: Option<i64>, rate: Option<f64>) -> Result<(Option<Arc<Dict>>, Option<i64>)> {
        let Some(rate) = rate else { return Ok((None, None)) };
        let mut b = bid.filter(|&b| b > 0).or_else(|| md5.and_then(|m| self.public.md5_bid.get(m).copied()));
        if b.is_some() && (md5.is_none() || b.and_then(|b| self.public.map_info(b)).map(|m| m.md5.as_str()) != md5) {
            b = None; // a stale id: not the snapshot's chart
        }
        let rate = round_digits(if rate == 0. { 1. } else { rate }, 3);
        if let Some(sha) = sha {
            let key = self.key(sha, rate);
            if let Some(f) = self.lookup(db, &key)? {
                return Ok((Some(f), b));
            }
        }
        if let Some(bid) = b {
            let chart = self.public.meta.charts.binary_search_by(|c| c.0.cmp(&bid)).ok().map(|i| &self.public.meta.charts[i]);
            if chart.is_some_and(|c| c.2 == self.calc && Some(c.1.as_str()) == md5) {
                // Kept, so one chart-rate is one object: the rows share it by address.
                let key = format!("public:{bid}@{rate}");
                if let Some(f) = self.cache.get(&key) {
                    return Ok((Some(f.clone()), b));
                }
                if let Some(Val::Dict(d)) = self.public.feature(bid, rate) {
                    let f = Arc::new(d);
                    self.cache.insert(key, f.clone());
                    return Ok((Some(f), b));
                }
            }
        }
        Ok((None, b))
    }
}

/// recommend.evidence_rows.
pub fn evidence_rows(db: &Db, source: &mut FeatureSource, c: &Consts) -> Result<Rows> {
    struct Raw { key: String, src: String, client: Option<String>, sha: Option<String>, md5: Option<String>, bid: Option<i64>, played: Option<String>,
                 mods: Option<String>, rate: Option<f64>, stats: Option<String>, acc: Option<f64>, n: Option<i64> }
    let mut statement = db.prepare("SELECT key,src,client,sha256,md5,beatmap_id,played,mods,rate,stats,acc,n FROM scores")?;
    let scores: Vec<Raw> = statement.query_map([], |r| Ok(Raw { key: r.get(0)?, src: r.get(1)?, client: r.get(2)?, sha: r.get(3)?, md5: r.get(4)?, bid: r.get(5)?,
        played: r.get(6)?, mods: r.get(7)?, rate: r.get(8)?, stats: r.get(9)?, acc: r.get(10)?, n: r.get(11)? }))?.collect::<Result<_>>()?;
    source.prefetch(db, scores.iter().filter(|r| r.rate.is_some()).filter_map(|r| {
        Some(format!("{}@{:.3}|{}", r.sha.as_ref()?, round_digits(r.rate?.max(0.) + f64::from(u8::from(r.rate? == 0.)), 3), source.calc))
    }).collect::<Vec<_>>().into_iter())?;
    let mut out = Rows::default();
    let mut dedupe: HashMap<*const Dict, usize> = HashMap::default();
    for r in &scores {
        let mods: Value = r.mods.as_deref().and_then(|m| serde_json::from_str(m).ok()).unwrap_or(Value::Array(Vec::new()));
        let mut rd = false;
        if let Some(bits) = mods.as_i64() {
            if bits & !STABLE_MODS != 0 {
                *out.skipped.entry("mods").or_default() += 1;
                continue;
            }
        } else {
            let acronyms: Vec<String> = mods.as_array().map_or(&[][..], Vec::as_slice).iter()
                .map(|m| m.get("acronym").and_then(Value::as_str).unwrap_or("").to_uppercase()).collect();
            rd = acronyms.iter().any(|a| a == "RD");
            if acronyms.iter().any(|a| !EVIDENCE_MODS.contains(&a.as_str())) || r.rate.is_none() {
                *out.skipped.entry("mods").or_default() += 1;
                continue;
            }
        }
        let acc = r.acc.unwrap_or(0.);
        if acc == 0. || acc < 0.5 || r.n.unwrap_or(0) < 100 {
            *out.skipped.entry("short/failed").or_default() += 1;
            continue;
        }
        let (f, b) = source.of(db, r.sha.as_deref(), r.md5.as_deref(), r.bid, r.rate)?;
        let Some(f) = f else {
            *out.skipped.entry("no features").or_default() += 1;
            continue;
        };
        let Some(t) = months(r.played.as_deref().unwrap_or("")) else { continue };
        let at = match dedupe.get(&Arc::as_ptr(&f)) {
            Some(&at) => at,
            None => {
                let Ok(feats) = Feats::of(&f, c) else { continue };
                out.feats.push(feats);
                dedupe.insert(Arc::as_ptr(&f), out.feats.len() - 1);
                out.feats.len() - 1
            }
        };
        let chart = r.md5.clone().filter(|m| !m.is_empty()).or_else(|| r.sha.clone()).unwrap_or_default();
        let ln = f.fget("ln", 0.);
        let family = f.dict("nps").and_then(|n| n.get("family")).filter(|v| v.truthy()).map_or_else(|| chart.clone(), |v| match v {
            Val::Str(s) => s.clone(),
            other => format!("{other:?}"),
        });
        let notes = f.fget("notes", 0.);
        let stats = r.stats.as_deref().and_then(parse_stats);
        out.rows.push(Evidence {
            y: y_of(acc),
            y_lazer: y_of(stats.as_ref().map_or(0., acc_lazer)),
            keys: out.feats[at].keys,
            t,
            chart,
            sha: r.sha.clone(),
            f: at,
            src: r.src.clone(),
            client: r.client.clone(),
            ts: r.played.as_deref().and_then(timestamp),
            b,
            rate: r.rate.unwrap_or(1.),
            key: r.key.clone(),
            rd,
            completed: completed(r.n.unwrap_or(0), notes, ln, lazer_judged(&r.key, r.client.as_deref())),
            ln,
            overall: out.feats[at].overall,
            family,
        });
    }
    Ok(out)
}

fn lazer_judged(key: &str, client: Option<&str>) -> bool {
    !key.starts_with("legacy|") && client.is_some_and(|c| c.ends_with("-lazer") || c == "lazer-live")
}

/// lazer judges a hold's head and tail apart; stable merges them into one judgement. Asking stable for both left 348 of
/// iess's 441 plays out of the Stats line, most of their hard ones.
fn completed(judged: i64, notes: f64, ln: f64, lazer: bool) -> bool {
    judged as f64 >= 0.95 * notes * if lazer { 1. + ln } else { 1. }
}

/// Rows the personal fit reads. Both forecasts target lazer plays. Stable merges a hold's head and tail into one
/// judgement; neither 305 nor 320 reweighting reconstructs the two separate judgements. Stable tap scores remain
/// transfer evidence. A keymode without lazer LN plays keeps charts up to STABLE_LN: most 7K charts hold some LN, so
/// the rule left a stable player's hardest plays out and their level on easy rice (Less 2026-11-01: 85 of 100 bests
/// dropped). G533QR's stable plays predicting its lazer LN plays: 7K RMSE .374 → .343, 4K .843 → .560; all of them
/// .434, 1.046.
pub fn stable_evidence<'a>(rows: impl Iterator<Item = &'a Evidence> + Clone) -> Vec<&'a Evidence> {
    let lazer = |r: &Evidence| lazer_judged(&r.key, r.client.as_deref());
    let mut lazer_ln: HashMap<i64, usize> = HashMap::default();
    rows.clone().filter(|r| r.ln != 0. && lazer(r)).for_each(|r| *lazer_ln.entry(r.keys).or_insert(0) += 1);
    rows.filter(|r| r.ln == 0. || lazer(r) || (lazer_ln.get(&r.keys).copied().unwrap_or(0) < MIN_LAZER_LN && r.ln <= STABLE_LN)).collect()
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    let at = q / 100. * (sorted.len() - 1) as f64;
    let (lo, hi) = (at.floor() as usize, at.ceil() as usize);
    sorted[lo] + (sorted[hi] - sorted[lo]) * (at - lo as f64)
}

/// Mean log stars of the plays within 70 % of the 90th percentile: where the player plays.
pub fn competence_level(log_ratings: &[f64]) -> f64 {
    let mut sorted = log_ratings.to_vec();
    sorted.sort_by(f64::total_cmp);
    let bar = percentile(&sorted, 90.) + 0.7f64.ln();
    let kept: Vec<f64> = log_ratings.iter().copied().filter(|&x| x >= bar).collect();
    kept.iter().sum::<f64>() / kept.len() as f64
}

/// {keys: competence level (log stars)} from the player's plays of the last year (all plays when fewer than 20), the
/// same zone the population fit centres its gap curves on.
pub fn play_levels(rows: &[&Evidence], now: f64) -> HashMap<i64, f64> {
    let mut by: HashMap<i64, Vec<&Evidence>> = HashMap::default();
    rows.iter().for_each(|r| by.entry(r.keys).or_default().push(r));
    by.into_iter().map(|(k, rs)| {
        let recent: Vec<&&Evidence> = rs.iter().filter(|r| r.t >= now - 12.).collect();
        let use_: Vec<f64> = if recent.len() >= 20 { recent.iter().map(|r| r.overall.max(0.05).ln()).collect() } else { rs.iter().map(|r| r.overall.max(0.05).ln()).collect() };
        (k, competence_level(&use_))
    }).collect()
}

/// Row of the Personal fit: `y` and `base` chosen by the caller.
pub fn personal_row(r: &Evidence, y: f64, base: f64) -> Row {
    Row {
        keys: r.keys, y, base, t: r.t, ts: r.ts.unwrap_or(0.), public: r.src == "public", local: r.src == "realm" || r.src == "live", rd: r.rd,
        completed: r.completed, chart: r.chart.clone(), rate: r.rate, rate3: round_digits(r.rate, 3), best: Best::Key(r.key.clone()),
        family: r.family.clone(), f: r.f,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stable_play_counts_each_hold_once() {
        assert!(completed(1000, 1000., 0.3, false));
        assert!(!completed(1000, 1000., 0.3, true));
        assert!(completed(1300, 1000., 0.3, true));
        assert!(!completed(900, 1000., 0.3, false));
        assert!(!lazer_judged("legacy|1", Some("osu-lazer")) && lazer_judged("x", Some("g835lx-lazer")) && !lazer_judged("st|x", Some("stable")));
    }
}
