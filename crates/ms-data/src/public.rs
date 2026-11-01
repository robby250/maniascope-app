//! The population evidence bundle (public.pkl) as one mapped file: small tables in a JSON header, each analysed
//! chart-rate a feature blob read only when asked. The pickle held everything as Python objects, ~0.9 GB resident.
use crate::blobs::{self, Blobs};
use crate::codec;
use crate::pickle::{self, P};
use ms_calc::val::{Dict, Key, Val};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;


#[derive(Clone, Default, Serialize, Deserialize)]
pub struct Population {
    pub slope: Vec<(i64, f64)>,
    pub window: Vec<(i64, f64)>,
    pub curve: Vec<(i64, f64)>,
    pub curve_knee: f64,
    pub gap: Vec<(i64, Vec<(String, Vec<f64>)>)>,
    pub share_mean: Vec<(i64, Vec<(String, f64)>)>,
    pub mb: Vec<(i64, f64, f64)>,
    pub mbr: Vec<(i64, f64, f64, f64)>,
    pub sig2: f64,
    pub tau2: f64,
    pub kappa: f64,
    pub rows: i64,
    pub players: i64,
    #[serde(default)]
    pub version: i64, // the evidence's link version: POP_VERSION
}

#[derive(Clone, Default, Serialize, Deserialize)]
pub struct MapInfo {
    pub bid: i64,
    pub keys: i64,
    pub od: f64,
    pub set: i64,
    pub approved: i64,
    pub file: String,
    pub md5: String,
}

#[derive(Clone, Default, Serialize, Deserialize)]
pub struct UserScore {
    pub score_id: i64,
    pub beatmap_id: i64,
    pub score: i64,
    pub stats: Vec<i64>,
    pub mods: i64,
    pub rate: f64,
    pub ranked: bool,
    pub date: String,
    pub pp: Option<f64>,
    pub hidden: i64,
}

#[derive(Clone, Default, Serialize, Deserialize)]
pub struct Meta {
    pub snapshot: String,
    pub built: String,
    pub calc: String,
    pub pop: Population,
    pub pop_lazer: Option<Population>,
    pub maps: Vec<MapInfo>,
    pub charts: Vec<(i64, String, String)>, // bid, md5, calc of the analysed charts
    pub user_id: i64,
    pub user_scores: Vec<UserScore>,
    pub user_stats: serde_json::Value,
    pub playcount: Vec<(i64, i64)>,
    pub elasticities: Vec<(String, f64)>, // warmup.rate_elasticities: fixed for a snapshot, expensive to read
}

pub struct Public {
    pub meta: Meta,
    blobs: Blobs,
    pub by_bid: HashMap<i64, usize>, // → meta.maps
    pub md5_bid: HashMap<String, i64>,
}

const MAGIC: &[u8; 8] = b"MSPUB001";

/// Evidence fitted by older links is replaced when the current bundle can be fetched.
pub const POP_VERSION: i64 = 2;

pub fn write(path: &Path, meta: &Meta, records: &[(i64, f64, Vec<u8>)]) -> Result<(), String> {
    blobs::write(path, MAGIC, &serde_json::to_vec(meta).map_err(|e| e.to_string())?, records)
}

fn num(p: &P, key: &str) -> f64 {
    p.get(key).and_then(P::f).unwrap_or(0.)
}

fn table(p: Option<&P>) -> Vec<(i64, f64)> {
    p.map_or(&[][..], P::items).iter().filter_map(|(k, v)| Some((k.i()?, v.f()?))).collect()
}

fn population(p: &P) -> Population {
    Population {
        slope: table(p.get("slope")),
        window: table(p.get("window")),
        curve: table(p.get("curve")),
        curve_knee: p.get("curve_knee").and_then(P::f).unwrap_or(4.),
        gap: p.get("gap").map_or(&[][..], P::items).iter().filter_map(|(k, parts)| {
            Some((k.i()?, parts.items().iter().filter_map(|(name, c)| Some((name.s()?.to_string(), c.seq().iter().filter_map(P::f).collect()))).collect()))
        }).collect(),
        share_mean: p.get("share_mean").map_or(&[][..], P::items).iter().filter_map(|(k, parts)| {
            Some((k.i()?, parts.items().iter().filter_map(|(name, v)| Some((name.s()?.to_string(), v.f()?))).collect()))
        }).collect(),
        mb: p.get("mb").map_or(&[][..], P::items).iter().filter_map(|(k, v)| Some((k.i()?, v.seq().first()?.f()?, v.seq().get(1)?.f()?))).collect(),
        mbr: p.get("mbr").map_or(&[][..], P::items).iter().filter_map(|(k, v)| {
            Some((k.seq().first()?.i()?, k.seq().get(1)?.f()?, v.seq().first()?.f()?, v.seq().get(1)?.f()?))
        }).collect(),
        sig2: num(p, "sig2"),
        tau2: num(p, "tau2"),
        kappa: num(p, "kappa"),
        rows: num(p, "rows") as i64,
        players: num(p, "players") as i64,
        version: p.get("version").and_then(P::f).map_or(1, |v| v as i64),
    }
}

/// A pickled value as a feature value.
pub fn value(p: &P) -> Val {
    match p {
        P::None => Val::None,
        P::Bool(b) => Val::Bool(*b),
        P::Int(i) => Val::Int(*i),
        P::F(x) => Val::F(*x),
        P::Str(s) => Val::Str(s.to_string()),
        P::Bytes(_) => Val::None,
        P::List(v) => Val::List(v.iter().map(value).collect()),
        P::Tuple(v) => Val::Tuple(v.iter().map(value).collect()),
        P::Dict(d) => Val::Dict(Dict(d.iter().map(|(k, v)| {
            (match k {
                P::Bool(b) => Key::B(*b),
                P::Int(i) => Key::F(*i as f64),
                P::F(x) => Key::F(*x),
                P::Str(s) => Key::S(s.to_string()),
                other => Key::S(format!("{other:?}")),
            }, value(v))
        }).collect())),
    }
}

/// public.pkl → the header and the sorted (bid, rate, blob) records.
pub fn from_pickle(raw: &[u8]) -> Result<(Meta, Vec<(i64, f64, Vec<u8>)>), String> {
    let root = pickle::load(raw)?;
    let text = |k: &str| root.get(k).and_then(P::s).unwrap_or("").to_string();
    let user = root.get("user").ok_or("public data has no user")?;
    let mut meta = Meta {
        snapshot: text("snapshot"),
        built: text("built"),
        calc: text("calc"),
        pop: population(root.get("pop").ok_or("public data has no population")?),
        pop_lazer: root.get("pop_lazer").map(population),
        maps: root.get("maps").map_or(&[][..], P::items).iter().filter_map(|(b, m)| {
            Some(MapInfo { bid: b.i()?, keys: num(m, "keys") as i64, od: num(m, "od"), set: num(m, "set") as i64, approved: num(m, "approved") as i64,
                           file: m.get("file").and_then(P::s).unwrap_or("").to_string(), md5: m.get("md5").and_then(P::s).unwrap_or("").to_string() })
        }).collect(),
        user_id: user.get("id").and_then(P::i).unwrap_or(0),
        user_scores: user.get("scores").map_or(&[][..], P::seq).iter().map(|s| UserScore {
            score_id: num(s, "score_id") as i64, beatmap_id: num(s, "beatmap_id") as i64, score: num(s, "score") as i64,
            stats: s.get("stats").map_or(&[][..], P::seq).iter().filter_map(P::i).collect(), mods: num(s, "mods") as i64, rate: num(s, "rate"),
            ranked: s.get("ranked").and_then(P::f).is_some_and(|x| x != 0.), date: s.get("date").and_then(P::s).unwrap_or("").to_string(),
            pp: s.get("pp").and_then(P::f), hidden: num(s, "hidden") as i64,
        }).collect(),
        user_stats: user.get("stats").map_or(serde_json::Value::Null, json),
        playcount: table(user.get("playcount")).into_iter().map(|(b, n)| (b, n as i64)).collect(),
        ..Meta::default()
    };
    let mut records = Vec::new();
    for (b, fs) in root.get("feats").map_or(&[][..], P::items) {
        let Some(bid) = b.i() else { continue };
        meta.charts.push((bid, fs.get("md5").and_then(P::s).unwrap_or("").to_string(), fs.get("calc").and_then(P::s).unwrap_or("").to_string()));
        for (rate, f) in fs.items() {
            if let P::F(rate) = rate {
                records.push((bid, *rate, codec::encode(&value(f))));
            }
        }
    }
    records.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
    meta.charts.sort_by_key(|c| c.0);
    Ok((meta, records))
}

fn json(p: &P) -> serde_json::Value {
    use serde_json::Value as J;
    match p {
        P::None => J::Null,
        P::Bool(b) => J::Bool(*b),
        P::Int(i) => J::from(*i),
        P::F(x) => J::from(*x),
        P::Str(s) => J::from(s.as_ref()),
        P::Bytes(_) => J::Null,
        P::List(v) | P::Tuple(v) => J::Array(v.iter().map(json).collect()),
        P::Dict(d) => J::Object(d.iter().map(|(k, v)| (k.s().map_or_else(|| format!("{k:?}"), str::to_string), json(v))).collect()),
    }
}

impl Public {
    pub fn open(path: &Path) -> Result<Public, String> {
        let (blobs, header) = Blobs::open(path, MAGIC)?;
        let meta: Meta = serde_json::from_slice(blobs.header(header)).map_err(|e| e.to_string())?;
        let by_bid = meta.maps.iter().enumerate().map(|(i, m)| (m.bid, i)).collect();
        let md5_bid = meta.maps.iter().filter(|m| !m.md5.is_empty()).map(|m| (m.md5.clone(), m.bid)).collect();
        Ok(Public { meta, blobs, by_bid, md5_bid })
    }

    pub fn map_info(&self, bid: i64) -> Option<&MapInfo> {
        self.by_bid.get(&bid).map(|&i| &self.meta.maps[i])
    }

    pub fn release(&self) {
        self.blobs.release();
    }

    pub fn rates(&self, bid: i64) -> Vec<f64> {
        self.blobs.rates(bid)
    }

    pub fn feature(&self, bid: i64, rate: f64) -> Option<Val> {
        self.blobs.feature(bid, rate)
    }

    /// Every chart-rate in (bid, rate) order.
    pub fn each(&self) -> impl Iterator<Item = (i64, f64, Val)> + '_ {
        self.blobs.each()
    }
}

/// The distributable metadata: population evidence without the builder's own scores (recdata.release_public).
pub fn released(meta: &Meta) -> Meta {
    Meta { user_id: 0, user_scores: Vec::new(), user_stats: serde_json::Value::Null, playcount: Vec::new(), ..meta.clone() }
}

const KEEP_ROLLING: usize = 2; // active + one previous, for rollback

pub fn active() -> Option<(String, Vec<serde_json::Value>)> {
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(crate::paths::rolling().join("active.json")).ok()?).ok()?;
    Some((v["name"].as_str()?.to_string(), v["history"].as_array().cloned().unwrap_or_default()))
}

/// Atomic switch of the active snapshot, then bounded pruning of superseded rolling snapshots only.
pub fn activate(name: &str) -> std::io::Result<()> {
    let root = crate::paths::rolling();
    let mut history = active().map_or_else(Vec::new, |a| a.1);
    history.retain(|h| h != name);
    history.push(serde_json::json!(name));
    let kept = history[history.len().saturating_sub(KEEP_ROLLING)..].to_vec();
    let tmp = root.join("active.json.part");
    std::fs::write(&tmp, serde_json::json!({"name": name, "history": kept, "t": crate::store::now()}).to_string())?;
    std::fs::rename(tmp, root.join("active.json"))?;
    for e in std::fs::read_dir(&root)?.flatten() {
        let n = e.file_name().to_string_lossy().into_owned();
        if e.path().is_dir() && !kept.iter().any(|k| k == n.as_str()) && !n.starts_with("staging-") {
            std::fs::remove_dir_all(e.path())?;
        }
    }
    Ok(())
}

pub const INDEX_URL: &str = "https://data.ppy.sh/";

/// Newest mania top-1000 snapshot on data.ppy.sh (one small request), recorded in kv. → the newer snapshot's name.
pub fn check_index(db: &crate::store::Db) -> Result<Option<String>, String> {
    let body = ureq::get(INDEX_URL).timeout(std::time::Duration::from_secs(20)).call().map_err(|e| e.to_string())?.into_string().map_err(|e| e.to_string())?;
    let names = regex_lite::Regex::new(r"(\d{4}_\d\d_\d\d)_performance_mania_top_1000\.tar\.bz2").unwrap();
    let newest = names.captures_iter(&body).map(|m| m[1].to_string()).max().map(|n| format!("{n}_performance_mania_top_1000"));
    let active: Option<String> = std::fs::read(crate::paths::rolling().join("active.json")).ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok()).and_then(|v| v["name"].as_str().map(str::to_string));
    let newer = matches!((&newest, &active), (Some(n), Some(a)) if n > a);
    crate::store::kv_set(db, "snapshot_check", &serde_json::json!({"t": crate::store::now(), "newest": newest, "active": active, "newer": newer})).map_err(|e| e.to_string())?;
    Ok(newest.filter(|_| newer))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn released_evidence_carries_nothing_of_the_builder() {
        let own = Meta { calc: "c".into(), snapshot: "s".into(), user_id: 5, user_scores: vec![UserScore::default()], user_stats: serde_json::json!({"a": 1}), playcount: vec![(1, 2)], ..Meta::default() };
        let out = released(&own);
        assert!(out.user_id == 0 && out.user_scores.is_empty() && out.user_stats.is_null() && out.playcount.is_empty());
        assert_eq!((out.calc.as_str(), own.user_scores.len()), ("c", 1));
    }
}
