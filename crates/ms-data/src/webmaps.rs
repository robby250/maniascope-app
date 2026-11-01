//! Website charts for the NPS and Skills lists: every status, analysed like installed maps. osu.direct metadata →
//! chart text (.osu only, no audio) for charts whose density can reach the NPS range at 0.70–1.50× → features on a
//! rate grid → WEB_DIR/web.msp, the fields the lists read, one blob per rate. Treated as not installed: Next sends
//! osu://b/<id> so the client offers the download. A calculator change needs another crawl (only analysis reruns;
//! files and metadata are cached). `stop` in WEB_DIR pauses a crawl, the next one resumes from state.db.
use crate::blobs::{self, Blobs};
use crate::codec;
use crate::paths;
use crate::pickle::{self, P};
use crate::public::value;
use crate::store::now;
use ms_calc::features::{features_batch};
use ms_calc::params::Params;
use ms_calc::val::{Dict, Val};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Duration;

const SEARCH: &str = "https://osu.direct/api/v2/search?mode=3&amount=100&sort=last_updated:desc";
// osu.direct first; ppy only for what the mirror lacks, slowly: both laptops share the home IP that the client uses,
// so a throttled IP at ppy would hurt play (user concern 2026-09-30).
pub const FILES: [(&str, f64); 2] = [("https://osu.direct/api/osu/", 0.6), ("https://osu.ppy.sh/osu/", 3.)];
const UA: &str = "maniascope/1.0 (personal osu!mania practice recommender)";
const STATUSES: [i64; 7] = [-2, -1, 0, 1, 2, 3, 4]; // graveyard, wip, pending, ranked, approved, qualified, loved
// HT/NM/DT only (user 2026-10-02: ~3x faster crawl, ~3x smaller file). NPS/Skills compute the exact rate a candidate
// needs on demand, as for installed maps.
pub const RATES: [f64; 3] = [0.75, 1., 1.5];
pub const PAUSE: f64 = 0.6;   // osu.direct allows 120 requests/minute
const PAGES: i64 = 4;     // pages in flight (~3 s each server-side → ~60/min)
// Before the first run on this PC (measured on G533QR 2026-10-01); runs learn their own.
const SEC_FETCH: f64 = 2.4;
const SEC_ANALYSE: f64 = 0.7;
const CHART_BYTES: i64 = 150_000;
const MAGIC: &[u8; 8] = b"MSWEB001";

pub fn web_dir() -> PathBuf {
    paths::rec_dir().join("web")
}

pub fn workers(n: Option<usize>) -> usize {
    n.unwrap_or_else(|| std::thread::available_parallelism().map_or(2, |x| x.get()).saturating_sub(2).max(1))
}

#[derive(Debug)]
pub enum CrawlError {
    Stopped,
    Failed(String),
}

impl<E: std::fmt::Display> From<E> for CrawlError {
    fn from(e: E) -> CrawlError {
        CrawlError::Failed(e.to_string())
    }
}

pub fn get(url: &str) -> Result<Vec<u8>, String> {
    let mut last = String::new();
    for attempt in 0..4u64 {
        match ureq::get(url).set("User-Agent", UA).timeout(Duration::from_secs(30)).call() {
            Ok(r) => {
                let mut out = Vec::new();
                match std::io::Read::read_to_end(&mut r.into_reader(), &mut out) {
                    Ok(_) => return Ok(out),
                    Err(e) => last = e.to_string(),
                }
            }
            Err(ureq::Error::Status(404, _)) => return Ok(Vec::new()),
            Err(ureq::Error::Status(429, _)) => {
                last = "429".into();
                std::thread::sleep(Duration::from_secs(60)); // rate limited, e.g. another client on this IP
            }
            Err(e) => last = e.to_string(),
        }
        std::thread::sleep(Duration::from_secs(5 * (attempt + 1)));
    }
    Err(last)
}

pub fn state_db() -> rusqlite::Result<Connection> {
    let dir = web_dir();
    let _ = std::fs::create_dir_all(dir.join("osu"));
    let db = Connection::open(dir.join("state.db"))?;
    db.busy_timeout(Duration::from_secs(5))?;
    db.execute_batch("CREATE TABLE IF NOT EXISTS diffs(bid INTEGER PRIMARY KEY, set_id INTEGER, status INTEGER, keys INTEGER,
          md5 TEXT, notes INTEGER, length REAL, playcount INTEGER, updated TEXT, artist TEXT, title TEXT,
          version TEXT, creator TEXT, sha TEXT, file_md5 TEXT, fetched TEXT, error TEXT, stars REAL);
        CREATE TABLE IF NOT EXISTS feats(sha TEXT, calc TEXT, data BLOB, PRIMARY KEY(sha, calc));
        CREATE TABLE IF NOT EXISTS kv(k TEXT PRIMARY KEY, v TEXT);")?;
    let has_stars: bool = db.prepare("PRAGMA table_info(diffs)")?.query_map([], |r| r.get::<_, String>(1))?.flatten().any(|n| n == "stars");
    if !has_stars {
        db.execute("ALTER TABLE diffs ADD COLUMN stars REAL", [])?; // nomod star rating from the listing
    }
    Ok(db)
}

pub fn kv_get(db: &Connection, k: &str) -> Option<Value> {
    db.query_row("SELECT v FROM kv WHERE k=?", [k], |r| r.get::<_, String>(0)).ok().and_then(|v| serde_json::from_str(&v).ok())
}

pub fn kv_set(db: &Connection, k: &str, v: &Value) -> rusqlite::Result<()> {
    db.execute("INSERT OR REPLACE INTO kv VALUES(?,?)", params![k, v.to_string()]).map(|_| ())
}

fn check_stop() -> Result<(), CrawlError> {
    if web_dir().join("stop").exists() { Err(CrawlError::Stopped) } else { Ok(()) }
}

/// The OS releases it when the crawl exits however it ends; None while another crawl holds it.
pub fn lock() -> Option<Connection> {
    let lock = Connection::open(web_dir().join("crawl.lock")).ok()?;
    lock.busy_timeout(Duration::ZERO).ok()?;
    lock.execute_batch("BEGIN EXCLUSIVE").ok().map(|()| lock)
}

pub fn running() -> bool {
    lock().is_none()
}

fn progress(db: &Connection, fields: Value) -> rusqlite::Result<()> {
    let mut p = kv_get(db, "progress").unwrap_or_else(|| json!({}));
    p["t"] = json!(now());
    if let (Some(o), Some(f)) = (p.as_object_mut(), fields.as_object()) {
        o.extend(f.iter().map(|(k, v)| (k.clone(), v.clone())));
    }
    kv_set(db, "progress", &p)
}

/// Time left for these counts at this PC's measured pace (defaults before its first run).
pub fn seconds(db: &Connection, fetch: i64, analyse: i64) -> f64 {
    let number = |k: &str| kv_get(db, k).and_then(|v| v.as_f64()).filter(|&x| x != 0.);
    fetch as f64 * number("sec_fetch").unwrap_or(SEC_FETCH) + analyse as f64 * number("sec_analyse").unwrap_or(SEC_ANALYSE / workers(None) as f64)
}

pub fn disk_bytes() -> u64 {
    let size = |d: PathBuf| std::fs::read_dir(d).into_iter().flatten().flatten().filter_map(|e| e.metadata().ok()).filter(|m| m.is_file()).map(|m| m.len()).sum::<u64>();
    size(web_dir()) + size(web_dir().join("osu"))
}

/// Charts without a listed rating stay in; the filter only drops what it can see.
fn in_stars(stars: Option<f64>, range: Option<(f64, f64)>) -> bool {
    match (stars, range) {
        (Some(s), Some((lo, hi))) => lo <= s && s <= hi,
        _ => true,
    }
}

/// The metadata density can reach the NPS range at 1.5× (the NPS ranking's density floor ≈ 20·k/7).
fn reachable(notes: i64, length: f64, keys: i64) -> bool {
    notes >= 100 && length >= 30. && notes as f64 / length * 1.5 >= 0.6 * 20. * keys as f64 / 7.
}

fn listed_keys(db: &Connection) -> rusqlite::Result<HashSet<i64>> {
    if let Some(saved) = kv_get(db, "keys").and_then(|v| v.as_array().cloned()) {
        return Ok(saved.iter().filter_map(Value::as_i64).collect());
    }
    db.prepare("SELECT DISTINCT keys FROM diffs")?.query_map([], |r| r.get(0))?.collect()
}

#[derive(Serialize, Clone, Debug)]
pub struct Estimate {
    pub fetch: i64,
    pub analyse: i64,
    pub ready: i64,
    pub new_bytes: i64,
    pub seconds: f64,
    pub disk: u64,
    pub checked: Option<f64>,
    pub unlisted: Vec<i64>,
}

/// What a crawl for these keymodes and ★ range still has to do.
pub fn estimate(db: &Connection, keys: &HashSet<i64>, stars: Option<(f64, f64)>, calc: &str) -> rusqlite::Result<Estimate> {
    let (mut fetch, mut analyse, mut ready) = (0, 0, 0);
    let mut q = db.prepare("SELECT d.keys,d.notes,d.length,d.stars,d.sha,d.fetched,d.md5,f.sha IS NOT NULL FROM diffs d LEFT JOIN feats f ON f.sha=d.sha AND f.calc=? WHERE d.error IS NULL")?;
    let rows = q.query_map([calc], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?, r.get::<_, f64>(2)?, r.get::<_, Option<f64>>(3)?, r.get::<_, Option<String>>(4)?,
                                           r.get::<_, Option<String>>(5)?, r.get::<_, Option<String>>(6)?, r.get::<_, bool>(7)?)))?;
    for row in rows {
        let (k, notes, length, s, sha, fetched, md5, done) = row?;
        if !keys.contains(&k) || !reachable(notes, length, k) || !in_stars(s, stars) {
            continue;
        }
        if sha.is_none() || fetched != md5 {
            fetch += 1;
        } else if !done {
            analyse += 1;
        } else {
            ready += 1;
        }
    }
    let listed = listed_keys(db)?;
    let mut unlisted: Vec<i64> = keys.difference(&listed).copied().collect();
    unlisted.sort_unstable();
    Ok(Estimate { fetch, analyse: fetch + analyse, ready, new_bytes: fetch * CHART_BYTES, seconds: seconds(db, fetch, fetch + analyse), disk: disk_bytes(),
                  checked: kv_get(db, "checked").and_then(|v| v.as_f64()), unlisted })
}

/// Newest-first per status; an incremental crawl stops a day past the last watermark.
/// Where a listing for these keymodes stopped: (status index, offset, sets, newest). A pause or a closed app lost a full pass
/// before this, so a keymode that needs one never got listed (user 2026-11-01: "it always checks the whole listing").
fn listing_at(db: &Connection, keys: &HashSet<i64>) -> Option<(usize, i64, i64, String)> {
    let mut wanted: Vec<i64> = keys.iter().copied().collect();
    wanted.sort_unstable();
    let l = kv_get(db, "listing").filter(|l| l["keys"] == json!(wanted))?;
    Some((l["status"].as_u64()? as usize, l["offset"].as_i64()?, l["sets"].as_i64()?, l["newest"].as_str()?.to_string()))
}

fn save_listing(db: &Connection, keys: &HashSet<i64>, at: (usize, i64, i64, &str)) -> rusqlite::Result<()> {
    let mut wanted: Vec<i64> = keys.iter().copied().collect();
    wanted.sort_unstable();
    kv_set(db, "listing", &json!({"keys": wanted, "status": at.0, "offset": at.1, "sets": at.2, "newest": at.3}))
}

fn crawl_meta(db: &Connection, keys: &HashSet<i64>, full: bool) -> Result<(), CrawlError> {
    // Rows crawled before the stars column need one full pass to learn their rating.
    let backfill = kv_get(db, "stars_backfilled").is_none_or(|v| !v.as_bool().unwrap_or(false));
    let listed = listed_keys(db)?;
    let stopped = listing_at(db, keys);
    let full = full || backfill || !keys.is_subset(&listed) || stopped.is_some();
    let mut sets = stopped.as_ref().map_or(0, |s| s.2 as usize);
    progress(db, json!({"phase": "check", "sets": sets, "full": full}))?;
    for (index, status) in STATUSES.into_iter().enumerate() {
        if stopped.as_ref().is_some_and(|s| s.0 > index) {
            continue;
        }
        let mark: String = db.query_row("SELECT v FROM kv WHERE k=?", [format!("mark{status}")], |r| r.get(0)).optional()?.unwrap_or_default();
        let stop = if full || mark.is_empty() { None } else {
            chrono::DateTime::parse_from_rfc3339(&mark.replace('Z', "+00:00")).ok().map(|t| (t - chrono::Duration::days(1)).format("%Y-%m-%dT%H:%M:%SZ").to_string())
        };
        let (mut newest, mut offset) = match stopped.as_ref().filter(|s| s.0 == index) {
            Some(s) => (mark.max(s.3.clone()), s.1),
            None => (mark, 0),
        };
        loop {
            let urls: Vec<String> = (0..PAGES).map(|i| format!("{SEARCH}&status={status}&offset={}", offset + 100 * i)).collect();
            let batch: Vec<Result<Vec<u8>, String>> = std::thread::scope(|s| {
                let handles: Vec<_> = urls.iter().map(|u| s.spawn(move || get(u))).collect();
                handles.into_iter().map(|h| h.join().unwrap_or_else(|_| Err("page thread panicked".into()))).collect()
            });
            std::thread::sleep(Duration::from_secs_f64(PAUSE));
            let mut pages: Vec<Vec<Value>> = Vec::new();
            for page in batch {
                let raw = page.map_err(CrawlError::Failed)?;
                if let Ok(Value::Array(p)) = serde_json::from_slice::<Value>(&raw) {
                    if !p.is_empty() {
                        pages.push(p);
                    }
                }
            }
            sets += pages.iter().map(Vec::len).sum::<usize>();
            progress(db, json!({"phase": "check", "sets": sets}))?;
            check_stop()?;
            let done = (pages.len() as i64) < PAGES || pages.iter().any(|p| p.len() < 100);
            if pages.is_empty() {
                break;
            }
            let tx = db.unchecked_transaction()?;
            for s in pages.iter().flatten() {
                for b in s["beatmaps"].as_array().map_or(&[][..], Vec::as_slice) {
                    let cs = b["cs"].as_f64().unwrap_or(0.) as i64;
                    if b["mode_int"].as_i64() != Some(3) || b["convert"].as_bool().unwrap_or(false) || !keys.contains(&cs) {
                        continue;
                    }
                    let notes = b["count_circles"].as_i64().unwrap_or(0) + b["count_sliders"].as_i64().unwrap_or(0);
                    tx.execute("INSERT INTO diffs(bid,set_id,status,keys,md5,notes,length,playcount,updated,artist,title,version,creator,stars) \
                                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(bid) DO UPDATE SET status=excluded.status, md5=excluded.md5, \
                                notes=excluded.notes, length=excluded.length, playcount=excluded.playcount, updated=excluded.updated, stars=excluded.stars, \
                                error=CASE WHEN diffs.md5 IS excluded.md5 THEN diffs.error END",
                               params![b["id"].as_i64(), s["id"].as_i64(), b["ranked"].as_i64().or_else(|| s["ranked"].as_i64()), cs, b["checksum"].as_str(), notes,
                                       b["hit_length"].as_f64().unwrap_or(0.), b["playcount"].as_i64().unwrap_or(0),
                                       b["last_updated"].as_str().or_else(|| s["last_updated"].as_str()), s["artist"].as_str(), s["title"].as_str(), b["version"].as_str(),
                                       s["creator"].as_str(), b["difficulty_rating"].as_f64()])?;
                }
            }
            tx.commit()?;
            newest = newest.max(pages[0][0]["last_updated"].as_str().unwrap_or("").to_string());
            let oldest = pages.last().and_then(|p| p.last()).and_then(|x| x["last_updated"].as_str()).unwrap_or("");
            if done || stop.as_deref().is_some_and(|s| oldest < s) {
                break;
            }
            offset += 100 * PAGES;
            if full {
                save_listing(db, keys, (index, offset, sets as i64, &newest))?;
            }
        }
        db.execute("INSERT OR REPLACE INTO kv VALUES(?,?)", params![format!("mark{status}"), newest])?;
        if full {
            save_listing(db, keys, (index + 1, 0, sets as i64, ""))?;
        }
    }
    db.execute("DELETE FROM kv WHERE k='listing'", [])?;
    if backfill {
        kv_set(db, "stars_backfilled", &json!(true))?;
    }
    let mut all: Vec<i64> = keys.union(&listed).copied().collect();
    all.sort_unstable();
    kv_set(db, "keys", &json!(all))?;
    kv_set(db, "checked", &json!(now()))?;
    Ok(())
}

/// Chart text once per revision, most-played first; empty body = unavailable.
fn fetch(db: &Connection, keys: &HashSet<i64>, limit: usize, stars: Option<(f64, f64)>) -> Result<usize, CrawlError> {
    let mut q = db.prepare("SELECT bid,md5,keys,notes,length,stars FROM diffs WHERE error IS NULL AND (sha IS NULL OR fetched IS NOT md5) ORDER BY playcount DESC")?;
    let rows: Vec<(i64, String)> = q.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, Option<String>>(1)?.unwrap_or_default(), r.get::<_, i64>(2)?, r.get::<_, i64>(3)?, r.get::<_, f64>(4)?, r.get::<_, Option<f64>>(5)?)))?
        .filter_map(Result::ok).filter(|r| keys.contains(&r.2) && reachable(r.3, r.4, r.2) && in_stars(r.5, stars)).take(limit).map(|r| (r.0, r.1)).collect();
    let (t0, done) = (std::time::Instant::now(), kv_get(db, "progress").and_then(|p| p["fetched"].as_u64()).unwrap_or(0) as usize);
    let total = rows.len();
    let (send, receive) = std::sync::mpsc::sync_channel::<(i64, Vec<u8>)>(1); // one ahead of the store loop, ~60/min
    std::thread::scope(|s| -> Result<(), CrawlError> {
        let rows = &rows;
        s.spawn(move || {
            for (bid, md5) in rows {
                // Dump copies for ranked and loved charts: no download for what is already here.
                let mut data = crate::paths::frozen_chart(*bid, Some(md5)).and_then(|p| std::fs::read(p).ok()).unwrap_or_default();
                for (url, pause) in FILES {
                    if !data.is_empty() {
                        break;
                    }
                    data = get(&format!("{url}{bid}")).unwrap_or_default();
                    std::thread::sleep(Duration::from_secs_f64(pause));
                }
                if send.send((*bid, data)).is_err() {
                    return; // a pause drops the queued downloads, not just stops reading them
                }
            }
        });
        for (i, (bid, data)) in receive.iter().enumerate() {
            store(db, bid, &data)?;
            progress(db, json!({"phase": "download", "fetched": done + i + 1}))?;
            if i >= 20 && i % 20 == 0 {
                kv_set(db, "sec_fetch", &json!(t0.elapsed().as_secs_f64() / (i + 1) as f64))?;
            }
            check_stop()?;
        }
        Ok(())
    })?;
    Ok(total)
}

fn store(db: &Connection, bid: i64, data: &[u8]) -> Result<(), CrawlError> {
    if data.iter().all(u8::is_ascii_whitespace) {
        db.execute("UPDATE diffs SET error='unavailable', fetched=md5 WHERE bid=?", [bid])?;
        return Ok(());
    }
    let sha: String = Sha256::digest(data).iter().map(|b| format!("{b:02x}")).collect();
    std::fs::write(web_dir().join("osu").join(format!("{sha}.osu")), data)?;
    db.execute("UPDATE diffs SET sha=?, file_md5=?, fetched=md5 WHERE bid=?", params![sha, format!("{:x}", md5::compute(data)), bid])?;
    Ok(())
}

/// A state.db feats cell: {rate: features} written by this program or the Python crawler; Err = its analysis failed.
fn read_cell(data: &[u8]) -> Result<Vec<(f64, Dict)>, String> {
    let rates = |items: Vec<(f64, Val)>| -> Vec<(f64, Dict)> {
        items.into_iter().filter_map(|(r, v)| match v { Val::Dict(d) => Some((r, d)), _ => None }).collect()
    };
    if data.first() == Some(&0x80) {
        let p = pickle::load(data)?;
        if p.get("error").is_some() {
            return Err("analysis failed".into());
        }
        return Ok(rates(p.items().iter().filter_map(|(r, f)| Some((r.f()?, value(f)))).collect()));
    }
    let Val::Dict(d) = codec::decode(data)? else { return Err("bad cell".into()) };
    if d.has("error") {
        return Err("analysis failed".into());
    }
    Ok(rates(d.0.into_iter().filter_map(|(k, v)| match k {
        ms_calc::val::Key::S(s) => Some((s.parse().ok()?, v)),
        ms_calc::val::Key::F(r) => Some((r, v)),
        ms_calc::val::Key::B(_) => None,
    }).collect()))
}

fn write_cell(fs: &[(f64, Dict)]) -> Vec<u8> {
    let mut d = Dict::new();
    fs.iter().for_each(|(r, f)| d.setf(*r, Val::Dict(f.clone())));
    codec::encode(&Val::Dict(d))
}

fn analyse(db: &Connection, calc: &str, prm: &Params, threads: Option<usize>) -> Result<(), CrawlError> {
    let todo: Vec<String> = db.prepare("SELECT DISTINCT sha FROM diffs WHERE sha IS NOT NULL AND sha NOT IN (SELECT sha FROM feats WHERE calc=?)")?
        .query_map([calc], |r| r.get(0))?.collect::<rusqlite::Result<_>>()?;
    let (t0, done) = (std::time::Instant::now(), kv_get(db, "progress").and_then(|p| p["analysed"].as_u64()).unwrap_or(0) as usize);
    for (n, chunk) in todo.chunks(64).enumerate() { // a pause stops between batches
        let jobs: Vec<(Vec<u8>, Vec<f64>)> = chunk.iter().map(|sha| (std::fs::read(web_dir().join("osu").join(format!("{sha}.osu"))).unwrap_or_default(), RATES.to_vec())).collect();
        let results = features_batch(&jobs, workers(threads), prm);
        let tx = db.unchecked_transaction()?;
        for (sha, fs) in chunk.iter().zip(results) {
            let cell = match fs {
                Ok(fs) => write_cell(&fs),
                Err(e) => {
                    let mut d = Dict::new();
                    d.set("error", e);
                    codec::encode(&Val::Dict(d))
                }
            };
            tx.execute("INSERT OR REPLACE INTO feats VALUES(?,?,?)", params![sha, calc, cell])?;
        }
        tx.commit()?;
        let i = n * 64 + chunk.len();
        progress(db, json!({"phase": "analyse", "analysed": done + i}))?;
        kv_set(db, "sec_analyse", &json!(t0.elapsed().as_secs_f64() / i as f64))?;
        check_stop()?;
    }
    Ok(())
}

// What NPS/Skills predictions, eligibility and practice matching read from a website chart. Full dicts cost ~90 KB per
// chart in memory.
const NPS_FIELDS: [&str; 13] = ["active", "burst", "chord", "drill", "family", "nps", "overlap", "play_span", "repeat", "rhythm", "version", "vibro_focused", "wide"];

fn compact(f: &Dict) -> Dict {
    let mut out = Dict::new();
    for k in ["keys", "overall", "od", "ln", "notes", "length", "stam", "description", "sk", "stars"] {
        out.set(k, f.get(k).cloned().unwrap_or(Val::None));
    }
    let mut endurance = Dict::new();
    endurance.set("load", f.dict("endurance").map_or(0., |e| e.fget("load", 0.)));
    out.set("endurance", endurance);
    let mut execution = Dict::new();
    execution.set("residual_vector", f.dict("execution").and_then(|e| e.get("residual_vector")).cloned().unwrap_or(Val::None));
    out.set("execution", execution);
    out.set("calibration_features", f.get("calibration_features").cloned().unwrap_or(Val::List(Vec::new())));
    let mut nps = Dict::new();
    if let Some(st) = f.dict("nps") {
        for k in NPS_FIELDS {
            if let Some(v) = st.get(k) {
                nps.set(k, v.clone());
            }
        }
    }
    out.set("nps", nps);
    out.set("tech_profile", f.get("tech_profile").cloned().unwrap_or_else(|| Val::Dict(Dict::new())));
    out
}

#[derive(Clone, Serialize, Deserialize)]
pub struct WebMap {
    pub sha: String,
    pub bid: i64,
    pub set_id: i64,
    pub status: Option<i64>,
    pub keys: i64,
    pub file_md5: String,
    pub playcount: i64,
    pub artist: String,
    pub title: String,
    pub version: String,
    pub creator: String,
    pub overall: Vec<(f64, f64)>,
}

/// Stored website features moved to the calculator `to`, web.msp with them: a calculator change would else crawl everything again.
pub fn carry(from: &str, to: &str, before: &Params, after: &Params) -> Result<(), String> {
    let text = |e: rusqlite::Error| e.to_string();
    if !web_dir().join("state.db").is_file() {
        return Ok(());
    }
    let db = state_db().map_err(text)?;
    let todo: Vec<String> = db.prepare("SELECT sha FROM feats WHERE calc=? AND sha NOT IN (SELECT sha FROM feats WHERE calc=?)").map_err(text)?
        .query_map([from, to], |r| r.get(0)).map_err(text)?.collect::<rusqlite::Result<_>>().map_err(text)?;
    for chunk in todo.chunks(500) {
        let tx = db.unchecked_transaction().map_err(text)?;
        for sha in chunk {
            let data: Vec<u8> = tx.query_row("SELECT data FROM feats WHERE sha=? AND calc=?", [sha.as_str(), from], |r| r.get(0)).map_err(text)?;
            let fs = read_cell(&data).unwrap_or_default();
            let moved: Vec<Option<Dict>> = fs.iter().map(|(_, f)| ms_calc::features::recalibrated(f, before, after)).collect();
            let cell = if moved.iter().any(Option::is_some) {
                write_cell(&fs.iter().zip(moved).map(|((r, f), m)| (*r, m.unwrap_or_else(|| f.clone()))).collect::<Vec<_>>())
            } else {
                data
            };
            tx.execute("INSERT OR REPLACE INTO feats VALUES(?,?,?)", params![sha, to, cell]).map_err(text)?;
        }
        tx.commit().map_err(text)?;
    }
    if !todo.is_empty() {
        let stars = kv_get(&db, "stars").and_then(|v| Some((v[0].as_f64()?, v[1].as_f64()?)));
        publish(&db, to, stars).map_err(|e| format!("{e:?}"))?;
    }
    Ok(())
}

#[derive(Serialize, Deserialize)]
struct WebMeta {
    calc: String,
    built: f64,
    maps: Vec<WebMap>,
}

fn publish(db: &Connection, calc: &str, stars: Option<(f64, f64)>) -> Result<(), CrawlError> {
    let mut q = db.prepare("SELECT d.sha,d.bid,d.set_id,d.status,d.keys,d.file_md5,d.playcount,d.artist,d.title,d.version,d.creator,d.stars,f.data \
                            FROM diffs d JOIN feats f ON f.sha=d.sha AND f.calc=? WHERE d.error IS NULL")?;
    let mut rows = q.query([calc])?;
    let text = |r: &rusqlite::Row<'_>, i: usize| r.get::<_, Option<String>>(i).map(Option::unwrap_or_default);
    let (mut maps, mut records, mut seen) = (Vec::new(), Vec::new(), HashSet::new());
    while let Some(r) = rows.next()? {
        let sha = text(r, 0)?;
        if !in_stars(r.get(11)?, stars) || !seen.insert(sha.clone()) {
            continue;
        }
        let Ok(fs) = read_cell(&r.get::<_, Vec<u8>>(12)?) else { continue };
        let at = maps.len() as i64;
        maps.push(WebMap { sha, bid: r.get(1)?, set_id: r.get::<_, Option<i64>>(2)?.unwrap_or(0), status: r.get(3)?, keys: r.get(4)?, file_md5: text(r, 5)?,
                           playcount: r.get::<_, Option<i64>>(6)?.unwrap_or(0), artist: text(r, 7)?, title: text(r, 8)?, version: text(r, 9)?, creator: text(r, 10)?,
                           overall: fs.iter().map(|(rate, f)| (*rate, f.f("overall"))).collect() });
        let mut sorted: Vec<_> = fs.iter().map(|(rate, f)| (at, *rate, codec::encode(&Val::Dict(compact(f))))).collect();
        sorted.sort_by(|a, b| a.1.total_cmp(&b.1));
        records.extend(sorted);
    }
    write_file(&web_dir().join("web.msp"), &WebMeta { calc: calc.to_string(), built: now(), maps }, &records)?;
    progress(db, json!({"ready": seen.len()}))?;
    Ok(())
}

fn write_file(path: &Path, meta: &WebMeta, records: &[(i64, f64, Vec<u8>)]) -> Result<(), String> {
    blobs::write(path, MAGIC, &serde_json::to_vec(meta).map_err(|e| e.to_string())?, records)
}

/// web.pkl of a Python install → web.msp, so an existing crawl keeps working.
pub fn convert_pickle(from: &Path, to: &Path) -> Result<(), String> {
    let root = pickle::load(&std::fs::read(from).map_err(|e| e.to_string())?)?;
    if root.get("format").and_then(P::i) != Some(3) {
        return Err("web.pkl has an old format".into());
    }
    let feats: HashMap<&str, &P> = root.get("feats").map_or(&[][..], P::items).iter().filter_map(|(k, v)| Some((k.s()?, v))).collect();
    let (mut maps, mut records) = (Vec::new(), Vec::new());
    for (sha, m) in root.get("maps").map_or(&[][..], P::items) {
        let Some(sha) = sha.s() else { continue };
        let num = |k: &str| m.get(k).and_then(P::i);
        let text = |k: &str| m.get(k).and_then(P::s).unwrap_or("").to_string();
        let at = maps.len() as i64;
        let mut rates: Vec<(f64, Vec<u8>)> = Vec::new();
        for (rate, blob) in feats.get(sha).map_or(&[][..], |f| f.items()) {
            if let (Some(rate), P::Bytes(b)) = (rate.f(), blob) {
                rates.push((rate, codec::encode(&value(&pickle::load(b)?))));
            }
        }
        rates.sort_by(|a, b| a.0.total_cmp(&b.0));
        records.extend(rates.into_iter().map(|(rate, blob)| (at, rate, blob)));
        maps.push(WebMap { sha: sha.to_string(), bid: num("bid").unwrap_or(0), set_id: num("set_id").unwrap_or(0), status: num("status"), keys: num("keys").unwrap_or(0),
                           file_md5: text("file_md5"), playcount: num("playcount").unwrap_or(0), artist: text("artist"), title: text("title"), version: text("version"),
                           creator: text("creator"), overall: m.get("overall").map_or(&[][..], P::items).iter().filter_map(|(r, o)| Some((r.f()?, o.f()?))).collect() });
    }
    let meta = WebMeta { calc: root.get("calc").and_then(P::s).unwrap_or("").to_string(), built: root.get("built").and_then(P::f).unwrap_or(0.), maps };
    write_file(to, &meta, &records)
}

pub struct Web {
    pub maps: Vec<WebMap>,
    pub calc: String,
    blobs: Blobs,
}

impl Web {
    pub fn open(path: &Path) -> Result<Web, String> {
        let (blobs, header) = Blobs::open(path, MAGIC)?;
        let meta: WebMeta = serde_json::from_slice(blobs.header(header)).map_err(|e| e.to_string())?;
        Ok(Web { maps: meta.maps, calc: meta.calc, blobs })
    }

    pub fn feature(&self, map: usize, rate: f64) -> Option<Val> {
        self.blobs.feature(map as i64, rate)
    }

    /// The rates to analyse a website chart from: those whose rating can land near the player's target. Two nearest
    /// the window's centre anchor the exact-rate refinement: all in-window rates were ~190k predictions after every
    /// score and the GUI lagged for minutes (G835LX 2026-10-02). None = outside every window.
    pub fn window_rates(&self, map: usize, reach: Option<(f64, f64)>) -> Option<Vec<f64>> {
        window_rates(&self.maps[map].overall, reach)
    }
}

fn window_rates(ov: &[(f64, f64)], reach: Option<(f64, f64)>) -> Option<Vec<f64>> {
    let (lo, hi) = reach.unwrap_or((0., f64::INFINITY));
    let mut rates: Vec<(f64, f64)> = ov.iter().copied().filter(|x| lo <= x.1 && x.1 <= hi).collect();
    if rates.is_empty() {
        // The window falls between two rated rates: both anchor an exact refinement.
        let below = ov.iter().copied().filter(|x| x.1 < lo).max_by(|a, b| a.1.total_cmp(&b.1));
        let above = ov.iter().copied().filter(|x| x.1 > hi).min_by(|a, b| a.1.total_cmp(&b.1));
        rates = vec![below?, above?];
    }
    if reach.is_some() && rates.len() > 2 {
        let mid = (lo * hi).ln() / 2.;
        rates.sort_by(|a, b| (a.1.ln() - mid).abs().total_cmp(&(b.1.ln() - mid).abs()));
        rates.truncate(2);
    }
    Some(rates.into_iter().map(|x| x.0).collect())
}

/// Everything a crawl does: listing, chart text, analysis, publication. `check` stops after the listing, so the app
/// can show counts, size and time before Start.
pub fn crawl(keys: &HashSet<i64>, stars: Option<(f64, f64)>, check: bool, full: bool, threads: Option<usize>, calc: &str, prm: &Params) -> Result<(), CrawlError> {
    let db = state_db()?;
    let Some(_held) = lock() else { return Err(CrawlError::Failed("another crawl is running".into())) };
    let _ = std::fs::remove_file(web_dir().join("stop"));
    let mut wanted: Vec<i64> = keys.iter().copied().collect();
    wanted.sort_unstable();
    kv_set(&db, "want_keys", &json!(wanted))?; // the app's window reopens with these
    kv_set(&db, "command", &json!(if check { "check" } else { "crawl" }))?;
    db.execute("DELETE FROM kv WHERE k='reopen'", [])?;
    if let Some((lo, hi)) = stars {
        kv_set(&db, "stars", &json!([lo, hi]))?; // remembered: later runs and the app keep the chosen range
    }
    let stars = kv_get(&db, "stars").and_then(|v| Some((v[0].as_f64()?, v[1].as_f64()?)));
    let run = || -> Result<(), CrawlError> {
        crawl_meta(&db, keys, full)?;
        if check {
            progress(&db, json!({"phase": "checked"}))?;
            return Ok(());
        }
        let plan = estimate(&db, keys, stars, calc)?;
        progress(&db, json!({"phase": "download", "fetched": 0, "fetch_total": plan.fetch, "analysed": 0, "analyse_total": plan.analyse, "error": null}))?;
        // Publish every chunk; small first chunks so a new player sees website maps within minutes, not after 5,000
        // downloads (fresh-install run 2026-10-02).
        for size in [300, 1500].into_iter().chain(std::iter::repeat(5000)) {
            let more = fetch(&db, keys, size, stars)?;
            analyse(&db, calc, prm, threads)?;
            publish(&db, calc, stars)?;
            if more == 0 {
                break;
            }
        }
        progress(&db, json!({"phase": "done"}))?;
        Ok(())
    };
    match run() {
        Err(CrawlError::Stopped) => progress(&db, json!({"phase": "paused"})).map_err(Into::into),
        Err(CrawlError::Failed(e)) => {
            progress(&db, json!({"phase": "failed", "error": e}))?;
            Err(CrawlError::Failed(e))
        }
        Ok(()) => Ok(()),
    }
}

/// {key: value} of the crawl's saved state, for the app's window.
pub fn state(db: &Connection) -> BTreeMap<String, Value> {
    ["progress", "want_keys", "stars", "checked", "reopen", "command"].iter().filter_map(|k| Some((k.to_string(), kv_get(db, k)?))).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_carries_what_the_app_needs_to_resume_a_paused_batch() {
        let db = Connection::open_in_memory().unwrap();
        db.execute("CREATE TABLE kv(k TEXT PRIMARY KEY, v TEXT)", []).unwrap();
        kv_set(&db, "reopen", &json!(true)).unwrap();
        kv_set(&db, "command", &json!("crawl")).unwrap();
        let s = state(&db);
        assert_eq!(s.get("reopen"), Some(&json!(true)));
        assert_eq!(s.get("command"), Some(&json!("crawl")));
    }

    #[test]
    fn rates_out_of_reach_are_never_loaded_and_two_anchor_a_window() {
        let ov = [(1., 6.), (1.2, 9.)];
        assert_eq!(window_rates(&ov, Some((5., 7.))), Some(vec![1.]));
        assert_eq!(window_rates(&ov, Some((10., 12.))), None);
        assert_eq!(window_rates(&ov, Some((7., 8.))), Some(vec![1., 1.2]), "a window between two rated rates keeps both");
        let many = [(0.8, 4.), (1., 6.), (1.2, 9.), (1.4, 11.)];
        let mut near = window_rates(&many, Some((3.5, 12.))).unwrap();
        near.sort_by(f64::total_cmp);
        assert_eq!(near, [1., 1.2], "only the two nearest the centre are predicted");
        assert_eq!(window_rates(&many, None).unwrap().len(), 4);
    }

    #[test]
    fn a_stopped_listing_continues_only_for_the_same_keymodes() {
        let db = Connection::open_in_memory().unwrap();
        db.execute_batch("CREATE TABLE kv(k TEXT PRIMARY KEY, v TEXT);").unwrap();
        let (ten, seven): (HashSet<i64>, HashSet<i64>) = ([10].into(), [7, 10].into());
        assert_eq!(listing_at(&db, &ten), None);
        save_listing(&db, &ten, (3, 4800, 52800, "2026-11-01T08:37:21Z")).unwrap();
        assert_eq!(listing_at(&db, &ten), Some((3, 4800, 52800, "2026-11-01T08:37:21Z".to_string())));
        assert_eq!(listing_at(&db, &seven), None, "other keymodes read the listing for themselves");
    }

    #[test]
    fn the_star_filter_keeps_unrated_charts() {
        let range = Some((6., 12.));
        assert!(in_stars(Some(7.), range) && in_stars(None, range) && in_stars(Some(2.), None));
        assert!(!in_stars(Some(5.9), range) && !in_stars(Some(12.1), range));
    }

    #[test]
    fn the_estimate_counts_what_a_crawl_still_has_to_do() {
        let db = Connection::open_in_memory().unwrap();
        db.execute_batch("CREATE TABLE kv(k TEXT PRIMARY KEY, v TEXT); CREATE TABLE feats(sha TEXT, calc TEXT);
                          CREATE TABLE diffs(bid INTEGER PRIMARY KEY, keys INTEGER, md5 TEXT, notes INTEGER, length REAL, playcount INTEGER, stars REAL, sha TEXT, fetched TEXT, error TEXT);").unwrap();
        for bid in 1..5 {
            db.execute("INSERT INTO diffs(bid,keys,md5,notes,length,playcount,stars) VALUES(?,7,?,2000,120,?,?)", rusqlite::params![bid, format!("m{bid}"), bid, if bid < 4 { 8. } else { 3. }]).unwrap();
        }
        let keys = |k: &[i64]| k.iter().copied().collect::<HashSet<i64>>();
        assert!(estimate(&db, &keys(&[7]), Some((6., 12.)), "c").unwrap().unlisted.is_empty(), "older crawls: their rows' keymodes");
        kv_set(&db, "keys", &json!([7])).unwrap();
        let e = estimate(&db, &keys(&[4, 7]), Some((6., 12.)), "c").unwrap();
        assert_eq!((e.fetch, e.analyse, e.ready, e.unlisted.clone()), (3, 3, 0, vec![4]));
        assert_eq!(e.new_bytes, 3 * CHART_BYTES);
        assert!(e.seconds > 0.);
        assert_eq!(estimate(&db, &keys(&[4]), Some((6., 12.)), "c").unwrap().fetch, 0);
        db.execute("UPDATE diffs SET sha='s1', fetched='m1' WHERE bid=1", []).unwrap();
        assert_eq!(estimate(&db, &keys(&[7]), Some((6., 12.)), "c").unwrap().fetch, 2, "a resume fetches only the rest");
        db.execute("INSERT INTO feats VALUES('s1','c')", []).unwrap();
        assert_eq!(estimate(&db, &keys(&[7]), Some((6., 12.)), "c").unwrap().ready, 1);
    }
}
