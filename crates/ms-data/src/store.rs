//! rec.db: schema, key-value, event log, tracking pauses, scores (recdata.py's store half).
use crate::pyjson::dumps;
use chrono::{DateTime, Local, NaiveDateTime, TimeZone, Utc};
use rusqlite::{params, params_from_iter, types::Value as Sql, Connection, OptionalExtension, Row};
use serde_json::{json, Map, Value};
use std::path::Path;

pub const SUPPORTED_KEYS: std::ops::RangeInclusive<i64> = 4..=10;
pub const RANKED_STATUS: [i64; 2] = [1, 2]; // ranked, approved; 4 = loved (warmup only)
const RANKED_MODS: [&str; 13] = ["NF", "EZ", "HD", "FI", "FL", "MR", "SD", "PF", "HT", "DC", "DT", "NC", "CL"];

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS scores(key TEXT PRIMARY KEY, src TEXT, local_id TEXT, online_id INTEGER,
  player_id INTEGER, sha256 TEXT, md5 TEXT, beatmap_id INTEGER, status INTEGER, played TEXT, imported TEXT,
  client TEXT, mods TEXT, rate REAL, ranked INTEGER, stats TEXT, acc REAL, n INTEGER, total INTEGER,
  pp REAL, fresh INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, t REAL, kind TEXT, beatmap TEXT, info TEXT);
CREATE TABLE IF NOT EXISTS kv(k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS tracking_pauses(start REAL PRIMARY KEY, end REAL);
CREATE TABLE IF NOT EXISTS ignored_scores(key TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS converts(source_sha TEXT, target INTEGER, converter TEXT, source_keys INTEGER, sha TEXT, md5 TEXT,
  artist TEXT, title TEXT, version TEXT, creator TEXT, set_id INTEGER, bid INTEGER, od REAL, length REAL, playcount INTEGER,
  web INTEGER, PRIMARY KEY(source_sha, target, converter));
CREATE INDEX IF NOT EXISTS converts_sha ON converts(sha);
CREATE TABLE IF NOT EXISTS feats(key TEXT PRIMARY KEY, data TEXT);
CREATE TABLE IF NOT EXISTS installed(sha256 TEXT PRIMARY KEY, beatmap_id INTEGER, set_id INTEGER, md5 TEXT,
  status INTEGER, keys INTEGER, od REAL, length REAL, title TEXT, artist TEXT, version TEXT);
CREATE TABLE IF NOT EXISTS installed_local(sha256 TEXT PRIMARY KEY, online_md5 TEXT, audio_sha TEXT,
  audio_required INTEGER, creator TEXT);
CREATE INDEX IF NOT EXISTS scores_bm ON scores(beatmap_id);
CREATE INDEX IF NOT EXISTS scores_sha ON scores(sha256);
CREATE INDEX IF NOT EXISTS scores_local ON scores(local_id);
CREATE INDEX IF NOT EXISTS installed_bid ON installed(beatmap_id);
CREATE INDEX IF NOT EXISTS events_t ON events(t);
";

pub type Db = Connection;

/// Stored chart features moved to the calculator `to`, so a calibration change doesn't re-analyse the library. Charts it doesn't
/// move keep their bytes. Rows are copied, as every older calculator's were left; a stopped run resumes.
pub fn carry_feats(db: &Db, from: &str, to: &str, before: &ms_calc::params::Params, after: &ms_calc::params::Params, flow: &dyn Fn(&str, f64) -> Option<ms_calc::flow::Flow>) -> Result<usize> {
    let there: std::collections::HashSet<String> = db.prepare("SELECT key FROM feats WHERE key LIKE ?")?.query_map([format!("%|{to}")], |r| r.get(0))?.collect::<Result<_>>()?;
    let all: Vec<String> = db.prepare("SELECT key FROM feats WHERE key LIKE ? ORDER BY key")?.query_map([format!("%|{from}")], |r| r.get(0))?.collect::<Result<_>>()?;
    // A launch during a crawl repeats the carry, and a second source calculator repeats most rows: only the missing cost anything.
    let keys: Vec<String> = all.into_iter().filter(|k| !there.contains(&format!("{}|{to}", k.rsplit_once('|').map_or("", |n| n.0)))).collect();
    for chunk in keys.chunks(500) {
        let tx = db.unchecked_transaction()?;
        for key in chunk {
            let data: Sql = tx.query_row("SELECT data FROM feats WHERE key=?", [key], |r| r.get(0))?;
            let f = match &data { // rows written before the binary form are JSON text
                Sql::Blob(b) => crate::codec::decode(b).ok(),
                Sql::Text(t) => serde_json::from_str(t).ok().map(|v| crate::codec::from_json(&v)),
                _ => None,
            };
            let name = key.rsplit_once('|').map_or("", |k| k.0);
            let (sha, rate) = name.rsplit_once('@').map_or((name, 1.), |(s, r)| (s, r.parse().unwrap_or(1.)));
            let moved = match f {
                Some(ms_calc::val::Val::Dict(f)) => ms_calc::features::recalibrated(&f, before, after, rate, &|| flow(sha, rate)),
                _ => None,
            };
            let data = moved.map_or(data, |f| Sql::Blob(crate::codec::encode(&ms_calc::val::Val::Dict(f))));
            tx.execute("INSERT OR IGNORE INTO feats VALUES (?,?)", params![format!("{name}|{to}"), data])?;
        }
        tx.commit()?;
    }
    Ok(keys.len())
}

/// Rows of every other calculator go and the file shrinks: they are only ever read to carry them, and one calculator costs
/// about 1 GB (G835LX held eight). Nothing without rows of `calc`, so a failed carry can't cost the analyses. A VACUUM that
/// other connections block leaves the deleted rows' pages free and the next call finishes it.
pub fn prune_feats(db: &Db, calc: &str) -> Result<Option<usize>> {
    let pattern = format!("%|{calc}");
    if db.query_row("SELECT 1 FROM feats WHERE key LIKE ? LIMIT 1", [&pattern], |_| Ok(())).optional()?.is_none() {
        return Ok(None);
    }
    let gone = db.execute("DELETE FROM feats WHERE key NOT LIKE ?", [&pattern])?;
    if gone == 0 {
        return Ok(Some(0));
    }
    db.execute_batch("VACUUM")?;
    db.query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |_| Ok(()))?; // VACUUM writes the whole file through the WAL
    Ok(Some(gone))
}

pub type Result<T> = rusqlite::Result<T>;

/// The worker, NPS builder, feature fill and import each hold a connection. With a rollback journal every
/// feature-batch commit made the selected card's reads wait (WAL: readers never wait for a writer).
pub fn connect(path: &Path) -> Result<Db> {
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let db = Connection::open(path)?;
    db.busy_timeout(std::time::Duration::from_secs(30))?;
    db.query_row("PRAGMA journal_mode=WAL", [], |_| Ok(()))?;
    db.execute_batch(SCHEMA)?;
    Ok(db)
}

pub fn kv_get(db: &Db, k: &str) -> Option<Value> {
    let text: Option<String> = db.query_row("SELECT v FROM kv WHERE k=?", [k], |r| r.get(0)).optional().ok().flatten();
    text.and_then(|t| serde_json::from_str(&t).ok())
}

pub fn kv_set(db: &Db, k: &str, v: &Value) -> Result<()> {
    db.execute("INSERT OR REPLACE INTO kv VALUES (?, ?)", params![k, dumps(v)]).map(|_| ())
}

pub fn now() -> f64 {
    match f64::from_bits(PINNED.load(std::sync::atomic::Ordering::Relaxed)) {
        0. => Utc::now().timestamp_micros() as f64 / 1e6,
        t => t,
    }
}

static PINNED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Stops the clock, so two benchmark runs compare predictions and lists.
pub fn pin_clock(t: f64) {
    PINNED.store(t.to_bits(), std::sync::atomic::Ordering::Relaxed);
}

/// time.strftime("%FT%T"): local wall clock.
pub fn local_stamp() -> String {
    Local::now().format("%Y-%m-%dT%H:%M:%S").to_string()
}

/// time.strftime of a timestamp, local wall clock.
pub fn local_format(t: f64, format: &str) -> String {
    Local.timestamp_opt(t as i64, 0).single().map_or_else(String::new, |d| d.format(format).to_string())
}

/// An event's info as Python's `json.loads` read it, except that NaN and infinities (which its `json.dumps` wrote) become null.
pub fn event_info(text: &str) -> Value {
    serde_json::from_str(text).or_else(|_| serde_json::from_str(&finite_json(text))).unwrap_or_else(|_: serde_json::Error| json!({}))
}

fn finite_json(text: &str) -> String {
    let (mut out, mut quoted, mut escaped) = (String::with_capacity(text.len()), false, false);
    let mut rest = text;
    while let Some(c) = rest.chars().next() {
        if !quoted {
            if let Some(token) = ["-Infinity", "Infinity", "NaN"].iter().find(|t| rest.starts_with(**t)) {
                out.push_str("null");
                rest = &rest[token.len()..];
                continue;
            }
        }
        (quoted, escaped) = match (quoted, escaped, c) {
            (true, true, _) => (true, false),
            (true, false, '\\') => (true, true),
            (_, _, '"') => (!quoted, false),
            _ => (quoted, false),
        };
        out.push(c);
        rest = &rest[c.len_utf8()..];
    }
    out
}

pub fn log_event(db: &Db, kind: &str, beatmap: Option<&str>, t: Option<f64>, info: &Value) -> Result<i64> {
    db.execute("INSERT INTO events(t, kind, beatmap, info) VALUES (?, ?, ?, ?)",
               params![t.unwrap_or_else(now), kind, beatmap, dumps(info)])?;
    Ok(db.last_insert_rowid())
}

/// Explicit player feedback beside the recorded attempt, never an automatic guess that an abort means failure.
/// A matching existing annotation is idempotent.
pub fn annotate_attempt(db: &Db, start_id: i64, reason: &str, note: &str, confidence: f64) -> std::result::Result<i64, String> {
    if !["failed", "interrupted", "experiment", "bored"].contains(&reason) {
        return Err("unsupported attempt reason".into());
    }
    if !confidence.is_finite() || !(0. ..=1.).contains(&confidence) {
        return Err("confidence must be between 0 and 1".into());
    }
    let start: Option<Option<String>> = db.query_row("SELECT beatmap FROM events WHERE id=? AND kind='start'", [start_id], |r| r.get(0))
        .optional().map_err(|e| e.to_string())?;
    let Some(beatmap) = start else { return Err("start event not found".into()) };
    let note: String = note.chars().take(1000).collect();
    let payload = json!({"start_id": start_id, "reason": reason, "note": note, "confidence": confidence, "source": "player"});
    let previous: Option<(i64, String)> = db.query_row(
        "SELECT id,info FROM events WHERE kind='attempt_note' AND json_extract(info,'$.start_id')=? ORDER BY id DESC LIMIT 1",
        [start_id], |r| Ok((r.get(0)?, r.get(1)?))).optional().map_err(|e| e.to_string())?;
    if let Some((id, info)) = previous {
        if serde_json::from_str::<Value>(&info).ok().as_ref() == Some(&payload) {
            return Ok(id);
        }
    }
    log_event(db, "attempt_note", beatmap.as_deref(), None, &payload).map_err(|e| e.to_string())
}

/// datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp(): no offset means local time.
pub fn timestamp(iso: &str) -> Option<f64> {
    let text = iso.replace('Z', "+00:00");
    if let Ok(t) = DateTime::parse_from_rfc3339(&text) {
        return Some(t.timestamp_micros() as f64 / 1e6);
    }
    for format in ["%Y-%m-%dT%H:%M:%S%.f%:z", "%Y-%m-%d %H:%M:%S%.f%:z"] {
        if let Ok(t) = DateTime::parse_from_str(&text, format) {
            return Some(t.timestamp_micros() as f64 / 1e6);
        }
    }
    let naive = ["%Y-%m-%dT%H:%M:%S%.f", "%Y-%m-%d %H:%M:%S%.f", "%Y-%m-%dT%H:%M"].iter()
        .find_map(|f| NaiveDateTime::parse_from_str(&text, f).ok())
        .or_else(|| chrono::NaiveDate::parse_from_str(&text, "%Y-%m-%d").ok().and_then(|d| d.and_hms_opt(0, 0, 0)))?;
    Local.from_local_datetime(&naive).earliest().map(|t| t.timestamp_micros() as f64 / 1e6)
}

pub fn tracking_enabled(db: &Db) -> Result<bool> {
    Ok(db.query_row("SELECT 1 FROM tracking_pauses WHERE end IS NULL", [], |_| Ok(())).optional()?.is_none())
}

/// Pausing halfway through a play excludes that whole attempt.
pub fn set_tracking(db: &Db, enabled: bool, t: Option<f64>, start: Option<f64>) -> Result<()> {
    let t = t.unwrap_or_else(now);
    if enabled {
        db.execute("UPDATE tracking_pauses SET end=? WHERE end IS NULL", [t])?;
    } else if tracking_enabled(db)? {
        db.execute("INSERT INTO tracking_pauses VALUES (?, NULL)", [start.map_or(t, |s| t.min(s))])?;
    }
    Ok(())
}

pub fn tracking_allowed(db: &Db, end: Option<f64>, start: Option<f64>) -> Result<bool> {
    let Some(end) = end else {
        // Undated imports cannot be proved to fall outside a controller session.
        return Ok(db.query_row("SELECT 1 FROM tracking_pauses LIMIT 1", [], |_| Ok(())).optional()?.is_none());
    };
    Ok(db.query_row("SELECT 1 FROM tracking_pauses WHERE start<=? AND (end IS NULL OR end>?) LIMIT 1",
                    params![end, start.unwrap_or(end)], |_| Ok(())).optional()?.is_none())
}

pub struct Event {
    pub id: i64,
    pub t: f64,
    pub kind: String,
    pub beatmap: Option<String>,
    pub info: Value,
}

/// Events outside tracking pauses, oldest first, in one query rather than a pause lookup per event.
/// fields: keep only these info keys (SQLite JSON), sparing the decode of large start payloads.
pub fn tracked_events(db: &Db, since: f64, kinds: &[&str], until: Option<f64>, fields: Option<&[&str]>) -> Result<Vec<Event>> {
    let info = fields.map_or_else(|| "e.info".to_string(), |fields| {
        let parts: Vec<String> = fields.iter().map(|k| format!("'{k}',json_extract(e.info,'$.{k}')")).collect();
        format!("json_object({}) AS info", parts.join(","))
    });
    let mut query = format!("SELECT e.id,e.t,e.kind,e.beatmap,{info} FROM events e WHERE e.t>=? AND NOT EXISTS (\
                             SELECT 1 FROM tracking_pauses p WHERE p.start<=e.t AND (p.end IS NULL OR p.end>e.t))");
    let mut args = vec![Sql::Real(since)];
    if !kinds.is_empty() {
        query += &format!(" AND e.kind IN ({})", vec!["?"; kinds.len()].join(","));
        args.extend(kinds.iter().map(|k| Sql::Text((*k).to_owned())));
    }
    if let Some(u) = until {
        query += " AND e.t<?";
        args.push(Sql::Real(u));
    }
    query += " ORDER BY e.t,e.id";
    let mut statement = db.prepare(&query)?;
    let rows = statement.query_map(params_from_iter(args), event)?;
    rows.collect()
}

fn event(r: &Row<'_>) -> Result<Event> {
    let info: Option<String> = r.get(4)?;
    Ok(Event { id: r.get(0)?, t: r.get(1)?, kind: r.get(2)?, beatmap: r.get(3)?,
               info: info.and_then(|i| serde_json::from_str(&i).ok()).unwrap_or_else(|| json!({})) })
}

/// lazer mod list → (rate | None, ranked). Custom DT/HT speeds and any other setting are unranked (osu! wiki,
/// Double Time (lazer)); adjust_pitch alone keeps a score ranked.
pub fn variant(mods: &[Value]) -> (Option<f64>, bool) {
    let (mut rate, mut ranked) = (1., true);
    for m in mods {
        let acronym = m.get("acronym").and_then(Value::as_str).unwrap_or("").to_uppercase();
        let empty = Map::new();
        let settings: Vec<(&String, &Value)> = m.get("settings").and_then(Value::as_object).unwrap_or(&empty).iter()
            .filter(|(k, _)| k.as_str() != "adjust_pitch").collect();
        if ["WU", "WD", "AS"].contains(&acronym.as_str()) {
            return (None, false);
        }
        let default = match acronym.as_str() {
            "HT" | "DC" => Some(0.75),
            "DT" | "NC" => Some(1.5),
            _ => None,
        };
        if let Some(default) = default {
            rate = settings.iter().find(|(k, _)| k.as_str() == "speed_change").and_then(|(_, v)| v.as_f64()).unwrap_or(default);
            ranked &= rate == default && settings.iter().all(|(k, _)| k.as_str() == "speed_change");
        } else {
            ranked &= RANKED_MODS.contains(&acronym.as_str()) && settings.is_empty();
        }
    }
    (Some(rate), ranked)
}

/// Only known constant-rate, unmodified-chart contexts enter live residuals.
pub fn predictable_context(mods: &[Value], rate: f64) -> bool {
    const ALLOWED: [&str; 11] = ["NF", "HD", "FI", "FL", "MR", "SD", "PF", "DT", "NC", "HT", "DC"];
    rate.is_finite() && rate > 0. && mods.iter().all(|m| {
        let acronym = m.get("acronym").and_then(Value::as_str).unwrap_or("").to_uppercase();
        ALLOWED.contains(&acronym.as_str()) && m.get("settings").and_then(Value::as_object)
            .is_none_or(|s| s.keys().all(|k| k == "speed_change" || k == "adjust_pitch"))
    })
}

/// (perfect, great, good, ok, meh, miss)
pub type Stats = [i64; 6];

/// Mania pp accuracy and judgement count.
pub fn acc320(s: &Stats) -> (f64, i64) {
    let n: i64 = s.iter().sum();
    (if n != 0 { (320 * s[0] + 300 * s[1] + 200 * s[2] + 100 * s[3] + 50 * s[4]) as f64 / (320 * n) as f64 } else { 0. }, n)
}

/// What stable's results screen shows: MAX and 300 both count 300.
pub fn acc_stable(s: &Stats) -> f64 {
    let n: i64 = s.iter().sum();
    if n != 0 { (300 * (s[0] + s[1]) + 200 * s[2] + 100 * s[3] + 50 * s[4]) as f64 / (300 * n) as f64 } else { 0. }
}

/// What lazer's results screen shows: PERFECT 305, GREAT 300.
pub fn acc_lazer(s: &Stats) -> f64 {
    let n: i64 = s.iter().sum();
    if n != 0 { (305 * s[0] + 300 * s[1] + 200 * s[2] + 100 * s[3] + 50 * s[4]) as f64 / (305 * n) as f64 } else { 0. }
}

pub fn parse_stats(text: &str) -> Option<Stats> {
    let v: Vec<i64> = serde_json::from_str(text).ok()?;
    v.try_into().ok()
}

/// (1 − stable accuracy) / (1 − lazer accuracy) of the same judgements: predictions stay in lazer units, stable shows
/// MAX as 300 (≈0.88 for a 95 % player). The player's own median when they have lazer plays.
pub fn stable_miss_ratio(db: &Db) -> Result<f64> {
    let mut statement = db.prepare("SELECT stats FROM scores WHERE stats IS NOT NULL ORDER BY played DESC LIMIT 2000")?;
    let stats = statement.query_map([], |r| r.get::<_, String>(0))?;
    let mut ratios: Vec<f64> = stats.filter_map(|s| parse_stats(&s.ok()?)).filter(|s| s.iter().sum::<i64>() >= 100 && acc_lazer(s) < 0.995)
        .map(|s| (1. - acc_stable(&s)) / (1. - acc_lazer(&s))).collect();
    ratios.sort_by(f64::total_cmp);
    Ok(if ratios.len() >= 20 { ratios[ratios.len() / 2] } else { 0.88 })
}

fn stats_text(stats: &Stats) -> String {
    stats.iter().map(i64::to_string).collect::<Vec<_>>().join(",")
}

/// One play across live/Realm sources, including identical achievements on different attempts.
pub fn score_key(sha: &str, total: i64, stats: &Stats, played: Option<&str>) -> String {
    let base = format!("lz|{sha}|{total}|{}", stats_text(stats));
    match played.and_then(timestamp) {
        Some(t) => format!("{base}|{t:.3}"),
        None => base,
    }
}

/// One stable play from scores.db and from tosu's results screen alike (their timestamps differ).
pub fn stable_key(sha: &str, total: i64, stats: &Stats) -> String {
    format!("st|{sha}|{total}|{}", stats_text(stats))
}

/// A row of `scores`; `key` may change to an older key of the same play.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Score {
    pub key: String,
    pub src: String,
    pub local_id: Option<String>,
    pub online_id: i64,
    pub player_id: Option<i64>,
    pub sha256: Option<String>,
    pub md5: Option<String>,
    pub beatmap_id: Option<i64>,
    pub status: Option<i64>,
    pub played: Option<String>,
    pub imported: String,
    pub client: Option<String>,
    pub mods: String,
    pub rate: Option<f64>,
    pub ranked: bool,
    pub stats: Stats,
    pub acc: f64,
    pub n: i64,
    pub total: i64,
    pub pp: Option<f64>,
    pub fresh: bool,
}

/// Insert, or fill in what a later source knows (online id, local id, status); never duplicates.
pub fn upsert_score(db: &Db, mut row: Score) -> Result<Option<String>> {
    if !tracking_allowed(db, row.played.as_deref().and_then(timestamp), None)?
        || db.query_row("SELECT 1 FROM ignored_scores WHERE key=?", [&row.key], |_| Ok(())).optional()?.is_some() {
        return Ok(None);
    }
    if row.key.starts_with("lz|") {
        let old = score_key(row.sha256.as_deref().unwrap_or(""), row.total, &row.stats, None);
        let previous: Option<Option<String>> = db.query_row("SELECT played FROM scores WHERE key=?", [&old], |r| r.get(0)).optional()?;
        if let Some(previous) = previous {
            if previous.as_deref().and_then(timestamp) == row.played.as_deref().and_then(timestamp) {
                row.key = old; // retain existing event references; no destructive migration
            }
        }
    }
    db.execute("INSERT INTO scores(key,src,local_id,online_id,player_id,sha256,md5,beatmap_id,status,played,imported,client,mods,\
                rate,ranked,stats,acc,n,total,pp,fresh) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) \
                ON CONFLICT(key) DO UPDATE SET online_id=MAX(COALESCE(online_id,0), COALESCE(excluded.online_id,0)), \
                local_id=COALESCE(local_id, excluded.local_id), status=COALESCE(excluded.status, status), \
                beatmap_id=COALESCE(beatmap_id, excluded.beatmap_id), md5=COALESCE(md5, excluded.md5), \
                played=COALESCE(played, excluded.played), ranked=MAX(COALESCE(ranked,0), COALESCE(excluded.ranked,0))",
               params![row.key, row.src, row.local_id, row.online_id, row.player_id, row.sha256, row.md5, row.beatmap_id, row.status,
                       row.played, row.imported, row.client, row.mods, row.rate, row.ranked as i64, dumps(&json!(row.stats)), row.acc, row.n,
                       row.total, row.pp, row.fresh as i64])?;
    Ok(Some(row.key))
}

/// Keymodes holding at least `share` of the player's latest n plays; empty before 20 plays.
pub fn played_keys(db: &Db, n: i64, share: f64) -> Result<Vec<i64>> {
    let mut statement = db.prepare("SELECT i.keys FROM scores s JOIN installed i ON i.sha256=s.sha256 WHERE s.src!='public' ORDER BY s.played DESC LIMIT ?")?;
    let rows: Vec<i64> = statement.query_map([n], |r| r.get::<_, Option<i64>>(0))?.filter_map(|k| k.ok().flatten()).filter(|k| SUPPORTED_KEYS.contains(k)).collect();
    if rows.len() < 20 {
        return Ok(Vec::new());
    }
    let mut out: Vec<i64> = rows.iter().copied().filter(|k| rows.iter().filter(|x| *x == k).count() as f64 >= share * rows.len() as f64).collect();
    out.sort_unstable();
    out.dedup();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn db() -> Db {
        connect(Path::new(":memory:")).unwrap()
    }

    #[test]
    fn pruning_takes_only_other_calculators_rows() {
        let dir = std::env::temp_dir().join(format!("ms-prune-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let db = connect(&dir.join("rec.db")).unwrap();
        log_event(&db, "start", None, Some(1.), &json!({})).unwrap();
        kv_set(&db, "k", &json!(1)).unwrap();
        for key in ["a@1.000|old", "b@1.000|old"] {
            db.execute("INSERT INTO feats VALUES (?,x'00')", [key]).unwrap();
        }
        assert_eq!(prune_feats(&db, "new").unwrap(), None, "no rows of the current calculator: nothing goes");
        db.execute("INSERT INTO feats VALUES ('a@1.000|new',x'00')", []).unwrap();
        assert_eq!(prune_feats(&db, "new").unwrap(), Some(2));
        assert_eq!(prune_feats(&db, "new").unwrap(), Some(0));
        let rows: i64 = db.query_row("SELECT COUNT(*) FROM feats", [], |r| r.get(0)).unwrap();
        let events: i64 = db.query_row("SELECT COUNT(*) FROM events", [], |r| r.get(0)).unwrap();
        assert_eq!((rows, events, kv_get(&db, "k")), (1, 1, Some(json!(1))));
        drop(db);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn carried_features_move_only_the_charts_the_calibration_moved() {
        use ms_calc::val::{Dict, Val};
        let after = ms_calc::features::standard_params();
        let before = ms_calc::features::params_of("3c695ee7a23f").unwrap();
        let chart = |keys: i64, cj: f64| {
            let mut f = Dict::new();
            let mut cf = vec![0.; 12];
            (cf[1], cf[6]) = (cj, 0.3);
            f.set("keys", keys);
            (f.set("overall", 5.), f.set("baseline_overall", 5.), f.set("raw_overall", 8.), f.set("od", 8.));
            f.set("calibration_features", Val::floats(&cf));
            crate::codec::encode(&Val::Dict(f))
        };
        let db = db();
        for (key, data) in [("a@1.000", chart(10, 0.95)), ("b@1.000", chart(10, 0.4)), ("c@1.000", chart(7, 0.95))] {
            db.execute("INSERT INTO feats VALUES (?,?)", params![format!("{key}|old"), data]).unwrap();
        }
        assert_eq!(carry_feats(&db, "old", "new", &before, &after, &|_, _| None).unwrap(), 3);
        assert_eq!(carry_feats(&db, "old", "new", &before, &after, &|_, _| None).unwrap(), 0, "a stopped run can repeat");
        let at = |key: &str| db.query_row("SELECT data FROM feats WHERE key=?", [key], |r| r.get::<_, Vec<u8>>(0)).unwrap();
        let Val::Dict(moved) = crate::codec::decode(&at("a@1.000|new")).unwrap() else { panic!() };
        assert!((moved.f("overall") - 5.).abs() > 0.1, "{}", moved.f("overall"));
        assert_ne!(at("b@1.000|new"), at("b@1.000|old"));
        assert_eq!(at("c@1.000|new"), at("c@1.000|old"));
    }

    #[test]
    fn accuracies_and_keys() {
        let s = [100, 50, 10, 5, 2, 1];
        assert!((acc_lazer(&s) - (305. * 100. + 300. * 50. + 2000. + 500. + 100.) / (305. * 168.)).abs() < 1e-12);
        assert_eq!(acc320(&[0; 6]), (0., 0));
        assert_eq!(score_key("a", 5, &[1, 2, 3, 4, 5, 6], None), "lz|a|5|1,2,3,4,5,6");
        assert_eq!(score_key("a", 5, &[1, 2, 3, 4, 5, 6], Some("2026-10-01T10:00:00+00:00")), "lz|a|5|1,2,3,4,5,6|1790848800.000");
    }

    #[test]
    fn mods_decide_rate_and_ranked() {
        assert_eq!(variant(&[]), (Some(1.), true));
        assert_eq!(variant(&[json!({"acronym": "DT"})]), (Some(1.5), true));
        assert_eq!(variant(&[json!({"acronym": "DT", "settings": {"speed_change": 1.3}})]), (Some(1.3), false));
        assert_eq!(variant(&[json!({"acronym": "HT", "settings": {"adjust_pitch": true}})]), (Some(0.75), true));
        assert_eq!(variant(&[json!({"acronym": "WU"})]), (None, false));
        assert!(!predictable_context(&[json!({"acronym": "AT"})], 1.));
        assert!(predictable_context(&[json!({"acronym": "NC", "settings": {"speed_change": 1.2}})], 1.2));
    }

    #[test]
    fn a_tracking_pause_hides_its_events_and_edits_are_never_stale() {
        let db = db();
        let start = log_event(&db, "start", Some("md5:a"), Some(100.), &json!({"keys": 7, "rate": 1.})).unwrap();
        log_event(&db, "finish", Some("md5:a"), Some(200.), &json!({"start_id": start, "y": -3.})).unwrap();
        let load = |db: &Db| crate::events::load_events(db, 0.).unwrap();
        assert_eq!(load(&db).len(), 2);
        set_tracking(&db, false, Some(150.), Some(100.)).unwrap();
        assert!(load(&db).is_empty());
        set_tracking(&db, true, Some(201.), None).unwrap();
        log_event(&db, "start", Some("md5:b"), Some(202.), &json!({"keys": 7, "rate": 1.1})).unwrap();
        assert_eq!(load(&db).iter().map(|e| e.beatmap.clone().unwrap()).collect::<Vec<_>>(), ["md5:b"]);
        db.execute("UPDATE events SET info=? WHERE t=202", [json!({"keys": 7, "rate": 1.2}).to_string()]).unwrap();
        assert_eq!(load(&db)[0].info["rate"], 1.2);
        for t in [99., 100., 150., 200., 201.] {
            log_event(&db, "selected", None, Some(t), &json!({})).unwrap();
        }
        let ids: Vec<(i64, f64)> = db.prepare("SELECT id,t FROM events WHERE t>=0 ORDER BY t,id").unwrap().query_map([], |r| Ok((r.get(0)?, r.get(1)?))).unwrap().map(Result::unwrap).collect();
        let allowed: Vec<i64> = ids.iter().filter(|x| tracking_allowed(&db, Some(x.1), None).unwrap()).map(|x| x.0).collect();
        assert_eq!(tracked_events(&db, 0., &[], None, None).unwrap().iter().map(|e| e.id).collect::<Vec<_>>(), allowed);
        assert!(tracked_events(&db, 0., &["start"], None, None).unwrap().iter().all(|e| e.kind == "start"));
    }

    #[test]
    fn events_and_tracking() {
        let db = db();
        let id = log_event(&db, "start", Some("md5:a"), Some(10.), &json!({"x": 1})).unwrap();
        log_event(&db, "finish", Some("md5:a"), Some(20.), &json!({"x": 2})).unwrap();
        assert_eq!(annotate_attempt(&db, id, "failed", "n", 1.).unwrap(), annotate_attempt(&db, id, "failed", "n", 1.).unwrap());
        set_tracking(&db, false, Some(15.), None).unwrap();
        assert!(!tracking_enabled(&db).unwrap());
        assert_eq!(tracked_events(&db, 0., &[], None, None).unwrap().len(), 1);
        set_tracking(&db, true, Some(25.), None).unwrap();
        assert!(tracking_allowed(&db, Some(26.), None).unwrap() && !tracking_allowed(&db, Some(20.), None).unwrap());
    }

    #[test]
    fn upsert_fills_in_later_knowledge() {
        let db = db();
        let row = Score { key: "lz|a|1|1,0,0,0,0,0".into(), src: "live".into(), sha256: Some("a".into()), total: 1, stats: [1, 0, 0, 0, 0, 0], ..Score::default() };
        upsert_score(&db, row.clone()).unwrap();
        upsert_score(&db, Score { online_id: 7, local_id: Some("u".into()), ..row }).unwrap();
        let (online, local): (i64, String) = db.query_row("SELECT online_id, local_id FROM scores", [], |r| Ok((r.get(0)?, r.get(1)?))).unwrap();
        assert_eq!((online, local.as_str()), (7, "u"));
    }
}

/// Hand freed heap back to the OS: glibc keeps it after the fit and each build, as RSS nothing uses.
pub fn release_memory() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    // Safety: no arguments beyond the padding, callable from any thread.
    unsafe {
        libc::malloc_trim(0);
    }
}
