//! Durable evidence for tracked attempts. Lazer's .osr payload is already compressed. Hashes identify files, not
//! machine-specific absolute paths. Never writes to the game's file store, and never archives scores from a
//! tracking-disabled interval.
use crate::imports::stats_of;
use crate::paths::{self, local_file, local_origin, long_path};
use crate::pyjson::dumps;
use crate::stable::osr_header;
use crate::store::{kv_get, parse_stats, score_key, timestamp, tracking_allowed, variant, Db};
use rusqlite::{params, OptionalExtension, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

pub const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS replay_evidence(
 score_key TEXT PRIMARY KEY, score_uuid TEXT, chart_sha256 TEXT, chart_md5 TEXT,
 replay_sha256 TEXT, archive_path TEXT, pauses TEXT, mods TEXT, rate REAL,
 played TEXT, status TEXT, error TEXT);
CREATE INDEX IF NOT EXISTS replay_score_uuid ON replay_evidence(score_uuid);
";

fn root(db: &Db) -> PathBuf {
    db.path().map_or_else(paths::rec_dir, |p| Path::new(p).parent().unwrap_or(Path::new(".")).to_path_buf())
}

fn hex(data: &[u8]) -> String {
    Sha256::digest(data).iter().map(|b| format!("{b:02x}")).collect()
}

fn keep(target: &Path, data: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    let dir = target.parent().unwrap();
    std::fs::create_dir_all(dir)?;
    let tmp = dir.join(format!(".replay-{}", std::process::id()));
    let done = (|| {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(data)?;
        f.sync_all()?;
        std::fs::rename(&tmp, target)
    })();
    let _ = std::fs::remove_file(&tmp);
    done
}

fn relative(sha: &str) -> String {
    format!("replays/{}/{sha}.osr", &sha[..2])
}

/// Link imported/live score keys to files; repeated calls are idempotent. Only observed attempts are archived: old
/// imported bests remain ordinary historical evidence, not fictitious sessions. Missing files keep a pointer and a
/// reason, so a subsequent import can finish archiving them.
pub fn archive(db: &Db, raw: &[Value], client: &str) -> Result<usize> {
    db.execute_batch(SCHEMA)?;
    let root = root(db);
    let origin = local_origin(client).1;
    let user = kv_get(db, "user_id").and_then(|v| v.as_i64());
    // One pass: a json_extract scan per score took 50 s on 12k scores.
    let finished: HashSet<String> = db.prepare("SELECT json_extract(info,'$.key') FROM events WHERE kind='finish'")?
        .query_map([], |r| r.get::<_, Option<String>>(0))?.filter_map(|k| k.ok().flatten()).collect();
    let mut count = 0;
    for item in raw {
        if item.get("map").is_some() || item["user_id"].as_i64() != user {
            continue;
        }
        let (Some(sha), Some(played)) = (item["hash"].as_str().filter(|s| !s.is_empty()), item["date"].as_str()) else { continue };
        let t = timestamp(played);
        if t.is_none() || !tracking_allowed(db, t, None)? {
            continue;
        }
        let key = score_key(sha, item["total"].as_i64().unwrap_or(0), &stats_of(item), Some(played));
        let score: Option<(String, i64)> = db.query_row("SELECT key,fresh FROM scores WHERE key=? OR local_id=? ORDER BY fresh DESC LIMIT 1", params![key, item["id"].as_str()],
                                                        |r| Ok((r.get(0)?, r.get::<_, Option<i64>>(1)?.unwrap_or(0)))).optional()?;
        let Some((key, fresh)) = score else { continue };
        if fresh == 0 && !finished.contains(&key) {
            continue;
        }
        let mut replay = item["replay_sha256"].as_str().map(str::to_string);
        if replay.is_none() {
            // a cleaned game-side reference must not erase our archive
            replay = db.query_row("SELECT replay_sha256 FROM replay_evidence WHERE score_key=? AND status='archived'", [&key], |r| r.get(0)).optional()?;
        }
        let (mut path, mut status, mut error) = (None, "missing", None);
        match replay.as_deref().filter(|s| s.len() == 64 && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))) {
            Some(sha) => {
                let rel = relative(sha);
                let target = root.join(&rel);
                let mut stored = target.is_file();
                if !stored {
                    let read = local_file(sha, &origin).ok_or_else(|| "no file".to_string()).and_then(|p| std::fs::read(p).map_err(|e| e.to_string()));
                    match read.and_then(|data| if hex(&data) == sha { keep(&target, &data).map_err(|e| e.to_string()) } else { Err("replay hash mismatch".to_string()) }) {
                        Ok(()) => { stored = true; count += 1; }
                        Err(e) => error = Some(e.chars().take(240).collect::<String>()),
                    }
                }
                if stored {
                    path = Some(rel);
                    status = "archived";
                }
            }
            None => error = Some("lazer has no replay file yet".to_string()),
        }
        let mods = item["mods"].as_str().filter(|m| !m.is_empty()).unwrap_or("[]");
        let parsed: Value = serde_json::from_str(mods).unwrap_or_else(|_| json!([]));
        db.execute("INSERT OR REPLACE INTO replay_evidence VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                   params![key, item["id"].as_str(), sha, item["md5"].as_str(), replay, path, dumps(item.get("pauses").unwrap_or(&json!([]))), mods,
                           variant(parsed.as_array().map_or(&[], Vec::as_slice)).0, played, status, error])?;
    }
    Ok(count)
}

/// archive for stable: its local replay of each tracked stable play (Data/r/<map md5>-<ticks>.osr, matched by score and
/// judgements in the header) and the pause positions tosu saw (the .osr has none; lazer's Realm does).
pub fn archive_stable(db: &Db, game: Option<&Path>) -> Result<usize> {
    db.execute_batch(SCHEMA)?;
    let root = root(db);
    let dir = game.map_or_else(paths::stable_data, Path::to_path_buf).join("Data").join("r");
    type Row = (String, Option<String>, Option<String>, Option<String>, i64, Option<i64>, String, Option<f64>, Option<String>, Option<String>);
    let rows: Vec<Row> = db.prepare("SELECT s.key,s.sha256,s.md5,s.stats,s.total,s.fresh,COALESCE(s.mods,'[]'),s.rate,s.played,e.status FROM scores s LEFT JOIN replay_evidence e ON e.score_key=s.key \
                                     WHERE (s.src='stable' OR s.client='stable-live') AND (e.status IS NULL OR e.status!='archived')")?
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?, r.get(6)?, r.get(7)?, r.get(8)?, r.get(9)?)))?.collect::<Result<_>>()?;
    let store = local_origin("stable").1;
    let mut count = 0;
    for (key, sha, md5, stats, total, fresh, mods, rate, played, _) in rows {
        let finish: Option<Option<String>> = db.query_row("SELECT info FROM events WHERE kind='finish' AND json_extract(info,'$.key')=? LIMIT 1", [&key], |r| r.get(0)).optional()?;
        if fresh.unwrap_or(0) == 0 && finish.is_none() {
            continue; // only observed attempts, as for lazer
        }
        let md5 = md5.or_else(|| {
            let path = local_file(sha.as_deref()?, &store)?;
            Some(format!("{:x}", md5::compute(std::fs::read(path).ok()?)))
        });
        let Some(md5) = md5 else { continue };
        let want = stats.as_deref().and_then(parse_stats);
        let mut data = None;
        for entry in std::fs::read_dir(long_path(&dir)).into_iter().flatten().flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if !(name.starts_with(&format!("{md5}-")) && name.ends_with(".osr")) {
                continue;
            }
            let Ok(bytes) = std::fs::read(entry.path()) else { continue };
            if osr_header(&bytes).is_ok_and(|h| i64::from(h.score) == total && want.is_some_and(|w| h.stats.map(i64::from) == w)) {
                data = Some(bytes);
                break;
            }
        }
        let (mut path, mut status, mut error, mut replay) = (None, "missing", Some("stable has no replay file yet"), None);
        if let Some(data) = data {
            let sha = hex(&data);
            let rel = relative(&sha);
            if !root.join(&rel).is_file() {
                keep(&root.join(&rel), &data).map_err(|e| rusqlite::Error::InvalidParameterName(e.to_string()))?;
                count += 1;
            }
            path = Some(rel);
            replay = Some(sha);
            status = "archived";
            error = None;
        }
        let pauses = finish.flatten().and_then(|f| serde_json::from_str::<Value>(&f).ok()).and_then(|f| f.get("pauses").filter(|p| !p.is_null()).cloned()).unwrap_or_else(|| json!([]));
        db.execute("INSERT OR REPLACE INTO replay_evidence VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                   params![key, None::<String>, sha, md5, replay, path, dumps(&pauses), mods, rate, played, status, error])?;
    }
    Ok(count)
}
