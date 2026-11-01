//! Scores and installed maps from the game (recdata.py's import half).
use crate::paths::{self, local_file, local_origin, long_path, stable_store};
use crate::pyjson::dumps;
use crate::realm::dump;
use crate::replays;
use crate::stable::{self, sha256_files};
use crate::store::{acc320, kv_get, kv_set, local_stamp, now, score_key, stable_key, stable_miss_ratio, timestamp, tracking_allowed, upsert_score, variant, Db, Score, Stats};
use chrono::DateTime;
use rusqlite::{params, OptionalExtension};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

type R<T> = Result<T, String>;

const STAT_NAMES: [&str; 6] = ["perfect", "great", "good", "ok", "meh", "miss"];
const HIT_NAMES: [&str; 6] = ["geki", "300", "katu", "100", "50", "0"];

fn sql(e: rusqlite::Error) -> String {
    e.to_string()
}

fn int(v: &Value, k: &str) -> i64 {
    v.get(k).and_then(|x| x.as_i64().or_else(|| x.as_f64().map(|f| f as i64))).unwrap_or(0)
}

fn text(v: &Value, k: &str) -> Option<String> {
    v.get(k).and_then(Value::as_str).map(str::to_string)
}

fn parsed(v: &Value, k: &str, default: &str) -> Value {
    serde_json::from_str(v.get(k).and_then(Value::as_str).filter(|s| !s.is_empty()).unwrap_or(default)).unwrap_or_else(|_| json!({}))
}

pub fn stats_of(raw: &Value) -> Stats {
    let st = parsed(raw, "stats", "{}");
    STAT_NAMES.map(|k| int(&st, k))
}

/// Shared by live results and feedback imports; tombstones prevent later Realm resurrection.
pub fn score_allowed(db: &Db, score: &Value, play: Option<&Value>, exclude: bool) -> rusqlite::Result<bool> {
    let h = score.get("hits").cloned().unwrap_or(Value::Null);
    let stats: Stats = HIT_NAMES.map(|k| int(&h, k));
    let played = score.get("played").and_then(Value::as_str);
    let key = score_key(score["sha256"].as_str().unwrap_or(""), int(score, "score"), &stats, played);
    let mut allowed = db.query_row("SELECT 1 FROM ignored_scores WHERE key=?", [&key], |_| Ok(())).optional()?.is_none()
        && tracking_allowed(db, played.and_then(timestamp), None)?;
    if let Some(play) = play {
        allowed = allowed && tracking_allowed(db, Some(play.get("end").and_then(Value::as_f64).unwrap_or_else(now)), play["t"].as_f64())?;
    }
    if !allowed && exclude {
        db.execute("INSERT OR IGNORE INTO ignored_scores VALUES (?)", [&key])?;
    }
    Ok(allowed)
}

/// lazer_scores.js lines → (score rows of user_id, installed-map rows).
pub fn realm_rows(raw: &[Value], user_id: Option<i64>) -> Vec<Score> {
    let imported = local_stamp();
    let mut out = Vec::new();
    for s in raw {
        if s.get("map").is_some() || s.get("user_id").and_then(Value::as_i64) != user_id {
            continue; // explicit account identity, not the most frequent name
        }
        let stats = stats_of(s);
        let (acc, n) = acc320(&stats);
        let mods = parsed(s, "mods", "[]");
        let (rate, ranked) = variant(mods.as_array().map_or(&[], Vec::as_slice));
        let (hash, total) = (s["hash"].as_str().unwrap_or(""), int(s, "total"));
        let date = s.get("date").and_then(Value::as_str);
        let key = if s["legacy"].as_bool().unwrap_or(false) && int(s, "legacy_online_id") > 0 {
            format!("legacy|{}", int(s, "legacy_online_id"))
        } else {
            score_key(hash, total, &stats, date)
        };
        out.push(Score { key, src: "realm".to_string(), local_id: text(s, "id"), online_id: int(s, "online_id").max(0), player_id: user_id, sha256: Some(hash.to_string()), md5: text(s, "md5"),
                         beatmap_id: s["beatmap_id"].as_i64(), status: s["status"].as_i64(), played: date.map(str::to_string), imported: imported.clone(), client: text(s, "client"), mods: dumps(&mods), rate, ranked,
                         stats, acc, n, total, pp: None, fresh: false });
    }
    out
}

/// A dump of the live file through a private copy: lazer rewrites freed pages under a reader.
pub fn snapshot(realm: Option<&Path>, maps: bool, recent: Option<(i64, i64)>) -> R<Vec<Value>> {
    let source = realm.map_or_else(|| paths::lazer_data().join("client.realm"), Path::to_path_buf);
    std::fs::create_dir_all(paths::cache()).map_err(|e| e.to_string())?;
    let copy = paths::cache().join(format!("client-{}.realm", std::process::id()));
    std::fs::copy(&source, &copy).map_err(|e| format!("{}: {e}", source.display()))?;
    let out = dump(&copy, maps, recent);
    let _ = std::fs::remove_file(&copy);
    out
}

fn set_origin(db: &Db, host: &str, root: &Path) -> rusqlite::Result<()> {
    kv_set(db, "manifest_origin", &json!({"host": host, "root": root.to_string_lossy()}))?;
    kv_set(db, "manifest_error", &Value::Null)
}

fn replace_library(db: &Db, maps: &[(String, i64, i64, Option<String>, Option<i64>, i64, Option<f64>, Option<f64>, Option<String>, Option<String>, Option<String>)],
                   local: &[(String, Option<String>, Option<String>, i64, Option<String>)]) -> rusqlite::Result<()> {
    db.execute("DELETE FROM installed", [])?;
    let mut q = db.prepare("INSERT OR REPLACE INTO installed VALUES (?,?,?,?,?,?,?,?,?,?,?)")?;
    for m in maps {
        q.execute(params![m.0, m.1, m.2, m.3, m.4, m.5, m.6, m.7, m.8, m.9, m.10])?;
    }
    db.execute("DELETE FROM installed_local", [])?;
    let mut q = db.prepare("INSERT OR REPLACE INTO installed_local VALUES (?,?,?,?,?)")?;
    for l in local {
        q.execute(params![l.0, l.1, l.2, l.3, l.4])?;
    }
    Ok(())
}

pub struct Imported {
    pub user_id: Option<i64>,
    pub seen: usize,
    pub new: i64,
    pub installed: usize,
    pub miss_ratio: Option<f64>,
}

pub fn import_realm(db: &Db, realm: Option<&Path>, user_id: Option<i64>) -> R<Imported> {
    let raw = snapshot(realm, true, None).inspect_err(|e| {
        let _ = kv_set(db, "manifest_error", &json!(e.lines().next().unwrap_or("").chars().take(160).collect::<String>()));
    })?;
    let mut user_id = user_id;
    if user_id.is_none() {
        let mut users: BTreeMap<i64, i64> = BTreeMap::new();
        for s in raw.iter().filter(|s| s.get("map").is_none()) {
            if let Some(u) = s["user_id"].as_i64().filter(|&u| u != 0) {
                *users.entry(u).or_default() += 1;
            }
        }
        user_id = kv_get(db, "user_id").and_then(|v| v.as_i64()).or_else(|| users.iter().max_by_key(|x| *x.1).map(|x| *x.0));
        if let Some(u) = user_id {
            kv_set(db, "user_id", &json!(u)).map_err(sql)?;
        }
    }
    let scores = realm_rows(&raw, user_id);
    let (mut maps, mut local) = (Vec::new(), Vec::new());
    for s in raw.iter().filter(|s| s.get("map").is_some_and(|m| m.as_str().is_some_and(|m| !m.is_empty()))) {
        let sha = s["map"].as_str().unwrap().to_string();
        if s["keys"].as_f64().is_some_and(|k| k != 0.) {
            maps.push((sha.clone(), int(s, "beatmap_id"), int(s, "set_id"), text(s, "md5"), s["status"].as_i64(), s["keys"].as_f64().unwrap() as i64, s["od"].as_f64(), s["length"].as_f64(),
                       text(s, "title"), text(s, "artist"), text(s, "version")));
        }
        local.push((sha, text(s, "online_md5"), text(s, "audio_sha"), i64::from(s["audio_required"].as_bool().unwrap_or(false)), text(s, "creator")));
    }
    let tx = db.unchecked_transaction().map_err(sql)?;
    let count = |db: &Db| db.query_row("SELECT COUNT(*) FROM scores", [], |r| r.get::<_, i64>(0));
    let before = count(db).map_err(sql)?;
    for r in &scores {
        upsert_score(db, r.clone()).map_err(sql)?;
    }
    // dump requires a whole file: a successful empty manifest really is empty.
    replace_library(db, &maps, &local).map_err(sql)?;
    let after = count(db).map_err(sql)?;
    tx.commit().map_err(sql)?;
    kv_set(db, "realm_import", &json!({"t": now(), "scores": scores.len(), "new": after - before, "maps": maps.len()})).map_err(sql)?;
    let (host, mut root) = local_origin("lazer");
    if let Some(realm) = realm {
        let real = std::fs::canonicalize(realm).unwrap_or_else(|_| realm.to_path_buf());
        if real != root.join("client.realm") {
            root = real.parent().unwrap_or(Path::new("")).to_path_buf();
        }
    }
    set_origin(db, &host, &root).map_err(sql)?;
    replays::archive(db, &raw, "lazer").map_err(sql)?;
    Ok(Imported { user_id, seen: scores.len(), new: after - before, installed: maps.len(), miss_ratio: None })
}

/// The Songs folder: osu!.<user>.cfg's BeatmapDirectory when it exists here, else <root>/Songs.
pub fn stable_songs(root: &Path) -> PathBuf {
    for entry in std::fs::read_dir(root).into_iter().flatten().flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with("osu!.") && name.ends_with(".cfg") {
            let Ok(bytes) = std::fs::read(entry.path()) else { continue };
            for line in String::from_utf8_lossy(&bytes).trim_start_matches('\u{feff}').lines() {
                if let Some((k, v)) = line.split_once('=') {
                    if k.trim() == "BeatmapDirectory" && !v.trim().is_empty() && root.join(v.trim()).is_dir() {
                        return root.join(v.trim());
                    }
                }
            }
        }
    }
    root.join("Songs")
}

/// The stable_store path of a stable .osu (hard link; a copy across file systems).
pub fn stable_link(path: &Path, sha: &str) -> std::io::Result<()> {
    let Some(dest) = local_file(sha, &stable_store()) else { return Ok(()) };
    if !dest.is_file() {
        std::fs::create_dir_all(dest.parent().unwrap())?;
        let tmp = dest.with_extension(format!("{}.part", std::process::id()));
        if std::fs::hard_link(path, &tmp).is_err() {
            std::fs::copy(path, &tmp)?;
        }
        std::fs::rename(&tmp, &dest)?;
    }
    Ok(())
}

fn stable_status(byte: u8) -> i64 {
    match byte {
        4 => 1,
        5 => 2,
        6 => 3,
        7 => 4,
        2 => 0,
        _ => -3,
    }
}

/// stable's osu!.db (mania charts) and scores.db (the account's mania scores), the import_realm of stable.
pub fn import_stable(db: &Db, root: Option<&Path>) -> R<Imported> {
    let root = root.map_or_else(paths::stable_data, Path::to_path_buf);
    // stable writes osu!.db when it first closes; until then its library is empty here, not an error
    let read = (root.join("osu!.db").is_file() || !root.join("osu!.exe").is_file())
        .then(|| std::fs::read(root.join("osu!.db")).map_err(|e| e.to_string()).and_then(|b| stable::library(&b)));
    let (player, charts) = match read {
        None => (String::new(), Vec::new()),
        Some(Ok(x)) => x,
        Some(Err(e)) => {
            kv_set(db, "manifest_error", &json!(e.lines().next().unwrap_or("").chars().take(160).collect::<String>())).map_err(sql)?;
            return Err(e);
        }
    };
    let songs = stable_songs(&root);
    let charts: Vec<(stable::Chart, PathBuf)> = charts.into_iter().filter(|c| c.mode == 3 && !c.md5.is_empty() && !c.file.is_empty())
        .map(|c| { let p = long_path(&songs.join(&c.folder).join(&c.file)); (c, p) }).collect();
    db.execute("CREATE TABLE IF NOT EXISTS stable_files(path TEXT PRIMARY KEY, mtime REAL, size INTEGER, sha256 TEXT)", []).map_err(sql)?;
    let mut known: HashMap<PathBuf, (f64, i64, String)> = HashMap::new();
    {
        let mut q = db.prepare("SELECT path, mtime, size, sha256 FROM stable_files").map_err(sql)?;
        let rows = q.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get(1)?, r.get(2)?, r.get(3)?))).map_err(sql)?;
        for row in rows.flatten() {
            known.insert(PathBuf::from(row.0), (row.1, row.2, row.3));
        }
    }
    let (mut stats, mut todo): (HashMap<PathBuf, (f64, i64)>, Vec<PathBuf>) = Default::default();
    for (_, path) in &charts {
        let Ok(st) = std::fs::metadata(path) else { continue };
        let mtime = st.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0., |d| d.as_secs_f64());
        let stamp = (mtime, st.len() as i64);
        if known.get(path).is_none_or(|k| (k.0, k.1) != stamp) {
            todo.push(path.clone());
        }
        stats.insert(path.clone(), stamp);
    }
    let hashed: HashMap<&PathBuf, Option<String>> = todo.iter().zip(sha256_files(&todo, 0)).collect();
    let sha_of = |p: &PathBuf| -> Option<String> { hashed.get(p).cloned().unwrap_or_else(|| known.get(p).map(|k| k.2.clone())) };
    let (mut maps, mut local, mut md5_sha, mut md5_bid) = (Vec::new(), Vec::new(), HashMap::new(), HashMap::new());
    for (c, path) in &charts {
        if !stats.contains_key(path) {
            continue;
        }
        let Some(sha) = sha_of(path) else { continue };
        stable_link(path, &sha).map_err(|e| e.to_string())?;
        md5_sha.insert(c.md5.clone(), sha.clone());
        md5_bid.insert(c.md5.clone(), (c.beatmap_id != 0).then_some(c.beatmap_id as i64));
        maps.push((sha.clone(), c.beatmap_id as i64, c.set_id as i64, Some(c.md5.clone()), Some(stable_status(c.status)), (c.cs as f64).round_ties_even() as i64, Some(c.od as f64), Some(c.total_ms as f64),
                   Some(c.title.clone()), Some(c.artist.clone()), Some(c.version.clone())));
        local.push((sha, Some(c.md5.clone()), None, 0, Some(c.creator.clone())));
    }
    let user_id = kv_get(db, "user_id").and_then(|v| v.as_i64());
    let imported = local_stamp();
    // no plays yet: stable writes scores.db on the first one
    let raw = std::fs::read(root.join("scores.db")).map_err(|e| e.to_string()).and_then(|b| stable::scores(&b)).unwrap_or_default();
    let mut scores = Vec::new();
    for s in raw {
        let Some(sha) = md5_sha.get(&s.md5) else { continue };
        if s.mode != 3 || s.player != player {
            continue;
        }
        let st: Stats = s.stats.map(i64::from);
        let (acc, n) = acc320(&st);
        let rate = if s.mods & (64 | 512) != 0 { 1.5 } else if s.mods & 256 != 0 { 0.75 } else { 1. };
        let ranked = s.mods & !(1 | 2 | 8 | 64 | 256 | 512 | 1024 | 1048576 | 1073741824 | 16384 | 32) == 0;
        let played = DateTime::from_timestamp(s.played.floor() as i64, 0).map(|t| t.format("%Y-%m-%dT%H:%M:%S").to_string());
        scores.push(Score { key: stable_key(sha, s.score as i64, &st), src: "stable".to_string(), local_id: None, online_id: s.online_id.max(0), player_id: user_id, sha256: Some(sha.clone()), md5: Some(s.md5.clone()),
                            beatmap_id: md5_bid[&s.md5], status: None, played, imported: imported.clone(), client: Some("stable".to_string()), mods: s.mods.to_string(), rate: Some(rate), ranked, stats: st, acc, n,
                            total: s.score as i64, pp: None, fresh: false });
    }
    let tx = db.unchecked_transaction().map_err(sql)?;
    let count = |db: &Db| db.query_row("SELECT COUNT(*) FROM scores", [], |r| r.get::<_, i64>(0));
    let before = count(db).map_err(sql)?;
    for r in &scores {
        upsert_score(db, r.clone()).map_err(sql)?;
    }
    replace_library(db, &maps, &local).map_err(sql)?;
    db.execute("DELETE FROM stable_files", []).map_err(sql)?;
    {
        let mut q = db.prepare("INSERT INTO stable_files VALUES (?,?,?,?)").map_err(sql)?;
        for (p, s) in &stats {
            if let Some(sha) = sha_of(p) {
                q.execute(params![p.to_string_lossy(), s.0, s.1, sha]).map_err(sql)?;
            }
        }
    }
    let after = count(db).map_err(sql)?;
    tx.commit().map_err(sql)?;
    kv_set(db, "realm_import", &json!({"t": now(), "scores": scores.len(), "new": after - before, "maps": maps.len()})).map_err(sql)?;
    kv_set(db, "stable_player", &json!(player)).map_err(sql)?;
    replays::archive_stable(db, Some(&root)).map_err(sql)?;
    let ratio = stable_miss_ratio(db).map_err(sql)?;
    kv_set(db, "stable_miss_ratio", &json!(ratio)).map_err(sql)?;
    let (host, store) = local_origin("stable");
    set_origin(db, &host, &store).map_err(sql)?;
    Ok(Imported { user_id, seen: scores.len(), new: after - before, installed: maps.len(), miss_ratio: Some(ratio) })
}

/// The active client's scores and installed maps.
pub fn import_library(db: &Db, client: &str) -> R<Imported> {
    if client == "stable" { import_stable(db, None) } else { import_realm(db, None, None) }
}

fn website(url: &str) -> R<(String, String)> {
    let reply = ureq::get(url).set("User-Agent", "maniascope (personal practice tool)").set("Accept", "application/json").timeout(std::time::Duration::from_secs(20)).call().map_err(|e| e.to_string())?;
    let last = reply.get_url().to_string();
    Ok((last, reply.into_string().map_err(|e| e.to_string())?))
}

fn quote(text: &str) -> String {
    text.bytes().map(|b| if b.is_ascii_alphanumeric() || b"-._~/".contains(&b) { (b as char).to_string() } else { format!("%{b:02X}") }).collect()
}

/// The player's top 100 mania plays from their public osu! profile (the website's own JSON; no API key, user
/// 2026-10-01): one lookup + one request. Bests outside the top 100 stay unknown (below #100).
pub fn import_website_best(db: &Db, username: &str) -> R<(i64, usize)> {
    let (last, page) = website(&format!("https://osu.ppy.sh/users/{}", quote(username.trim())))?;
    // The profile page names its numeric id in the canonical link (a redirect may also carry it).
    let canonical = regex_lite::Regex::new(r#"rel="canonical" href="https://osu\.ppy\.sh/users/(\d+)""#).unwrap();
    let ending = regex_lite::Regex::new(r"/users/(\d+)$").unwrap();
    let id = canonical.captures(&page).or_else(|| ending.captures(&last)).and_then(|m| m[1].parse::<i64>().ok()).ok_or_else(|| format!("osu! user '{username}' not found"))?;
    let (_, body) = website(&format!("https://osu.ppy.sh/users/{id}/scores/best?mode=mania&limit=100"))?;
    let rows: Value = serde_json::from_str(&body).map_err(|e| e.to_string())?;
    Ok((id, store_website_best(db, username, id, &rows)?))
}

fn store_website_best(db: &Db, username: &str, id: i64, rows: &Value) -> R<usize> {
    let imported = local_stamp();
    let mut count = 0;
    let tx = db.unchecked_transaction().map_err(sql)?;
    // A changed username replaces the profile, it doesn't add a second player's bests (re-fetched if changed back).
    db.execute("DELETE FROM scores WHERE src='public' AND key LIKE 'web|%' AND player_id<>?", [id]).map_err(sql)?;
    for s in rows.as_array().map_or(&[][..], Vec::as_slice) {
        let (st, b) = (&s["statistics"], &s["beatmap"]);
        let stats: Stats = STAT_NAMES.map(|k| int(st, k));
        let mods = s.get("mods").cloned().filter(|m| !m.is_null()).unwrap_or_else(|| json!([]));
        let (rate, ranked) = variant(mods.as_array().map_or(&[], Vec::as_slice));
        let pp = s["pp"].as_f64().filter(|&p| p != 0.);
        if rate.is_none() || pp.is_none() {
            continue;
        }
        let (acc, n) = acc320(&stats);
        let played = text(s, "ended_at").filter(|t| !t.is_empty()).unwrap_or_else(|| imported.clone()).replace('Z', "");
        upsert_score(db, Score { key: format!("web|{}", int(s, "id")), src: "public".to_string(), local_id: None, online_id: int(s, "id"), player_id: Some(id), sha256: None, md5: text(b, "checksum"),
                                 beatmap_id: s["beatmap_id"].as_i64(), status: b["ranked"].as_i64(), played: played.split('+').next().map(str::to_string), imported: imported.clone(),
                                 client: Some("website".to_string()), mods: dumps(&mods), rate, ranked, stats, acc, n, total: int(s, "total_score"), pp, fresh: false }).map_err(sql)?;
        count += 1;
    }
    tx.commit().map_err(sql)?;
    kv_set(db, "website_user", &json!({"name": username, "id": id, "t": now(), "scores": count})).map_err(sql)?;
    Ok(count)
}

/// The snapshot's stable high scores of this user (their own namespace; pp as the server had it).
pub fn import_public_user(db: &Db, public: &crate::public::Public) -> rusqlite::Result<()> {
    let (imported, tx) = (local_stamp(), db.unchecked_transaction()?);
    for s in &public.meta.user_scores {
        let mut stats: Stats = [0; 6];
        stats.iter_mut().zip(&s.stats).for_each(|(a, b)| *a = *b);
        let (acc, n) = acc320(&stats);
        let map = public.map_info(s.beatmap_id);
        upsert_score(db, Score { key: format!("legacy|{}", s.score_id), src: "public".to_string(), local_id: None, online_id: s.score_id, player_id: Some(public.meta.user_id), sha256: None,
                                 md5: map.map(|m| m.md5.clone()).filter(|m| !m.is_empty()), beatmap_id: Some(s.beatmap_id), status: map.map(|m| m.approved), played: Some(s.date.clone()),
                                 imported: imported.clone(), client: Some("stable".to_string()), mods: s.mods.to_string(), rate: Some(s.rate), ranked: s.ranked, stats, acc, n, total: s.score, pp: s.pp, fresh: false })?;
    }
    tx.commit()
}

/// A results-screen observation from tosu. fresh=false: a past score being viewed (history only). → (key, new)
pub fn add_live_score(db: &Db, score: &Value, fresh: bool, play: Option<&Value>) -> rusqlite::Result<(Option<String>, bool)> {
    let h = score.get("hits").cloned().unwrap_or(Value::Null);
    let stats: Stats = HIT_NAMES.map(|k| int(&h, k));
    let (acc, n) = acc320(&stats);
    let mods = score.get("mods_list").filter(|m| !m.is_null());
    let (rate, ranked) = mods.map_or_else(|| (score["rate"].as_f64(), false), |m| variant(m.as_array().map_or(&[], Vec::as_slice))); // [] is NoMod: ranked
    let mut score = score.clone();
    if fresh && !score["played"].as_str().is_some_and(|p| !p.is_empty()) {
        score["played"] = json!(chrono::Utc::now().to_rfc3339());
    }
    if !score_allowed(db, &score, play.filter(|_| fresh), true)? {
        return Ok((None, false));
    }
    let (sha, total) = (score["sha256"].as_str().unwrap_or("").to_string(), int(&score, "score"));
    let played = text(&score, "played").filter(|p| !p.is_empty());
    let stable = score["client"].as_str() == Some("stable");
    let key = if stable { stable_key(&sha, total, &stats) } else { score_key(&sha, total, &stats, played.as_deref()) };
    let tx = db.unchecked_transaction()?;
    let mut known: Option<bool> = db.query_row("SELECT fresh FROM scores WHERE key=?", [&key], |r| r.get::<_, Option<i64>>(0)).optional()?.map(|f| f.unwrap_or(0) != 0);
    let old: Option<(Option<String>, Option<i64>)> = db.query_row("SELECT played, fresh FROM scores WHERE key=?", [score_key(&sha, total, &stats, None)], |r| Ok((r.get(0)?, r.get(1)?))).optional()?;
    if let Some(old) = old {
        if old.0.as_deref().and_then(timestamp) == played.as_deref().and_then(timestamp) {
            known = Some(old.1.unwrap_or(0) != 0);
        }
    }
    let stamp = local_stamp();
    let mods_text = mods.map_or_else(|| "[]".to_string(), dumps);
    let key = upsert_score(db, Score { key, src: "live".to_string(), local_id: None, online_id: 0, player_id: kv_get(db, "user_id").and_then(|v| v.as_i64()), sha256: Some(sha), md5: text(&score, "md5"),
                                       beatmap_id: score["beatmap_id"].as_i64(), status: score["status"].as_i64(), played: Some(played.unwrap_or_else(|| stamp.clone())), imported: stamp,
                                       client: Some(if stable { "stable-live" } else { "lazer-live" }.to_string()), mods: mods_text, rate, ranked, stats, acc, n, total, pp: None, fresh })?;
    if fresh && known == Some(false) {
        if let Some(key) = &key {
            db.execute("UPDATE scores SET fresh=1 WHERE key=?", [key])?;
        }
    }
    tx.commit()?;
    Ok((key, known.is_none()))
}

#[cfg(test)]
mod website_tests {
    use super::*;

    #[test]
    fn website_top_plays_become_ledger_bests() {
        let dir = std::env::temp_dir().join(format!("ms-web-best-{}", std::process::id()));
        let db = crate::store::connect(&dir.join("rec.db")).unwrap();
        let rows = json!([{"id": 7, "beatmap_id": 11, "pp": 500.5, "ended_at": "2026-05-03T17:09:43Z", "total_score": 900000, "mods": [{"acronym": "DT"}],
                           "beatmap": {"checksum": "m".repeat(32), "ranked": 1}, "statistics": {"perfect": 900, "great": 80, "good": 10, "ok": 5, "meh": 3, "miss": 2}},
                          {"id": 8, "beatmap_id": 12, "pp": null, "mods": [], "beatmap": {}, "statistics": {}}]);
        assert_eq!(store_website_best(&db, "someone", 42, &rows).unwrap(), 1);
        let row: (i64, f64, f64, String, String) = db.query_row("SELECT beatmap_id, rate, pp, md5, client FROM scores WHERE key='web|7'", [], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?))).unwrap();
        assert_eq!(row, (11, 1.5, 500.5, "m".repeat(32), "website".to_string()));
        assert_eq!(crate::store::kv_get(&db, "website_user").unwrap()["id"], 42);
        drop(db);
        let _ = std::fs::remove_dir_all(dir);
    }
}
