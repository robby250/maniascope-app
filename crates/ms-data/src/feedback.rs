//! Local evidence for calibration, read by `calib feedback`: ~/.config/maniascope/feedback.json (per
//! machine). An entry is a chart identity (sha256 of the file) at a rate; judgments compare two entries.
use crate::imports::score_allowed;
use crate::paths;
use crate::store::{local_stamp, Db};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

fn file() -> PathBuf {
    paths::config().join("feedback.json")
}

fn array(m: &Value) -> Option<&Vec<Value>> {
    m.get("array").unwrap_or(m).as_array()
}

/// tosu `play.mods` → (constant rate | None, note). Unknown is never 1.0.
pub fn rate_from_mods(mods: &Value) -> (Option<f64>, String) {
    let Some(arr) = array(mods) else { return (None, "rate not reported".to_string()) };
    let mut rate = 1.;
    for m in arr {
        let acronym = m["acronym"].as_str().unwrap_or("").to_uppercase();
        match acronym.as_str() {
            "WU" | "WD" | "AS" => return (None, format!("{acronym} is variable speed — unsupported, set the rate manually")),
            "DT" | "NC" => rate = m["settings"]["speed_change"].as_f64().unwrap_or(1.5),
            "HT" | "DC" => rate = m["settings"]["speed_change"].as_f64().unwrap_or(0.75),
            _ => {}
        }
    }
    if (0.1..=5.).contains(&rate) { (Some(rate), String::new()) } else { (None, format!("implausible rate {rate}")) }
}

/// tosu `play.mods` include Constant Speed (scroll changes removed).
pub fn constant_speed(mods: &Value) -> bool {
    array(mods).is_some_and(|a| a.iter().any(|m| m["acronym"].as_str().unwrap_or("").eq_ignore_ascii_case("CS")))
}

pub fn load() -> Value {
    let mut db: Value = std::fs::read(file()).ok().and_then(|b| serde_json::from_slice(&b).ok()).filter(Value::is_object).unwrap_or_else(|| json!({}));
    for (key, empty) in [("maps", json!({})), ("judgments", json!([])), ("scores", json!([]))] {
        if db.get(key).is_none() {
            db[key] = empty;
        }
    }
    db
}

pub fn save(db: &Value) -> std::io::Result<()> {
    use serde::Serialize;
    let target = file();
    std::fs::create_dir_all(target.parent().unwrap())?;
    let mut out = Vec::new();
    db.serialize(&mut serde_json::Serializer::with_formatter(&mut out, serde_json::ser::PrettyFormatter::with_indent(b" "))).map_err(std::io::Error::other)?;
    let tmp = target.with_extension("json.part");
    std::fs::write(&tmp, out)?;
    std::fs::rename(tmp, target)
}

pub fn entry_id(sha: &str, rate: f64) -> String {
    format!("{sha}@{rate:.2}")
}

/// Stored path if it exists here, else the same sha in this machine's lazer store.
pub fn chart_path(m: &Value) -> Option<PathBuf> {
    let stored = Path::new(m["path"].as_str().unwrap_or(""));
    if stored.is_file() {
        return Some(stored.to_path_buf());
    }
    paths::local_file(m["sha256"].as_str()?, &paths::lazer_data())
}

/// Record one of the user's own results (from tosu's results screen); false if known.
pub fn add_score(db: &mut Value, score: &Value, path: &str, store: &Db) -> rusqlite::Result<bool> {
    if !score_allowed(store, score, None, false)? {
        return Ok(false);
    }
    const SAME: [&str; 5] = ["sha256", "score", "accuracy", "max_combo", "played"];
    let scores = db["scores"].as_array_mut().expect("load made it a list");
    let id = score.get("id").filter(|v| v.as_str().is_some_and(|s| !s.is_empty()));
    if scores.iter().any(|s| SAME.iter().all(|k| s.get(*k) == score.get(*k)) || (id.is_some() && s.get("id") == id)) {
        return Ok(false);
    }
    let mut row = score.clone();
    row["path"] = json!(path);
    row["seen"] = json!(local_stamp()[..10]);
    scores.push(row);
    Ok(true)
}

pub fn label(m: &Value) -> String {
    format!("{} [{}] · {}K · {:.2}×", m["title"].as_str().unwrap_or(""), m["version"].as_str().unwrap_or(""), m["keys"], m["rate"].as_f64().unwrap_or(0.))
}
