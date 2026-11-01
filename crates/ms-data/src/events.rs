//! Reading the event log the way sessions need it: exposures of the last two weeks, exact chart identities.
use crate::store::{acc_lazer, parse_stats, tracked_events, Db, Event, Result};
use rusqlite::OptionalExtension;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

pub const HISTORY_S: f64 = 3. * 86400.;       // events that shape form and predictions
pub const EXPOSURE_S: f64 = 14. * 86400.;     // offers and plays that keep a map back
const EXPOSURE_KINDS: [&str; 9] = ["offer", "next", "open", "skip", "start", "finish", "abort", "reset", "void"];
const EXPOSURE_FIELDS: [&str; 8] = ["md5", "sha", "bid", "mode", "length", "played_seconds", "progress", "start_id"];

pub fn positive_id(v: &Value) -> Option<i64> {
    let n = match v {
        Value::Number(n) => n.as_i64().or_else(|| n.as_f64().filter(|x| x.fract() == 0.).map(|x| x as i64)),
        Value::String(s) => s.trim().parse().ok(),
        _ => None,
    };
    n.filter(|&n| n > 0)
}

fn text(v: Option<&Value>) -> Option<String> {
    match v {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) if s.is_empty() => None,
        Some(Value::String(s)) => Some(s.clone()),
        Some(other) => Some(other.to_string()),
    }
}

/// Exact revision across rates; a sentinel online id is never an identity.
pub fn event_key(c: &Value) -> Option<String> {
    if let Some(md5) = text(c.get("md5")) {
        return Some(format!("md5:{md5}"));
    }
    if let Some(sha) = text(c.get("sha")).or_else(|| text(c.get("sha256"))) {
        return Some(format!("sha:{sha}"));
    }
    c.get("bid").and_then(positive_id).map(|b| format!("bid:{b}"))
}

pub fn legacy_key(value: &str) -> Option<String> {
    if value.starts_with("md5:") || value.starts_with("sha:") || value.starts_with("bid:") {
        return Some(value.to_string());
    }
    match value.len() {
        32 => Some(format!("md5:{value}")),
        64 => Some(format!("sha:{value}")),
        _ => positive_id(&json!(value)).map(|b| format!("bid:{b}")),
    }
}

fn exposures(db: &Db, start: f64, until: f64) -> Result<Vec<Event>> {
    let mut out = tracked_events(db, start, &EXPOSURE_KINDS, Some(until), Some(&EXPOSURE_FIELDS))?;
    for e in &mut out {
        if let Some(o) = e.info.as_object_mut() {
            o.retain(|_, v| !v.is_null());
        }
    }
    Ok(out)
}

/// Events since `since`, plus the older exposures back to two weeks, with `beatmap` resolved to the exact chart.
pub fn load_events(db: &Db, since: f64) -> Result<Vec<Arc<Event>>> {
    let hour = (since / 3600.).floor() * 3600.;
    let mut events = exposures(db, hour - (EXPOSURE_S - HISTORY_S), hour)?;
    events.extend(exposures(db, hour, since)?);
    events.extend(tracked_events(db, since, &[], None, None)?);
    resolve(db, &mut events)?;
    // A start stores the chart's whole analysis; a session only ever reads its sk, and 14 days of it is tens of MB.
    for e in &mut events {
        if let Some(f) = e.info.get_mut("features").filter(|f| f.is_object()) {
            *f = json!({"sk": f.get("sk").cloned().unwrap_or(Value::Null)});
        }
    }
    Ok(events.into_iter().map(Arc::new).collect())
}

/// Events from `since` on, no older exposures.
pub fn events_since(db: &Db, since: f64) -> Result<Vec<Arc<Event>>> {
    let mut events = tracked_events(db, since, &[], None, None)?;
    resolve(db, &mut events)?;
    Ok(events.into_iter().map(Arc::new).collect())
}

fn resolve(db: &Db, events: &mut [Event]) -> Result<()> {
    let offers: HashMap<i64, Value> = events.iter().filter(|e| e.kind == "offer").map(|e| (e.id, e.info.clone())).collect();
    for e in events.iter_mut() {
        if e.kind == "finish" && e.info.get("y_lazer").is_none_or(Value::is_null) {
            if let Some(key) = text(e.info.get("key")) {
                let stats: Option<String> = db.query_row("SELECT stats FROM scores WHERE key=?", [key], |r| r.get(0)).optional()?.flatten();
                if let Some(s) = stats.as_deref().and_then(parse_stats) {
                    let a = acc_lazer(&s).min(0.999);
                    e.info["y_lazer"] = json!((1. - a).max(1e-3).ln());
                }
            }
        }
        if e.kind == "start" {
            let offered = e.info.get("offer_id").and_then(Value::as_i64).and_then(|id| offers.get(&id)).and_then(|o| o.get("rate")).cloned();
            if let (Some(rate), true) = (offered, e.info.get("offered_rate").is_none()) {
                e.info["offered_rate"] = rate;
            }
        }
    }
    let (mut sha_aliases, mut starts, mut observed) = (HashMap::<String, Option<String>>::new(), HashMap::new(), HashMap::<String, HashSet<String>>::new());
    for e in events.iter_mut() {
        if let (Some(sha), None) = (text(e.info.get("sha")), text(e.info.get("md5"))) {
            if !sha_aliases.contains_key(&sha) {
                let md5 = db.query_row("SELECT md5 FROM installed WHERE sha256=?", [&sha], |r| r.get::<_, Option<String>>(0)).optional()?.flatten();
                sha_aliases.insert(sha.clone(), md5);
            }
            if let Some(md5) = sha_aliases[&sha].clone() {
                e.info["md5"] = json!(md5);
            }
        }
        if let Some(key) = event_key(&json!({"md5": e.info.get("md5"), "sha": e.info.get("sha")})) {
            if e.kind == "start" {
                starts.insert(e.id, key.clone());
            }
            if let Some(b) = e.beatmap.as_ref().filter(|b| positive_id(&json!(b)).is_some()) {
                observed.entry(b.clone()).or_default().insert(key);
            }
        }
    }
    let mut aliases: HashMap<String, Option<String>> = observed.iter().filter(|(_, v)| v.len() == 1)
        .map(|(old, v)| (old.clone(), v.iter().next().cloned())).collect();
    for e in events.iter_mut() {
        let start = e.info.get("start_id").and_then(Value::as_i64).and_then(|s| starts.get(&s));
        let own = event_key(&json!({"md5": e.info.get("md5"), "sha": e.info.get("sha")}));
        if let (true, Some(key)) = (e.kind == "finish" || e.kind == "abort", start) {
            e.beatmap = Some(key.clone());
        } else if own.is_some() {
            e.beatmap = own;
        } else if let Some(old) = e.beatmap.clone() {
            if !aliases.contains_key(&old) {
                let mut matches: Vec<String> = Vec::new();
                if let Some(bid) = positive_id(&json!(old)) {
                    let mut q = db.prepare_cached("SELECT DISTINCT md5 FROM installed WHERE beatmap_id=? AND md5 IS NOT NULL LIMIT 2")?;
                    matches = q.query_map([bid], |r| r.get(0))?.collect::<Result<_>>()?;
                }
                aliases.insert(old.clone(), if matches.len() == 1 { Some(format!("md5:{}", matches[0])) } else { legacy_key(&old) });
            }
            e.beatmap = aliases[&old].clone();
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identities() {
        assert_eq!(event_key(&json!({"md5": "a", "sha": "b"})).as_deref(), Some("md5:a"));
        assert_eq!(event_key(&json!({"sha256": "b"})).as_deref(), Some("sha:b"));
        assert_eq!(event_key(&json!({"bid": -1})), None);
        assert_eq!(event_key(&json!({"bid": 5})).as_deref(), Some("bid:5"));
        assert_eq!(legacy_key("12").as_deref(), Some("bid:12"));
        assert_eq!(legacy_key(&"a".repeat(32)).as_deref(), Some(format!("md5:{}", "a".repeat(32)).as_str()));
    }
}
