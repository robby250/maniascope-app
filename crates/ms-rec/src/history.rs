//! Saved results and their frozen predictions (recommend.play_history, recover_results).
use crate::evidence::y_of;
use crate::pp::PESSIMISM;
use ms_data::store::{acc_lazer, event_info, log_event, now, parse_stats, timestamp, tracking_allowed, Db, Result, Stats};
use rusqlite::params;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};

pub fn client_name(client: &str) -> String {
    format!("osu!{client}")
}

/// Before tosu has named one: lazer when its database is here, else an installed stable.
pub fn default_client() -> &'static str {
    use ms_data::paths;
    if paths::lazer_data().join("client.realm").is_file() || !paths::stable_data().join("osu!.db").is_file() { "lazer" } else { "stable" }
}

pub struct Shown {
    pub client: String,
    pub miss_ratio: f64,
}

impl Shown {
    /// A lazer-unit accuracy (predictions, targets inside the model) as the active client displays it.
    pub fn acc(&self, acc: f64) -> f64 {
        if self.client == "stable" { 1. - (1. - acc) * self.miss_ratio } else { acc }
    }

    /// shown acc's inverse: a displayed accuracy (the target slider) in the model's lazer units.
    pub fn model(&self, acc: f64) -> f64 {
        if self.client == "stable" { 1. - (1. - acc) / self.miss_ratio } else { acc }
    }

    /// A lazer-unit accuracy difference (points) in the active client's units.
    pub fn points(&self, points: f64) -> f64 {
        if self.client == "stable" { points * self.miss_ratio } else { points }
    }

    /// A play's accuracy as the active client shows it: on stable, stable's formula on the same judgements.
    pub fn play(&self, stats: &Stats, lazer: f64) -> f64 {
        if self.client == "stable" { ms_data::store::acc_stable(stats) } else { lazer }
    }
}

/// Pair imported results only with a matching, already recorded attempt. Never predict after a play.
pub fn recover_results(db: &Db) -> Result<usize> {
    let current = now();
    let starts: Vec<(i64, f64, Option<String>, Value)> = db.prepare("SELECT id,t,beatmap,info FROM events WHERE kind='start' AND t>=? ORDER BY t, id")?
        .query_map([current - 3. * 86400.], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get::<_, String>(3)?)))?
        .map(|x| x.map(|(a, b, c, d)| (a, b, c, serde_json::from_str(&d).unwrap_or(Value::Null)))).collect::<Result<_>>()?;
    let mut finished: HashSet<String> = db.prepare("SELECT json_extract(info,'$.key') FROM events WHERE kind='finish'")?
        .query_map([], |r| r.get::<_, Option<String>>(0))?.filter_map(|k| k.ok().flatten()).collect();
    let mut added = 0;
    for (i, (id, t, beatmap, info)) in starts.iter().enumerate() {
        let next = starts.get(i + 1).map_or(current, |s| s.1);
        let end = next.min(t + info["length"].as_f64().filter(|&l| l != 0.).unwrap_or(900.) + 120.);
        let mut wanted: Vec<String> = info["mods"].as_array().map_or_else(Vec::new, |m| m.iter().filter_map(|x| x.as_str().map(str::to_string)).collect());
        wanted.sort();
        let scores: Vec<(String, Option<f64>, Option<String>, Option<String>, f64, Option<String>)> = db.prepare_cached("SELECT key,rate,mods,played,acc,stats FROM scores WHERE sha256=?")?
            .query_map([info["sha"].as_str()], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?)))?.collect::<Result<_>>()?;
        let matches: Vec<_> = scores.into_iter().filter(|r| {
            let mods: Value = r.2.as_deref().and_then(|m| serde_json::from_str(m).ok()).unwrap_or(json!([]));
            let mut names: Vec<String> = mods.as_array().map_or_else(Vec::new, |m| m.iter().map(|x| x["acronym"].as_str().unwrap_or("").to_string()).collect());
            names.sort();
            let played = r.3.as_deref().and_then(timestamp).unwrap_or(0.);
            !finished.contains(&r.0) && r.1 == info["rate"].as_f64() && names == wanted && *t < played && played <= end
        }).collect();
        let [r] = matches.as_slice() else { continue };
        let played = r.3.as_deref().and_then(timestamp).unwrap_or(0.);
        if !tracking_allowed(db, Some(played), Some(*t))? || info["predicted_at"].as_f64().unwrap_or(*t) > played {
            continue;
        }
        let y_lazer = r.5.as_deref().and_then(parse_stats).map_or(0., |s| y_of(acc_lazer(&s)));
        log_event(db, "finish", beatmap.as_deref(), Some(played), &json!({"key": r.0, "start_id": id, "recovered": true, "y": y_of(r.4), "y_lazer": y_lazer}))?;
        db.execute("UPDATE scores SET fresh=1 WHERE key=?", [&r.0])?;
        finished.insert(r.0.clone());
        added += 1;
    }
    Ok(added)
}

#[derive(Default)]
pub struct Played {
    pub key: String,
    pub played: Option<String>,
    pub title: String,
    pub rate: Option<f64>,
    pub keys: Option<i64>,
    pub actual: f64,
    pub expected: Option<f64>,
    pub z: Option<f64>,
    pub prediction: Value,
    pub sha256: Option<String>,
    pub stats: Stats,
    pub pauses: Option<Value>,
    pub recovered: bool,
    pub mods: Value,
    pub start_id: Option<i64>,
    pub start_t: Option<f64>,
    pub finish_id: Option<i64>,
    pub finish_t: Option<f64>,
    pub forecast_at: Option<f64>,
    pub forecast_status: &'static str, // saved, missing or identity_or_time_mismatch
    pub prediction_valid: Option<bool>,
}

struct Ev {
    id: i64,
    t: f64,
    kind: String,
    beatmap: Option<String>,
    info: Value,
}

/// Saved results and their frozen predictions; unobserved historical plays have no prediction.
pub fn play_history(db: &Db, limit: usize, outliers: bool, until: Option<f64>) -> Result<Vec<Played>> {
    let mut pauses: HashMap<String, Value> = HashMap::new();
    let has: bool = db.query_row("SELECT 1 FROM sqlite_master WHERE type='table' AND name='replay_evidence'", [], |_| Ok(true)).unwrap_or(false);
    if has {
        for row in db.prepare("SELECT score_key,pauses FROM replay_evidence")?.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, Option<String>>(1)?)))? {
            let (k, p) = row?;
            if let Some(v) = p.and_then(|p| serde_json::from_str(&p).ok()) {
                pauses.insert(k, v);
            }
        }
    }
    let events: Vec<Ev> = db.prepare("SELECT id,t,kind,beatmap,info FROM events WHERE kind IN ('start','finish','void') AND (? IS NULL OR t<=?) ORDER BY t,id")?
        .query_map(params![until, until], |r| Ok(Ev { id: r.get(0)?, t: r.get(1)?, kind: r.get(2)?, beatmap: r.get(3)?, info: event_info(&r.get::<_, String>(4)?) }))?.collect::<Result<_>>()?;
    let starts: HashMap<i64, usize> = events.iter().enumerate().filter(|x| x.1.kind == "start").map(|x| (x.1.id, x.0)).collect();
    let (mut paired, mut pending): (HashMap<String, (usize, usize)>, Option<usize>) = Default::default();
    for (i, e) in events.iter().enumerate() {
        match e.kind.as_str() {
            "start" => pending = Some(i),
            "void" => pending = None,
            _ => {
                let start = if e.info.get("start_id").is_some() { e.info["start_id"].as_i64().and_then(|s| starts.get(&s).copied()) } else { pending };
                if let (Some(s), Some(key)) = (start, e.info["key"].as_str()) {
                    if events[s].beatmap == e.beatmap {
                        paired.insert(key.to_string(), (s, i));
                    }
                }
                pending = None;
            }
        }
    }
    type Row = (String, Option<String>, Option<String>, Option<f64>, f64, Option<String>, Option<String>, Option<String>, Option<String>, Option<i64>, Option<String>);
    let scores: Vec<Row> = db.prepare("SELECT s.key,s.played,s.stats,s.rate,s.acc,s.sha256,i.artist,i.title,i.version,i.keys,s.mods FROM scores s LEFT JOIN installed i ON i.sha256=s.sha256 WHERE s.src!='public' ORDER BY s.played DESC")?
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?, r.get(6)?, r.get(7)?, r.get(8)?, r.get(9)?, r.get(10)?)))?.collect::<Result<_>>()?;
    let mut out = Vec::new();
    for (key, played, stats, rate, acc, sha, artist, title, version, keys, mods) in scores {
        let t = played.as_deref().and_then(timestamp);
        if until.is_some_and(|u| t.is_none_or(|t| t > u)) {
            continue;
        }
        let stats = stats.as_deref().and_then(parse_stats).unwrap_or([0; 6]);
        let actual = acc_lazer(&stats);
        let pair = paired.get(&key).map(|&(s, e)| (&events[s], &events[e]));
        let mut info = pair.map_or(Value::Null, |p| p.0.info.clone());
        let finish = pair.map(|p| &p.1.info);
        let mut forecast_status = if info["expected"].is_null() { "missing" } else { "saved" };
        let mut forecast_at = None;
        if let Some((start, end)) = pair {
            // A stored null or non-number is not "absent": it stays an unusable time.
            forecast_at = match info.get("predicted_at") { Some(v) => v.as_f64(), None => Some(start.t) };
            if info["sha"].as_str() != sha.as_deref() || info["rate"].as_f64() != rate || forecast_at.is_none_or(|f| !f.is_finite() || t.is_some_and(|t| f > t))
                || until.is_some_and(|u| start.t > end.t || forecast_at.is_some_and(|f| f > u)) {
                info = Value::Null;
                forecast_status = "identity_or_time_mismatch";
            }
        }
        let number = |k: &str| info[k].as_f64();
        let z = match (number("display_mu"), number("display_sd").filter(|&s| s != 0.), number("mu"), number("sd").filter(|&s| s != 0.)) {
            (Some(m), Some(s), _, _) => Some((m - y_of(actual)) / s),
            (_, _, Some(m), Some(s)) => Some((number("shown_mu").unwrap_or_else(|| m + PESSIMISM * number("sdm").unwrap_or(0.)) - y_of(acc)) / s),
            _ => None,
        };
        let name = title.as_ref().filter(|t| !t.is_empty()).map(|ti| format!("{} - {ti} [{}]", artist.as_deref().unwrap_or(""), version.as_deref().unwrap_or("")))
            .or_else(|| info["title"].as_str().map(str::to_string)).or_else(|| sha.clone()).unwrap_or_default();
        out.push(Played { mods: mods.as_deref().and_then(|m| serde_json::from_str(m).ok()).unwrap_or_else(|| json!([])), start_id: pair.map(|p| p.0.id), start_t: pair.map(|p| p.0.t), finish_id: pair.map(|p| p.1.id),
                          finish_t: pair.map(|p| p.1.t), forecast_at, forecast_status, prediction_valid: finish.and_then(|f| f["prediction_valid"].as_bool()), pauses: pauses.get(&key).cloned(), recovered: finish.is_some_and(|f| f["recovered"].as_bool().unwrap_or(false)), key, played, title: name, rate, keys, actual, expected: number("expected"), z,
                          prediction: info.clone(), sha256: sha, stats });
    }
    out.sort_by(|a, b| {
        let at = |p: &Played| p.played.as_deref().and_then(timestamp).unwrap_or(0.);
        at(b).total_cmp(&at(a))
    });
    if outliers {
        out.retain(|r| r.z.is_some_and(|z| z.abs() >= 1.5));
        out.sort_by(|a, b| b.z.unwrap().abs().total_cmp(&a.z.unwrap().abs()));
    }
    if limit != 0 {
        out.truncate(limit);
    }
    Ok(out)
}

pub fn history_text(rows: &[Played], shown: &Shown) -> String {
    use chrono::{Local, TimeZone};
    let lines: Vec<String> = rows.iter().map(|r| {
        let when = r.played.as_deref().and_then(timestamp).and_then(|t| Local.timestamp_opt(t as i64, 0).single()).map_or_else(|| "unknown time".to_string(), |t| t.format("%m-%d %H:%M").to_string());
        let rate = r.rate.map_or_else(|| "unknown rate".to_string(), |x| format!("{x:.2}×"));
        let expected = r.expected.map_or_else(|| "expected % not recorded".to_string(), |e| format!("expected {:.2}%", 100. * shown.acc(e)));
        let z = r.z.map_or_else(String::new, |z| format!(" · {z:+.2}σ ({} expected)", if z > 0. { "above" } else { "below" }));
        let n = r.pauses.as_ref().and_then(Value::as_array).map_or(0, Vec::len);
        let pause = if n > 0 { format!(" · {n} play pause{} (duration unknown)", if n != 1 { "s" } else { "" }) } else { String::new() };
        let recovered = if r.recovered { format!(" · recovered from {}", client_name(&shown.client)) } else { String::new() };
        format!("{when} · {} · {rate}\n  {:.2}% · {expected}{z}{recovered}{pause}", r.title, 100. * shown.play(&r.stats, r.actual))
    }).collect();
    if lines.is_empty() { "No plays match this view yet.".to_string() } else { lines.join("\n\n") }
}
