//! The installed library and what a shown list entry holds.
use ms_data::paths;
use ms_data::store::{kv_get, Db, Result};
use ms_model::predict::{Feats, HashMap, Prediction};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub const SUPPORTED_KEYS: std::ops::RangeInclusive<i64> = 4..=10;

/// A chart of the library, or a website chart in the same shape.
#[derive(Clone, Debug, Default)]
pub struct Inst {
    pub sha256: String,
    pub beatmap_id: Option<i64>,
    pub set_id: Option<i64>,
    pub md5: Option<String>,
    pub status: Option<i64>,
    pub keys: i64,
    pub od: Option<f64>,
    pub length: Option<f64>,
    pub title: String,
    pub artist: String,
    pub version: String,
    pub online_md5: Option<String>,
    pub audio_sha: Option<String>,
    pub audio_required: bool,
    pub creator: Option<String>,
    pub path: Option<PathBuf>,
    pub installed: bool,
    pub playcount: Option<i64>,
    pub convert_of: Option<String>, // the source chart's sha of a keymode convert
    pub source_keys: i64,
    pub source_web: bool,
}

impl Inst {
    pub fn name(&self) -> String {
        format!("{} - {} [{}]", self.artist, self.title, self.version)
    }

    /// recommend.event_key
    pub fn identity(&self) -> String {
        self.md5.as_ref().filter(|m| !m.is_empty()).map_or_else(|| format!("sha:{}", self.sha256), |m| format!("md5:{m}"))
    }
}

const COLUMNS: &str = "i.sha256,i.beatmap_id,i.set_id,i.md5,i.status,i.keys,i.od,i.length,i.title,i.artist,i.version,\
                       l.online_md5,l.audio_sha,l.audio_required,l.creator";

fn row(r: &rusqlite::Row<'_>) -> rusqlite::Result<Inst> {
    let text = |i: usize| r.get::<_, Option<String>>(i).map(Option::unwrap_or_default);
    Ok(Inst { sha256: r.get(0)?, beatmap_id: r.get(1)?, set_id: r.get(2)?, md5: r.get(3)?, status: r.get(4)?, keys: r.get::<_, Option<i64>>(5)?.unwrap_or(0), od: r.get(6)?,
              length: r.get(7)?, title: text(8)?, artist: text(9)?, version: text(10)?, online_md5: r.get(11)?, audio_sha: r.get(12)?,
              audio_required: r.get::<_, Option<i64>>(13)?.unwrap_or(0) != 0, creator: r.get(14)?, path: None, installed: true, playcount: None, ..Inst::default() })
}

fn select(db: &Db, tail: &str, args: &[&dyn rusqlite::ToSql]) -> Result<Vec<Inst>> {
    let mut q = db.prepare_cached(&format!("SELECT {COLUMNS} FROM installed i LEFT JOIN installed_local l USING(sha256){tail}"))?;
    let rows = q.query_map(args, row)?;
    rows.collect()
}

/// {md5: chart} of the library.
pub fn installed_by_md5(db: &Db) -> Result<HashMap<String, Arc<Inst>>> {
    Ok(select(db, "", &[])?.into_iter().filter_map(|i| Some((i.md5.clone().filter(|m| !m.is_empty())?, Arc::new(i)))).collect())
}

pub fn one_by_md5(db: &Db, md5: &str) -> Result<Option<Inst>> {
    Ok(select(db, " WHERE i.md5=?", &[&md5])?.into_iter().next())
}

pub fn one_by_sha(db: &Db, sha: &str) -> Result<Option<Inst>> {
    Ok(select(db, " WHERE i.sha256=?", &[&sha])?.into_iter().next())
}

pub fn local_shas(db: &Db) -> Result<std::collections::HashSet<String>> {
    let mut q = db.prepare("SELECT sha256 FROM installed")?;
    let rows = q.query_map([], |r| r.get(0))?;
    rows.collect()
}

pub use ms_data::paths::local_origin;

/// A manifest is machine-local, even when portable feature caches are copied. → (the library's charts on disk, error)
pub fn library(db: &Db, client: &str) -> Result<(Vec<Inst>, String)> {
    let (host, root) = local_origin(client);
    if let Some(origin) = kv_get(db, "manifest_origin") {
        if origin["host"].as_str() != Some(&host) || origin["root"].as_str().map(Path::new) != Some(root.as_path()) {
            return Ok((Vec::new(), "Library belongs to another location — import local maps".to_string()));
        }
    }
    // Exhausted before the file checks, which can wait on disk while another connection needs to commit a live event.
    let all = select(db, "", &[])?;
    let out = all.into_iter().filter(|d| {
        let file = paths::local_file(&d.sha256, &root);
        if !SUPPORTED_KEYS.contains(&d.keys) || !file.is_some_and(|p| p.is_file()) {
            return false;
        }
        let audio = d.audio_sha.as_deref().and_then(|a| paths::local_file(a, &root));
        !d.audio_required || audio.is_some_and(|p| p.is_file())
    }).collect();
    Ok((out, kv_get(db, "manifest_error").and_then(|v| v.as_str().map(str::to_string)).unwrap_or_default()))
}

/// One list entry of any tab: chart, prediction, why it is offered.
#[derive(Clone, Default)]
pub struct Pick {
    pub mode: String,
    pub purpose: String,
    pub selection: String,
    pub sha: Option<String>,
    pub md5: Option<String>,
    pub bid: Option<i64>,
    pub title: String,
    pub artist: Option<String>,
    pub song: Option<String>,
    pub version: Option<String>,
    pub creator: Option<String>,
    pub keys: i64,
    pub rate: f64,
    pub var: String,
    pub installed: bool,
    pub online_match: bool,
    pub ranked: bool,
    pub nps: Option<f64>,
    pub length: f64,
    pub skill: Option<String>,
    pub description: String,
    pub family: Option<String>,
    pub group: Option<String>,
    pub profile: Vec<f64>,
    pub playcount: Option<i64>,
    pub value: f64,
    pub weight: f64,
    pub category_share: f64,
    pub why: String,
    pub practice_skills: Vec<String>,
    pub matched_skills: Vec<String>,
    pub practice_description: String,
    pub pred: Prediction,            // the display target
    pub pp_pred: Option<Prediction>, // the PP target, for a recorded play
    pub gain: Option<f64>,
    pub p_up: Option<f64>,
    pub best: Option<f64>,
    pub pp_lo: Option<f64>,
    pub pp_hi: Option<f64>,
    pub pp_mid: Option<f64>,
    pub best_rank: i64,
    pub push_only: bool,
    pub challenge: f64,
    pub stars: Option<f64>,
    pub scale: Option<f64>, // pp_scale of the chart-rate
    pub fresh: f64,
    pub offer_id: Option<i64>,
    pub f: Option<Arc<Feats>>,
    pub generated_at: f64,
    pub convert_of: Option<String>,
    pub source_keys: i64,
    pub source_web: bool,
    pub set_id: Option<i64>,
}

impl Pick {
    /// Which how-often slider the map answers to (ms_model::choices).
    pub fn stream(&self) -> u8 {
        if self.convert_of.is_some() { ms_model::choices::CONVERT } else if self.installed { ms_model::choices::REAL } else { ms_model::choices::WEB }
    }

    /// recommend.event_key
    pub fn identity(&self) -> Option<String> {
        ms_data::events::event_key(&serde_json::json!({"md5": self.md5, "sha": self.sha, "bid": self.bid}))
    }

    pub fn chart(&self) -> &str {
        self.md5.as_deref().or(self.sha.as_deref()).unwrap_or("")
    }

    /// Newer features' list fields, the `why` line included. → whether anything changed.
    pub fn refresh(&mut self, m: &ms_model::choices::Metadata, practice: bool) -> bool {
        let same = self.description == m.description && self.family.as_deref() == Some(m.family.as_str()) && self.group.as_deref() == Some(m.group.as_str()) && self.nps == Some(m.nps)
            && self.length == m.length && self.profile == m.profile && (!practice || (self.matched_skills == m.matched && self.practice_description == m.practice));
        if same {
            return false;
        }
        let headline = |nps: f64, skills: &str| if practice { skills.to_string() } else { format!("{nps:.1} raw NPS") };
        let old = format!("{} · {} · ", headline(self.nps.unwrap_or(0.), &self.practice_description), self.description);
        if let Some(tail) = self.why.strip_prefix(&old) {
            self.why = format!("{} · {} · {tail}", headline(m.nps, &m.practice), m.description);
        }
        self.skill = Some(m.description.clone());
        self.description.clone_from(&m.description);
        self.family = Some(m.family.clone());
        self.group = Some(m.group.clone());
        self.nps = Some(m.nps);
        self.length = m.length;
        self.profile.clone_from(&m.profile);
        self.matched_skills.clone_from(&m.matched);
        self.practice_description.clone_from(&m.practice);
        true
    }
}

pub fn rate_label_pp(rate: f64) -> String {
    let r = ms_calc::py::round_digits(rate, 2);
    if r == 0.75 { "HT".into() } else if r == 1. { "NM".into() } else if r == 1.5 { "DT".into() } else { format!("{rate:.2}×") }
}

/// nps.rate_label
pub fn rate_label(rate: f64) -> String {
    format!("{} {rate:.2}×", if rate == 1. { "NM" } else if rate < 1. { "HT" } else { "DT" })
}

pub fn num(v: &Value, k: &str) -> Option<f64> {
    v.get(k).and_then(Value::as_f64)
}
