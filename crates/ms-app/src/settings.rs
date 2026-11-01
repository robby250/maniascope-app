//! ui.json: same file and keys as ever; observed rates are never persisted.
use ms_data::paths::config;
use ms_model::skills;
use ms_rec::builder::{normalize_focus, normalize_target, DEFAULT_TARGET};
use ms_rec::catalog::SUPPORTED_KEYS;
use ms_rec::converts::{default_sources, Setting};
use ms_rec::pp::PpSettings;
use ms_rec::worker::PpMix;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::PathBuf;

pub const MODES: [&str; 3] = ["pp", "nps", "skills"];

pub fn mode_name(m: &str) -> &'static str {
    match m { "pp" => "PP", "nps" => "NPS", _ => "Skills" }
}

#[derive(Clone, PartialEq, Debug)]
pub struct Display {
    pub rating: bool,
    pub nps: bool,
    pub dan: bool,
    pub ln: bool,
    pub timeline: bool,
    pub skills: usize,
}

pub struct Ui {
    pub mode: String,
    pub keys: HashMap<String, Vec<i64>>,
    pub keys_auto: BTreeSet<String>, // never chosen: follow the keymodes actually played
    pub practice: Vec<String>,
    pub skill_weights: HashMap<String, f64>,
    pub skill_avoid: HashMap<String, f64>,
    pub skill_other: f64,
    pub nps_focus: f64,
    pub web_bias: HashMap<String, f64>,
    pub web_weighted: HashMap<String, bool>,
    pub converts: HashMap<String, BTreeMap<i64, Setting>>, // tab → target keymode
    pub targets: HashMap<String, f64>,
    pub pp: PpSettings,
    pub mix: PpMix,
    pub display: Display,
    pub next_action: String,
    pub auto_next: bool,
    pub keep_above: bool,
    pub overlay: bool,
    pub details: bool,
    pub win: (f32, f32),
    pub details_win: (f32, f32),
    pub recs_win: (f32, f32),
    pub win_pos: Option<(f32, f32)>,
    pub recs_pos: Option<(f32, f32)>,
    pub recs_open: bool,
}

pub fn normalize_keys(v: &Value, default: &[i64]) -> Vec<i64> {
    match v {
        Value::Null => default.to_vec(),
        Value::String(s) if s == "all" => SUPPORTED_KEYS.collect(),
        Value::Array(a) => {
            let mut out: Vec<i64> = a.iter().filter_map(Value::as_i64).filter(|k| SUPPORTED_KEYS.contains(k)).collect();
            out.sort_unstable();
            out.dedup();
            out
        }
        _ => Vec::new(),
    }
}

fn file() -> PathBuf {
    config().join("ui.json")
}

fn pair(v: &Value, a: &str, b: &str, default: (f32, f32)) -> (f32, f32) {
    (v[a].as_f64().map_or(default.0, |x| x as f32), v[b].as_f64().map_or(default.1, |x| x as f32))
}

fn point(v: &Value) -> Option<(f32, f32)> {
    Some((v[0].as_f64()? as f32, v[1].as_f64()? as f32))
}

fn pp_settings(v: &Value) -> PpSettings {
    let d = PpSettings::default();
    let (f, b) = (|k: &str, x: f64| v[k].as_f64().unwrap_or(x), |k: &str, x: bool| v[k].as_bool().unwrap_or(x));
    PpSettings { min_gain: f("min_gain", d.min_gain), min_chance: f("min_chance", d.min_chance), ht: b("ht", d.ht), ht_chance: f("ht_chance", d.ht_chance), ht_gain: f("ht_gain", d.ht_gain), dt: b("dt", d.dt), push: f("push", d.push), cooldown_days: f("cooldown_days", d.cooldown_days), top100: b("top100", d.top100) }
}

fn pp_mix(v: &Value) -> PpMix {
    let d = PpMix::default();
    let f = |k: &str, x: f64| v[k].as_f64().unwrap_or(x);
    PpMix { on: v["on"].as_bool().unwrap_or(d.on), every: v["every"].as_f64().map_or(d.every, |x| x as usize), min_gain: f("min_gain", d.min_gain), chance: f("chance", d.chance), ht_chance: f("ht_chance", d.ht_chance) }
}

impl Ui {
    pub fn skill_tuning(&self) -> ms_model::skills::Tuning {
        ms_model::skills::Tuning { avoid: self.skill_avoid.clone(), other: self.skill_other }
    }

    pub fn load() -> Ui {
        let v: Value = std::fs::read(file()).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or(Value::Null);
        Ui::from_json(&v)
    }

    pub fn from_json(v: &Value) -> Ui {
        let mode = v["recommendation_mode"].as_str().filter(|m| MODES.contains(m)).unwrap_or("pp").to_string();
        let all: Vec<i64> = SUPPORTED_KEYS.collect();
        let keys = [("pp".to_string(), normalize_keys(&v["pp_keys"], &all)), ("nps".to_string(), normalize_keys(&v["nps_keys"], &[7])), ("skills".to_string(), normalize_keys(&v["skills_keys"], &[7]))].into_iter().collect();
        let keys_auto = ["nps", "skills"].into_iter().filter(|m| v[format!("{m}_keys")].is_null()).map(str::to_string).collect();
        let d = &v["display"];
        let flag = |k: &str| d[k].as_bool().unwrap_or(true);
        let selected: Vec<String> = v["practice_skills"].as_array().map_or_else(Vec::new, |a| a.iter().filter_map(|s| s.as_str().map(str::to_string)).collect());
        Ui {
            mode, keys, keys_auto, practice: skills::normalize(&selected), skill_weights: v["skill_weights"].as_object().map_or_else(HashMap::new, |o| o.iter().filter_map(|(k, x)| Some((k.clone(), x.as_f64()?))).collect()),
            skill_avoid: v["skill_avoid"].as_object().map_or_else(HashMap::new, |o| o.iter().filter_map(|(k, x)| Some((k.clone(), x.as_f64()?))).collect()), skill_other: v["skill_other"].as_f64().unwrap_or(0.),
            nps_focus: normalize_focus(v["nps_focus"].as_f64().unwrap_or(0.5)), web_bias: ["nps", "skills"].into_iter().map(|m| (m.to_string(), normalize_focus(v["web_bias"][m].as_f64().unwrap_or(0.5)))).collect(),
            web_weighted: ["nps", "skills"].into_iter().map(|m| (m.to_string(), v["web_weighted"][m].as_bool().unwrap_or(true))).collect(),
            converts: ["nps", "skills"].into_iter().map(|m| (m.to_string(), v["converts"][m].as_object().map_or_else(BTreeMap::new, |o| o.iter().filter_map(|(k, s)| {
                let target: i64 = k.parse().ok().filter(|k| SUPPORTED_KEYS.contains(k))?;
                let sources = s["sources"].as_array().map_or_else(|| default_sources(target), |a| a.iter().filter_map(|x| x.as_i64()).filter(|x| SUPPORTED_KEYS.contains(x) && *x != target).collect());
                Some((target, Setting { on: s["on"].as_bool().unwrap_or(false), sources, often: normalize_focus(s["often"].as_f64().unwrap_or(0.5)), weighted: s["weighted"].as_bool().unwrap_or(true) }))
            }).collect()))).collect(),
            targets: ["nps", "skills"].into_iter().map(|m| (m.to_string(), normalize_target(v[format!("{m}_acc_target")].as_f64().unwrap_or(DEFAULT_TARGET)))).collect(),
            pp: pp_settings(&v["pp_settings"]), mix: pp_mix(&v["pp_mix"]),
            display: Display { rating: flag("rating"), nps: flag("nps"), dan: flag("dan"), ln: flag("ln"), timeline: flag("timeline"), skills: d["skills"].as_u64().map_or(3, |x| x as usize) },
            next_action: "auto".to_string(), auto_next: v["auto_next"].as_bool().unwrap_or(true), keep_above: v["keep_above"].as_bool().unwrap_or(false), overlay: cfg!(windows) && v["overlay"].as_bool().unwrap_or(false),
            details: v["details"].as_bool().unwrap_or(false),
            // Defaults: the user's own sizes on G835LX (2026-10-30).
            win: pair(v, "w", "h", (442., 214.)), details_win: pair(v, "details_w", "details_h", (441., 663.)), recs_win: pair(v, "recs_w", "recs_h", (442., 420.)),
            win_pos: point(&v["pos"]), recs_pos: point(&v["recs_pos"]), recs_open: v["recs_open"].as_bool().unwrap_or(false),
        }
    }

    pub fn to_json(&self) -> Value {
        let keys = |m: &str| if self.keys_auto.contains(m) { Value::Null } else { json!(self.keys[m]) };
        let d = &self.display;
        let (p, m) = (&self.pp, &self.mix);
        json!({"mode": "auto", "recommendation_mode": self.mode, "next_action": self.next_action, "auto_next": self.auto_next, "keep_above": self.keep_above, "overlay": self.overlay,
               "pp_keys": self.keys["pp"], "nps_keys": keys("nps"), "skills_keys": keys("skills"), "practice_skills": self.practice, "skill_weights": self.skill_weights, "skill_avoid": self.skill_avoid, "skill_other": self.skill_other, "nps_focus": self.nps_focus, "web_bias": self.web_bias,
               "web_weighted": self.web_weighted, "converts": self.converts.iter().map(|(m, t)| (m.clone(), Value::Object(t.iter().map(|(k, s)| (k.to_string(), json!({"on": s.on, "sources": s.sources, "often": s.often, "weighted": s.weighted}))).collect()))).collect::<serde_json::Map<_, _>>(),
               "nps_acc_target": self.targets["nps"], "skills_acc_target": self.targets["skills"],
               "pp_settings": {"min_gain": p.min_gain, "min_chance": p.min_chance, "ht": p.ht, "ht_chance": p.ht_chance, "ht_gain": p.ht_gain, "dt": p.dt, "push": p.push, "cooldown_days": p.cooldown_days, "top100": p.top100},
               "pp_mix": {"on": m.on, "every": m.every, "min_gain": m.min_gain, "chance": m.chance, "ht_chance": m.ht_chance},
               "display": {"rating": d.rating, "nps": d.nps, "dan": d.dan, "ln": d.ln, "timeline": d.timeline, "skills": d.skills}, "details": self.details,
               "w": self.win.0, "h": self.win.1, "details_w": self.details_win.0, "details_h": self.details_win.1, "recs_w": self.recs_win.0, "recs_h": self.recs_win.1,
               "recs_pos": self.recs_pos.map(|p| [p.0, p.1]), "pos": self.win_pos.map(|p| [p.0, p.1]), "recs_open": self.recs_open})
    }

    pub fn save(&self) {
        let path = file();
        let tmp = path.with_extension("json.tmp");
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        if std::fs::write(&tmp, self.to_json().to_string()).is_ok() {
            let _ = std::fs::rename(tmp, path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_python_ui_file_keeps_its_choices_through_a_round_trip() {
        let old = json!({"mode": "manual", "recommendation_mode": "skills", "pp_keys": [4, 7], "nps_keys": [7], "skills_keys": null, "practice_skills": ["!ln/release", "chordstream"], "skill_weights": {"chordstream": 2.0},
                         "nps_focus": 0.25, "web_bias": {"nps": 0.1, "skills": 0.9}, "nps_acc_target": 0.93, "pp_settings": {"min_gain": 2.0, "ht": false}, "pp_mix": {"on": true, "every": 3}, "display": {"nps": false, "skills": 5}, "rate": 1.05, "w": 500, "h": 300, "pos": [10.0, 20.0], "recs_open": true});
        let ui = Ui::from_json(&old);
        assert_eq!(ui.keys["pp"], [4, 7]);
        assert!(ui.keys_auto.contains("skills") && !ui.keys_auto.contains("nps"));
        assert!(!ui.pp.ht && ui.mix.on && !ui.display.nps);
        let back = ui.to_json();
        assert_eq!((back["mode"].as_str(), back["next_action"].as_str()), (Some("auto"), Some("auto")));
        for k in ["recommendation_mode", "pp_keys", "skills_keys", "practice_skills", "skill_weights", "nps_focus", "recs_open", "pos"] {
            assert_eq!(back[k], old[k], "{k}");
        }
        assert_eq!(Ui::from_json(&back).to_json(), back);
    }
}
