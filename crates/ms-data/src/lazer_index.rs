//! An index of lazer's hash-named file store: the string lazer prints in its log ("Artist - Title (Creator) [Version]")
//! → the .osu path, for both romanised and unicode metadata. Only used when tosu is not running. Cached in
//! ~/.cache/maniascope/index.json, rebuilt when a selected map is missing from it.
use crate::paths;
use ms_calc::background::background_map;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let path = entry.path();
        if entry.file_type().is_ok_and(|t| t.is_dir()) { walk(&path, out) } else { out.push(path) }
    }
}

/// The [Metadata] fields of an .osu, if it is one.
pub fn metadata(path: &Path) -> Option<HashMap<String, String>> {
    let data = std::fs::read(path).ok()?;
    if !data.starts_with(b"osu file format v") {
        return None;
    }
    let text = String::from_utf8_lossy(&data);
    let (mut md, mut inside): (HashMap<String, String>, bool) = Default::default();
    for line in text.lines() {
        if line.starts_with('[') {
            if inside {
                break;
            }
            inside = line == "[Metadata]";
        } else if inside {
            if let Some((k, v)) = line.split_once(':') {
                md.insert(k.to_string(), v.trim().to_string());
            }
        }
    }
    Some(md)
}

/// The display strings of an .osu, if it is one.
fn names(path: &Path) -> Option<Vec<String>> {
    let md = metadata(path)?;
    let get = |k: &str| md.get(k).map_or("", String::as_str);
    let or = |k: &str, b: &str| md.get(k).map(String::as_str).filter(|v| !v.is_empty()).unwrap_or(b).to_string();
    let (title, artist) = (get("Title"), get("Artist"));
    let name = |a: &str, t: &str| format!("{a} - {t} ({}) [{}]", get("Creator"), get("Version"));
    Some(vec![name(artist, title), name(&or("ArtistUnicode", artist), &or("TitleUnicode", title))])
}

pub fn build_index(store: &Path) -> HashMap<String, String> {
    let mut files = Vec::new();
    walk(store, &mut files);
    let mut index = HashMap::new();
    for (path, found) in files.iter().zip(background_map(&files, 0, |p| Ok(names(p)))) {
        for name in found.ok().flatten().into_iter().flatten() {
            index.entry(name).or_insert_with(|| path.to_string_lossy().into_owned());
        }
    }
    index
}

/// Load the cached index, else (re)build and cache it. A cache written under another alias of the same store is
/// re-pointed instead of rebuilt.
pub fn load_or_build(force: bool) -> HashMap<String, String> {
    let store = std::env::var("LAZER_FILES").map_or_else(|_| paths::lazer_data().join("files"), PathBuf::from);
    let store = std::fs::canonicalize(&store).unwrap_or(store);
    let cache = paths::cache().join("index.json");
    let root = store.to_string_lossy().into_owned();
    if !force {
        if let Some(cached) = std::fs::read(&cache).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok()) {
            if let (Some(old), Some(index)) = (cached["store"].as_str(), cached["index"].as_object()) {
                let index: HashMap<String, String> = index.iter().filter_map(|(k, v)| {
                    let p = v.as_str()?;
                    Some((k.clone(), p.strip_prefix(old).map_or_else(|| p.to_string(), |rest| format!("{root}{rest}"))))
                }).collect();
                if index.values().take(20).any(|p| Path::new(p).exists()) {
                    return index;
                }
            }
        }
    }
    let index = build_index(&store);
    if std::fs::create_dir_all(paths::cache()).is_ok() {
        let tmp = cache.with_extension("json.part");
        if std::fs::write(&tmp, json!({"store": root, "index": index}).to_string()).is_ok() {
            let _ = std::fs::rename(tmp, cache);
        }
    }
    index
}
