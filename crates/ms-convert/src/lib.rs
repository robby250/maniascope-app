//! Keymode conversion: Torosu's key converter (MIT, see LICENSE-Torosu), commit d4d97c9, reduced to what ManiaScope uses.
//! Every convert is a separate new set on both clients, with no online ids: lazer marks an installed set for deletion
//! when an import has its OnlineID (BeatmapImporter.PreImport), so Torosu's merge into the original set must never
//! reach it, and one output per chart keeps its sha and md5 the same on either client.
mod analysis;
mod key_converter;
mod mania;
mod osu_fixups;
pub mod osz;
mod timing;

use anyhow::{bail, Result};
use key_converter::{Density, KeyConverterParams};

/// Changes whenever the output bytes of any chart can change: stored with each index row.
pub const CONVERTER_ID: &str = "torosu-d4d97c9-1";

pub const KEYS: std::ops::RangeInclusive<usize> = 1..=mania::STABLE_MAX_KEYS;

pub struct Converted {
    pub osu: Vec<u8>,
    pub file_name: String,
    pub version: String,
    pub keys: usize,
}

/// Going up adds notes: krrcream's 10K preset does too (rows of 2 to 8), Torosu's default None leaves maps thinner than
/// real ones of that keymode, and Full keeps handstreams as handstreams. One fixed setting: "too many options, better to
/// have good defaults" (user 2026-11-01). Going down never adds notes.
fn density(source: usize, target: usize) -> Density {
    if target > source { Density::Full } else { Density::None }
}

/// The `[Events]` lines rosu-map doesn't encode, with `[Variables]` expanded: storyboard and video.
fn raw_events(text: &str) -> Vec<String> {
    let (mut section, mut variables, mut events) = ("", Vec::<(&str, &str)>::new(), Vec::new());
    for line in text.lines() {
        let line = line.trim_start_matches('\u{feff}').trim_end();
        let trimmed = line.trim();
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            section = trimmed;
            continue;
        }
        match section {
            "[Variables]" => {
                if let Some((name, value)) = trimmed.split_once('=').filter(|x| x.0.starts_with('$')) {
                    variables.push((name, value));
                }
            }
            "[Events]" if !trimmed.is_empty() && !trimmed.starts_with("//") => {
                let encoded = !line.starts_with([' ', '_']) && matches!(line.split(',').next().map(str::trim), Some("0" | "Background" | "2" | "Break"));
                if !encoded {
                    events.push(line.to_string());
                }
            }
            _ => {}
        }
    }
    variables.sort_by_key(|v| std::cmp::Reverse(v.0.len())); // so $ab is not replaced by the value of $a
    for line in &mut events {
        for (name, value) in &variables {
            if line.contains(name) {
                *line = line.replace(name, value);
            }
        }
    }
    events
}

/// Appended at the end of the `[Events]` section of an encoded `.osu`.
fn inject_events(osu: String, events: &[String]) -> String {
    const HEADER: &str = "[Events]\n";
    let Some(start) = osu.find(HEADER).filter(|_| !events.is_empty()) else { return osu };
    let body = start + HEADER.len();
    let at = osu[body..].find("\n[").map_or(osu.len(), |p| body + p);
    let mut out = String::with_capacity(osu.len() + events.iter().map(|e| e.len() + 1).sum::<usize>());
    out.push_str(&osu[..at]);
    if !out.ends_with('\n') {
        out.push('\n');
    }
    for line in events {
        out.push_str(line);
        out.push('\n');
    }
    out.push_str(&osu[at..]);
    out
}

fn sanitize_filename(name: &str) -> String {
    name.chars().filter(|c| !matches!(c, '<' | '>' | ':' | '"' | '/' | '\\' | '|' | '?' | '*') && !c.is_control()).collect()
}

/// A chart at another key count. Deterministic: the same bytes give the same bytes.
pub fn convert(source: &[u8], target: usize) -> Result<Converted> {
    // Website charts are anyone's files: an odd one fails alone, not the thread indexing thousands.
    std::panic::catch_unwind(|| convert_chart(source, target)).unwrap_or_else(|_| bail!("the converter failed on this chart"))
}

fn convert_chart(source: &[u8], target: usize) -> Result<Converted> {
    let mut map: rosu_map::Beatmap = rosu_map::from_bytes(source)?;
    let keys = mania::key_count(&map);
    if keys == target {
        bail!("the map is already {target}K");
    }
    key_converter::apply(&mut map, &KeyConverterParams { target_keys: target as u8, density: density(keys, target), keep_jacks: true, playstyle: None })?;
    map.version = format!("{} ({keys}K to {target}K)", map.version);
    let events = raw_events(&String::from_utf8_lossy(source));
    let encoded = map.encode_to_string()?;
    let osu = osu_fixups::sanitize(inject_events(encoded, &events));
    let file_name = sanitize_filename(&format!("{} - {} ({}) [{}].osu", map.artist, map.title, map.creator, map.version));
    Ok(Converted { osu: osu.into_bytes(), file_name, version: map.version, keys: target })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stacked_notes_are_one_note() {
        let mut osu = source();
        for _ in 0..14 {
            osu.push_str("36,192,30000,1,0,0:0:0:0:\n");
        }
        let out = String::from_utf8(convert(osu.as_bytes(), 10).unwrap().osu).unwrap();
        assert_eq!(out.lines().filter(|l| l.split(',').nth(2) == Some("30000")).count(), 1);
    }

    fn source() -> String {
        let mut osu = "osu file format v14\n\n[General]\nAudioFilename: a.mp3\nMode: 3\n\n[Metadata]\nTitle:T\nArtist:A\nCreator:C\nVersion:Insane\nBeatmapID:123\nBeatmapSetID:456\n\n\
                       [Difficulty]\nCircleSize:7\nOverallDifficulty:8\n\n[Variables]\n$x=99\n\n[Events]\n0,0,\"bg.jpg\",0,0\nSprite,Foreground,Centre,\"s.png\",$x,240\n _F,0,0,1000,1,0\n\n\
                       [TimingPoints]\n0,500,4,1,0,60,1,0\n\n[HitObjects]\n".to_string();
        for step in 0..200 {
            osu.push_str(&format!("{},192,{},1,0,0:0:0:0:\n", (step * 3 % 7) * 512 / 7 + 36, step * 125));
        }
        osu
    }

    #[test]
    fn same_bytes_give_same_bytes() {
        assert_eq!(convert(source().as_bytes(), 10).unwrap().osu, convert(source().as_bytes(), 10).unwrap().osu);
    }

    /// lazer marks an installed set for deletion when an import has its OnlineID.
    #[test]
    fn a_convert_has_no_online_ids() {
        let out = convert(source().as_bytes(), 10).unwrap();
        let text = String::from_utf8(out.osu).unwrap();
        assert!(!text.lines().any(|l| l.starts_with("BeatmapID") || l.starts_with("BeatmapSetID")), "{text}");
    }

    #[test]
    fn keeps_the_storyboard_and_names_the_convert() {
        let out = convert(source().as_bytes(), 4).unwrap();
        assert_eq!(out.version, "Insane (7K to 4K)");
        assert_eq!(out.file_name, "A - T (C) [Insane (7K to 4K)].osu");
        let text = String::from_utf8(out.osu).unwrap();
        assert!(text.contains("Sprite,Foreground,Centre,\"s.png\",99,240\r\n _F,0,0,1000,1,0\r\n"), "{text}");
        assert!(text.lines().any(|l| l.replace(" ", "") == "CircleSize:4") && text.lines().any(|l| l == "Version: Insane (7K to 4K)" || l == "Version:Insane (7K to 4K)"), "{text}");
    }

    #[test]
    fn refuses_the_same_key_count() {
        assert!(convert(source().as_bytes(), 7).is_err());
    }

    #[test]
    fn going_up_adds_notes_going_down_never() {
        let rows = |osu: &[u8]| String::from_utf8_lossy(osu).split("[HitObjects]").nth(1).unwrap().lines().filter(|l| !l.is_empty()).count();
        let original = rows(source().as_bytes());
        assert!(rows(&convert(source().as_bytes(), 10).unwrap().osu) > original);
        assert!(rows(&convert(source().as_bytes(), 4).unwrap().osu) <= original);
    }
}
