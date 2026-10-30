//! `.osu` parsing: notes (start, end, column) and the lazer mania scroll multiplier.
use crate::py;
use std::collections::BTreeMap;

#[derive(Debug)]
pub enum ChartError {
    /// Not a usable native osu!mania chart (`skill_calc.ChartError`).
    Chart(String),
    /// Python would have raised something else (ValueError/OverflowError) while parsing.
    Value(String),
}

#[derive(Clone, Debug, Default)]
pub struct Chart {
    pub keys: usize,
    /// Structure of arrays, sorted by (t, col, end); ms (integral for parsed charts, fractional for
    /// rolled_execution's reference charts).
    pub t: Vec<f64>,
    pub end: Vec<f64>,
    pub col: Vec<u8>,
    /// (ms, multiplier); empty = constant scroll.
    pub sv: Vec<(f64, f64)>,
    pub od: f64,
    pub title: String,
    pub artist: String,
    pub version: String,
    pub creator: String,
}

impl Chart {
    pub fn len(&self) -> usize {
        self.t.len()
    }
    pub fn last_end(&self) -> f64 {
        self.end.iter().copied().fold(None, |m: Option<f64>, x| Some(match m { Some(m) if m >= x => m, _ => x })).unwrap_or(0.0)
    }
}

/// Python text mode: utf-8 with errors="ignore", universal newlines.
fn lines(data: &[u8]) -> Vec<String> {
    let text = String::from_utf8_lossy(data).replace('\u{FFFD}', "");
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut it = text.chars().peekable();
    while let Some(ch) = it.next() {
        match ch {
            '\n' => {
                out.push(std::mem::take(&mut cur));
            }
            '\r' => {
                if it.peek() == Some(&'\n') {
                    it.next();
                }
                out.push(std::mem::take(&mut cur));
            }
            _ => cur.push(ch),
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Python `str.strip()` (whitespace as `str.isspace`).
fn strip(s: &str) -> &str {
    s.trim_matches(|c: char| c.is_whitespace() || ('\x1c'..='\x1f').contains(&c))
}

pub fn parse(data: &[u8]) -> Result<Chart, ChartError> {
    let lines = lines(data);
    let first = lines.first().map(|l| l.trim_start_matches('\u{FEFF}')).unwrap_or("");
    if !first.starts_with("osu file format") {
        return Err(ChartError::Chart("not an .osu file".into()));
    }
    let mut meta: Vec<(String, String)> = Vec::new();
    let mut keys: Option<i64> = None;
    let mut od = 5.0f64;
    let mut mode = 0i64;
    let mut section = String::new();
    let mut raw: Vec<(i64, i64, i64)> = Vec::new();
    let mut tps: Vec<(f64, f64, bool)> = Vec::new();
    for line in &lines[1..] {
        let line = strip(line);
        if line.is_empty() {
            continue;
        }
        if line.starts_with('[') {
            section = line.to_string();
            continue;
        }
        if section == "[HitObjects]" {
            let p: Vec<&str> = line.split(',').collect();
            if p.len() < 4 {
                continue;
            }
            let x = match py::int_float(p[0]) {
                Ok(v) => v,
                Err(true) => return Err(ChartError::Value("cannot convert float infinity to integer".into())),
                Err(false) => continue,
            };
            let t = match py::int_float(p[2]) {
                Ok(v) => v,
                Err(true) => return Err(ChartError::Value("cannot convert float infinity to integer".into())),
                Err(false) => continue,
            };
            let typ = match py::int(p[3]) {
                Some(v) => v,
                None => continue,
            };
            let end = if typ & 128 != 0 {
                if p.len() < 6 {
                    continue;
                }
                match py::int_float(p[5].split(':').next().unwrap()) {
                    Ok(v) => v,
                    Err(true) => return Err(ChartError::Value("cannot convert float infinity to integer".into())),
                    Err(false) => continue,
                }
            } else {
                t
            };
            raw.push((t, end.max(t), x));
        } else if section == "[TimingPoints]" {
            let p: Vec<&str> = line.split(',').collect();
            if p.len() < 2 {
                continue;
            }
            let (Some(a), Some(b)) = (py::float(p[0]), py::float(p[1])) else { continue };
            let red = if p.len() > 6 { !strip(p[6]).starts_with('0') } else { true };
            tps.push((a, b, red));
        } else if let Some((k, v)) = line.split_once(':') {
            if section == "[General]" && k == "Mode" {
                let v = strip(v);
                mode = if v.is_empty() { 0 } else { py::int(v).ok_or_else(|| ChartError::Value(format!("invalid literal for int(): {v:?}")))? };
            } else if section == "[Difficulty]" && k == "CircleSize" {
                let f = py::float(v).ok_or_else(|| ChartError::Value(format!("could not convert string to float: {v:?}")))?;
                if !f.is_finite() {
                    return Err(ChartError::Value("cannot convert float to integer".into()));
                }
                keys = Some(py::round(f) as i64);
            } else if section == "[Difficulty]" && k == "OverallDifficulty" {
                od = py::float(v).ok_or_else(|| ChartError::Value(format!("could not convert string to float: {v:?}")))?;
            } else if section == "[Metadata]" {
                let v = strip(v).to_string();
                match meta.iter_mut().find(|(mk, _)| mk == k) {
                    Some(e) => e.1 = v,
                    None => meta.push((k.to_string(), v)),
                }
            }
        }
    }
    if mode != 3 {
        return Err(ChartError::Chart("not a native osu!mania chart (converts are unsupported)".into()));
    }
    let keys = match keys {
        Some(k) if (1..=18).contains(&k) => k,
        k => return Err(ChartError::Chart(format!("unsupported key count: {}", k.map_or("None".into(), |k| k.to_string())))),
    };
    // Normalise: column, dedupe, order-independent sort, no same-column overlap.
    let mut notes: Vec<(i64, i64, i64)> = raw
        .into_iter()
        .map(|(t, e, x)| (t, e, (keys - 1).min(0i64.max(py::floordiv(x.saturating_mul(keys), 512)))))
        .collect();
    notes.sort_by_key(|&(t, e, c)| (t, c, e));
    notes.dedup();
    let mut ch = Chart { keys: keys as usize, ..Default::default() };
    let mut last: Vec<Option<usize>> = vec![None; keys as usize];
    for (t, e, c) in notes {
        if let Some(i) = last[c as usize] {
            if ch.t[i] == t as f64 {
                continue; // duplicate head
            }
            if ch.end[i] > t as f64 {
                ch.end[i] = t as f64; // impossible overlap: cut the earlier hold
            }
        }
        last[c as usize] = Some(ch.t.len());
        ch.t.push(t as f64);
        ch.end.push(e as f64);
        ch.col.push(c as u8);
    }
    ch.od = if od.is_finite() { od } else { 5.0 };
    let get = |k: &str| meta.iter().find(|(mk, _)| mk == k).map(|(_, v)| v.clone());
    let nonempty = |k: &str| get(k).filter(|v| !v.is_empty());
    ch.title = nonempty("TitleUnicode").or_else(|| get("Title")).unwrap_or_default();
    ch.artist = nonempty("ArtistUnicode").or_else(|| get("Artist")).unwrap_or_default();
    ch.version = get("Version").unwrap_or_default();
    ch.creator = get("Creator").unwrap_or_default();
    ch.sv = scroll(&tps, ch.last_end());
    Ok(ch)
}

/// [(ms, scroll multiplier)] as osu!lazer's mania playfield scrolls (ppy/osu 577d29f,
/// DrawableScrollingRuleset + LegacyBeatmapDecoder): multiplier = SV (100 / -beatLength of a
/// green line, 1 at a red line, clamped 0.01-10) × most common beat length / current beat length;
/// last point at a timestamp wins; points after the last object are dropped. [] = constant.
fn scroll(tps: &[(f64, f64, bool)], last: f64) -> Vec<(f64, f64)> {
    let mut pts: Vec<(f64, f64, f64)> = Vec::new();
    let mut bl: Option<f64> = None;
    let mut sv = 1.0f64;
    for &(t, v, red) in tps {
        if red && v > 0.0 {
            bl = Some(60000f64.min(6f64.max(v)));
            sv = 1.0;
        } else if !red {
            sv = if v < 0.0 { 10f64.min(0.01f64.max(100.0 / -v)) } else { 1.0 };
        }
        if let Some(b) = bl {
            if t <= last {
                pts.push((t, b, sv));
            }
        }
    }
    let reds: Vec<(f64, f64)> = tps.iter().filter(|&&(_, v, red)| red && v > 0.0).map(|&(t, v, _)| (t, v)).collect();
    if pts.is_empty() || reds.is_empty() {
        return Vec::new();
    }
    // most common beat length by duration (first red starts at 0); dict order = first appearance
    let mut dur: Vec<(f64, f64)> = Vec::new();
    for (i, &(t, v)) in reds.iter().enumerate() {
        if t > last {
            continue;
        }
        let end = if i + 1 < reds.len() { reds[i + 1].0 } else { last };
        let k = py::round_digits(60000f64.min(6f64.max(v)), 3);
        let add = 0f64.max(end.min(last) - if i == 0 { 0.0 } else { t });
        match dur.iter_mut().find(|(dk, _)| *dk == k) {
            Some(e) => e.1 += add,
            None => dur.push((k, add)),
        }
    }
    let base = if dur.is_empty() {
        pts[0].1
    } else {
        let mut best = dur[0];
        for &d in &dur[1..] {
            if d.1 > best.1 {
                best = d;
            }
        }
        best.0
    };
    let mut out: BTreeMap<u64, (f64, f64)> = BTreeMap::new(); // keyed by ordered t bits
    for &(t, b, v) in &pts {
        out.insert(order_key(t), (t, v * base / b));
    }
    let sv: Vec<(f64, f64)> = out.into_values().collect();
    if sv.iter().all(|&(_, m)| (m - 1.0).abs() < 1e-3) {
        Vec::new()
    } else {
        sv
    }
}

/// Total order of finite doubles as u64 (for sorted maps keyed by time).
fn order_key(x: f64) -> u64 {
    let x = if x == 0.0 { 0.0 } else { x }; // -0.0 == 0.0 as a dict key
    let b = x.to_bits();
    if b >> 63 == 1 {
        !b
    } else {
        b | (1 << 63)
    }
}
