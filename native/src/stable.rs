//! osu!stable's osu!.db (library) and scores.db (local scores), read-only.
//! Format: https://github.com/ppy/osu/wiki/Legacy-database-file-structure (star pairs are Int-Float from 20250107).
//! Not part of the calculator identity (native_backend.identity_files leaves it out).
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use sha2::{Digest, Sha256};

use crate::background::background_map;

struct Reader<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.at.checked_add(n).filter(|&e| e <= self.b.len()).ok_or("truncated database")?;
        let s = &self.b[self.at..end];
        self.at = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }
    fn i16(&mut self) -> Result<i16, String> {
        Ok(i16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn i32(&mut self) -> Result<i32, String> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn i64(&mut self) -> Result<i64, String> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn count(&mut self) -> Result<usize, String> {
        usize::try_from(self.i32()?).map_err(|_| "negative count".to_string())
    }
    fn string(&mut self) -> Result<String, String> {
        match self.u8()? {
            0 => Ok(String::new()),
            0x0b => {
                let (mut len, mut shift) = (0usize, 0);
                loop {
                    let byte = self.u8()?;
                    len |= ((byte & 0x7f) as usize) << shift;
                    if byte & 0x80 == 0 {
                        break;
                    }
                    shift += 7;
                    if shift > 28 {
                        return Err("bad string length".into());
                    }
                }
                Ok(String::from_utf8_lossy(self.take(len)?).into_owned())
            }
            b => Err(format!("bad string marker {b:#x} at {}", self.at - 1)),
        }
    }
}

/// .NET DateTime ticks (UTC) → Unix seconds.
fn unix(ticks: i64) -> f64 {
    (ticks - 621_355_968_000_000_000) as f64 / 1e7
}

pub struct Chart {
    pub artist: String,
    pub title: String,
    pub creator: String,
    pub version: String,
    pub audio: String,
    pub md5: String,
    pub file: String,
    pub folder: String,
    pub status: u8,
    pub cs: f32,
    pub od: f32,
    pub total_ms: i32,
    pub beatmap_id: i32,
    pub set_id: i32,
    pub mode: u8,
}

/// (player name, mania and convertible charts).
pub fn library(b: &[u8]) -> Result<(String, Vec<Chart>), String> {
    let mut r = Reader { b, at: 0 };
    let version = r.i32()?;
    if version < 20191106 {
        return Err(format!("osu!.db version {version} is older than this reader (update osu!)"));
    }
    r.i32()?; // folders
    r.u8()?; // account unlocked
    r.i64()?; // unlock date
    let player = r.string()?;
    let n = r.count()?;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let artist = r.string()?;
        r.string()?;
        let title = r.string()?;
        r.string()?;
        let creator = r.string()?;
        let version_name = r.string()?;
        let audio = r.string()?;
        let md5 = r.string()?;
        let file = r.string()?;
        let status = r.u8()?;
        r.take(2 + 2 + 2 + 8)?; // circles, sliders, spinners, modified
        r.f32()?; // AR
        let cs = r.f32()?;
        r.f32()?; // HP
        let od = r.f32()?;
        r.take(8)?; // slider velocity
        for _ in 0..4 {
            let pairs = r.count()?;
            r.take(pairs * if version >= 20250107 { 10 } else { 14 })?;
        }
        r.i32()?; // drain seconds
        let total_ms = r.i32()?;
        r.i32()?; // preview
        let points = r.count()?;
        r.take(points * 17)?;
        let beatmap_id = r.i32()?;
        let set_id = r.i32()?;
        r.i32()?; // thread
        r.take(4 + 2 + 4)?; // grades, local offset, stack leniency
        let mode = r.u8()?;
        r.string()?; // source
        r.string()?; // tags
        r.i16()?; // online offset
        r.string()?; // title font
        r.take(1 + 8 + 1)?; // unplayed, last played, osz2
        let folder = r.string()?;
        r.take(8 + 5 + 4 + 1)?; // last checked, 5 flags, modification, scroll speed
        out.push(Chart { artist, title, creator, version: version_name, audio, md5, file, folder, status, cs, od, total_ms, beatmap_id, set_id, mode });
    }
    Ok((player, out))
}

pub struct Score {
    pub mode: u8,
    pub md5: String,
    pub player: String,
    pub stats: [i32; 6], // MAX(geki), 300, 200(katu), 100, 50, miss: recdata's (perfect, great, good, ok, meh, miss)
    pub score: i32,
    pub combo: i16,
    pub mods: i32,
    pub played: f64,
    pub online_id: i64,
}

pub fn scores(b: &[u8]) -> Result<Vec<Score>, String> {
    let mut r = Reader { b, at: 0 };
    r.i32()?;
    let maps = r.count()?;
    let mut out = Vec::new();
    for _ in 0..maps {
        r.string()?;
        for _ in 0..r.count()? {
            let mode = r.u8()?;
            r.i32()?;
            let md5 = r.string()?;
            let player = r.string()?;
            r.string()?; // replay md5
            let (c300, c100, c50, geki, katu, miss) = (r.i16()?, r.i16()?, r.i16()?, r.i16()?, r.i16()?, r.i16()?);
            let score = r.i32()?;
            let combo = r.i16()?;
            r.u8()?; // perfect
            let mods = r.i32()?;
            r.string()?; // life bar
            let played = unix(r.i64()?);
            r.i32()?;
            let online_id = r.i64()?;
            if mods & (1 << 23) != 0 {
                r.take(8)?; // Target Practice accuracy
            }
            let stats = [geki, c300, katu, c100, c50, miss].map(|x| x as u16 as i32);
            out.push(Score { mode, md5, player, stats, score, combo, mods, played, online_id });
        }
    }
    Ok(out)
}

fn read(path: &str) -> PyResult<Vec<u8>> {
    std::fs::read(path).map_err(|e| PyValueError::new_err(format!("{path}: {e}")))
}

/// osu!.db → (player name, [chart dicts]).
#[pyfunction]
pub fn stable_library<'py>(py: Python<'py>, path: &str) -> PyResult<(String, Vec<Bound<'py, PyDict>>)> {
    let (player, charts) = library(&read(path)?).map_err(PyValueError::new_err)?;
    let mut out = Vec::with_capacity(charts.len());
    for c in charts {
        let d = PyDict::new(py);
        d.set_item("artist", c.artist)?;
        d.set_item("title", c.title)?;
        d.set_item("creator", c.creator)?;
        d.set_item("version", c.version)?;
        d.set_item("audio", c.audio)?;
        d.set_item("md5", c.md5)?;
        d.set_item("file", c.file)?;
        d.set_item("folder", c.folder)?;
        d.set_item("status", c.status)?;
        d.set_item("cs", c.cs)?;
        d.set_item("od", c.od)?;
        d.set_item("length", c.total_ms)?;
        d.set_item("beatmap_id", c.beatmap_id)?;
        d.set_item("set_id", c.set_id)?;
        d.set_item("mode", c.mode)?;
        out.push(d);
    }
    Ok((player, out))
}

/// scores.db → [score dicts].
#[pyfunction]
pub fn stable_scores<'py>(py: Python<'py>, path: &str) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let mut out = Vec::new();
    for s in scores(&read(path)?).map_err(PyValueError::new_err)? {
        let d = PyDict::new(py);
        d.set_item("mode", s.mode)?;
        d.set_item("md5", s.md5)?;
        d.set_item("player", s.player)?;
        d.set_item("stats", s.stats.to_vec())?;
        d.set_item("score", s.score)?;
        d.set_item("combo", s.combo)?;
        d.set_item("mods", s.mods)?;
        d.set_item("played", s.played)?;
        d.set_item("online_id", s.online_id)?;
        out.push(d);
    }
    Ok(out)
}

/// A replay's header (.osr without its frames): which play it holds.
#[pyfunction]
pub fn osr_header<'py>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyDict>> {
    let data = read(path)?;
    let parse = || -> Result<(u8, String, String, [i32; 6], i32, i32, f64), String> {
        let mut r = Reader { b: &data, at: 0 };
        let mode = r.u8()?;
        r.i32()?;
        let md5 = r.string()?;
        let player = r.string()?;
        r.string()?;
        let (c300, c100, c50, geki, katu, miss) = (r.i16()?, r.i16()?, r.i16()?, r.i16()?, r.i16()?, r.i16()?);
        let score = r.i32()?;
        r.take(2 + 1)?; // combo, perfect
        let mods = r.i32()?;
        r.string()?;
        let played = unix(r.i64()?);
        Ok((mode, md5, player, [geki, c300, katu, c100, c50, miss].map(|x| x as u16 as i32), score, mods, played))
    };
    let (mode, md5, player, stats, score, mods, played) = parse().map_err(PyValueError::new_err)?;
    let d = PyDict::new(py);
    d.set_item("mode", mode)?;
    d.set_item("md5", md5)?;
    d.set_item("player", player)?;
    d.set_item("stats", stats.to_vec())?;
    d.set_item("score", score)?;
    d.set_item("mods", mods)?;
    d.set_item("played", played)?;
    Ok(d)
}

/// SHA-256 hex of each file (None when unreadable): stable's .osu files get lazer's content identity.
#[pyfunction]
#[pyo3(signature = (paths, threads=0))]
pub fn sha256_files(py: Python<'_>, paths: Vec<String>, threads: usize) -> Vec<Option<String>> {
    py.detach(|| {
        background_map(&paths, threads, |p| {
            let data = std::fs::read(p).map_err(|e| e.to_string())?;
            Ok(Sha256::digest(&data).iter().map(|b| format!("{b:02x}")).collect::<String>())
        })
        .into_iter()
        .map(Result::ok)
        .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(out: &mut Vec<u8>, v: &str) {
        out.push(0x0b);
        out.push(v.len() as u8);
        out.extend(v.as_bytes());
    }

    #[test]
    fn scores_roundtrip() {
        let mut b = vec![];
        b.extend(20250107i32.to_le_bytes());
        b.extend(1i32.to_le_bytes());
        s(&mut b, "abc");
        b.extend(1i32.to_le_bytes());
        b.push(3);
        b.extend(20250107i32.to_le_bytes());
        s(&mut b, "abc");
        s(&mut b, "me");
        s(&mut b, "r");
        for v in [900i16, 30, 5, 1200, 40, 7] {
            b.extend(v.to_le_bytes());
        }
        b.extend(987654i32.to_le_bytes());
        b.extend(321i16.to_le_bytes());
        b.push(0);
        b.extend((64i32 | (1 << 23)).to_le_bytes());
        b.push(0);
        b.extend((621_355_968_000_000_000i64 + 1_700_000_000 * 10_000_000).to_le_bytes());
        b.extend((-1i32).to_le_bytes());
        b.extend(55i64.to_le_bytes());
        b.extend(0f64.to_le_bytes());
        let out = scores(&b).unwrap();
        assert_eq!(out.len(), 1);
        let x = &out[0];
        assert_eq!((x.mode, x.md5.as_str(), x.player.as_str()), (3, "abc", "me"));
        assert_eq!(x.stats, [1200, 900, 40, 30, 5, 7]);
        assert_eq!((x.score, x.combo, x.mods, x.online_id), (987654, 321, 64 | (1 << 23), 55));
        assert_eq!(x.played, 1_700_000_000.);
        assert!(scores(&b[..b.len() - 1]).is_err());
    }
}
