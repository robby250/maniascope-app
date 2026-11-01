//! The osu! state ManiaScope needs, read from the game's memory instead of running tosu next to it. Ported from tosu v4.26.3
//! (LGPL-3.0, see LICENSE-tosu): stable's patterns and offsets and lazer's object paths are tosu's work, reduced to the fields
//! used here. lazer's field offsets come from `layout`, computed from the game's DLLs.
//! `Reader::poll` returns tosu's /websocket/v2 shape, the watcher's input since before this crate existed.
pub mod process;
mod lazer;
pub mod layout;
mod stable;

use process::{exes, Process};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// One look at the game. Empty strings: not known yet.
#[derive(Default)]
struct Seen {
    state: &'static str,
    songs: String,
    folder: String,
    file: String,
    game: String,
    id: i64,
    time: f64,
    length: f64,
    mods: Value,
    hits: Value,
    replay: bool,
    profile: String,
    results: Value,
}

enum Game {
    Stable(stable::Stable),
    Lazer(lazer::Lazer),
}

pub struct Reader {
    game: Option<Game>,
    client: &'static str,
    time: Option<f64>,
    chart: Option<(String, (i64, f64, f64))>,
    last: Option<Value>,
    tried: Option<Instant>,
}

const RETRY: Duration = Duration::from_secs(60);
/// The first scans come before the game has loaded; a build the patterns don't fit must not scan forever at that pace.
const SCAN_WAIT: Duration = Duration::from_secs(5);
const SCAN_WAIT_MAX: Duration = Duration::from_secs(300);

/// stable and lazer share the name osu!.exe on Windows; lazer is a .NET app with its deps file beside it. (osu!.db can't tell:
/// a fresh stable has none until it first closes.)
fn running() -> Vec<(&'static str, u32, PathBuf)> {
    let mut found: Vec<_> = exes("osu!.exe").into_iter().map(|(pid, path)| (if cfg!(unix) || !path.with_file_name("osu!.deps.json").is_file() { "stable" } else { "lazer" }, pid, path)).collect();
    if cfg!(unix) {
        found.extend(exes("osu!").into_iter().map(|(pid, path)| ("lazer", pid, path)));
    }
    found
}

/// (ruleset, first object, end of the last) of a .osu file. Unreadable (stable under Wine reports paths only Wine knows):
/// `lastObject` 0 tells the watcher to go by the file it finds itself.
fn chart_times(path: &Path) -> (i64, f64, f64) {
    let Ok(bytes) = std::fs::read(path) else { return (0, 0., 0.) };
    let text = String::from_utf8_lossy(&bytes);
    let (mut mode, mut first, mut last, mut objects) = (0, None, 0f64, false);
    for line in text.lines() {
        if let Some(m) = line.strip_prefix("Mode:") {
            mode = m.trim().parse().unwrap_or(0);
        } else if line.trim_end() == "[HitObjects]" {
            objects = true;
        } else if line.starts_with('[') {
            objects = false;
        } else if objects {
            let fields: Vec<&str> = line.split(',').collect();
            let Some(start) = fields.get(2).and_then(|t| t.parse::<f64>().ok()) else { continue };
            let held = fields.get(3).and_then(|t| t.parse::<i64>().ok()).is_some_and(|t| t & 128 != 0);
            let end = fields.get(5).filter(|_| held).and_then(|e| e.split(':').next()).and_then(|e| e.parse().ok()).unwrap_or(start);
            first.get_or_insert(start);
            last = last.max(end);
        }
    }
    (mode, first.unwrap_or(0.), last)
}

impl Default for Reader {
    fn default() -> Reader {
        Reader::new()
    }
}

impl Reader {
    pub fn new() -> Reader {
        Reader { game: None, client: "", time: None, chart: None, last: None, tried: None }
    }

    fn attach(&mut self, prefer: &str) {
        if self.tried.is_some_and(|t| t.elapsed() < RETRY) {
            return; // a lazer whose DLLs can't be read: no retry every second
        }
        let games = running();
        let Some((client, pid, path)) = games.iter().find(|g| g.0 == prefer).or(games.first()) else { return };
        let game = if *client == "stable" {
            Process::open(*pid, false).map(|p| Game::Stable(stable::Stable::new(p, path)))
        } else {
            Process::open(*pid, true).and_then(|p| lazer::Lazer::new(p, path)).map(Game::Lazer)
        };
        self.tried = game.is_none().then(Instant::now);
        self.game = game;
        self.client = client;
    }

    /// None: no game this reader can follow. A failed read mid-load repeats the last frame, as tosu keeps its last state.
    pub fn poll(&mut self, prefer: &str) -> Option<Value> {
        if self.game.is_none() {
            self.attach(prefer);
        }
        let alive = match self.game.as_ref()? {
            Game::Stable(g) => g.alive(),
            Game::Lazer(g) => g.alive(),
        };
        if !alive {
            (self.game, self.time, self.chart, self.last) = (None, None, None, None);
            return None;
        }
        let seen = match self.game.as_mut()? {
            Game::Stable(g) => g.frame(),
            Game::Lazer(g) => g.frame(),
        };
        let Some(seen) = seen else { return self.last.clone() };
        let paused = self.time == Some(seen.time);
        self.time = Some(seen.time);
        let path = Path::new(&seen.songs).join(&seen.folder).join(&seen.file);
        let key = path.to_string_lossy().into_owned();
        if self.chart.as_ref().is_none_or(|c| c.0 != key) {
            self.chart = Some((key, chart_times(&path)));
        }
        let (mode, first, last) = self.chart.as_ref()?.1;
        let frame = json!({"client": self.client, "state": {"name": seen.state}, "game": {"paused": paused, "replay": seen.replay},
                           "play": {"mods": seen.mods, "hits": seen.hits}, "profile": {"name": seen.profile}, "resultsScreen": seen.results,
                           "beatmap": {"id": seen.id, "mode": {"number": mode}, "time": {"live": seen.time, "firstObject": first, "lastObject": last, "mp3Length": seen.length}},
                           "files": {"beatmap": seen.file}, "folders": {"songs": seen.songs, "beatmap": seen.folder, "game": seen.game}});
        self.last = Some(frame.clone());
        Some(frame)
    }
}

/// .NET DateTime.ToBinary split in two, as an ISO string.
fn net_date(high: i32, low: i32) -> Option<String> {
    let ticks = ((high as i64 & 0x3fff_ffff) << 32) | (low as u32 as i64);
    let at = chrono::DateTime::from_timestamp_millis((ticks - 621_355_968_000_000_000) / 10_000)?;
    Some(at.to_rfc3339_opts(chrono::SecondsFormat::Millis, true))
}

/// mania's letters on both clients.
fn grade(accuracy: f64, silver: bool) -> &'static str {
    match accuracy {
        a if a >= 100. => if silver { "XH" } else { "X" },
        a if a >= 95. => if silver { "SH" } else { "S" },
        a if a >= 90. => "A",
        a if a >= 80. => "B",
        a if a >= 70. => "C",
        _ => "D",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_chart_gives_its_ruleset_and_the_end_of_its_last_hold() {
        let path = std::env::temp_dir().join(format!("ms-osumem-{}.osu", std::process::id()));
        std::fs::write(&path, "[General]\nMode: 3\n\n[HitObjects]\n36,192,1000,1,0,0:0:0:0:\n109,192,2000,128,0,5000:0:0:0:0:\n182,192,3000,1,0,0:0:0:0:\n").unwrap();
        assert_eq!(chart_times(&path), (3, 1000., 5000.));
        let _ = std::fs::remove_file(path);
        assert_eq!(chart_times(Path::new("/nowhere.osu")), (0, 0., 0.));
    }

    #[test]
    fn dates_decode_from_net_binary() {
        // 2026-11-01T12:00:00Z in .NET ticks, Kind flags in the top bits
        let ticks: i64 = 639_000_000_000_000_000 + 0;
        let iso = net_date(((ticks >> 32) as i32) | 1 << 30, ticks as i32).unwrap();
        let back = chrono::DateTime::parse_from_rfc3339(&iso).unwrap().timestamp_millis();
        assert_eq!(back * 10_000 + 621_355_968_000_000_000, ticks);
    }
}
