//! Observing the game: tosu's WebSocket first, lazer's log as the rate-less fallback. Emits an observation whenever the
//! selection, rate or connection changes; tracks each attempt from gameplay to its result.
use ms_data::feedback::{constant_speed, rate_from_mods};
use ms_data::imports::{stable_link, stable_songs};
use ms_data::paths::{lazer_data, local_file, long_path, stable_data, stable_store};
use ms_data::store::{now, timestamp};
use ms_rec::navigation::game_running;
use ms_rec::worker::{Queue, Task};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};
use tungstenite::{Message, WebSocket};

pub const TOSU_HOST: &str = "127.0.0.1:24050";
const GAMEPLAY_FRAME: Duration = Duration::from_millis(300); // tosu pushes every POLL_RATE; the clock needs 3 Hz

#[derive(Clone, Debug, PartialEq, Default)]
pub struct Obs {
    pub source: Option<&'static str>, // "tosu" | "log"
    pub path: Option<PathBuf>,
    pub rate: Option<f64>, // None: unknown, never No Mod
    pub cs: bool,
    pub mods: Option<Vec<Value>>,
    pub bid: Option<i64>,
    pub client: Option<String>,
    pub note: String,
}

pub enum Event {
    Obs(Obs),
    Status(String),
    Progress(Option<f64>), // playback ms while in gameplay
    Score { score: Value, path: PathBuf, fresh: bool, attempt: Option<Value> },
}

fn waiting() -> Obs {
    Obs { source: Some("tosu"), note: "Waiting for osu!: start it and select a mania map".into(), ..Obs::default() }
}

fn basename(p: Option<&PathBuf>) -> Option<String> {
    p.and_then(|p| p.file_name()).map(|n| n.to_string_lossy().into_owned())
}

/// The attempt lifecycle over tosu's states, without any I/O of its own but the chart lookup.
pub struct Tracker {
    pub tx: Sender<Event>,
    pub q: Arc<Queue>,
    pub play: Arc<Mutex<Option<Value>>>, // the attempt in gameplay; tracking pauses read its start
    done: Option<Value>,                 // the one whose result comes next
    last_score: Option<Value>,
    last: Option<Obs>,
    live: Option<f64>,
    last_live: Option<f64>,
    needs_import: bool,
    disconnected: bool,
    stable_root: Option<PathBuf>,
    stable_game: Option<String>,
    stable_files: HashMap<(PathBuf, Option<SystemTime>, u64), PathBuf>,
    pub replay_watching: fn() -> bool,
}

/// stable: tosu's StreamCompanion status (its v2 JSON lacks the flag) is 8 while a replay plays.
fn file_is_mania(path: &Path) -> bool {
    std::fs::read(path).is_ok_and(|b| String::from_utf8_lossy(&b).lines().any(|l| l.strip_prefix("Mode:").is_some_and(|m| m.trim() == "3")))
}

fn replay_status() -> bool {
    ureq::get(&format!("http://{TOSU_HOST}/json/sc")).timeout(Duration::from_millis(600)).call().ok()
        .and_then(|r| r.into_string().ok()).and_then(|s| serde_json::from_str::<Value>(&s).ok()).is_some_and(|v| v["status"] == 8)
}

impl Tracker {
    pub fn new(tx: Sender<Event>, q: Arc<Queue>, play: Arc<Mutex<Option<Value>>>) -> Tracker {
        Tracker { tx, q, play, done: None, last_score: None, last: None, live: None, last_live: None, needs_import: false, disconnected: false, stable_root: None, stable_game: None, stable_files: HashMap::new(), replay_watching: replay_status }
    }

    fn send(&self, e: Event) {
        let _ = self.tx.send(e);
    }

    /// One observation pass: tosu's state or nothing (`tosu` false: the log fallback's).
    pub fn observed(&mut self, obs: Obs, tosu: bool) {
        if !tosu || obs.path.is_none() {
            self.needs_import = true;
            self.disconnected = true;
        } else if self.needs_import && self.play.lock().unwrap().is_none() {
            self.needs_import = false;
            self.q.put(Task::Import); // missed results survive in lazer's database
        }
        if self.last.as_ref() != Some(&obs) {
            self.last = Some(obs.clone());
            self.send(Event::Obs(obs));
        }
        if self.live != self.last_live {
            self.last_live = self.live;
            self.send(Event::Progress(self.live));
        }
    }

    fn chart(&mut self, d: &Value) -> Option<PathBuf> {
        let rel = d["files"]["beatmap"].as_str().filter(|s| !s.is_empty())?;
        let songs = d["folders"]["songs"].as_str().filter(|s| !s.is_empty());
        if d["client"] != "stable" {
            let store = lazer_data().join("files");
            return [songs.map(PathBuf::from), Some(store)].into_iter().flatten().map(|r| r.join(rel)).find(|p| p.is_file());
        }
        if let Some(game) = d["folders"]["game"].as_str().filter(|g| !g.is_empty() && self.stable_game.as_deref() != Some(*g)) {
            ms_data::paths::remember_stable(game);
            self.stable_game = Some(game.to_string());
        }
        // The chart's .osu linked into stable_store under its SHA-256 (tosu under Wine can report a Songs folder that only
        // exists inside Wine).
        let root = self.stable_root.get_or_insert_with(|| stable_songs(&stable_data())).clone();
        let folder = d["folders"]["beatmap"].as_str().unwrap_or("");
        let real = [songs.map(PathBuf::from), Some(root)].into_iter().flatten().map(|r| long_path(&r.join(folder).join(rel))).find(|p| p.is_file())?;
        let st = std::fs::metadata(&real).ok()?;
        let key = (real.clone(), st.modified().ok(), st.len());
        if let Some(hit) = self.stable_files.get(&key) {
            return Some(hit.clone());
        }
        let data = std::fs::read(&real).ok()?;
        let sha: String = Sha256::digest(&data).iter().map(|b| format!("{b:02x}")).collect();
        stable_link(&real, &sha).ok()?;
        if self.stable_files.len() > 256 {
            self.stable_files.clear();
        }
        let linked = local_file(&sha, &stable_store())?;
        self.stable_files.insert(key, linked.clone());
        Some(linked)
    }

    /// Attempt lifecycle: start on entering gameplay (a jump back in time is a restart), abort on leaving it anywhere but
    /// the results screen; the result shown next belongs to this attempt.
    fn track(&mut self, state: &str, d: &Value, path: Option<&PathBuf>) {
        let rel = basename(path);
        let tm = &d["beatmap"]["time"];
        let mods = &d["play"]["mods"];
        // Autoplay/Cinema is the game playing, not the player: no attempt
        let auto = mods["array"].as_array().is_some_and(|a| a.iter().any(|m| matches!(m["acronym"].as_str(), Some("AT" | "CN"))));
        let mut current = self.play.lock().unwrap().take();
        if let (true, Some(sha), false) = (state == "play", rel.as_ref(), auto) {
            let num = |v: &Value| v.as_f64().unwrap_or(0.);
            let (live, first) = (num(&tm["live"]), num(&tm["firstObject"]));
            let last = Some(num(&tm["lastObject"])).filter(|l| *l > first).unwrap_or(num(&tm["mp3Length"]));
            let stale = current.as_ref().is_none_or(|p| p["sha"] != sha.as_str() || live + 1500. < num(&p["pos"]));
            if stale {
                // A jump back within the first seconds is gameplay taking over from the song-select preview position, not
                // a restart: the attempt starts again without an abort.
                if let Some(mut p) = current.take().filter(|p| now() - num(&p["t"]) > 5. || p["sha"] != sha.as_str()) {
                    p["end"] = json!(now());
                    self.q.put(if self.disconnected { Task::Void } else { Task::Abort(p) });
                }
                let own = d["client"] != "stable" || !(self.replay_watching)();
                let p = json!({"sha": sha, "path": path.map(|p| p.to_string_lossy()), "own": own, "rate": rate_from_mods(mods).0, "t": now(), "pos": live, "progress": 0.,
                               "played_seconds": 0., "mods": mods.get("array").cloned().unwrap_or(Value::Null)});
                self.q.put(Task::Start(p.clone()));
                current = Some(p);
            }
            let p = current.as_mut().unwrap();
            let delta = live - first.max(num(&p["pos"]));
            if delta > 0. && delta < 2000. {
                if let Some(rate) = p["rate"].as_f64().filter(|r| *r != 0.) {
                    p["played_seconds"] = json!(num(&p["played_seconds"]) + delta / 1000. / rate);
                }
            }
            let paused = d["game"]["paused"].as_bool().unwrap_or(false);
            // stable reports the lead-in as paused (2026-11-01, real play): before the first note nothing is interrupted
            if paused && live > first && !p["paused_now"].as_bool().unwrap_or(false) {
                // stable's .osr has no pause list (lazer's Realm does)
                match p["pauses"].as_array_mut() {
                    Some(list) => list.push(json!(live as i64)),
                    None => p["pauses"] = json!([live as i64]),
                }
            }
            p["paused_now"] = json!(paused);
            p["partial_stats"] = d["play"]["hits"].clone();
            p["pos"] = json!(live);
            p["progress"] = json!(((live - first) / (last - first).max(1.)).clamp(0., 1.));
            *self.play.lock().unwrap() = current;
        } else if let Some(mut p) = current {
            if state == "resultscreen" {
                self.done = Some(p);
            } else {
                // auto: the first gameplay poll comes before Autoplay's mods do
                p["end"] = json!(now());
                self.q.put(if self.disconnected || auto { Task::Void } else { Task::Abort(p) });
            }
        }
        self.disconnected = false;
    }

    /// Hand the user's own mania result to the UI; fresh only right after its own gameplay (a replay or an old result
    /// being viewed is history, not today's form).
    fn note_score(&mut self, d: &Value, path: Option<&PathBuf>) {
        let rs = &d["resultsScreen"];
        let rel = basename(path);
        // tosu 4.26.2 reads stable's player name as "" or garbage ("wd" as guest, 2026-10-31): on stable the result right
        // after this app's own tracked, not-a-replay attempt on the chart is the player's
        let own = rs["playerName"] == d["profile"]["name"]
            || d["client"] == "stable" && self.done.as_ref().is_some_and(|p| p["sha"].as_str() == rel.as_deref() && p["own"] == true);
        let (Some(rel), Some(path), true) = (rel, path, own) else { return };
        if rs["mode"]["name"].as_str().unwrap_or("").to_lowercase() != "mania" {
            return;
        }
        let played = rs["createdAt"].as_str().unwrap_or("");
        let score = json!({"sha256": rel, "rate": rate_from_mods(&rs["mods"]).0, "client": d["client"], "mods": rs["mods"]["name"].as_str().unwrap_or(""),
                           "accuracy": rs["accuracy"], "hits": rs["hits"], "rank": rs["rank"], "max_combo": rs["maxCombo"], "score": rs["score"], "played": played,
                           "mods_list": rs["mods"]["array"]});
        let hits: i64 = ["geki", "300", "katu", "100", "50", "0"].iter().filter_map(|k| rs["hits"][*k].as_i64()).sum();
        if hits == 0 || timestamp(played).is_none() {
            return; // the results payload is still filling in
        }
        if self.last_score.as_ref() == Some(&score) {
            return;
        }
        self.last_score = Some(score.clone());
        let mut fresh = false;
        if let Some(att) = self.done.as_ref().filter(|a| a["sha"] == rel.as_str()) {
            fresh = timestamp(played).is_none_or(|ts| ts >= att["t"].as_f64().unwrap_or(0.) - 30.);
            if !fresh {
                self.q.put(Task::Void);
            }
        }
        let attempt = self.done.take();
        self.send(Event::Score { score, path: path.clone(), fresh, attempt });
    }

    /// One frame of /websocket/v2 or /json/v2.
    pub fn frame(&mut self, d: &Value) -> Obs {
        self.live = None;
        if d.get("error").is_some_and(|e| !e.is_null()) || d.get("beatmap").is_none() {
            return waiting();
        }
        // stable opens on an osu!standard map; converts are unsupported, so only mania charts are analysed. tosu that can't
        // reach the file (stable under Wine) leaves `mode` and `lastObject` at 0: then the file decides.
        let reported = d["beatmap"]["mode"]["number"].as_i64().unwrap_or(3) == 3;
        let unparsed = d["beatmap"]["time"]["lastObject"].as_f64() == Some(0.);
        let path = if reported || unparsed { self.chart(d).filter(|p| reported || file_is_mania(p)) } else { None };
        let mania = reported || path.is_some();
        let Some(state) = d["state"]["name"].as_str() else { return Obs { source: Some("tosu"), note: "unexpected tosu payload".into(), ..Obs::default() } };
        self.track(&state.to_lowercase(), d, path.as_ref());
        if state == "resultScreen" {
            self.note_score(d, path.as_ref());
            if let Some(last) = self.last.as_ref().filter(|o| o.source == Some("tosu")) {
                return last.clone(); // a past score's mods are not the selection
            }
        }
        let mods = &d["play"]["mods"];
        let (rate, note) = rate_from_mods(mods);
        if state.eq_ignore_ascii_case("play") {
            self.live = d["beatmap"]["time"]["live"].as_f64();
        }
        let list = mods.get("array").or(Some(mods)).and_then(Value::as_array).cloned();
        let note = if path.is_some() { note } else if mania { "selected map file not found".into() } else { "not an osu!mania map".into() };
        Obs { source: Some("tosu"), path, rate, cs: constant_speed(mods), mods: list, bid: d["beatmap"]["id"].as_i64(), client: d["client"].as_str().map(str::to_string), note }
    }
}

/// The rate-less fallback: lazer's own log names the working beatmap.
struct LogTail {
    log: Option<(PathBuf, u64)>,
    display: Option<String>,
    index: Option<HashMap<String, String>>,
    reindexed: Option<Instant>,
    job: Option<std::thread::JoinHandle<HashMap<String, String>>>,
}

impl LogTail {
    fn newest() -> Option<PathBuf> {
        let files = std::fs::read_dir(lazer_data().join("logs")).ok()?;
        files.flatten().filter(|e| e.file_name().to_string_lossy().ends_with("runtime.log"))
            .max_by_key(|e| e.metadata().and_then(|m| m.modified()).ok()).map(|e| e.path())
    }

    fn poll(&mut self, client: &str, tx: &Sender<Event>) -> Obs {
        if client == "stable" {
            return Obs { note: "osu!stable needs tosu (starting it…)".into(), ..Obs::default() };
        }
        if !game_running("lazer") { // an old log is not a live selection
            (self.log, self.display) = (None, None);
            return Obs { note: "osu!lazer is not running".into(), ..Obs::default() };
        }
        if let Some(newest) = Self::newest() {
            if self.log.as_ref().is_none_or(|l| l.0 != newest) {
                self.log = Some((newest.clone(), 0));
            }
            let at = self.log.as_ref().map_or(0, |l| l.1);
            let mut text = Vec::new();
            let read = std::fs::File::open(&newest).and_then(|mut f| { f.seek(SeekFrom::Start(at))?; f.read_to_end(&mut text) });
            if let Ok(n) = read {
                self.log = Some((newest, at + n as u64));
                const MARK: &str = "Game-wide working beatmap updated to ";
                for line in String::from_utf8_lossy(&text).lines() {
                    if let Some(i) = line.find(MARK) {
                        self.display = Some(line[i + MARK.len()..].trim_end().to_string());
                    }
                }
            }
        }
        if self.job.as_ref().is_some_and(|j| j.is_finished()) {
            self.index = self.job.take().and_then(|j| j.join().ok());
        }
        let path = self.display.clone().and_then(|d| self.resolve(&d, tx));
        let note = match (&self.display, &path) {
            (Some(d), None) => format!("not in library index: {d}"),
            _ => "no live rate without tosu".to_string(),
        };
        Obs { source: Some("log"), path, note, ..Obs::default() }
    }

    fn resolve(&mut self, display: &str, tx: &Sender<Event>) -> Option<PathBuf> {
        let path = self.index.as_ref().and_then(|i| i.get(display)).map(PathBuf::from).filter(|p| p.exists());
        if path.is_none() && self.job.is_none() && self.reindexed.is_none_or(|t| t.elapsed() > Duration::from_secs(600)) {
            self.reindexed = Some(Instant::now());
            let force = self.index.is_some();
            let _ = tx.send(Event::Status("tosu unavailable — indexing library in background; recovering scores from lazer".into()));
            let display = display.to_string();
            self.job = Some(std::thread::spawn(move || {
                let index = ms_data::lazer_index::load_or_build(force);
                if !force && !index.contains_key(&display) { ms_data::lazer_index::load_or_build(true) } else { index }
            }));
        }
        path
    }
}

fn connect() -> Option<WebSocket<TcpStream>> {
    let addr = TOSU_HOST.to_socket_addrs().ok()?.next()?;
    let stream = TcpStream::connect_timeout(&addr, Duration::from_millis(600)).ok()?;
    stream.set_read_timeout(Some(Duration::from_millis(600))).ok()?;
    let _ = stream.set_nodelay(true);
    tungstenite::client(format!("ws://{TOSU_HOST}/websocket/v2").as_str(), stream).ok().map(|x| x.0)
}

/// Sleeps on the socket: no frame, no work. `client` reads the player's chosen client for the fallback's message.
pub fn run(mut t: Tracker, halt: Arc<AtomicBool>, client: impl Fn() -> String) {
    let mut tail = LogTail { log: None, display: None, index: None, reindexed: None, job: None };
    while !halt.load(Ordering::Relaxed) {
        let Some(mut socket) = connect() else {
            t.live = None;
            let obs = tail.poll(&client(), &t.tx);
            t.observed(obs, false);
            std::thread::sleep(Duration::from_secs(1));
            continue;
        };
        let mut handled: Option<Instant> = None;
        while !halt.load(Ordering::Relaxed) {
            match socket.read() {
                Ok(Message::Text(text)) => {
                    if t.live.is_some() && handled.is_some_and(|h| h.elapsed() < GAMEPLAY_FRAME) {
                        continue; // gameplay: never spend CPU next to the game on frames nothing shows
                    }
                    handled = Some(Instant::now());
                    let Ok(d) = serde_json::from_str::<Value>(&text) else { continue };
                    let obs = t.frame(&d);
                    t.observed(obs, true);
                }
                Ok(_) => {}
                // No frame for a while: tosu is up without a game.
                Err(tungstenite::Error::Io(e)) if matches!(e.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut) => {
                    t.live = None;
                    t.observed(waiting(), true);
                }
                Err(_) => break,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::{channel, Receiver};

    fn tracker() -> (Tracker, Receiver<Event>, Arc<Queue>) {
        let (tx, rx) = channel();
        let q = Arc::new(Queue::default());
        (Tracker::new(tx, q.clone(), Arc::default()), rx, q)
    }

    fn state(name: &str, live: f64, chart: &Path) -> Value {
        json!({"client": "lazer", "state": {"name": name}, "game": {"paused": false}, "play": {"mods": {"array": [], "rate": 1.0}, "hits": {"300": 1}},
               "beatmap": {"id": 5, "mode": {"number": 3}, "time": {"live": live, "firstObject": 0, "lastObject": 10000}},
               "files": {"beatmap": chart.file_name().unwrap().to_str().unwrap()}, "folders": {"songs": chart.parent().unwrap().to_str().unwrap()}})
    }

    fn chart() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ms-watch-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // Each test its own file: a rewrite under a parallel test changes the mtime its cache key was made from.
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let p = dir.join(format!("chart{}", N.fetch_add(1, Ordering::Relaxed)));
        std::fs::write(&p, "x").unwrap();
        p
    }

    fn drain(q: &Queue) -> Vec<&'static str> {
        std::iter::from_fn(|| q.get(Duration::from_millis(1))).map(|t| match t {
            Task::Start(_) => "start",
            Task::Abort(_) => "abort",
            Task::Void => "void",
            Task::Import => "import",
            _ => "other",
        }).collect()
    }

    #[test]
    fn a_map_tosu_could_not_parse_goes_by_its_file() {
        let (mut t, _rx, _q) = tracker();
        let c = chart();
        let mut d = state("selectPlay", 0., &c);
        d["beatmap"]["mode"]["number"] = json!(0);
        d["beatmap"]["time"]["lastObject"] = json!(0);
        d["beatmap"]["time"]["mp3Length"] = json!(1000);
        assert_eq!(t.frame(&d).note, "not an osu!mania map");
        std::fs::write(&c, "osu file format v14\n\n[General]\nMode: 3\n").unwrap();
        assert_eq!(t.frame(&d).path.as_ref(), Some(&c));
        d["state"]["name"] = json!("play");
        d["beatmap"]["time"]["live"] = json!(500);
        t.frame(&d);
        assert_eq!(t.play.lock().unwrap().as_ref().unwrap()["progress"], 0.5);
    }

    #[test]
    fn an_attempt_starts_on_gameplay_and_aborts_on_leaving_it() {
        let (mut t, _rx, q) = tracker();
        let c = chart();
        t.frame(&state("play", 100., &c));
        t.frame(&state("play", 400., &c));
        let p = t.play.lock().unwrap().clone().unwrap();
        assert_eq!(p["pos"], 400.);
        assert!((p["played_seconds"].as_f64().unwrap() - 0.3).abs() < 1e-9);
        let s = t.frame(&state("menu", 0., &c));
        assert_eq!(s.path.as_ref(), Some(&c));
        assert_eq!(drain(&q), ["start", "abort"]);
        assert!(t.play.lock().unwrap().is_none());
    }

    #[test]
    fn the_preview_position_is_not_a_restart_but_a_rewind_is() {
        let (mut t, _rx, q) = tracker();
        let c = chart();
        t.frame(&state("play", 9000., &c));
        t.frame(&state("play", 100., &c)); // within 5 s: gameplay took over from the preview
        assert_eq!(drain(&q), ["start", "start"]);
        t.play.lock().unwrap().as_mut().unwrap()["t"] = json!(now() - 10.);
        t.play.lock().unwrap().as_mut().unwrap()["pos"] = json!(9000.);
        t.frame(&state("play", 100., &c));
        assert_eq!(drain(&q), ["abort", "start"]);
    }

    #[test]
    fn a_result_after_gameplay_is_fresh_and_goes_out_once() {
        let (mut t, rx, q) = tracker();
        let c = chart();
        t.frame(&state("play", 100., &c));
        let mut r = state("resultScreen", 0., &c);
        r["profile"] = json!({"name": "me"});
        r["resultsScreen"] = json!({"playerName": "me", "mode": {"name": "mania"}, "createdAt": "2030-01-01T00:00:00Z", "mods": {"array": [], "name": ""},
                                    "hits": {"geki": 3, "300": 1}, "accuracy": 99.0, "rank": "S", "maxCombo": 4, "score": 1000});
        t.frame(&r);
        t.frame(&r);
        let got: Vec<Event> = rx.try_iter().collect();
        let scores: Vec<_> = got.iter().filter_map(|e| if let Event::Score { fresh, attempt, .. } = e { Some((*fresh, attempt.is_some())) } else { None }).collect();
        assert_eq!(scores, [(true, true)]);
        assert_eq!(drain(&q), ["start"]);
    }

    #[test]
    fn autoplay_is_never_an_attempt() {
        let (mut t, _rx, q) = tracker();
        let c = chart();
        let mut f = state("play", 100., &c);
        f["play"]["mods"]["array"] = json!([{"acronym": "AT"}]);
        t.frame(&f);
        assert!(drain(&q).is_empty());
    }

    #[test]
    fn the_first_poll_before_autoplays_mods_is_voided() {
        let (mut t, _rx, q) = tracker();
        let c = chart();
        let mut f = state("play", 100., &c);
        t.frame(&f); // tosu's first gameplay poll, before Autoplay's mods
        f["play"]["mods"]["array"] = json!([{"acronym": "AT"}]);
        t.frame(&f);
        t.frame(&state("menu", 0., &c));
        assert_eq!(drain(&q), ["start", "void"]);
        let mut dt = state("play", 100., &c);
        dt["play"]["mods"]["array"] = json!([{"acronym": "DT"}]);
        t.frame(&dt);
        t.frame(&state("menu", 0., &c));
        assert_eq!(drain(&q), ["start", "abort"]);
    }

    #[test]
    fn an_outage_is_not_evidence_of_a_failed_play() {
        let (mut t, _rx, q) = tracker();
        let c = chart();
        t.frame(&state("play", 100., &c));
        t.observed(waiting(), true);
        t.frame(&state("menu", 0., &c));
        assert_eq!(drain(&q), ["start", "void"]);
    }

    #[test]
    fn pause_positions_are_recorded_during_play() {
        let (mut t, _rx, _q) = tracker();
        let c = chart();
        for (live, paused) in [(-1600., true), (1000., false), (2000., false), (2100., true), (2100., true), (2200., false), (5000., true)] {
            let mut f = state("play", live, &c);
            f["game"]["paused"] = json!(paused);
            t.frame(&f);
        }
        assert_eq!(t.play.lock().unwrap().as_ref().unwrap()["pauses"], json!([2100, 5000]));
        t.frame(&state("resultScreen", 5000., &c));
        assert_eq!(t.done.as_ref().unwrap()["pauses"], json!([2100, 5000]));
    }

    #[test]
    fn a_stable_result_counts_only_after_an_own_attempt_not_a_watched_replay() {
        for own in [true, false] {
            let (mut t, rx, _q) = tracker();
            let c = chart();
            let sha = c.file_name().unwrap().to_str().unwrap().to_string();
            // Already linked: the test must not write the real stable store.
            let st = std::fs::metadata(&c).unwrap();
            t.stable_files.insert((c.clone(), st.modified().ok(), st.len()), c.clone());
            t.stable_root = c.parent().map(Path::to_path_buf);
            t.done = Some(json!({"sha": sha, "t": 0, "own": own}));
            let mut r = state("resultScreen", 0., &c);
            r["client"] = json!("stable");
            r["profile"] = json!({"name": "Guest"});
            r["resultsScreen"] = json!({"playerName": "wd", "mode": {"name": "mania"}, "mods": {"array": []}, "hits": {"geki": 9, "300": 1, "katu": 0, "100": 0, "50": 0, "0": 0}, "score": 1, "createdAt": "2026-10-31T00:00:00Z"}); // tosu: stable's own name
            t.frame(&r);
            let got = rx.try_iter().filter(|e| matches!(e, Event::Score { .. })).count();
            assert_eq!(got, usize::from(own));
        }
    }

    #[test]
    fn a_silent_tosu_reconnects_with_an_import() {
        let (mut t, rx, q) = tracker();
        let c = chart();
        t.observed(waiting(), true);
        let o = t.frame(&state("menu", 0., &c));
        t.observed(o, true);
        assert_eq!(drain(&q), ["import"]);
        assert_eq!(rx.try_iter().filter(|e| matches!(e, Event::Obs(_))).count(), 2);
    }

    #[test]
    fn websocket_frames_reach_the_tracker() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (s, _) = listener.accept().unwrap();
            let mut ws = tungstenite::accept(s).unwrap();
            ws.send(Message::text(json!({"error": "not_ready"}).to_string())).unwrap();
            let _ = ws.close(None);
        });
        let stream = TcpStream::connect(addr).unwrap();
        let (mut ws, _) = tungstenite::client(format!("ws://{addr}/websocket/v2").as_str(), stream).unwrap();
        let Ok(Message::Text(text)) = ws.read() else { panic!("no frame") };
        let (mut t, _rx, _q) = tracker();
        assert_eq!(t.frame(&serde_json::from_str(&text).unwrap()).note, waiting().note);
        server.join().unwrap();
    }
}
