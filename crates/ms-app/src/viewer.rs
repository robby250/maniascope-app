//! The selected chart: its own analysis, independent of the recommender. The newest request replaces pending ones.
use ms_calc::card::{card, describe, display_name};
use ms_calc::chart::{parse, Chart};
use ms_calc::features::compute_selected;
use ms_calc::params::Params;
use ms_calc::skills::skills;
use ms_calc::val::{Dict, Val};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};

const REMEMBERED: usize = 64;

pub struct Entry {
    pub name: String,
    pub rating: f64,
    pub parts: Vec<String>,
}

pub struct Tech {
    pub t: f64,
    pub pattern: f64,
    pub rhythm: f64,
    pub why: String,
}

/// One analysis, in display form: frames read this, never the result dictionary.
pub struct Analysis {
    pub path: PathBuf,
    pub rate: f64,
    pub sv: bool,
    pub keys: i64,
    pub title: String,
    pub artist: String,
    pub version: String,
    pub creator: String,
    pub overall: f64,
    pub notes: usize,
    pub ln_notes: f64,
    pub nps: Option<f64>,
    pub span: Option<(f64, f64)>, // first note, last end, ms
    pub timeline: Vec<f64>,
    pub entries: Vec<Entry>,
    pub tags: Vec<String>,
    pub bars: Vec<(String, f64)>, // display name, value: this keymode's skills
    pub names: Vec<(String, String)>, // key, display name
    pub tech: Vec<Tech>,
    pub sv_ignored: bool,
    pub dan_vibro: ms_rec::dans::Vibro,
}

/// recdata.raw_nps: chord-inclusive heads over first-to-last head seconds.
pub fn raw_nps(c: &Chart, rate: f64) -> Option<f64> {
    let span = c.t.last()? - c.t.first()?;
    (rate.is_finite() && rate > 0. && span > 0.).then(|| c.len() as f64 * rate * 1000. / span)
}

fn strings(v: Option<&Val>) -> Vec<String> {
    match v {
        Some(Val::List(l) | Val::Tuple(l)) => l.iter().filter_map(|x| if let Val::Str(s) = x { Some(s.clone()) } else { None }).collect(),
        _ => Vec::new(),
    }
}

fn text(d: &Dict, k: &str) -> String {
    if let Some(Val::Str(s)) = d.get(k) { s.clone() } else { String::new() }
}

pub fn analyse(path: &Path, rate: f64, sv: bool, prm: &Params) -> Result<Analysis, String> {
    let data = std::fs::read(path).map_err(|e| e.to_string())?;
    let chart = parse(&data).map_err(|e| format!("{e:?}"))?;
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| compute_selected(&data, &chart, rate, sv, prm))).map_err(|_| "internal error (see crash.log)".to_string())??;
    let sc = res.dict("scores").ok_or("no scores")?;
    let keys = chart.keys as i64;
    let names: Vec<(String, String)> = skills(chart.keys).iter().map(|s| (s.name().to_string(), display_name(*s).to_string())).collect();
    let entries = card(&res).iter().filter_map(|e| if let Val::Dict(e) = e { Some(Entry { name: text(e, "name"), rating: e.f("rating"), parts: strings(e.get("parts")) }) } else { None }).collect();
    let tech = match res.get("tech_where") {
        Some(Val::List(l)) => l.iter().filter_map(|w| if let Val::Dict(w) = w { Some(Tech { t: w.f("t"), pattern: w.f("pattern"), rhythm: w.f("rhythm"), why: text(w, "why") }) } else { None }).collect(),
        _ => Vec::new(),
    };
    let notes: Vec<(f64, usize)> = chart.t.iter().zip(&chart.col).map(|(t, c)| (*t, *c as usize)).collect();
    Ok(Analysis {
        path: path.to_path_buf(), rate, sv, keys, overall: sc.f("overall"), notes: res.fget("notes", 0.) as usize, ln_notes: res.fget("ln_notes", 0.), nps: raw_nps(&chart, rate),
        span: (chart.len() >= 2).then(|| (chart.t[0], chart.last_end())), timeline: res.get("timeline").map_or_else(Vec::new, Val::list_f), entries, tags: describe(&res, &prm.k).1,
        bars: names.iter().map(|(k, n)| (n.clone(), sc.fget(k, 0.))).collect(), names, tech, sv_ignored: matches!(res.get("sv"), Some(Val::Str(s)) if s == "ignored"),
        dan_vibro: ms_rec::dans::vibro_runs(&notes, rate), title: chart.title.clone(), artist: chart.artist.clone(), version: chart.version.clone(), creator: chart.creator.clone(),
    })
}

#[derive(Clone, PartialEq)]
pub struct Request {
    pub generation: u64,
    pub path: PathBuf,
    pub rate: f64,
    pub sv: bool,
}

pub struct Done {
    pub request: Request,
    pub result: Result<Arc<Analysis>, String>,
}

#[derive(Default)]
struct Slot {
    pending: Option<Request>,
    stopped: bool,
}

#[derive(Default)]
pub struct Viewer {
    slot: Mutex<Slot>,
    wake: Condvar,
    memory: Mutex<VecDeque<(Option<std::time::SystemTime>, Arc<Analysis>)>>,
}

fn stamp(p: &Path) -> Option<std::time::SystemTime> {
    std::fs::metadata(p).ok()?.modified().ok()
}

impl Viewer {
    pub fn cached(&self, r: &Request) -> Option<Arc<Analysis>> {
        let mut memory = self.memory.lock().unwrap();
        let now = stamp(&r.path);
        let at = memory.iter().position(|(at, a)| a.path == r.path && a.rate == r.rate && a.sv == r.sv && *at == now)?;
        let hit = memory.remove(at)?;
        memory.push_back(hit.clone());
        Some(hit.1)
    }

    pub fn request(&self, r: Option<Request>) {
        self.slot.lock().unwrap().pending = r;
        self.wake.notify_one();
    }

    pub fn stop(&self) {
        self.slot.lock().unwrap().stopped = true;
        self.wake.notify_one();
    }

    /// Runs on its own thread until `stop`.
    pub fn run(&self, prm: &Params, done: impl Fn(Done)) {
        loop {
            let request = {
                let mut slot = self.slot.lock().unwrap();
                while slot.pending.is_none() && !slot.stopped {
                    slot = self.wake.wait(slot).unwrap();
                }
                if slot.stopped {
                    return;
                }
                slot.pending.take().unwrap()
            };
            let before = stamp(&request.path);
            let result = analyse(&request.path, ms_calc::py::round_digits(request.rate, 4), request.sv, prm).map(Arc::new);
            if before != stamp(&request.path) {
                continue; // the chart changed under the analysis: a later observation asks again
            }
            if let Ok(a) = &result {
                let mut memory = self.memory.lock().unwrap();
                memory.push_back((before, a.clone()));
                if memory.len() > REMEMBERED {
                    memory.pop_front();
                }
            }
            done(Done { request, result });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ms_calc::features::standard_params;
    use std::sync::mpsc::channel;

    fn write(path: &Path, gap: i64) {
        let mut text = "osu file format v14\n\n[General]\nMode: 3\n\n[Difficulty]\nCircleSize:4\nOverallDifficulty:8\n\n[TimingPoints]\n0,500,4,1,0,100,1,0\n\n[HitObjects]\n".to_string();
        for i in 0..300 {
            text.push_str(&format!("{},192,{},1,0,0:0:0:0:\n", (i % 4) * 128 + 64, 1000 + i * gap));
        }
        std::fs::write(path, text).unwrap();
    }

    #[test]
    fn content_rate_and_sv_each_miss_the_memory_and_the_latest_request_wins() {
        let dir = std::env::temp_dir().join(format!("ms-viewer-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (old, new) = (dir.join("old.osu"), dir.join("new.osu"));
        write(&old, 80);
        write(&new, 90);
        let viewer = Arc::new(Viewer::default());
        let request = |path: &PathBuf, generation| Request { generation, path: path.clone(), rate: 1., sv: true };
        viewer.request(Some(request(&old, 1)));
        viewer.request(Some(request(&new, 2))); // the viewer never got to the old one
        let (tx, rx) = channel();
        let thread = { let v = viewer.clone(); std::thread::spawn(move || { let prm = standard_params(); v.run(&prm, |d| tx.send(d).unwrap()); }) };
        let done = rx.recv().unwrap();
        assert_eq!(done.request.generation, 2);
        let hit = request(&new, 3);
        assert!(viewer.cached(&hit).is_some());
        assert!(viewer.cached(&Request { rate: 1.1, ..hit.clone() }).is_none());
        assert!(viewer.cached(&Request { sv: false, ..hit.clone() }).is_none());
        std::thread::sleep(std::time::Duration::from_millis(20));
        write(&new, 70);
        assert!(viewer.cached(&hit).is_none(), "an edited chart is not the analysed one");
        viewer.stop();
        thread.join().unwrap();
        let _ = std::fs::remove_dir_all(dir);
    }
}
