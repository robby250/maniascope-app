//! The app's state and everything that is not drawing: messages from the worker, the watcher and the analysis thread.
use crate::fmt::{self, Line};
use crate::settings::{Ui as Settings, MODES};
use crate::tosu::Tosu;
use crate::viewer::{self, Analysis, Viewer};
use crate::watcher::{self, Event, Obs, Tracker};
use eframe::egui;
use ms_calc::features::standard_params;
use ms_data::feedback;
use ms_data::imports::score_allowed;
use ms_data::paths::rec_dir;
use ms_data::store::{connect, kv_get, kv_set, now, played_keys, set_tracking, tracking_enabled, Db};
use ms_rec::catalog::Pick;
use ms_rec::engine::Card;
use ms_rec::history::{default_client, Shown};
use ms_rec::navigation::{perform, same_map};
use ms_rec::stats::Current;
use ms_rec::worker::{spawn, Config, Done as Sent, Hooks, Out, Queue, Task, Verdict};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

pub const DEBUG_ENV: &str = "MANIASCOPE_DEBUG";

pub enum Msg {
    Watch(Event),
    Rec(Out),
    Analysed(viewer::Done),
    Raise,
    Status(String),
    TosuReady,
    Web(Box<crate::windows::WebInfo>),
}

/// Sends to the UI and wakes it: a hidden window still runs `logic`.
#[derive(Clone)]
pub struct Mail {
    tx: Sender<Msg>,
    ctx: egui::Context,
}

impl Mail {
    pub fn send(&self, m: Msg) {
        let _ = self.tx.send(m);
        self.ctx.request_repaint();
    }
}

pub struct View {
    pub shown: Vec<Pick>,
    pub pending: Option<Out>,
    pub ready: bool,
    pub target: Option<Pick>,
    pub target_msg: String,
    pub target_tip: Option<String>,
    pub note: String,
    pub note_tip: String,
    pub loaded: bool,
    pub analysing: Option<(String, usize, usize)>,
    pub revision: u64,
    pub hover: bool,
    pub settings: bool,
}

pub struct Selected {
    pub path: PathBuf,
    pub rate: f64,
    pub sv: bool,
    pub result: Result<Arc<Analysis>, String>,
}

/// A newer selection is being analysed: its own title from the file header, the old analysis cleared so it can't be read as
/// the new map's.
pub struct Pending {
    pub title: String,
    pub artist: String,
    pub version: String,
    pub creator: String,
    pub context: String,
}

#[derive(Default)]
pub struct Open {
    pub menu: bool,
    pub profile: Option<bool>, // first run
    pub profile_text: String,
    pub overlay_confirm: bool,
    pub history: bool,
    pub history_only: bool,
    pub history_text: String,
    pub web: bool,
    pub compare: bool,
    pub open_osu: bool,
    pub open_text: String,
}

pub struct App {
    pub s: Settings,
    pub rx: Receiver<Msg>,
    pub mail: Mail,
    pub q: Arc<Queue>,
    pub halt: Arc<AtomicBool>,
    pub play: Arc<Mutex<Option<Value>>>,
    pub busy: Arc<AtomicBool>,
    pub game: Arc<AtomicBool>,
    pub client_name: Arc<Mutex<String>>,
    pub db: Db,
    pub viewer: Arc<Viewer>,
    pub obs: Obs,
    pub local: Option<PathBuf>,
    pub shown: Option<Selected>,
    pub pending: Option<Pending>,
    pub generation: u64,
    pub requested: Option<(PathBuf, Option<SystemTime>, f64, bool)>,
    pub live: Option<f64>,
    pub card: Option<Box<Card>>,
    pub card_note: Option<String>,
    pub views: HashMap<&'static str, View>,
    pub stats: Option<Box<Current>>,
    pub stats_open: bool,
    pub stats_keys: Option<i64>, // None: all keymodes
    pub stats_wait: Option<(String, usize, usize)>,
    pub expanded: HashSet<(i64, String)>,
    pub chips_open: HashSet<(i64, String)>,
    pub revision: u64,
    pub selection_id: u64,
    pub units: Shown,
    pub result: Option<(Vec<Line>, String)>,
    pub status_text: String,
    pub bg_progress: Option<(usize, usize)>,
    pub blocker: Option<String>,
    pub foot: String,
    pub open: Open,
    pub save_at: Option<Instant>,
    pub tosu: Tosu,
    pub tracking: Option<bool>, // None: release builds have no pause switch
    pub debug: bool,
    pub web: crate::windows::WebView,
    pub compare: crate::windows::CompareView,
    pub resized: Option<(f32, f32)>,
    pub clipboard: Option<arboard::Clipboard>,
    keys_checked: Option<Instant>,
}

fn view() -> View {
    View { shown: Vec::new(), pending: None, ready: false, target: None, target_msg: String::new(), target_tip: None, note: String::new(), note_tip: String::new(), loaded: false, analysing: None, revision: 0, hover: false, settings: false }
}

/// Only source/debug runs get the pause switch (user 2026-10-01). Release builds have no way back on, so an old pause is
/// closed instead of excluding every later play.
/// The welcome prompt: once, and never on a linked install (Python only asked in the exe, so source runs never set the flag).
fn asks_profile(db: &Db) -> bool {
    ["first_run_asked", "user_id", "website_username"].iter().all(|k| kv_get(db, k).is_none())
}

fn tracking_switch(db: &Db, debug: bool) -> Option<bool> {
    if debug {
        return Some(tracking_enabled(db).unwrap_or(true));
    }
    let _ = set_tracking(db, true, None, None);
    None
}

impl App {
    /// The window's state, with no worker, watcher or tosu behind it.
    fn bare(ctx: &egui::Context, s: Settings, db_path: &std::path::Path) -> Result<App, String> {
        let (tx, rx) = channel();
        let mail = Mail { tx, ctx: ctx.clone() };
        let db = connect(db_path).map_err(|e| e.to_string())?;
        let debug = std::env::var(DEBUG_ENV).is_ok_and(|v| v == "1");
        let client = kv_get(&db, "client").and_then(|v| v.as_str().map(str::to_string)).unwrap_or_else(|| default_client().to_string());
        let miss_ratio = kv_get(&db, "stable_miss_ratio").and_then(|v| v.as_f64()).unwrap_or(0.88);
        let (q, halt) = (Arc::new(Queue::default()), Arc::new(AtomicBool::new(false)));
        let (busy, game) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicBool::new(false)));
        let play = Arc::new(Mutex::new(None));
        let tracking = tracking_switch(&db, debug);
        let mut app = App {
            s, rx, mail: mail.clone(), q: q.clone(), halt: halt.clone(), play: play.clone(), busy: busy.clone(), game: game.clone(), client_name: Arc::new(Mutex::new(client.clone())), viewer: Arc::default(), db,
            obs: Obs { note: "starting…".into(), ..Obs::default() }, local: None, shown: None, pending: None, generation: 0, requested: None, live: None, card: None, card_note: None,
            views: MODES.into_iter().map(|m| (m, view())).collect(), stats: None, stats_open: false, stats_keys: Some(7), stats_wait: None, expanded: HashSet::new(), chips_open: HashSet::new(), revision: 0, selection_id: 0,
            units: Shown { client, miss_ratio }, result: None, status_text: String::new(), bg_progress: None, blocker: None, foot: String::new(), open: Open::default(), save_at: None, tosu: Tosu::new(mail.clone()), tracking, debug,
            web: Default::default(), compare: Default::default(), resized: None, clipboard: arboard::Clipboard::new().ok(), keys_checked: None,
        };
        app.auto_keys(true);
        if !debug && asks_profile(&app.db) {
            let _ = kv_set(&app.db, "first_run_asked", &json!(true));
            app.open.profile = Some(true);
            app.open.profile_text = kv_get(&app.db, "website_username").and_then(|v| v.as_str().map(str::to_string)).unwrap_or_default();
        }
        Ok(app)
    }

    pub fn new(ctx: &egui::Context) -> Result<App, String> {
        let db_path = rec_dir().join("rec.db");
        let mut app = App::bare(ctx, Settings::load(), &db_path)?;
        app.tosu.start(app.s.overlay);
        app.sync_rec();
        // The worker: the UI thread never waits on it.
        let (mail, q, halt, play, busy, game) = (app.mail.clone(), app.q.clone(), app.halt.clone(), app.play.clone(), app.busy.clone(), app.game.clone());
        let m = mail.clone();
        let hooks = Hooks { emit: Arc::new(move |o| m.send(Msg::Rec(o))), busy: { let b = busy.clone(); Arc::new(move || b.load(Ordering::Relaxed)) }, game: { let g = game.clone(); Arc::new(move || g.load(Ordering::Relaxed)) } };
        let _ = spawn(crate::public::load, q.clone(), hooks, halt.clone(), db_path);
        let (v, m) = (app.viewer.clone(), mail.clone());
        std::thread::Builder::new().name("selected".into()).spawn(move || { let prm = standard_params(); v.run(&prm, |d| m.send(Msg::Analysed(d))); }).map_err(|e| e.to_string())?;
        let (m, stop) = (mail, halt);
        let (tx2, rx2) = channel();
        std::thread::spawn(move || for e in rx2 { m.send(Msg::Watch(e)); });
        let names = app.client_name.clone();
        let tracker = Tracker::new(tx2, q, play);
        std::thread::Builder::new().name("watcher".into()).spawn(move || watcher::run(tracker, stop, move || names.lock().unwrap().clone())).map_err(|e| e.to_string())?;
        Ok(app)
    }

    /// Until the first lazer import lands (or the player picks), recheck what they play.
    fn auto_keys(&mut self, first: bool) {
        if self.s.keys_auto.is_empty() {
            return;
        }
        if !first && self.keys_checked.is_some_and(|t| t.elapsed() < Duration::from_secs(15)) {
            return;
        }
        self.keys_checked = Some(Instant::now());
        let played = played_keys(&self.db, 300, 0.1).unwrap_or_default();
        if played.is_empty() {
            return;
        }
        for m in std::mem::take(&mut self.s.keys_auto) {
            if first {
                self.s.keys.insert(m, played.clone());
            } else {
                self.set_keys(&m, played.clone());
            }
        }
    }

    pub fn request_save(&mut self) {
        self.save_at = Some(Instant::now() + Duration::from_millis(700));
    }

    pub fn sync_rec(&mut self) {
        self.revision += 1;
        // Settings apply at once; this answers "how do I refresh?" (Less, 2026-10-31).
        let mode = self.mode();
        if !self.s.keys[mode].is_empty() {
            self.views.get_mut(mode).expect("a tab").note = "Applied · updating the list…".into();
        }
        for v in self.views.values_mut() {
            v.revision = self.revision;
        }
        let s = &self.s;
        self.q.put(Task::Configure(Box::new(Config {
            mode: s.mode.clone(), keys: s.keys.clone(), action: s.next_action.clone(), revision: self.revision, practice_skills: Some(s.practice.clone()), nps_focus: Some(s.nps_focus),
            acc_targets: s.targets.clone(), web_bias: s.web_bias.clone(), web_weighted: s.web_weighted.clone(), converts: s.converts.clone(), pp_settings: Some(s.pp.clone()), pp_mix: Some(s.mix.clone()), skill_weights: Some(s.skill_weights.clone()), skill_tuning: Some(s.skill_tuning()),
        })));
        self.request_save();
    }

    pub fn mode(&self) -> &'static str {
        MODES.into_iter().find(|m| *m == self.s.mode).unwrap_or("pp")
    }

    pub fn set_mode(&mut self, mode: &str) {
        if MODES.contains(&mode) && mode != self.s.mode {
            self.s.mode = mode.to_string();
            self.sync_rec();
        }
    }

    pub fn set_keys(&mut self, mode: &str, keys: Vec<i64>) {
        self.s.keys.insert(mode.to_string(), keys.clone());
        self.s.keys_auto.remove(mode);
        let v = self.views.get_mut(mode).expect("a tab");
        v.shown.clear();
        (v.ready, v.pending) = (false, None);
        if v.target.as_ref().is_some_and(|t| !keys.contains(&t.keys)) {
            v.target = None;
            v.target_msg.clear();
        }
        v.note = if keys.is_empty() { "Select at least one keymode".into() } else { "Refreshing…".into() };
        self.sync_rec();
    }

    pub fn set_skills(&mut self, selected: Vec<String>) {
        let selected = ms_model::skills::normalize(&selected);
        if selected == self.s.practice {
            return;
        }
        self.s.practice = selected;
        let v = self.views.get_mut("skills").expect("a tab");
        v.shown.clear();
        (v.ready, v.pending, v.target) = (false, None, None);
        v.target_msg.clear();
        v.note = format!("Finding {} maps at your current level…", ms_model::skills::label(&self.s.practice));
        self.sync_rec();
    }

    pub fn practice_from_stats(&mut self, keys: i64, group: &str) {
        self.set_keys("skills", vec![keys]);
        self.s.skill_other = 0.; // the Stats link means this skill only
        self.set_skills(vec![group.to_string()]);
        self.set_mode("skills");
        self.set_stats(false);
    }

    pub fn set_stats(&mut self, open: bool) {
        if open != self.stats_open {
            self.stats_open = open;
            self.q.put(Task::Stats(open));
        }
    }

    pub fn next_enabled(&self) -> bool {
        !self.s.keys[self.mode()].is_empty() && self.views[self.mode()].ready
    }

    pub fn next(&mut self, skip: bool) {
        let mode = self.mode();
        self.flush_pool(mode);
        if !self.views[mode].shown.is_empty() {
            self.activate(mode, 0, Some(skip));
        }
    }

    pub fn activate(&mut self, mode: &'static str, index: usize, skip: Option<bool>) {
        let view = &self.views[mode];
        let Some(mut c) = view.shown.get(index).cloned() else { return };
        if view.revision != self.revision || !self.s.keys[mode].contains(&c.keys) || (mode == "skills" && c.practice_skills != self.s.practice) {
            return;
        }
        c.mode = mode.to_string();
        c.selection = if skip.is_none() { "list" } else { "next" }.to_string();
        let selected_at = now();
        let action = if self.live.is_some() { "copy" } else { self.s.next_action.as_str() };
        let done = perform(&mut c, action, &self.units.client);
        self.selection_id += 1;
        let (sel, rev) = (self.selection_id, self.revision);
        let result = Some(Sent { copy: done.copy.clone(), msg: done.msg.clone(), sent: done.sent, unavailable: done.unavailable });
        // Navigation uses the completed snapshot now; the worker only logs this selection and refreshes the rest.
        self.q.put(match skip {
            None => Task::Open { pick: Box::new(c.clone()), mode: Some(mode.into()), revision: Some(rev), result, selected_at: Some(selected_at), selection_id: sel },
            Some(skip) => Task::Next { skip, mode: Some(mode.into()), revision: Some(rev), candidate: Some(Box::new(c.clone())), result, selected_at: Some(selected_at), selection_id: sel },
        });
        if done.unavailable {
            self.status_text = done.msg.clone();
        } else {
            let msg = format!("{} · set {} manually", done.msg, c.var);
            self.on_rec(Out::Target { mode: mode.to_string(), revision: rev, selection_id: sel, c: Some(Box::new(c.clone())), msg, copy: done.copy.clone() });
        }
        let id = c.identity();
        for v in self.views.values_mut() {
            v.shown.retain(|o| o.identity() != id);
            v.ready = !v.shown.is_empty();
        }
    }

    pub fn flush_pool(&mut self, mode: &'static str) {
        let (revision_now, selection) = (self.revision, self.selection_id);
        let view = self.views.get_mut(mode).expect("a tab");
        let Some(Out::Pool { revision, selection_id, phase, shown, note, summary, progress, .. }) = view.pending.take() else { return };
        if revision != revision_now || selection_id < selection {
            return;
        }
        view.loaded = true;
        if !shown.is_empty() {
            self.blocker = None;
        }
        let phase = match phase.as_str() { "warmup" => "Warming up", "build" => "Warmed up", "push" => "Strong form", "recover" => "Easing off", "fading" => "Fading", other => other };
        view.note = if note.is_empty() { phase.to_string() } else { format!("{phase} · {note}") };
        view.note_tip = [view.note.as_str(), summary.as_str()].into_iter().filter(|t| !t.is_empty()).collect::<Vec<_>>().join("\n");
        let (done, total) = progress.unwrap_or((0, 0));
        view.analysing = (total > 0).then(|| (format!("Analysing maps · {} / {}", ms_rec::engine::commas(done), ms_rec::engine::commas(total)), done, total));
        view.ready = !shown.is_empty();
        view.shown = shown;
    }

    fn set_target(&mut self, mode: &str, c: Option<Box<Pick>>, msg: String, copy: Option<String>) {
        let Some(view) = MODES.into_iter().find(|m| *m == mode).and_then(|m| self.views.get_mut(m)) else { return };
        view.target_tip = Some([Some(msg.as_str()), c.as_ref().map(|c| c.why.as_str()), copy.as_deref()].into_iter().flatten().filter(|t| !t.is_empty()).collect::<Vec<_>>().join("\n")).filter(|t| !t.is_empty());
        view.target = c.map(|c| *c);
        view.target_msg = msg;
        if let (Some(text), Some(clip)) = (copy, self.clipboard.as_mut()) {
            let _ = clip.set_text(text);
        }
        self.overlay_update();
    }

    pub(crate) fn on_rec(&mut self, msg: Out) {
        match msg {
            Out::Stats(data) => self.stats = Some(data),
            Out::Status { text, progress, blocking, learning } => {
                if blocking {
                    self.blocker = Some(text.clone());
                }
                // A later import/replay note must not hide why there are no recommendations at all.
                let joined: Vec<&str> = self.blocker.iter().map(String::as_str).chain([text.as_str()]).filter(|t| !t.is_empty()).collect();
                self.status_text = if joined.len() == 2 && joined[0] == joined[1] { joined[0].to_string() } else { joined.join("\n") };
                let (done, total) = progress.unwrap_or((0, 0));
                self.bg_progress = (total > 0).then_some((done, total));
                if let Some(on) = learning { // Stats has no list to show meanwhile: it keeps its own banner
                    let eta = text.rsplit_once(" · ~").map(|x| format!(" · ~{}", x.1)).unwrap_or_default();
                    self.stats_wait = (on && total > 0).then(|| (format!("Learning your level from your plays · {} / {} charts analysed{eta} — your stats fill in as this runs", ms_rec::engine::commas(done), ms_rec::engine::commas(total)), done, total));
                }
            }
            Out::Pool { mode, revision, selection_id, phase, shown, note, summary, progress } => {
                let Some(m) = MODES.into_iter().find(|m| *m == mode) else { return };
                if revision != self.revision || selection_id < self.selection_id {
                    return;
                }
                let view = self.views.get_mut(m).expect("a tab");
                view.pending = Some(Out::Pool { mode, revision, selection_id, phase, shown, note, summary, progress });
                if !view.hover || !view.ready {
                    self.flush_pool(m);
                }
            }
            Out::Result(v) => {
                let v: Box<Verdict> = v;
                self.result = Some((fmt::result_lines(&v), format!("{}\n{}", v.title, v.text)));
                if self.s.auto_next && v.targets.iter().any(|m| *m == self.s.mode) && self.next_enabled() {
                    self.next(false);
                }
            }
            Out::History(text) => self.open.history_text = text,
            Out::Card { c, note, path, rate } => {
                if path.is_some() && (path != self.obs.path || rate != self.obs.rate) {
                    return;
                }
                self.card = c;
                self.card_note = note;
                self.overlay_update();
            }
            Out::Copy { text, msg } => {
                if let Some(clip) = self.clipboard.as_mut() {
                    let _ = clip.set_text(text);
                }
                self.status_text = msg;
            }
            Out::Target { mode, revision, selection_id, c, msg, copy } => {
                if revision != self.revision || selection_id < self.selection_id {
                    return;
                }
                self.set_target(&mode, c, msg, copy);
            }
        }
    }

    fn on_obs(&mut self, obs: Obs) {
        if let Some(client) = obs.client.as_ref().filter(|c| matches!(c.as_str(), "lazer" | "stable")) {
            if Some(client) != self.obs.client.as_ref() {
                self.q.put(Task::Client(client.clone())); // the worker ignores the one it already serves
                self.units.client = client.clone();
                self.units.miss_ratio = kv_get(&self.db, "stable_miss_ratio").and_then(|v| v.as_f64()).unwrap_or(self.units.miss_ratio);
                *self.client_name.lock().unwrap() = client.clone();
            }
        }
        if (&obs.path, obs.rate, &obs.mods) != (&self.obs.path, self.obs.rate, &self.obs.mods) {
            self.q.put(Task::Selected { path: obs.path.clone(), rate: obs.rate, mods: obs.mods.clone() });
            self.on_rec(Out::Card { c: None, note: obs.path.as_ref().map(|_| "Updating selected-map prediction…".to_string()), path: None, rate: None });
        }
        self.game.store(obs.source == Some("tosu") && obs.path.is_some(), Ordering::Relaxed);
        let down = obs.source != Some("tosu");
        self.obs = obs;
        if down {
            self.tosu.start(self.s.overlay); // helper died or was never up: bring it back
        }
        self.refresh();
    }

    fn on_score(&mut self, score: Value, path: PathBuf, fresh: bool, attempt: Option<Value>) {
        if !score_allowed(&self.db, &score, attempt.as_ref().filter(|_| fresh), true).unwrap_or(true) {
            self.foot = "play excluded: tracking was paused".into();
            return;
        }
        self.q.put(Task::Score { score: score.clone(), path: path.clone(), fresh, play: attempt });
        let mut score = score;
        if let Some(o) = score.as_object_mut() {
            o.remove("mods_list");
        }
        let mut db = feedback::load();
        if feedback::add_score(&mut db, &score, &path.to_string_lossy(), &self.db).unwrap_or(false) {
            let _ = feedback::save(&db);
            self.foot = format!("score saved for calibration: {:.2}% {} ({} total)", score["accuracy"].as_f64().unwrap_or(0.), score["rank"].as_str().unwrap_or(""), db["scores"].as_array().map_or(0, Vec::len));
        }
    }

    pub fn tracking_toggled(&mut self, enabled: bool) {
        self.tracking = Some(enabled);
        let start = self.play.lock().unwrap().as_ref().and_then(|p| p["t"].as_f64());
        let _ = set_tracking(&self.db, enabled, None, start);
        self.q.put(Task::Tracking);
        self.status_text = if enabled { "Tracking resumed for the next play" } else { "Tracking paused — controller plays will be excluded" }.into();
    }

    /// → (path, rate, rate known) for what should be on display now.
    pub fn effective(&self) -> (Option<PathBuf>, f64, bool) {
        let path = self.local.clone().or_else(|| self.obs.path.clone());
        if self.local.is_some() {
            return (path, 1., true);
        }
        (path, self.obs.rate.filter(|r| *r != 0.).unwrap_or(1.), self.obs.rate.is_some())
    }

    /// Scroll changes count unless lazer's selection has Constant Speed.
    fn sv(&self) -> bool {
        !(self.local.is_none() && self.obs.cs)
    }

    /// Reconcile the display with the effective state and ask for an analysis if needed.
    pub fn refresh(&mut self) {
        let (path, rate, _) = self.effective();
        let sv = self.sv();
        let Some(path) = path else {
            self.generation += 1; // invalidate anything in flight
            self.requested = None;
            self.viewer.request(None);
            (self.shown, self.pending) = (None, None);
            self.update_status();
            return;
        };
        if self.shown.as_ref().is_some_and(|s| (&s.path, s.rate, s.sv) == (&path, rate, sv)) {
            // Returning to the already displayed chart must invalidate a newer in-flight selection too. Otherwise its
            // delayed result can replace the chart the user has actually settled on.
            if self.requested.take().is_some() {
                self.generation += 1;
                self.viewer.request(None);
            }
            self.pending = None;
        } else {
            // Repeated observations while this selection is computing must not invalidate its own result or queue it again.
            let stamp = std::fs::metadata(&path).ok().and_then(|m| m.modified().ok());
            let key = (path.clone(), stamp, rate, sv);
            if self.requested.as_ref() != Some(&key) {
                self.requested = Some(key);
                self.generation += 1;
                let request = viewer::Request { generation: self.generation, path: path.clone(), rate, sv };
                match self.viewer.cached(&request) {
                    Some(hit) => {
                        self.viewer.request(None);
                        self.on_analysed(viewer::Done { request, result: Ok(hit) });
                    }
                    None => {
                        self.pending = Some(self.header(&path, rate));
                        self.viewer.request(Some(request));
                    }
                }
            }
        }
        self.update_status();
    }

    fn header(&self, path: &std::path::Path, rate: f64) -> Pending {
        let meta = ms_data::lazer_index::metadata(path).unwrap_or_default();
        let get = |a: &str, b: &str| meta.get(a).or_else(|| meta.get(b)).cloned().unwrap_or_default();
        let keys = get("CircleSize", "CircleSize").split('.').next().unwrap_or("").to_string();
        let mut title = get("TitleUnicode", "Title");
        if title.is_empty() {
            title = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        }
        Pending { title, artist: get("ArtistUnicode", "Artist"), version: get("Version", "Version"), creator: get("Creator", "Creator"), context: format!("{}{rate:.2}×", if keys.is_empty() { String::new() } else { format!("{keys}K  ·  ") }) }
    }

    fn on_analysed(&mut self, d: viewer::Done) {
        if d.request.generation != self.generation {
            return; // includes callbacks queued before a newer selection
        }
        self.requested = None;
        self.pending = None;
        let r = d.request;
        self.shown = Some(Selected { path: r.path, rate: r.rate, sv: r.sv, result: d.result });
        self.update_status();
        self.overlay_update();
    }

    pub fn update_status(&mut self) {
        let o = &self.obs;
        self.foot = if self.local.is_some() {
            "local file — osu! selection ignored".to_string()
        } else if o.source == Some("tosu") && o.path.is_some() {
            // Working is silent: this line is for what needs a look (user 2026-11-01).
            [o.note.as_str(), if o.rate.is_none() { "showing 1.00×" } else { "" }].iter().filter(|t| !t.is_empty()).copied().collect::<Vec<_>>().join(" · ")
        } else if o.source == Some("log") {
            format!("lazer via log — {}{}", o.note, if o.path.is_some() { " · showing 1.00×, tosu reads the real rate" } else { "" })
        } else {
            o.note.clone()
        };
        if self.web.continued && self.web.info.as_ref().is_some_and(|i| i.running) {
            self.foot = [self.foot.as_str(), "website charts continuing"].iter().filter(|t| !t.is_empty()).copied().collect::<Vec<_>>().join(" · ");
        }
    }

    pub fn analysis(&self) -> Option<&Arc<Analysis>> {
        if self.pending.is_some() { None } else { self.shown.as_ref().and_then(|s| s.result.as_ref().ok()) }
    }

    /// What the main window shows, plus the expected accuracy and the pick, for the overlay page.
    pub fn overlay_update(&mut self) {
        if !self.s.overlay {
            return;
        }
        let mut state = json!({});
        if let Some(a) = self.analysis() {
            let step = (a.timeline.len() / 120).max(1);
            let dan = self.s.display.dan.then(|| ms_rec::dans::label(a.keys, a.overall, a.ln_notes / a.notes.max(1) as f64, Some(&a.dan_vibro))).flatten();
            state = json!({"title": a.title, "artist": a.artist, "version": format!("[{}]  {}", a.version, a.creator), "rating": a.overall, "context": fmt::context(a, dan.as_deref(), ""), "nps": a.nps,
                           "ln": (100. * a.ln_notes / a.notes.max(1) as f64).round(), "skills": a.entries.iter().take(self.s.display.skills).map(|e| json!({"name": e.name, "rating": e.rating})).collect::<Vec<_>>(),
                           "timeline": a.timeline.chunks(step).map(|c| c.iter().copied().fold(f64::MIN, f64::max)).collect::<Vec<_>>()});
        }
        if let Some(c) = &self.card {
            state["expected"] = json!(format!("~{:.1}%", 100. * self.units.acc(c.acc_mid)));
            state["form"] = json!(fmt::form_text(c.form, c.acc_mid, &self.units));
        }
        if let Some(p) = &self.views[self.mode()].target {
            state["pick"] = json!({"title": p.title, "expected": format!("~{:.1}%", 100. * self.units.acc(p.pred.acc_mid)), "set": format!("{:.2}× {}", p.rate, fmt::wanted_mod(p.rate))});
        }
        ms_data::tools::write_overlay_state(&state);
    }

    pub fn toggle_overlay(&mut self, on: bool) {
        self.s.overlay = on;
        self.request_save();
        self.tosu.restart(on);
        self.overlay_update();
    }

    /// A PP map picked from the NPS list reads like one in the PP list (user 2026-10-31).
    pub fn card_is_pp(&self) -> bool {
        let pick = self.views[self.mode()].target.as_ref();
        self.mode() == "pp" || matches!((&self.card, pick), (Some(c), Some(p)) if p.purpose == "nps_mix" && same_map(p, None, Some(&c.md5), c.bid))
    }

    pub fn pump(&mut self, ctx: &egui::Context) {
        while let Ok(m) = self.rx.try_recv() {
            match m {
                Msg::Rec(o) => self.on_rec(o),
                Msg::Analysed(d) => self.on_analysed(d),
                Msg::Status(t) => self.foot = t,
                Msg::Raise => ctx.send_viewport_cmd(egui::ViewportCommand::Focus),
                Msg::TosuReady => self.tosu.start(self.s.overlay),
                Msg::Web(info) => self.web_show(*info),
                Msg::Watch(Event::Obs(o)) => self.on_obs(o),
                Msg::Watch(Event::Status(t)) => self.foot = t,
                Msg::Watch(Event::Score { score, path, fresh, attempt }) => self.on_score(score, path, fresh, attempt),
                Msg::Watch(Event::Progress(live)) => {
                    self.live = live;
                    self.busy.store(live.is_some(), Ordering::Relaxed);
                }
            }
        }
        self.auto_keys(false);
        if let Some(at) = self.save_at {
            let left = at.saturating_duration_since(Instant::now());
            if left.is_zero() {
                self.save_at = None;
                self.s.save();
            } else {
                ctx.request_repaint_after(left);
            }
        }
        self.web_tick(ctx);
    }

    pub fn shutdown(&mut self) {
        self.s.save();
        self.halt.store(true, Ordering::Relaxed);
        self.viewer.stop();
        self.tosu.stop();
        self.web_pause();
    }
}

#[cfg(test)]
#[path = "ui_tests.rs"]
mod ui_tests;
