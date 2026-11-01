//! The worker owns the recommendation state and every write to its event log; the UI only queues tasks and renders
//! what `emit` hands back.
use crate::builder::{self, rank, taste, web_factor, Builder, Published, RankInputs, Request, Update, DEFAULT_TARGET, PROFILE, STRUCTURE_VERSION};
use crate::catalog::{one_by_md5, one_by_sha, Inst, Pick};
use crate::engine::{forecast_chart, hashes, refit, Card, Engine, Now, Refit, CALC_ID, PREDICTOR_ID};
use crate::evidence::y_of;
use crate::history::{client_name, default_client, history_text, play_history, recover_results, Shown};
use crate::navigation::{perform, same_map, ACTIONS};
use crate::pp::{self, PpSettings, PESSIMISM};
use crate::stats;
use ms_calc::features::features_batch;
use ms_data::imports::{self, add_live_score, import_library, Imported};
use ms_data::public::Public;
use ms_data::store::{acc_lazer, acc320, connect, kv_get, kv_set, log_event, now, parse_stats, predictable_context, stable_miss_ratio, timestamp, tracking_allowed, Db, Result, Stats, SUPPORTED_KEYS};
use ms_data::{replays, webmaps};
use ms_model::choices::{mix_order, ChoiceList, Prepared, CONVERT, REAL, WEB};
use crate::converts::{self, Setting};
use ms_model::predict::{Consts, FeatStore, HashMap};
use ms_model::session::SESSION_GAP;
use ms_model::skills;
use rusqlite::OptionalExtension;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::time::{Duration, Instant};

pub const MODES: [&str; 3] = ["pp", "nps", "skills"];
const CONVERT_WAIT: Duration = Duration::from_secs(2); // skipping through a list imports nothing
const NOT_RUNNING: &str = "osu! isn't running";
const NPS_CLOCK: f64 = 300.; // s: NPS/Skills predictions follow warm-up cooling without plays this often

#[derive(Clone, Debug, PartialEq)]
pub struct PpMix {
    pub on: bool,
    pub every: usize,
    pub min_gain: f64,
    pub chance: f64,
    pub ht_chance: f64,
}

impl Default for PpMix {
    fn default() -> Self {
        PpMix { on: false, every: 4, min_gain: 3., chance: 0.4, ht_chance: 0.7 }
    }
}

#[derive(Clone, Default)]
pub struct Config {
    pub mode: String,
    pub keys: std::collections::HashMap<String, Vec<i64>>,
    pub action: String,
    pub revision: u64,
    pub practice_skills: Option<Vec<String>>,
    pub nps_focus: Option<f64>,
    pub acc_targets: std::collections::HashMap<String, f64>,
    pub web_bias: std::collections::HashMap<String, f64>,
    pub pp_settings: Option<PpSettings>,
    pub pp_mix: Option<PpMix>,
    pub skill_weights: Option<std::collections::HashMap<String, f64>>,
    pub skill_tuning: Option<skills::Tuning>,
    pub web_weighted: std::collections::HashMap<String, bool>,
    pub converts: std::collections::HashMap<String, BTreeMap<i64, Setting>>,
}

const SELECTED: &str = "selected"; // a convert's mode when it comes from the selected map, not a list

pub enum Task {
    ConvertDone { pick: Box<Pick>, mode: String, error: Option<String> },
    ConvertSelected(i64),
    Configure(Box<Config>),
    Selected { path: Option<PathBuf>, rate: Option<f64>, mods: Option<Vec<Value>> },
    FeatureReady { sha: String, features: Option<Vec<(f64, ms_calc::val::Dict)>>, refit: bool },
    Start(Value),
    Abort(Value),
    Void,
    Score { score: Value, path: PathBuf, fresh: bool, play: Option<Value> },
    Next { skip: bool, mode: Option<String>, revision: Option<u64>, candidate: Option<Box<Pick>>, result: Option<Done>, selected_at: Option<f64>, selection_id: u64 },
    Open { pick: Box<Pick>, mode: Option<String>, revision: Option<u64>, result: Option<Done>, selected_at: Option<f64>, selection_id: u64 },
    Publish,
    Refresh,
    WebUpdated,
    Stats(bool),
    NpsUpdate(Update),
    SkipWarmup,
    Lock(Option<i64>),
    Import,
    Client(String),
    WebsiteUser(String),
    Imported { result: Option<Imported>, recovered: usize, changed: bool, error: Option<String> },
    Tracking,
    History(bool),
    Refit,
    RefitDone(Option<Box<Refit>>),
    ReplayDone { pending: bool, error: Option<String>, fresh: Vec<String> },
}

/// A navigation the UI already performed (the clicked moment's clipboard stays on its thread).
#[derive(Clone)]
pub struct Done {
    pub copy: Option<String>,
    pub msg: String,
    pub sent: bool,
    pub unavailable: bool,
}

pub struct Verdict {
    pub title: String,
    pub text: String,
    pub acc: f64,
    pub expected: Option<f64>,
    pub z: Option<f64>,
    pub rate: Option<f64>,
    pub when: Option<f64>,
    pub news: Vec<String>,
    pub targets: Vec<String>,
}

pub enum Out {
    Status { text: String, progress: Option<(usize, usize)>, blocking: bool, learning: Option<bool> },
    Pool { mode: String, revision: u64, selection_id: u64, phase: String, shown: Vec<Pick>, note: String, summary: String, progress: Option<(usize, usize)> },
    Target { mode: String, revision: u64, selection_id: u64, c: Option<Box<Pick>>, msg: String, copy: Option<String> },
    Copy { text: String, msg: String },
    Card { c: Option<Box<Card>>, note: Option<String>, path: Option<PathBuf>, rate: Option<f64> },
    Result(Box<Verdict>),
    History(String),
    Stats(Box<stats::Current>),
}

/// FIFO, except the selected map's card: it no longer waits behind post-play rebuilds (G835LX 2026-10-05: score, publish
/// and abort queued 10–30 s ahead of it). Post-play rebuilds go after every user action.
#[derive(Default)]
pub struct Queue {
    lanes: Mutex<[VecDeque<Task>; 3]>,
    wake: Condvar,
    pub config_revision: Mutex<u64>,
    pub selected_at: Mutex<Option<Instant>>,
}

impl Queue {
    pub fn put(&self, task: Task) {
        let lane = match task {
            Task::Selected { .. } | Task::FeatureReady { .. } => 0,
            Task::Refresh => 2,
            _ => 1,
        };
        match &task {
            Task::Configure(c) => {
                let mut r = self.config_revision.lock().unwrap();
                *r = (*r).max(c.revision);
            }
            Task::Selected { .. } => *self.selected_at.lock().unwrap() = Some(Instant::now()),
            _ => {}
        }
        self.lanes.lock().unwrap()[lane].push_back(task);
        self.wake.notify_one();
    }

    pub fn get(&self, timeout: Duration) -> Option<Task> {
        let mut lanes = self.lanes.lock().unwrap();
        if lanes.iter().all(VecDeque::is_empty) {
            lanes = self.wake.wait_timeout(lanes, timeout).unwrap().0;
        }
        lanes.iter_mut().find_map(VecDeque::pop_front)
    }

    fn urgent(&self) -> Option<Task> {
        self.lanes.lock().unwrap()[0].pop_front()
    }

    fn take_configures(&self) -> Vec<Task> {
        let mut lanes = self.lanes.lock().unwrap();
        let (configs, rest): (VecDeque<Task>, VecDeque<Task>) = std::mem::take(&mut lanes[1]).into_iter().partition(|t| matches!(t, Task::Configure(_)));
        lanes[1] = rest;
        configs.into()
    }
}

#[derive(Clone, Default)]
struct ListState {
    candidates: Arc<ChoiceList>,
    catalog: Arc<HashMap<String, Arc<Inst>>>,
    total: usize,
    analyzed: usize,
    failed: usize,
    error: String,
    partial: bool,
    restored: bool,
    build_failed: bool,
}

impl ListState {
    /// Charts that cannot be analysed (converts, broken files) are done too, or the bar waits forever.
    fn progress(&self) -> Option<(usize, usize)> {
        (!self.build_failed && self.total != 0 && self.analyzed + self.failed < self.total).then_some((self.analyzed + self.failed, self.total))
    }
}

type Check = Arc<dyn Fn() -> bool + Send + Sync>;

/// What the UI gives the worker.
#[derive(Clone)]
pub struct Hooks {
    pub emit: Arc<dyn Fn(Out) + Send + Sync>,
    pub busy: Check, // a map is being played
    pub game: Check, // the game is running
}

struct BuildSlot {
    request: Mutex<Option<Request>>,
    wake: Condvar,
}

pub struct Worker {
    q: Arc<Queue>,
    h: Hooks,
    halt: Arc<AtomicBool>,
    db: Db,
    db_path: PathBuf,
    c: Arc<Consts>,
    rec: Option<Engine>,
    public: Option<Arc<Public>>,
    store: Arc<RwLock<FeatStore>>,
    slot: Arc<BuildSlot>,
    generation: u64,
    local_request: u64,
    mode: String,
    action: String,
    revision: u64,
    selection_id: u64,
    keys: std::collections::HashMap<&'static str, Vec<i64>>,
    target: Option<Box<Pick>>,
    targets: std::collections::HashMap<String, Box<Pick>>,
    playlists: std::collections::HashMap<&'static str, Vec<Pick>>,
    consumed: VecDeque<String>,
    nps_focus: f64,
    web_bias: std::collections::HashMap<&'static str, f64>,
    web_weighted: std::collections::HashMap<&'static str, bool>,
    converts: std::collections::HashMap<&'static str, BTreeMap<i64, Setting>>,
    odds: std::collections::HashMap<&'static str, [f64; 3]>,
    convert_turn: Arc<AtomicU64>, // the newest convert to deliver: a skipped one never reaches the game
    skill_weights: std::collections::HashMap<String, f64>,
    skill_tuning: skills::Tuning,
    acc_targets: std::collections::HashMap<&'static str, f64>,
    pp_mix: PpMix,
    settings: PpSettings,
    practice_skills: Vec<String>,
    states: std::collections::HashMap<&'static str, ListState>,
    credible: std::collections::HashMap<String, HashSet<String>>,
    restored: HashSet<String>,
    shown: Shown,
    session_cache: Option<((Option<i64>, i64), f64, Arc<Now>)>,
    taste_cache: (Option<i64>, HashMap<(i64, String), f64>),
    sel: (Option<PathBuf>, Option<f64>, Option<Vec<Value>>),
    loading: Option<String>,
    learning: Arc<Mutex<Option<String>>>,
    first_import: bool,
    refresh_queued: bool,
    refit_due: bool,
    refit_running: bool,
    refit_again: bool,
    last_refit: f64,
    last_idle_check: Instant,
    nps_requested: Instant,
    replay_due: Option<Instant>,
    replay_retries: u32,
    jobs: Arc<Jobs>,
    feature_errors: Arc<Mutex<std::collections::HashMap<(String, u64), (Instant, String)>>>,
    feature_pending: Arc<Mutex<(Option<(String, PathBuf, f64, bool)>, bool)>>,
    stats_visible: bool,
    stats_cache: Option<(u64, Arc<stats::Profile>)>,
    current: Option<String>,
    yielding: bool,
}

#[derive(Default)]
struct Jobs {
    importing: AtomicBool,
    archiving: AtomicBool,
    filling: AtomicBool,
}

fn key_mode(mode: &str) -> &'static str {
    MODES.into_iter().find(|m| *m == mode).unwrap_or("pp")
}

fn normalize_keys(v: &[i64]) -> Vec<i64> {
    let mut out: Vec<i64> = v.iter().copied().filter(|k| SUPPORTED_KEYS.contains(k)).collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// PP rows among a list: one in `every` offers, `since` NPS picks already made since the last PP offer.
fn place_every(mut out: Vec<Pick>, picks: Vec<Pick>, every: usize, since: usize) -> Vec<Pick> {
    let mut i = every.saturating_sub(1).saturating_sub(since);
    for c in picks {
        if i > out.len() {
            break;
        }
        out.insert(i, c);
        i += every;
    }
    out.truncate(20);
    out
}

fn verdict_words(z: f64) -> &'static str {
    if z > 1.5 { "way above expected" } else if z > 0.5 { "better than expected" } else if z < -1.5 { "way below expected" } else if z < -0.5 { "below expected" } else { "about as expected" }
}

fn reason(c: &Pick) -> String {
    let chance = 100. * c.p_up.unwrap_or(0.);
    match c.best.filter(|&b| b != 0.) {
        Some(b) => format!("refarm · best {b:.0}pp · {chance:.0}% to beat"),
        None => format!("new map · {chance:.0}% to count"),
    }
}

fn parse_mods(v: Option<&Value>) -> Option<Vec<Value>> {
    v.and_then(Value::as_array).cloned()
}

/// Spawned on its own thread: runs until `stop`. `public` is opened by the caller (the UI thread never waits on it).
pub fn spawn(public: impl FnOnce(&Hooks) -> std::result::Result<Arc<Public>, String> + Send + 'static, q: Arc<Queue>, h: Hooks, halt: Arc<AtomicBool>, db_path: PathBuf) -> std::thread::JoinHandle<()> {
    std::thread::Builder::new().name("recommend".into()).spawn(move || {
        let db = match connect(&db_path) {
            Ok(db) => db,
            Err(e) => {
                (h.emit)(Out::Status { text: format!("Could not open your data: {e}"), progress: None, blocking: true, learning: None });
                return;
            }
        };
        let mut w = Worker::new(q, h, halt, db, db_path);
        w.run(public);
    }).expect("worker thread")
}

impl Worker {
    fn new(q: Arc<Queue>, h: Hooks, halt: Arc<AtomicBool>, db: Db, db_path: PathBuf) -> Worker {
        let c = Arc::new(Consts::standard());
        let store = FeatStore::new(skills::feat_trees(), PROFILE.iter().map(|s| s.to_string()).collect(), c.clone(), STRUCTURE_VERSION).expect("skill trees");
        let client = kv_get(&db, "client").and_then(|v| v.as_str().map(str::to_string)).unwrap_or_else(|| default_client().to_string());
        let miss_ratio = kv_get(&db, "stable_miss_ratio").and_then(|v| v.as_f64()).unwrap_or(0.88);
        let first_import = kv_get(&db, "realm_import").is_none();
        let empty = || MODES.into_iter().map(|m| (m, Vec::new())).collect();
        Worker {
            q, h, halt, db, db_path, c, rec: None, public: None, store: Arc::new(RwLock::new(store)), slot: Arc::new(BuildSlot { request: Mutex::new(None), wake: Condvar::new() }), generation: 0, local_request: 0,
            mode: "pp".into(), action: "auto".into(), revision: 0, selection_id: 0,
            keys: [("pp", SUPPORTED_KEYS.collect()), ("nps", vec![7]), ("skills", vec![7])].into_iter().collect(),
            target: None, targets: Default::default(), playlists: empty(), consumed: VecDeque::new(), nps_focus: 0.5, web_bias: [("nps", 0.5), ("skills", 0.5)].into_iter().collect(), skill_weights: Default::default(), skill_tuning: Default::default(), web_weighted: [("nps", true), ("skills", true)].into_iter().collect(),
            converts: [("nps", BTreeMap::new()), ("skills", BTreeMap::new())].into_iter().collect(), odds: [("nps", [1., 1., 0.]), ("skills", [1., 1., 0.])].into_iter().collect(), convert_turn: Arc::new(AtomicU64::new(0)),
            acc_targets: [("nps", DEFAULT_TARGET), ("skills", DEFAULT_TARGET)].into_iter().collect(), pp_mix: PpMix::default(), settings: PpSettings::default(), practice_skills: Vec::new(),
            states: [("nps", ListState::default()), ("skills", ListState::default())].into_iter().collect(), credible: Default::default(), restored: HashSet::new(),
            shown: Shown { client, miss_ratio }, session_cache: None, taste_cache: (None, HashMap::default()), sel: (None, None, None), loading: None, learning: Arc::default(), first_import, refresh_queued: false,
            refit_due: false, refit_running: false, refit_again: false, last_refit: now(), last_idle_check: Instant::now(), nps_requested: Instant::now(), replay_due: Some(Instant::now() + Duration::from_secs(10)),
            replay_retries: 0, jobs: Arc::default(), feature_errors: Arc::default(), feature_pending: Arc::default(), stats_visible: false, stats_cache: None, current: None, yielding: false,
        }
    }

    fn status(&self, text: impl Into<String>) {
        (self.h.emit)(Out::Status { text: text.into(), progress: None, blocking: false, learning: None });
    }

    /// What startup is waiting on: the status line, its progress bar, and the selected map's card.
    fn stage(&mut self, text: Option<String>, blocking: bool) {
        self.loading.clone_from(&text);
        (self.h.emit)(Out::Status { text: text.unwrap_or_default(), progress: None, blocking, learning: None });
        if self.rec.is_none() && self.sel.0.is_some() {
            self.card();
        }
    }

    fn run(&mut self, public: impl FnOnce(&Hooks) -> std::result::Result<Arc<Public>, String>) {
        self.stage(Some("Opening your saved plays and recommendations…".into()), false);
        // The UI queues its real tab and filters before start: honoured before any invisible local-playlist work.
        for task in self.q.take_configures() {
            self.run_task(task);
        }
        match self.startup(public) {
            Ok(()) => self.stage(None, false),
            Err(e) => self.stage(Some(format!("Recommendations unavailable: {}", e.chars().take(240).collect::<String>())), true),
        }
        self.begin_import();
        while !self.halt.load(Ordering::Relaxed) {
            if self.refit_due && !(self.h.busy)() {
                self.t_refit();
            }
            self.catch_up_replays();
            let Some(task) = self.q.get(Duration::from_secs(5)) else {
                self.idle();
                continue;
            };
            self.run_task(task);
        }
    }

    fn idle(&mut self) {
        let busy = (self.h.busy)();
        if self.rec.is_some() && !busy {
            let last: Option<f64> = self.db.query_row("SELECT MAX(t) FROM events WHERE kind IN ('finish','abort')", [], |r| r.get(0)).unwrap_or(None);
            if last.is_some_and(|l| self.last_refit < l && now() - l > SESSION_GAP) {
                self.t_refit();
            }
            if self.nps_requested.elapsed().as_secs_f64() > NPS_CLOCK {
                self.request_nps(); // warm-up cools with the clock: re-predict NPS/Skills now and then
            }
        }
        if !busy && self.last_idle_check.elapsed().as_secs() > 3600 {
            self.last_idle_check = Instant::now();
            if now() - kv_get(&self.db, "snapshot_check").and_then(|v| v["t"].as_f64()).unwrap_or(0.) > 86400. {
                if let Ok(Some(newest)) = ms_data::public::check_index(&self.db) {
                    self.status(&format!("newer public snapshot {} — refresh --download (personal analysis; see licence)", &newest[..10]));
                }
            }
        }
    }

    fn run_task(&mut self, task: Task) {
        let started = Instant::now();
        if let Err(e) = self.dispatch(task) {
            eprintln!("recommendation task failed: {e}");
            self.status("recommendation task failed (see maniascope.log)");
        }
        let took = started.elapsed().as_secs_f64();
        if took > 1.5 {
            eprintln!("slow recommender task {took:.1} s"); // every later tab switch or Next waits behind it
        }
    }

    fn startup(&mut self, public: impl FnOnce(&Hooks) -> std::result::Result<Arc<Public>, String>) -> std::result::Result<(), String> {
        let public = public(&self.h)?;
        self.carry_calculator().map_err(|e| e.to_string())?;
        self.prune_calculators();
        self.stage(Some("Fitting your prediction model to your scores…".into()), false);
        ms_data::imports::import_public_user(&self.db, &public).map_err(|e| e.to_string())?;
        let r = refit(&self.db, &public, &self.c).map_err(|e| e.to_string())?;
        self.public = Some(public.clone());
        self.rec = Some(Engine::new(public, self.c.clone(), r, self.settings.clone()));
        self.generation += 1;
        self.spawn_builder();
        self.restore_local(&self.mode.clone()).map_err(|e| e.to_string())?;
        self.request_nps();
        self.publish(None).map_err(|e| e.to_string())?;
        self.start_fill();
        Ok(())
    }

    /// Once per calculator change that only moved its calibration: everything stored follows without re-analysing a chart.
    fn carry_calculator(&mut self) -> std::result::Result<(), String> {
        let done = format!("carried_{CALC_ID}");
        if kv_get(&self.db, &done).is_some() {
            return Ok(());
        }
        self.stage(Some("Updating stored ratings (once)…".into()), false);
        crate::engine::carry_rows(&self.db)?;
        // A crawl from before the update can still hold it; a failure here waits for the next launch, not for the player.
        if ms_data::webmaps::running() {
            return Ok(());
        }
        if let Err(e) = crate::engine::carry_web(&self.db) {
            eprintln!("website features not carried: {e}");
            return Ok(());
        }
        kv_set(&self.db, &done, &json!(true)).map_err(|e| e.to_string())
    }

    /// Once, after the carry: a player never needs the older calculators' rows again (user 2026-11-01).
    fn prune_calculators(&mut self) {
        let (carried, done) = (format!("carried_{CALC_ID}"), format!("pruned_{CALC_ID}"));
        if kv_get(&self.db, &carried).is_none() || kv_get(&self.db, &done).is_some() {
            return;
        }
        self.stage(Some("Compacting stored ratings (once)…".into()), false);
        match ms_data::store::prune_feats(&self.db, CALC_ID) {
            Ok(None) => {} // an older database re-analyses instead of carrying: its old rows go next launch
            Ok(Some(_)) => { let _ = kv_set(&self.db, &done, &json!(true)); }
            Err(e) => eprintln!("old ratings kept for now: {e}"),
        }
    }

    fn spawn_builder(&self) {
        let (slot, q, h, store, path, halt) = (self.slot.clone(), self.q.clone(), self.h.clone(), self.store.clone(), self.db_path.clone(), self.halt.clone());
        let prm = self.rec.as_ref().unwrap().prm.clone();
        std::thread::Builder::new().name("nps-build".into()).spawn(move || {
            let mut builder = Builder::new(store, prm);
            let (s, stop) = (slot.clone(), halt.clone());
            builder.wanted = Arc::new(move || stop.load(Ordering::Relaxed) || s.request.lock().unwrap().is_some());
            builder.busy = h.busy.clone();
            builder.game = h.game.clone();
            let out = q.clone();
            builder.emit = Arc::new(move |u| out.put(Task::NpsUpdate(u)));
            let Ok(db) = connect(&path) else { return };
            while !halt.load(Ordering::Relaxed) {
                let request = {
                    let mut pending = slot.request.lock().unwrap();
                    while pending.is_none() && !halt.load(Ordering::Relaxed) {
                        pending = slot.wake.wait_timeout(pending, Duration::from_secs(2)).unwrap().0;
                    }
                    pending.take()
                };
                let Some(r) = request else { continue };
                if let Err(e) = builder.build(&db, &r) {
                    builder.reset();
                    (builder.emit)(Update::Failed { error: e.to_string(), mode: r.mode.clone(), revision: r.revision, generation: r.generation });
                }
                r.public.release();
                ms_data::store::release_memory();
            }
        }).expect("builder thread");
    }

    fn dispatch(&mut self, task: Task) -> Result<()> {
        match task {
            Task::Configure(c) => self.t_configure(*c)?,
            Task::ConvertDone { pick, mode, error } => self.t_convert_done(pick, mode, error)?,
            Task::Selected { path, rate, mods } => self.t_selected(path, rate, mods)?,
            Task::FeatureReady { sha, features, refit } => {
                if let Some(rec) = self.rec.as_mut() {
                    let have = features.is_some();
                    if let Some(fs) = &features {
                        rec.store(&self.db, &sha, fs)?;
                    }
                    self.card();
                    if have && refit {
                        self.refresh_later();
                    }
                }
            }
            Task::Start(play) => self.t_start(&play)?,
            Task::Abort(play) => self.t_abort(&play)?,
            Task::Void => {
                log_event(&self.db, "void", self.current.as_deref(), None, &json!({}))?;
            }
            Task::Score { score, path, fresh, play } => self.t_score(&score, &path, fresh, play.as_ref())?,
            Task::Next { skip, mode, revision, candidate, result, selected_at, selection_id } => self.t_next(skip, mode, revision, candidate, result, selected_at, selection_id)?,
            Task::Open { pick, mode, revision, result, selected_at, selection_id } => self.t_open(pick, mode, revision, result, selected_at, selection_id)?,
            Task::Publish => self.publish(None)?,
            Task::Refresh => {
                self.refresh_queued = false;
                self.request_nps();
                if self.mode == "pp" {
                    self.publish(None)?; // NPS/Skills: the Builder's first publication of this request follows
                }
            }
            Task::WebUpdated => self.request_nps(),
            Task::ConvertSelected(target) => self.t_convert_selected(target),
            Task::Stats(v) => self.t_stats(v)?,
            Task::NpsUpdate(u) => self.t_nps_update(u)?,
            Task::SkipWarmup => {
                log_event(&self.db, "skip_warmup", None, None, &json!({}))?;
                self.playlists.values_mut().for_each(|l| l.retain(|c| c.purpose != "warmup"));
                self.request_nps();
                self.publish(None)?;
            }
            Task::Lock(keys) => {
                let m = key_mode(&self.mode);
                self.keys.insert(m, keys.map_or_else(|| SUPPORTED_KEYS.collect(), |k| normalize_keys(&[k])));
                self.request_nps();
                self.publish(None)?;
            }
            Task::Import => self.begin_import(),
            Task::Client(name) => self.t_client(&name)?,
            Task::WebsiteUser(name) => {
                let name = name.trim();
                kv_set(&self.db, "website_username", &if name.is_empty() { Value::Null } else { json!(name) })?;
                kv_set(&self.db, "website_user", &json!({}))?;
                self.begin_import();
            }
            Task::Imported { result, recovered, changed, error } => self.t_imported(result, recovered, changed, error)?,
            Task::Tracking | Task::Refit => self.t_refit(),
            Task::History(outliers) => (self.h.emit)(Out::History(history_text(&play_history(&self.db, 100, outliers, None)?, &self.shown))),
            Task::RefitDone(r) => self.t_refit_done(r)?,
            Task::ReplayDone { pending, error, fresh } => self.t_replay_done(pending, error, fresh)?,
        }
        Ok(())
    }

    fn target_of(&self, mode: &str) -> f64 {
        self.shown.model(self.acc_targets.get(mode).copied().unwrap_or(DEFAULT_TARGET))
    }

    fn session(&mut self) -> Result<Arc<Now>> {
        let id: Option<i64> = self.db.query_row("SELECT MAX(id) FROM events", [], |r| r.get(0))?;
        let stamp = (id, (now() / 2.) as i64);
        let target = self.target_of(&self.mode.clone());
        if let Some((s, t, n)) = &self.session_cache {
            if *s == stamp && *t == target {
                return Ok(n.clone());
            }
        }
        let n = Arc::new(self.rec.as_ref().expect("a fitted model").now_state(&self.db, target)?);
        self.session_cache = Some((stamp, target, n.clone()));
        Ok(n)
    }

    fn publish(&mut self, mode: Option<String>) -> Result<()> {
        let Some(rec) = self.rec.as_ref() else { return Ok(()) };
        if self.stats_visible {
            return Ok(());
        }
        // The player's level is still unknown: an empty list says why, not beginner maps.
        if rec.r.fitted.rows.rows.len() < 30 && (self.learning.lock().unwrap().is_some() || self.first_import) {
            return Ok(());
        }
        let mode = key_mode(mode.as_deref().unwrap_or(&self.mode));
        let st = self.states.get(mode).cloned();
        let kept = st.as_ref().filter(|s| s.partial).map(|_| (self.playlists[mode].clone(), self.credible.get(mode).cloned()));
        let (phase, shown, note) = self.pool(mode)?;
        let mut shown = self.prepared_playlist(mode, shown)?;
        if mode == "nps" && self.pp_mix.on {
            shown = self.mix_pp(shown)?;
            self.playlists.insert(mode, shown.clone());
        }
        if let Some((list, credible)) = kept {
            // a first look from the previous leaders: the full list still decides the queue
            self.playlists.insert(mode, list);
            match credible {
                Some(c) => self.credible.insert(mode.to_string(), c),
                None => self.credible.remove(mode),
            };
        }
        let progress = st.as_ref().and_then(ListState::progress);
        (self.h.emit)(Out::Pool { mode: mode.to_string(), revision: self.revision, selection_id: self.selection_id, phase, shown, note, summary: self.summary(), progress });
        Ok(())
    }

    fn summary(&self) -> String {
        let Some(rec) = self.rec.as_ref() else { return String::new() };
        let levels: Vec<String> = SUPPORTED_KEYS.filter_map(|k| rec.model().level.get(&k).map(|l| format!("{k}K {:+.2}", l.0))).collect();
        format!("ledger {:.1}pp over {} maps · {} lazer scores without a local chart · {} evidence rows · levels {} · retry {:.3}", rec.r.ledger.total(), rec.r.ledger.best.len(), rec.r.missing,
                rec.r.fitted.rows.rows.len(), levels.join(", "), rec.model().retry)
    }

    /// Which how-often slider each of this tab's last offers answered to, oldest first.
    fn recent(n: &Now, mode: &str) -> Vec<u8> {
        let of = |e: &Arc<ms_data::store::Event>| if e.info["convert"].as_bool() == Some(true) { CONVERT } else if e.info["web"].as_bool() == Some(true) { WEB } else { REAL };
        let all: Vec<u8> = n.session.all.iter().filter(|e| e.kind == "offer" && e.info["mode"].as_str() == Some(mode)).map(of).collect();
        all[all.len().saturating_sub(40)..].to_vec()
    }

    /// The sets of what was just played or offered, and of the converts among them: one song, whichever side came first.
    fn recent_sets(&self, n: &Now) -> Result<(HashSet<i64>, HashSet<i64>)> {
        let played = n.session.attempts.iter().rev().take(20).filter_map(|a| a.beatmap.clone());
        let (mut sets, mut convert_sets) = (HashSet::new(), HashSet::new());
        let mut library = self.db.prepare_cached("SELECT set_id FROM installed WHERE md5=? OR sha256=?")?;
        let mut converts = self.db.prepare_cached("SELECT set_id FROM converts WHERE md5=? OR sha=?")?;
        for id in self.consumed.iter().cloned().chain(played) {
            let hash = id.split_once(':').map_or(id.as_str(), |x| x.1);
            let rows: Vec<Option<i64>> = library.query_map([hash, hash], |r| r.get(0))?.collect::<Result<_>>()?;
            sets.extend(rows.into_iter().flatten().filter(|&s| s > 0));
            let rows: Vec<Option<i64>> = converts.query_map([hash, hash], |r| r.get(0))?.collect::<Result<_>>()?;
            convert_sets.extend(rows.into_iter().flatten().filter(|&s| s > 0));
        }
        for t in self.targets.values() {
            if let Some(set) = t.set_id.filter(|&s| s > 0) {
                sets.insert(set);
                if t.convert_of.is_some() {
                    convert_sets.insert(set);
                }
            }
        }
        sets.extend(convert_sets.iter().copied());
        Ok((sets, convert_sets))
    }

    fn local_shas(&self) -> Arc<HashSet<String>> {
        self.rec.as_ref().expect("a fitted model").r.local_shas.clone()
    }

    fn prepared(&self, mode: &'static str) -> (ListState, Prepared) {
        let st = self.states[mode].clone();
        let shas = self.local_shas();
        let p = st.candidates.prepare(&self.keys[mode], &|s| shas.contains(s));
        (st, p)
    }

    fn pool(&mut self, mode: &'static str) -> Result<(String, Vec<Pick>, String)> {
        let n = self.session()?;
        if self.keys[mode].is_empty() {
            return Ok((n.session.phase().to_string(), Vec::new(), "Select at least one keymode".to_string()));
        }
        if mode == "pp" {
            let keys = self.keys["pp"].clone();
            let (phase, shown, note) = self.rec.as_mut().unwrap().pool(&self.db, &keys, &n, 20)?;
            return Ok((phase.to_string(), shown, note));
        }
        let (st, prepared) = self.prepared(mode);
        // Explicit skill preferences must not be erased by unrelated NPS taste.
        let none = HashMap::default();
        if mode == "nps" {
            let id: Option<i64> = self.db.query_row("SELECT MAX(id) FROM events", [], |r| r.get(0))?;
            if id != self.taste_cache.0 {
                self.taste_cache = (id, taste(&self.db, now())?);
            }
        }
        self.yield_urgent(); // a selected map's card goes before the library-wide weighting
        let shares = (mode == "skills").then(|| skills::shares(&self.practice_skills, &self.skill_weights, self.skill_tuning.other)).flatten();
        let seed = n.session.all.iter().filter(|e| matches!(e.kind.as_str(), "offer" | "finish" | "abort" | "reset")).map(|e| e.id).max().unwrap_or(0);
        let seed = (mode == "nps" || shares.is_some()).then(|| seed.to_string());
        let consumed: HashSet<String> = self.consumed.iter().cloned().collect();
        let active: Vec<&Setting> = self.keys[mode].iter().filter_map(|k| self.converts[mode].get(k)).filter(|s| s.on && !s.sources.is_empty()).collect();
        let convert = if active.is_empty() { 0. } else { web_factor(active.iter().map(|s| s.often).sum::<f64>() / active.len() as f64) }; // several keymodes share one list
        let (recent_sets, recent_convert_sets) = if active.is_empty() { Default::default() } else { self.recent_sets(&n)? };
        let store = self.store.read().unwrap();
        let (ranked, odds) = rank(&prepared, &st.catalog, &n.session, &store, RankInputs {
            keys: self.keys[mode].clone(), mode, web: web_factor(self.web_bias[mode]), convert, weighted: [self.web_weighted[mode], active.iter().any(|s| s.weighted)], recent_sets: &recent_sets, recent_convert_sets: &recent_convert_sets, taste: if mode == "nps" { &self.taste_cache.1 } else { &none }, consumed: &consumed, seed, focus: self.nps_focus, count: 60,
            recent: Self::recent(&n, mode), skill_weights: shares, selected_skills: &self.practice_skills, miss_ratio: (self.shown.client == "stable").then_some(self.shown.miss_ratio),
        });
        drop(store);
        self.odds.insert(mode, odds);
        let shown: Vec<Pick> = ranked.iter().take(20).cloned().collect();
        // A queued map outside the current top 60 was queued under another state (cold start, restart list): G835LX
        // 2026-10-22 offered dEKA, 6 NPS, ranked 2242 of 2245, after seven such maps.
        self.credible.insert(mode.to_string(), ranked.iter().filter_map(Pick::identity).collect());
        let target = self.acc_targets.get(mode).copied().unwrap_or(DEFAULT_TARGET);
        let mut note = format!("target ~{:.0}%", 100. * target);
        let incomplete = st.analyzed + st.failed < st.total;
        if st.build_failed {
            note += " · local map analysis stopped";
        } else if st.total == 0 {
            note += " · no maps of these keymodes found yet";
        } else if !incomplete {
            note += &format!(" · {} maps", crate::engine::commas(st.total)); // else the window's bottom bar counts them
        }
        let missing = shown.iter().filter(|c| c.stream() == WEB).count();
        if missing != 0 {
            note += &format!(" · {missing} to download");
        }
        if mode == "skills" {
            note = format!("{} · {note}", skills::label(&self.practice_skills));
        }
        if shown.is_empty() && !st.build_failed {
            note += if incomplete { " · looking for maps that fit your target…" } else { " · no map fits your accuracy target at 0.70–1.50× (try another target or keymode)" };
        }
        if st.failed != 0 {
            note += &format!(" · {} unreadable", st.failed);
        }
        if !st.error.is_empty() {
            note += &format!(" · {}", st.error);
        }
        if st.restored && !st.build_failed {
            note += " · showing the last list while it updates";
        }
        Ok((n.session.phase().to_string(), shown, note))
    }

    /// PP maps past the mix bars at every N-th place of the NPS list (user 2026-10-30).
    fn mix_pp(&mut self, shown: Vec<Pick>) -> Result<Vec<Pick>> {
        let n = self.session()?;
        let keys = self.keys["nps"].clone();
        let cands = self.rec.as_mut().unwrap().scored(&self.db, &keys, &n)?;
        let m = self.pp_mix.clone();
        let listed: HashSet<String> = shown.iter().filter_map(Pick::identity).collect();
        // Time-based freshness (no pool): an offered PP map stays back ~half an hour, a played one ~6 plays.
        let mut picks: Vec<Pick> = cands.into_iter().filter(|c| {
            let id = c.identity().unwrap_or_default();
            c.ranked && c.gain.unwrap_or(0.) >= m.min_gain && c.p_up.unwrap_or(0.) >= (if c.rate < 1. { m.ht_chance } else { m.chance }) && !listed.contains(&id)
                && n.session.freshness(&id, c.length, Some("pp"), None, ms_model::session::EXPOSURE_S) >= 0.5
        }).map(|mut c| {
            c.why = format!("{} · PP map in the NPS list", reason(&c));
            c.purpose = "nps_mix".into();
            c
        }).collect();
        let quality = |c: &Pick| c.p_up.unwrap_or(0.) * (0.3 + (c.gain.unwrap_or(0.) / 0.05).ln_1p().powf(0.65));
        picks.sort_by(|a, b| quality(b).total_cmp(&quality(a)));
        // NPS picks since the last PP offer place the first one: one in `every` offers is PP.
        let mut since = 0;
        for e in n.session.all.iter().rev().filter(|e| e.kind == "offer") {
            if e.info["mode"].as_str() == Some("pp") {
                break;
            }
            since += 1;
        }
        Ok(place_every(shown, picks, m.every, since))
    }

    /// Keep the displayed order; update predictions, then append new maps.
    fn prepared_playlist(&mut self, mode: &'static str, fresh: Vec<Pick>) -> Result<Vec<Pick>> {
        let n = self.session()?;
        let local = mode != "pp";
        let fresh_by: std::collections::HashMap<String, Pick> = fresh.iter().filter_map(|c| Some((c.identity()?, c.clone()))).collect();
        let prepared = local.then(|| self.prepared(mode));
        let credible = if local && !fresh.is_empty() { self.credible.get(mode).cloned() } else { None };
        let shas = self.local_shas();
        let selected = self.practice_skills.clone();
        let (warm, target) = (n.view.warm_flag, n.target);
        let mut shown = Vec::new();
        let queue = self.playlists[mode].clone();
        for mut c in queue {
            if c.purpose == "nps_mix" {
                continue; // _mix_pp re-adds it; refreshed here it turned into a plain NPS row
            }
            let key = c.identity().unwrap_or_default();
            if credible.as_ref().is_some_and(|set| !set.contains(&key)) || !self.keys[mode].contains(&c.keys) || (mode == "skills" && c.practice_skills != selected) {
                continue;
            }
            let sha = c.sha.clone().unwrap_or_default();
            if c.installed && !shas.contains(&sha) {
                continue;
            }
            let alternative = match &prepared {
                Some((st, p)) => p.find(&key).and_then(|i| builder::prepared_pick(p, i, &st.catalog, mode, &selected, &self.store.read().unwrap())),
                None => fresh_by.get(&key).cloned(),
            };
            if let Some(a) = alternative.as_ref().filter(|a| a.rate == c.rate && (c.md5.is_some() || !sha.is_empty())) {
                let (why, weight, value) = (c.why.clone(), c.weight, c.value);
                c = a.clone(); // refresh exact features without changing queue order
                if local {
                    (c.why, c.weight, c.value) = (why, weight, value);
                }
            }
            let rec = self.rec.as_ref().unwrap();
            let Some(f) = c.f.clone() else { continue };
            let chart = c.md5.clone().unwrap_or_else(|| sha.clone());
            let Some(mut expected) = rec.predict(&f, &chart, c.bid, c.rate, &n) else { continue };
            if let Some(a) = alternative.as_ref().filter(|a| a.rate != c.rate) {
                let central = rec.model().central_target(&f, target, warm);
                if (a.pred.acc_mid - central).abs() < (expected.shown.acc_mid - central).abs() {
                    if let (Some(af), Some(e)) = (a.f.clone(), a.f.as_ref().and_then(|af| rec.predict(af, &chart, c.bid, a.rate, &n))) {
                        let _ = af;
                        c = a.clone();
                        expected = e;
                    }
                }
            }
            let Some(f) = c.f.clone() else { continue };
            if mode == "skills" && alternative.is_none() {
                continue; // no longer matches the selection: the kernel left it out
            }
            if local && !rec.model().eligible(&f, &expected.shown, target, warm) {
                continue;
            }
            c.pred = expected.shown.clone();
            c.pp_pred = Some(expected.pp.clone());
            if mode == "pp" && c.purpose != "warmup" {
                if let Some(scale) = c.scale {
                    let (lo, hi, best, gain, up) = pp::estimates_at(&expected.pp, scale, c.bid, &rec.r.ledger);
                    (c.pp_lo, c.pp_hi, c.best, c.gain, c.p_up) = (Some(lo), Some(hi), best, Some(gain), Some(up));
                }
            }
            shown.push(c);
        }
        let mut seen: HashSet<String> = shown.iter().filter_map(Pick::identity).chain(self.consumed.iter().cloned()).collect();
        for c in &fresh {
            if let Some(key) = c.identity() {
                if seen.insert(key) {
                    shown.push(c.clone());
                }
            }
        }
        if local {
            // Kept maps you don't have outlive the played ones and piled up at the top (user 2026-10-26).
            let streams: Vec<u8> = shown.iter().map(Pick::stream).collect();
            let order = mix_order(&streams, self.odds[mode], &Self::recent(&n, mode));
            shown = order.into_iter().map(|i| shown[i].clone()).collect();
        }
        shown.truncate(20);
        if shown.is_empty() && !fresh.is_empty() {
            // A small credible library can cycle after its prepared queue ends.
            let last = self.consumed.back().cloned();
            shown = fresh.iter().filter(|c| c.identity() != last).take(20).cloned().collect();
            if shown.is_empty() {
                shown = fresh[..1].to_vec();
            }
        }
        self.playlists.insert(mode, shown.clone());
        Ok(shown)
    }

    fn refresh_later(&mut self) {
        if !self.refresh_queued {
            self.refresh_queued = true;
            self.q.put(Task::Refresh);
        }
    }

    /// Run a waiting map/rate selection inside a long rebuild, so its card never waits for it.
    fn yield_urgent(&mut self) {
        if self.yielding {
            return;
        }
        self.yielding = true;
        while let Some(task) = self.q.urgent() {
            self.run_task(task);
        }
        self.yielding = false;
    }

    fn request_of(&mut self, mode: &str) -> Result<Request> {
        let n = self.session()?;
        let rec = self.rec.as_ref().unwrap();
        let view = n.view.clone();
        Ok(Request { model: rec.r.fitted.display.as_ref().map_or_else(|| rec.r.fitted.pp.model.clone(), |d| d.model.clone()), tables_for: Arc::new(move |k| view.tables(k)), session: n.session.clone(),
                     public: rec.public.clone(), calc: CALC_ID.to_string(), predictor: PREDICTOR_ID.to_string(), client: self.shown.client.clone(), target: n.target, keys: self.keys["nps"].clone(),
                     skills_keys: self.keys["skills"].clone(), selected_skills: self.practice_skills.clone(), tuning: self.skill_tuning.clone(), converts: self.converts.get(key_mode(mode)).map_or_else(Vec::new, |s| converts::pairs(s, &self.keys[key_mode(mode)])), mode: mode.to_string(), revision: self.local_request, generation: self.generation })
    }

    fn restore_local(&mut self, mode: &str) -> Result<()> {
        if self.rec.is_none() || mode == "pp" || !self.restored.insert(mode.to_string()) {
            return Ok(());
        }
        let m = key_mode(mode);
        let Some(saved) = kv_get(&self.db, &format!("playlist_cache:{mode}")).filter(|s| s["version"] == 1) else { return Ok(()) };
        let refs: Vec<(String, f64)> = saved["choices"].as_array().map_or_else(Vec::new, |a| a.iter().filter_map(|x| Some((x[0].as_str()?.to_string(), x[1].as_f64()?))).collect());
        let installed: Vec<Arc<Inst>> = self.rec.as_ref().unwrap().r.installed.values().filter(|i| self.keys[m].contains(&i.keys)).cloned().collect();
        let request = self.request_of(mode)?;
        let mut builder = Builder::new(self.store.clone(), self.rec.as_ref().unwrap().prm.clone());
        let (candidates, catalog) = builder.restore(&self.db, &request, &refs, &installed)?;
        let total = installed.len();
        self.states.insert(m, ListState { candidates, catalog, total, analyzed: total.min(saved["analyzed"].as_u64().unwrap_or(0) as usize), restored: true, ..ListState::default() });
        Ok(())
    }

    fn request_nps(&mut self) {
        if self.rec.is_none() {
            return;
        }
        self.local_request += 1;
        self.nps_requested = Instant::now();
        let mode = if self.stats_visible { "stats".to_string() } else { self.mode.clone() };
        if let Ok(r) = self.request_of(&mode) {
            *self.slot.request.lock().unwrap() = Some(r);
            self.slot.wake.notify_all();
        }
    }

    fn t_stats(&mut self, visible: bool) -> Result<()> {
        self.stats_visible = visible;
        self.request_nps();
        if !visible {
            return self.publish(None);
        }
        if self.rec.is_none() {
            return Ok(());
        }
        if self.stats_cache.as_ref().is_none_or(|c| c.0 != self.generation) {
            let profile = stats::build(self.rec.as_ref().unwrap(), &self.db)?;
            self.stats_cache = Some((self.generation, Arc::new(profile)));
        }
        let n = self.session()?;
        let profile = self.stats_cache.as_ref().unwrap().1.clone();
        (self.h.emit)(Out::Stats(Box::new(stats::current(&profile, &n, self.rec.as_ref().unwrap()))));
        Ok(())
    }

    fn t_nps_update(&mut self, update: Update) -> Result<()> {
        match update {
            Update::Failed { error, mode, revision, generation } => {
                if generation != self.generation || revision != self.local_request {
                    return Ok(());
                }
                // Keep the usable shortlist; a later ready publication replaces this terminal status.
                let m = key_mode(&mode);
                if let Some(st) = self.states.get_mut(m) {
                    st.error = error;
                    st.build_failed = true;
                }
                if mode == self.mode {
                    self.publish(Some(mode))?;
                }
            }
            Update::Ready(p) => {
                if p.generation != self.generation || p.revision != self.local_request || p.keys_by_mode.0 != self.keys["nps"] || p.keys_by_mode.1 != self.keys["skills"] || p.selected_skills != self.practice_skills {
                    return Ok(());
                }
                self.accept(p)?;
            }
        }
        Ok(())
    }

    fn accept(&mut self, p: Published) -> Result<()> {
        let mode = key_mode(&p.mode);
        // The restart cache holds no website charts; the first real build replaces that placeholder list instead
        // of keeping it, or website charts never show up.
        if self.states.get(mode).is_some_and(|s| s.restored) {
            self.playlists.insert(mode, Vec::new());
        }
        // Exact new features of a chart that left the candidates: the retained queue predicts and reads from them at once.
        let mut refreshed = HashMap::default();
        {
            let store = self.store.read().unwrap();
            let practice = mode == "skills";
            for c in self.playlists.get_mut(mode).into_iter().flatten() {
                let Some(sha) = c.sha.clone() else { continue };
                if let Some(f) = store.feats(&sha, c.rate) {
                    c.f = Some(Arc::new(f.clone()));
                }
                let Some(rec) = self.rec.as_ref() else { continue };
                let Some(mut m) = store.metadata(&sha, c.rate, &self.practice_skills, &self.skill_tuning, practice, rec.model()) else { continue };
                m.family = Some(m.family).filter(|f| !f.is_empty()).or(c.family.clone()).unwrap_or_default();
                m.nps = Some(m.nps).filter(|n| *n != 0.).or(c.nps).unwrap_or(0.);
                if c.refresh(&m, practice) {
                    if let Some(key) = c.identity() {
                        refreshed.insert((key, c.rate.to_bits()), m);
                    }
                }
            }
        }
        // The features may have moved on after the Builder queued this publication: its same-rate choice must not undo that.
        let mut p = p;
        if !refreshed.is_empty() {
            p.candidates = Arc::new(p.candidates.patched(&refreshed));
        }
        // A quick partial list (the previous leaders only) would shrink the restart references.
        if !p.partial {
            let refs: Vec<Value> = p.candidates.restart_refs().into_iter().map(|(s, r)| json!([s, r])).collect();
            kv_set(&self.db, &format!("playlist_cache:{mode}"), &json!({"version": 1, "calc": CALC_ID, "choices": refs, "analyzed": p.analyzed}))?;
        }
        self.states.insert(mode, ListState { candidates: p.candidates, catalog: p.catalog, total: p.total, analyzed: p.analyzed, failed: p.failed, error: p.error, partial: p.partial, restored: false, build_failed: false });
        if mode == self.mode {
            self.publish(Some(mode.to_string()))?;
        }
        Ok(())
    }

    fn t_configure(&mut self, c: Config) -> Result<()> {
        // Fast clicks need one search for the final selection, not a full catalogue pass for every intermediate state.
        if !MODES.contains(&c.mode.as_str()) || c.revision < self.revision.max(*self.q.config_revision.lock().unwrap()) {
            return Ok(());
        }
        let old = (self.keys["nps"].clone(), self.keys["skills"].clone(), self.practice_skills.clone(), self.skill_tuning.clone());
        let old_mode = self.mode.clone();
        self.mode.clone_from(&c.mode);
        self.action = if ACTIONS.contains(&c.action.as_str()) { c.action.clone() } else { "auto".to_string() };
        self.revision = c.revision;
        for m in MODES {
            self.keys.insert(m, c.keys.get(m).map_or_else(|| if m == "pp" { SUPPORTED_KEYS.collect() } else { vec![7] }, |k| normalize_keys(k)));
        }
        self.practice_skills = skills::normalize(c.practice_skills.as_deref().unwrap_or(&[]));
        if let Some(focus) = c.nps_focus.map(builder::normalize_focus).filter(|f| *f != self.nps_focus) {
            self.nps_focus = focus;
            self.playlists.insert("nps", Vec::new());
        }
        for (m, value) in &c.web_bias {
            let m = key_mode(m);
            if self.web_bias.get(m).is_some_and(|v| v != value) {
                self.web_bias.insert(m, *value);
                self.playlists.insert(m, Vec::new());
            }
        }
        let mut rebuild = false;
        for (m, on) in &c.web_weighted {
            let m = key_mode(m);
            if self.web_weighted.get(m).is_some_and(|v| v != on) {
                self.web_weighted.insert(m, *on);
                self.playlists.insert(m, Vec::new());
            }
        }
        for (m, value) in c.converts {
            let m = key_mode(&m);
            if self.converts.get(m).is_some_and(|v| *v != value) {
                rebuild = true;
                self.converts.insert(m, value);
                self.playlists.insert(m, Vec::new());
            }
        }
        if let Some(w) = c.skill_weights.filter(|w| *w != self.skill_weights) {
            self.skill_weights = w;
            self.playlists.insert("skills", Vec::new());
        }
        if let Some(t) = c.skill_tuning.filter(|t| *t != self.skill_tuning) {
            self.skill_tuning = t;
            self.playlists.insert("skills", Vec::new());
        }
        let mut retarget = false;
        for (m, value) in &c.acc_targets {
            let (m, value) = (key_mode(m), builder::normalize_target(*value));
            if self.acc_targets.get(m).is_some_and(|v| *v != value) {
                self.acc_targets.insert(m, value);
                retarget = true;
                self.playlists.insert(m, Vec::new());
            }
        }
        if let Some(s) = c.pp_settings.filter(|s| *s != self.settings) {
            self.settings = s.clone();
            self.playlists.insert("pp", Vec::new());
            if let Some(rec) = self.rec.as_mut() {
                rec.settings = s;
            }
        }
        if let Some(m) = c.pp_mix.filter(|m| *m != self.pp_mix) {
            self.pp_mix = m;
            self.playlists.insert("nps", Vec::new());
        }
        self.restore_local(&c.mode)?;
        if rebuild || retarget || old != (self.keys["nps"].clone(), self.keys["skills"].clone(), self.practice_skills.clone(), self.skill_tuning.clone()) || old_mode != self.mode {
            self.request_nps();
        }
        self.publish(None)
    }

    fn t_start(&mut self, play: &Value) -> Result<()> {
        let t = play["t"].as_f64().unwrap_or_else(now);
        if !tracking_allowed(&self.db, Some(t), None)? {
            return Ok(());
        }
        let sha = play["sha"].as_str().unwrap_or("").to_string();
        let path = play["path"].as_str().filter(|p| !p.is_empty()).map(PathBuf::from);
        let rate = play["rate"].as_f64();
        let mods = parse_mods(play.get("mods"));
        let data = path.as_ref().and_then(|p| std::fs::read(p).ok());
        let md5 = data.as_ref().map(|d| hashes(d).1);
        let predictable = self.rec.is_some() && mods.as_ref().is_some_and(|m| rate.is_some_and(|r| predictable_context(m, r)));
        let mut exp = None;
        if let (true, Some(md5), Some(rate)) = (predictable, md5.as_deref(), rate) {
            let n = self.session()?;
            exp = self.rec.as_mut().unwrap().expectation(&self.db, &sha, md5, rate, &n)?;
        }
        let inst = one_by_sha(&self.db, &sha)?;
        let meta = if inst.is_none() { path.as_deref().and_then(ms_data::lazer_index::metadata) } else { None };
        let title = inst.as_ref().map(Inst::name).or_else(|| meta.as_ref().filter(|m| m.contains_key("Title")).map(|m| {
            let g = |k: &str| m.get(k).map_or("", String::as_str);
            format!("{} - {} [{}]", g("Artist"), g("Title"), g("Version"))
        })).unwrap_or_else(|| sha.clone());
        let bid = inst.as_ref().and_then(|i| i.beatmap_id).or_else(|| meta.as_ref().and_then(|m| m.get("BeatmapID")?.parse().ok()));
        let identity_md5 = md5.clone().or_else(|| inst.as_ref().and_then(|i| i.md5.clone()));
        let mut c = self.target.clone().filter(|c| same_map(c, Some(&sha), md5.as_deref(), bid));
        if c.is_none() {
            // Restore the offer linkage after an application restart. Only the latest preceding offer can match; an
            // unrelated manual pick cannot.
            let offer: Option<(i64, String)> = self.db.query_row("SELECT id,info FROM events WHERE kind='offer' AND t<=? AND t>=? ORDER BY t DESC,id DESC LIMIT 1", [t, t - 1800.], |r| Ok((r.get(0)?, r.get(1)?))).optional()?;
            if let Some((id, info)) = offer {
                let o: Value = serde_json::from_str(&info).unwrap_or(Value::Null);
                let p = Pick { sha: o["sha"].as_str().map(str::to_string), md5: o["md5"].as_str().map(str::to_string), bid: o["bid"].as_i64(), ..Pick::default() };
                if same_map(&p, Some(&sha), identity_md5.as_deref(), bid) {
                    let mut p = p;
                    p.offer_id = Some(id);
                    p.purpose = o["purpose"].as_str().unwrap_or("").to_string();
                    p.mode = o["mode"].as_str().unwrap_or("").to_string();
                    p.rate = o["rate"].as_f64().unwrap_or(0.);
                    p.pred.acc_mid = o["accuracy"].as_f64().unwrap_or(f64::NAN);
                    p.pred.opening_acc = o["opening_accuracy"].as_f64().unwrap_or(f64::NAN);
                    p.selection = o["selection"].as_str().unwrap_or("").to_string();
                    p.practice_skills = o["practice_skills"].as_array().map_or_else(Vec::new, |a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect());
                    c = Some(Box::new(p));
                }
            }
        }
        let matched = c.is_some();
        let keys = exp.as_ref().map(|e| e.2.keys).or(inst.as_ref().map(|i| i.keys));
        let length = exp.as_ref().map(|e| e.1.f("length") / 1000. / rate.unwrap_or(1.)).unwrap_or_else(|| inst.as_ref().and_then(|i| i.length).zip(rate).map_or(0., |(l, r)| l / 1000. / r));
        let key = if let Some(m) = identity_md5.as_ref().filter(|m| !m.is_empty()) { format!("md5:{m}") } else { format!("sha:{sha}") };
        for list in self.playlists.values_mut() {
            list.retain(|p| p.identity().as_deref() != Some(&key));
        }
        let (mut pattern, mut family) = (None, None);
        if let Some(f) = exp.as_ref().map(|e| &e.1) {
            if f.dict("nps").is_some_and(|n| n.get("version").is_some_and(|v| v.truthy())) {
                pattern = Some(ms_model::predict::profile_of(f, &PROFILE.map(str::to_string)));
                family = f.dict("nps").and_then(|n| n.get("family")).map(ms_data::codec::to_json);
            }
        }
        let rec = self.rec.as_ref();
        let chart = identity_md5.clone().unwrap_or_else(|| sha.clone());
        let forecast = exp.as_ref().and_then(|e| forecast_chart(e.0.forecast(&chart, rate.unwrap_or(1.)), Some(&sha), identity_md5.as_deref()));
        let mut info = json!({
            "sha": sha, "rate": rate, "purpose": c.as_ref().map_or("manual", |c| c.purpose.as_str()), "mode": c.as_ref().map_or("manual", |c| c.mode.as_str()), "offer_id": c.as_ref().and_then(|c| c.offer_id),
            "offered_rate": c.as_ref().map(|c| c.rate), "offered_accuracy": c.as_ref().map(|c| c.pred.acc_mid).filter(|a| !a.is_nan()), "offered_opening_accuracy": c.as_ref().map(|c| c.pred.opening_acc).filter(|a| !a.is_nan()),
            "practice_skills": c.as_ref().map(|c| c.practice_skills.clone()).filter(|_| matched), "choice": c.as_ref().map(|c| c.selection.clone()),
            "profile": pattern.or_else(|| c.as_ref().map(|c| c.profile.clone()).filter(|p| !p.is_empty())), "family": family.or_else(|| c.as_ref().and_then(|c| c.family.clone()).map(Value::from)),
            "mods": mods.as_ref().map(|m| m.iter().map(|x| x["acronym"].clone()).collect::<Vec<_>>()).unwrap_or_default(), "mods_list": mods, "title": title, "predicted_at": now(),
            "predictor": forecast.as_ref().map_or(json!(PREDICTOR_ID), |f| f["predictor"].clone()), "calculator": forecast.as_ref().map_or(json!(CALC_ID), |f| f["calculator"].clone()),
            "snapshot": rec.map(|r| r.public.meta.snapshot.clone()), "features": exp.as_ref().map(|e| ms_data::codec::to_json(&ms_calc::val::Val::Dict((*e.1).clone()))),
            "difficulty": exp.as_ref().map(|e| ms_calc::features::displayed_feature(&e.1, &rec.unwrap().prm)), "expected": exp.as_ref().map(|e| e.0.shown.acc_mid), "expected_lo": exp.as_ref().map(|e| e.0.shown.acc_lo),
            "expected_hi": exp.as_ref().map(|e| e.0.shown.acc_hi), "accuracy_forecast": forecast, "accuracy_forecast_recorded_at": now(), "shown_mu": exp.as_ref().map(|e| e.0.pp.mu),
            "md5": identity_md5, "keys": keys, "length": length,
        });
        if let Some((e, f, _, b)) = &exp {
            let rec = rec.unwrap();
            let map = info.as_object_mut().unwrap();
            map.extend(e.fields());
            map.insert("skill".into(), json!(crate::engine::top_skill(f)));
            let meta = b.and_then(|b| rec.public.map_info(b));
            map.insert("ranked".into(), json!(meta.is_some_and(|m| ms_data::store::RANKED_STATUS.contains(&m.approved))));
        }
        log_event(&self.db, "start", Some(&key), Some(t), &info)?;
        self.current = Some(key);
        Ok(())
    }

    fn t_abort(&mut self, play: &Value) -> Result<()> {
        let t = play["t"].as_f64();
        let end = play["end"].as_f64();
        if !tracking_allowed(&self.db, Some(end.unwrap_or_else(now)), t)? {
            return Ok(());
        }
        let start: Option<(i64, Option<String>)> = self.db.query_row("SELECT id,beatmap FROM events WHERE kind='start' AND t=? ORDER BY id DESC LIMIT 1", [t], |r| Ok((r.get(0)?, r.get(1)?))).optional()?;
        let beatmap = start.as_ref().and_then(|s| s.1.clone()).or_else(|| self.current.clone());
        log_event(&self.db, "abort", beatmap.as_deref(), end, &json!({"start_id": start.map(|s| s.0), "progress": ms_calc::py::round_digits(play["progress"].as_f64().unwrap_or(0.), 3),
                                                                     "played_seconds": play["played_seconds"].as_f64().unwrap_or(0.), "partial_stats": play.get("partial_stats")}))?;
        self.refresh_later();
        Ok(())
    }

    fn t_score(&mut self, score: &Value, path: &Path, fresh: bool, play: Option<&Value>) -> Result<()> {
        let md5 = std::fs::read(path).ok().map(|d| hashes(&d).1); // None: still stored; the next import fills the ids
        let inst = match &md5 {
            Some(m) => one_by_md5(&self.db, m)?,
            None => None,
        };
        let bid = inst.as_ref().and_then(|i| i.beatmap_id);
        let sha = score["sha256"].as_str().unwrap_or("").to_string();
        let rate = score["rate"].as_f64();
        let before = match self.rec.is_some() {
            true => {
                self.session_cache = None;
                let n = self.session()?;
                let rec = self.rec.as_ref().unwrap();
                Some((rec.r.ledger.total(), bid.map_or(0., |b| rec.r.ledger.best_of(b)), n.session.phase()))
            }
            false => None,
        };
        let mut row = score.clone();
        row["md5"] = json!(md5);
        row["beatmap_id"] = json!(bid);
        row["status"] = json!(inst.as_ref().and_then(|i| i.status));
        let (key, new) = add_live_score(&self.db, &row, fresh, play)?;
        let Some(key) = key else {
            self.status("play excluded: tracking was paused");
            return Ok(());
        };
        let t = play.and_then(|p| p["t"].as_f64());
        let mut start: Option<(i64, Option<String>, Value)> = match t {
            Some(t) => self.db.query_row("SELECT id,beatmap,info FROM events WHERE kind='start' AND t=? ORDER BY id DESC LIMIT 1", [t], |r| Ok((r.get(0)?, r.get(1)?, r.get::<_, String>(2)?))).optional()?
                .map(|(a, b, c)| (a, b, serde_json::from_str(&c).unwrap_or(Value::Null))),
            None => None,
        };
        if start.as_ref().is_some_and(|s| s.2["sha"].as_str() != Some(&sha) || s.2["rate"].as_f64() != rate) {
            start = None;
        }
        let mut info = start.as_ref().map_or(Value::Null, |s| s.2.clone());
        // a delayed startup must not masquerade as a pre-play prediction
        if info["predicted_at"].as_f64().unwrap_or(0.) > score["played"].as_str().and_then(timestamp).unwrap_or_else(now) {
            info = Value::Null;
        }
        let title = inst.as_ref().map(Inst::name).or_else(|| info["title"].as_str().map(str::to_string)).unwrap_or_else(|| sha.clone());
        if fresh {
            self.replay_due = Some(Instant::now() + Duration::from_secs(3));
            self.replay_retries = 0;
            let (acc, stats): (f64, String) = self.db.query_row("SELECT acc,stats FROM scores WHERE key=?", [&key], |r| Ok((r.get(0)?, r.get(1)?)))?;
            let display = parse_stats(&stats).map_or(0., |s| acc_lazer(&s));
            if self.db.query_row("SELECT 1 FROM events WHERE kind='finish' AND json_extract(info,'$.key')=?", [&key], |_| Ok(())).optional()?.is_some() {
                return Ok(());
            }
            log_event(&self.db, "finish", start.as_ref().and_then(|s| s.1.as_deref()), None, &json!({"key": key, "start_id": start.as_ref().map(|s| s.0), "prediction_valid": info["mu"].is_number(), "y": y_of(acc),
                                                                                                  "y_lazer": y_of(display), "pauses": play.and_then(|p| p.get("pauses")).filter(|p| !p.is_null()).cloned().unwrap_or(json!([]))}))?;
            // Show the saved play even if subsequent feature calculation/refitting fails.
            let mut v = self.verdict(&key, bid, None, &info, &title)?;
            v.targets = self.targets.iter().filter(|(_, c)| same_map(c, Some(&sha), md5.as_deref(), bid)).map(|x| x.0.clone()).collect();
            (self.h.emit)(Out::Result(Box::new(v)));
        }
        if self.rec.is_some() && (new || fresh) {
            let mods = parse_mods(score.get("mods_list"));
            if mods.as_ref().is_some_and(|m| rate.is_some_and(|r| predictable_context(m, r))) {
                self.request_feature(&sha, path, rate, true)?;
            }
            self.rec.as_mut().unwrap().observe(&self.db, &key)?;
            if let (true, Some((_, b0, _)), Some(s), Some(b)) = (fresh, before.as_ref(), start.as_ref(), bid) {
                log_event(&self.db, "pp_result", s.1.as_deref(), None, &json!({"start_id": s.0, "before": b0, "after": self.rec.as_ref().unwrap().r.ledger.best_of(b)}))?;
            }
            if !fresh && new {
                self.t_refit(); // a newly viewed old score is historical evidence, unlike a live result handled by form
            }
            self.card();
            self.refresh_later();
            if self.stats_visible {
                self.t_stats(true)?;
            }
            if fresh {
                let v = self.verdict(&key, bid, before, &info, &title)?;
                (self.h.emit)(Out::Result(Box::new(v)));
            }
        }
        Ok(())
    }

    /// A finished play against what was shown, and what it did.
    fn verdict(&mut self, key: &str, bid: Option<i64>, before: Option<(f64, f64, &'static str)>, info: &Value, title: &str) -> Result<Verdict> {
        let text: String = self.db.query_row("SELECT stats FROM scores WHERE key=?", [key], |r| r.get(0))?;
        let st: Stats = parse_stats(&text).unwrap_or([0; 6]);
        let (acc, _) = acc320(&st);
        // Shown as the connected game shows it; the verdict compares lazer units (stable: shown acc of the forecast).
        let shown = self.shown.play(&st, acc_lazer(&st));
        let mut out = vec![format!("{:.2}%", 100. * shown)];
        let expected = info["expected"].as_f64().map(|e| self.shown.acc(e));
        let mut z = None;
        if let (Some(mu), Some(sd)) = (info["mu"].as_f64(), info["sd"].as_f64().filter(|&s| s != 0.)) {
            let shown_mu = info["shown_mu"].as_f64().unwrap_or(mu + PESSIMISM * info["sdm"].as_f64().unwrap_or(0.));
            let mut value = (shown_mu - y_of(acc)) / sd;
            if let (Some(m), Some(s)) = (info["display_mu"].as_f64(), info["display_sd"].as_f64().filter(|&s| s != 0.)) {
                value = (m - y_of(acc_lazer(&st))) / s;
            }
            z = Some(value);
            out.push(format!("{}{}", expected.map_or_else(String::new, |e| format!("expected ~{:.2}% · ", 100. * e)), verdict_words(value)));
        }
        let mut news = Vec::new();
        if let Some((t0, b0, ph0)) = before {
            let rec = self.rec.as_ref().unwrap();
            let total = rec.r.ledger.total();
            let b1 = bid.map_or(0., |b| rec.r.ledger.best_of(b));
            if b1 > b0 + 0.05 {
                news.push(if b0 != 0. { format!("new best {b1:.0}pp (was {b0:.0}) · account {:+.1}pp", total - t0) } else { format!("new map {b1:.0}pp · account {:+.1}pp", total - t0) });
            }
            self.session_cache = None;
            let ph1 = self.session()?.session.phase();
            if ph1 != ph0 {
                news.push(match ph1 {
                    "build" => "warmed up — building",
                    "push" => "in form — pushing",
                    "recover" => "easing off",
                    "fading" => "fading",
                    _ => "warming up",
                }.to_string());
            }
            out.extend(news.iter().cloned());
        }
        Ok(Verdict { title: title.to_string(), text: out.join(" · "), acc: shown, expected, z, rate: info["rate"].as_f64(), when: Some(now()), news, targets: Vec::new() })
    }

    fn card(&mut self) {
        let (path, rate, mods) = self.sel.clone();
        let Some(path) = path.filter(|_| self.rec.is_some()) else {
            let note = self.sel.0.as_ref().map(|_| self.loading.clone().unwrap_or_else(|| "Loading prediction model…".to_string()));
            (self.h.emit)(Out::Card { c: None, note, path: self.sel.0.clone(), rate });
            return;
        };
        let (mut c, mut note) = (None, None);
        match (rate, mods) {
            (Some(r), Some(m)) if predictable_context(&m, r) => {
                if std::fs::metadata(&path).is_err() {
                    note = Some("Selected chart file is unavailable".to_string());
                } else {
                    if let Ok(n) = self.session() {
                        c = self.rec.as_mut().unwrap().card(&self.db, &path, r, &n).ok().flatten();
                    }
                    if c.is_none() {
                        let sha = std::fs::read(&path).map(|d| hashes(&d).0).unwrap_or_default();
                        let failed = self.feature_errors.lock().unwrap().get(&(sha, ms_calc::py::round_digits(r, 3).to_bits())).map(|x| x.1.clone());
                        note = Some(failed.map_or_else(|| format!("Calculating prediction at {r:.2}×…"), |f| format!("Prediction unavailable at {r:.2}×: {f}")));
                    }
                }
            }
            (Some(_), Some(_)) => note = Some("Prediction unavailable for these mods".to_string()),
            _ => note = Some("Waiting for osu! to report the playback rate and mods".to_string()),
        }
        if c.is_some() {
            if let Some(asked) = self.q.selected_at.lock().unwrap().take() {
                if asked.elapsed().as_secs_f64() > 1. {
                    eprintln!("selected card latency {:.1} s", asked.elapsed().as_secs_f64());
                }
            }
        }
        (self.h.emit)(Out::Card { c: c.map(Box::new), note, path: Some(path), rate });
    }

    #[allow(clippy::too_many_arguments)]
    fn t_next(&mut self, skip: bool, mode: Option<String>, revision: Option<u64>, candidate: Option<Box<Pick>>, result: Option<Done>, selected_at: Option<f64>, selection_id: u64) -> Result<()> {
        let mode = mode.unwrap_or_else(|| self.mode.clone());
        let m = key_mode(&mode);
        if !MODES.contains(&mode.as_str()) || (result.is_none() && (revision.is_some_and(|r| r != self.revision) || self.keys[m].is_empty())) || self.rec.is_none() {
            return Ok(());
        }
        if skip {
            if let Some(t) = self.targets.get(&mode) {
                log_event(&self.db, "skip", t.identity().as_deref(), None, &json!({"mode": mode}))?;
            }
        }
        match candidate.or_else(|| self.playlists[m].first().cloned().map(Box::new)) {
            Some(pick) => {
                log_event(&self.db, "next", pick.identity().as_deref(), selected_at, &json!({"purpose": pick.purpose, "var": pick.var, "mode": mode}))?;
                self.open(*pick, &mode, "next", result, selected_at, selection_id)?;
            }
            None => (self.h.emit)(Out::Target { mode, revision: self.revision, selection_id: self.selection_id, c: None, msg: "Preparing the next playlist…".to_string(), copy: None }),
        }
        // Let pending selections log before refreshing the remainder.
        self.q.put(Task::Publish);
        Ok(())
    }

    fn t_open(&mut self, pick: Box<Pick>, mode: Option<String>, revision: Option<u64>, result: Option<Done>, selected_at: Option<f64>, selection_id: u64) -> Result<()> {
        let mode = mode.or_else(|| Some(pick.mode.clone()).filter(|m| !m.is_empty())).unwrap_or_else(|| self.mode.clone());
        if !MODES.contains(&mode.as_str()) {
            return Ok(());
        }
        let m = key_mode(&mode);
        if result.is_none() && (revision.is_some_and(|r| r != self.revision) || !self.keys[m].contains(&pick.keys) || (mode == "skills" && pick.practice_skills != self.practice_skills)) {
            return Ok(());
        }
        self.open(*pick, &mode, "list", result, selected_at, selection_id)?;
        self.publish(None)
    }

    fn open(&mut self, mut c: Pick, mode: &str, selection: &str, result: Option<Done>, selected_at: Option<f64>, selection_id: u64) -> Result<()> {
        c.mode = mode.to_string();
        c.selection = selection.to_string();
        let performed = result.is_some();
        let done = result.unwrap_or_else(|| {
            let action = if (self.h.busy)() { "copy" } else { self.action.as_str() };
            let p = perform(&mut c, action, &self.shown.client);
            Done { copy: p.copy, msg: p.msg, sent: p.sent, unavailable: p.unavailable }
        });
        self.selection_id = self.selection_id.max(selection_id);
        let key = c.identity();
        let logged = if c.purpose == "nps_mix" { "pp" } else { mode }; // a PP map in the NPS list is a PP offer
        if let Some(k) = &key {
            self.consumed.push_back(k.clone());
            if self.consumed.len() > 20 {
                self.consumed.pop_front();
            }
        }
        for list in self.playlists.values_mut() {
            list.retain(|p| p.identity() != key);
        }
        if done.unavailable {
            self.status(done.msg);
            return Ok(());
        }
        let shown = (mode == "skills").then(|| (c.practice_skills.clone(), c.matched_skills.clone()));
        let forecast = self.rec.as_ref().filter(|r| r.r.fitted.display.is_some()).and_then(|_| {
            let e = crate::engine::Expected { pp: c.pred.clone(), shown: c.pred.clone(), direct: true };
            forecast_chart(e.forecast(c.chart(), c.rate), c.sha.as_deref(), c.md5.as_deref())
        });
        let info = json!({
            "mode": logged, "keys": c.keys, "group": c.group, "bid": c.bid, "rate": c.rate, "selection": selection, "length": c.length, "accuracy": c.pred.acc_mid, "title": c.title,
            "opening_accuracy": c.pred.opening_acc, "activation_minutes": c.pred.activation, "purpose": c.purpose, "md5": c.md5, "sha": c.sha, "web": c.stream() == WEB, "convert": c.convert_of.is_some(), "p_up": c.p_up, "gain": c.gain, "best_pp": c.best,
            "pp_mid": c.pp_mid, "practice_skills": shown.as_ref().map(|s| &s.0), "matched_skills": shown.as_ref().map(|s| &s.1), "accuracy_forecast": forecast, "accuracy_forecast_recorded_at": now(),
            "predictor": forecast.as_ref().map_or(json!(PREDICTOR_ID), |f| f["predictor"].clone()), "calculator": forecast.as_ref().map_or(json!(CALC_ID), |f| f["calculator"].clone()),
        });
        c.offer_id = Some(log_event(&self.db, "offer", key.as_deref(), selected_at, &info)?);
        let boxed = Box::new(c);
        self.target = Some(boxed.clone());
        self.targets.insert(mode.to_string(), boxed.clone());
        if boxed.convert_of.is_some() && !boxed.installed {
            self.deliver(boxed.clone(), mode.to_string());
        }
        if done.sent {
            log_event(&self.db, "open", key.as_deref(), selected_at, &json!({"var": boxed.var, "mode": mode}))?;
        }
        let msg = format!("{} · set {} manually", done.msg, boxed.var);
        (self.h.emit)(Out::Target { mode: mode.to_string(), revision: self.revision, selection_id: self.selection_id, c: Some(boxed), msg, copy: if performed { None } else { done.copy } });
        Ok(())
    }

    /// A convert's turn: written and imported once it stays the pick for a moment and no map is being played.
    fn deliver(&mut self, pick: Box<Pick>, mode: String) {
        let turn = self.convert_turn.fetch_add(1, Ordering::SeqCst) + 1;
        let (q, h, halt, turns, path, client) = (self.q.clone(), self.h.clone(), self.halt.clone(), self.convert_turn.clone(), self.db_path.clone(), self.shown.client.clone());
        let _ = std::thread::Builder::new().name("convert".into()).spawn(move || {
            std::thread::sleep(CONVERT_WAIT);
            while (h.busy)() && !halt.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_secs(1));
            }
            if turns.load(Ordering::SeqCst) != turn || halt.load(Ordering::Relaxed) {
                return;
            }
            let root = crate::catalog::local_origin(&client).1;
            let error = if crate::navigation::game_running(&client) {
                connect(&path).map_err(|e| e.to_string()).and_then(|db| converts::deliver(&pick, &db, &client, &root)).err()
            } else {
                Some(NOT_RUNNING.to_string())
            };
            q.put(Task::ConvertDone { pick, mode, error });
        });
    }

    /// The map selected in osu! as a convert, on request and outside any list: delivered like a list's.
    fn t_convert_selected(&mut self, target: i64) {
        let Some(path) = self.sel.0.clone() else { return self.status("Select a map in osu! first") };
        let converted = std::fs::read(&path).map_err(|e| e.to_string()).and_then(|data| Ok((hashes(&data).0, ms_convert::convert(&data, target as usize).map_err(|e| e.to_string())?)));
        let (source, out) = match converted {
            Ok(done) => done,
            Err(e) => return self.status(format!("Couldn't convert that map: {e}")),
        };
        let m = ms_data::lazer_index::metadata(&path).unwrap_or_default();
        let (sha, md5) = hashes(&out.osu);
        let text = |k: &str| m.get(k).cloned();
        let pick = Pick { sha: Some(sha), md5: Some(md5), title: text("Title").unwrap_or_default(), artist: text("Artist"), song: text("Title"), version: Some(out.version), creator: text("Creator"),
                          keys: target, set_id: text("BeatmapSetID").and_then(|s| s.parse().ok()), convert_of: Some(source), ..Pick::default() };
        self.status("Converting…");
        self.deliver(Box::new(pick), SELECTED.to_string());
    }

    fn t_convert_done(&mut self, pick: Box<Pick>, mode: String, error: Option<String>) -> Result<()> {
        let current = self.targets.get(&mode).is_some_and(|t| t.identity() == pick.identity());
        match error {
            None if mode == SELECTED => {
                let first = if self.shown.client == "stable" { "press F5 in song select, then Ctrl+V" } else { "press Ctrl+V in song select" };
                (self.h.emit)(Out::Copy { text: crate::navigation::song_search(&pick, &self.shown.client), msg: format!("Ready — {first}") });
            }
            None if current => {
                let mut c = pick;
                c.installed = true;
                let copy = Some(crate::navigation::song_search(&c, &self.shown.client));
                let first = if self.shown.client == "stable" { "press F5 in song select, then Ctrl+V" } else { "press Ctrl+V in song select" };
                let msg = format!("Ready — {first} · set {} manually", c.var);
                self.targets.insert(mode.clone(), c.clone());
                self.target = Some(c.clone());
                (self.h.emit)(Out::Target { mode, revision: self.revision, selection_id: self.selection_id, c: Some(c), msg, copy });
            }
            None => {}
            Some(e) if e == NOT_RUNNING => self.status(format!("{e} — start it, then press Next")),
            Some(e) => {
                eprintln!("convert failed: {} {e}", pick.title);
                self.status(format!("Couldn't convert that map: {e}"));
                if current {
                    self.q.put(Task::Next { skip: false, mode: Some(mode), revision: None, candidate: None, result: None, selected_at: None, selection_id: self.selection_id });
                }
            }
        }
        Ok(())
    }

    fn t_selected(&mut self, path: Option<PathBuf>, rate: Option<f64>, mods: Option<Vec<Value>>) -> Result<()> {
        let previous = self.sel.0.clone();
        self.sel = (path.clone(), rate, mods.clone());
        let mut hashed = None;
        if let Some(p) = &path {
            if let Ok(data) = std::fs::read(p) {
                let (sha, md5) = hashes(&data);
                if mods.as_ref().is_some_and(|m| rate.is_some_and(|r| predictable_context(m, r))) {
                    self.request_feature(&sha, p, rate, false)?;
                }
                if self.rec.is_some() && previous.as_ref() != Some(p) && !self.local_shas().contains(&sha) {
                    self.begin_import();
                }
                hashed = Some((sha, md5));
            }
        }
        self.card();
        let (Some(target), Some(p), Some((sha, md5))) = (self.target.clone(), path, hashed) else { return Ok(()) };
        let bid = ms_data::lazer_index::metadata(&p).and_then(|m| m.get("BeatmapID")?.parse().ok());
        if same_map(&target, Some(&sha), Some(&md5), bid) {
            let updated = target.md5.as_deref() != Some(md5.as_str());
            log_event(&self.db, "selected", Some(&format!("md5:{md5}")), None, &json!({"offered_revision": target.identity(), "updated": updated}))?;
            let msg = format!("✓ {}chart selected — intended {}; {}", if updated { "updated " } else { "" }, target.var, rate.map_or_else(|| "actual rate unknown".to_string(), |r| format!("observed {r:.2}×")));
            (self.h.emit)(Out::Target { mode: target.mode.clone(), revision: self.revision, selection_id: self.selection_id, c: Some(target), msg, copy: None });
        }
        Ok(())
    }

    fn request_feature(&mut self, sha: &str, path: &Path, rate: Option<f64>, refit: bool) -> Result<()> {
        let (Some(rate), Some(rec)) = (rate, self.rec.as_mut()) else { return Ok(()) };
        if sha.is_empty() || rec.features(&self.db, Some(sha), None, None, rate)?.0.is_some() {
            return Ok(());
        }
        let id = (sha.to_string(), ms_calc::py::round_digits(rate, 3).to_bits());
        {
            let mut errors = self.feature_errors.lock().unwrap();
            if errors.get(&id).is_some_and(|f| f.0 > Instant::now()) {
                return Ok(());
            }
            errors.remove(&id);
        }
        let mut pending = self.feature_pending.lock().unwrap();
        pending.0 = Some((sha.to_string(), path.to_path_buf(), rate, refit));
        if pending.1 {
            return Ok(()); // the newest request replaces a pending old selection
        }
        pending.1 = true;
        let (queue, errors, slot, halt, h, prm) = (self.q.clone(), self.feature_errors.clone(), self.feature_pending.clone(), self.halt.clone(), self.h.clone(), self.rec.as_ref().unwrap().prm.clone());
        std::thread::Builder::new().name("selected-feature".into()).spawn(move || loop {
            let task = {
                let mut p = slot.lock().unwrap();
                let task = p.0.take();
                if task.is_none() || halt.load(Ordering::Relaxed) {
                    p.1 = false;
                    return;
                }
                task.unwrap()
            };
            // Not deferred during play: waiting left the card on "Calculating…" for the whole song after an in-game
            // rate change (2026-10-10).
            let (sha, path, rate, refit) = task;
            let id = (sha.clone(), ms_calc::py::round_digits(rate, 3).to_bits());
            match Engine::analyse(&path, ms_calc::py::round_digits(rate, 3), &prm) {
                Ok(fs) => {
                    errors.lock().unwrap().remove(&id);
                    queue.put(Task::FeatureReady { sha, features: Some(fs), refit });
                }
                Err(e) => {
                    let mut list = errors.lock().unwrap();
                    if list.len() > 64 {
                        list.clear();
                    }
                    list.insert(id, (Instant::now() + Duration::from_secs(30), e.chars().take(240).collect()));
                    drop(list);
                    (h.emit)(Out::Status { text: format!("Selected chart prediction unavailable: {}", e.chars().take(160).collect::<String>()), progress: None, blocking: false, learning: None });
                    queue.put(Task::FeatureReady { sha, features: None, refit: false });
                }
            }
        }).expect("feature thread");
        Ok(())
    }

    fn t_client(&mut self, name: &str) -> Result<()> {
        if name != "lazer" && name != "stable" {
            return Ok(());
        }
        kv_set(&self.db, "client", &json!(name))?;
        if name == self.shown.client {
            return Ok(());
        }
        self.shown.client = name.to_string();
        self.shown.miss_ratio = stable_miss_ratio(&self.db)?;
        self.first_import = true;
        self.begin_import();
        Ok(())
    }

    /// Restored recommendations are usable before the game's files are read. One import is coalesced; the live event and
    /// model owner remains this worker.
    fn begin_import(&mut self) {
        if self.jobs.importing.swap(true, Ordering::SeqCst) {
            return;
        }
        let game = client_name(&self.shown.client);
        self.status(if self.first_import { format!("Reading your {game} scores and installed maps (first time: about a minute)…") } else { format!("Using saved history · checking {game} for new plays in the background…") });
        let (q, h, halt, jobs, path, client) = (self.q.clone(), self.h.clone(), self.halt.clone(), self.jobs.clone(), self.db_path.clone(), self.shown.client.clone());
        std::thread::Builder::new().name("score-import".into()).spawn(move || {
            while (h.busy)() && !halt.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_secs(1));
            }
            if halt.load(Ordering::Relaxed) {
                return;
            }
            let run = || -> std::result::Result<(Imported, usize, bool), String> {
                let db = connect(&path).map_err(|e| e.to_string())?;
                let signature = |db: &Db| -> std::result::Result<Vec<u8>, String> {
                    use sha2::{Digest, Sha256};
                    let mut digest = Sha256::new();
                    let mut q = db.prepare("SELECT sha256,md5,status,keys FROM installed ORDER BY sha256").map_err(|e| e.to_string())?;
                    let rows = q.query_map([], |r| Ok((r.get::<_, Option<String>>(0)?, r.get::<_, Option<String>>(1)?, r.get::<_, Option<i64>>(2)?, r.get::<_, Option<i64>>(3)?))).map_err(|e| e.to_string())?;
                    for row in rows.flatten() {
                        digest.update(format!("{row:?}"));
                    }
                    Ok(digest.finalize().to_vec())
                };
                let before = signature(&db)?;
                let result = import_library(&db, &client)?;
                let recovered = recover_results(&db).map_err(|e| e.to_string())?;
                let mut changed = signature(&db)? != before;
                // Website top 100 for a named profile: bests from stable/other PCs (user 2026-10-01).
                let name = kv_get(&db, "website_username").and_then(|v| v.as_str().map(str::to_string));
                let last = kv_get(&db, "website_user").and_then(|v| v["t"].as_f64()).unwrap_or(0.);
                if let Some(name) = name.filter(|_| now() - last > 6. * 3600.) {
                    match imports::import_website_best(&db, &name) {
                        Ok(_) => changed = true,
                        Err(e) => (h.emit)(Out::Status { text: format!("osu! profile scores unavailable: {e}").chars().take(160).collect(), progress: None, blocking: false, learning: None }),
                    }
                }
                Ok((result, recovered, changed))
            };
            let task = match run() {
                Ok((result, recovered, changed)) => Task::Imported { result: Some(result), recovered, changed, error: None },
                Err(e) => Task::Imported { result: None, recovered: 0, changed: false, error: Some(e.lines().next().unwrap_or("").chars().take(160).collect()) },
            };
            jobs.importing.store(false, Ordering::SeqCst);
            q.put(task);
        }).expect("import thread");
    }

    fn t_imported(&mut self, result: Option<Imported>, recovered: usize, changed: bool, error: Option<String>) -> Result<()> {
        let first = std::mem::take(&mut self.first_import);
        let game = client_name(&self.shown.client);
        if let Some(e) = error {
            if first {
                self.stage(Some(format!("Could not read your {game} scores: {e}")), true);
            } else {
                self.status(format!("Saved recommendations retained; {game} import pending: {e}"));
            }
            return Ok(());
        }
        let Some(r) = result else { return Ok(()) };
        if let Some(m) = r.miss_ratio {
            self.shown.miss_ratio = m;
        }
        let head = if r.new != 0 { format!("Read {} new plays and {} installed maps from {game}", crate::engine::commas(r.new as usize), crate::engine::commas(r.installed)) } else { format!("Up to date with {game} ({} maps)", crate::engine::commas(r.installed)) };
        self.status(format!("{head}{}", if recovered != 0 { format!(" · {recovered} predictions recovered") } else { String::new() }));
        if r.new != 0 || recovered != 0 {
            let rows = play_history(&self.db, 1, false, None)?;
            if let Some(p) = rows.first() {
                (self.h.emit)(Out::Result(Box::new(Verdict { title: p.title.clone(), text: history_text(&rows, &self.shown), acc: self.shown.play(&p.stats, p.actual), expected: p.expected.map(|e| self.shown.acc(e)), z: p.z, rate: p.rate,
                                                             when: p.played.as_deref().and_then(timestamp), news: Vec::new(), targets: Vec::new() })));
            }
        }
        if r.new != 0 && self.rec.is_some() {
            self.start_fill(); // before the refit: its publish must already know the level is pending
        }
        if r.new != 0 || recovered != 0 || changed {
            self.t_refit();
        }
        Ok(())
    }

    fn first_learning(&self) -> bool {
        self.rec.as_ref().is_none_or(|r| r.r.fitted.rows.rows.len() < 30)
    }

    /// Features for the player's own charts that the public table lacks (custom rates, unranked maps). A first run imports
    /// thousands of plays after this started once with none (fresh install 2026-10-02: 5,632 chart/rates never analysed,
    /// so PP suggested 4K Beginner maps to a 7K player). One at a time; an import that brings new plays starts another.
    fn start_fill(&mut self) {
        if self.jobs.filling.swap(true, Ordering::SeqCst) {
            return;
        }
        let first = self.first_learning();
        if first {
            *self.learning.lock().unwrap() = Some("Checking which of your played charts still need analysing…".to_string());
        }
        let rec = self.rec.as_ref().unwrap();
        let (q, h, halt, jobs, path, public, prm, learning, c) = (self.q.clone(), self.h.clone(), self.halt.clone(), self.jobs.clone(), self.db_path.clone(), rec.public.clone(), rec.prm.clone(), self.learning.clone(), self.c.clone());
        let installed = rec.r.installed.clone();
        std::thread::Builder::new().name("features".into()).spawn(move || {
            let mut filled = false;
            let mut run = || -> Result<()> {
                let db = connect(&path)?;
                let mut source = crate::evidence::FeatureSource { public: public.clone(), calc: CALC_ID.to_string(), cache: HashMap::default() };
                let rows: Vec<(Option<String>, Option<String>, Option<i64>, Option<f64>)> = db.prepare("SELECT sha256, md5, beatmap_id, rate FROM scores WHERE sha256 IS NOT NULL")?
                    .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))?.collect::<Result<_>>()?;
                source.prefetch(&db, rows.iter().filter_map(|r| Some(format!("{}@{:.3}|{CALC_ID}", r.0.as_ref()?, ms_calc::py::round_digits(r.3.filter(|&x| x != 0.).unwrap_or(1.), 3)))))?;
                let mut need: std::collections::BTreeMap<String, Vec<f64>> = Default::default();
                let mut want = |source: &mut crate::evidence::FeatureSource, sha: &str, md5: Option<&str>, bid: Option<i64>, rate: f64| -> Result<()> {
                    if source.of(&db, Some(sha), md5, bid, Some(rate))?.0.is_none() && pp::chart_path(Some(sha)).is_some() {
                        let rate = ms_calc::py::round_digits(rate, 3);
                        let list = need.entry(sha.to_string()).or_default();
                        if !list.contains(&rate) {
                            list.push(rate);
                        }
                    }
                    Ok(())
                };
                for r in &rows {
                    want(&mut source, r.0.as_deref().unwrap(), r.1.as_deref(), r.2, r.3.filter(|&x| x != 0.).unwrap_or(1.))?;
                }
                for inst in pp::newer_installed(&public, &installed) {
                    for (rate, _) in pp::VARIANTS {
                        want(&mut source, &inst.sha256, inst.md5.as_deref(), None, rate)?;
                    }
                }
                let _ = &c;
                let jobs_list: Vec<(String, Vec<f64>)> = need.into_iter().collect();
                if jobs_list.is_empty() {
                    return Ok(());
                }
                filled = true;
                let (t0, total) = (Instant::now(), jobs_list.len());
                let mut refits = 0;
                let emit = |done: usize| {
                    let speed = done as f64 / t0.elapsed().as_secs_f64().max(1e-3);
                    let left = if speed > 0. { (total - done) as f64 / speed } else { 0. };
                    let eta = if left >= 90. { format!(" · ~{:.0} min left", left / 60.) } else if left > 0. { format!(" · ~{left:.0} s left") } else { String::new() };
                    let (d, t) = (crate::engine::commas(done), crate::engine::commas(total));
                    let text = if first { format!("Analysing the charts you have played, to learn your level · {d} / {t} charts{eta}") } else { format!("Analysing {t} new or updated maps in the background · {d} done{eta}") };
                    *learning.lock().unwrap() = first.then(|| text.clone());
                    (h.emit)(Out::Status { text, progress: Some((done, total)), blocking: false, learning: Some(first) });
                };
                emit(0);
                // First learning is all the player waits for: every core but two, paused while playing.
                let workers = if first { webmaps::workers(None) } else { 1 };
                let chunk = 12.max(4 * workers);
                for (i, part) in jobs_list.chunks(chunk).enumerate() {
                    while (h.busy)() && !halt.load(Ordering::Relaxed) {
                        std::thread::sleep(Duration::from_secs(2));
                    }
                    if halt.load(Ordering::Relaxed) {
                        return Ok(());
                    }
                    let data: Vec<(Vec<u8>, Vec<f64>)> = part.iter().map(|(sha, rates)| (pp::chart_path(Some(sha)).and_then(|p| std::fs::read(p).ok()).unwrap_or_default(), rates.clone())).collect();
                    for ((sha, _), res) in part.iter().zip(features_batch(&data, workers, &prm)) {
                        let Ok(fs) = res else { continue };
                        let tx = db.unchecked_transaction()?;
                        for (rate, f) in fs {
                            tx.execute("INSERT OR REPLACE INTO feats VALUES (?, ?)", rusqlite::params![source.key(sha, rate), ms_data::codec::encode(&ms_calc::val::Val::Dict(f))])?;
                        }
                        tx.commit()?;
                    }
                    let done = total.min((i + 1) * chunk);
                    emit(done);
                    // Only while the level is unknown: each refit restarts the NPS/Skills build.
                    if first && done - refits >= 200.max(total / 5) && done < total {
                        refits = done;
                        q.put(Task::Refit); // suggestions sharpen while the rest is analysed
                    }
                }
                Ok(())
            };
            let outcome = run();
            *learning.lock().unwrap() = None;
            jobs.filling.store(false, Ordering::SeqCst);
            match outcome {
                Ok(()) if filled => {
                    q.put(Task::Refit);
                    (h.emit)(Out::Status { text: String::new(), progress: None, blocking: false, learning: Some(false) });
                }
                Ok(()) => {}
                Err(e) => {
                    eprintln!("feature fill: {e}");
                    (h.emit)(Out::Status { text: "Some score analyses could not be completed; available recommendations are kept".to_string(), progress: None, blocking: false, learning: Some(false) });
                }
            }
        }).expect("feature fill thread");
    }

    /// Archive independently: neither a busy queue nor slow game-file I/O blocks Next.
    fn catch_up_replays(&mut self) {
        if self.replay_due.is_none_or(|d| Instant::now() < d) || (self.h.busy)() || self.jobs.importing.load(Ordering::SeqCst) || self.jobs.archiving.swap(true, Ordering::SeqCst) {
            return;
        }
        self.replay_due = None;
        let (q, jobs, path, client) = (self.q.clone(), self.jobs.clone(), self.db_path.clone(), self.shown.client.clone());
        std::thread::Builder::new().name("replay-archive".into()).spawn(move || {
            let run = || -> std::result::Result<(bool, Vec<String>), String> {
                let e = |x: rusqlite::Error| x.to_string();
                let db = connect(&path).map_err(e)?;
                if client == "stable" {
                    recover_results(&db).map_err(e)?;
                    replays::archive_stable(&db, None).map_err(e)?; // stable's scores came live; its replay file follows
                } else {
                    // A small post-result catch-up: recent scores only, no full library scan.
                    let since = chrono::Utc::now().timestamp() - 3 * 86400;
                    let raw = imports::snapshot(None, false, Some((since, 0)))?;
                    let user = kv_get(&db, "user_id").and_then(|v| v.as_i64());
                    for row in imports::realm_rows(&raw, user) {
                        ms_data::store::upsert_score(&db, row).map_err(e)?;
                    }
                    recover_results(&db).map_err(e)?;
                    replays::archive(&db, &raw, "lazer").map_err(e)?;
                }
                let pending = db.query_row("SELECT 1 FROM replay_evidence WHERE status='missing' AND julianday(played)>=julianday('now','-3 days') LIMIT 1", [], |_| Ok(())).optional().map_err(e)?.is_some();
                let fresh: Vec<String> = db.prepare("SELECT key FROM scores WHERE fresh=1 AND julianday(played)>=julianday('now','-3 days')").map_err(e)?.query_map([], |r| r.get(0)).map_err(e)?.flatten().collect();
                Ok((pending, fresh))
            };
            let task = match run() {
                Ok((pending, fresh)) => Task::ReplayDone { pending, error: None, fresh },
                Err(e) => Task::ReplayDone { pending: true, error: Some(e.chars().take(120).collect()), fresh: Vec::new() },
            };
            jobs.archiving.store(false, Ordering::SeqCst);
            q.put(task);
        }).expect("replay thread");
    }

    fn t_replay_done(&mut self, pending: bool, error: Option<String>, fresh: Vec<String>) -> Result<()> {
        if self.rec.is_some() && !fresh.is_empty() {
            for key in &fresh {
                self.rec.as_mut().unwrap().observe(&self.db, key)?;
            }
            self.refresh_later();
        }
        if pending {
            self.replay_retries += 1;
            if self.replay_retries <= 3 {
                self.replay_due = Some(Instant::now() + Duration::from_secs(20));
            }
        } else {
            self.replay_retries = 0;
        }
        if let Some(e) = error {
            self.status(format!("Replay archive pending: {e}"));
        }
        Ok(())
    }

    /// A refit runs on its own connection and thread; the worker keeps answering cards from the previous model (a 15–40 s
    /// refit held the selected card that long, G835LX run.log 2026-10).
    fn t_refit(&mut self) {
        if (self.h.busy)() {
            self.refit_due = true;
            return;
        }
        self.refit_due = false;
        let Some(rec) = self.rec.as_ref() else { return };
        self.last_refit = now();
        if self.refit_running {
            self.refit_again = true; // newer evidence arrived: fit once more after this one
            return;
        }
        self.refit_running = true;
        *rec.observing.lock().unwrap() = Some(Vec::new());
        let (q, public, c, path) = (self.q.clone(), rec.public.clone(), self.c.clone(), self.db_path.clone());
        std::thread::Builder::new().name("refit".into()).spawn(move || {
            let result = connect(&path).and_then(|db| refit(&db, &public, &c));
            if let Err(e) = &result {
                eprintln!("refit: {e}");
            }
            q.put(Task::RefitDone(result.ok().map(Box::new)));
        }).expect("refit thread");
    }

    fn t_refit_done(&mut self, r: Option<Box<Refit>>) -> Result<()> {
        self.refit_running = false;
        if let (Some(r), Some(rec)) = (r, self.rec.as_mut()) {
            let observed = rec.observing.lock().unwrap().take().unwrap_or_default();
            rec.apply(&self.db, *r, observed)?;
            self.generation += 1;
            self.session_cache = None;
            self.request_nps();
            if self.mode == "pp" {
                self.publish(None)?;
            }
            self.card();
            if self.stats_visible {
                self.t_stats(true)?;
            }
        } else if let Some(rec) = self.rec.as_ref() {
            *rec.observing.lock().unwrap() = None;
        }
        if std::mem::take(&mut self.refit_again) {
            self.t_refit();
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "worker_tests.rs"]
mod tests;
