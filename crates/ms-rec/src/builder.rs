//! nps.py: installed-map density and Skills lists. One coalescing, low-priority builder: sparse work is persistent,
//! feature batches enrich the cache and each publication uses exact analyses, never a scaling guess.
use crate::catalog::{library, Inst, Pick, rate_label};
use ms_calc::features::features_batch;
use ms_calc::params::Params;
use ms_calc::val::Val;
use ms_data::codec;
use ms_data::public::Public;
use ms_data::store::{kv_get, tracked_events, Db, Result};
use ms_data::webmaps::{web_dir, workers, Web};
use ms_model::choices::{ChoiceList, Choices, Forecast, Ranking};
use ms_model::predict::{Consts, FeatStore, Feats, HashMap, PredModel, PredSession, ScoreJob, MAX_RATE, MIN_RATE};
use ms_model::session::Session;
use ms_model::skills;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet, VecDeque};
use std::sync::{Arc, RwLock};

pub const STRUCTURE_VERSION: i64 = 2;
pub const DEFAULT_TARGET: f64 = 0.95;
const LEAD: usize = 300; // charts re-predicted first after a play or refit (the shown 20 come from the top ~60)
const STABLE_RATES: [f64; 3] = [0.75, 1., 1.5]; // stable has HT/NM/DT only (user 2026-10-31)
pub const PROFILE: [&str; 21] = ["stream", "delay", "jumpstream", "handstream", "chordstream", "chordjack", "jack", "jackspeed", "minijack", "longjack", "jumptrill",
    "splittrill", "bracket", "ln", "release", "inverse", "hybrid", "shield", "patterntech", "rhythmtech", "sv"];

pub fn offerable(rate: f64, client: &str) -> bool {
    client != "stable" || STABLE_RATES.contains(&rate)
}

/// The nearest rate the client can play.
fn snap(rate: f64, client: &str) -> f64 {
    if client != "stable" {
        return rate;
    }
    STABLE_RATES.into_iter().min_by(|a, b| (a / rate).ln().abs().total_cmp(&(b / rate).ln().abs())).unwrap()
}

pub fn normalize_focus(value: f64) -> f64 {
    if value.is_finite() { value.clamp(0., 1.) } else { 0.5 }
}

/// 'Maps you don't have' slider → their odds against your own maps. Their list share rises linearly: 0 at the left
/// end, half in the middle, 80 % at the right (an exponent jumped 0 → 22 %, user 2026-10-26).
pub fn web_factor(value: f64) -> f64 {
    let value = normalize_focus(value);
    let share = if value <= 0.5 { value } else { 0.5 + 0.6 * (value - 0.5) };
    share / (1. - share)
}

/// The tab's displayed-accuracy target (Settings slider): 88–98 %, default 95 %.
pub fn normalize_target(value: f64) -> f64 {
    if value.is_finite() { value.clamp(0.88, 0.98) } else { DEFAULT_TARGET }
}

/// {keys: (lo, hi)} ratings that can land near the target: website charts outside are never loaded. The level +
/// population curve alone put the target at ov; every NPS candidate of 2026-10-01 (4,513 website chart-rates) lay
/// within 0.64–1.00·ov, so 0.55–1.10 loses none.
pub fn reach(model: &PredModel, keys: &[i64], target: f64, c: &Consts) -> HashMap<i64, (f64, f64)> {
    keys.iter().map(|&k| {
        let level = model.level.get(&k).map_or(model.glob + 0.3, |l| l.0);
        let y = (1. - target).ln();
        let (mut lo, mut hi) = (0.5f64.ln(), 60f64.ln());
        for _ in 0..40 {
            let mid = (lo + hi) / 2.;
            if level + model.base_of(&Feats::bare(k, mid.exp(), 8., c), None, 0., None).0 < y { lo = mid } else { hi = mid }
        }
        let tenth = |x: f64| (x * 10.).round() / 10.;
        (k, (tenth(0.55 * lo.exp()), tenth(1.1 * lo.exp())))
    }).collect()
}

/// Shrunk completion and offer rates per (keys, group). Next alone is never a positive vote.
pub fn taste(db: &Db, now: f64) -> Result<HashMap<(i64, String), f64>> {
    let reset: f64 = db.query_row("SELECT MAX(t) FROM events WHERE kind='taste_reset'", [], |r| r.get::<_, Option<f64>>(0))?.unwrap_or(0.);
    let since = reset.max(now - 90. * 86400.);
    let (mut offers, mut starts, mut completed) = (Vec::new(), HashMap::<i64, Option<i64>>::default(), HashSet::new());
    for e in tracked_events(db, since, &["offer", "start", "finish"], None, None)? {
        if e.t <= since {
            continue;
        }
        match e.kind.as_str() {
            "offer" if e.info.get("mode").and_then(Value::as_str) == Some("nps") => offers.push((e.id, e.info, 0.5f64.powf((now - e.t) / (30. * 86400.)))),
            "start" => { starts.insert(e.id, e.info.get("offer_id").and_then(Value::as_i64)); }
            "finish" => {
                if let Some(Some(o)) = e.info.get("start_id").and_then(Value::as_i64).and_then(|s| starts.get(&s)) {
                    completed.insert(*o);
                }
            }
            _ => {}
        }
    }
    type Context = (Option<i64>, u8, i64);
    let (mut counts, mut contexts): (BTreeMap<(Option<i64>, Option<String>, Context), (f64, f64)>, BTreeMap<Context, (f64, f64)>) = Default::default();
    for (id, info, weight) in &offers {
        let length = info.get("length").and_then(Value::as_f64).filter(|&l| l != 0.).unwrap_or(240.);
        let accuracy = info.get("accuracy").and_then(Value::as_f64).filter(|&a| a != 0.).unwrap_or(0.95);
        let keys = info.get("keys").and_then(Value::as_i64);
        let context = (keys, if length < 240. { 0 } else if length < 600. { 1 } else { 2 }, (accuracy * 100. / 3.).floor() as i64);
        let done = if completed.contains(id) { *weight } else { 0. };
        let g = counts.entry((keys, info.get("group").and_then(Value::as_str).map(str::to_string), context)).or_default();
        g.0 += weight;
        g.1 += done;
        let x = contexts.entry(context).or_default();
        x.0 += weight;
        x.1 += done;
    }
    let mut result: HashMap<(i64, String), (f64, f64)> = HashMap::default();
    for ((keys, group, context), (a, b)) in counts {
        let (ca, cb) = contexts[&context];
        let baseline = (cb + 2.) / (ca + 4.);
        if let (Some(k), Some(g)) = (keys, group) {
            let at = result.entry((k, g)).or_default();
            at.0 += a * ((b + 6. * baseline) / (a + 6.) - baseline);
            at.1 += a;
        }
    }
    Ok(result.into_iter().map(|(g, (score, count))| (g, (score / count.max(1.)).clamp(-0.25, 0.25))).collect())
}

/// What one builder request depends on.
#[derive(Clone)]
pub struct Request {
    pub model: Arc<PredModel>,         // the display target
    pub tables_for: Arc<dyn Fn(&[i64]) -> PredSession + Send + Sync>,
    pub session: Arc<Session>,
    pub public: Arc<Public>,
    pub calc: String,
    pub predictor: String,
    pub client: String,
    pub target: f64,
    pub keys: Vec<i64>,
    pub skills_keys: Vec<i64>,
    pub selected_skills: Vec<String>,
    pub tuning: skills::Tuning,
    pub converts: Vec<(i64, Vec<i64>)>, // this tab's enabled (target, source keymodes)
    pub mode: String,                  // nps, skills, or a tab that never waits for this
    pub revision: u64,
    pub generation: u64,
}

/// One publication of a request.
pub struct Published {
    pub mode: String,
    pub candidates: Arc<ChoiceList>,
    pub catalog: Arc<HashMap<String, Arc<Inst>>>,
    pub selected_skills: Vec<String>,
    pub total: usize,
    pub analyzed: usize,
    pub failed: usize,
    pub error: String,
    pub partial: bool,
    pub revision: u64,
    pub generation: u64,
    pub keys_by_mode: (Vec<i64>, Vec<i64>),
}

pub enum Update {
    Ready(Published),
    Failed { error: String, mode: String, revision: u64, generation: u64 },
}

pub struct Builder {
    pub store: Arc<RwLock<FeatStore>>,
    catalog: HashMap<String, Arc<Inst>>,
    order: Vec<String>, // catalogue order: library, then website; equal-valued copies of a chart resolve by it
    catalog_key: Option<(String, Vec<i64>, Option<f64>, Option<std::time::SystemTime>, String, Vec<(i64, Vec<i64>)>)>,
    index_size: usize, // converts indexed before the next publication; grows, so the first show up quickly
    catalog_error: String,
    loaded_rowid: i64,
    web: HashSet<String>,
    window: Option<Vec<(i64, u64, u64)>>,
    installed_md5: HashSet<String>,
    lead: BTreeMap<String, Vec<String>>,
    pub prm: Arc<Params>,
    pub wanted: Arc<dyn Fn() -> bool + Send + Sync>, // a newer request is waiting, or the app is closing
    pub busy: Arc<dyn Fn() -> bool + Send + Sync>,   // the existing gameplay pause policy
    pub game: Arc<dyn Fn() -> bool + Send + Sync>,   // the game is running
    pub emit: Arc<dyn Fn(Update) + Send + Sync>,
}

fn job(sha: &str, inst: &Inst) -> ScoreJob {
    ScoreJob { sha: sha.to_string(), key: inst.md5.clone().filter(|m| !m.is_empty()).unwrap_or_else(|| inst.sha256.clone()), bid: None, md5: inst.md5.clone().unwrap_or_default(),
               keys: inst.keys, installed: inst.installed, playcount: inst.playcount.unwrap_or(0) as f64, convert: inst.convert_of.is_some() }
}

fn publish_job(sha: &str, inst: &Inst, public: &Public) -> ScoreJob {
    ScoreJob { bid: inst.md5.as_ref().and_then(|m| public.md5_bid.get(m).copied()), ..job(sha, inst) }
}

impl Builder {
    pub fn new(store: Arc<RwLock<FeatStore>>, prm: Arc<Params>) -> Builder {
        Builder { store, catalog: HashMap::default(), order: Vec::new(), catalog_key: None, catalog_error: String::new(), loaded_rowid: 0, web: HashSet::new(), window: None, installed_md5: HashSet::new(), index_size: 256,
                  lead: BTreeMap::new(), prm, wanted: Arc::new(|| false), busy: Arc::new(|| false), game: Arc::new(|| false), emit: Arc::new(|_| {}) }
    }

    /// A failed build's catalogue is not reusable.
    pub fn reset(&mut self) {
        self.catalog_key = None;
    }

    /// Worker._restore_local: the last list's references under this session, before the library is read. Saved choices
    /// are only chart/rate references, so they survive a calculator update and predictions are always current.
    pub fn restore(&mut self, db: &Db, r: &Request, refs: &[(String, f64)], installed: &[Arc<Inst>]) -> Result<(Arc<ChoiceList>, Arc<HashMap<String, Arc<Inst>>>)> {
        self.catalog = installed.iter().map(|i| (i.sha256.clone(), i.clone())).collect();
        self.apply_converts(db, &r.converts)?; // a saved list is half converts on 10K, none with a library row
        let mut source = crate::evidence::FeatureSource { public: r.public.clone(), calc: r.calc.clone(), cache: Default::default() };
        let mut store = FeatStore::new(skills::feat_trees(), PROFILE.iter().map(|s| s.to_string()).collect(), r.model.c.clone(), STRUCTURE_VERSION).expect("skill trees");
        let (mut catalog, mut order): (HashMap<String, Arc<Inst>>, Vec<String>) = Default::default();
        for (sha, rate) in refs {
            let Some(inst) = self.catalog.get(sha) else { continue };
            let key = source.key(sha, *rate);
            if let Some(f) = source.lookup(db, &key)? {
                store.put(sha, *rate, &f);
                if catalog.insert(sha.clone(), inst.clone()).is_none() {
                    order.push(sha.clone());
                }
            }
        }
        let allowed: Vec<i64> = if r.mode == "nps" { r.keys.clone() } else { r.skills_keys.clone() };
        let tables = (r.tables_for)(&allowed);
        let selected = skills::normalize(&r.selected_skills);
        let mut memo = Choices::new(order.clone(), Forecast { calc: r.calc.clone(), predictor: r.predictor.clone(), band_z: ms_model::predict::BAND_Z });
        let jobs: Vec<ScoreJob> = order.iter().map(|s| publish_job(s, &catalog[s], &r.public)).collect();
        r.model.score_into(&mut memo, &tables, &store, &jobs, &selected, &r.tuning, r.target, r.mode == "skills", skills::wants_vibro(&selected), !selected.is_empty(), ms_data::store::now())
            .map_err(rusqlite::Error::InvalidParameterName)?;
        *self.store.write().unwrap() = store;
        Ok((Arc::new(memo.snapshot()), Arc::new(catalog)))
    }

    fn wait_idle(&self) -> bool {
        while (self.busy)() {
            if (self.wanted)() {
                return false;
            }
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
        !(self.wanted)()
    }

    /// Website charts join NPS and Skills alike, marked not installed (Next offers the download). Only charts leaving or
    /// entering the window change; the installed library's analyses stay.
    fn swap_web(&mut self, db: &Db, r: &Request, allowed: &[i64], reaches: Option<&HashMap<i64, (f64, f64)>>, web: Option<&Web>) -> Result<()> {
        let mut wanted: HashMap<String, (usize, Vec<f64>)> = HashMap::default();
        if let Some(web) = web {
            for (i, m) in web.maps.iter().enumerate() {
                let chart = web_dir().join("osu").join(format!("{}.osu", m.sha));
                if !allowed.contains(&m.keys) || self.installed_md5.contains(&m.file_md5) || !chart.is_file() {
                    continue;
                }
                if let Some(rates) = web.window_rates(i, reaches.and_then(|x| x.get(&m.keys)).copied()) {
                    wanted.insert(m.sha.clone(), (i, rates));
                }
            }
        }
        let mut store = self.store.write().unwrap();
        for sha in self.web.iter().filter(|s| !wanted.contains_key(*s)).cloned().collect::<Vec<_>>() {
            self.catalog.remove(&sha);
            store.drop_chart(&sha);
        }
        let mut added = Vec::new();
        let mut adding: Vec<(&String, &(usize, Vec<f64>))> = wanted.iter().filter(|x| !self.catalog.contains_key(x.0)).collect();
        adding.sort_by_key(|x| x.1 .0);
        for (sha, (i, rates)) in adding {
            let (web, m) = (web.unwrap(), &web.unwrap().maps[*i]);
            let inst = Inst { sha256: sha.clone(), beatmap_id: Some(m.bid), set_id: Some(m.set_id), md5: Some(m.file_md5.clone()), online_md5: Some(m.file_md5.clone()), status: m.status, keys: m.keys,
                              artist: m.artist.clone(), title: m.title.clone(), version: m.version.clone(), creator: Some(m.creator.clone()), path: Some(web_dir().join("osu").join(format!("{sha}.osu"))),
                              installed: false, playcount: Some(m.playcount), ..Inst::default() };
            let bid = r.public.md5_bid.get(&m.file_md5).copied();
            for &rate in rates {
                if !offerable(rate, &r.client) {
                    continue;
                }
                // Replace only offered web anchors; current SQLite below still wins.
                let public = bid.filter(|&b| public_current(&r.public, b, &m.file_md5, &r.calc))
                    .and_then(|b| r.public.feature(b, rate)).filter(|f| matches!(f, Val::Dict(d) if d.f("keys") as i64 == m.keys));
                if let Some(Val::Dict(f)) = public.or_else(|| web.feature(*i, rate)) {
                    store.put(sha, rate, &f);
                }
            }
            self.catalog.insert(sha.clone(), Arc::new(inst));
            added.push(sha.clone());
        }
        self.order.retain(|s| self.catalog.contains_key(s));
        self.order.extend(added.iter().cloned());
        if self.loaded_rowid != 0 {
            // The SQLite cursor already passed these charts' rows: read just theirs (primary-key range per chart).
            let mut q = db.prepare("SELECT key,data FROM feats WHERE key>=? AND key<?")?;
            for sha in &added {
                let rows: Vec<(String, Vec<u8>)> = q.query_map([format!("{sha}@"), format!("{sha}A")], read_row)?.collect::<Result<_>>()?;
                for (key, data) in rows {
                    if let Some((rate, calc)) = key.split_once('@').and_then(|x| x.1.split_once('|')) {
                        let rate: f64 = rate.parse().unwrap_or(0.);
                        if calc == r.calc && offerable(rate, &r.client) {
                            if let Ok(Val::Dict(f)) = codec::load(&data) {
                                store.put(sha, rate, &f);
                            }
                        }
                    }
                }
            }
        }
        self.web.retain(|s| wanted.contains_key(s));
        self.web.extend(added);
        Ok(())
    }

    /// Indexed converts join the catalogue when enabled. An imported one is a library row by now: it stays a convert, and
    /// leaves with its setting.
    fn apply_converts(&mut self, db: &Db, enabled: &[(i64, Vec<i64>)]) -> Result<()> {
        let mut gone: HashSet<String> = HashSet::new();
        for row in crate::converts::rows(db)? {
            let enabled = enabled.iter().any(|(t, s)| *t == row.keys && s.contains(&row.source_keys));
            let sha = row.sha256.clone();
            match self.catalog.get_mut(&sha) {
                Some(_) if !enabled => {
                    self.catalog.remove(&sha);
                    gone.insert(sha);
                }
                Some(inst) => {
                    let inst = Arc::make_mut(inst);
                    (inst.convert_of, inst.source_keys, inst.source_web) = (row.convert_of, row.source_keys, row.source_web);
                }
                None if enabled => {
                    self.order.push(sha.clone());
                    self.catalog.insert(sha, Arc::new(row));
                }
                None => {}
            }
        }
        if !gone.is_empty() {
            self.order.retain(|s| !gone.contains(s));
        }
        Ok(())
    }

    /// Converts the next charts, analyses them at 1× and adds them to the catalogue; false: nothing left.
    fn index_converts(&mut self, db: &Db, r: &Request, web: Option<&Web>, reaches: Option<&HashMap<i64, (f64, f64)>>) -> Result<bool> {
        if r.converts.is_empty() {
            return Ok(false);
        }
        let (library, _) = library(db, &r.client)?;
        let (none, root) = (HashMap::default(), crate::catalog::local_origin(&r.client).1);
        let mut source = crate::evidence::FeatureSource { public: r.public.clone(), calc: r.calc.clone(), cache: Default::default() };
        let mut rated = |sha: &str| {
            let key = source.key(sha, 1.);
            source.lookup(db, &key).ok().flatten().map(|f| f.f("overall"))
        };
        let jobs = crate::converts::pending(db, &library, web, &self.installed_md5, &r.converts, reaches.unwrap_or(&none), &mut rated, &root, self.index_size)?;
        if jobs.is_empty() {
            return Ok(false);
        }
        self.index_size = (self.index_size * 4).min(4096);
        let cores = std::thread::available_parallelism().map_or(4, |x| x.get());
        let threads = if (self.game)() { (cores / 4).max(1) } else { workers(None) };
        let prm = self.prm.clone();
        let done = ms_calc::background::background_map(&jobs, threads, |j| crate::converts::run(j, &prm));
        let tx = db.unchecked_transaction()?;
        let mut store = self.store.write().unwrap();
        for (job, result) in jobs.iter().zip(&done) {
            crate::converts::store(db, job, result.as_ref().ok())?;
            let Ok(d) = result else { continue };
            for (rate, f) in &d.features {
                tx.execute("INSERT OR REPLACE INTO feats VALUES (?,?)", rusqlite::params![format!("{}@{rate:.3}|{}", d.sha, r.calc), codec::encode(&Val::Dict(f.clone()))])?;
                store.put(&d.sha, *rate, f);
            }
            if !self.catalog.contains_key(&d.sha) {
                self.order.push(d.sha.clone());
                self.catalog.insert(d.sha.clone(), Arc::new(crate::converts::inst(job, d)));
            }
        }
        drop(store);
        tx.commit()?;
        Ok(true)
    }

    /// Features stored since the last request. Rowids are monotone for the append/replace-only feature table.
    fn load_new(&mut self, db: &Db, r: &Request) -> Result<bool> {
        let upper: i64 = db.query_row("SELECT COALESCE(MAX(rowid),0) FROM feats", [], |x| x.get(0))?;
        let pattern = format!("%|{}", r.calc);
        let mut q = db.prepare("SELECT rowid,key,data FROM feats WHERE rowid>? AND rowid<=? AND key LIKE ? ORDER BY rowid LIMIT 256")?;
        while self.loaded_rowid < upper {
            if !self.wait_idle() {
                return Ok(false);
            }
            // Each bounded page is exhausted before decoding: a live cursor would hold up event commits.
            let rows: Vec<(i64, String, Vec<u8>)> = q.query_map(rusqlite::params![self.loaded_rowid, upper, pattern], |x| Ok((x.get(0)?, read_row_at(x, 1)?, blob_at(x, 2)?)))?.collect::<Result<_>>()?;
            if rows.is_empty() {
                self.loaded_rowid = upper; // skip an irrelevant calculator tail
                break;
            }
            let mut store = self.store.write().unwrap();
            for (rowid, key, data) in rows {
                self.loaded_rowid = rowid;
                let Some((sha, rest)) = key.split_once('@') else { continue };
                let Ok(rate) = rest.split('|').next().unwrap_or("").parse::<f64>() else { continue };
                if self.catalog.contains_key(sha) && offerable(rate, &r.client) {
                    if let Ok(Val::Dict(f)) = codec::load(&data) {
                        store.put(sha, rate, &f); // a damaged derived entry is recomputed
                    }
                }
            }
        }
        Ok(true)
    }

    fn snapshot(&self, r: &Request, list: ChoiceList, partial: bool, subset: Option<HashMap<String, Arc<Inst>>>, failed: usize) -> Update {
        let store = self.store.read().unwrap();
        let total = self.catalog.len();
        let analyzed = self.catalog.keys().filter(|s| store.current(s)).count();
        Update::Ready(Published { mode: r.mode.clone(), candidates: Arc::new(list), catalog: Arc::new(subset.unwrap_or_else(|| self.catalog.clone())), selected_skills: r.selected_skills.clone(),
                                  total, analyzed, failed, error: self.catalog_error.clone(), partial, revision: r.revision, generation: r.generation,
                                  keys_by_mode: (r.keys.clone(), r.skills_keys.clone()) })
    }

    /// One request, published as it improves; returns when finished or superseded.
    pub fn build(&mut self, db: &Db, r: &Request) -> Result<()> {
        if r.mode != "nps" && r.mode != "skills" {
            return Ok(()); // PP never waits for an invisible tab
        }
        let allowed: Vec<i64> = if r.mode == "nps" { r.keys.clone() } else { r.skills_keys.clone() };
        let imported = kv_get(db, "realm_import").and_then(|v| v["t"].as_f64());
        let web_path = web_dir().join("web.msp");
        let web_mtime = std::fs::metadata(&web_path).and_then(|m| m.modified()).ok();
        let web = web_mtime.and_then(|_| Web::open(&web_path).ok());
        let reaches = (web_mtime.is_some() || !r.converts.is_empty()).then(|| reach(&r.model, &allowed, r.target, &r.model.c));
        let window = reaches.as_ref().map(|x| {
            let mut v: Vec<(i64, u64, u64)> = x.iter().map(|(k, (a, b))| (*k, a.to_bits(), b.to_bits())).collect();
            v.sort_unstable();
            v
        });
        // The website window follows the accuracy target: a target change swaps only the website charts. Keying the
        // whole catalogue on it re-decoded every analysed chart (user 2026-10-31).
        let catalog_key = Some((r.calc.clone(), allowed.clone(), imported, web_mtime, r.client.clone(), r.converts.clone()));
        if catalog_key != self.catalog_key {
            let (full, error) = library(db, &r.client)?;
            self.catalog_error = error;
            self.installed_md5 = full.iter().filter_map(|i| i.md5.clone()).collect();
            self.order = full.iter().filter(|i| allowed.contains(&i.keys)).map(|i| i.sha256.clone()).collect();
            self.catalog = full.into_iter().filter(|i| allowed.contains(&i.keys)).map(|i| (i.sha256.clone(), Arc::new(i))).collect();
            self.apply_converts(db, &r.converts)?;
            self.index_size = 256;
            self.catalog_key = catalog_key;
            *self.store.write().unwrap() = FeatStore::new(skills::feat_trees(), PROFILE.iter().map(|s| s.to_string()).collect(), r.model.c.clone(), STRUCTURE_VERSION).expect("skill trees");
            self.loaded_rowid = 0;
            self.web.clear();
            self.swap_web(db, r, &allowed, reaches.as_ref(), web.as_ref())?;
            self.window = window;
        } else if window != self.window {
            self.swap_web(db, r, &allowed, reaches.as_ref(), web.as_ref())?;
            self.window = window;
        }
        if !self.load_new(db, r)? {
            return Ok(());
        }
        // Public features of the installed revisions fill what the local cache lacks.
        {
            let mut store = self.store.write().unwrap();
            for (sha, inst) in &self.catalog {
                let Some(md5) = inst.md5.as_ref() else { continue };
                let Some(&bid) = r.public.md5_bid.get(md5) else { continue };
                if !public_current(&r.public, bid, md5, &r.calc) {
                    continue;
                }
                for rate in r.public.rates(bid) {
                    if offerable(rate, &r.client) && !store.has_rate(sha, rate) {
                        if let Some(Val::Dict(f)) = r.public.feature(bid, rate).filter(|f| matches!(f, Val::Dict(d) if d.f("keys") as i64 == inst.keys)) {
                            store.put(sha, rate, &f);
                        }
                    }
                }
            }
        }
        let tables = (r.tables_for)(&allowed);
        let skills_mode = r.mode == "skills";
        let selected = skills::normalize(&r.selected_skills);
        let forecast = Forecast { calc: r.calc.clone(), predictor: r.predictor.clone(), band_z: ms_model::predict::BAND_Z };
        let score = |choices: &mut Choices, shas: &[String], catalog: &HashMap<String, Arc<Inst>>, store: &FeatStore, now: f64| -> std::result::Result<(), String> {
            let jobs: Vec<ScoreJob> = shas.iter().filter_map(|s| catalog.get(s).map(|i| publish_job(s, i, &r.public))).collect();
            r.model.score_into(choices, &tables, store, &jobs, &selected, &r.tuning, r.target, skills_mode, skills::wants_vibro(&selected), !selected.is_empty(), now)
        };
        let now = ms_data::store::now();
        // A play or refit re-predicts the whole library. The previous request's leading charts go first, so the shown
        // list follows the new session within a second.
        let lead: Vec<String> = self.lead.get(&r.mode).map_or_else(Vec::new, |l| l.iter().filter(|s| self.catalog.contains_key(*s)).cloned().collect());
        if !lead.is_empty() {
            let subset: HashMap<String, Arc<Inst>> = lead.iter().map(|s| (s.clone(), self.catalog[s].clone())).collect();
            let mut quick = Choices::new(lead.clone(), forecast.clone());
            let store = self.store.read().unwrap();
            if score(&mut quick, &lead, &subset, &store, now).is_ok() {
                drop(store);
                (self.emit)(self.snapshot(r, quick.snapshot(), true, Some(subset), 0));
            }
        }
        // Deterministic per-chart order spreads unplayed coverage across collections.
        let (mut unknown, mut enrich): (Vec<String>, Vec<String>) = (Vec::new(), Vec::new());
        {
            let store = self.store.read().unwrap();
            for sha in &self.order {
                if !store.contains(sha) {
                    unknown.push(sha.clone());
                } else if !store.current(sha) {
                    enrich.push(sha.clone());
                }
            }
            unknown.sort_by_key(|s| Sha256::digest(s.as_bytes()));
            // Don't analyse only the global density leaders before variety exists: strong representatives of every
            // pattern and mode interleave; all of the remaining library still receives coverage.
            let nearest = |s: &str| store.rates(s).into_iter().min_by(|a, b| (a - 1.).abs().total_cmp(&(b - 1.).abs())).unwrap_or(1.);
            let mut sorted: Vec<(f64, String)> = enrich.drain(..).map(|s| {
                let v = r.model.initial_value(&store, &publish_job(&s, &self.catalog[&s], &r.public), nearest(&s), &tables, r.target);
                (v.unwrap_or(0.), s)
            }).collect();
            sorted.sort_by(|a, b| b.0.total_cmp(&a.0));
            let mut groups: BTreeMap<(i64, &'static str), VecDeque<String>> = BTreeMap::new();
            for (_, sha) in sorted {
                let g = store.group(&sha, nearest(&sha), &r.model).unwrap_or("rice");
                groups.entry((self.catalog[&sha].keys, g)).or_default().push_back(sha);
            }
            while !groups.is_empty() {
                groups.retain(|_, q| {
                    if let Some(s) = q.pop_front() {
                        enrich.push(s);
                    }
                    !q.is_empty()
                });
            }
        }
        let memo_forecast = forecast.clone();
        let mut memo = Choices::new(self.order.clone(), forecast);
        let (mut bad, mut count, mut last_publish) = (HashMap::<String, String>::default(), 0usize, None::<std::time::Instant>);
        let (mut refined, mut refined_counts, mut groups_done): (Vec<(String, f64)>, HashMap<String, u32>, BTreeMap<(i64, String), u32>) = Default::default();
        let mut pending_unknown: VecDeque<String> = unknown.into();
        let mut pending_enrich: VecDeque<String> = enrich.into();
        loop {
            if !self.wait_idle() {
                return Ok(());
            }
            let pass = std::time::Instant::now();
            let mut front = {
                let store = self.store.read().unwrap();
                let unscored = memo.unscored();
                score(&mut memo, &unscored, &self.catalog, &store, ms_data::store::now()).map_err(|e| rusqlite::Error::InvalidParameterName(e))?;
                let blocked: HashSet<String> = refined_counts.iter().filter(|x| *x.1 >= 6).map(|x| x.0.clone()).chain(bad.keys().cloned()).collect();
                memo.proposals(&refined, &blocked)
            };
            let publish_now = last_publish.is_none_or(|t| t.elapsed().as_secs_f64() > 10.) || count == 0;
            let snapshot = memo.snapshot();
            if publish_now {
                self.lead.insert(r.mode.clone(), snapshot.lead(LEAD));
                (self.emit)(self.snapshot(r, snapshot.clone(), false, None, bad.len()));
                last_publish = Some(std::time::Instant::now());
            }
            let mut jobs: Vec<(String, f64)> = Vec::new();
            {
                let store = self.store.read().unwrap();
                let group_of = |sha: &str, rate: f64| -> (i64, String) {
                    let near = store.rates(sha).into_iter().min_by(|a, b| (a - rate).abs().total_cmp(&(b - rate).abs())).unwrap_or(rate);
                    (self.catalog[sha].keys, store.group(sha, near, &r.model).unwrap_or("rice").to_string())
                };
                front.sort_by(|a, b| {
                    let score = |p: &(f64, String, f64)| p.0 - 0.45 * f64::from(*groups_done.get(&group_of(&p.1, p.2)).unwrap_or(&0)).ln_1p();
                    score(b).total_cmp(&score(a))
                });
                if count < 160 {
                    for (_, sha, wanted) in &front {
                        let rate = snap(*wanted, &r.client);
                        if rate != *wanted {
                            refined.push((sha.clone(), *wanted)); // the frontier stops offering a rate this client lacks
                        }
                        if store.has_rate(sha, rate) {
                            continue;
                        }
                        let seen = refined.iter().any(|x| x.0 == *sha && x.1 == rate);
                        if !seen && !bad.contains_key(sha) && refined_counts.get(sha).copied().unwrap_or(0) < 6 {
                            jobs.push((sha.clone(), rate));
                            refined.push((sha.clone(), rate));
                            *refined_counts.entry(sha.clone()).or_insert(0) += 1;
                            *groups_done.entry(group_of(sha, rate)).or_insert(0) += 1;
                            // 32 a pass: every pass rescans the library; 32 analyse in ~0.1 s.
                            if jobs.len() >= 32 {
                                break;
                            }
                        }
                    }
                }
            }
            let take = |pending_enrich: &mut VecDeque<String>, pending_unknown: &mut VecDeque<String>, bad: &HashMap<String, String>, store: &FeatStore| -> Vec<(String, f64)> {
                let mut out = Vec::new();
                for (queue, n) in [(pending_enrich, 24), (pending_unknown, 40)] {
                    for _ in 0..n.min(queue.len()) {
                        let sha = queue.pop_front().unwrap();
                        if !bad.contains_key(&sha) {
                            let rate = store.rates(&sha).into_iter().min_by(|a, b| (a - 1.).abs().total_cmp(&(b - 1.).abs())).unwrap_or(1.);
                            out.push((sha, rate));
                        }
                    }
                }
                out
            };
            jobs.extend(take(&mut pending_enrich, &mut pending_unknown, &bad, &self.store.read().unwrap()));
            // A useful exact shortlist needs no endless refinement, but unanalysed maps keep going until the library is
            // done: a calculator change left ~10k waiting for two weeks at 160 per update (user 2026-10-25).
            if jobs.is_empty() {
                self.lead.insert(r.mode.clone(), snapshot.lead(LEAD));
                (self.emit)(self.snapshot(r, snapshot, false, None, bad.len()));
                if !self.wait_idle() || !self.index_converts(db, r, web.as_ref(), reaches.as_ref())? {
                    return Ok(());
                }
                memo = Choices::new(self.order.clone(), memo_forecast.clone());
                continue;
            }
            // A pass analyses for at least as long as its bookkeeping took: a fixed ~50 maps a pass left the tail of a
            // re-analysis at ~10/s.
            let (bookkeeping, analysing) = (pass.elapsed(), std::time::Instant::now());
            while !jobs.is_empty() {
                if !self.wait_idle() {
                    return Ok(());
                }
                let root = crate::catalog::local_origin(&r.client).1;
                let data: Vec<(Vec<u8>, Vec<f64>)> = jobs.iter().map(|(sha, rate)| {
                    let inst = &self.catalog[sha];
                    let converted = inst.convert_of.is_some() && !inst.installed;
                    let path = inst.path.clone().or_else(|| ms_data::paths::local_file(sha, &root));
                    (if converted { crate::converts::bytes(inst, &root) } else { path.and_then(|p| std::fs::read(p).ok()) }.unwrap_or_default(), vec![*rate])
                }).collect();
                // Beside a running game a full-core batch costs a laptop its turbo and heat headroom.
                let cores = std::thread::available_parallelism().map_or(4, |x| x.get());
                let threads = if (self.game)() { (cores / 4).max(1) } else { workers(None) };
                let done = features_batch(&data, threads, &self.prm);
                let tx = db.unchecked_transaction()?;
                {
                    let mut store = self.store.write().unwrap();
                    for ((sha, _), result) in jobs.iter().zip(done) {
                        match result {
                            Err(e) => { bad.insert(sha.clone(), e); }
                            Ok(fs) => {
                                for (rate, f) in fs {
                                    tx.execute("INSERT OR REPLACE INTO feats VALUES (?,?)", rusqlite::params![format!("{sha}@{rate:.3}|{}", r.calc), codec::encode(&Val::Dict(f.clone()))])?;
                                    store.put(sha, rate, &f);
                                }
                                memo.drop_chart(sha);
                                count += 1;
                            }
                        }
                    }
                }
                tx.commit()?;
                jobs = if analysing.elapsed() < bookkeeping { take(&mut pending_enrich, &mut pending_unknown, &bad, &self.store.read().unwrap()) } else { Vec::new() };
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
        }
    }
}

/// The public features of a beatmap are of this calculator and this file revision.
fn public_current(public: &Public, bid: i64, md5: &str, calc: &str) -> bool {
    let chart = public.meta.charts.binary_search_by(|c| c.0.cmp(&bid)).ok().map(|x| &public.meta.charts[x]);
    !md5.is_empty() && chart.is_some_and(|c| c.2 == calc && c.1 == md5) && public.map_info(bid).is_some_and(|m| m.md5 == md5)
}

fn blob_at(r: &rusqlite::Row<'_>, i: usize) -> Result<Vec<u8>> {
    Ok(match r.get_ref(i)? {
        rusqlite::types::ValueRef::Blob(b) | rusqlite::types::ValueRef::Text(b) => b.to_vec(),
        _ => Vec::new(),
    })
}

fn read_row_at(r: &rusqlite::Row<'_>, i: usize) -> Result<String> {
    r.get(i)
}

fn read_row(r: &rusqlite::Row<'_>) -> Result<(String, Vec<u8>)> {
    Ok((r.get(0)?, blob_at(r, 1)?))
}

/// What the tab ranks with this session.
pub struct RankInputs<'a> {
    pub keys: Vec<i64>,
    pub mode: &'a str,
    pub web: f64,
    pub convert: f64,
    pub weighted: [bool; 2],
    pub taste: &'a HashMap<(i64, String), f64>,
    pub consumed: &'a HashSet<String>,
    pub recent_sets: &'a HashSet<i64>, // what was just played or offered: a convert of those waits, it's the same song
    pub recent_convert_sets: &'a HashSet<i64>, // the converts among them: their real maps wait the same way
    pub seed: Option<String>,
    pub focus: f64,
    pub count: usize,
    pub recent: Vec<u8>,
    pub skill_weights: Option<std::collections::HashMap<String, f64>>,
    pub selected_skills: &'a [String],
    pub miss_ratio: Option<f64>, // the active client is stable: recdata.shown_acc
}

/// A convert and its source are one song: a convert waits while its set was just played or offered, a real map only
/// after a convert of its set. Real maps otherwise never wait for a set: that's the normal list.
fn same_song_waits(convert: bool, inst: Option<&Inst>, recent_sets: &HashSet<i64>, recent_convert_sets: &HashSet<i64>) -> bool {
    inst.and_then(|i| i.set_id).is_some_and(|s| if convert { recent_sets } else { recent_convert_sets }.contains(&s))
}

/// nps.rank: the first `count` of a publication's prepared pool as list entries, in shown order, and the streams' odds.
pub fn rank(prepared: &ms_model::choices::Prepared, catalog: &HashMap<String, Arc<Inst>>, session: &Session, store: &FeatStore, inputs: RankInputs<'_>) -> (Vec<Pick>, [f64; 3]) {
    let recent: Vec<_> = session.attempts.iter().rev().take(6).rev().filter(|a| a.a.get("profile").is_some_and(|p| !p.is_null()) && a.meaningful).collect();
    let mode = inputs.mode;
    let freshness = |key: &str, length: f64, pool: f64| session.freshness(key, length, Some(mode), Some(pool), ms_model::session::EXPOSURE_S);
    let eventful = |key: &str| session.eventful(key);
    let consumed = |c: &ms_model::choices::Choice| inputs.consumed.contains(&c.identity()) || same_song_waits(c.convert, catalog.get(&c.sha).map(Arc::as_ref), inputs.recent_sets, inputs.recent_convert_sets);
    let (rows, relax, odds) = prepared.rank(Ranking {
        keys: inputs.keys, nps: mode == "nps", web: inputs.web, convert: inputs.convert, weighted: inputs.weighted, taste: inputs.taste, eventful: &eventful, freshness: &freshness,
        recent_families: recent.iter().rev().take(4).rev().filter_map(|a| a.text("family").map(str::to_string)).collect(),
        recent_profiles: recent.iter().filter_map(|a| a.a.get("profile")?.as_array().map(|p| p.iter().filter_map(Value::as_f64).collect())).collect(),
        current_keys: session.current_keys(), consumed: &consumed,
        shuffle: inputs.seed.map(|s| (s, if mode == "nps" { inputs.focus } else { 0. })), count: Some(inputs.count), recent: inputs.recent, skill_weights: inputs.skill_weights,
    });
    let picks = rows.into_iter().filter_map(|(i, value, weight, share)| {
        let mut p = prepared_pick(prepared, i, catalog, mode, inputs.selected_skills, store)?;
        let c = &prepared.list.items[i];
        let headline = if mode == "nps" { format!("{:.1} raw NPS", c.nps) } else { c.practice.clone() };
        let shown = inputs.miss_ratio.map_or(c.acc_mid, |m| 1. - (1. - c.acc_mid) * m);
        p.value = value;
        p.weight = weight;
        p.category_share = share;
        p.why = format!("{headline} · {} · expect ~{:.1}%{}", c.description, 100. * shown, if relax { " · freshness relaxed" } else { "" });
        Some(p)
    }).collect();
    (picks, odds)
}

/// nps.prepared_choice: entry `i` of a prepared pool as a list entry.
pub fn prepared_pick(prepared: &ms_model::choices::Prepared, i: usize, catalog: &HashMap<String, Arc<Inst>>, mode: &str, selected: &[String], store: &FeatStore) -> Option<Pick> {
    let c = &prepared.list.items[i];
    let inst = catalog.get(&c.sha)?;
    Some(Pick {
        mode: mode.to_string(), purpose: mode.to_string(), sha: Some(c.sha.clone()), md5: inst.md5.clone(), bid: inst.beatmap_id.filter(|&b| b > 0), title: inst.name(), artist: Some(inst.artist.clone()),
        song: Some(inst.title.clone()), version: Some(inst.version.clone()), creator: inst.creator.clone(), keys: c.keys, rate: c.rate, var: rate_label(c.rate), installed: inst.installed,
        online_match: inst.beatmap_id.is_some_and(|b| b > 0) && inst.online_md5.is_some() && inst.online_md5 == inst.md5, nps: Some(c.nps), length: c.length, skill: Some(c.description.clone()),
        description: c.description.clone(), family: Some(c.family.clone()), group: Some(c.group.clone()), profile: c.profile.clone(), playcount: inst.playcount, value: c.value,
        practice_skills: if mode == "skills" { skills::normalize(selected) } else { Vec::new() }, matched_skills: c.matched.clone(), practice_description: c.practice.clone(),
        f: store.feats(&c.sha, c.rate).cloned().map(Arc::new), pred: c.pred.clone(), generated_at: c.generated_at, best_rank: 100000, convert_of: inst.convert_of.clone(), source_keys: inst.source_keys,
        source_web: inst.source_web, set_id: inst.set_id, ..Pick::default()
    })
}

/// Keeps MIN_RATE/MAX_RATE reachable for callers that show the rate range.
pub const RATE_RANGE: (f64, f64) = (MIN_RATE, MAX_RATE);

#[cfg(test)]
mod convert_tests {
    use super::*;
    use std::path::Path;

    fn row(db: &Db, source: &str, sha: &str, target: i64, keys: i64) {
        db.execute("INSERT INTO converts VALUES (?,?,?,?,?,'m','A','T','V (7K to 10K)','C',5,NULL,8,90,1,0)", rusqlite::params![source, target, ms_convert::CONVERTER_ID, keys, sha]).unwrap();
    }

    fn builder() -> Builder {
        let c = Arc::new(ms_model::predict::Consts::standard());
        Builder::new(Arc::new(RwLock::new(FeatStore::new(skills::feat_trees(), PROFILE.iter().map(|s| s.to_string()).collect(), c, STRUCTURE_VERSION).unwrap())), Arc::new(ms_calc::features::standard_params()))
    }

    #[test]
    fn an_imported_convert_is_one_row_and_stays_a_convert() {
        let dir = std::env::temp_dir().join(format!("ms-convert-test-{}", std::process::id()));
        let db = ms_data::store::connect(&dir.join("rec.db")).unwrap();
        row(&db, "src", "out", 10, 7);
        let mut b = builder();
        // the realm import gave it a library row
        b.catalog.insert("out".into(), Arc::new(Inst { sha256: "out".into(), keys: 10, installed: true, ..Inst::default() }));
        b.order.push("out".into());
        b.apply_converts(&db, &[(10, vec![7])]).unwrap();
        assert_eq!((b.catalog.len(), b.order.len()), (1, 1));
        assert!(b.catalog["out"].installed && b.catalog["out"].convert_of.as_deref() == Some("src"));
        // not imported: the row itself joins
        row(&db, "src2", "out2", 10, 7);
        b.apply_converts(&db, &[(10, vec![7])]).unwrap();
        assert!(!b.catalog["out2"].installed && b.catalog.len() == 2);
        // turned off: gone, imported or not
        b.apply_converts(&db, &[]).unwrap();
        assert!(b.catalog.is_empty() && b.order.is_empty(), "{:?}", b.order);
        let _ = std::fs::remove_dir_all(Path::new(&dir));
    }

    #[test]
    fn a_convert_and_its_source_wait_for_each_other() {
        let inst = Inst { set_id: Some(5), ..Inst::default() };
        let recent: HashSet<i64> = [5].into_iter().collect();
        let none = HashSet::new();
        assert!(same_song_waits(true, Some(&inst), &recent, &none));
        assert!(!same_song_waits(false, Some(&inst), &recent, &none));
        assert!(same_song_waits(false, Some(&inst), &recent, &recent));
        assert!(!same_song_waits(true, Some(&inst), &none, &none));
    }
}
