//! The Recommender: one fit's models and everything derived from it, the card of a selected chart, the PP pool.
use crate::catalog::{installed_by_md5, local_shas, rate_label_pp, Inst, Pick};
use crate::evidence::{now_months, FeatureSource};
use crate::pp::{self, build_ledger, candidates, estimates, home_keys, newer_ranked, prepare, reference, score, status_of, typical, warmups, Ledger, PpSettings, Reference, Scoring, Variant};
use crate::recommender::{family_transfer, fit, Fitted};
use ms_calc::features::{displayed_feature, standard_params};
use ms_calc::params::Params;
use ms_calc::py::round_digits;
use ms_calc::val::{Dict, Val};
use ms_data::codec;
use ms_data::events::load_events;
use ms_data::public::Public;
use ms_data::store::{connect, now, Db, Result, RANKED_STATUS};
use ms_model::choices::preview_order;
use ms_model::predict::{Consts, Feats, HashMap, PredSession, Prediction, BAND_Z};
use ms_model::session::{Session, HISTORY_S};
use rusqlite::OptionalExtension;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// recommend.predictor_id of the model the Rust port reproduces: forecasts stay comparable across the port. A change to
/// the prediction model takes a new id, so old forecasts are not read as this model's.
pub const PREDICTOR_ID: &str = "dec244ae8a09";
pub const CALC_ID: &str = "a4e17d09c2b6";
pub const CARRIED_CALC: &str = "3c695ee7a23f"; // differs in calibration only (calib/carry.json). Each carry keeps its source rows.
const SURE_PLAYS: usize = 50; // of the player's own, on a keymode: fewer and the model is mostly the population's guess
pub const ACTIVE_KEYS: std::ops::RangeInclusive<i64> = 4..=10;

/// One event log state: the session, its display view, the frozen tables of both.
pub struct Now {
    pub session: Arc<Session>,
    pub view: Arc<Session>,
    pub tables: PredSession,
    pub view_tables: PredSession,
    pub target: f64,
}

pub struct Refit {
    pub fitted: Arc<Fitted>,
    pub installed: Arc<HashMap<String, Arc<Inst>>>,
    pub local_shas: Arc<HashSet<String>>,
    pub ledger: Ledger,
    pub missing: usize,
    pub ranks: HashMap<i64, i64>,
    pub references: HashMap<i64, Reference>,
}

pub fn refit(db: &Db, public: &Arc<Public>, c: &Arc<Consts>) -> Result<Refit> {
    let installed = Arc::new(installed_by_md5(db)?);
    let (ledger, missing) = build_ledger(db, public, &installed)?;
    let (ranks, references) = reference(db, &ledger)?;
    Ok(Refit { fitted: Arc::new(fit(db, public, CALC_ID, c)?), local_shas: Arc::new(local_shas(db)?), installed, ledger, missing, ranks, references })
}

pub struct Engine {
    pub public: Arc<Public>,
    pub c: Arc<Consts>,
    pub prm: Arc<Params>,
    pub r: Refit,
    pub source: FeatureSource,
    pub settings: PpSettings,
    pub observing: Arc<Mutex<Option<Vec<String>>>>, // results that land while a refit runs
    variants: Option<(Vec<i64>, Arc<Vec<Variant>>)>,
    transfer: Arc<Vec<Vec<f64>>>,
}

/// Both targets of one prediction: `pp` (acc320) and what the player reads.
pub struct Expected {
    pub pp: Prediction,
    pub shown: Prediction,
    pub direct: bool, // `shown` comes from a directly fitted display target
}

impl Expected {
    /// recommend.Predictor.predict's forecast: the display state, chart-bound when `chart` is a hash.
    pub fn forecast(&self, chart: &str, rate: f64) -> Option<Value> {
        self.direct.then(|| {
            let p = &self.shown;
            let (sha, md5) = (chart.len() == 64, chart.len() == 32);
            json!({"schema": 1, "target": "native_displayed_lazer", "kind": "usual_range", "generated_at": now(), "calculator": CALC_ID, "predictor": PREDICTOR_ID,
                   "sha256": if sha { Some(chart) } else { None }, "md5": if md5 { Some(chart) } else { None }, "rate": rate, "band_z": BAND_Z,
                   "mid": p.acc_mid, "lo": p.acc_lo, "hi": p.acc_hi, "mu": p.mu, "sdm": p.sdm, "sda": p.sda, "sd": p.sd})
        })
    }

    /// The expectation's fields of a start event.
    pub fn fields(&self) -> Map<String, Value> {
        let (p, s) = (&self.pp, &self.shown);
        let mut out = Map::new();
        for (k, v) in [("mu", p.mu), ("base_mu", p.base_mu), ("cold_penalty", p.cold), ("activation_minutes", s.activation), ("opening_acc", s.opening_acc), ("rate_slope", s.slope), ("sd", p.sd), ("sdm", p.sdm)] {
            out.insert(k.to_string(), json!(v));
        }
        if self.direct {
            for (k, v) in [("mu", s.mu), ("base_mu", s.base_mu), ("cold_penalty", s.cold), ("sdm", s.sdm), ("sda", s.sda), ("sd", s.sd)] {
                out.insert(format!("display_{k}"), json!(v));
            }
        }
        out
    }
}

/// accuracy_targets.forecast_chart: an exact chart identity, without changing prediction provenance.
pub fn forecast_chart(state: Option<Value>, sha: Option<&str>, md5: Option<&str>) -> Option<Value> {
    let mut state = state?;
    for (k, v) in [("sha256", sha), ("md5", md5)] {
        match (state[k].as_str(), v) {
            (Some(a), Some(b)) if a != b => return None,
            (None, Some(b)) => state[k] = json!(b),
            _ => {}
        }
    }
    Some(state)
}

#[derive(Default)]
pub struct Card {
    pub bid: Option<i64>,
    pub md5: String,
    pub keys: i64,
    pub title: String,
    pub var: String,
    pub rate: f64,
    pub ranked: bool,
    pub artist: Option<String>,
    pub song: Option<String>,
    pub version: Option<String>,
    pub acc_mid: f64,
    pub acc_lo: f64,
    pub acc_hi: f64,
    pub sure: bool, // enough of the player's own plays on this keymode to show acc_mid unasked
    pub form: (Option<&'static str>, f64),
    pub length: f64,
    pub skill: Option<String>,
    pub nps: Option<f64>,
    pub pp: Option<(f64, f64, Option<f64>, f64, f64)>, // pp_lo, pp_hi, best, gain, p_up
    pub plays_like: Option<(f64, i64)>,
}

/// recommend.top_skill
pub fn top_skill(f: &Dict) -> Option<String> {
    let sk = f.dict("sk")?;
    let mut best: Option<(&str, f64)> = None;
    for (k, v) in &sk.0 {
        if let ms_calc::val::Key::S(k) = k {
            if k != "stamina" && best.is_none_or(|b| v.f() > b.1) {
                best = Some((k, v.f()));
            }
        }
    }
    best.map(|b| b.0.to_string())
}

/// Hex of a digest.
pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn hashes(data: &[u8]) -> (String, String) {
    (hex(&Sha256::digest(data)), format!("{:x}", md5::compute(data)))
}

impl Engine {
    pub fn new(public: Arc<Public>, c: Arc<Consts>, r: Refit, settings: PpSettings) -> Engine {
        let source = FeatureSource { public: public.clone(), calc: CALC_ID.to_string(), cache: HashMap::default() };
        Engine { transfer: Arc::new(family_transfer(&c)), public, c, prm: Arc::new(standard_params()), r, source, settings, observing: Arc::default(), variants: None }
    }

    /// A finished refit replaces the models; results that landed meanwhile are observed again.
    pub fn apply(&mut self, db: &Db, r: Refit, observed: Vec<String>) -> Result<()> {
        self.r = r;
        self.variants = None; // ~77k variants: built on the first PP-tab request, not on every refit
        for key in observed {
            self.observe(db, &key)?;
        }
        Ok(())
    }

    pub fn model(&self) -> &ms_model::predict::PredModel {
        self.r.fitted.display.as_ref().map_or(&self.r.fitted.pp.model, |d| &d.model)
    }

    pub fn now_state(&self, db: &Db, target: f64) -> Result<Now> {
        self.state_at(db, target, now())
    }

    pub fn state_at(&self, db: &Db, target: f64, current: f64) -> Result<Now> {
        let events = load_events(db, ((current - HISTORY_S) / 60.).floor() * 60.)?;
        let session = Session::new(events, current, Some(self.r.fitted.pp.warmup.clone()), self.c.clone(), self.transfer.clone());
        let view = self.r.fitted.display.as_ref().map_or_else(|| session.display_view(self.r.fitted.pp.warmup.clone()), |d| session.display_view(d.warmup.clone()));
        let all: Vec<i64> = ACTIVE_KEYS.collect();
        let (tables, view_tables) = (session.tables(&all), view.tables(&all));
        Ok(Now { session: Arc::new(session), view: Arc::new(view), tables, view_tables, target })
    }

    /// recommend.Predictor.predict: PP means and spreads stay; playlist quality and intervals use the display target.
    pub fn predict(&self, f: &Feats, chart: &str, bid: Option<i64>, rate: f64, n: &Now) -> Option<Expected> {
        let pp = self.r.fitted.pp.model.predict(f, chart, bid, rate, &n.tables).ok()?;
        let Some(d) = self.r.fitted.display.as_ref() else {
            return Some(Expected { shown: pp.clone(), pp, direct: false });
        };
        let shown = d.model.predict(f, chart, bid, rate, &n.view_tables).ok()?;
        Some(Expected { pp, shown, direct: true })
    }

    /// Features of a chart-rate: the local table, then the public one when the md5 matches.
    pub fn features(&mut self, db: &Db, sha: Option<&str>, md5: Option<&str>, bid: Option<i64>, rate: f64) -> Result<(Option<Arc<Dict>>, Option<i64>)> {
        self.source.of(db, sha, md5, bid, Some(rate))
    }

    /// FeatureSource._store
    pub fn store(&mut self, db: &Db, sha: &str, features: &[(f64, Dict)]) -> Result<()> {
        let tx = db.unchecked_transaction()?;
        for (rate, f) in features {
            let key = self.source.key(sha, *rate);
            tx.execute("INSERT OR REPLACE INTO feats VALUES (?, ?)", rusqlite::params![key, codec::encode(&Val::Dict(f.clone()))])?;
            self.source.cache.insert(key, Arc::new(f.clone()));
        }
        tx.commit()
    }

    /// A result updates the ledger and session, not the frozen baseline model: rebuilding it on every play blocked Next.
    /// Session correction already consumes this result; refits remain for startup, imports and old scores.
    pub fn observe(&mut self, db: &Db, key: &str) -> Result<()> {
        if let Some(list) = self.observing.lock().unwrap().as_mut() {
            list.push(key.to_string());
        }
        type Row = (Option<i64>, Option<i64>, Option<String>, String, Option<i64>, Option<String>, Option<f64>, Option<String>, Option<String>, Option<String>, Option<String>, Option<i64>);
        let row: Option<Row> = db.query_row("SELECT ranked,beatmap_id,md5,src,online_id,client,pp,sha256,stats,mods,played,status FROM scores WHERE key=?", [key],
                                            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?, r.get(6)?, r.get(7)?, r.get(8)?, r.get(9)?, r.get(10)?, r.get(11)?))).optional()?;
        let Some((ranked, bid, md5, src, online, client, stored, sha, stats, mods, played, status)) = row else { return Ok(()) };
        let Some(bid) = bid.filter(|&b| b > 0 && ranked.unwrap_or(0) != 0) else { return Ok(()) };
        let counted = status_of(status, md5.as_deref(), Some(bid), &self.public, &self.r.installed).is_some_and(|s| RANKED_STATUS.contains(&s));
        if !counted || !(src != "realm" || online.unwrap_or(0) != 0 || client.as_deref() == Some("lazer-live")) {
            return Ok(());
        }
        let mods: Value = mods.as_deref().and_then(|m| serde_json::from_str(m).ok()).unwrap_or(json!([]));
        let lazer = !client.as_deref().unwrap_or("").starts_with("stable");
        let pp = stored.or_else(|| Some(pp::score_pp(&pp::scored_chart(sha.as_deref(), Some(bid), md5.as_deref())?, &ms_data::store::parse_stats(stats.as_deref()?)?, &mods, lazer)?));
        let Some(pp) = pp else { return Ok(()) };
        db.execute("UPDATE scores SET pp=? WHERE key=?", rusqlite::params![pp, key])?;
        if pp > self.r.ledger.best_of(bid) {
            let mut best = self.r.ledger.best.clone();
            best.insert(bid, (pp, true, played));
            self.r.ledger = Ledger::new(best);
            (self.r.ranks, self.r.references) = reference(db, &self.r.ledger)?;
        }
        Ok(())
    }

    fn variants(&mut self, db: &Db, keys: &[i64]) -> Result<Arc<Vec<Variant>>> {
        if let Some((k, v)) = &self.variants {
            if k == keys {
                return Ok(v.clone());
            }
        }
        let newer = newer_ranked(db, &mut self.source, &self.r.installed, keys, &self.c)?;
        let all = Arc::new(candidates(&self.public, &self.r.installed, keys, &self.c, newer));
        self.variants = Some((keys.to_vec(), all.clone()));
        Ok(all)
    }

    /// All PP candidates under this session, the keymodes' best first.
    pub fn scored(&mut self, db: &Db, keys: &[i64], n: &Now) -> Result<Vec<Pick>> {
        let variants = self.variants(db, keys)?;
        let month = now_months();
        let typ = typical(&self.r.fitted.rows, month);
        let threads = std::thread::available_parallelism().map_or(1, |x| x.get()).saturating_sub(2).max(1);
        let f = &self.r.fitted;
        let s = Scoring { pp_model: &f.pp.model, pp_tables: &n.tables, display: f.display.as_ref().map(|d| (&*d.model, &n.view_tables)), shown: f.shown, session: &n.session, ledger: &self.r.ledger,
                          ranks: &self.r.ranks, references: &self.r.references, settings: &self.settings, typical: &typ, installed: &self.r.installed, public: &self.public, now: n.session.now };
        Ok(score(&variants, &s, threads))
    }

    /// recommend.pool with choose=False: (phase, the first n_show of the eligible maps, note).
    pub fn pool(&mut self, db: &Db, keys: &[i64], n: &Now, n_show: usize) -> Result<(&'static str, Vec<Pick>, String)> {
        let list = self.scored(db, keys, n)?;
        let phase = n.session.phase();
        let month = now_months();
        let typ = typical(&self.r.fitted.rows, month);
        let f = self.r.fitted.clone();
        let home = home_keys(&self.r.ledger, &self.public, &f.rows, month);
        let (mut rows, relaxed) = prepare(&list, &n.session, &self.r.ledger, &self.r.references, &self.settings, home);
        if rows.is_empty() && phase == "warmup" {
            let k0 = n.session.current_keys().filter(|k| keys.contains(k)).or(Some(home).filter(|k| keys.contains(k))).or(keys.first().copied());
            if let Some(k0) = k0 {
                let by_chart = |chart: &str| self.r.installed.get(chart).cloned();
                rows = warmups(&f, &list, &n.session, k0, &n.tables, &n.view_tables, &by_chart, &typ);
            }
        }
        let seed = n.session.all.iter().filter(|e| matches!(e.kind.as_str(), "offer" | "finish" | "abort" | "reset")).map(|e| e.id).max().unwrap_or(0);
        let ids: Vec<(String, f64)> = rows.iter().map(|c| (c.md5.clone().or_else(|| c.sha.clone()).or_else(|| c.bid.map(|b| b.to_string())).unwrap_or_else(|| c.title.clone()), c.weight)).collect();
        let order = preview_order(&ids, &seed.to_string(), 0.);
        let shown: Vec<Pick> = order.into_iter().take(n_show).map(|i| rows[i].clone()).collect();
        let top = rows.iter().filter(|c| c.best_rank <= 100).count();
        let mut note = if rows.is_empty() { "No likely PP gains at this readiness — try Skills or an easier map".to_string() } else { format!("{} eligible · {top} top-100", commas(rows.len())) };
        if relaxed {
            note += " · all cycled, repeats allowed";
        }
        Ok((phase, shown, note))
    }

    /// Pre-play expectation for a chart about to be played (frozen into the start event). None: no features yet.
    pub fn expectation(&mut self, db: &Db, sha: &str, md5: &str, rate: f64, n: &Now) -> Result<Option<(Expected, Arc<Dict>, Feats, Option<i64>)>> {
        let rate = round_digits(rate, 3);
        let (f, b) = self.features(db, Some(sha), Some(md5), None, rate)?;
        let Some(f) = f else { return Ok(None) };
        let Ok(feats) = Feats::of(&f, &self.c) else { return Ok(None) };
        Ok(self.predict(&feats, md5, b, rate, n).map(|e| (e, f, feats, b)))
    }

    fn own_plays(&self, keys: i64) -> usize {
        self.r.fitted.rows.rows.iter().filter(|r| r.keys == keys && r.src != "public" && r.completed).count()
    }

    /// Stats for the chart the game has selected, at its rate: the same numbers a list entry shows.
    pub fn card(&mut self, db: &Db, path: &Path, rate: f64, n: &Now) -> Result<Option<Card>> {
        let Ok(data) = std::fs::read(path) else { return Ok(None) };
        let (sha, md5) = hashes(&data);
        let Some((e, f, feats, b)) = self.expectation(db, &sha, &md5, rate, n)? else { return Ok(None) };
        let inst = self.r.installed.get(&md5).cloned();
        let bid = b.or(inst.as_ref().and_then(|i| i.beatmap_id).filter(|&b| b != 0));
        let approved = b.and_then(|b| self.public.map_info(b)).is_some_and(|m| RANKED_STATUS.contains(&m.approved));
        let stars = f.get("stars").map(Val::f).or_else(|| b.and_then(|b| self.public.feature(b, round_digits(rate, 3)))
            .and_then(|v| if let Val::Dict(d) = v { d.get("stars").map(Val::f) } else { None }));
        let hits = f.get("hits").map(Val::f).unwrap_or_else(|| f.fget("notes", 0.));
        let ranked = (approved || inst.as_ref().is_some_and(|i| i.status.is_some_and(|s| RANKED_STATUS.contains(&s)))) && [0.75, 1., 1.5].contains(&round_digits(rate, 2)) && stars.is_some();
        let meta = if inst.is_none() { ms_data::lazer_index::metadata(path) } else { None };
        let get = |k: &str| meta.as_ref().and_then(|m| m.get(k).cloned());
        let (artist, song, version) = inst.as_ref().map_or_else(|| (get("Artist"), get("Title"), get("Version")), |i| (Some(i.artist.clone()), Some(i.title.clone()), Some(i.version.clone())));
        let title = if inst.is_some() || meta.is_some() {
            format!("{} - {} [{}]", artist.as_deref().unwrap_or(""), song.as_deref().unwrap_or(""), version.as_deref().unwrap_or(""))
        } else {
            path.file_name().map_or_else(String::new, |n| n.to_string_lossy().into_owned())
        };
        let pp = ranked.then(|| estimates(&e.pp, stars.unwrap(), hits, bid, &self.r.ledger));
        let plays_like = self.plays_like(&f, b, rate);
        let nps = f.dict("nps").map(|d| d.f("nps"));
        Ok(Some(Card { bid, md5, keys: feats.keys, title, var: rate_label_pp(rate), rate, ranked, artist, song, version, acc_mid: e.shown.acc_mid, acc_lo: e.shown.acc_lo, acc_hi: e.shown.acc_hi, sure: self.own_plays(feats.keys) >= SURE_PLAYS,
                       form: n.session.form_state(feats.keys), length: f.f("length") / 1000. / rate, skill: top_skill(&f), nps, pp, plays_like }))
    }

    /// (stars, scores) the public scores say a chart plays like at this rate, its stars moved by its map effect; None
    /// with few scores or a small shift (user 2026-10-01: Aim Burst 12.5★ has 98 % top scores; the same players score
    /// 92 % on a 12.5★ Camellia map).
    pub fn plays_like(&self, f: &Dict, b: Option<i64>, rate: f64) -> Option<(f64, i64)> {
        let pop = self.r.fitted.display.as_ref().map_or(&self.r.fitted.pp.model.pop, |d| &d.model.pop);
        let (ratio, n) = plays_like_ratio(pop, f.f("keys") as i64, f.f("overall"), b?, rate)?;
        // The card shows display stars; scale the rating before the display conversion, not after it.
        let mut shown = f.clone();
        shown.set("overall", f.f("overall") * ratio);
        if f.has("preunit_overall") {
            shown.set("preunit_overall", f.f("preunit_overall") * ratio);
        }
        Some((displayed_feature(&shown, &self.prm), n))
    }

    /// Selected-chart analysis: a panic in one chart loses that chart, not the app.
    pub fn analyse(path: &Path, rate: f64, prm: &Params) -> std::result::Result<Vec<(f64, Dict)>, String> {
        let data = std::fs::read(path).map_err(|e| e.to_string())?;
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ms_calc::features::selected_features(&data, rate, prm))).map_err(|_| "the analysis crashed on this chart".to_string())?
    }
}

/// "12,345"
pub fn commas(n: usize) -> String {
    let text = n.to_string();
    let bytes = text.as_bytes();
    bytes.iter().enumerate().flat_map(|(i, b)| {
        let sep = i > 0 && (bytes.len() - i) % 3 == 0;
        sep.then_some(',').into_iter().chain(std::iter::once(*b as char))
    }).collect()
}

/// A connection of its own for a background job.
pub fn second_connection(db: &Db) -> Result<Db> {
    connect(Path::new(db.path().unwrap_or("")))
}

/// How much harder (> 1) or easier a chart plays than its stars by the public map effect, with its score count. None
/// with few scores or a small shift.
pub fn plays_like_ratio(pop: &ms_model::predict::Pop, k: i64, overall: f64, b: i64, rate: f64) -> Option<(f64, i64)> {
    const MIN_SCORES: f64 = 30.;
    const MIN_SHIFT: f64 = 0.05;
    let lr_rate = round_digits(rate, 3).ln();
    let (mb, n) = pop.mbr.get(&b).and_then(|v| v.iter().find(|x| (x.0 - lr_rate).abs() < 1e-9)).map(|x| (x.1, x.2)).or_else(|| pop.mb.get(&b).copied())?;
    if n < MIN_SCORES {
        return None;
    }
    // Invert the whole link (slope·x + curve·(x − knee)²), not the slope alone: above the knee the link is ~3× steeper
    // at 9★, and slope-only turned a displayed 9.2★ (mb −.17) into "plays like 7.4★" where the full link gives 8.9★
    // (user 2026-10-02).
    let (s, c, knee) = (pop.slope(k), pop.curve(k), pop.knee.ln());
    let g = |x: f64| s * x + c * (x - knee).max(0.).powi(2);
    let lr = overall.max(0.05).ln();
    let target = g(lr) + mb; // more misses = harder
    let x = if target <= g(knee) || c <= 0. { target / s } else { knee + (-s + (s * s + 4. * c * (target - s * knee)).sqrt()) / (2. * c) };
    let ratio = (x - lr).exp();
    ((ratio - 1.).abs() >= MIN_SHIFT).then_some((ratio, n as i64))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ms_model::predict::{ByKeys, Pop};

    fn pop() -> Pop {
        let mut p = Pop { slope: ByKeys { map: [(0, 2.), (4, 2.)].into_iter().collect() }, knee: 4., ..Pop::default() };
        p.mb = [(1, (-0.5, 300.)), (2, (0.02, 300.)), (3, (-0.5, 5.))].into_iter().collect();
        p.mbr = [(1, vec![(1.5f64.ln(), 0.4, 80.)])].into_iter().collect();
        p
    }

    #[test]
    fn plays_like_moves_stars_by_the_public_map_effect() {
        let p = pop();
        let (ratio, n) = plays_like_ratio(&p, 4, 10., 1, 1.).unwrap();
        assert!(n == 300 && (ratio - (-0.25f64).exp()).abs() < 1e-9, "fewer misses than its stars: easier");
        assert!(plays_like_ratio(&p, 4, 10., 1, 1.5).unwrap().0 > 1., "its own 1.5× evidence wins");
        assert!(plays_like_ratio(&p, 4, 10., 2, 1.).is_none() && plays_like_ratio(&p, 4, 10., 3, 1.).is_none());
        // Above the knee the link bends: the shift inverts slope and curve together (Chinmoku 1.5×: slope-only said
        // 7.4★ for a 9.2★ chart the full link puts at 8.9★).
        let mut curved = pop();
        curved.curve = ByKeys { map: [(0, 1.6), (4, 1.6)].into_iter().collect() };
        let (ratio, _) = plays_like_ratio(&curved, 4, 10., 1, 1.).unwrap();
        let g = |x: f64| 2. * x + 1.6 * (x - 4f64.ln()).max(0.).powi(2);
        assert!((g((10. * ratio).ln()) - (g(10f64.ln()) - 0.5)).abs() < 1e-9 && ratio > (-0.25f64).exp());
    }
}

/// Stored ratings of the past calculators moved to this one, without their charts. Rows already there stay.
pub fn carry_rows(db: &ms_data::store::Db) -> std::result::Result<(), String> {
    ms_data::store::carry_feats(db, CARRIED_CALC, CALC_ID, &ms_calc::features::params_of(CARRIED_CALC)?, &ms_calc::features::standard_params()).map(|_| ()).map_err(|e| e.to_string())
}
