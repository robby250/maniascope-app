//! A small synthetic player and population for tests: the real fit, no mocks. Fake data folders, nothing real.
use ms_calc::features::{chart_features, standard_params};
use ms_calc::val::Val;
use ms_data::codec;
use ms_data::public::{self, MapInfo, Meta, Population, Public};
use ms_data::store::{connect, upsert_score, Db, Score, Stats};
use ms_model::predict::Consts;
use ms_rec::engine::{hashes, refit, Engine};
use ms_rec::pp::PpSettings;
use std::path::PathBuf;
use std::sync::Arc;

pub struct World {
    pub dir: PathBuf,
    pub db: Db,
    pub public: Arc<Public>,
    pub charts: Vec<(String, String, f64)>, // sha, md5, overall
}

impl Drop for World {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

pub fn osu(keys: usize, gap: f64, seed: usize) -> Vec<u8> {
    let mut text = format!("osu file format v14\n\n[General]\nMode: 3\n\n[Metadata]\nTitle:song {seed}\nArtist:a\nCreator:c\nVersion:v\n\n[Difficulty]\nCircleSize:{keys}\nOverallDifficulty:8\n\n[TimingPoints]\n0,500,4,1,0,100,1,0\n\n[HitObjects]\n");
    for i in 0..2400 {
        let x = (((i + seed * (i / 8)) % keys) * 512 + 256) / keys;
        text.push_str(&format!("{x},192,{},1,0,0:0:0:0:\n", 1000 + (i as f64 * gap) as i64));
    }
    text.into_bytes()
}

pub fn population() -> Population {
    Population { slope: vec![(7, 2.)], window: vec![(7, 1.5)], curve: vec![(7, 0.)], curve_knee: 4., sig2: 0.1, tau2: 0.05, kappa: 5., rows: 1000, players: 50, ..Population::default() }
}

/// 1000 judgements whose lazer accuracy is `acc`.
pub fn stats(acc: f64) -> Stats {
    let good = (1000. * (1. - acc) * 305. / 105.).round() as i64;
    [1000 - good, 0, good, 0, 0, 0]
}


pub fn iso(t: f64) -> String {
    let days = (t / 86400.).floor() as i64;
    let (mut y, mut d) = (1970, days);
    let leap = |y: i64| (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    while d >= 365 + i64::from(leap(y)) {
        d -= 365 + i64::from(leap(y));
        y += 1;
    }
    let lengths = [31, 28 + i64::from(leap(y)), 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut m = 0;
    while d >= lengths[m] {
        d -= lengths[m];
        m += 1;
    }
    let s = (t - days as f64 * 86400.) as i64;
    format!("{y}-{:02}-{:02}T{:02}:{:02}:{:02}+00:00", m + 1, d + 1, s / 3600, s / 60 % 60, s % 60)
}

pub fn chart_bytes(i: usize) -> Vec<u8> {
    osu(7, 150. - 4. * i as f64, i)
}

pub fn world(tag: &str, acc: impl Fn(f64) -> f64, plays: usize) -> World {
    dated(tag, plays, 30., |o, _| acc(o))
}

/// `plays` over the last `days`; `acc(overall, age in days)`.
pub fn dated(tag: &str, plays: usize, days: f64, acc: impl Fn(f64, f64) -> f64) -> World {
    let dir = std::env::temp_dir().join(format!("ms-world-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let db = connect(&dir.join("rec.db")).unwrap();
    let prm = standard_params();
    let (mut maps, mut records, mut charts, mut analysed) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for i in 0..24usize {
        let data = chart_bytes(i);
        let (sha, md5) = hashes(&data);
        let (rate, f) = chart_features(&data, &[1.], &prm).unwrap().remove(0);
        let bid = 1000 + i as i64;
        charts.push((sha, md5.clone(), f.f("overall")));
        maps.push(MapInfo { bid, keys: 7, od: 8., set: bid, approved: if i == 23 { 3 } else { 1 }, file: String::new(), md5: md5.clone() }); // the last one qualified
        analysed.push((bid, md5, ms_rec::engine::CALC_ID.to_string()));
        records.push((bid, rate, codec::encode(&Val::Dict(f))));
    }
    let meta = Meta { snapshot: "fixture".into(), calc: ms_rec::engine::CALC_ID.into(), pop: population(), pop_lazer: Some(population()), maps, charts: analysed, ..Meta::default() };
    let path = dir.join("public.msp");
    public::write(&path, &meta, &records).unwrap();
    let public = Arc::new(Public::open(&path).unwrap());
    let now = ms_data::store::now();
    let spacing = days * 86400. / plays.max(1) as f64;
    for i in 0..plays {
        let (sha, md5, overall) = &charts[i % charts.len()];
        let at = now - 86400. - (plays - i) as f64 * spacing;
        let s = stats(acc(*overall, (now - at) / 86400.));
        let (accuracy, n) = ms_data::store::acc320(&s);
        upsert_score(&db, Score { key: format!("s{i}"), src: "realm".into(), online_id: 5000 + i as i64, player_id: Some(1), sha256: Some(sha.clone()), md5: Some(md5.clone()),
            beatmap_id: Some(1000 + (i % charts.len()) as i64), status: Some(1), played: Some(iso(at)), imported: "2026-11-01T00:00:00+00:00".into(), client: Some("2026.921.0-lazer".into()),
            mods: "[]".into(), rate: Some(1.), ranked: true, stats: s, acc: accuracy, n, total: 900000, pp: Some(1.), fresh: false, local_id: None }).unwrap();
    }
    World { dir, db, public, charts }
}

pub fn engine(w: &World) -> Engine {
    let c = Arc::new(Consts::standard());
    let r = refit(&w.db, &w.public, &c).unwrap();
    Engine::new(w.public.clone(), c, r, PpSettings::default())
}

/// Fake data folders, once per process: nothing here may reach the real lazer or stable data.
pub fn sandbox() -> PathBuf {
    static DIR: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    DIR.get_or_init(|| {
        let dir = std::env::temp_dir().join(format!("ms-world-{}-data", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // SAFETY: before any worker thread of this test binary reads them, the first call of every test.
        unsafe {
            std::env::set_var("LAZER_DATA", &dir);
            std::env::set_var("STABLE_DATA", dir.join("stable"));
            std::env::set_var("MANIASCOPE_REC", dir.join("rec"));
        }
        dir
    }).clone()
}

/// Puts the charts in the fake lazer store and the installed table.
pub fn install(w: &World) -> Vec<(String, String)> {
    let root = sandbox();
    let mut out = Vec::new();
    for (i, (sha, md5, _)) in w.charts.iter().enumerate() {
        let data = chart_bytes(i);
        let path = ms_data::paths::local_file(sha, &root).unwrap();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, data).unwrap();
        let f = chart_features(&chart_bytes(i), &[1.], &standard_params()).unwrap().remove(0).1;
        w.db.execute("INSERT OR REPLACE INTO feats VALUES (?,?)", rusqlite::params![format!("{sha}@1.000|{}", ms_rec::engine::CALC_ID), codec::encode(&Val::Dict(f))]).unwrap();
        w.db.execute("INSERT OR REPLACE INTO installed VALUES (?,?,?,?,1,7,8,180,?,'a','v')", rusqlite::params![sha, 1000 + i as i64, 1000 + i as i64, md5, format!("song {i}")]).unwrap();
        out.push((sha.clone(), md5.clone()));
    }
    out
}

pub fn sloped(o: f64) -> f64 {
    1. - 0.06 * (o / 2.).powi(2)
}
