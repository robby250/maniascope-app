//! osu!stable support: library and scores from stable's databases, the same chart identity as lazer, HT/NM/DT only
//!. One test: it sets the process's data folders.
use ms_data::imports::{add_live_score, import_library};
use ms_data::paths::{local_file, stable_store};
use ms_data::replays::archive_stable;
use ms_data::store::{acc_lazer, acc_stable, connect, log_event};
use ms_rec::builder::offerable;
use ms_rec::catalog::Pick;
use ms_rec::history::{history_text, Played, Shown};
use ms_rec::navigation::song_search;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::Path;

const OSU: &[u8] = b"osu file format v14\n\n[General]\nMode: 3\n\n[Difficulty]\nCircleSize:7\nOverallDifficulty:8\n";

fn s(v: &str) -> Vec<u8> {
    let mut out = vec![0x0b, v.len() as u8];
    out.extend_from_slice(v.as_bytes());
    out
}

struct Bytes(Vec<u8>);

impl Bytes {
    fn add(&mut self, b: &[u8]) -> &mut Bytes {
        self.0.extend_from_slice(b);
        self
    }
    fn i32(&mut self, v: i32) -> &mut Bytes {
        self.add(&v.to_le_bytes())
    }
    fn i16(&mut self, v: i16) -> &mut Bytes {
        self.add(&v.to_le_bytes())
    }
    fn i64(&mut self, v: i64) -> &mut Bytes {
        self.add(&v.to_le_bytes())
    }
    fn f32(&mut self, v: f32) -> &mut Bytes {
        self.add(&v.to_le_bytes())
    }
    fn f64(&mut self, v: f64) -> &mut Bytes {
        self.add(&v.to_le_bytes())
    }
    fn text(&mut self, v: &str) -> &mut Bytes {
        self.add(&s(v))
    }
}

struct Listed<'a> {
    artist: &'a str,
    title: &'a str,
    version: &'a str,
    md5: &'a str,
    file: &'a str,
    folder: &'a str,
    bid: i32,
    mode: u8,
}

/// Minimal osu!.db (version 20250107: star ratings are Int-Float pairs).
fn osu_db(player: &str, charts: &[Listed]) -> Vec<u8> {
    let mut b = Bytes(Vec::new());
    b.i32(20250107).i32(1).add(&[1]).i64(0).text(player).i32(charts.len() as i32);
    for c in charts {
        for x in [c.artist, "", c.title, "", "mapper", c.version, "audio.mp3", c.md5, c.file] {
            b.text(x);
        }
        b.add(&[4]).i16(1).i16(0).i16(0).i64(0).f32(9.).f32(7.).f32(8.).f32(8.).f64(1.4);
        b.i32(1).add(&[8]).i32(0).add(&[0x0c]).f32(4.); // osu! stars
        b.i32(0).i32(0).i32(0);
        b.i32(90).i32(90000).i32(0).i32(0); // drain, total, preview, points
        b.i32(c.bid).i32(77).i32(0).add(&[0; 4]).i16(0).f32(0.7).add(&[c.mode]);
        b.text("").text("").i16(0).text("").add(&[1]).i64(0).add(&[0]);
        b.text(c.folder).i64(0).add(&[0; 5]).i32(0).add(&[20]);
    }
    b.i32(0);
    b.0
}

fn replay_head(b: &mut Bytes, md5: &str, player: &str, stats: [i16; 6], score: i32, mods: i32) {
    let [geki, c300, katu, c100, c50, miss] = stats;
    b.add(&[3]).i32(20250107).text(md5).text(player).text("r");
    b.i16(c300).i16(c100).i16(c50).i16(geki).i16(katu).i16(miss).i32(score).i16(500).add(&[0]).i32(mods);
    b.text("");
}

fn scores_db(md5: &str, player: &str, rows: &[([i16; 6], i32, i32, i64)]) -> Vec<u8> {
    let mut b = Bytes(Vec::new());
    b.i32(20250107).i32(1).text(md5).i32(rows.len() as i32);
    for &(stats, score, mods, unix) in rows {
        replay_head(&mut b, md5, player, stats, score, mods);
        b.i64(621_355_968_000_000_000 + unix * 10_000_000).i32(-1).i64(0);
    }
    b.0
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn live(sha: &str, score: i64) -> Value {
    json!({"sha256": sha, "client": "stable", "hits": {"geki": 900, "300": 80, "katu": 10, "100": 5, "50": 2, "0": 3}, "score": score, "played": "2026-10-31T01:00:00", "mods_list": [{"acronym": "DT"}]})
}

fn replay_file(path: &Path, head: &[u8]) {
    std::fs::write(path, head).unwrap();
}

#[test]
fn stable_library_scores_replays_and_units() {
    let tmp = std::env::temp_dir().join(format!("ms-stable-{}", std::process::id()));
    let root = tmp.join("osu!");
    std::fs::create_dir_all(root.join("Songs/123 set")).unwrap();
    std::fs::create_dir_all(root.join("Data/r")).unwrap();
    std::fs::write(root.join("Songs/123 set/a.osu"), OSU).unwrap();
    let std_chart = String::from_utf8_lossy(OSU).replace("Mode: 3", "Mode: 0");
    std::fs::write(root.join("Songs/123 set/b.osu"), std_chart).unwrap();
    let (sha, md5) = (hex(&Sha256::digest(OSU)), ms_rec::engine::hashes(OSU).1);
    let zero = "0".repeat(32);
    std::fs::write(root.join("osu!.db"), osu_db("me", &[
        Listed { artist: "Art", title: "Song", version: "Hard", md5: &md5, file: "a.osu", folder: "123 set", bid: 55, mode: 3 },
        Listed { artist: "Art", title: "Std", version: "Hard", md5: &zero, file: "b.osu", folder: "123 set", bid: 56, mode: 0 }])).unwrap();
    let stats = [900, 80, 10, 5, 2, 3];
    std::fs::write(root.join("scores.db"), scores_db(&md5, "me", &[(stats, 812345, 64, 1_700_000_000), (stats, 700000, 0, 1_700_000_100)])).unwrap();
    std::env::set_var("HOME", tmp.join("home"));
    std::env::set_var("XDG_DATA_HOME", tmp.join("home/.local/share"));
    std::env::set_var("MANIASCOPE_DATA", tmp.join("data"));
    std::env::set_var("STABLE_DATA", &root);
    let db = connect(&tmp.join("rec.db")).unwrap();

    // library and scores import with lazer identity, mania only
    let r = import_library(&db, "stable").unwrap();
    assert_eq!((r.user_id, r.seen, r.new, r.installed), (None, 2, 2, 1));
    let row: (String, i64, String, i64, i64) = db.query_row("SELECT sha256,beatmap_id,md5,status,keys FROM installed", [], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?))).unwrap();
    assert_eq!(row, (sha.clone(), 55, md5.clone(), 1, 7));
    assert_eq!(std::fs::read(local_file(&sha, &stable_store()).unwrap()).unwrap(), OSU, "every sha → file lookup reaches stable's chart");
    let mut rows: Vec<(i64, f64, String, i64)> = db.prepare("SELECT total,rate,client,beatmap_id FROM scores").unwrap().query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?))).unwrap().map(Result::unwrap).collect();
    rows.sort_by_key(|r| r.0);
    assert_eq!(rows, [(700000, 1., "stable".to_string(), 55), (812345, 1.5, "stable".to_string(), 55)]);
    assert_eq!(import_library(&db, "stable").unwrap().new, 0, "idempotent, files not re-hashed");
    // tosu's results screen of the same play lands on the same row, whatever its timestamp
    assert!(!add_live_score(&db, &live(&sha, 812345), false, None).unwrap().1);
    assert_eq!(db.query_row("SELECT COUNT(*) FROM scores", [], |r| r.get::<_, i64>(0)).unwrap(), 2);

    // rates, search and shown accuracy
    assert!(offerable(1.5, "stable") && !offerable(1.1, "stable") && offerable(1.1, "lazer"));
    let pick = Pick { artist: Some("Art".into()), song: Some("Song \"X\"".into()), version: Some("Hard".into()), keys: 7, ..Pick::default() };
    assert_eq!(song_search(&pick, "stable"), "Art Song X Hard keys=7");
    assert!(song_search(&pick, "lazer").starts_with("artist=\"Art\""));
    let st = [900, 80, 10, 5, 2, 3];
    assert!(acc_stable(&st) > acc_lazer(&st));
    let shown = Shown { client: "stable".into(), miss_ratio: 0.8 };
    assert!((shown.model(shown.acc(0.95)) - 0.95).abs() < 1e-12 && shown.acc(0.95) > 0.95 && shown.points(1.) < 1.);
    let played = Played { key: "k".into(), played: Some("2026-10-31T01:00:00".into()), title: "T".into(), rate: Some(1.5), keys: Some(7), actual: acc_lazer(&st), expected: Some(0.95), z: Some(0.5),
                          prediction: Value::Null, sha256: None, stats: st, pauses: None, recovered: true, ..Default::default() };
    let text = history_text(&[played], &shown);
    assert!(text.contains(&format!("{:.2}%", 100. * acc_stable(&st))) && text.contains(&format!("expected {:.2}%", 100. * shown.acc(0.95))), "{text}");
    assert!(text.contains("from osu!stable"), "{text}");
    assert_eq!(Shown { client: "lazer".into(), miss_ratio: 0.8 }.acc(0.95), 0.95);

    // a tracked stable play archives its replay and tosu's pauses
    let head = |score: i32| {
        let mut b = Bytes(Vec::new());
        replay_head(&mut b, &md5, "me", stats, score, 64);
        b.i64(0);
        b.0
    };
    let mut own = head(812345);
    own.extend_from_slice(&0i32.to_le_bytes());
    replay_file(&root.join(format!("Data/r/{md5}-638000000000000000.osr")), &own);
    replay_file(&root.join(format!("Data/r/{md5}-638000000000000001.osr")), &head(812344)); // another play
    let (key, _) = add_live_score(&db, &live(&sha, 812345), true, None).unwrap();
    let key = key.unwrap();
    log_event(&db, "finish", None, None, &json!({"key": key, "pauses": [12000, 30500]})).unwrap();
    add_live_score(&db, &live(&sha, 700000), false, None).unwrap();
    assert_eq!(archive_stable(&db, Some(&root)).unwrap(), 1);
    let (score_key, status, pauses, rate, path): (String, String, String, f64, String) = db.query_row("SELECT score_key,status,pauses,rate,archive_path FROM replay_evidence", [], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?))).unwrap();
    assert_eq!((score_key.as_str(), status.as_str(), rate), (key.as_str(), "archived", 1.5));
    assert_eq!(serde_json::from_str::<Value>(&pauses).unwrap(), json!([12000, 30500]));
    assert_eq!(std::fs::read(tmp.join(&path)).unwrap(), own);
    assert_eq!(db.query_row("SELECT COUNT(*) FROM replay_evidence", [], |r| r.get::<_, i64>(0)).unwrap(), 1, "only observed attempts");
    assert_eq!(archive_stable(&db, Some(&root)).unwrap(), 0, "idempotent");
    let _ = std::fs::remove_dir_all(tmp);
}
