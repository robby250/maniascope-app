//! osu!stable: 32-bit .NET, no field offset tables; byte patterns in its JIT code lead to its statics.
use crate::process::{Pattern, Process};
use crate::Seen;
use serde_json::{json, Value};
use std::path::Path;
use std::time::{Duration, Instant};

/// tosu's patterns with its offsets applied to the hit. Names as in tosu's StableMemory.
const PATTERNS: [(&str, i64); 9] = [
    ("F8 01 74 04 83 65", 0),                               // baseAddr
    ("5E 5F 5D C3 A1 ?? ?? ?? ?? 89 ?? 04", 0),             // playTimeAddr
    ("7D 15 A1 ?? ?? ?? ?? 85 C0", 0),                      // rulesetsAddr
    ("48 83 F8 04 73 1E", -0x4),                            // statusPtr
    ("C8 FF ?? ?? ?? ?? ?? 81 0D ?? ?? ?? ?? ?? 08 00 00", 0x9), // menuModsPtr
    ("55 8B EC 83 EC 08 A1 ?? ?? ?? ?? 85 C0", 0x7),        // getAudioLengthPtr
    ("FF 15 ?? ?? ?? ?? A1 ?? ?? ?? ?? 8B 48 54 33 D2", 0x7), // userProfilePtr
    ("55 8B EC 80 3D ?? ?? ?? ?? 00 75 26 80 3D", 0),       // canRunSlowlyAddr
    ("83 E0 20 85 C0 7E 2F", 0),                            // settingsClassAddr
];
const STATES: [&str; 14] = ["menu", "edit", "play", "exit", "selectEdit", "selectPlay", "selectDrawings", "resultScreen", "update", "busy", "unknown", "lobby", "matchSetup", "selectMulti"];
const SCORE_V2: u32 = 1 << 29;

const MODS: [(u32, &str); 31] = [(1, "NF"), (2, "EZ"), (4, "TD"), (8, "HD"), (16, "HR"), (32, "SD"), (64, "DT"), (128, "RX"), (256, "HT"), (512, "NC"), (1024, "FL"), (2048, "AT"),
    (4096, "SO"), (8192, "AP"), (16384, "PF"), (32768, "4K"), (65536, "5K"), (131072, "6K"), (262144, "7K"), (524288, "8K"), (1 << 20, "FI"), (1 << 21, "RD"), (1 << 22, "CN"), (1 << 23, "TP"),
    (1 << 24, "9K"), (1 << 25, "CP"), (1 << 26, "1K"), (1 << 27, "3K"), (1 << 28, "2K"), (SCORE_V2, "V2"), (1 << 30, "MR")];

/// tosu's list form; DT and SD stay implied by NC and PF.
fn mods(bits: u32) -> Value {
    let on: Vec<&str> = MODS.iter().filter(|m| bits & m.0 != 0).map(|m| m.1).filter(|m| !(*m == "DT" && bits & 512 != 0 || *m == "SD" && bits & 16384 != 0)).collect();
    json!({"array": on.iter().map(|m| json!({"acronym": m})).collect::<Vec<_>>(), "name": on.concat(), "number": bits})
}

/// stable's own accuracy, in percent; ScoreV2 weighs a rainbow 305.
fn accuracy(h: &[i64; 6], v2: bool) -> f64 {
    let [geki, great, katu, ok, meh, miss] = *h;
    let total = geki + great + katu + ok + meh + miss;
    if total == 0 {
        return 0.;
    }
    let perfect = if v2 { 61 } else { 60 };
    100. * (perfect * geki + 60 * great + 40 * katu + 20 * ok + 10 * meh) as f64 / (total * 60) as f64
}

pub struct Stable {
    p: Process,
    game: String,
    found: [Option<u64>; 9],
    scanned: Option<Instant>,
    wait: Duration,
    file: (String, String, i64),
    length: f64,
}

impl Stable {
    pub fn new(p: Process, exe: &Path) -> Stable {
        let game = if cfg!(windows) { exe.parent().map_or(String::new(), |d| d.to_string_lossy().into_owned()) } else { String::new() };
        Stable { p, game, found: [None; 9], scanned: None, wait: crate::SCAN_WAIT, file: Default::default(), length: 0. }
    }

    pub fn alive(&self) -> bool {
        self.p.alive()
    }

    fn scan(&mut self) -> Option<[u64; 9]> {
        if self.found.iter().any(Option::is_none) && self.scanned.is_none_or(|t| t.elapsed() > self.wait) {
            self.wait = if self.scanned.is_some() { (self.wait * 2).min(crate::SCAN_WAIT_MAX) } else { self.wait }; // JIT code appears as the game reaches each screen
            self.scanned = Some(Instant::now());
            let patterns: Vec<Pattern> = PATTERNS.iter().map(|p| Pattern::new(p.0)).collect();
            let mut hits = [None; 9];
            self.p.scan(&patterns, &mut hits);
            for (i, hit) in hits.iter().enumerate() {
                self.found[i] = self.found[i].or(hit.map(|h| (h as i64 + PATTERNS[i].1) as u64));
            }
        }
        let mut all = [0; 9];
        for (slot, hit) in all.iter_mut().zip(&self.found) {
            *slot = (*hit)?;
        }
        Some(all)
    }

    fn i32(&self, address: u64) -> Option<u64> {
        self.p.ptr(address)
    }

    fn hits(&self, score: u64) -> Option<(Value, [i64; 6])> {
        let at = |o: u64| self.p.i16(score + o).map(i64::from);
        let (ok, great, meh, geki, katu, miss) = (at(0x88)?, at(0x8a)?, at(0x8c)?, at(0x8e)?, at(0x90)?, at(0x92)?);
        Some((json!({"300": great, "geki": geki, "100": ok, "katu": katu, "50": meh, "0": miss}), [geki, great, katu, ok, meh, miss]))
    }

    fn score_mods(&self, score: u64) -> Option<u32> {
        let m = self.i32(score + 0x1c)?;
        Some(self.p.i32(m + 0xc)? as u32 ^ self.p.i32(m + 0x8)? as u32)
    }

    fn results(&self, ruleset: u64) -> Option<Value> {
        let base = self.i32(ruleset + 0x38)?;
        if base == 0 {
            return None;
        }
        let bits = self.score_mods(base)?;
        let (hits, counts) = self.hits(base)?;
        let accuracy = accuracy(&counts, bits & SCORE_V2 != 0);
        let mode = self.p.i32(base + 0x64)?;
        Some(json!({"playerName": self.p.string(self.i32(base + 0x28)?)?, "mode": {"number": mode, "name": if mode == 3 { "mania" } else { "" }}, "score": self.p.i32(base + 0x78)?,
                    "accuracy": accuracy, "hits": hits, "mods": mods(bits), "maxCombo": self.p.i16(base + 0x68)?, "rank": crate::grade(accuracy, bits & (8 | 1024) != 0),
                    "createdAt": crate::net_date(self.p.i32(base + 0xa4)?, self.p.i32(base + 0xa0)?)?}))
    }

    pub fn frame(&mut self) -> Option<Seen> {
        let [base, play_time, rulesets, status, menu_mods, audio, profile, slow, settings] = self.scan()?;
        let state = self.p.deref(status)? as usize;
        let time = self.p.i32(self.i32(play_time + 5)?)? as f64;
        let beatmap = self.p.deref(base - 0xc)?;
        if beatmap != 0 {
            let name = self.p.string(self.i32(beatmap + 0x90)?)?;
            if name.ends_with(".osu") { // mid-switch it reads anything else
                self.file = (self.p.string(self.i32(beatmap + 0x78)?)?, name, self.p.i32(beatmap + 0xc8)? as i64);
            }
        }
        let ruleset = self.i32(self.i32(rulesets - 0xb)? + 4)?;
        let mut seen = Seen { state: STATES.get(state).copied().unwrap_or("menu"), time, replay: self.p.u8(self.i32(slow + 0x46)?)? == 1, ..Seen::default() };
        let mut bits = self.p.deref(menu_mods)? as u32;
        if seen.state == "play" && ruleset != 0 {
            let gameplay = self.i32(ruleset + 0x64)?;
            let score = if gameplay != 0 { self.i32(gameplay + 0x38)? } else { 0 };
            if score != 0 {
                bits = self.score_mods(score)?;
                seen.hits = self.hits(score)?.0;
            }
            self.length = self.p.f64(self.p.deref(audio)? + 4)?.round();
        }
        if seen.state == "resultScreen" && ruleset != 0 {
            seen.results = self.results(ruleset).unwrap_or(Value::Null);
        }
        let set = self.i32(settings + 8)?;
        let folder = self.p.string(self.i32(self.i32(set + 0xb8)? + 4)?)?;
        seen.songs = Path::new(&self.game).join(folder).to_string_lossy().into_owned();
        seen.game = self.game.clone();
        (seen.folder, seen.file, seen.id) = self.file.clone();
        seen.length = self.length;
        seen.mods = mods(bits);
        seen.profile = self.p.string(self.i32(self.p.deref(profile)? + 0x30)?)?;
        Some(seen)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mods_read_as_tosu_lists_them() {
        assert_eq!(mods(64 | 8)["name"], "HDDT");
        assert_eq!(mods(64 | 512)["name"], "NC");
        assert_eq!(mods(256)["array"][0]["acronym"], "HT");
    }

    #[test]
    fn stable_accuracy_counts_a_rainbow_as_a_300_without_score_v2() {
        assert_eq!(accuracy(&[1, 1, 0, 0, 0, 0], false), 100.);
        assert!(accuracy(&[1, 1, 0, 0, 0, 0], true) > 100.);
        assert_eq!(accuracy(&[0, 1, 0, 0, 0, 1], false), 50.);
    }
}
