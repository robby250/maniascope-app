//! osu!lazer: 64-bit .NET. The game object is found from a byte pattern, then everything is a field offset. Those move with
//! every game release (13 of tosu's 184 between August and October 2026), so they are computed from the game's DLLs at attach.
use crate::layout::Layouts;
use crate::process::{Pattern, Process};
use crate::Seen;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const ANCHOR: &str = "01 01 00 00 00 00 80 44 00 00 40 44"; // OsuGame's ScalingContainerTargetDrawSize
const ASSEMBLIES: [&str; 3] = ["osu!", "osu.Game", "osu.Framework"];
const GAME_CLASS: &str = "osu.Desktop.OsuGameDesktop";
const CLASSES: [&str; 28] = [GAME_CLASS, "osu.Framework.Audio.Track.Track", "osu.Framework.Platform.Storage", "osu.Framework.Screens.ScreenStack", "osu.Framework.Timing.DecouplingFramedClock",
    "osu.Framework.Timing.FramedClock", "osu.Game.Beatmaps.BeatmapInfo", "osu.Game.Beatmaps.FramedBeatmapClock", "osu.Game.Beatmaps.WorkingBeatmapCache+BeatmapManagerWorkingBeatmap",
    "osu.Game.IO.WrappedStorage", "osu.Game.Models.RealmUser", "osu.Game.Online.API.APIAccess", "osu.Game.Online.API.LocalUserState", "osu.Game.Online.API.Requests.Responses.APIUser",
    "osu.Game.Online.Chat.ExternalLinkOpener", "osu.Game.OsuGame", "osu.Game.OsuGameBase", "osu.Game.Rulesets.RulesetInfo", "osu.Game.Scoring.ScoreInfo", "osu.Game.Screens.Edit.Editor",
    "osu.Game.Screens.OsuScreen", "osu.Game.Screens.Play.Player", "osu.Game.Screens.Play.PlayerLoader", "osu.Game.Screens.Play.SubmittingPlayer", "osu.Game.Screens.Ranking.SoloResultsScreen",
    "osu.Game.Screens.Select.SoloSongSelect", "osu.Game.Screens.SelectV2.SoloSongSelect", "osu.Game.Screens.Select.SongSelect"];

/// mania's mods in the game's category order. The game's lists hold them by type, so position gives the acronym.
const CATEGORIES: [&[&str]; 6] = [
    &["EZ", "NF", "HT", "DC", "NR"],
    &["HR", "SD", "PF", "DT", "NC", "FI", "HD", "CO", "FL", "AC"],
    &["RD", "DS", "MR", "DA", "CL", "IN", "CS", "HO", "1K", "2K", "3K", "4K", "5K", "6K", "7K", "8K", "9K", "10K"],
    &["AT", "CN"],
    &["WU", "WD", "MU", "AS"],
    &["TD", "SV2"],
];

pub struct Lazer {
    p: Process,
    o: HashMap<&'static str, HashMap<String, u32>>,
    size: i64, // the game object's size, the upper half of the first word of its method table
    base: u64,
    scanned: Option<Instant>,
    wait: Duration,
    mods: HashMap<u64, &'static str>,
    menu_mods: Value,
}

/// The program's folder. /proc/<pid>/root: flatpak's program path is its own.
fn dlls(pid: u32, exe: &Path) -> Option<PathBuf> {
    let dir = exe.parent()?;
    Some(if cfg!(unix) { PathBuf::from(format!("/proc/{pid}/root")).join(dir.strip_prefix("/").ok()?) } else { dir.to_path_buf() })
}

impl Lazer {
    pub fn new(p: Process, exe: &Path) -> Option<Lazer> {
        let mut layouts = Layouts::new(&dlls(p.pid, exe)?);
        let size = layouts.class(&ASSEMBLIES, GAME_CLASS)?.base_size() as i64;
        let o = CLASSES.iter().filter_map(|c| Some((*c, layouts.class(&ASSEMBLIES, c)?.fields))).collect();
        Some(Lazer { p, o, size, base: 0, scanned: None, wait: crate::SCAN_WAIT, mods: HashMap::new(), menu_mods: json!({"array": [], "name": ""}) })
    }

    pub fn alive(&self) -> bool {
        self.p.alive()
    }

    fn off(&self, class: &str, field: &str) -> Option<u64> {
        self.o.get(class)?.get(field).map(|v| *v as u64)
    }

    fn at(&self, address: u64, class: &str, field: &str) -> Option<u64> {
        self.p.ptr(address + self.off(class, field)?)
    }

    fn is_game(&self, address: u64) -> bool {
        self.p.ptr(address).filter(|v| *v != 0).and_then(|v| self.p.i64(v)).is_some_and(|word| word >> 32 == self.size)
    }

    /// The field's distance from the anchor shifts between game builds; the vtable check picks the right one.
    fn from_anchor(&self, anchor: u64) -> Option<u64> {
        [0x24, 0x28, 0x2c, 0x20, 0x30, 0x1c, 0x34].iter().find_map(|delta| {
            let opener = self.p.ptr(anchor - delta)?;
            let api = self.at(opener, "osu.Game.Online.Chat.ExternalLinkOpener", "<api>k__BackingField")?;
            self.at(api, "osu.Game.Online.API.APIAccess", "game").filter(|g| self.is_game(*g))
        })
    }

    fn game(&mut self) -> Option<u64> {
        if !self.is_game(self.base) && self.scanned.is_none_or(|t| t.elapsed() > self.wait) {
            self.wait = if self.scanned.is_some() { (self.wait * 2).min(crate::SCAN_WAIT_MAX) } else { self.wait };
            self.scanned = Some(Instant::now());
            self.base = self.p.scan_all(&Pattern::new(ANCHOR)).into_iter().find_map(|a| self.from_anchor(a)).unwrap_or(0);
        }
        (self.base != 0 && self.is_game(self.base)).then_some(self.base)
    }

    fn items(&self, list: u64) -> Option<Vec<u64>> {
        // tosu's check: an array keeps its method table where a List keeps its count
        let array = self.p.i32(list + 0x10)? > 10_000_000;
        let (size, items) = if array { (self.p.i32(list + 8)?, list) } else { (self.p.i32(list + 0x10)?, self.p.ptr(list + 8)?) };
        (0..size.clamp(0, 256) as u64).map(|i| self.p.ptr(items + 0x10 + 8 * i)).collect()
    }

    fn mod_types(&self, list: u64) -> Option<Vec<u64>> {
        let mut types = Vec::new();
        for item in self.items(list)? {
            let t = self.p.ptr(item)?;
            if self.p.i32(t)? == 0x1000000 && self.p.i32(t + 3)? == 8193 { // MultiMod: its members count by position
                let inner = self.p.ptr(item + 0x10)?;
                for i in 0..self.p.i32(inner + 8)?.clamp(0, 64) as u64 {
                    types.push(self.p.ptr(self.p.ptr(inner + 0x10 + 8 * i)?)?);
                }
            } else {
                types.push(t);
            }
        }
        Some(types)
    }

    fn map_mods(&mut self, game: u64) -> Option<()> {
        let dict = self.p.ptr(self.at(game, "osu.Desktop.OsuGameDesktop", "AvailableMods")? + 0x20)?;
        let (entries, count) = (self.p.ptr(dict + 0x10)?, self.p.i32(dict + 0x38)?);
        for (i, names) in CATEGORIES.iter().enumerate().take(count.max(0) as usize) {
            let types = self.mod_types(self.p.ptr(entries + 0x10 + 0x18 * i as u64)?)?;
            self.mods.extend(types.into_iter().zip(names.iter().copied()));
        }
        Some(())
    }

    fn selected_mods(&mut self, game: u64) -> Option<()> {
        if self.mods.is_empty() {
            self.map_mods(game)?;
        }
        let bindable = self.at(game, "osu.Desktop.OsuGameDesktop", "SelectedMods")?;
        if self.p.u8(bindable + 0x50)? == 1 {
            return Some(()); // disabled in gameplay: the list keeps what the score's mods say
        }
        let mut list = Vec::new();
        for item in self.items(self.p.ptr(bindable + 0x20)?)? {
            let Some(name) = self.mods.get(&self.p.ptr(item)?).copied() else { continue };
            let mut entry = json!({"acronym": name});
            if matches!(name, "DT" | "NC" | "HT" | "DC") {
                entry["settings"] = json!({"speed_change": self.p.f64(self.p.ptr(item + 0x10)? + 0x40)?});
            }
            list.push(entry);
        }
        let name: String = list.iter().filter_map(|m| m["acronym"].as_str()).collect();
        self.menu_mods = json!({"array": list, "name": name});
        Some(())
    }

    fn score_mods(&self, info: u64) -> Option<Value> {
        let text = self.p.string(self.at(info, "osu.Game.Scoring.ScoreInfo", "<ModsJson>k__BackingField")?)?;
        let list: Vec<Value> = serde_json::from_str(&text).unwrap_or_default();
        let name: String = list.iter().filter_map(|m| m["acronym"].as_str()).collect();
        Some(json!({"array": list, "name": name}))
    }

    fn hits(&self, info: u64) -> Option<Value> {
        let mut hits = json!({"300": 0, "geki": 0, "100": 0, "katu": 0, "50": 0, "0": 0});
        let dict = self.at(info, "osu.Game.Scoring.ScoreInfo", "statistics")?;
        if dict == 0 {
            return Some(hits);
        }
        let (entries, count) = (self.p.ptr(dict + 0x10)?, self.p.i32(dict + 0x38)?);
        for i in 0..count.clamp(0, 64) as u64 {
            let item = entries + 0x10 + 0x10 * i;
            let name = match self.p.i32(item + 8)? {
                1 => "0",
                2 => "50",
                3 => "100",
                4 => "katu",
                5 => "300",
                6 => "geki",
                _ => continue,
            };
            hits[name] = json!(self.p.i32(item + 0xc)?);
        }
        Some(hits)
    }

    fn results(&self, screen: u64) -> Option<Value> {
        let info = self.p.ptr(self.at(screen, "osu.Game.Screens.Ranking.SoloResultsScreen", "SelectedScore")? + 0x20)?;
        let (class, date) = ("osu.Game.Scoring.ScoreInfo", self.off("osu.Game.Scoring.ScoreInfo", "<Date>k__BackingField")?);
        let date = crate::net_date(self.p.i32(info + date + 0xc)?, self.p.i32(info + date + 8)?)?;
        let user = self.at(info, class, "<RealmUser>k__BackingField")?;
        let ruleset = self.at(info, class, "<Ruleset>k__BackingField")?;
        let mode = self.p.i32(ruleset + self.off("osu.Game.Rulesets.RulesetInfo", "<OnlineID>k__BackingField")?)?;
        let accuracy = self.p.f64(info + self.off(class, "<Accuracy>k__BackingField")?)? * 100.;
        let mods = self.score_mods(info)?;
        let silver = mods["array"].as_array().is_some_and(|a| a.iter().any(|m| matches!(m["acronym"].as_str(), Some("HD" | "FL"))));
        Some(json!({"playerName": self.p.string(self.at(user, "osu.Game.Models.RealmUser", "<Username>k__BackingField")?)?, "mode": {"number": mode, "name": if mode == 3 { "mania" } else { "" }},
                    "score": self.p.i64(info + self.off(class, "<TotalScore>k__BackingField")?)?, "accuracy": accuracy, "hits": self.hits(info)?, "mods": mods,
                    "maxCombo": self.p.i32(info + self.off(class, "<MaxCombo>k__BackingField")?)?, "rank": crate::grade(accuracy, silver), "createdAt": date}))
    }

    fn user(&self, game: u64) -> Option<String> {
        let api = self.at(game, "osu.Game.OsuGameBase", "<API>k__BackingField")?;
        let state = self.at(api, "osu.Game.Online.API.APIAccess", "localUserState")?;
        let user = self.p.ptr(self.at(state, "osu.Game.Online.API.LocalUserState", "localUser")? + 0x20)?;
        self.p.string(self.at(user, "osu.Game.Online.API.Requests.Responses.APIUser", "<Username>k__BackingField")?)
    }

    fn same(&self, screen: u64, class: &str, field: &str, game: u64, base_class: &str, base_field: &str) -> bool {
        self.at(screen, class, field).is_some_and(|a| Some(a) == self.at(game, base_class, base_field))
    }

    fn state(&self, game: u64, screen: u64) -> (&'static str, bool) {
        const GAME: &str = "osu.Game.OsuGameBase";
        let api = |class: &str| self.same(screen, class, "<api>k__BackingField", game, GAME, "<API>k__BackingField");
        let logo = self.same(screen, "osu.Game.Screens.OsuScreen", "<logo>k__BackingField", game, "osu.Game.OsuGame", "osuLogo");
        let player = api("osu.Game.Screens.Play.SubmittingPlayer") && self.same(screen, "osu.Game.Screens.Play.SubmittingPlayer", "<spectatorClient>k__BackingField", game, GAME, "<SpectatorClient>k__BackingField")
            || api("osu.Game.Screens.Play.Player") && self.same(screen, "osu.Game.Screens.Play.Player", "<scoreManager>k__BackingField", game, GAME, "<ScoreManager>k__BackingField");
        let loader = logo && self.same(screen, "osu.Game.Screens.Play.PlayerLoader", "<leaderboardManager>k__BackingField", game, GAME, "<LeaderboardManager>k__BackingField");
        if player || loader {
            return ("play", player);
        }
        if api("osu.Game.Screens.Ranking.SoloResultsScreen") && logo {
            return ("resultScreen", false);
        }
        let select = ["osu.Game.Screens.Select.SoloSongSelect", "osu.Game.Screens.SelectV2.SoloSongSelect", "osu.Game.Screens.Select.SongSelect"].into_iter().find(|c| self.o.contains_key(c));
        if select.is_some_and(|c| self.at(screen, c, "<game>k__BackingField") == Some(game)) {
            return ("selectPlay", false);
        }
        if api("osu.Game.Screens.Edit.Editor") && self.same(screen, "osu.Game.Screens.Edit.Editor", "<realm>k__BackingField", game, GAME, "realm") {
            return ("edit", false);
        }
        ("menu", false)
    }

    pub fn frame(&mut self) -> Option<Seen> {
        const GAME: &str = "osu.Game.OsuGameBase";
        const WORKING: &str = "osu.Game.Beatmaps.WorkingBeatmapCache+BeatmapManagerWorkingBeatmap";
        let game = self.game()?;
        let stack_field = if self.off("osu.Game.OsuGame", "ScreenStack").is_some() { "ScreenStack" } else { "<ScreenStack>k__BackingField" };
        let stack = self.at(self.at(game, "osu.Game.OsuGame", stack_field)?, "osu.Framework.Screens.ScreenStack", "stack")?;
        let count = self.p.i32(stack + 0x10)? as i64;
        let screen = self.p.ptr(self.p.ptr(stack + 8)? + (0x10 + 8 * (count - 1)) as u64)?;
        let (state, in_player) = self.state(game, screen);
        self.selected_mods(game);
        let working = self.p.ptr(self.at(game, GAME, "<Beatmap>k__BackingField")? + 0x20)?;
        let info = self.at(working, WORKING, "BeatmapInfo")?;
        let hash = self.p.string(self.at(info, "osu.Game.Beatmaps.BeatmapInfo", "<Hash>k__BackingField")?)?;
        let clock = self.at(game, GAME, "beatmapClock")?;
        let source = self.at(clock, "osu.Game.Beatmaps.FramedBeatmapClock", "finalClockSource")?;
        let time = self.p.f64(source + self.off("osu.Framework.Timing.FramedClock", "<CurrentTime>k__BackingField")?)?.round();
        let storage = self.at(self.at(game, GAME, "<Storage>k__BackingField")?, "osu.Game.IO.WrappedStorage", "<UnderlyingStorage>k__BackingField")?;
        let songs = Path::new(&self.p.string(self.at(storage, "osu.Framework.Platform.Storage", "<BasePath>k__BackingField")?)?).join("files");
        let mut seen = Seen { state, time, mods: self.menu_mods.clone(), songs: songs.to_string_lossy().into_owned(), id: self.p.i32(info + self.off("osu.Game.Beatmaps.BeatmapInfo", "<OnlineID>k__BackingField")?)? as i64, ..Seen::default() };
        if hash.len() > 2 {
            seen.file = Path::new(&hash[..1]).join(&hash[..2]).join(&hash).to_string_lossy().into_owned();
        }
        if in_player {
            let score = self.at(screen, "osu.Game.Screens.Play.Player", "<Score>k__BackingField")?;
            let info = self.p.ptr(score + 8).filter(|i| score != 0 && *i != 0)?;
            // Empty for a moment after the player loads (the old tosu feed showed nomod there): the selection still holds.
            seen.mods = Some(self.score_mods(info)?).filter(|m| m["array"].as_array().is_some_and(|a| !a.is_empty())).unwrap_or(seen.mods);
            seen.hits = self.hits(info)?;
            seen.replay = false; // a replay's result names its player, which the watcher compares
            let track = self.at(self.at(clock, "osu.Game.Beatmaps.FramedBeatmapClock", "decoupledTrack")?, "osu.Framework.Timing.DecouplingFramedClock", "<Source>k__BackingField")?;
            seen.length = self.p.f64(track + self.off("osu.Framework.Audio.Track.Track", "length")?)?.round();
        }
        if state == "resultScreen" {
            seen.results = self.results(screen).unwrap_or(Value::Null);
            seen.profile = self.user(game).unwrap_or_default();
        }
        Some(seen)
    }
}
