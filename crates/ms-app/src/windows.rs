//! The small windows: ⋯ menu, rate, display, profile, overlay warning, website charts, history, saved-map comparisons. Each is
//! a real window the compositor places, so it stays on screen wherever the small main window sits (a popup inside the main
//! window would be cut off by it).
use crate::app::{App, Mail, Msg};
use crate::fmt::{self, keys_label};
use crate::theme::{self, DIM};
use eframe::egui::{self, RichText, ScrollArea, Ui, ViewportBuilder, ViewportCommand, ViewportId};
use ms_data::paths::cache;
use ms_data::store::{kv_get, now};
use ms_data::webmaps::{self, Estimate};
use ms_rec::catalog::SUPPORTED_KEYS;
use ms_rec::worker::Task;
use serde_json::Value;
use std::collections::HashSet;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

pub const DEFAULT_SKILLS: usize = 3;

fn window(ctx: &egui::Context, id: &str, title: &str, size: [f32; 2], above: bool, mut add: impl FnMut(&mut Ui)) -> bool {
    let mut builder = ViewportBuilder::default().with_title(title).with_inner_size(size).with_app_id("maniascope");
    if above {
        builder = builder.with_always_on_top();
    }
    ctx.show_viewport_immediate(ViewportId::from_hash_of(id), builder, |ui, _| {
        egui::CentralPanel::default().show(ui, |ui| add(ui));
        ui.ctx().input(|i| i.viewport().close_requested())
    })
}

#[derive(Default)]
pub struct WebInfo {
    pub progress: Value,
    pub running: bool,
    pub disk: f64,
    pub estimate: Option<Estimate>,
    pub left: f64,
    pub done: f64,
    pub error: Option<String>,
    pub reopen: Option<(String, Vec<i64>, Option<(f64, f64)>)>, // the batch closing the app paused: command, keymodes, ★
}

#[derive(Default)]
pub struct WebView {
    polling: bool,
    pub continued: bool,
    next: Option<Instant>,
    child: Option<Child>,
    pub info: Option<WebInfo>,
    ready: Option<i64>,
    keys: Option<HashSet<i64>>,
    stars: Option<(f64, f64)>,
    saved: bool,
}

fn poll(mail: Mail, view: Option<(HashSet<i64>, Option<(f64, f64)>)>) {
    std::thread::spawn(move || {
        let read = || -> rusqlite::Result<WebInfo> {
            let db = webmaps::state_db()?;
            let running = webmaps::running();
            let state = webmaps::state(&db);
            let p = state.get("progress").cloned().unwrap_or(Value::Null);
            let estimate = match view.as_ref().filter(|_| !running) {
                Some((keys, stars)) => Some(webmaps::estimate(&db, keys, *stars, ms_rec::engine::CALC_ID)?),
                None => None,
            };
            let num = |k: &str| p[k].as_i64().unwrap_or(0);
            let (mut left, mut done) = (0., 0.);
            if running && !p["fetch_total"].is_null() {
                left = webmaps::seconds(&db, num("fetch_total") - num("fetched"), num("analyse_total") - num("analysed"));
                done = 1. - left / webmaps::seconds(&db, num("fetch_total"), num("analyse_total")).max(1.);
            }
            let reopen = state.get("reopen").map(|_| {
                let keys = state.get("want_keys").and_then(|v| v.as_array()).map_or_else(Vec::new, |a| a.iter().filter_map(Value::as_i64).collect());
                (state.get("command").and_then(Value::as_str).unwrap_or("crawl").to_string(), keys, state.get("stars").and_then(|v| Some((v[0].as_f64()?, v[1].as_f64()?))))
            });
            Ok(WebInfo { disk: webmaps::disk_bytes() as f64, running, estimate, left, done, progress: p, error: None, reopen })
        };
        mail.send(Msg::Web(Box::new(read().unwrap_or_else(|e| WebInfo { error: Some(e.to_string()), ..WebInfo::default() }))));
    });
}

impl App {
    pub fn web_label(&self) -> String {
        let Some(info) = &self.web.info else { return "Website charts…".into() };
        let phase = info.progress["phase"].as_str().unwrap_or("");
        if info.running && phase != "check" {
            format!("Website charts… {:.0}%", 100. * info.done.max(0.))
        } else if matches!(phase, "paused" | "failed") {
            "Website charts… paused".into()
        } else {
            "Website charts…".into()
        }
    }

    fn web_poll(&mut self) {
        if self.web.polling {
            return;
        }
        self.web.polling = true;
        let view = self.open.web.then(|| (self.web.keys.clone().unwrap_or_default(), self.web.stars));
        poll(self.mail.clone(), view);
    }

    pub fn web_show(&mut self, info: WebInfo) {
        self.web.polling = false;
        let ready = info.progress["ready"].as_i64().unwrap_or(0);
        if self.web.ready.is_some_and(|r| r != ready) {
            self.q.put(Task::WebUpdated); // new website charts join the open playlist without a restart
        }
        self.web.ready = Some(ready);
        if let Some((command, keys, stars)) = info.reopen.as_ref().filter(|_| !info.running && self.web.child.is_none()) { // the child clears the flag a moment after it starts
            self.web_run(command, keys, *stars);
            self.web.continued = true;
        }
        let child = self.web.child.as_mut().is_some_and(|c| matches!(c.try_wait(), Ok(None)));
        self.web.next = (self.open.web || info.running || child).then(|| Instant::now() + Duration::from_secs(2));
        self.web.info = Some(info);
    }

    pub fn web_tick(&mut self, ctx: &egui::Context) {
        if self.web.ready.is_none() && !self.web.polling {
            self.web_poll(); // a paused download shows on the button after a restart
        }
        if let Some(at) = self.web.next {
            let left = at.saturating_duration_since(Instant::now());
            if left.is_zero() {
                self.web.next = None;
                self.web_poll();
            } else {
                ctx.request_repaint_after(left);
            }
        }
    }

    fn web_run(&mut self, command: &str, keys: &[i64], stars: Option<(f64, f64)>) {
        let Ok(exe) = std::env::current_exe() else { return };
        let _ = std::fs::create_dir_all(cache());
        let Ok(log) = std::fs::File::create(cache().join("webmaps.log")) else { return };
        let list = keys.iter().map(i64::to_string).collect::<Vec<_>>().join(",");
        let mut c = Command::new(exe);
        c.args(["--webmaps", command, "--keys", &list]);
        if let Some((lo, hi)) = stars {
            c.args(["--stars", &format!("{lo}-{hi}")]);
        }
        c.stdin(Stdio::null()).stderr(log.try_clone().map_or(Stdio::null(), Stdio::from)).stdout(log);
        self.web.child = c.spawn().ok();
        self.web_poll();
    }

    pub fn web_pause(&mut self) {
        if self.web.child.as_mut().is_some_and(|c| matches!(c.try_wait(), Ok(None))) { // paused, not killed: the next launch continues
            let _ = std::fs::write(webmaps::web_dir().join("stop"), "");
            if let Ok(db) = webmaps::state_db() {
                let _ = webmaps::kv_set(&db, "reopen", &Value::Bool(true));
            }
        }
    }

    /// osu! ★ range (1.0×) covering the player's recent plays in these keymodes at 0.9–1.2×, or None.
    fn recommended_stars(&self, keys: &HashSet<i64>) -> Option<(f64, f64)> {
        let mut stars: Vec<f64> = Vec::new();
        if let Ok(mut q) = self.db.prepare("SELECT info FROM events WHERE kind='start' ORDER BY id DESC LIMIT 400") {
            let rows = q.query_map([], |r| r.get::<_, Option<String>>(0)).ok()?;
            for info in rows.flatten().flatten() {
                let Ok(v) = serde_json::from_str::<Value>(&info) else { continue };
                let f = &v["features"];
                if let (Some(s), Some(k)) = (f["stars"].as_f64().filter(|s| *s != 0.), f["keys"].as_i64()) {
                    if keys.contains(&k) {
                        stars.push(s);
                    }
                }
            }
        }
        if stars.len() < 30 {
            // A new install has no tracked starts yet: its imported lazer plays say the same thing.
            stars.clear();
            let calc = ms_rec::engine::CALC_ID;
            let mut q = self.db.prepare("SELECT f.data FROM scores s JOIN feats f ON f.key = s.sha256 || '@' || printf('%.3f', s.rate) || '|' || ? WHERE s.src != 'public' AND s.rate BETWEEN .9 AND 1.2 ORDER BY s.played DESC LIMIT 400").ok()?;
            let rows = q.query_map([calc], |r| r.get::<_, Vec<u8>>(0)).ok()?;
            for data in rows.flatten() {
                if let Ok(ms_calc::val::Val::Dict(f)) = ms_data::codec::decode(&data) {
                    if f.fget("stars", 0.) != 0. && keys.contains(&(f.fget("keys", 0.) as i64)) {
                        stars.push(f.f("stars"));
                    }
                }
            }
        }
        if stars.len() < 30 {
            return None;
        }
        stars.sort_by(f64::total_cmp);
        Some(((stars[stars.len() / 10] / 1.2 * 2.).round() / 2., (stars[stars.len() * 9 / 10] / 0.9 * 2.).round() / 2.))
    }

    fn web_ui(&mut self, ui: &mut Ui) {
        if !self.web.saved {
            self.web.saved = true;
            if let Ok(db) = webmaps::state_db() {
                let state = webmaps::state(&db);
                self.web.keys = state.get("want_keys").and_then(|v| v.as_array()).map(|a| a.iter().filter_map(Value::as_i64).collect());
                self.web.stars = state.get("stars").and_then(|v| Some((v[0].as_f64()?, v[1].as_f64()?)));
            }
            let chosen: HashSet<i64> = self.s.keys["nps"].iter().chain(&self.s.keys["skills"]).copied().collect();
            let keys = self.web.keys.get_or_insert(chosen).clone();
            self.web.stars = self.web.stars.or_else(|| self.recommended_stars(&keys)).or(Some((5., 10.)));
            self.web_poll();
        }
        ui.label("Downloads the chart text (no audio) of maps you don't have from osu.direct, analyses it and adds those maps to the NPS and Skills playlists; Next then offers the download in osu!. Runs at low priority. Pause any time; closing ManiaScope pauses it too, and it continues where it stopped the next time you open the app.");
        let (running, phase) = self.web.info.as_ref().map_or((false, String::new()), |i| (i.running, i.progress["phase"].as_str().unwrap_or("").to_string()));
        let mut changed = false;
        let keys = self.web.keys.clone().unwrap_or_default();
        ui.add_enabled_ui(!running, |ui| {
            ui.horizontal(|ui| {
                ui.label("Keymodes");
                for k in SUPPORTED_KEYS {
                    let mut on = keys.contains(&k);
                    if ui.checkbox(&mut on, format!("{k}K")).changed() {
                        let set = self.web.keys.get_or_insert_default();
                        if on { set.insert(k); } else { set.remove(&k); }
                        changed = true;
                    }
                }
            });
            ui.horizontal(|ui| {
                ui.label("osu! ★ at 1.0×");
                let (mut lo, mut hi) = self.web.stars.unwrap_or((5., 10.));
                changed |= ui.add(egui::DragValue::new(&mut lo).range(0.0..=15.).speed(0.5).fixed_decimals(1)).changed();
                changed |= ui.add(egui::DragValue::new(&mut hi).range(0.0..=15.).speed(0.5).fixed_decimals(1)).changed();
                self.web.stars = Some((lo, hi));
                if let Some(r) = self.recommended_stars(&keys) {
                    theme::dim(ui, format!("(your plays suggest {}–{})", r.0, r.1));
                }
            });
        });
        if changed {
            self.web_poll();
        }
        let selected: Vec<i64> = { let mut v: Vec<i64> = self.web.keys.iter().flatten().copied().collect(); v.sort_unstable(); v };
        let stars = self.web.stars.map(|(a, b)| (a.min(b), a.max(b)));
        let c = ms_rec::engine::commas;
        let i64n = |v: i64| c(v.max(0) as usize);
        let mut go_ok = false;
        match &self.web.info {
            None => { ui.label("Counting the charts in this range…"); }
            Some(info) => {
                let p = &info.progress;
                let num = |k: &str| p[k].as_i64().unwrap_or(0);
                if let Some(e) = &info.error {
                    ui.label(format!("Could not read the website chart state: {e}"));
                } else if running && phase == "check" {
                    let what = if p["full"].as_bool().unwrap_or(false) { "A keymode not listed yet needs the whole mania listing, once: about 15–30 min, and it continues from here if you pause or close." } else { "Only what changed since the last check." };
                    ui.label(format!("Checking osu.direct: {} map sets listed so far. {what}", i64n(num("sets"))));
                } else if running {
                    if self.web.continued {
                        ui.label("Continuing the batch you left unfinished.");
                    }
                    ui.label(format!("Downloaded {} of {}, analysed {} of {}. {} charts are in the playlist already (the first after 300, then in batches).\nAbout {} left · {} on disk now.",
                                     i64n(num("fetched")), i64n(num("fetch_total")), i64n(num("analysed")), i64n(num("analyse_total")), i64n(num("ready")), fmt::duration(info.left), fmt::size(info.disk)));
                    theme::bar(ui, info.done as f32);
                } else if let Some(est) = &info.estimate {
                    let mut lines = Vec::new();
                    if phase == "failed" {
                        lines.push(format!("Stopped with an error: {}. Resume retries.", p["error"].as_str().unwrap_or("?")));
                    } else if phase == "paused" {
                        lines.push("Paused.".to_string());
                    }
                    if !est.unlisted.is_empty() {
                        lines.push(format!("{} not listed yet. Start first reads osu.direct's mania listing (about 15–30 min the first time, seconds after that), then downloads; Check only reads the listing, to see the size first.", keys_label(&est.unlisted)));
                    }
                    if selected.is_empty() {
                        lines.push("Pick at least one keymode.".to_string());
                    } else if est.fetch > 0 || est.analyse > 0 {
                        lines.push(format!("To do: {} charts to download (about {}) and {} to analyse, about {} on this PC.", i64n(est.fetch), fmt::size(est.new_bytes as f64), i64n(est.analyse), fmt::duration(est.seconds)));
                        lines.push(format!("On disk now {}, afterwards about {}. {} charts in this range are in the playlist already.", fmt::size(est.disk as f64), fmt::size(est.disk as f64 + est.new_bytes as f64), i64n(est.ready)));
                    } else if est.unlisted.is_empty() {
                        lines.push(format!("Up to date: {} charts in this range are in the playlist, {} on disk.", i64n(est.ready), fmt::size(est.disk as f64)));
                    }
                    if let Some(t) = est.checked.filter(|t| *t != 0.) {
                        lines.push(format!("Listing checked {} ago.", fmt::duration(now() - t)));
                    }
                    ui.label(lines.join("\n"));
                    // Start lists unchecked keymodes itself: a first run needs no separate Check (fresh install).
                    go_ok = est.fetch > 0 || est.analyse > 0 || !est.unlisted.is_empty() || phase == "paused";
                }
            }
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
            let label = if running { "Pause" } else if matches!(phase.as_str(), "check" | "download" | "analyse" | "paused" | "failed") { "Resume" } else { "Start" };
            if ui.add_enabled(running || go_ok, egui::Button::new(label)).clicked() {
                if running {
                    let _ = std::fs::write(webmaps::web_dir().join("stop"), ""); // pauses after the current chart
                } else {
                    self.web_run("crawl", &selected, stars);
                }
            }
            if ui.add_enabled(!running && !selected.is_empty(), egui::Button::new("Check for new charts")).clicked() {
                self.web_run("check", &selected, None);
            }
        });
    }

    pub fn menu_ui(&mut self, ui: &mut Ui) {
        let mut close = false;
        let selected = self.analysis().map(|a| a.keys);
        ui.horizontal(|ui| {
            ui.label("Convert this map to");
            for k in SUPPORTED_KEYS.filter(|k| Some(*k) != selected) {
                if ui.add_enabled(selected.is_some(), egui::Button::new(format!("{k}K"))).on_hover_text("Adds it to osu! as its own difficulty, without the playlist").clicked() {
                    self.q.put(Task::ConvertSelected(k));
                    close = true;
                }
            }
        });
        ui.separator();
        if ui.button("Skip warmup").on_hover_text("Already warm: remove the cold-session prior for PP, NPS and Skills").clicked() {
            self.q.put(Task::SkipWarmup);
            close = true;
        }
        if ui.checkbox(&mut self.s.auto_next, "Auto next after finishing the pick").on_hover_text("Finishing the recommended map picks the next one and copies its search, as if you pressed Next").changed() {
            self.request_save();
        }
        let mut above = self.s.keep_above;
        if ui.checkbox(&mut above, "Always on top").on_hover_text("Keep ManiaScope and the playlist above other windows. Not above exclusive fullscreen; for that add ManiaScope to tosu's in-game overlay.").changed() {
            self.s.keep_above = above;
            self.request_save();
            ui.ctx().send_viewport_cmd_to(ViewportId::ROOT, ViewportCommand::WindowLevel(level(above)));
            ui.ctx().send_viewport_cmd_to(playlist_id(), ViewportCommand::WindowLevel(level(above)));
        }
        if cfg!(windows) {
            let mut on = self.s.overlay;
            if ui.checkbox(&mut on, "ManiaScope in your tosu overlay").on_hover_text("If you run tosu yourself: ManiaScope adds its page there. In osu! press Ctrl+Shift+Space, right-click, add ManiaScope, then press Ctrl+Shift+Space again.").changed() {
                self.toggle_overlay(on);
            }
        }
        if ui.button("osu! profile…").on_hover_text("Your username: your public top 100 plays fill in bests that lazer doesn't have locally").clicked() {
            self.open.profile = if self.open.profile.is_some() { None } else { Some(false) };
            self.open.profile_text = kv_get(&self.db, "website_username").and_then(|v| v.as_str().map(str::to_string)).unwrap_or_default();
        }
        if self.open.profile == Some(false) {
            self.profile_ui(ui);
        }
        ui.separator();
        ui.label("Main window shows");
        self.display_ui(ui);
        if self.debug && ui.button("Play & prediction history…").clicked() {
            self.open.history = true;
            self.q.put(Task::History(false));
            close = true;
        }
        if close {
            self.open.menu = false;
        }
    }

    fn profile_ui(&mut self, ui: &mut Ui) {
        ui.label("Your osu! username or profile number. ManiaScope reads your public top 100 mania plays so PP suggestions know your bests. Empty = off.");
        let edit = ui.text_edit_singleline(&mut self.open.profile_text);
        let enter = edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
        let first = self.open.profile == Some(true);
        ui.horizontal(|ui| {
            if ui.button("Save").clicked() || enter {
                self.q.put(Task::WebsiteUser(self.open.profile_text.clone()));
                self.open.profile = None;
            }
            if ui.button(if first { "Skip" } else { "Cancel" }).clicked() {
                self.open.profile = None;
            }
        });
    }

    fn display_ui(&mut self, ui: &mut Ui) {
        let d = &mut self.s.display;
        let mut changed = false;
        changed |= ui.checkbox(&mut d.rating, "Difficulty rating").changed();
        changed |= ui.checkbox(&mut d.dan, "Dan level").changed();
        changed |= ui.checkbox(&mut d.nps, "NPS").changed();
        changed |= ui.checkbox(&mut d.ln, "LN %").changed();
        changed |= ui.checkbox(&mut d.timeline, "Difficulty timeline").changed();
        ui.horizontal(|ui| {
            ui.label("Skills shown");
            changed |= ui.add(egui::DragValue::new(&mut d.skills).range(1..=6)).on_hover_text(format!("Default {DEFAULT_SKILLS}")).changed();
        });
        if changed {
            self.request_save();
        }
    }

    pub fn side_windows(&mut self, ctx: &egui::Context) {
        let above = self.s.keep_above;
        if self.open.profile == Some(true) && window(ctx, "profile", "Welcome to ManiaScope", [380., 150.], above, |ui| self.profile_ui(ui)) {
            self.open.profile = None;
        }
        if self.open.menu {
            self.open.menu &= !window(ctx, "menu", "ManiaScope — Settings", [430., 470.], above, |ui| self.menu_ui(ui));
            if !self.open.menu {
                self.open.profile = self.open.profile.filter(|first| *first);
            }
        }
        if self.open.web {
            self.open.web = !window(ctx, "web", "ManiaScope — Website charts", [480., 300.], above, |ui| self.web_ui(ui));
            if !self.open.web {
                self.web.saved = false;
            }
        }
        if self.open.history {
            self.open.history = !window(ctx, "history", "Play & prediction history", [760., 520.], above, |ui| {
                if ui.checkbox(&mut self.open.history_only, "Only unusually high or low results (±1.5σ)").changed() {
                    self.q.put(Task::History(self.open.history_only));
                }
                ui.label(RichText::new("Latest 100 plays; unusual results sorted by prediction error. Accuracy is lazer accuracy.").color(DIM));
                ScrollArea::vertical().show(ui, |ui| ui.label(&self.open.history_text));
            });
        }
    }
}

pub fn level(above: bool) -> egui::WindowLevel {
    if above { egui::WindowLevel::AlwaysOnTop } else { egui::WindowLevel::Normal }
}

pub fn playlist_id() -> ViewportId {
    ViewportId::from_hash_of("playlist")
}
