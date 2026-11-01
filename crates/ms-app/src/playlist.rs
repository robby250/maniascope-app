//! The playlist window, its own so the main one can stay an OBS source: selected map, last play, the tabs and their settings.
use crate::app::App;
use crate::fmt::{self, keys_label};
use crate::settings::{mode_name, MODES};
use crate::theme::{self, ACCENT, BORDER, CHIP_OFF, CHIP_ON, DIM};
use eframe::egui::{self, containers::menu::{MenuButton, MenuConfig}, Color32, PopupCloseBehavior, RichText, ScrollArea, Sense, Stroke, Ui, UiBuilder};
use ms_data::webmaps::web_dir;
use ms_model::skills::{self, Skill, AVOID};
use ms_rec::builder::{normalize_focus, normalize_target, web_factor, DEFAULT_TARGET};
use ms_rec::catalog::{Pick, SUPPORTED_KEYS};
use ms_rec::converts::Setting;

/// The side the slider leans to, so the left half doesn't read "20% NPS" (user 2026-11-01).
pub fn focus_label(v: f64) -> String {
    match (100. * v).round() as i64 {
        50 => "Even mix".to_string(),
        n if n < 50 => format!("{}% variety", 100 - n),
        n => format!("{n}% NPS"),
    }
}

const GEAR: f32 = 19.; // the tab cog's glyph; 11 was "way too small" (user 2026-11-01)
const NAME: f32 = 134.; // the longest row name fits
const VALUE: f32 = 106.; // "95.0% (default)" fits
const WEIGHTS: [f64; 6] = [0.25, 0.5, 0.75, 1., 1.5, 2.]; // multiplies a skill's share of the list (user 2026-11-01: "0.25x-2x")
const OTHER: [f64; 7] = [0., 0.25, 0.5, 0.75, 1., 1.5, 2.];
const CHIP_GAP: f32 = 4.;

fn row(ui: &mut Ui, name: &str, tip: Option<&str>, add: impl FnOnce(&mut Ui) -> bool) -> bool {
    let r = ui.horizontal(|ui| {
        ui.allocate_ui_with_layout(egui::vec2(NAME, 16.), egui::Layout::left_to_right(egui::Align::Center), |ui| {
            ui.set_min_width(NAME);
            ui.add(egui::Label::new(name).truncate());
        });
        add(ui)
    });
    if let Some(t) = tip {
        r.response.on_hover_text(t);
    }
    r.inner
}

/// A settings row: name, slider, current value.
fn slider(ui: &mut Ui, name: &str, value: &mut f64, range: (f64, f64, f64), show: impl Fn(f64) -> String, tip: Option<&str>) -> bool {
    row(ui, name, tip, |ui| {
        ui.spacing_mut().slider_width = (ui.available_width() - VALUE).max(60.);
        let changed = ui.add(egui::Slider::new(value, range.0..=range.1).step_by(range.2).show_value(false)).on_hover_text(name).changed();
        ui.label(show(*value));
        changed
    })
}

/// The slider steps through a list.
fn stepped(ui: &mut Ui, name: &str, value: &mut f64, values: &[f64], show: impl Fn(f64) -> String, tip: Option<&str>) -> bool {
    let mut at = values.iter().enumerate().min_by(|a, b| (a.1 - *value).abs().total_cmp(&(b.1 - *value).abs())).map_or(0, |x| x.0);
    let changed = row(ui, name, tip, |ui| {
        ui.spacing_mut().slider_width = (ui.available_width() - VALUE).max(60.);
        let c = ui.add(egui::Slider::new(&mut at, 0..=values.len() - 1).show_value(false)).on_hover_text(name).changed();
        ui.label(show(values[at]));
        c
    });
    if changed {
        *value = values[at];
    }
    changed
}

/// A chip's width, before it exists: rows are cut from this. A `ui.horizontal` inside `horizontal_wrapped` has no size when
/// the wrap is decided, so the row ran off the window (user 2026-11-01: "the buttons go offscreen").
fn chip_width(ui: &Ui, text: &str) -> f32 {
    let mark = ui.painter().layout_no_wrap("✓ ".to_string(), egui::FontId::proportional(theme::BODY), theme::TEXT).size().x;
    let text = ui.painter().layout_no_wrap(text.to_string(), egui::FontId::proportional(theme::BODY), theme::TEXT).size().x;
    text + mark + 2. * ui.spacing().button_padding.x + 2.
}

/// The indices of each row, in order: a unit wider than the whole width still gets a row of its own.
pub fn wrap_rows(widths: &[f32], available: f32, gap: f32) -> Vec<Vec<usize>> {
    let mut rows: Vec<Vec<usize>> = Vec::new();
    let mut used = 0.;
    for (i, w) in widths.iter().enumerate() {
        match rows.last_mut() {
            Some(row) if used + gap + w <= available => {
                row.push(i);
                used += gap + w;
            }
            _ => {
                rows.push(vec![i]);
                used = *w;
            }
        }
    }
    rows
}

fn weight_text(value: f64, rest: f64) -> String {
    format!("{value}× · {:.0}%", 100. * value / (value + rest))
}

fn switch(ui: &mut Ui, name: &str, value: &mut bool, tip: Option<&str>) -> bool {
    let r = ui.checkbox(value, name);
    let changed = r.changed();
    if let Some(t) = tip {
        r.on_hover_text(t);
    }
    changed
}

const WEIGHT_TIP: &str = "On: the slider sets the chance per map, so at the middle every map is equally likely and more maps mean more picks. Off: the slider is a plain share of your picks.";

fn how_often(value: f64, weighted: bool) -> String {
    let f = web_factor(value);
    if f <= 0. {
        "never".to_string()
    } else if weighted {
        if (f - 1.).abs() < 1e-9 { "same per map".to_string() } else { format!("{f:.1}× per map") }
    } else {
        pct(f / (1. + f)).replace('%', "% of picks")
    }
}

fn pct(v: f64) -> String {
    format!("{:.0}%", 100. * v)
}

fn pp_text(v: f64) -> String {
    format!("{v}pp")
}

fn row_text(c: &Pick, mode: &str, shown: &ms_rec::history::Shown) -> (String, Vec<String>) {
    let length = fmt::clock(c.length);
    let acc = 100. * shown.acc(c.pred.acc_mid);
    let convert = c.convert_of.as_ref().map(|_| format!("{}K→{}K", c.source_keys, c.keys));
    let (number, mut sub) = if c.purpose == "nps_mix" || (mode == "pp" && c.purpose != "warmup") {
        let n = format!("{}+{:.1}pp · {:.0}%", if mode != "pp" { "PP " } else { "" }, c.gain.unwrap_or(0.), 100. * c.p_up.unwrap_or(0.));
        (n, vec![c.var.clone(), c.stars.map_or(String::new(), |s| format!("{s:.1}★")), c.best.filter(|b| *b != 0.).map_or("new".to_string(), |b| format!("best {b:.0}pp")), length])
    } else if mode == "pp" {
        (format!("warm-up ~{acc:.1}%"), vec![length])
    } else {
        let mut sub = if mode == "nps" { vec![format!("{:.1} NPS", c.nps.unwrap_or(0.))] } else { Vec::new() };
        sub.extend([length, if c.description.is_empty() { c.skill.clone().unwrap_or_default() } else { c.description.clone() }]);
        (format!("~{acc:.1}%"), sub)
    };
    if let Some(c) = convert {
        sub.push(c);
    } else if !c.installed {
        sub.push(if mode != "pp" { "download" } else { "not installed" }.to_string());
    }
    (number, sub.into_iter().filter(|s| !s.is_empty()).collect())
}

impl App {
    fn key_picker(&mut self, ui: &mut Ui, mode: &str) {
        let keys = self.s.keys[mode].clone();
        let mut next = None;
        ui.horizontal(|ui| {
            ui.allocate_ui_with_layout(egui::vec2(NAME, 16.), egui::Layout::left_to_right(egui::Align::Center), |ui| {
                ui.set_min_width(NAME);
                ui.label("Keymodes");
            });
            let config = MenuConfig::new().close_behavior(PopupCloseBehavior::CloseOnClickOutside);
            MenuButton::new(keys_label(&keys)).config(config).ui(ui, |ui| {
                if ui.button("All supported modes (4K–10K)").clicked() {
                    next = Some(SUPPORTED_KEYS.collect::<Vec<_>>());
                }
                for k in SUPPORTED_KEYS {
                    let mut on = keys.contains(&k);
                    if ui.checkbox(&mut on, format!("{k}K")).changed() {
                        let mut v: Vec<i64> = SUPPORTED_KEYS.filter(|x| if *x == k { on } else { keys.contains(x) }).collect();
                        v.sort_unstable();
                        next = Some(v);
                    }
                }
            }).0.on_hover_text("Choose any combination independently for this tab. All covers 4K–10K; empty selection stays empty.");
        });
        if let Some(v) = next {
            self.set_keys(mode, v);
        }
    }

    fn acc_target(&mut self, ui: &mut Ui, mode: &str) {
        let mut v = 100. * self.s.targets[mode];
        if slider(ui, "Accuracy target", &mut v, (88., 98., 0.5), |v| format!("{v:.1}%{}", if v / 100. == DEFAULT_TARGET { " (default)" } else { "" }), Some("Maps and rates are chosen so the predicted accuracy lands near this.")) {
            self.s.targets.insert(mode.to_string(), normalize_target(v / 100.));
            self.sync_rec();
        }
    }

    /// How often this tab suggests website charts (maps you don't have) next to your own.
    fn web_bias(&mut self, ui: &mut Ui, mode: &str) {
        // Without website charts it can change nothing (Less, 2026-10-31: set "Never", saw no difference).
        let have = web_dir().join("web.msp").is_file() || web_dir().join("web.pkl").is_file();
        let tip = if have { "Website charts (Website charts…): their share of your picks. Middle: half. Right end: 80 %. Left end: never, only maps you own." } else { "Nothing to mix yet: add charts with Website charts… on the NPS tab. Until then every pick is a map you own." };
        let mut v = self.s.web_bias[mode];
        let weighted = self.s.web_weighted[mode];
        if ui.add_enabled_ui(have, |ui| slider(ui, "Maps you don't have", &mut v, (0., 1., 0.05), |v| how_often(v, weighted), Some(tip))).inner {
            self.s.web_bias.insert(mode.to_string(), normalize_focus(v));
            self.sync_rec();
        }
        let mut on = weighted;
        if ui.add_enabled_ui(have, |ui| switch(ui, "Weight website maps by map count", &mut on, Some(WEIGHT_TIP))).inner {
            self.s.web_weighted.insert(mode.to_string(), on);
            self.sync_rec();
        }
    }

    fn web_button(&mut self, ui: &mut Ui) {
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
            let label = self.web_label();
            if ui.button(label).on_hover_text("Add maps you don't have to the playlists (chart text only, no audio).").clicked() {
                self.open.web = true;
            }
        });
    }

    /// Keymode converts for each of the tab's keymodes: which keymodes feed it and how often they come up.
    fn converts(&mut self, ui: &mut Ui, mode: &'static str) {
        theme::section(ui, "CONVERTS");
        for target in self.s.keys[mode].clone() {
            let mut s = self.s.converts[mode].get(&target).cloned().unwrap_or_else(|| Setting::new(target));
            let mut changed = switch(ui, &format!("Convert other keymodes to {target}K"), &mut s.on, Some("Maps of other keymodes join this list, converted when their turn comes. Their difficulty is known ahead."));
            if s.on {
                changed |= row(ui, "From", Some("The keymodes to convert from."), |ui| {
                    let mut changed = false;
                    ui.spacing_mut().item_spacing.x = 3.;
                    for k in SUPPORTED_KEYS.filter(|k| *k != target) {
                        let mut on = s.sources.contains(&k);
                        if ui.toggle_value(&mut on, format!("{k}K")).changed() {
                            s.sources.retain(|x| *x != k);
                            if on {
                                s.sources.push(k);
                                s.sources.sort_unstable();
                            }
                            changed = true;
                        }
                    }
                    changed
                });
                let weighted = s.weighted;
                changed |= slider(ui, "How often", &mut s.often, (0., 1., 0.05), |v| how_often(v, weighted), Some("Converts against real maps of this keymode."));
                changed |= switch(ui, "Weight converts by map count", &mut s.weighted, Some(WEIGHT_TIP));
            }
            if changed {
                s.often = normalize_focus(s.often);
                self.s.converts.get_mut(mode).unwrap().insert(target, s);
                self.sync_rec();
            }
        }
    }

    fn chip(&mut self, ui: &mut Ui, key: &str, name: &str, tip: &str) {
        let (state, mixed) = (skills::state(&self.s.practice, key), skills::mixed(&self.s.practice, key));
        let word = if mixed { "mixed" } else { match state { 1 => "included", -1 => "excluded", _ => "off" } };
        let (mark, colour) = if mixed { ("– ", DIM) } else { match state { 1 => ("✓ ", CHIP_ON), -1 => ("✕ ", CHIP_OFF), _ => ("", BORDER) } };
        let mut text = egui::text::LayoutJob::default();
        let plain = egui::TextFormat { font_id: egui::FontId::proportional(theme::BODY), color: theme::TEXT, ..Default::default() };
        text.append(mark, 0., egui::TextFormat { color: colour, ..plain.clone() });
        text.append(name, 0., plain);
        let mut b = egui::Button::new(text).stroke(Stroke::new(1., colour));
        if !mixed && state != 0 {
            b = b.fill(colour.gamma_multiply(0.16));
        }
        let tip = if tip.is_empty() { "Broad practice for this family; ▾ opens its subskills.".to_string() } else { tip.to_string() };
        if ui.add(b).on_hover_text(format!("{name}: {word}\n{tip}")).clicked() {
            let next = skills::cycle(&self.s.practice, key);
            self.set_skills(next);
        }
    }

    /// The wrapped units of a chip row: a unit is drawn by `add`.
    fn chip_rows<T>(&mut self, ui: &mut Ui, units: &[T], width: impl Fn(&Ui, &T) -> f32, mut add: impl FnMut(&mut Self, &mut Ui, &T)) {
        let widths: Vec<f32> = units.iter().map(|u| width(ui, u)).collect();
        for row in wrap_rows(&widths, ui.available_width(), CHIP_GAP) {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing = egui::vec2(CHIP_GAP, CHIP_GAP);
                for i in row {
                    add(self, ui, &units[i]);
                }
            });
        }
    }

    fn any_chip(&mut self, ui: &mut Ui) {
        let on = self.s.practice.is_empty();
        let mut b = egui::Button::new("Any").stroke(Stroke::new(1., if on { CHIP_ON } else { BORDER }));
        if on {
            b = b.fill(CHIP_ON.gamma_multiply(0.16));
        }
        if ui.add(b).on_hover_text("Regular practice: no skill chosen or avoided.").clicked() {
            self.set_skills(Vec::new());
        }
    }

    /// Skills chips, over the same taxonomy as player Stats. Replaced a popover of checkboxes whose third state read as dots
    /// (user 2026-10-31: "kinda scuffed").
    fn skill_picker(&mut self, ui: &mut Ui) {
        theme::dim_small(ui, "Click: include → avoid → off. ▾ opens subskills.");
        let keys = self.s.keys["skills"].clone();
        let modes: Vec<i64> = [keys.contains(&4).then_some(4), keys.iter().any(|k| *k >= 5).then_some(7)].into_iter().flatten().collect();
        if modes.is_empty() {
            ui.label("Select a keymode first.");
        }
        for (at, mode) in modes.iter().copied().enumerate() {
            if modes.len() > 1 {
                ui.label(RichText::new(if mode == 4 { "4K" } else { "5K–10K" }).family(egui::FontFamily::Name("bold".into())));
            }
            let mut units: Vec<Option<&Skill>> = if at == 0 { vec![None] } else { Vec::new() };
            units.extend(skills::tree(mode).iter().map(Some));
            self.chip_rows(ui, &units, |ui, u| u.map_or(chip_width(ui, "Any"), |p| chip_width(ui, p.name) + if p.children.is_empty() { 0. } else { chip_width(ui, "▾") }), |app, ui, u| {
                let Some(parent) = u else { return app.any_chip(ui) };
                // Together: a chip and its ▾ wrap as one.
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 0.;
                    app.chip(ui, &parent.key, parent.name, parent.tip);
                    if !parent.children.is_empty() {
                        let id = (mode, parent.key.clone());
                        let open = app.chips_open.contains(&id);
                        if ui.selectable_label(open, "▾").on_hover_text(format!("{} subskills", parent.name)).clicked() {
                            if open { app.chips_open.remove(&id); } else { app.chips_open.insert(id); }
                        }
                    }
                });
            });
            for parent in skills::tree(mode) {
                if !parent.children.is_empty() && self.chips_open.contains(&(mode, parent.key.clone())) {
                    theme::dim(ui, parent.name);
                    let children: Vec<&Skill> = parent.children.iter().collect();
                    self.chip_rows(ui, &children, |ui, c| chip_width(ui, c.name), |app, ui, c| app.chip(ui, &c.key, c.name, c.tip));
                }
            }
        }
        self.skill_sliders(ui);
    }

    /// How much of each included skill, and how strongly each avoided one is kept out.
    fn skill_sliders(&mut self, ui: &mut Ui) {
        let (included, avoided): (Vec<String>, Vec<String>) = self.s.practice.iter().cloned().partition(|k| !k.starts_with('!'));
        if !included.is_empty() {
            theme::section(ui, "HOW MUCH OF EACH");
            let total = included.iter().map(|k| self.s.skill_weights.get(k).copied().unwrap_or(1.)).sum::<f64>() + self.s.skill_other;
            for key in included {
                let mut v = self.s.skill_weights.get(&key).copied().unwrap_or(1.);
                let before = v;
                if stepped(ui, &format!("✓ {}", skills::name(&key)), &mut v, &WEIGHTS, |x| weight_text(x, total - before), Some("How much of the list this skill gets, against the others here.")) {
                    self.s.skill_weights.insert(key, v);
                    self.sync_rec();
                }
            }
            let mut v = self.s.skill_other;
            if stepped(ui, "Other maps", &mut v, &OTHER, |x| if x == 0. { "Off".to_string() } else { weight_text(x, total - self.s.skill_other) }, Some("Off: only the skills above. On: maps without them join the list at this share.")) {
                self.s.skill_other = v;
                self.sync_rec();
            }
        }
        if !avoided.is_empty() {
            theme::section(ui, "AVOID");
            for token in avoided {
                let key = token.trim_start_matches('!').to_string();
                let hold = self.s.skill_tuning().hold(&key);
                let mut v = AVOID.iter().position(|a| a.0 == hold).unwrap_or(1) as f64;
                if stepped(ui, &format!("✕ {}", skills::name(&key)), &mut v, &[0., 1., 2.], |x| AVOID[x as usize].1.to_string(), Some("How much of it a map may hold before it leaves the list.")) {
                    self.s.skill_avoid.insert(key, AVOID[v as usize].0);
                    self.sync_rec();
                }
            }
        }
    }

    fn settings_page(&mut self, ui: &mut Ui, mode: &'static str) {
        ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            ui.set_max_width(ui.available_width() - 10.); // the scroll bar's place
            ui.spacing_mut().item_spacing.y = 6.;
            match mode {
                "nps" => {
                    theme::section(ui, "MAPS");
                    self.key_picker(ui, mode);
                    self.acc_target(ui, mode);
                    theme::section(ui, "MIX");
                    let mut focus = self.s.nps_focus;
                    if slider(ui, "Variety to high NPS", &mut focus, (0., 1., 0.05), focus_label, Some("Left: pattern types (jacks, streams, LN…) get an even share of the list. Right: the highest NPS maps, mostly chordstreams and brackets. Both aim at the accuracy target.")) {
                        self.s.nps_focus = normalize_focus(focus);
                        self.sync_rec();
                    }
                    self.web_bias(ui, mode);
                    self.web_button(ui);
                    self.converts(ui, mode);
                    theme::section(ui, "PP MAPS IN THIS LIST");
                    let mut changed = false;
                    let m = &mut self.s.mix;
                    changed |= switch(ui, "Mix in likely PP gains", &mut m.on, Some("PP maps you are likely to beat now, between NPS maps. Off: no PP scoring while you play NPS."));
                    if m.on {
                        let mut every = m.every as f64;
                        if slider(ui, "One PP map every", &mut every, (2., 8., 1.), |v| format!("{v:.0} picks"), None) {
                            m.every = every as usize;
                            changed = true;
                        }
                        changed |= stepped(ui, "Min gain", &mut m.min_gain, &[1., 2., 3., 5., 8., 10., 15.], pp_text, None);
                        changed |= slider(ui, "Min chance", &mut m.chance, (0.2, 0.9, 0.05), pct, Some("Chance to beat your best, NM and DT. It already includes this session's form."));
                        changed |= slider(ui, "Min chance (HT)", &mut m.ht_chance, (0.4, 0.95, 0.05), pct, None);
                    }
                    if changed {
                        self.sync_rec();
                    }
                }
                "skills" => {
                    theme::section(ui, "SKILLS");
                    self.skill_picker(ui);
                    theme::section(ui, "MAPS");
                    self.key_picker(ui, mode);
                    self.acc_target(ui, mode);
                    self.web_bias(ui, mode);
                    self.converts(ui, mode);
                }
                _ => {
                    theme::section(ui, "CHOICE");
                    self.key_picker(ui, mode);
                    let mut changed = false;
                    let o = &mut self.s.pp;
                    changed |= stepped(ui, "Min gain", &mut o.min_gain, &[0.01, 0.1, 0.25, 0.5, 1., 2., 3., 5., 10.], pp_text, Some("Expected account pp gain a map needs."));
                    changed |= slider(ui, "Min chance", &mut o.min_chance, (0.05, 0.6, 0.01), pct, Some("Chance to beat your best. Maps above your usual level keep at least 25%."));
                    changed |= switch(ui, "Top-100 maps only", &mut o.top100, Some("Only maps that are, or would enter, your top 100."));
                    theme::section(ui, "MODS");
                    changed |= switch(ui, "DT", &mut o.dt, None);
                    changed |= switch(ui, "HT", &mut o.ht, Some("HT only as a likely, real improvement."));
                    if o.ht {
                        changed |= slider(ui, "HT min chance", &mut o.ht_chance, (0.3, 0.95, 0.05), pct, None);
                        changed |= stepped(ui, "HT min gain", &mut o.ht_gain, &[1., 2., 3., 5., 8., 10., 15., 20.], pp_text, None);
                    }
                    theme::section(ui, "STYLE");
                    changed |= slider(ui, "Safe to Push", &mut o.push, (-0.5, 0.5, 0.05), |v| if v.abs() < 0.01 { "auto".to_string() } else { format!("{} {:.2}", if v > 0. { "push" } else { "safe" }, v.abs()) }, Some("Middle: follows your form. Push: higher-pp, lower-accuracy maps. Safe: comfortable ones."));
                    theme::section(ui, "REPEATS");
                    changed |= slider(ui, "Repeat cooldown", &mut o.cooldown_days, (1., 14., 1.), |v| format!("{v:.0} days"), Some("Offers and plays older than this no longer hold a map back."));
                    if changed {
                        self.sync_rec();
                    }
                }
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
                if ui.button("Done").clicked() {
                    self.views.get_mut(mode).expect("a tab").settings = false;
                }
            });
        });
    }

    fn list_page(&mut self, ui: &mut Ui, mode: &'static str) {
        let v = &self.views[mode];
        if let Some(c) = &v.target {
            let block = |ui: &mut Ui| { theme::block(ui, &[(fmt::target_lines(c, &self.obs, &self.units)[0].clone(), 1), (fmt::target_lines(c, &self.obs, &self.units)[1].clone(), 2)], v.target_tip.as_deref()); };
            egui::Frame::new().fill(theme::ROW_CURRENT).inner_margin(egui::Margin::symmetric(6, 5)).show(ui, |ui| {
                ui.set_width(ui.available_width());
                block(ui);
            });
        } else if !v.target_msg.is_empty() {
            theme::block(ui, &[(vec![fmt::plain(v.target_msg.clone())], 1)], v.target_tip.as_deref());
        }
        let note_height = if v.note.is_empty() { 0. } else { 18. };
        let mut clicked = None;
        let height = (ui.available_height() - note_height).max(40.);
        let (shown, loaded, status) = (v.shown.clone(), v.loaded, self.status_text.clone());
        let units = (self.units.client.clone(), self.units.miss_ratio);
        let area = ScrollArea::vertical().id_salt(mode).max_height(height).auto_shrink([false, false]).show(ui, |ui| {
            if shown.is_empty() {
                ui.vertical_centered(|ui| {
                    ui.add_space(24.);
                    let head = if loaded { "Nothing to suggest yet.".to_string() } else { format!("Loading {} suggestions…", mode_name(mode)) };
                    theme::dim(ui, if status.is_empty() { head } else { format!("{head}\n\n{status}") });
                });
            }
            let units = ms_rec::history::Shown { client: units.0.clone(), miss_ratio: units.1 };
            ui.spacing_mut().item_spacing.y = 0.;
            for (i, c) in shown.iter().enumerate() {
                let (number, sub) = row_text(c, mode, &units);
                let r = ui.scope_builder(UiBuilder::new().id_salt(i).sense(Sense::click()), |ui| {
                    ui.style_mut().interaction.selectable_labels = false; // selectable text took the hover from its row (user 2026-11-01)
                    ui.spacing_mut().item_spacing.y = 2.;
                    let fill = if ui.response().hovered() { theme::ROW_HOVER } else if i % 2 == 1 { theme::BUTTON } else { Color32::TRANSPARENT };
                    let frame = egui::Frame::new().inner_margin(egui::Margin::symmetric(6, 7)).fill(fill);
                    frame.show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.horizontal(|ui| {
                            let width = ui.available_width() - 104.;
                            let mut line = vec![fmt::bold(format!("{:.2}×", c.rate)), fmt::plain("  ")];
                            line.extend(fmt::pick_line(c));
                            ui.scope(|ui| {
                                ui.set_max_width(width.max(60.));
                                theme::line(ui, &line, 3);
                            });
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
                                ui.label(RichText::new(&number).family(egui::FontFamily::Name("bold".into())));
                            });
                        });
                        theme::dim_small(ui, sub.join(" · "));
                    });
                }).response;
                let r = r.on_hover_cursor(egui::CursorIcon::PointingHand).on_hover_text(&c.why);
                if r.clicked() {
                    clicked = Some(i);
                }
            }
        });
        let hover = ui.rect_contains_pointer(area.inner_rect);
        let view = self.views.get_mut(mode).expect("a tab");
        let left = view.hover && !hover;
        view.hover = hover;
        if left {
            self.flush_pool(mode); // a list under the pointer never moves
        }
        let v = &self.views[mode];
        if !v.note.is_empty() {
            let (note, tip) = (v.note.clone(), v.note_tip.clone());
            let r = theme::dim_small(ui, note);
            if !tip.is_empty() {
                r.on_hover_text(tip);
            }
        }
        if let Some(i) = clicked {
            self.activate(mode, i, None);
        }
    }

    pub fn playlist_ui(&mut self, ui: &mut Ui) {
        let mode = self.mode();
        let enabled = self.next_enabled();
        ui.horizontal(|ui| {
            let next = egui::Button::new(RichText::new(format!("Next · {}", mode_name(mode))).color(Color32::WHITE)).fill(ACCENT.gamma_multiply(0.7));
            let tip = if enabled { "Pick from the active PP/NPS/Skills tab. Auto copies a local song-select search; set rate manually.".to_string() } else { self.status_text.clone() };
            if ui.add_enabled(enabled, next).on_hover_text(tip).clicked() {
                self.next(false);
            }
            if ui.add_enabled(enabled, egui::Button::new("⏭ Skip")).on_hover_text("Skip: not this map (avoided for about 12 h)").clicked() {
                self.next(true);
            }
            if let Some(on) = self.tracking {
                if ui.selectable_label(on, if on { "Tracking on" } else { "Tracking off" }).on_hover_text("Paused plays stay excluded after imports and restarts. Resume for the next play.").clicked() {
                    self.tracking_toggled(!on);
                }
            }
        });
        ui.add_space(6.);
        let (lines, tip) = fmt::card_lines(self.card.as_deref(), self.card_is_pp(), self.card_note.as_deref(), &self.units);
        let rows = [1, 2, 1];
        theme::block(ui, &lines.into_iter().zip(rows).collect::<Vec<_>>(), tip.as_deref());
        if let Some((lines, tip)) = &self.result {
            theme::block(ui, &lines.iter().cloned().zip([1, 1, 2]).collect::<Vec<_>>(), Some(tip));
        }
        ui.add_space(6.);
        ui.horizontal(|ui| {
            for m in MODES {
                let active = !self.stats_open && m == mode;
                let tab = ui.selectable_label(active, mode_name(m));
                if tab.clicked() {
                    self.set_stats(false);
                    self.set_mode(m);
                    if active {
                        self.views.get_mut(m).expect("a tab").settings = false; // back to the list, like the cog
                    }
                }
                if active {
                    let view = self.views.get_mut(m).expect("a tab");
                    // Painted, not laid out: a glyph this big as a button would make the row taller (user 2026-11-01).
                    let (rect, gear) = ui.allocate_exact_size(egui::vec2(GEAR + 4., tab.rect.height()), egui::Sense::click());
                    gear.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Button, true, "⚙"));
                    let color = ui.style().interact_selectable(&gear, view.settings).text_color();
                    ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, "⚙", egui::FontId::proportional(GEAR), color);
                    if gear.on_hover_text("Settings for this tab").clicked() {
                        view.settings = !view.settings;
                    }
                }
            }
            if ui.selectable_label(self.stats_open, "Stats").clicked() {
                self.set_stats(true);
            }
        });
        ui.separator();
        // The one status line and bar for the whole window, at the bottom (user 2026-10-30: "Analysing" three times).
        let analysing = if self.stats_open || self.bg_progress.is_some() { None } else { self.views[mode].analysing.clone() };
        let paused = analysing.as_ref().filter(|_| self.live.is_some()).map_or("", |_| "paused while you play");
        let text: Vec<&str> = [self.status_text.as_str(), analysing.as_ref().map_or("", |a| a.0.as_str()), paused].into_iter().filter(|t| !t.is_empty()).collect();
        let text = text.join(" · ").replace('\n', " · ");
        let (done, total) = self.bg_progress.or(analysing.as_ref().map(|a| (a.1, a.2))).unwrap_or((0, 0));
        egui::Panel::bottom("rec-status").show_separator_line(false).show(ui, |ui| {
            if total > 0 {
                theme::bar(ui, done as f32 / total as f32);
            }
            if !text.is_empty() {
                theme::dim_small(ui, text);
            }
        });
        egui::CentralPanel::default().frame(egui::Frame::new()).show(ui, |ui| {
            if self.stats_open {
                self.stats_ui(ui);
            } else if self.views[mode].settings {
                self.settings_page(ui, mode);
            } else {
                self.list_page(ui, mode);
            }
        });
    }

    /// Only the player's resize (and the saved size) sets the window: content never does (user 2026-10-30, twice).
    pub fn track_playlist(&mut self, ctx: &egui::Context) {
        let (rect, outer, fullscreen, maximized, minimized) = ctx.input(|i| {
            let v = i.viewport();
            (v.inner_rect, v.outer_rect, v.fullscreen, v.maximized, v.minimized)
        });
        if fullscreen == Some(true) || maximized == Some(true) || minimized == Some(true) {
            return;
        }
        let (size, at) = (rect.map(|r| (r.width(), r.height())), outer.map(|o| (o.left(), o.top())));
        if let Some(size) = size.filter(|s| s.0 > 1. && *s != self.s.recs_win) {
            self.s.recs_win = size;
            self.request_save();
        }
        if at.is_some() && at != self.s.recs_pos {
            self.s.recs_pos = at;
            self.request_save();
        }
    }
}
