//! The player profile: rendering only, data comes from ms_rec::stats.
use crate::app::App;
use crate::theme::{self, ACCENT, DIM, TEXT};
use eframe::egui::{self, pos2, vec2, Align2, Color32, FontId, Rect, RichText, ScrollArea, Sense, Stroke, Ui};
use ms_model::skills::{self, Skill};
use ms_rec::catalog::SUPPORTED_KEYS;
use ms_rec::dans;
use ms_rec::stats::{FormHistory, Group, Page, TARGET};

const MONTHS: [&str; 12] = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const FORM: [(&str, Color32); 4] = [("warm-up", Color32::from_rgb(89, 168, 230)), ("peak", Color32::from_rgb(87, 194, 110)), ("bad", Color32::from_rgb(230, 84, 74)), ("fading", Color32::from_rgb(230, 161, 59))];

fn month_name(m: i64) -> String {
    format!("{} {}", MONTHS[m.rem_euclid(12) as usize], 2000 + m.div_euclid(12))
}

/// [(level, short name)] of the keymode's main dan series inside lo..hi: the level matching each dan (its rating / the
/// page's dan scale).
fn dan_ticks(keys: i64, lo: f64, hi: f64, factor: f64) -> Vec<(f64, String)> {
    let Some(series) = dans::series(keys, "rice") else { return Vec::new() };
    let prefix = dans::short(series).trim();
    dans::tiers(series).map_or_else(Vec::new, |t| t.iter().map(|(tier, v)| (v.exp() / factor, tier)).filter(|x| lo <= x.0 && x.0 <= hi).map(|(v, tier)| {
        let name = tier.replace(prefix, "");
        (v, if name.trim().is_empty() { tier.clone() } else { name.trim().to_string() })
    }).collect())
}

/// Shared stars → x mapping so every skill row and the dan axis line up.
#[derive(Clone, Copy)]
struct Scale {
    lo: f64,
    hi: f64,
    dan: f64,
}

impl Scale {
    fn x(&self, v: f64, r: Rect) -> f32 {
        r.left() + 6. + (r.width() - 12.) * (((v - self.lo) / (self.hi - self.lo)).clamp(0., 1.)) as f32
    }
}

fn small_font() -> FontId {
    FontId::proportional(9.)
}

fn date(t: f64, f: &str) -> String {
    ms_data::store::local_format(t, f)
}

fn dan_axis(ui: &mut Ui, keys: i64, scale: Scale, width: f32) {
    let (r, _) = ui.allocate_exact_size(vec2(width, 18.), Sense::hover());
    let p = ui.painter();
    let mut last = f32::MIN;
    for (stars, name) in dan_ticks(keys, scale.lo, scale.hi, scale.dan) {
        let x = scale.x(stars, r);
        let g = p.layout_no_wrap(name, small_font(), TEXT.gamma_multiply(0.55));
        let left = (x - g.size().x / 2.).max(r.left()).min(r.right() - g.size().x); // edge names stay whole
        if left < last + 4. {
            continue; // crowded tiers: keep every other name readable
        }
        last = left + g.size().x;
        p.galley(pos2(left, r.top()), g, TEXT);
        p.vline(x, (r.top() + 13.)..=r.bottom(), Stroke::new(1., TEXT.gamma_multiply(0.55)));
    }
}

/// ● your level, a faint band for its uncertainty, ○ this session (only once warm).
fn skill_bar(ui: &mut Ui, keys: i64, g: Option<&Group>, scale: Scale, session: bool, width: f32) -> egui::Response {
    let (r, response) = ui.allocate_exact_size(vec2(width, 22.), Sense::hover());
    let p = ui.painter();
    let y = r.center().y;
    let ink = TEXT.gamma_multiply(0.10);
    p.hline((r.left() + 6.)..=(r.right() - 6.), y, Stroke::new(2., ink));
    for (stars, _) in dan_ticks(keys, scale.lo, scale.hi, scale.dan) {
        p.vline(scale.x(stars, r), (y - 4.)..=(y + 4.), Stroke::new(1., ink));
    }
    let Some(g) = g else { return response };
    let Some(value) = g.value else { return response };
    let (lo, hi) = (scale.x(g.low, r), scale.x(g.high, r));
    p.rect_filled(Rect::from_min_max(pos2(lo, y - 5.), pos2(hi.max(lo + 1.), y + 5.)), 0., ACCENT.gamma_multiply(0.25));
    p.circle_filled(pos2(scale.x(value, r), y), 5.5, ACCENT);
    if let (true, Some(c)) = (session, g.current) {
        p.circle_stroke(pos2(scale.x(c, r), y), 6.5, Stroke::new(2., TEXT.gamma_multiply(0.95)));
    }
    response
}

/// Overall level per month, dan lines behind it, peak and today marked.
fn history_chart(ui: &mut Ui, keys: i64, h: &[(i64, f64)], peak: (i64, f64), dan: f64) {
    let (r, _) = ui.allocate_exact_size(vec2(ui.available_width(), 110.), Sense::hover());
    let values: Vec<f64> = h.iter().map(|x| x.1).collect();
    let lo = values.iter().copied().fold(f64::MAX, f64::min) * 0.97;
    let hi = values.iter().copied().fold(f64::MIN, f64::max) * 1.03;
    let (m0, m1) = (h[0].0, h[h.len() - 1].0);
    let x = |m: i64| r.left() + 8. + (r.width() - 82.) * ((m - m0) as f32 / (m1 - m0).max(1) as f32);
    let y = |v: f64| r.bottom() - 16. - (r.height() - 24.) * ((v - lo) / (hi - lo).max(1e-6)) as f32;
    let p = ui.painter();
    let mut ticks = dan_ticks(keys, lo, hi, dan);
    ticks.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut last: Option<f32> = None;
    for (stars, name) in ticks {
        if last.is_some_and(|l| y(stars) - l < 11.) {
            continue; // crowded low tiers: every other line stays readable
        }
        last = Some(y(stars));
        p.hline((r.left() + 8.)..=(r.right() - 68.), y(stars), Stroke::new(1., TEXT.gamma_multiply(0.12)));
        p.text(pos2(r.right() - 62., y(stars)), Align2::LEFT_CENTER, name, small_font(), TEXT.gamma_multiply(0.55));
    }
    p.add(egui::Shape::line(h.iter().map(|&(m, v)| pos2(x(m), y(v))).collect(), Stroke::new(2., ACCENT)));
    p.circle_stroke(pos2(x(peak.0), y(peak.1)), 4., Stroke::new(2., ACCENT));
    let end = h[h.len() - 1];
    p.circle_filled(pos2(x(end.0), y(end.1)), 4., ACCENT);
    p.text(pos2(x(m0), r.bottom() - 3.), Align2::LEFT_BOTTOM, month_name(m0), small_font(), TEXT.gamma_multiply(0.55));
    p.text(pos2(x(m1), r.bottom() - 3.), Align2::RIGHT_BOTTOM, month_name(m1), small_font(), TEXT.gamma_multiply(0.55));
}

/// The last sessions play by play, against the rested expectation; hover a dot for its session and mark.
fn form_chart(ui: &mut Ui, form: &FormHistory) {
    let (mut dots, mut gaps, mut count) = (Vec::new(), Vec::new(), 0usize);
    for s in &form.sessions {
        gaps.push(count);
        let day = date(s.t, "%d %b %H:%M");
        for &(clock, value, mark) in &s.plays {
            dots.push((count, value, mark, format!("{day} · {clock:.0} min in · {value:+.1}%{}", if mark.is_empty() { String::new() } else { format!(" · {mark}") })));
            count += 1;
        }
        count += 2;
    }
    let top = dots.iter().map(|d| d.1.abs()).fold(0.5, f64::max);
    let (r, response) = ui.allocate_exact_size(vec2(ui.available_width(), 90.), Sense::hover());
    let x = |i: usize| r.left() + 6. + (r.width() - 12.) * i as f32 / count.max(1) as f32;
    let y = |v: f64| r.center().y - (r.height() / 2. - 6.) * (v / top) as f32;
    let p = ui.painter();
    let ink = Stroke::new(1., TEXT.gamma_multiply(0.15));
    p.hline(r.x_range(), y(0.), ink);
    for start in gaps.iter().skip(1) {
        p.vline(x(start - 1), (r.top() + 2.)..=(r.bottom() - 2.), ink);
    }
    for (i, value, mark, _) in &dots {
        match FORM.iter().find(|f| f.0 == *mark) {
            Some(f) => p.circle_filled(pos2(x(*i), y(*value)), 3., f.1),
            None => p.circle_filled(pos2(x(*i), y(*value)), 2., TEXT.gamma_multiply(0.45)),
        };
    }
    if let Some(at) = response.hover_pos() {
        if let Some(dot) = dots.iter().min_by(|a, b| (x(a.0) - at.x).abs().total_cmp(&(x(b.0) - at.x).abs())) {
            response.on_hover_text_at_pointer(&dot.3);
        }
    }
}

struct Layout {
    name: f32,
    tail: f32,
    bar: f32,
}

impl App {
    fn skill_row(&mut self, ui: &mut Ui, keys: i64, node: &Skill, page: &Page, scale: Scale, l: &Layout, indent: bool, toggle: Option<&mut bool>) {
        let row = page.groups.get(&node.key);
        let value = row.and_then(|g| g.value);
        let dan = value.and_then(|v| dans::ability(keys, v, &node.key, &page.dan_scale));
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 8.;
            ui.scope(|ui| {
                ui.set_width(l.name);
                ui.add_space(if indent { 22. } else { 0. });
                let label = match &toggle {
                    Some(open) => ui.selectable_label(false, format!("{} {}", if **open { "▾" } else { "▸" }, node.name)),
                    None => ui.label(node.name),
                };
                if label.clicked() {
                    if let Some(open) = toggle {
                        *open = !*open;
                    }
                }
                if !node.tip.is_empty() {
                    label.on_hover_text(node.tip);
                }
            });
            let bar = skill_bar(ui, keys, row, scale, page.session, l.bar);
            if let Some(g) = row {
                if let Some(v) = g.value {
                    let examples: Vec<String> = g.examples.iter().map(|(t, r, d)| format!("{t} · {r:.2}× · difficulty {d:.1}")).collect();
                    bar.on_hover_text(format!("Level {v:.2}; likely between {:.2} and {:.2}. {} plays on {} charts.{}", g.low, g.high, g.plays, g.maps, if examples.is_empty() { String::new() } else { format!("\n{}", examples.join("\n")) }));
                } else if g.extrapolated {
                    bar.on_hover_text(format!("Evidence reaches difficulty {:.1}; the fitted limit is not measured.", g.tested_to));
                }
            }
            ui.scope(|ui| {
                ui.set_width(l.tail);
                ui.horizontal(|ui| {
                    match value {
                        Some(v) => {
                            ui.label(RichText::new(format!("{v:.1}")).family(egui::FontFamily::Name("bold".into())));
                            if let Some(d) = &dan {
                                theme::dim_small(ui, format!("≈ {d}"));
                            }
                        }
                        None => { ui.label("—"); }
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.add(egui::Button::new("Practice").frame(false)).on_hover_text(format!("Practice {} in {keys}K", node.name)).clicked() {
                            self.practice_from_stats(keys, &node.key);
                        }
                    });
                });
            });
        });
        let confidence = row.map_or("Not enough evidence", |g| g.confidence);
        let maps = row.map_or(0, |g| g.maps);
        let session = row.filter(|g| page.session && g.value.is_some()).and_then(|g| g.current).map_or(String::new(), |c| format!(" · this session {c:.1}"));
        ui.horizontal(|ui| {
            ui.add_space(l.name + 8.);
            theme::dim_small(ui, format!("{confidence} · {} charts{session}", ms_rec::engine::commas(maps)));
        });
    }

    fn page_ui(&mut self, ui: &mut Ui, keys: i64, page: &Page) {
        let groups = &page.groups;
        let overall = groups.get("overall");
        let value = overall.and_then(|g| g.value);
        let dan_scale = &page.dan_scale;
        ui.horizontal(|ui| match value {
            Some(v) => {
                ui.label(RichText::new(format!("{v:.1}")).size(theme::BODY * 2.2).family(egui::FontFamily::Name("bold".into())));
                ui.label(RichText::new(format!("your {keys}K level")).family(egui::FontFamily::Name("bold".into())));
                if let Some(d) = dans::ability(keys, v, "overall", dan_scale) {
                    ui.label(format!("·  ≈ {d}"));
                }
            }
            None => { ui.label(RichText::new(format!("{keys}K · not enough history yet")).family(egui::FontFamily::Name("bold".into()))); }
        });
        let peak = page.peak;
        // Months with too few plays leave the history: its last month is not "now" after a break (user 2026-10-30).
        let stale = page.history.last().is_some_and(|h| page.now - h.0 as f64 > 2.);
        let mut lines = Vec::new();
        if let (Some(v), Some(p)) = (value, peak) {
            if p.1 > v + 0.05 {
                lines.push(format!("Peak {:.1} in {}{}", p.1, month_name(p.0), dans::ability(keys, p.1, "overall", dan_scale).map_or(String::new(), |d| format!(" (≈ {d})"))));
            } else if !stale {
                lines.push("This is your peak so far.".to_string());
            }
        }
        if let (Some(_), true, Some(last)) = (value, stale, page.history.last()) {
            lines.push(format!("Last measured in {}: too few {keys}K plays since to tell your form now.", month_name(last.0)));
        }
        match overall.and_then(|g| g.current).filter(|_| value.is_some() && page.session) {
            Some(c) => lines.push(format!("This session: {c:.1} (○ on the bars)")),
            None if value.is_some() => lines.push("Not warmed up: the first maps of a session play lower than this, so no session number is shown until you have played a few.".to_string()),
            None => {}
        }
        if !lines.is_empty() {
            ui.label(lines.join("\n"));
        }
        let target = format!("{:.3}", 100. * self.units.acc(TARGET));
        let target = format!("{}%", target.trim_end_matches('0').trim_end_matches('.'));
        let rice = dan_scale.get("rice").copied().unwrap_or(1.);
        if page.history.len() >= 2 {
            history_chart(ui, keys, &page.history, peak.unwrap_or(page.history[page.history.len() - 1]), rice);
        }
        if let Some(form) = page.form.as_ref().filter(|f| !f.sessions.is_empty()) {
            ui.label("Form, play by play (last sessions)");
            form_chart(ui, form);
            ui.horizontal(|ui| {
                for (name, colour) in FORM {
                    ui.label(RichText::new("●").color(colour).size(theme::SMALL));
                    theme::dim_small(ui, name);
                }
            });
            let fatigue = if form.fatigue != 0. { format!("you lose about {:.1}% an hour.", form.fatigue.abs()) } else { "no drop shows yet.".to_string() };
            theme::dim_small(ui, format!("The first minutes of a session cost up to {:.1}%. After 2 hours {fatigue}", form.warmup.abs()));
        }
        let (local, public, newest) = page.evidence;
        let c = ms_rec::engine::commas;
        let based = format!("{} local plays · {} website bests{}", c(local), c(public), if newest != 0. { format!(" · newest {}", date(newest, "%Y-%m-%d")) } else { String::new() });
        theme::dim(ui, format!("{based}. {} charts · {} plays. Level = the chart difficulty you play at ~{target}, from your own recent completed plays. ≈ dan: the hardest dan whose songs you would play at 95% on your better half of plays, after grinding (one song, not passing the course).", c(page.maps), c(page.plays)));
        let measured: Vec<&Group> = groups.values().filter(|g| g.value.is_some()).collect();
        let ends = peak.map(|p| p.1);
        let lo = measured.iter().map(|g| g.low).chain(ends).fold(f64::MAX, f64::min);
        let hi = measured.iter().map(|g| g.high).chain(ends).fold(f64::MIN, f64::max);
        let (lo, hi) = if lo == f64::MAX { (0.8, 10.2) } else { (lo - 0.2, hi + 0.2) };
        let scale = Scale { lo, hi: hi.max(lo + 0.5), dan: rice };
        // Every row's bar starts and ends at the same x, under the dan axis.
        let width = ui.available_width();
        let l = Layout { name: 110., tail: 150., bar: (width - 110. - 150. - 16.).max(80.) };
        theme::dim(ui, format!("● level   band: uncertainty{}", if page.session { "   ○ this session" } else { "" }));
        ui.horizontal(|ui| {
            ui.add_space(l.name + 8.);
            dan_axis(ui, keys, scale, l.bar);
        });
        let parents = skills::tree(keys);
        for parent in parents {
            if parent.children.is_empty() {
                self.skill_row(ui, keys, parent, page, scale, &l, false, None);
                continue;
            }
            let id = (keys, parent.key.clone());
            let mut open = self.expanded.contains(&id);
            self.skill_row(ui, keys, parent, page, scale, &l, false, Some(&mut open));
            if open {
                self.expanded.insert(id);
                for child in &parent.children {
                    self.skill_row(ui, keys, child, page, scale, &l, true, None);
                }
            } else {
                self.expanded.remove(&id);
            }
        }
        let reliable: Vec<(&str, &Group)> = parents.iter().filter_map(|p| groups.get(&p.key).filter(|g| g.value.is_some() && g.maps >= 10).map(|g| (p.name, g))).collect();
        if value.is_some() {
            let by = |a: &&(&str, &Group), b: &&(&str, &Group)| a.1.value.total_cmp_opt(&b.1.value);
            if let (Some(high), Some(low)) = (reliable.iter().max_by(by), reliable.iter().min_by(by)) {
                if high.1.low > low.1.high {
                    ui.label(format!("Profile contrast: {} is better supported at higher difficulty than {}.", high.0, low.0));
                }
            }
        }
        egui::CollapsingHeader::new("Prediction strengths & blind spots").show(ui, |ui| {
            let mut messages: Vec<String> = page.diagnostics.iter().map(|(skill, delta, maps)| format!("{}: {:.1} accuracy points {} the saved pre-play predictions across {maps} charts.", skills::name(skill), self.units.points(*delta).abs(), if *delta > 0. { "above" } else { "below" })).collect();
            if messages.is_empty() {
                messages.push("No repeatable large prediction bias supported by enough distinct tracked charts yet.".to_string());
            }
            messages.push("These are model diagnostics, not proof of a personal weakness. Mixed patterns, session form and older predictor versions may contribute.".to_string());
            theme::dim(ui, messages.join("\n\n"));
        });
        if page.mash_maps > 0 {
            theme::dim(ui, format!("Special-pattern evidence: Mash appears materially in {} charts. It is an execution/difficulty correction, not a separate practice goal.", page.mash_maps));
        }
    }

    pub fn stats_ui(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            let label = self.stats_keys.map_or("All keymodes".to_string(), |k| format!("{k}K"));
            egui::ComboBox::from_id_salt("stats-keys").selected_text(label).show_ui(ui, |ui| {
                ui.selectable_value(&mut self.stats_keys, None, "All keymodes");
                for k in SUPPORTED_KEYS {
                    ui.selectable_value(&mut self.stats_keys, Some(k), format!("{k}K"));
                }
            });
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(RichText::new("PLAYER STATS").color(DIM));
            });
        });
        if let Some((text, done, total)) = self.stats_wait.clone() {
            theme::dim(ui, text);
            if total > 0 {
                theme::bar(ui, done as f32 / total as f32);
            }
        }
        ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            ui.spacing_mut().item_spacing.y = 8.;
            let Some(data) = self.stats.take() else {
                ui.label("Reading your score history… (it appears here as soon as your plays are read and analysed; progress is shown above)");
                return;
            };
            match self.stats_keys {
                None => {
                    ui.label("Your keymodes");
                    theme::dim(ui, "Each rating uses its own keymode's difficulty scale. There is no cross-keymode average.");
                    for k in SUPPORTED_KEYS {
                        let page = data.pages.get(&k);
                        let value = page.and_then(|p| p.groups.get("overall")).and_then(|g| g.value).map_or("—".to_string(), |v| format!("{v:.1}"));
                        let (maps, plays) = page.map_or((0, 0), |p| (p.maps, p.plays));
                        if ui.add(egui::Button::new(format!("{k}K     {value}    ·    {} charts / {} plays", ms_rec::engine::commas(maps), ms_rec::engine::commas(plays))).frame(false)).clicked() {
                            self.stats_keys = Some(k);
                        }
                    }
                }
                Some(k) => {
                    if let Some(page) = data.pages.get(&k) {
                        self.page_ui(ui, k, page);
                    }
                }
            }
            if self.stats.is_none() { // a practice click or a newer profile replaced it meanwhile
                self.stats = Some(data);
            }
        });
    }
}

trait Cmp {
    fn total_cmp_opt(&self, other: &Self) -> std::cmp::Ordering;
}

impl Cmp for Option<f64> {
    fn total_cmp_opt(&self, other: &Self) -> std::cmp::Ordering {
        self.unwrap_or(0.).total_cmp(&other.unwrap_or(0.))
    }
}
