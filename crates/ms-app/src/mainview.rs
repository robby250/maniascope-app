//! The main window: glanceable difficulty of the selected map, kept small as an OBS source.
use crate::app::App;
use crate::fmt::{self, bold, plain, small};
use crate::theme::{self, ACCENT, BODY, DIM, TEXT};
use crate::viewer::Analysis;
use eframe::egui::{self, epaint::{text::{LayoutJob, TextFormat, TextWrapping}, Mesh, Vertex, WHITE_UV}, Align, Color32, FontId, Layout, RichText, Sense, Shape, TextStyle, Ui, Vec2};
use std::sync::Arc;

const BAR_FULL: f32 = 10.; // fixed scale for the skill bars: comparable across maps
const BAR_ROW: f32 = 18.;
const BAR_LABEL: f32 = 96.;  // the longest skill name
const BAR_VALUE: f32 = 40.;  // "10.00" and its gap
const HEAD: f32 = 52.; // three lines of skills, or the rating badge

fn strip(rect: egui::Rect, points: &[(f32, f32)], color: Color32) -> Shape {
    let mut m = Mesh::default();
    for (i, &(x, y)) in points.iter().enumerate() {
        m.vertices.push(Vertex { pos: egui::pos2(x, y), uv: WHITE_UV, color });
        m.vertices.push(Vertex { pos: egui::pos2(x, rect.bottom()), uv: WHITE_UV, color });
        if i > 0 {
            let n = (2 * i) as u32;
            m.indices.extend([n - 2, n - 1, n, n - 1, n + 1, n]);
        }
    }
    Shape::mesh(m)
}

fn shown_bars(a: &Analysis) -> impl Iterator<Item = &(String, f64)> {
    a.bars.iter().filter(|b| b.1 >= 0.05)
}

impl App {
    fn timeline(&self, ui: &mut Ui, a: &Arc<Analysis>, height: f32) {
        let (rect, response) = ui.allocate_exact_size(Vec2::new(ui.available_width(), height), Sense::hover());
        response.on_hover_text("Difficulty over time (2 s horizon of the rating's own demand curve, scaled to this map's peak). Fills while you play.");
        if a.timeline.is_empty() {
            return;
        }
        let step = (a.timeline.len() / (rect.width() as usize).max(1)).max(1); // one point per pixel at most
        let pts: Vec<f64> = a.timeline.chunks(step).map(|c| c.iter().copied().fold(f64::MIN, f64::max)).collect();
        let top = (pts.iter().copied().fold(0., f64::max) * 1.05).max(1e-9); // own peak: no dead band under the text
        let n = (pts.len() - 1).max(1) as f32;
        let points: Vec<(f32, f32)> = pts.iter().enumerate().map(|(i, v)| (rect.left() + i as f32 * rect.width() / n, rect.bottom() - rect.height() * (*v / top) as f32)).collect();
        let p = ui.painter_at(rect);
        p.add(strip(rect, &points, TEXT.gamma_multiply(0.35)));
        if let (Some(live), Some((t0, t1)), None) = (self.live, a.span, &self.local) {
            let f = (((live - t0) / (t1 - t0).max(1.)) as f32).clamp(0., 1.);
            let x = rect.left() + rect.width() * f;
            let mut done: Vec<(f32, f32)> = points.iter().copied().take_while(|q| q.0 <= x).collect();
            if let (Some(last), Some(next)) = (done.last().copied(), points.get(done.len())) {
                let t = (x - last.0) / (next.0 - last.0).max(1e-3);
                done.push((x, last.1 + (next.1 - last.1) * t));
            }
            let blue = Color32::from_rgba_unmultiplied(54, 133, 227, 230);
            p.add(strip(rect, &done, blue));
            p.rect_filled(egui::Rect::from_min_max(egui::pos2(x - 1., rect.top()), egui::pos2(x + 1., rect.bottom())), 0., Color32::from_rgb(54, 133, 227));
        }
    }

    /// Badge, skills, NPS, all centred on one fixed height: the timeline below keeps its size from map to map, and the NPS
    /// never pushes the skill rows apart (user 2026-11-01: the gap under the top skill).
    fn head(&self, ui: &mut Ui, a: &Arc<Analysis>) {
        let d = &self.s.display;
        let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), HEAD), Sense::hover());
        let mut ui = ui.new_child(egui::UiBuilder::new().max_rect(rect).layout(Layout::left_to_right(Align::Center)));
        if d.rating {
            let v = ms_calc::py::round_digits(a.overall, 2);
            let (text, pill) = fmt::rating_colours(v);
            theme::gradient(&mut ui, &format!("{v:.2}"), BODY * 2.4, None, &text, Some(pill))
                .on_hover_text("Overall difficulty: level a well-rounded player needs for controlled, accurate play. Provisional calibration — one decimal is already generous.");
            ui.add_space(4.);
        }
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            if let (true, Some(v)) = (d.nps, a.nps) {
                theme::gradient(ui, &format!("{v:.1}"), BODY * 1.9, Some((" NPS", BODY * 0.9)), &fmt::nps_colours(v), None).on_hover_text("Notes per second, first to last note, at this rate");
            }
            let first = a.entries.first().map_or(1., |e| e.rating);
            // Type size and opacity shrink with distance from the top entry.
            let format = |e: &crate::viewer::Entry| {
                let r = e.rating / first;
                let f = ((r - 0.3) / 0.7).min(1.) as f32;
                let family = if r >= 0.85 { egui::FontFamily::Name("bold".into()) } else { egui::FontFamily::Proportional };
                TextFormat { font_id: FontId::new(BODY * (0.70 + 0.55 * f), family), color: TEXT.gamma_multiply(0.55 + 0.45 * f), ..TextFormat::default() }
            };
            let width = ui.available_width();
            let row = |entries: &[crate::viewer::Entry]| {
                let mut j = LayoutJob::default();
                for (i, e) in entries.iter().enumerate() {
                    j.append(&format!("{} {:.1}", e.name, e.rating), if i > 0 { 8. } else { 0. }, format(e));
                }
                j.wrap = TextWrapping::truncate_at_width(width);
                j
            };
            let top = a.entries.first().map(|e| row(std::slice::from_ref(e)));
            let rest = row(a.entries.get(1..d.skills.min(a.entries.len())).unwrap_or(&[]));
            let tags = (!a.tags.is_empty()).then(|| a.tags.join(" · "));
            let lines = ui.fonts_mut(|f| {
                let mut h = top.clone().map_or(BODY * 1.2, |j| f.layout_job(j).size().y);
                h += f.layout_job(rest.clone()).size().y;
                h + tags.as_ref().map_or(0., |_| BODY * 1.2)
            });
            ui.allocate_ui_with_layout(ui.available_size(), Layout::top_down(Align::Min), |ui| {
                ui.spacing_mut().item_spacing.y = 0.;
                ui.add_space(((HEAD - lines) / 2.).max(0.));
                match (top, a.entries.first()) {
                    (Some(job), Some(_)) => {
                        let mut tip: Vec<String> = a.entries.iter().filter(|e| e.parts.len() > 1).map(|e| format!("{}: {}", e.name, e.parts.iter().map(|p| a.names.iter().find(|n| n.0 == *p).map_or(p.as_str(), |n| n.1.as_str())).collect::<Vec<_>>().join(" + "))).collect();
                        if a.entries.iter().any(|e| e.parts.iter().any(|p| p == "technical")) {
                            tip.push("Tech passages (song time):".into());
                            tip.extend(a.tech.iter().map(|w| format!("  {}:{:04.1}  pattern {:.0}% / rhythm {:.0}%{}", (w.t / 60.) as i64, w.t % 60., 100. * w.pattern, 100. * w.rhythm, if w.why.is_empty() { String::new() } else { format!(" — {}", w.why) })));
                        }
                        let r = ui.label(job);
                        if !tip.is_empty() {
                            r.on_hover_text(tip.join("\n"));
                        }
                    }
                    _ => { ui.label("too sparse to characterise"); }
                }
                ui.label(rest);
                if let Some(t) = tags {
                    theme::dim(ui, t);
                }
            });
        });
    }

    /// One row per skill the map has: zeros said nothing and pushed the footer off the window.
    fn details(&self, ui: &mut Ui, a: &Arc<Analysis>) {
        let rows: Vec<&(String, f64)> = shown_bars(a).collect();
        let (area, hover) = ui.allocate_exact_size(Vec2::new(ui.available_width(), rows.len() as f32 * BAR_ROW), Sense::hover());
        let p = ui.painter();
        let font = egui::FontId::proportional(theme::BODY);
        for (i, (name, value)) in rows.into_iter().enumerate() {
            let row = egui::Rect::from_min_size(area.min + Vec2::new(0., i as f32 * BAR_ROW), Vec2::new(area.width(), BAR_ROW));
            p.text(row.left_center(), egui::Align2::LEFT_CENTER, name, font.clone(), theme::TEXT);
            p.text(row.right_center(), egui::Align2::RIGHT_CENTER, format!("{value:.2}"), font.clone(), theme::TEXT);
            let track = egui::Rect::from_min_max(egui::pos2(row.left() + BAR_LABEL, row.center().y - 3.), egui::pos2(row.right() - BAR_VALUE, row.center().y + 3.));
            p.rect_filled(track, 3., theme::BUTTON);
            let mut fill = track;
            fill.set_width(track.width() * (*value as f32 / BAR_FULL).clamp(0., 1.));
            p.rect_filled(fill, 3., ACCENT);
        }
        hover.on_hover_text(format!("Each skill is the rating of the sections that show it, on the same scale as the overall number: a section can count toward several skills (Delay and Technical, Chordstream and Bracket); overall charges its work once. Bars: fixed 0–{BAR_FULL}, comparable across maps. Stamina is the hardest 4 minutes, a shorter map counted with rest."));
    }

    fn context_line(&self, a: &Arc<Analysis>) -> String {
        let (_, _, known) = self.effective();
        let mode = if self.local.is_some() { " local file" } else if known { "" } else { " (osu! rate unknown)" };
        let dan = self.s.display.dan.then(|| ms_rec::dans::label(a.keys, a.overall, a.ln_notes / a.notes.max(1) as f64, Some(&a.dan_vibro))).flatten();
        let mut text = fmt::context(a, dan.as_deref(), mode);
        // Chart length, or time left (real seconds at this rate) in play.
        if let Some((t0, t1)) = a.span {
            let playing = self.live.filter(|_| self.local.is_none());
            let left = (t1 - playing.unwrap_or(t0).max(t0)).max(0.) / 1000. / a.rate;
            text += &format!("  ·  {}{}", fmt::clock(left), if playing.is_some() { " left" } else { "" });
        }
        text
    }

    pub fn main_ui(&mut self, ui: &mut Ui) {
        ui.spacing_mut().item_spacing.y = 2.;
        ui.horizontal(|ui| {
            if ui.add(egui::Button::new("☰  Playlist").selected(self.s.recs_open)).on_hover_text("Show or hide the playlist window (Next, Skip, Tracking, PP/NPS/Skills lists)").clicked() {
                self.s.recs_open = !self.s.recs_open;
                self.request_save();
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("⋯").on_hover_text("Settings").clicked() {
                    self.open.menu = !self.open.menu;
                }
            });
        });
        ui.add_space(4.);
        let analysis = self.analysis().cloned();
        let foot = 22.;
        let rows = analysis.as_ref().map_or(0, |a| shown_bars(a).count()) as f32;
        let detail_height = if self.s.details { rows * BAR_ROW + 6. } else { 0. };
        if let Some(a) = &analysis {
            theme::line(ui, &vec![bold(if a.title.is_empty() { "?" } else { &a.title }), plain("  "), small(a.artist.clone())], 1);
            ui.horizontal(|ui| {
                theme::dim(ui, format!("[{}]  {}", a.version, a.creator));
                if self.s.display.ln {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        theme::dim_small(ui, format!("LN {:.0}%", 100. * a.ln_notes / a.notes.max(1) as f64)).on_hover_text("Long-note heads / all note heads. Each LN counts once; tails are not extra notes.");
                    });
                }
            });
            let context = self.context_line(a);
            theme::dim(ui, &context).on_hover_text(context);
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.set_width(ui.available_width() - 26.);
                    self.head(ui, a);
                });
                let arrow = if self.s.details { "▴" } else { "▾" };
                if ui.add(egui::Button::new(arrow).frame(false)).on_hover_text("Skill details").clicked() {
                    self.toggle_details(ui.ctx());
                }
            });
            if self.s.display.timeline {
                self.timeline(ui, a, (ui.available_height() - foot - detail_height).max(30.));
            }
            if self.s.details {
                self.details(ui, a);
            }
        } else if let Some(p) = &self.pending {
            theme::line(ui, &vec![bold(p.title.clone()), plain("  "), small(p.artist.clone())], 1);
            theme::dim(ui, format!("[{}]  {}", p.version, p.creator));
            theme::dim(ui, p.context.clone());
            ui.horizontal(|ui| {
                ui.label(RichText::new("…").size(BODY * 3.).color(Color32::from_gray(0x55)));
                ui.label("Calculating…");
            });
        } else {
            let text = self.shown.as_ref().and_then(|s| s.result.as_ref().err().cloned()).unwrap_or_else(|| "Select a mania map in osu!".to_string());
            if let Some(s) = &self.shown {
                ui.label(s.path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default());
            }
            ui.horizontal(|ui| {
                ui.label(RichText::new("–").size(BODY * 3.).color(Color32::from_gray(0x55)));
                ui.label(text);
            });
        }
        ui.with_layout(egui::Layout::bottom_up(egui::Align::Min), |ui| {
            ui.horizontal(|ui| {
                let foot = self.foot.clone();
                if self.local.is_some() && ui.button("Follow osu!").clicked() {
                    self.local = None;
                    self.refresh();
                }
                if let Some(c) = self.card.as_ref().filter(|c| c.sure) {
                    ui.label(RichText::new(format!("Expected ~{:.1}%", 100. * self.units.acc(c.acc_mid))).text_style(TextStyle::Small)).on_hover_text("Your expected accuracy on this map at this rate, from your own plays");
                }
                ui.add(egui::Label::new(RichText::new(&foot).color(DIM).text_style(TextStyle::Small)).truncate()).on_hover_text(foot);
            });
        });
    }

    /// Details grow the window; collapsing returns to the compact size rather than letting the timeline take the rows' space.
    fn toggle_details(&mut self, ctx: &egui::Context) {
        self.s.details = !self.s.details;
        let (w, h) = if self.s.details { self.s.details_win } else { self.s.win };
        self.resized = Some((w, h));
        ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(Vec2::new(w, h)));
        self.request_save();
    }

    /// Remember the sizes and places the player gave the window. `resized`: our own request has not landed yet.
    pub fn track_window(&mut self, ctx: &egui::Context) {
        let (rect, outer, fullscreen, maximized, minimized) = ctx.input(|i| {
            let v = i.viewport();
            (v.inner_rect, v.outer_rect, v.fullscreen, v.maximized, v.minimized)
        });
        if fullscreen == Some(true) || maximized == Some(true) || minimized == Some(true) {
            return;
        }
        if let Some(r) = rect {
            let size = (r.width(), r.height());
            if self.resized.take_if(|w| (w.0 - size.0).abs() < 2. && (w.1 - size.1).abs() < 2.).is_none() && self.resized.is_none() && size.0 > 1. {
                let slot = if self.s.details { &mut self.s.details_win } else { &mut self.s.win };
                if *slot != size {
                    *slot = size;
                    self.request_save();
                }
            }
        }
        if let Some(o) = outer {
            let p = Some((o.left(), o.top()));
            if self.s.win_pos != p {
                self.s.win_pos = p;
                self.request_save();
            }
        }
    }
}
