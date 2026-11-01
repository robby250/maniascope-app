//! ManiaScope: glanceable osu!mania difficulty for the selected map, and what to play next.
#![cfg_attr(windows, windows_subsystem = "windows")]
mod app;
mod fmt;
mod mainview;
mod playlist;
mod public;
mod settings;
mod statsview;
mod theme;
mod viewer;
mod watcher;
mod windows;

use app::{App, Msg};
use eframe::egui::{self, ViewportBuilder};
use ms_data::paths::cache;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};

const INSTANCE_PORT: u16 = 24051; // a second launch raises the first instance's window

const ICON: &[u8] = include_bytes!("../../../assets/icon.png");

/// Linux has no installer: this puts the running exe in the application menu.
fn install_desktop() -> i32 {
    let (Ok(exe), Some(home)) = (std::env::current_exe(), std::env::var_os("HOME")) else { return 1 };
    // The XDG spec ignores an empty or relative value.
    let apps = std::env::var_os("XDG_DATA_HOME").map(std::path::PathBuf::from).filter(|p| p.is_absolute()).unwrap_or_else(|| std::path::Path::new(&home).join(".local/share"));
    let (icon, entry) = (apps.join("icons/maniascope.png"), apps.join("applications/maniascope.desktop"));
    let write = || -> std::io::Result<()> {
        std::fs::create_dir_all(apps.join("icons"))?;
        std::fs::create_dir_all(apps.join("applications"))?;
        std::fs::write(&icon, ICON)?;
        std::fs::write(&entry, format!(include_str!("../../../assets/maniascope.desktop"), exe = exe.display(), icon = icon.display()))
    };
    match write() {
        Ok(()) => { println!("{}", entry.display()); 0 }
        Err(e) => { eprintln!("{e}"); 1 }
    }
}

fn lifecycle(message: &str) {
    println!("[{}] viewer pid={} {message}", ms_data::store::local_stamp(), std::process::id());
}

/// The windowed exe has no console: prints, panics and native warnings go to maniascope.log (previous run:
/// maniascope.prev.log) in the cache folder.
fn log_to_file() {
    let dir = cache();
    let log = dir.join("maniascope.log");
    let _ = std::fs::create_dir_all(&dir);
    let _ = std::fs::rename(&log, dir.join("maniascope.prev.log"));
    let Ok(file) = std::fs::OpenOptions::new().create(true).append(true).open(&log) else { return };
    #[cfg(unix)]
    {
        use std::os::fd::AsRawFd;
        extern "C" {
            fn dup2(old: i32, new: i32) -> i32;
        }
        for fd in [1, 2] {
            // SAFETY: both descriptors are open; dup2 replaces the standard stream for the whole process.
            unsafe { dup2(file.as_raw_fd(), fd) };
        }
    }
    #[cfg(windows)]
    {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::System::Console::{SetStdHandle, STD_ERROR_HANDLE, STD_OUTPUT_HANDLE};
        // SAFETY: the handle stays open for the process' life (`forget` below).
        unsafe {
            SetStdHandle(STD_OUTPUT_HANDLE, file.as_raw_handle() as _);
            SetStdHandle(STD_ERROR_HANDLE, file.as_raw_handle() as _);
        }
    }
    std::mem::forget(file);
    println!("[{}] ManiaScope {} calc {}, {} {}, {}", ms_data::store::local_stamp(), env!("CARGO_PKG_VERSION"), ms_rec::engine::CALC_ID, std::env::consts::OS, std::env::consts::ARCH, std::env::current_exe().map(|p| p.display().to_string()).unwrap_or_default());
}

fn crash_log() {
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let text = format!("\n=== {} @ {} ===\n{info}\n{}\n", std::thread::current().name().unwrap_or("?"), ms_data::store::local_stamp(), std::backtrace::Backtrace::force_capture());
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(cache().join("crash.log")) {
            let _ = f.write_all(text.as_bytes());
        }
        default(info);
    }));
}

/// CI/support check without a game: calculator identity, a full analysis, dan table.
fn self_test() -> i32 {
    let data = include_bytes!("../../../packaging/selftest.osu");
    let chart = ms_calc::chart::parse(data).expect("self-test chart");
    let prm = ms_calc::features::standard_params();
    let res = ms_calc::compute::compute(&chart, 1., true, &prm).expect("analysis");
    let overall = res.dict("scores").expect("scores").f("overall");
    assert!(1. < overall && overall < 20., "{overall}");
    assert!(ms_rec::dans::tiers("7K Regular").is_some(), "dan table missing or built by another calculator");
    println!("self-test OK {} {overall:.2}", ms_rec::engine::CALC_ID);
    0
}

fn webmaps(args: &[String]) -> i32 {
    let command = args.first().map_or("", String::as_str);
    let value = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1));
    let keys = value("--keys").map_or_else(Default::default, |k| k.split(',').filter_map(|x| x.parse().ok()).collect());
    let stars = value("--stars").and_then(|s| { let (a, b) = s.split_once('-')?; Some((a.parse().ok()?, b.parse().ok()?)) });
    let prm = ms_calc::features::standard_params();
    match ms_data::webmaps::crawl(&keys, stars, command == "check", false, None, ms_rec::engine::CALC_ID, &prm) {
        Ok(()) => 0,
        Err(ms_data::webmaps::CrawlError::Stopped) => 0,
        Err(ms_data::webmaps::CrawlError::Failed(e)) => {
            eprintln!("{e}");
            1
        }
    }
}

struct Shell {
    app: Option<App>,
    error: Option<String>,
}

impl eframe::App for Shell {
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.app.is_none() && self.error.is_none() {
            theme::apply(ctx);
            match App::new(ctx) {
                Ok(app) => {
                    lifecycle("opened");
                    listen(app.mail.clone());
                    self.app = Some(app);
                }
                Err(e) => self.error = Some(e), // otherwise the process lingers with no window (Windows friends, v0.3.8)
            }
        }
        if let Some(app) = &mut self.app {
            app.pump(ctx);
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        let Some(app) = &mut self.app else {
            egui::CentralPanel::default().show(ui, |ui| {
                ui.label("ManiaScope could not start");
                ui.label(format!("{}\nPlease send the log files in\n{}", self.error.as_deref().unwrap_or(""), cache().display()));
            });
            return;
        };
        egui::CentralPanel::default().frame(egui::Frame::central_panel(ui.style()).inner_margin(egui::Margin { left: 12, right: 12, top: 8, bottom: 2 })).show(ui, |ui| app.main_ui(ui));
        app.track_window(&ctx);
        if !app.s.recs_open {
            app.recs_opened = None;
        } else {
            // Kept from opening: fed the tracked values, egui re-sends every change as a resize or move on each repaint, and
            // repaints run constantly in game (Less: only this window, only in game, only once moved and resized).
            let ((w, h), at) = *app.recs_opened.get_or_insert((app.s.recs_win, app.s.recs_pos));
            let mut b = ViewportBuilder::default().with_title("ManiaScope — Playlist").with_inner_size([w, h]).with_app_id("maniascope");
            if let Some(p) = at {
                b = b.with_position(p); // Windows/X11 only: Wayland ignores it (a compositor window rule places them there)
            }
            if app.s.keep_above {
                b = b.with_always_on_top();
            }
            let closed = ctx.show_viewport_immediate(windows::playlist_id(), b, |ui, _| {
                egui::CentralPanel::default().frame(egui::Frame::central_panel(ui.style()).inner_margin(14)).show(ui, |ui| app.playlist_ui(ui));
                let c = ui.ctx().clone();
                app.track_playlist(&c);
                c.input(|i| i.viewport().close_requested())
            });
            if closed {
                app.s.recs_open = false;
                app.request_save();
            }
        }
        app.side_windows(&ctx);
        if ctx.input(|i| i.viewport().close_requested()) {
            lifecycle("close requested");
        }
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        if let Some(app) = &mut self.app {
            app.shutdown();
        }
        lifecycle("destroy");
    }
}

fn listen(mail: app::Mail) {
    let Ok(listener) = TcpListener::bind(("127.0.0.1", INSTANCE_PORT)) else { return };
    std::thread::spawn(move || {
        for s in listener.incoming().flatten() {
            let mut hello = [0u8; 10];
            if matches!((&s).read(&mut hello), Ok(10)) && &hello == b"maniascope" {
                mail.send(Msg::Raise);
            }
        }
    });
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("--self-test") => std::process::exit(self_test()),
        Some("--install-desktop") => std::process::exit(install_desktop()),
        Some("--webmaps") => std::process::exit(webmaps(&args[1..])),
        _ => {}
    }
    if let Ok(mut other) = TcpStream::connect(("127.0.0.1", INSTANCE_PORT)) {
        let _ = other.write_all(b"maniascope");
        return; // single instance: the running window comes forward
    }
    log_to_file();
    crash_log();
    let s = settings::Ui::load();
    let (w, h) = if s.details { s.details_win } else { s.win };
    let mut view = ViewportBuilder::default().with_title("ManiaScope").with_app_id("maniascope").with_inner_size([w, h]).with_min_inner_size([200., 120.]);
    if let Some(p) = s.win_pos {
        view = view.with_position(p);
    }
    if s.keep_above {
        view = view.with_always_on_top();
    }
    if let Ok(icon) = eframe::icon_data::from_png_bytes(ICON) {
        view = view.with_icon(icon);
    }
    // Windows: DirectX first. A VM or remote desktop has no OpenGL 2 ("egui_glow requires opengl 2.0+").
    #[cfg(windows)]
    let renderers = [eframe::Renderer::Wgpu, eframe::Renderer::Glow];
    #[cfg(not(windows))]
    let renderers = [eframe::Renderer::Glow];
    for renderer in renderers {
        let options = eframe::NativeOptions { viewport: view.clone(), renderer, ..Default::default() };
        match eframe::run_native("ManiaScope", options, Box::new(|_| Ok(Box::new(Shell { app: None, error: None })))) {
            Ok(()) => break,
            Err(e) => eprintln!("ManiaScope could not start with {renderer}: {e}"),
        }
    }
    lifecycle("application loop exited");
}
