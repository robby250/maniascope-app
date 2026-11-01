//! The tosu helper: only a tosu we started is ever stopped. Packaged builds fetch it once, unasked: rate tracking is part of
//! "just works".
use crate::app::{Mail, Msg};
use ms_data::paths::cache;
use ms_data::tools::{install_overlay, install_tosu, set_tosu_env, tosu_bin};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const RETRY: Duration = Duration::from_secs(10);

pub struct Tosu {
    mail: Mail,
    child: Option<Child>,
    tried: Option<Instant>,
    downloading: bool,
}

/// The Linux build reads ~/.config/tosu; the Windows build reads tosu.env beside its executable.
fn env_file() -> PathBuf {
    if cfg!(windows) { tosu_bin().with_file_name("tosu.env") } else { ms_data::paths::home().join(".config/tosu/tosu.env") }
}

pub fn installed() -> bool {
    tosu_bin().is_file()
}

fn answering() -> bool {
    use std::net::{TcpStream, ToSocketAddrs};
    crate::watcher::TOSU_HOST.to_socket_addrs().ok().and_then(|mut a| a.next()).is_some_and(|a| TcpStream::connect_timeout(&a, Duration::from_millis(300)).is_ok())
}

impl Tosu {
    pub fn new(mail: Mail) -> Tosu {
        Tosu { mail, child: None, tried: None, downloading: false }
    }

    pub fn install(&mut self) {
        if std::mem::replace(&mut self.downloading, true) {
            return;
        }
        let mail = self.mail.clone();
        std::thread::spawn(move || match install_tosu() {
            Ok(_) => {
                mail.send(Msg::Status("tosu installed — rate tracking starts".into()));
                mail.send(Msg::TosuReady);
            }
            // offline / GitHub unreachable: the log fallback still works
            Err(e) => mail.send(Msg::Status(format!("tosu download failed: {e}").chars().take(120).collect())),
        });
    }

    pub fn start(&mut self, overlay: bool) {
        if self.tried.is_some_and(|t| t.elapsed() < RETRY) {
            return;
        }
        self.tried = Some(Instant::now());
        if !installed() {
            if std::env::var(crate::app::DEBUG_ENV).is_err() {
                self.install();
            }
            return;
        }
        if self.child.as_mut().is_some_and(|c| matches!(c.try_wait(), Ok(None))) || answering() {
            return; // ours is still coming up, or someone's tosu already answers
        }
        let flag = if overlay { "true" } else { "false" };
        let rate = if overlay { "10" } else { "100" };
        // tosu only takes these from its config file; without the first every start opens a browser tab.
        let mut env = vec![("OPEN_DASHBOARD_ON_STARTUP", "false")];
        if cfg!(windows) {
            // Nothing here reads keys or live hit errors: tosu's 100 Hz gameplay loop only costs the game CPU. Other overlays
            // in the in-game layer may want them (tosu's own defaults).
            env.extend([("ENABLE_INGAME_OVERLAY", flag), ("ENABLE_KEY_OVERLAY", flag), ("PRECISE_DATA_POLL_RATE", rate)]);
        }
        let bin = tosu_bin();
        // The patched build (patches/tosu-loop-waits.patch) refreshes song select in ms, the WebSocket pushes at POLL_RATE:
        // a slower push would be the selection latency. The official build refreshes every 150 ms anyway.
        if bin.with_file_name("procfix-build.json").is_file() {
            env.push(("POLL_RATE", "30"));
        }
        let _ = set_tosu_env(&env_file(), &env);
        if overlay {
            let _ = install_overlay();
        }
        let _ = std::fs::create_dir_all(cache());
        let Ok(out) = std::fs::File::create(cache().join("tosu.log")) else { return };
        let mut c = Command::new(&bin);
        c.current_dir(bin.parent().unwrap_or(&bin)).stdin(Stdio::null()).stderr(out.try_clone().map_or(Stdio::null(), Stdio::from)).stdout(out);
        #[cfg(windows)]
        std::os::windows::process::CommandExt::creation_flags(&mut c, 0x0800_0000); // CREATE_NO_WINDOW
        match c.spawn() {
            Ok(child) => self.child = Some(child),
            Err(e) => eprintln!("start tosu: {e}"),
        }
    }

    pub fn stop(&mut self) {
        if let Some(mut c) = self.child.take() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }

    pub fn restart(&mut self, overlay: bool) {
        if self.child.is_none() {
            self.mail.send(Msg::Status("Restart tosu to apply the overlay setting".into()));
            return;
        }
        self.stop();
        self.tried = None;
        self.start(overlay);
    }
}
