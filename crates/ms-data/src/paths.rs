//! Where everything lives, per OS (paths.py).
use std::env;
use std::path::{Path, PathBuf};

pub const WINDOWS: bool = cfg!(windows);

pub fn home() -> PathBuf {
    env::var_os(if WINDOWS { "USERPROFILE" } else { "HOME" }).map_or_else(|| PathBuf::from("."), PathBuf::from)
}

fn win(var: &str, parts: &[&str]) -> PathBuf {
    let mut p = env::var_os(var).map_or_else(
        || home().join("AppData").join(if var == "APPDATA" { "Roaming" } else { "Local" }),
        PathBuf::from,
    );
    parts.iter().for_each(|x| p.push(x));
    p
}

/// `~` at the start of an environment override, as os.path.expanduser did.
fn expand(value: &str) -> PathBuf {
    value.strip_prefix("~/").map_or_else(|| PathBuf::from(value), |rest| home().join(rest))
}

/// lazer's data directory (client.realm, files/, logs/).
pub fn lazer_data() -> PathBuf {
    if let Ok(v) = env::var("LAZER_DATA") {
        if !v.is_empty() {
            return expand(&v);
        }
    }
    if WINDOWS {
        return win("APPDATA", &["osu"]);
    }
    if cfg!(target_os = "macos") {
        return home().join("Library/Application Support/osu");
    }
    let flatpak = home().join(".var/app/sh.ppy.osu/data/osu");
    let native = home().join(".local/share/osu");
    if flatpak.is_dir() || !native.is_dir() { flatpak } else { native }
}

/// osu!stable's folder: STABLE_DATA, else the usual install. Linux runs it under Wine: osu-winello's saved path,
/// its default, then the default Wine prefix.
pub fn stable_data() -> PathBuf {
    if let Ok(v) = env::var("STABLE_DATA") {
        if !v.is_empty() {
            return expand(&v);
        }
    }
    if let Some(seen) = std::fs::read_to_string(stable_seen()).ok().map(PathBuf::from).filter(|d| d.join("osu!.db").is_file()) {
        return seen;
    }
    if WINDOWS {
        return win("LOCALAPPDATA", &["osu!"]);
    }
    let mut found = Vec::new();
    if let Ok(saved) = std::fs::read_to_string(home().join(".local/share/osuconfig/osupath")) {
        found.push(PathBuf::from(saved.trim()));
    }
    let user = env::var("USER").unwrap_or_else(|_| home().file_name().map_or(String::new(), |n| n.to_string_lossy().into_owned()));
    found.push(home().join(".local/share/osu-wine/osu!"));
    found.push(home().join(".wine/drive_c/users").join(user).join("AppData/Local/osu!"));
    found.iter().find(|d| !d.as_os_str().is_empty() && d.join("osu!.db").is_file()).unwrap_or(&found[0]).clone()
}

fn stable_seen() -> PathBuf {
    rec_dir().join("stable_path")
}

/// A Wine path on this machine's own disk; everything else as given.
fn native_path(path: &str) -> PathBuf {
    match path.strip_prefix("Z:\\").or_else(|| path.strip_prefix("z:\\")) {
        Some(rest) if !WINDOWS => PathBuf::from(format!("/{}", rest.replace('\\', "/"))),
        _ => PathBuf::from(path),
    }
}

/// The folder of the stable that tosu sees running, kept so a custom install needs no setup and works with the game closed.
pub fn remember_stable(game: &str) {
    let folder = native_path(game);
    if folder.join("osu!.db").is_file() && stable_data() != folder {
        let _ = std::fs::write(stable_seen(), folder.to_string_lossy().as_bytes());
    }
}

/// The frozen September dump; only the calibration machine has one.
pub fn frozen() -> PathBuf {
    home().join(".cache/maniascope/dump")
}

/// A chart's copy in a dump folder, when it is that revision (no md5: any). Beatmap ids keep their map across revisions.
pub fn dump_chart(root: &Path, bid: i64, md5: Option<&str>) -> Option<PathBuf> {
    let path = root.join("osu").join(format!("{bid}.osu"));
    let same = |want: &str| std::fs::read(&path).is_ok_and(|data| format!("{:x}", md5::compute(data)) == want);
    (path.is_file() && md5.is_none_or(same)).then_some(path)
}

pub fn frozen_chart(bid: i64, md5: Option<&str>) -> Option<PathBuf> {
    dump_chart(&frozen(), bid, md5)
}

pub fn data() -> PathBuf {
    if WINDOWS { win("LOCALAPPDATA", &["maniascope"]) } else { home().join(".local/share/maniascope") }
}

pub fn cache() -> PathBuf {
    if WINDOWS { win("LOCALAPPDATA", &["maniascope", "cache"]) } else { home().join(".cache/maniascope") }
}

pub fn config() -> PathBuf {
    if WINDOWS { win("APPDATA", &["maniascope"]) } else { home().join(".config/maniascope") }
}

pub fn rec_dir() -> PathBuf {
    env::var("MANIASCOPE_REC").map_or_else(|_| data().join("rec"), |v| expand(&v))
}

pub fn rolling() -> PathBuf {
    rec_dir().join("rolling")
}

/// A chart or replay in lazer's content-addressed layout under `root`.
pub fn local_file(sha: &str, root: &Path) -> Option<PathBuf> {
    (sha.len() == 64 && sha.bytes().all(|b| b.is_ascii_hexdigit())).then(|| root.join("files").join(&sha[..1]).join(&sha[..2]).join(sha))
}

/// Windows: stable's Songs paths often pass MAX_PATH (260); the long-path (\\?\) form has no limit.
pub fn long_path(path: &Path) -> PathBuf {
    if WINDOWS && !path.to_string_lossy().starts_with(r"\\?\") {
        let absolute = std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
        PathBuf::from(format!(r"\\?\{}", absolute.display()))
    } else {
        path.to_path_buf()
    }
}

/// stable's .osu files under their SHA-256 in lazer's layout (hard links): every sha → path lookup works for both
/// clients, and an edited chart keeps its old revision's identity.
pub fn stable_store() -> PathBuf {
    data().join("stable")
}

fn hostname() -> String {
    std::fs::read_to_string("/proc/sys/kernel/hostname").map(|s| s.trim().to_string())
        .or_else(|_| env::var("COMPUTERNAME")).unwrap_or_default()
}

/// The manifest's machine and file store: (host, root).
pub fn local_origin(client: &str) -> (String, PathBuf) {
    let root = if client == "stable" { stable_store() } else { std::fs::canonicalize(lazer_data()).unwrap_or_else(|_| lazer_data()) };
    (hostname(), root)
}

#[cfg(test)]
mod dump_tests {
    #[test]
    fn a_wine_path_on_the_root_drive_is_a_local_path() {
        if !super::WINDOWS {
            assert_eq!(super::native_path("Z:\\home\\x\\osu!").to_str(), Some("/home/x/osu!"));
            assert_eq!(super::native_path("/home/x/osu!").to_str(), Some("/home/x/osu!"));
        }
    }

    use super::*;

    #[test]
    fn a_dump_chart_counts_only_at_its_revision() {
        let root = std::env::temp_dir().join(format!("ms-dump-{}", std::process::id()));
        std::fs::create_dir_all(root.join("osu")).unwrap();
        std::fs::write(root.join("osu/7.osu"), b"chart").unwrap();
        let md5 = format!("{:x}", md5::compute(b"chart"));
        assert_eq!(dump_chart(&root, 7, Some(&md5)), Some(root.join("osu/7.osu")));
        assert_eq!(dump_chart(&root, 7, None), Some(root.join("osu/7.osu")));
        assert_eq!(dump_chart(&root, 7, Some("other")), None);
        assert_eq!(dump_chart(&root, 8, None), None);
        std::fs::remove_dir_all(root).unwrap();
    }
}
