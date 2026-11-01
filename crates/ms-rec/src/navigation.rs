//! Explicit Next actions. Search syntax checked against ppy/osu FilterQueryParser.cs and FilterCriteria.cs: quoted
//! values followed by ! match the complete field; backslash-escaped quotes are NOT supported. A local SHA/MD5 is an
//! identity, not a song-search keyword.
use crate::catalog::Pick;
use ms_data::paths;
use std::path::PathBuf;
use std::process::{Command, Stdio};

pub const ACTIONS: [&str; 3] = ["auto", "copy", "link"];

/// A submitted chart can update its bytes without changing its map ID.
pub fn same_map(p: &Pick, sha: Option<&str>, md5: Option<&str>, bid: Option<i64>) -> bool {
    (sha.is_some() && sha == p.sha.as_deref()) || (md5.is_some() && md5 == p.md5.as_deref()) || (bid.is_some_and(|b| b > 0) && bid == p.bid)
}

fn field(name: &str, value: &str) -> String {
    let value = value.trim();
    // The lazer parser has no quoted-string escape convention. A literal safe fragment is preferable to a
    // syntactically broken 'escaped' exact match.
    let fragments: Vec<&str> = value.split(['"', '\r', '\n', '\t']).collect();
    let fragment = fragments.iter().copied().max_by_key(|f| f.chars().count()).unwrap_or("").trim();
    if fragment.is_empty() {
        return String::new();
    }
    format!("{name}=\"{fragment}\"{}", if fragments.len() == 1 { "!" } else { "" })
}

/// Search the local difficulty; intended playback rate is deliberately absent.
pub fn song_search(p: &Pick, client: &str) -> String {
    let (artist, title, version, creator) = (p.artist.as_deref().unwrap_or(""), p.song.as_deref().unwrap_or(""), p.version.as_deref().unwrap_or(""), p.creator.as_deref().unwrap_or(""));
    if client == "stable" {
        // stable has no quoted field values: its words match artist/title/difficulty/creator/tags, keys= filters
        // (osu! wiki, Beatmap search)
        let words = format!("{artist} {title} {version}").replace(['"', '=', '<', '>', '!'], " ");
        let query = words.split_whitespace().collect::<Vec<_>>().join(" ");
        return if query.is_empty() { query } else if p.keys != 0 { format!("{query} keys={}", p.keys) } else { query };
    }
    let fields = [field("artist", artist), field("title", title), field("diff", version), field("creator", creator)];
    let query = fields.iter().filter(|f| !f.is_empty()).cloned().collect::<Vec<_>>().join(" ");
    if !query.is_empty() {
        return if p.keys != 0 { format!("{query} cs={}", p.keys) } else { query };
    }
    p.bid.filter(|&b| b > 0).map_or_else(|| p.title.trim().to_string(), |b| b.to_string())
}

/// Full paths of the running osu!.exe processes. lazer and stable share that file name.
#[cfg(windows)]
fn osu_exes() -> Vec<PathBuf> {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::ProcessStatus::EnumProcesses;
    use windows_sys::Win32::System::Threading::{OpenProcess, QueryFullProcessImageNameW, PROCESS_NAME_WIN32, PROCESS_QUERY_LIMITED_INFORMATION};
    let (mut pids, mut used) = (vec![0u32; 16384], 0u32);
    // SAFETY: the buffers outlive the calls, and every opened handle is closed.
    unsafe {
        if EnumProcesses(pids.as_mut_ptr(), (pids.len() * 4) as u32, &mut used) == 0 {
            return Vec::new();
        }
        let mut found = Vec::new();
        for &pid in &pids[..used as usize / 4] {
            let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
            if handle.is_null() {
                continue;
            }
            let (mut name, mut size) = (vec![0u16; 32768], 32768u32);
            if QueryFullProcessImageNameW(handle, PROCESS_NAME_WIN32, name.as_mut_ptr(), &mut size) != 0 {
                let path = PathBuf::from(String::from_utf16_lossy(&name[..size as usize]));
                if path.file_name().is_some_and(|n| n.eq_ignore_ascii_case("osu!.exe")) {
                    found.push(path);
                }
            }
            CloseHandle(handle);
        }
        found
    }
}

#[cfg(not(windows))]
fn osu_exes() -> Vec<PathBuf> {
    Vec::new()
}

/// The active client's running osu!.exe (Windows); stable's sits beside its osu!.db.
fn client_exe(client: &str) -> Option<PathBuf> {
    osu_exes().into_iter().find(|e| e.with_file_name("osu!.db").is_file() == (client == "stable"))
}

/// The active client is running (stable is osu!.exe under Wine).
pub fn game_running(client: &str) -> bool {
    if paths::WINDOWS {
        return client_exe(client).is_some();
    }
    let want = if client == "stable" { "osu!.exe" } else { "osu!" };
    std::fs::read_dir("/proc").into_iter().flatten().flatten().any(|e| {
        e.file_name().to_string_lossy().bytes().all(|b| b.is_ascii_digit()) && std::fs::read_to_string(e.path().join("comm")).is_ok_and(|c| c.trim() == want)
    })
}

/// Hand an osu:// link to the running client: a second process of the same exe only forwards it over IPC. On Windows
/// never through the URL handler: osu:// belongs to whichever client registered it, so a lazer player got stable
/// launched (user 2026-10-31).
fn open_url(url: &str, client: &str) -> std::io::Result<()> {
    let home = std::env::var("HOME").unwrap_or_default();
    let mut command = if paths::WINDOWS {
        let mut c = Command::new(client_exe(client).ok_or_else(|| std::io::Error::other("osu! has closed"))?);
        c.arg(url);
        c
    } else if client == "lazer" && std::path::Path::new(&home).join(".var/app/sh.ppy.osu").is_dir() {
        let mut c = Command::new("flatpak");
        c.args(["run", "sh.ppy.osu", url]);
        c
    } else {
        let mut c = Command::new("xdg-open");
        c.arg(url);
        c
    };
    command.stdout(Stdio::null()).stderr(Stdio::null()).spawn().map(|_| ())
}

/// Hand a file to the running client: a second process of the same exe forwards it over IPC (Program.cs), and lazer
/// imports it. Linux lazer: the running binary itself, or Flatpak's; xdg-open only knows what's registered.
pub fn hand_over(path: &std::path::Path, client: &str) -> Result<(), String> {
    if client == "stable" && !paths::WINDOWS {
        return Ok(()); // the .osz is in Songs and stable imports it on entering song select (Wine, 2026-11-01); xdg-open would run whatever is registered, lazer included
    }
    let home = std::env::var("HOME").unwrap_or_default();
    let mut command = if paths::WINDOWS {
        let mut c = Command::new(client_exe(client).ok_or("osu! has closed")?);
        c.arg(path);
        c
    } else if client == "lazer" && std::path::Path::new(&home).join(".var/app/sh.ppy.osu").is_dir() {
        let mut c = Command::new("flatpak");
        c.args(["run", "sh.ppy.osu"]).arg(path);
        c
    } else if let Some(exe) = running_lazer().filter(|_| client == "lazer") {
        let mut c = Command::new(exe);
        c.arg(path);
        c
    } else {
        let mut c = Command::new("xdg-open");
        c.arg(path);
        c
    };
    command.stdout(Stdio::null()).stderr(Stdio::null()).spawn().map(|_| ()).map_err(|e| e.to_string())
}

fn running_lazer() -> Option<PathBuf> {
    std::fs::read_dir("/proc").ok()?.flatten().find(|e| std::fs::read_to_string(e.path().join("comm")).is_ok_and(|c| c.trim() == "osu!")).and_then(|e| std::fs::read_link(e.path().join("exe")).ok())
}

#[derive(Default)]
pub struct Performed {
    pub copy: Option<String>,
    pub msg: String,
    pub sent: bool,
    pub unavailable: bool,
}

/// copy/msg/sent; never starts the game or automatically changes mods.
pub fn perform(p: &mut Pick, action: &str, client: &str) -> Performed {
    if p.convert_of.is_some() && !p.installed {
        return Performed { msg: format!("Converting to {}K…", p.keys), ..Performed::default() }; // the worker delivers it
    }
    if p.installed {
        let path = p.sha.as_deref().and_then(|s| crate::pp::chart_path(Some(s)));
        if path.is_none() {
            if p.mode == "nps" || p.mode == "skills" {
                return Performed { msg: "Chart is no longer installed — refresh the library".to_string(), unavailable: true, ..Performed::default() };
            }
            p.installed = false;
        }
    }
    let action = if ACTIONS.contains(&action) { action } else { "auto" };
    let bid = p.bid.filter(|&b| b > 0);
    // A known edited revision must never open the online original instead.
    let trusted = bid.is_some() && p.online_match;
    if action == "copy" || (action == "auto" && p.installed) || !trusted {
        let query = song_search(p, client);
        if query.is_empty() {
            return Performed { msg: "No searchable metadata for this chart".to_string(), ..Performed::default() };
        }
        let msg = if !p.installed {
            "Search copied, but you don't have this map — download it, then Ctrl+V in song select"
        } else if action == "link" && !trusted {
            "No matching online map — search copied, press Ctrl+V in song select"
        } else {
            "Search copied — press Ctrl+V in song select"
        };
        return Performed { copy: Some(query), msg: msg.to_string(), ..Performed::default() };
    }
    let url = format!("osu://b/{}", bid.unwrap());
    if !game_running(client) {
        return Performed { copy: Some(url), msg: "osu! is not running — copied beatmap link".to_string(), ..Performed::default() };
    }
    if let Err(e) = open_url(&url, client) {
        return Performed { copy: Some(url), msg: format!("Could not open beatmap panel ({e}) — copied link"), ..Performed::default() };
    }
    // Auto still copies the search, so after downloading it can be pasted in song select (user 2026-10-09).
    let query = (action == "auto").then(|| song_search(p, client)).filter(|q| !q.is_empty());
    let msg = format!("Sent beatmap link to osu! — choose Go to beatmap; mods unchanged{}", if query.is_some() { " · search copied, Ctrl+V in song select" } else { "" });
    Performed { copy: query, msg, sent: true, unavailable: false }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_fragments_a_quoted_title_and_never_escapes() {
        assert_eq!(field("title", "plain"), "title=\"plain\"!");
        assert_eq!(field("title", "a \"long part\" b"), "title=\"long part\"");
        assert_eq!(field("title", "  "), "");
    }

    #[test]
    fn stable_search_has_no_field_syntax() {
        let p = Pick { artist: Some("A=B".into()), song: Some("T!".into()), version: Some("V".into()), keys: 7, ..Pick::default() };
        assert_eq!(song_search(&p, "stable"), "A B T V keys=7");
        assert_eq!(song_search(&p, "lazer"), "artist=\"A=B\"! title=\"T!\"! diff=\"V\"! cs=7");
    }
}
