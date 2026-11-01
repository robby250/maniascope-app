//! One-click helpers for the optional external tools.
use crate::paths;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::time::Duration;

const UA: &str = "maniascope";

pub fn tosu_bin() -> PathBuf {
    paths::data().join("tosu").join(if paths::WINDOWS { "tosu.exe" } else { "tosu" })
}

/// Latest official tosu release for this platform → data/tosu. tosu (LGPL-3.0) stays a separate program downloaded
/// from its own releases, never bundled. The official build is right for Windows: patches/tosu-proc-read.patch fixes
/// a crash in the Linux-only /proc reader, and tosu-loop-waits.patch only shortens song-select polling.
pub fn install_tosu() -> Result<PathBuf, String> {
    let e = |x: &dyn std::fmt::Display| x.to_string();
    let body = ureq::get("https://api.github.com/repos/tosuapp/tosu/releases/latest").set("User-Agent", UA).timeout(Duration::from_secs(20)).call().map_err(|x| e(&x))?
        .into_string().map_err(|x| e(&x))?;
    let release: Value = serde_json::from_str(&body).map_err(|x| e(&x))?;
    let want = if paths::WINDOWS { "tosu-windows-" } else { "tosu-linux-" };
    let url = release["assets"].as_array().and_then(|a| a.iter().find(|x| x["name"].as_str().is_some_and(|n| n.starts_with(want) && n.ends_with(".zip"))))
        .and_then(|a| a["browser_download_url"].as_str()).ok_or("no tosu asset for this platform")?;
    let mut data = Vec::new();
    std::io::Read::read_to_end(&mut ureq::get(url).set("User-Agent", UA).timeout(Duration::from_secs(120)).call().map_err(|x| e(&x))?.into_reader(), &mut data).map_err(|x| e(&x))?;
    let exe = tosu_bin();
    let target = exe.parent().unwrap();
    std::fs::create_dir_all(target).map_err(|x| e(&x))?;
    zip::ZipArchive::new(std::io::Cursor::new(data)).map_err(|x| e(&x))?.extract(target).map_err(|x| e(&x))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&exe).map_err(|x| e(&x))?.permissions().mode();
        std::fs::set_permissions(&exe, std::fs::Permissions::from_mode(mode | 0o100)).map_err(|x| e(&x))?;
    }
    Ok(exe)
}

/// Set KEY=value lines in tosu.env, keeping the others; returns whether the file changed.
pub fn set_tosu_env(path: &Path, values: &[(&str, &str)]) -> std::io::Result<bool> {
    let old = std::fs::read_to_string(path).unwrap_or_default();
    let mut lines: Vec<String> = old.lines().map(str::to_string).collect();
    for (key, value) in values {
        let line = format!("{key}={value}");
        let prefix = format!("{key}=");
        let mut hit = false;
        for l in lines.iter_mut().filter(|l| l.starts_with(&prefix)) {
            *l = line.clone();
            hit = true;
        }
        if !hit {
            lines.push(line);
        }
    }
    let mut new = lines.join("\n");
    if !new.is_empty() {
        new.push('\n');
    }
    let changed = new != old;
    if changed {
        std::fs::create_dir_all(path.parent().unwrap())?;
        std::fs::write(path, new)?;
    }
    Ok(changed)
}

/// The ManiaScope page in tosu's static folder (its default on Windows: ./static beside tosu.exe).
pub fn overlay_dir() -> PathBuf {
    tosu_bin().parent().unwrap().join("static").join("ManiaScope")
}

pub fn install_overlay() -> std::io::Result<()> {
    let dir = overlay_dir();
    std::fs::create_dir_all(&dir)?;
    std::fs::write(dir.join("index.html"), include_str!("../../../assets/overlay/ManiaScope/index.html"))?;
    std::fs::write(dir.join("metadata.txt"), include_str!("../../../assets/overlay/ManiaScope/metadata.txt"))
}

/// state.json for the page; tmp + rename, so a fetch never reads half a file. A busy file skips one update.
pub fn write_overlay_state(state: &Value) {
    let path = overlay_dir().join("state.json");
    let tmp = path.with_extension("json.tmp");
    if std::fs::write(&tmp, state.to_string()).is_ok() {
        let _ = std::fs::rename(tmp, path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tosu_env_keeps_other_lines_and_reports_changes() {
        let dir = std::env::temp_dir().join(format!("ms-tosu-{}", std::process::id()));
        let env = dir.join("cfg/tosu.env");
        assert!(set_tosu_env(&env, &[("A", "false"), ("B", "100")]).unwrap());
        std::fs::write(&env, "A=true\nKEEP=1").unwrap();
        assert!(set_tosu_env(&env, &[("A", "false"), ("B", "100")]).unwrap());
        assert_eq!(std::fs::read_to_string(&env).unwrap(), "A=false\nKEEP=1\nB=100\n");
        assert!(!set_tosu_env(&env, &[("A", "false")]).unwrap());
        let _ = std::fs::remove_dir_all(dir);
    }
}
