//! ManiaScope's page in a tosu the player runs themselves.
use serde_json::Value;
use std::path::{Path, PathBuf};

/// The ManiaScope page of the tosu at `exe`, by tosu's own rule (getStaticPath): tosu.env's STATIC_FOLDER_PATH, and its
/// default ./static means the folder beside the program.
pub fn overlay_dir(exe: &Path) -> PathBuf {
    let env = std::fs::read_to_string(exe.with_file_name("tosu.env")).unwrap_or_default();
    let set = env.lines().find_map(|l| l.strip_prefix("STATIC_FOLDER_PATH=")).map(str::trim).filter(|v| !matches!(*v, "" | "./static" | ".\\static"));
    exe.with_file_name(set.unwrap_or("static")).join("ManiaScope")
}

pub fn install_overlay(dir: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    std::fs::write(dir.join("index.html"), include_str!("../../../assets/overlay/ManiaScope/index.html"))?;
    std::fs::write(dir.join("metadata.txt"), include_str!("../../../assets/overlay/ManiaScope/metadata.txt"))
}

/// state.json for the page; tmp + rename, so a fetch never reads half a file. A busy file skips one update.
pub fn write_overlay_state(dir: &Path, state: &Value) {
    let path = dir.join("state.json");
    let tmp = path.with_extension("json.tmp");
    if std::fs::write(&tmp, state.to_string()).is_ok() {
        let _ = std::fs::rename(tmp, path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_page_goes_where_that_tosu_reads_its_static_folder() {
        let dir = std::env::temp_dir().join(format!("ms-static-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let exe = dir.join("tosu.exe");
        assert_eq!(overlay_dir(&exe), dir.join("static/ManiaScope"), "no tosu.env");
        for (line, folder) in [("STATIC_FOLDER_PATH=./static", dir.join("static")), ("STATIC_FOLDER_PATH=.\\static", dir.join("static")), ("STATIC_FOLDER_PATH=counters", dir.join("counters")), ("STATIC_FOLDER_PATH=/srv/pages", "/srv/pages".into())] {
            std::fs::write(dir.join("tosu.env"), format!("POLL_RATE=100\n{line}\n")).unwrap();
            assert_eq!(overlay_dir(&exe), folder.join("ManiaScope"), "{line}");
        }
        let _ = std::fs::remove_dir_all(dir);
    }

}
