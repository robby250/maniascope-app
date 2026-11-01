//! The population evidence: the active snapshot's mapped file, a public.pkl of an older install converted once, or the
//! published copy for this calculator.
use ms_data::paths::rolling;
use ms_data::public::{activate, active, Public};
use ms_rec::engine::CALC_ID;
use ms_rec::worker::{Hooks, Out};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn release_url() -> String {
    let repo = std::env::var("MANIASCOPE_RELEASE_REPO").unwrap_or_else(|_| "robby250/maniascope".into());
    std::env::var("MANIASCOPE_RELEASE_URL").unwrap_or_else(|_| format!("https://github.com/{repo}/releases/latest/download"))
}

fn candidates(dir: &Path) -> [PathBuf; 2] {
    [dir.join("calculators").join(CALC_ID).join("public.msp"), dir.join("public.msp")]
}

fn local() -> Option<Result<Public, String>> {
    let (name, _) = active()?;
    let dir = rolling().join(name);
    if let Some(p) = candidates(&dir).iter().find(|p| p.is_file()) {
        return Some(ms_model::public::open(p));
    }
    // An install from before the mapped format.
    for pkl in [dir.join("calculators").join(CALC_ID).join("public.pkl"), dir.join("public.pkl")] {
        if pkl.is_file() {
            let to = pkl.with_extension("msp");
            return Some(ms_model::public::convert(&pkl, &to).and_then(|()| ms_model::public::open(&to)));
        }
    }
    None
}

fn fetch(h: &Hooks) -> Result<Public, String> {
    let url = format!("{}/public-{CALC_ID}.msp", release_url());
    let response = ureq::get(&url).set("User-Agent", "maniascope").timeout(std::time::Duration::from_secs(600)).call().map_err(|e| e.to_string())?;
    let total: usize = response.header("Content-Length").and_then(|v| v.parse().ok()).unwrap_or(0);
    let dir = rolling().join("release").join("calculators").join(CALC_ID);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let part = dir.join("public.msp.part");
    let mut file = std::fs::File::create(&part).map_err(|e| e.to_string())?;
    let (mut reader, mut chunk, mut done, start) = (response.into_reader(), vec![0u8; 1 << 20], 0usize, std::time::Instant::now());
    loop {
        let n = reader.read(&mut chunk).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        std::io::Write::write_all(&mut file, &chunk[..n]).map_err(|e| e.to_string())?;
        done += n;
        let speed = done as f64 / start.elapsed().as_secs_f64().max(1e-3);
        let left = if total > done { format!(" · ~{:.0} s left", (total - done) as f64 / speed) } else { String::new() };
        let of = if total > 0 { format!(" / {:.0}", total as f64 / 1e6) } else { String::new() };
        (h.emit)(Out::Status { text: format!("Downloading population data (what other players score) · {:.0}{of} MB{left}", done as f64 / 1e6), progress: Some((done, total)), blocking: false, learning: None });
    }
    drop(file);
    let path = dir.join("public.msp");
    std::fs::rename(&part, &path).map_err(|e| e.to_string())?;
    let public = ms_model::public::open(&path)?;
    if public.meta.calc != CALC_ID {
        return Err("published evidence belongs to another calculator".into());
    }
    activate("release").map_err(|e| e.to_string())?;
    Ok(public)
}

pub fn load(h: &Hooks) -> Result<Arc<Public>, String> {
    let found = local().and_then(Result::ok).filter(|p| p.meta.calc == CALC_ID);
    let public = match found {
        Some(p) if p.meta.pop.version >= ms_data::public::POP_VERSION => p,
        Some(old) => fetch(h).unwrap_or(old), // offline: the older link still works
        None => fetch(h).map_err(|e| format!("could not download the population data ({e}); check the internet connection and restart ManiaScope"))?,
    };
    Ok(Arc::new(public))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ms_data::public::{write, Meta, Population};

    /// The only test of this binary that sets the data folder and release address.
    #[test]
    fn evidence_of_another_calculator_is_refused_and_older_links_are_kept_offline() {
        let dir = std::env::temp_dir().join(format!("ms-public-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        // SAFETY: nothing else in this binary reads these two.
        unsafe {
            std::env::set_var("MANIASCOPE_REC", &dir);
            std::env::set_var("MANIASCOPE_RELEASE_URL", "http://127.0.0.1:1"); // never reaches anything
        }
        let folder = rolling().join("fixture");
        std::fs::create_dir_all(&folder).unwrap();
        std::fs::write(rolling().join("active.json"), serde_json::json!({"name": "fixture"}).to_string()).unwrap();
        let save = |calc: &str, version| write(&folder.join("public.msp"), &Meta { calc: calc.into(), pop: Population { version, ..Population::default() }, ..Meta::default() }, &[]).unwrap();
        let h = Hooks { emit: Arc::new(|_| {}), busy: Arc::new(|| false), game: Arc::new(|| false) };
        save("not-this", ms_data::public::POP_VERSION);
        let refused = load(&h).err().unwrap_or_default();
        assert!(refused.contains("could not download"), "{refused}");
        save(CALC_ID, 1);
        assert_eq!(load(&h).unwrap().meta.pop.version, 1, "a failed download keeps the old evidence");
        save(CALC_ID, ms_data::public::POP_VERSION);
        assert!(load(&h).is_ok());
        let _ = std::fs::remove_dir_all(dir);
    }
}
