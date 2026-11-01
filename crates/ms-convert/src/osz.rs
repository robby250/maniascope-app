//! Writing `.osz` files: zip archives of a beatmap set (Torosu's `osz.rs`).
use anyhow::Result;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipWriter};

pub enum Entry<'a> {
    Bytes(&'a [u8]),
    File(&'a Path),
}

/// Text and raw audio deflate well; audio, images and videos are compressed already.
fn compression(name: &str) -> CompressionMethod {
    let extension = name.rsplit_once('.').map(|(_, e)| e.to_ascii_lowercase());
    match extension.as_deref() {
        Some("osu" | "osb" | "wav" | "txt" | "bmp") => CompressionMethod::Deflated,
        _ => CompressionMethod::Stored,
    }
}

/// Written beside the target and renamed: the client picks up any .osz, and a half-written one imports as a broken set.
pub fn write(path: &Path, entries: &[(String, Entry<'_>)]) -> Result<()> {
    let part = path.with_extension("osz.part");
    let attempt = || -> Result<()> {
        let mut zip = ZipWriter::new(File::create(&part)?);
        for (name, entry) in entries {
            zip.start_file(name.as_str(), SimpleFileOptions::default().compression_method(compression(name)))?;
            match entry {
                Entry::Bytes(b) => zip.write_all(b)?,
                Entry::File(p) => { std::io::copy(&mut File::open(p)?, &mut zip)?; }
            }
        }
        zip.finish()?.sync_all()?;
        std::fs::rename(&part, path)?;
        Ok(())
    };
    attempt().inspect_err(|_| { let _ = std::fs::remove_file(&part); })
}
