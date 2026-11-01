//! The Windows exe's icon, as a linked resource: mingw's windres, the only toolchain the release uses.
use std::{env, path::Path, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    let out = env::var("OUT_DIR").unwrap();
    let (rc, object) = (Path::new(&out).join("icon.rc"), Path::new(&out).join("icon.o"));
    let icon = Path::new(&env::var("CARGO_MANIFEST_DIR").unwrap()).join("../../assets/icon.ico");
    std::fs::write(&rc, format!("1 ICON \"{}\"\n", icon.display().to_string().replace('\\', "/"))).unwrap();
    let windres = env::var("WINDRES").unwrap_or_else(|_| "x86_64-w64-mingw32-windres".into());
    let status = Command::new(windres).arg(&rc).args(["-O", "coff", "-o"]).arg(&object).status().expect("windres");
    assert!(status.success());
    println!("cargo:rustc-link-arg-bins={}", object.display());
}
