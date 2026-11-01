//! Computed offsets against a published offsets file: `dump <dir with DLLs and truth.json>`.
use serde_json::Value;

fn main() {
    let dir = std::path::PathBuf::from(std::env::args().nth(1).unwrap());
    let truth: Value = serde_json::from_slice(&std::fs::read(dir.join("truth.json")).unwrap()).unwrap();
    let mut layouts = ms_osumem::layout::Layouts::new(&dir);
    let asms = ["osu!", "osu.Game", "osu.Framework", "osu.Game.Rulesets.Osu", "Sentry"];
    let (mut good, mut bad, mut missing, mut shown) = (0, 0, 0, 0);
    for (class, fields) in truth.as_object().unwrap() {
        let Some(fields) = fields.as_object() else { continue };
        let found = layouts.class(&asms, class);
        for (field, want) in fields {
            let got = found.as_ref().and_then(|l| l.fields.get(field)).copied();
            match got {
                Some(g) if Some(g as u64) == want.as_u64() => good += 1,
                Some(g) => {
                    bad += 1;
                    if shown < 40 {
                        shown += 1;
                        println!("{class} {field}: got {g} want {want}");
                    }
                }
                None => {
                    missing += 1;
                    if shown < 40 {
                        shown += 1;
                        println!("{class} {field}: not found");
                    }
                }
            }
        }
    }
    println!("{good} right, {bad} wrong, {missing} missing");
    if let Some(l) = layouts.class(&asms, "osu.Desktop.OsuGameDesktop") {
        println!("base size {} vtable word {:?} → {}", l.base_size(), truth["GameBaseVtable"], truth["GameBaseVtable"].as_u64().unwrap_or(0) >> 32);
    }
}
