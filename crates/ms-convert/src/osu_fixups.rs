//! Fixups applied to the output of `Beatmap::encode_to_string()`, which is not
//! quite a `.osu` that osu! stable accepts.

pub fn sanitize(osu: String) -> String {
    let osu = fix_general_sample_set(osu);
    let osu = fix_combo_colours(osu);
    let osu = strip_orphan_inherited_timing_points(osu);

    to_crlf(osu)
}

/// Drops inherited timing points that precede the first uninherited one.
///
/// `encode_timing_points` calls `collect_samples()`, which derives sample points
/// from hit objects and emits a line for each. The first hit object usually sits
/// before the first red line, so an orphan green line ends up first in the file:
///
/// ```text
/// 324,-100,4,0,1,40,0,0    <- inherited, and first in the file
/// 574,250,4,2,1,40,1,0     <- the actual first red line
/// ```
///
/// The first timing point must always be uninherited — an inherited one has no
/// parent BPM to inherit. osu! stable hangs on the loading screen trying to
/// resolve it: the map lists fine and opens in the editor, but never starts.
fn strip_orphan_inherited_timing_points(osu: String) -> String {
    let mut out = String::with_capacity(osu.len());
    let mut in_section = false;
    let mut seen_uninherited = false;
    let mut dropped = 0;

    for line in osu.split_inclusive('\n') {
        let trimmed = line.trim();

        if trimmed.starts_with('[') {
            in_section = trimmed == "[TimingPoints]";
            out.push_str(line);
            continue;
        }

        if in_section && !seen_uninherited && !trimmed.is_empty() {
            // time,beatLength,meter,sampleSet,sampleIndex,volume,uninherited,effects
            match trimmed.split(',').nth(6).map(str::trim) {
                Some("1") => seen_uninherited = true,
                Some(_) => {
                    
                    dropped += 1;
                    continue;
                }
                None => {}
            }
        }

        out.push_str(line);
    }

    if dropped > 0 {
        
    }

    out
}

/// `encode_general` writes `SampleSet` as a number, but osu! expects the name
/// there. In `[TimingPoints]` the sample set is numeric, so only this key is
/// touched.
fn fix_general_sample_set(osu: String) -> String {
    replace_lines(osu, "SampleSet: ", |value| {
        let name = match value.trim() {
            "0" => "None",
            "1" => "Normal",
            "2" => "Soft",
            "3" => "Drum",
            _ => return None,
        };

        Some(format!("SampleSet: {}", name))
    })
}

/// `encode_colors` emits `Combo1: r,g,b,a`; osu! expects three components.
fn fix_combo_colours(osu: String) -> String {
    let mut out = String::with_capacity(osu.len());

    for line in osu.split_inclusive('\n') {
        let trimmed = line.trim_end_matches(['\n', '\r']);

        let fixed = trimmed
            .split_once(':')
            .filter(|(key, _)| key.trim().starts_with("Combo"))
            .and_then(|(key, value)| {
                let parts: Vec<&str> = value.split(',').map(str::trim).collect();
                (parts.len() == 4).then(|| {
                    format!("{} : {},{},{}\n", key.trim(), parts[0], parts[1], parts[2])
                })
            });

        match fixed {
            Some(f) => out.push_str(&f),
            None => out.push_str(line),
        }
    }

    out
}

fn replace_lines(osu: String, prefix: &str, f: impl Fn(&str) -> Option<String>) -> String {
    let mut out = String::with_capacity(osu.len());

    for line in osu.split_inclusive('\n') {
        let trimmed = line.trim_end_matches(['\n', '\r']);

        match trimmed.strip_prefix(prefix).and_then(|v| f(v)) {
            Some(replacement) => {
                out.push_str(&replacement);
                out.push('\n');
            }
            None => out.push_str(line),
        }
    }

    out
}

/// `.osu` files use CRLF; rosu-map's `writeln!` only emits `\n`.
fn to_crlf(osu: String) -> String {
    osu.replace("\r\n", "\n").replace('\n', "\r\n")
}
