//! What each line says, as styled spans: no widgets here, so the wording is testable.
use crate::watcher::Obs;
use crate::viewer::Analysis;
use eframe::egui::Color32;
use ms_rec::catalog::Pick;
use ms_rec::engine::Card;
use ms_rec::history::Shown;
use ms_rec::navigation::same_map;
use ms_rec::worker::Verdict;

pub const GREEN: Color32 = Color32::from_rgb(0x23, 0x87, 0x47);
pub const RED: Color32 = Color32::from_rgb(0xd9, 0x38, 0x4f);
pub const GREY: Color32 = Color32::from_rgb(0x8a, 0x90, 0x99);

#[derive(Clone, Debug, PartialEq)]
pub struct Span {
    pub text: String,
    pub bold: bool,
    pub small: bool,
    pub color: Option<Color32>,
}

pub fn plain(t: impl Into<String>) -> Span {
    Span { text: t.into(), bold: false, small: false, color: None }
}

pub fn bold(t: impl Into<String>) -> Span {
    Span { bold: true, ..plain(t) }
}

pub fn small(t: impl Into<String>) -> Span {
    Span { small: true, ..plain(t) }
}

pub type Line = Vec<Span>;

/// One line: bold title, then small difficulty name and keymode.
pub fn map_line(title: &str, artist: Option<&str>, song: Option<&str>, version: Option<&str>, keys: i64) -> Line {
    let (mut name, mut diff) = (title.to_string(), version.map(str::to_string));
    if let Some(song) = song.filter(|s| !s.is_empty()) {
        name = [artist.filter(|a| !a.is_empty()), Some(song)].into_iter().flatten().collect::<Vec<_>>().join(" — ");
    } else if let (Some(i), true) = (name.find(" ["), name.trim_end().ends_with(']')) {
        let tail = name[i + 2..].trim_end();
        diff = Some(tail[..tail.len() - 1].to_string());
        name.truncate(i);
    }
    let name = name.strip_suffix(".osu").unwrap_or(&name).to_string();
    vec![bold(name), plain(" "), small(format!("{}{keys}K", diff.map_or(String::new(), |v| format!("[{v}] "))))]
}

pub fn pick_line(c: &Pick) -> Line {
    map_line(&c.title, c.artist.as_deref(), c.song.as_deref(), c.version.as_deref(), c.keys)
}

pub fn wanted_mod(rate: f64) -> &'static str {
    if (rate - 1.).abs() < 0.0005 { "NM" } else if rate < 1. { "HT" } else { "DT" }
}

/// The picked map: [rate to set (green once osu! shows it) and expectation, title], so they never move.
pub fn target_lines(c: &Pick, obs: &Obs, shown: &Shown) -> Vec<Line> {
    let wanted = wanted_mod(c.rate);
    let known = obs.source == Some("game") && obs.rate.is_some();
    let observed = obs.path.as_ref().and_then(|p| p.file_name()).map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let same = same_map(c, Some(&observed), None, obs.bid);
    let mods: Vec<String> = obs.mods.iter().flatten().filter_map(|m| m["acronym"].as_str().map(str::to_uppercase)).collect();
    let has = |set: &[&str]| mods.iter().any(|m| set.contains(&m.as_str()));
    let speed = has(&["DT", "NC", "HT", "DC", "WU", "WD", "AS"]);
    let mod_ok = match wanted { "NM" => !speed, "HT" => has(&["HT", "DC"]), _ => has(&["DT", "NC"]) };
    let correct = known && same && obs.rate.is_some_and(|a| (a - c.rate).abs() < 0.0005) && mod_ok;
    let colour = if correct { GREEN } else if known { RED } else { GREY };
    let expected = if same && c.sha.as_deref().is_some_and(|s| s != observed) { "Updated chart selected".to_string() } else { format!("~{:.1}%", 100. * shown.acc(c.pred.acc_mid)) };
    vec![vec![Span { text: format!("{} SET {:.2}× · {wanted}", if correct { "✓" } else { "→" }, c.rate), bold: true, small: false, color: Some(colour) }, plain("  "), bold(expected)], pick_line(c)]
}

/// 'warming up −1.2%': the form reason and what it does to this map's expected accuracy.
pub fn form_text(form: (Option<&str>, f64), acc_mid: f64, shown: &Shown) -> String {
    let (Some(reason), shift) = form else { return String::new() };
    let rested = 1. - (1. - acc_mid) * (-shift).exp();
    format!("{reason} {:+.1}%", 100. * (shown.acc(acc_mid) - shown.acc(rested))).replace('-', "−")
}

pub fn clock(seconds: f64) -> String {
    let s = seconds as i64;
    format!("{}:{:02}", s / 60, s % 60)
}

/// The selected map: [numbers, title, details], numbers at a fixed place whatever the title (user 2026-10-30). PP gain
/// leads the details line, so the numbers line never wraps. The full card is its tooltip.
pub fn card_lines(c: Option<&Card>, pp: bool, note: Option<&str>, shown: &Shown) -> (Vec<Line>, Option<String>) {
    let Some(c) = c else { return (vec![vec![small(note.unwrap_or("Select a map to see its prediction"))]], None) };
    let pct = |x: f64| 100. * shown.acc(x);
    let (mid, lo, hi) = (pct(c.acc_mid), pct(c.acc_lo), pct(c.acc_hi));
    let skill = c.skill.as_deref().and_then(ms_calc::skills::Sk::from_name).map_or_else(|| c.skill.clone().unwrap_or_default(), |s| ms_calc::card::display_name(s).to_string());
    let length = clock(c.length);
    let line = vec![small("SELECTED"), plain("  expected "), bold(format!("~{mid:.1}%")), plain(" "), small(format!("({lo:.1}–{hi:.1}%)")), plain("  "), bold(format!("{:.2}×", c.rate))];
    let rest = [length.clone(), skill.clone(), c.nps.map_or(String::new(), |n| format!("{n:.1} NPS"))].into_iter().filter(|x| !x.is_empty()).collect::<Vec<_>>().join(" · ");
    let mut details = Vec::new();
    let form = form_text(c.form, c.acc_mid, shown);
    if !form.is_empty() {
        details.extend([Span { bold: true, ..small(form.clone()) }, small(" · ")]);
    }
    if let (true, true, Some((_, _, best, gain, up))) = (pp, c.ranked, c.pp) {
        details.extend([Span { bold: true, ..small(format!("+{gain:.1}pp")) }, small(format!(" · {:.0}% to {} · ", 100. * up, if best.is_some() { "beat" } else { "count" }))]);
    }
    details.push(small(rest));
    let mut tip = vec![format!("Expected ~{mid:.1}%, usually {lo:.1}–{hi:.1}%"), format!("{:.2}× · {length} · {skill}", c.rate)];
    if !form.is_empty() {
        tip.insert(1, format!("Form now: {form} against your rested level on this map"));
    }
    match c.pp.filter(|_| c.ranked) {
        None => tip.push("No pp (unranked or unranked rate)".into()),
        Some((low, high, best, gain, up)) => tip.push(format!("{low:.0}–{high:.0}pp · {}+{gain:.1}pp expected · {:.0}% to {}", best.map_or(String::new(), |b| format!("best {b:.0} · ")), 100. * up, if best.is_some() { "beat" } else { "count" })),
    }
    if let Some(n) = c.nps {
        tip.push(format!("{n:.1} raw NPS"));
    }
    if let Some((stars, n)) = c.plays_like {
        // User 2026-10-02 could not tell whose stars or whose scores ("7.4* osu SR? or our SR?").
        tip.push(format!("Top-1000 players score on it like a {stars:.1}★ map (ManiaScope ★, their best of {} scores at {:.2}×)", ms_rec::engine::commas(n as usize), c.rate));
    }
    let title = map_line(&c.title, c.artist.as_deref(), c.song.as_deref(), c.version.as_deref(), c.keys);
    (vec![line, title, details], Some(tip.join("\n")))
}

pub fn verdict_words(z: f64) -> &'static str {
    if z > 1.5 { "way above expected" } else if z > 0.5 { "better than expected" } else if z < -1.5 { "way below expected" } else if z < -0.5 { "below expected" } else { "about as expected" }
}

/// The last play: [rate · news, big accuracy vs expected, title]. No clock time: "00:44" read as the song's length (user
/// 2026-10-31).
pub fn result_lines(v: &Verdict) -> Vec<Line> {
    let mut head = vec![v.rate.map_or("unknown rate".to_string(), |r| format!("{r:.2}×"))];
    head.extend(v.news.iter().cloned());
    let mut acc = vec![bold(format!("{:.2}%", 100. * v.acc))];
    if let Some(e) = v.expected {
        acc.push(plain(format!(" · expected ~{:.2}%", 100. * e)));
    }
    if let Some(z) = v.z {
        // Words, not σ (user 2026-10-31 preferred "better than expected").
        let colour = if z > 0.5 { GREEN } else if z < -0.5 { RED } else { GREY };
        acc.extend([plain(" · "), Span { text: verdict_words(z).into(), bold: true, small: false, color: Some(colour) }]);
    }
    vec![vec![small("LAST PLAY"), plain("  "), plain(head.join(" · "))], acc, vec![bold(v.title.clone())]]
}

pub fn keys_label(keys: &[i64]) -> String {
    if keys == ms_rec::catalog::SUPPORTED_KEYS.collect::<Vec<_>>() {
        return "All".into();
    }
    if keys.len() >= 3 && keys.windows(2).all(|w| w[1] == w[0] + 1) {
        return format!("{}K–{}K", keys[0], keys[keys.len() - 1]);
    }
    if keys.is_empty() { "None".into() } else { keys.iter().map(|k| format!("{k}K")).collect::<Vec<_>>().join(", ") }
}

pub fn size(n: f64) -> String {
    if n >= 1e9 { format!("{:.1} GB", n / 1e9) } else { format!("{:.0} MB", n / 1e6) }
}

pub fn duration(s: f64) -> String {
    let m = (s / 60.) as i64;
    if m < 1 { "under a minute".into() } else if m < 60 { format!("{m} min") } else if m < 1440 { format!("{} h {} min", m / 60, m % 60) } else { format!("{} d {} h", m / 1440, m / 60 % 24) }
}

/// Context line of the main window: dan before the rate, so the rate note goes first when the line is cut.
pub fn context(a: &Analysis, dan: Option<&str>, mode: &str) -> String {
    format!("{}K{}  ·  {:.2}×{mode}{}", a.keys, dan.map_or(String::new(), |d| format!("  ·  {d}")), a.rate, if a.sv_ignored { "  ·  SV ignored (Constant Speed)" } else { "" })
}

// osu!lazer star badge (osu.Game/Graphics/OsuColour.cs): the pill takes STAR, its text is 75 % black below 6.5 ★, Orange1 up
// to 9 ★, then STAR_TEXT. From 10 ★ the text runs as a smooth gradient from 9 ★ yellow up to its own colour. NPS badge:
// WoW item-rarity colours centred on each 10 NPS band and blended between them.
const STAR: [(f64, u32); 12] = [(0.1, 0xaaaaaa), (0.1, 0x4290fb), (1.25, 0x4fc0ff), (2.0, 0x4fffd5), (2.5, 0x7cff4f), (3.3, 0xf6f05c), (4.2, 0xff8068), (4.9, 0xff4e6f), (5.8, 0xc645b8), (6.7, 0x6563de), (7.7, 0x18158e), (9.0, 0x000000)];
const STAR_TEXT: [(f64, u32); 5] = [(9.0, 0xf6f05c), (9.9, 0xff8068), (10.6, 0xff4e6f), (11.5, 0xc645b8), (12.4, 0x6563de)];
const NPS: [(f64, u32); 6] = [(5., 0x9d9d9d), (15., 0x1eff00), (25., 0x0070dd), (35., 0xa335ee), (45., 0xff8000), (55., 0xff3c00)];

fn rgb(c: u32) -> Color32 {
    Color32::from_rgb((c >> 16) as u8, (c >> 8) as u8, c as u8)
}

fn mix(a: u32, b: u32, f: f64) -> u32 {
    [16, 8, 0].iter().fold(0, |out, s| {
        let (x, y) = (f64::from((a >> s) & 255), f64::from((b >> s) & 255));
        out | (((x * (1. - f) + y * f).round() as u32) << s)
    })
}

fn sample(spec: &[(f64, u32)], x: f64) -> u32 {
    if x <= spec[0].0 {
        return spec[0].1;
    }
    for w in spec.windows(2) {
        let ((a, ca), (b, cb)) = (w[0], w[1]);
        if x <= b {
            return mix(ca, cb, if b > a { (x - a) / (b - a) } else { 1. });
        }
    }
    spec[spec.len() - 1].1
}

/// (text colours left to right, pill colour) of the big rating.
pub fn rating_colours(v: f64) -> (Vec<Color32>, Color32) {
    let v = ms_calc::py::round_digits(v, 2);
    let text = if v < 6.5 {
        vec![Color32::from_rgba_unmultiplied(0, 0, 0, 0xbf)]
    } else if v < 9. {
        vec![rgb(0xffd966)]
    } else if v < 10. {
        vec![rgb(sample(&STAR_TEXT, v))]
    } else {
        STAR_TEXT.iter().filter(|x| x.0 < v).map(|x| rgb(x.1)).chain([rgb(sample(&STAR_TEXT, v))]).collect()
    };
    (text, rgb(sample(&STAR, v)))
}

/// The gradient runs over the number, " NPS" takes its last colour.
pub fn nps_colours(v: f64) -> Vec<Color32> {
    let lo = v - 10. * ((v - 20.) / 10.).clamp(0., 1.);
    std::iter::once(sample(&NPS, lo)).chain(NPS.iter().filter(|x| lo < x.0 && x.0 < v).map(|x| x.1)).chain([sample(&NPS, v)]).map(rgb).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line_text(l: &Line) -> String {
        l.iter().map(|s| s.text.as_str()).collect()
    }

    #[test]
    fn a_title_keeps_its_difficulty_apart() {
        assert_eq!(line_text(&map_line("A - B [Hard]", None, None, None, 7)), "A - B [Hard] 7K");
        assert_eq!(map_line("A - B [Hard]", None, None, None, 7)[0].text, "A - B");
        assert_eq!(line_text(&map_line("x.osu", Some("Art"), Some("Song"), Some("V"), 4)), "Art — Song [V] 4K");
    }

    #[test]
    fn keymodes_read_short() {
        assert_eq!(keys_label(&[4, 5, 6, 7, 8, 9, 10]), "All");
        assert_eq!(keys_label(&[4, 5, 6]), "4K–6K");
        assert_eq!(keys_label(&[4, 7]), "4K, 7K");
        assert_eq!(keys_label(&[]), "None");
    }

    #[test]
    fn spectrum_blends_between_stops() {
        assert_eq!(sample(&NPS, 10.), mix(0x9d9d9d, 0x1eff00, 0.5));
        assert_eq!(nps_colours(12.).len(), 2);
        assert_eq!(nps_colours(30.).len(), 3); // 20 to 30 runs through the 25 NPS stop
        assert_eq!(rating_colours(5.).0.len(), 1);
    }

    #[test]
    fn form_text_says_what_it_costs() {
        let shown = Shown { client: "lazer".into(), miss_ratio: 0.88 };
        assert_eq!(form_text((Some("warming up"), 0.2), 0.95, &shown), "warming up −0.9%");
        assert_eq!(form_text((None, 0.), 0.95, &shown), "");
    }
}
