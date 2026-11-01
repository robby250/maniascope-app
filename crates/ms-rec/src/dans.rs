//! Dan equivalent of a chart: where it falls among the community dan charts (calib/dans.json). Anchors are the dan charts
//! themselves, rated by this calculator, not another calculator's scale (user 2026-10-01). Only chart content decides the
//! series and the place; titles never enter (user: "no cheating").
//!
//! Series: LN-heavy charts are placed among LN dans, the rest among rice dans; 4K charts with sustained same-column runs
//! are also placed among vibro dans. A chart mostly made of vibro runs shows the vibro dan alone: its mashing reads as fast
//! rice/LN to the calculator (user 2026-10-01), unless the notes outside the runs are dense on their own: REFORM
//! Iota/Kappa are 26/43 % vibro yet hard rice besides (user 2026-10-11). Vibro dans are placed by vibro speed together with
//! stars: stars alone put Vibro 2–8 within 8.5–9.2★ out of order.
use ms_data::codec::from_json;
use ms_calc::val::Val;
use ms_model::predict::{Consts, Feats};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::OnceLock;

const TABLE: &str = include_str!("../../../calib/dans.json");
const VIBRO_GAP: f64 = 100.;  // ms: a column re-hit this fast keeps a vibro run going (10 hits/s)
const VIBRO_RUN: usize = 6;   // hits in one column before it counts as vibro, not a jack
pub const VIBRO_SHARE: f64 = 0.10; // share of notes in such runs: 117/120 vibro dan charts, 0.5% of ranked 4K (2026-10-01)
const VIBRO_ONLY: f64 = 0.25; // 44/48 vibro dan charts reach it; 4/113 REFORM charts, no LN dan chart
const VIBRO_REST: f64 = 13.;  // notes/s outside vibro runs: vibro dan charts <= 12.3, REFORM charts >= 13.5 at 10%+ vibro
const LN_SHARE: f64 = 0.22;   // LN dans' charts hold >= 25% LN notes, rice dans' <= 19% (2026-10-01)
const LN_MAJORITY: f64 = 0.5; // from here holds are most notes: the LN dan alone

pub fn series(keys: i64, kind: &str) -> Option<&'static str> {
    match (keys, kind) {
        (4, "rice") => Some("4K REFORM"),
        (4, "ln") => Some("4K LN"),
        (6, "rice") => Some("6K Regular"),
        (6, "ln") => Some("6K LN"),
        (7, "rice") => Some("7K Regular"),
        (7, "ln") => Some("7K LN"),
        (10, "rice") => Some("10K Regular"),
        _ => None,
    }
}

pub fn short(series: &str) -> &'static str {
    match series {
        "7K Regular" => "Reg ",
        "7K LN" | "4K LN" => "LN ",
        "6K Regular" => "6K ",
        "6K LN" => "6K LN ",
        "10K Regular" => "10K ",
        _ => "",
    }
}

type Tiers = Vec<(String, f64)>;

struct Data {
    series: HashMap<String, Tiers>,
    weights: (f64, f64),
}

fn data() -> &'static Data {
    static D: OnceLock<Data> = OnceLock::new();
    D.get_or_init(|| {
        let v: Value = serde_json::from_str(TABLE).unwrap_or(Value::Null);
        // Another calculator's ratings would misplace every chart.
        let v = if v["calc"] == crate::engine::CALC_ID { v } else { Value::Null };
        let series = v["series"].as_object().map_or_else(HashMap::new, |o| o.iter().map(|(k, tiers)| {
            (k.clone(), tiers.as_array().map_or_else(Vec::new, |t| t.iter().filter_map(|x| Some((x["tier"].as_str()?.to_string(), x["value"].as_f64()?))).collect()))
        }).collect());
        Data { series, weights: (v["vibro_weights"][0].as_f64().unwrap_or(1.), v["vibro_weights"][1].as_f64().unwrap_or(0.)) }
    })
}

/// The accuracy on a dan's songs, on the player's better plays after grinding, that reads as passing it, in every series.
/// Real passes are 96 % regular, 95 % LN (user 2026-11-01, no source online). One value: at 96 % for regular and 95 % for LN
/// on one shift, the user's 7K LN would read well above Azimuth (estimated from the tier table, not run).
pub const PEAK: f64 = 0.95;

pub fn tiers(series: &str) -> Option<&'static Tiers> {
    data().series.get(series)
}

/// {tier: [features at 1.0×]} of the series' dan songs, in tier order of the table.
pub fn songs(series: &str, c: &Consts) -> HashMap<String, Vec<Feats>> {
    // Read for Stats only: held for the app's life the parsed songs were 12 MB.
    let table: Value = serde_json::from_str(TABLE).unwrap_or(Value::Null);
    table["songs"][series].as_object().map_or_else(HashMap::new, |o| o.iter().map(|(tier, list)| {
        let feats = list.as_array().map_or_else(Vec::new, |l| l.iter().filter_map(|f| match from_json(f) {
            Val::Dict(d) => Feats::of(&d, c).ok(),
            _ => None,
        }).collect());
        (tier.clone(), feats)
    }).collect())
}

/// Rating at continuous tier index p (position's inverse), linear between and beyond the anchors.
pub fn rating_at(series: &str, p: f64) -> Option<f64> {
    let t = tiers(series).filter(|t| !t.is_empty())?;
    if t.len() == 1 {
        return Some(t[0].1.exp());
    }
    let j = (p.floor().max(0.) as usize).min(t.len() - 2);
    Some((t[j].1 + (p - j as f64) * (t[j + 1].1 - t[j].1)).exp())
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Vibro {
    pub share: f64,
    pub speed: f64,
    pub rest: f64,
}

/// Share of notes in same-column runs (>= VIBRO_RUN hits, <= VIBRO_GAP apart at this rate), their median hits per second,
/// and the other notes per second of chart. Rolls count too: their columns repeat just as fast.
pub fn vibro_runs(notes: &[(f64, usize)], rate: f64) -> Vibro {
    let mut cols: HashMap<usize, Vec<f64>> = HashMap::new();
    for &(start, col) in notes {
        cols.entry(col).or_default().push(start / rate);
    }
    let (mut inrun, mut gaps) = (0, Vec::new());
    for times in cols.values_mut() {
        times.sort_by(f64::total_cmp);
        let mut run = vec![times[0]];
        for &t in times[1..].iter().chain(std::iter::once(&f64::INFINITY)) {
            if t - run[run.len() - 1] <= VIBRO_GAP {
                run.push(t);
            } else {
                if run.len() >= VIBRO_RUN {
                    inrun += run.len();
                    gaps.extend(run.windows(2).map(|w| w[1] - w[0]));
                }
                run = vec![t];
            }
        }
    }
    gaps.sort_by(f64::total_cmp);
    let (lo, hi) = notes.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |a, n| (a.0.min(n.0), a.1.max(n.0)));
    let span = if notes.is_empty() { 0. } else { (hi - lo) / rate / 1000. };
    Vibro { share: inrun as f64 / notes.len().max(1) as f64, speed: if gaps.is_empty() { 0. } else { 1000. / gaps[gaps.len() / 2] },
            rest: if span != 0. { (notes.len() - inrun) as f64 / span } else { 0. } }
}

/// Mostly vibro runs with little else: the rice/LN dan would name the wrong skill.
pub fn vibro_only(v: &Vibro) -> bool {
    v.share >= VIBRO_ONLY && v.rest < VIBRO_REST
}

fn value(series: &str, rating: f64, vibro: Option<&Vibro>) -> f64 {
    value_with(data().weights, series, rating, vibro)
}

/// The quantity a series' anchors are in: log stars, or for vibro dans weighted log speed + log stars.
pub fn value_with(weights: (f64, f64), series: &str, rating: f64, vibro: Option<&Vibro>) -> f64 {
    if series == "4K Vibro" {
        return weights.0 * vibro.map_or(1f64, |v| v.speed.max(1.)).ln() + weights.1 * rating.max(1e-3).ln();
    }
    rating.max(1e-3).ln()
}

/// Continuous tier index (0 = first anchor) of value v, linear between and beyond the anchors.
pub fn position(t: &[(String, f64)], v: f64) -> f64 {
    if t.len() == 1 {
        return 0.;
    }
    let j = if v <= t[0].1 { 0 } else if v >= t[t.len() - 1].1 { t.len() - 2 } else { (0..t.len() - 1).find(|&i| t[i].1 <= v && v <= t[i + 1].1).unwrap_or(0) };
    let (lo, hi) = (t[j].1, t[j + 1].1);
    if hi - lo < 1e-9 { j as f64 + 0.5 } else { j as f64 + (v - lo) / (hi - lo) }
}

/// Nearest dan, with −/+ when the chart sits more than a quarter of the way to the next one.
pub fn place(series: &str, rating: f64, vibro: Option<&Vibro>) -> Option<String> {
    let t = tiers(series).filter(|t| !t.is_empty())?;
    let prefix = short(series);
    let p = position(t, value(series, rating, vibro));
    if p < -0.5 {
        return Some(format!("below {prefix}{}", t[0].0));
    }
    if p > t.len() as f64 - 0.5 {
        return Some(format!("{prefix}{}+", t[t.len() - 1].0));
    }
    let j = (p.round_ties_even().max(0.) as usize).min(t.len() - 1);
    let off = p - j as f64;
    Some(format!("{prefix}{}{}", t[j].0, if off > 0.25 { "+" } else if off < -0.25 { "−" } else { "" }))
}

pub fn kind(ln_share: f64) -> &'static str {
    if ln_share >= LN_SHARE { "ln" } else { "rice" }
}

pub fn label(keys: i64, rating: f64, ln_share: f64, vibro: Option<&Vibro>) -> Option<String> {
    // Vibro last: the rice/LN dan is the main reading (user 2026-10-01).
    let vibro = vibro.filter(|_| keys == 4);
    // Mixed charts (LN dan share but mostly taps) read on both, the tap dans first (user 2026-10-01: a 29 % LN rice
    // stamina chart showed only "LN Gamma").
    let kinds: Vec<&str> = if !(LN_SHARE..LN_MAJORITY).contains(&ln_share) { vec![kind(ln_share)] } else { vec!["rice", "ln"] };
    let mut parts: Vec<String> = kinds.iter().filter(|_| !vibro.is_some_and(vibro_only)).filter_map(|k| place(series(keys, k)?, rating, None)).collect();
    parts.extend(vibro.filter(|v| v.share >= VIBRO_SHARE).and_then(|v| place("4K Vibro", rating, Some(v))));
    if parts.is_empty() { None } else { Some(parts.join(" · ")) }
}

/// The dan series kind a Stats skill is placed on: LN skills on the LN dans.
pub fn ability_kind(skill: &str) -> &'static str {
    if skill.split('/').next() == Some("ln") { "ln" } else { "rice" }
}

/// Dan whose charts sit at a player's ~94 % difficulty for this Stats skill. scale: {kind: factor} from the player's
/// predicted accuracy on the dan songs; without it the level's rating is placed. None for vibro (its dans need a vibro
/// speed) and keymodes without dans.
pub fn ability(keys: i64, value: f64, skill: &str, scale: &HashMap<String, f64>) -> Option<String> {
    if skill.ends_with("vibro") {
        return None;
    }
    let k = ability_kind(skill);
    place(series(keys, k)?, value * scale.get(k).copied().unwrap_or(1.), None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ladder_reads_its_own_anchors() {
        let t = tiers("7K Regular").expect("7K dans");
        assert_eq!(place("7K Regular", t[3].1.exp(), None), Some(format!("Reg {}", t[3].0)));
        assert!((rating_at("7K Regular", 3.).unwrap() - t[3].1.exp()).abs() < 1e-9);
    }

    #[test]
    fn a_long_vibro_run_counts_and_a_short_one_does_not() {
        let run: Vec<(f64, usize)> = (0..10).map(|i| (f64::from(i) * 50., 0)).collect();
        assert_eq!(vibro_runs(&run, 1.).share, 1.);
        assert_eq!(vibro_runs(&run[..4], 1.).share, 0.);
    }
}

#[cfg(test)]
mod shipped_tests {
    use super::*;

    #[test]
    fn shipped_dan_table_is_stamped_for_this_calculator() {
        // A calculator change without a new table hides every dan reading.
        assert!(tiers("7K Regular").is_some_and(|t| !t.is_empty()));
    }
}
