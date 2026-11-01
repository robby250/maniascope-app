//! Calculator behaviour on generated .osu files.
mod common;
use common::*;
use ms_calc::card::{card, ranked};
use ms_calc::chart::{parse, ChartError, Chart};
use ms_calc::compute::compute;
use ms_calc::features::standard_params;
use ms_calc::params::Params;
use ms_calc::skills::{skills, Sk};
use ms_calc::val::{Dict, Val};
use serde_json::{json, Value};

const DEFAULT_TIMING: &str = "0,500,4,1,0,100,1,0";

/// (ms, column, hold end).
type Note = (i64, usize, Option<i64>);

fn osu(notes: &[Note], keys: usize, mode: i32, shuffle: bool, timing: &[String]) -> Vec<u8> {
    let mut lines: Vec<String> = notes.iter().map(|&(t, c, end)| {
        let x = ((c as f64 + 0.5) * 512. / keys as f64) as i64;
        match end {
            Some(e) => format!("{x},192,{t},128,0,{e}:0:0:0:0:"),
            None => format!("{x},192,{t},1,0,0:0:0:0:"),
        }
    }).collect();
    if shuffle {
        lines.reverse();
    }
    format!("osu file format v14\n\n[General]\nMode: {mode}\n\n[Metadata]\nTitle:t\nVersion:v\n\n[Difficulty]\nCircleSize:{keys}\n\n[TimingPoints]\n{}\n\n[HitObjects]\n{}\n",
            timing.join("\n"), lines.join("\n")).into_bytes()
}

fn file(notes: &[Note], keys: usize) -> Chart {
    parse(&osu(notes, keys, 3, false, &[DEFAULT_TIMING.to_string()])).unwrap()
}

fn res(notes: &[Note], keys: usize) -> Dict {
    run(&file(notes, keys), 1.)
}

fn sc(notes: &[Note], keys: usize) -> Dict {
    res(notes, keys).dict("scores").unwrap().clone()
}

fn empty_model() -> Params {
    Params::new(json!({}), &Value::Null).unwrap()
}

fn top(notes: &[Note], keys: usize, n: usize) -> Vec<&'static str> {
    ranked(&res(notes, keys)).into_iter().take(n).map(Sk::name).collect()
}

fn name_of(e: &Val) -> String {
    let Val::Dict(d) = e else { panic!() };
    match d.get("name") {
        Some(Val::Str(s)) => s.clone(),
        _ => panic!(),
    }
}

fn parts_of(e: &Val) -> Vec<String> {
    let Val::Dict(d) = e else { panic!() };
    match d.get("parts") {
        Some(Val::List(l) | Val::Tuple(l)) => l.iter().map(|p| if let Val::Str(s) = p { s.clone() } else { panic!() }).collect(),
        _ => panic!(),
    }
}

fn names(notes: &[Note], keys: usize) -> Vec<String> {
    card(&res(notes, keys)).iter().map(name_of).collect()
}

fn stream(n: i64, gap: i64, keys: i64) -> Vec<Note> {
    stream_at(n, gap, keys, 1000)
}

fn stream_at(n: i64, gap: i64, keys: i64, start: i64) -> Vec<Note> {
    (0..n).map(|i| (start + gap * i, (i % keys) as usize, None)).collect()
}

/// Rows of `gap` ms cycling through column patterns.
fn rows(pattern: &[&[usize]], gap: i64, n: i64) -> Vec<Note> {
    (0..n).flat_map(|i| pattern[i as usize % pattern.len()].iter().map(move |&c| (1000 + gap * i, c, None))).collect()
}

/// One note per row, hands alternating irregularly, gaps (ms) cycling.
fn dump(gaps: &[i64], n: usize, cols: &[usize]) -> Vec<Note> {
    let mut t = 1000;
    (0..n).map(|i| {
        let note = (t, cols[i % cols.len()], None);
        t += gaps[i % gaps.len()];
        note
    }).collect()
}

const DUMP_COLS: [usize; 12] = [0, 3, 1, 2, 3, 0, 2, 1, 0, 2, 3, 1];

#[test]
fn parsing_and_rate() {
    let base = stream(1200, 100, 7);
    let a = sc(&base, 7);
    assert!(a.f("overall") > 1. && a.values_f().iter().all(|v| *v >= 0.));
    // simultaneous-line order and absolute offset do not matter
    let chords: Vec<Note> = (0..300).flat_map(|i| [i as usize % 3, 4, 6].map(|c| (1000 + 150 * i, c, None))).collect();
    let shuffled = parse(&osu(&chords, 7, 3, true, &[DEFAULT_TIMING.to_string()])).unwrap();
    assert_eq!(sc(&chords, 7).values_f(), run(&shuffled, 1.).dict("scores").unwrap().values_f());
    assert!((sc(&stream_at(1200, 100, 7, 91234), 7).f("overall") - a.f("overall")).abs() < 1e-9);
    // rate applied exactly once: 100 ms stream at 1.25x == 80 ms stream at 1.0x, plus the rate factor. The physical
    // core invariant, before the fitted score-link transform.
    let raw = empty_model();
    let (r, g) = (1.25f64, raw.k.RATE_G);
    let fast = run_with(&file(&base, 7), r, &raw).dict("scores").unwrap().f("overall");
    let sped = run_with(&file(&stream(1200, 80, 7), 7), 1., &raw).dict("scores").unwrap().f("overall");
    assert!((fast / r.powf(g) - sped).abs() < 1e-6);
    // every keymode parses; non-mania is explicit
    for k in 4..=10 {
        assert!(sc(&stream(400, 100, k as i64), k).f("overall") > 0.);
    }
    assert!(matches!(parse(&osu(&base, 7, 0, false, &[DEFAULT_TIMING.to_string()])), Err(ChartError::Chart(_))));
    // short charts are rated, not zeroed; one note is simply 0
    let lns: Vec<Note> = (0..7).map(|i| (1000 + 100 * i, i as usize % 7, Some(1050 + 100 * i))).collect();
    assert!(sc(&lns, 7).f("overall") > 0.);
    assert_eq!(sc(&[(1000, 0, None)], 7).f("overall"), 0.);
}

#[test]
fn rate_sweep_is_smooth_and_uncapped() {
    let notes: Vec<Note> = (0..900i64).flat_map(|i| {
        let mut cols = vec![(i % 7) as usize];
        if i % 4 == 0 {
            cols.push(((i + 3) % 7) as usize);
        }
        cols.into_iter().map(move |c| (1000 + 120 * i, c, None))
    }).collect();
    let c = file(&notes, 7);
    let mut previous = 0.;
    for r in (70..=200).step_by(2) {
        let v = run(&c, r as f64 / 100.).dict("scores").unwrap().f("overall");
        // 0.5 s bins shift under the notes as rate changes: a sub-display wobble. This chart dips 1.1 % at 1.70→1.72×.
        assert!(previous == 0. || (previous * 0.985 < v && v < previous * 1.07), "{r} {previous} {v}");
        previous = v;
    }
}

#[test]
fn horizon_units_match_rate_adjusted_overall() {
    let raw = empty_model();
    let (a, b) = (run_with(&file(&stream(1200, 100, 7), 7), 1.25, &raw), run_with(&file(&stream(1200, 80, 7), 7), 1., &raw));
    let factor = 1.25f64.powf(raw.k.RATE_G);
    let level = |r: &Dict, h: f64| match r.get("levels") {
        Some(Val::Dict(d)) => d.0.iter().find(|(k, _)| matches!(k, ms_calc::val::Key::F(x) if *x == h)).map(|x| x.1.f()).unwrap(),
        _ => panic!(),
    };
    let Some(Val::Dict(levels)) = a.get("levels") else { panic!() };
    assert!(!levels.0.is_empty());
    for (h, value) in &levels.0 {
        let ms_calc::val::Key::F(h) = h else { panic!() };
        assert!((value.f() / factor / level(&b, *h) - 1.).abs() < 1e-10, "{h}");
    }
    let share = |r: &Dict| level(r, 120.) / r.dict("scores").unwrap().f("overall");
    assert!((share(&a) / share(&b) - 1.).abs() < 1e-10);
    let feature = ms_calc::features::features_of(&a, 1.25, &raw);
    assert!((feature.f("stam") / (share(&a) / factor) - 1.).abs() < 1e-10);
}

#[test]
fn recovery_and_burst_vs_sustained() {
    let shift = |notes: &[Note], from: usize| -> Vec<Note> { notes.iter().enumerate().map(|(i, n)| (n.0 + if i >= from { 60000 } else { 0 }, n.1, n.2)).collect() };
    let base = stream(1200, 100, 7);
    let (a, b) = (sc(&base, 7), sc(&shift(&base, 600), 7));
    assert!(b.f("stamina") < a.f("stamina") && b.f("overall") <= a.f("overall"));
    // same pattern: 3 s burst in a 2 min map < 3 min sustained; easy padding does not erase a bottleneck
    let mut burst = stream(30, 100, 7);
    burst.push((125000, 0, None));
    assert!(sc(&burst, 7).f("overall") < 0.9 * sc(&stream(1800, 100, 7), 7).f("overall"));
    let mut padded = stream(600, 400, 7);
    padded.extend(stream_at(100, 70, 7, 300000));
    assert!(sc(&padded, 7).f("overall") > 1.3 * sc(&stream(600, 400, 7), 7).f("overall"));
    let mut sprint = stream(60, 70, 7);
    sprint.push((125000, 0, None));
    assert_eq!(res(&sprint, 7).f("horizon"), 2.);
    // Stamina is endurance, not a relabelled overall: a short burst has little, a long even run nearly all of its
    // rating; rest never adds to it
    let (short, marathon) = (sc(&stream(60, 100, 7), 7), sc(&stream(3600, 100, 7), 7));
    assert!(short.f("stamina") < 0.6 * short.f("overall") && marathon.f("stamina") > 0.95 * marathon.f("overall"));
    assert!(sc(&shift(&stream(3600, 100, 7), 1800), 7).f("stamina") < marathon.f("stamina"));
}

#[test]
fn hand_patterns() {
    let col = |c: usize, gap: i64, n: i64| -> Vec<Note> { (0..n).map(|i| (1000 + gap * i, c, None)).collect() };
    // same row rate, different hands: one-hand trill > 7-key staircase roll
    let trill: Vec<Note> = (0..600).map(|i| (1000 + 100 * i, (i % 2) as usize, None)).collect();
    assert!(sc(&trill, 7).f("overall") > 1.2 * sc(&stream(600, 100, 7), 7).f("overall"));
    // single-column jack is Jack (7K) / Jackspeed (4K); repeated chords are chordjack
    assert_eq!(top(&col(2, 150, 400), 7, 1), ["jack"]);
    assert_eq!(top(&col(2, 150, 400), 4, 1), ["jackspeed"]);
    // an anchor with the other hand in between is still a jack in 7K (the hand repeats)
    let anchor: Vec<Note> = (0..800i64).map(|i| (1000 + 120 * i, if i % 2 == 1 { 1 } else { (5 + (i / 2) % 2) as usize }, None)).collect();
    assert!(top(&anchor, 7, 2).contains(&"jack"));
    let cj = rows(&[&[0, 3, 4, 5], &[0, 2, 4, 6]], 160, 400);
    assert_eq!(top(&cj, 7, 1), ["chordjack"]);
    assert!(!top(&cj, 7, 3).contains(&"bracket"));
    // chords without repeats are chordstream, fast light rice is delay (7777's Delay packs)
    let hs = sc(&rows(&[&[1, 3, 6], &[0, 2, 5]], 160, 400), 7);
    assert!(hs.f("chordstream") > hs.f("chordjack"));
    let rice = |gap: i64| -> Vec<Note> {
        (0..1500i64).flat_map(|i| {
            let mut cols = vec![((i * 3) % 7) as usize];
            if i % 3 == 0 {
                cols.push(((i * 3 + 4) % 7) as usize);
            }
            cols.into_iter().map(move |c| (1000 + gap * i, c, None))
        }).collect()
    };
    assert_eq!(top(&rice(60), 7, 1), ["delay"]);
    // the same texture slower is Stream, not Delay
    assert_eq!(top(&rice(100), 7, 1), ["stream"]);
    // bracket: each hand trills between disjoint finger groups, one a chord (13 / 2)
    let br = rows(&[&[1, 5], &[0, 2, 4, 6]], 120, 800);
    assert!(top(&br, 7, 3).contains(&"bracket"));
    assert!(sc(&br, 7).f("bracket") > 2. * sc(&stream(800, 120, 7), 7).f("bracket"));
    // disjoint chord moves that never interleave are chordstream (user 2026-10-11: cigarettes)
    let cs = rows(&[&[0, 1, 4, 5], &[2, 6]], 120, 800);
    assert_eq!(top(&cs, 7, 1), ["chordstream"]);
    assert!(!top(&cs, 7, 3).contains(&"bracket"));
    // a smooth roll is not technical; broken rhythm is
    assert!(sc(&stream(600, 100, 7), 7).f("technical") < 0.25);
    let swing: Vec<Note> = (0..600).map(|i| (1000 + 100 * i + if i % 3 == 1 { 37 } else { 0 }, (i % 7) as usize, None)).collect();
    assert!(sc(&swing, 7).f("technical") > 1. && top(&swing, 7, 3).contains(&"technical"));
    // smooth fast rolls back onto the 1/4 grid and tap -> LN accents are not technical
    let roll: Vec<Note> = (0..300).flat_map(|i| [(0, 0), (19, 1), (38, 2), (57, 3), (150, 5), (225, 6)].map(|(dt, c)| (1000 + 300 * i + dt, c, None))).collect();
    assert!(sc(&roll, 7).f("technical") < 0.25);
    let accent: Vec<Note> = (0..500i64).flat_map(|i| [(1000 + 150 * i, ((i * 3) % 7) as usize, None), (1075 + 150 * i, ((i * 3) % 7) as usize, Some(1135 + 150 * i))]).collect();
    assert!(sc(&accent, 7).f("technical") < 0.25);
}

#[test]
fn sv_reading() {
    let base = stream(1200, 100, 7);
    let plain = sc(&base, 7);
    let timed = |timing: &[String]| parse(&osu(&base, 7, 3, false, timing)).unwrap();
    // decorative lines (SV 1, or changes after the last note) change nothing
    let harmless = ["0,500,4,1,0,100,1,0", "5000,-100,4,1,0,100,0,0", "999000,-25,4,1,0,100,0,0"].map(String::from);
    assert_eq!(run(&timed(&harmless), 1.).dict("scores").unwrap().values_f(), plain.values_f());
    // scroll jumping between 0.5x and 2x while notes approach is reading work
    let mut active = vec![DEFAULT_TIMING.to_string()];
    active.extend((0..300).map(|k| format!("{},{},4,1,0,100,0,0", 1000 + 400 * k, if k % 2 == 1 { -200 } else { -50 })));
    let sv = run(&timed(&active), 1.);
    assert!(score(&sv, "overall") > 1.05 * plain.f("overall") && score(&sv, "sv") > 0.);
    assert_eq!(sv.get("sv"), Some(&Val::Str("included".into())));
    // Constant Speed: the same chart without scroll changes
    let cs = compute(&timed(&active), 1., false, &standard_params()).unwrap();
    assert!((score(&cs, "overall") - plain.f("overall")).abs() < 1e-9);
    assert_eq!(cs.get("sv"), Some(&Val::Str("ignored".into())));
}

#[test]
fn ln_interaction() {
    let taps = |hold: fn(i64, usize) -> Option<i64>| -> Vec<Note> { (0..300).flat_map(|i| [0, 1].map(|c| (1000 + 300 * i, c, hold(1000 + 300 * i, c)))).collect() };
    let (a, b, c) = (sc(&taps(|_, _| None), 7), sc(&taps(|t, _| Some(t + 250)), 7), sc(&taps(|t, c| Some(t + 250 - 70 * c as i64)), 7));
    assert!(a.f("ln") == 0. && a.f("overall") < b.f("overall") && b.f("overall") < c.f("overall"));
    // a long simple hold is not harder than a shorter one
    let holds = |len: i64| -> Vec<Note> { (0..40).map(|i| (1000 + 3000 * i, 3, Some(1000 + len + 3000 * i))).collect() };
    assert!((sc(&holds(2400), 7).f("overall") - sc(&holds(1400), 7).f("overall")).abs() < 0.12);
    // tapping under a same-hand hold costs more than under an other-hand hold
    let under = |low: i64| -> Vec<Note> {
        let mut n: Vec<Note> = (0..30).map(|i| (1000 + 4000 * i, 0, Some(4900 + 4000 * i))).collect();
        n.extend((0..600).filter(|i| (200 * i) % 4000 > 200).map(|i| (1000 + 200 * i, (low + i % 2) as usize, None)));
        n
    };
    assert!(sc(&under(1), 7).f("ln") > 1.3 * sc(&under(5), 7).f("ln"));
}

#[test]
fn mashed_rolls_read_as_vibro_not_stream() {
    // "Vibro Dansen": a 4-key roll 21 ms apart, short holds, each column every 83 ms, is mashed
    let order = [2, 3, 0, 1];
    let roll: Vec<Note> = (0..1600).map(|i| (1000 + 21 * i, order[i as usize % 4], Some(1000 + 21 * i + 42))).collect();
    let r = res(&roll, 4);
    assert_eq!(name_of(&card(&r)[0]), "Vibro");
    assert!(score(&r, "vibro") > 2. * score(&r, "stream").max(score(&r, "dump")).max(score(&r, "ln")));
    // the same roll at a hittable 60 ms per row stays rice
    let slow: Vec<Note> = (0..600).map(|i| (1000 + 60 * i, order[i as usize % 4], None)).collect();
    assert!(sc(&slow, 4).f("vibro") < 0.1);
}

#[test]
fn card_taxonomy() {
    // a long single-column run is a Longjack (it replaces Jack); a lone pair inside rice is a Minijack
    let long: Vec<Note> = (0..400).map(|i| (1000 + 150 * i, 2, None)).collect();
    assert_eq!(names(&long, 7)[0], "Longjack");
    let mini: Vec<Note> = (0..1200i64).map(|i| (1000 + 110 * i, (if i % 4 < 2 { (i / 2) * 3 % 7 } else { (i * 3) % 7 }) as usize, None)).collect();
    let n = names(&mini, 7);
    assert!(n.iter().any(|x| x == "Minijack") && !n.iter().any(|x| x == "Longjack"));
    // tap -> hold in the same column (1/4 apart) is Shield; release -> re-press right away is Inverse
    let shield: Vec<Note> = (0..300i64).flat_map(|i| [(1000 + 400 * i, (i % 7) as usize, None), (1100 + 400 * i, (i % 7) as usize, Some(1350 + 400 * i))]).collect();
    assert!(names(&shield, 7).iter().any(|x| x == "Shield"));
    let inverse: Vec<Note> = (0..300).flat_map(|i| [1, 5].map(|c| (1000 + 300 * i, c, Some(1250 + 300 * i)))).collect();
    assert!(names(&inverse, 7).iter().any(|x| x == "Inverse"));
    // broken rhythm reads as Tech <pattern>, never a bare Technical; a component never repeats
    let swing: Vec<Note> = (0..600).map(|i| (1000 + 100 * i + if i % 3 == 1 { 37 } else { 0 }, (i % 7) as usize, None)).collect();
    let entries = card(&res(&swing, 7));
    assert!(name_of(&entries[0]).starts_with("Tech ") && !entries.iter().any(|e| name_of(e) == "Technical"));
    let mut parts: Vec<String> = entries.iter().flat_map(parts_of).collect();
    let all = parts.len();
    parts.sort();
    parts.dedup();
    assert_eq!(parts.len(), all);
}

#[test]
fn chord_trills_and_dump() {
    // names only: overall comes from the same demand whatever the rows are called
    assert_eq!(top(&rows(&[&[0, 1], &[2, 3]], 90, 800), 4, 1), ["jumptrill"]);
    assert_eq!(top(&rows(&[&[0, 2], &[1, 3]], 90, 800), 4, 1), ["splittrill"]);
    assert_eq!(top(&rows(&[&[0, 1, 2], &[4, 5, 6]], 90, 800), 7, 1), ["jumptrill"]);
    assert!(!skills(7).contains(&Sk::splittrill) && !skills(7).contains(&Sk::dump), "7K: Bracket, Delay");
    assert!(["chordstream", "bracket"].contains(&top(&rows(&[&[0, 2, 4], &[1, 3, 5]], 90, 800), 7, 1)[0]));
    // A B A inside jumpstream is not a trill; A B A B is
    let js = rows(&[&[0, 2], &[1, 3], &[0, 2], &[1], &[3], &[0, 1], &[2], &[3]], 90, 800);
    assert_eq!(sc(&js, 4).f("splittrill"), 0.);
    // 4K Dump: fast one-finger-per-hand rice, overlapping the stream family (Delay's rule)
    let (fast, slow) = (sc(&stream(1200, 45, 4), 4), sc(&stream(1200, 110, 4), 4));
    assert!(fast.f("dump") >= 0.9 * fast.f("stream") && slow.f("dump") == 0.);
}

#[test]
fn changing_jumptrill() {
    let changing: &[&[usize]] = &[&[0, 1], &[4, 5], &[1, 2], &[5, 6]]; // 12 / 56 / 23 / 67
    for keys in [7, 8] {
        let c = file(&rows(changing, 90, 400), keys);
        // Recognition does not change physical demand. The score calibration is pattern-sensitive, checked separately.
        let result = run_with(&c, 1., &empty_model());
        assert_eq!(ranked(&result)[0].name(), "jumptrill");
        let plain = run_with(&c, 1., &Params::new(json!({}), &json!({"TRILL_GAP": 0})).unwrap());
        for k in ["overall", "stamina"] {
            assert_eq!(score(&result, k), score(&plain, k));
        }
        for k in ["levels", "horizon", "timeline"] {
            assert_eq!(result.get(k), plain.get(k));
        }
    }
    assert_eq!(top(&rows(&[&[0, 1, 2], &[4, 5], &[1, 2], &[4, 5, 6]], 90, 400), 7, 1), ["jumptrill"]);
    // Alternating hands already costs far less than repeating chords at the same row speed.
    let chordjack: &[&[usize]] = &[&[0, 1], &[0, 1], &[1, 2], &[1, 2]];
    assert!(sc(&rows(chordjack, 100, 400), 7).f("overall") > 2. * sc(&rows(changing, 100, 400), 7).f("overall"));
    let same_hand: &[&[usize]] = &[&[0, 1], &[1, 2], &[4, 5], &[5, 6]];
    let single: &[&[usize]] = &[&[0, 1], &[4], &[1, 2], &[5, 6]];
    let both_hands: &[&[usize]] = &[&[0, 4], &[1, 5], &[2, 6], &[1, 5]];
    for pattern in [same_hand, single, both_hands] {
        assert_eq!(sc(&rows(pattern, 90, 400), 7).f("jumptrill"), 0.);
    }
    assert_eq!(sc(&rows(changing, 90, 3), 7).f("jumptrill"), 0.);
    assert_eq!(sc(&rows(changing, 220, 400), 7).f("jumptrill"), 0.);
}

#[test]
fn fast_changing_chords_lose_mirror_discount() {
    let changing = rows(&[&[1, 2, 5, 6], &[0, 1, 4, 5]], 110, 400);
    let fixed = rows(&[&[0, 1, 4, 5]], 110, 400);
    let slow: Vec<Note> = changing.iter().map(|n| (n.0 * 2, n.1, None)).collect();
    let mirror = Params::new(standard_params().model.clone(), &json!({"MIRROR": 1.0})).unwrap();
    let with = |notes: &[Note]| score(&run_with(&file(notes, 7), 1., &mirror), "overall");
    assert_eq!(with(&changing), sc(&changing, 7).f("overall"), "fast partial repeats are separate movements");
    assert!(with(&fixed) > sc(&fixed, 7).f("overall"), "complete repeats still share one bounce");
    assert!(with(&slow) > sc(&slow, 7).f("overall"), "slower changing chords keep the discount");
}

#[test]
fn rhythm_tech() {
    // 1/4 <-> 1/6 switching inside fast rice is rhythm tech; the same rice on one pulse is not
    let switching = dump(&[40, 40, 40, 60, 60], 900, &DUMP_COLS);
    let (mixed, even) = (sc(&switching, 4), sc(&dump(&[50], 900, &DUMP_COLS), 4));
    assert!(mixed.f("rhythmtech") > 0.8 * mixed.f("overall") && even.f("rhythmtech") == 0.);
    // a whole multiple of the pulse stays on it (50 50 50 150); a long roll changing speed is one sweep
    assert_eq!(sc(&dump(&[50, 50, 50, 150], 900, &DUMP_COLS), 4).f("rhythmtech"), 0.);
    let mut t = 1000;
    let roll: Vec<Note> = (0..900usize).map(|i| {
        let note = (t, 6 - i % 7, None);
        t += [25, 37, 50][(i / 14) % 3];
        note
    }).collect();
    let r = sc(&roll, 7);
    assert!(r.f("rhythmtech") < 0.2 * r.f("overall"));
    // scroll changes do not move Tech: the same notes with and without SV
    let mut sv = vec![DEFAULT_TIMING.to_string()];
    sv.extend((1..30).map(|i| format!("{},-{},4,1,0,100,0,0", 2000 * i, 50 + 50 * (i % 3))));
    let raw = empty_model();
    let (a, b) = (run_with(&file(&switching, 4), 1., &raw), run_with(&parse(&osu(&switching, 4, 3, false, &sv)).unwrap(), 1., &raw));
    assert!(score(&b, "sv") > 0. && (score(&a, "rhythmtech") - score(&b, "rhythmtech")).abs() < 1e-6 * score(&a, "rhythmtech"));
}

#[test]
fn pattern_tech() {
    // a minijack inserted into fast moving rice is pattern tech; the same rice without it, or every note doubled, is not
    let plain = sc(&dump(&[60], 900, &[0, 2, 1, 3, 0, 3, 1, 2]), 4);
    let inserted = sc(&dump(&[60], 900, &[0, 2, 1, 3, 3, 0, 1, 2, 0, 3, 1, 2]), 4); // 3 3: entered and left from motion
    let doubled = sc(&dump(&[60], 900, &[0, 0, 2, 2, 1, 1, 3, 3]), 4);
    assert!(plain.f("patterntech") == 0. && doubled.f("patterntech") == 0.);
    assert!(inserted.f("patterntech") > 0.8 * inserted.f("overall"));
    // Smooth rice <-> jumptrill transitions alone are NOT technical: they need repeated-column or rhythm-control evidence.
    let row = |pattern: &[&[usize]]| -> Vec<Note> { pattern.iter().enumerate().flat_map(|(i, cs)| cs.iter().map(move |&c| (1000 + 70 * i as i64, c, None)).collect::<Vec<_>>()).collect() };
    let single: [&[usize]; 4] = [&[0], &[2], &[1], &[3]];
    let chords: [&[usize]; 4] = [&[0, 1], &[2, 3], &[0, 1], &[2, 3]];
    let switching: Vec<&[usize]> = (0..160).flat_map(|i| if i % 2 == 1 { single } else { chords }).collect();
    let sw = sc(&row(&switching), 4);
    assert!(sw.f("patterntech") < 0.1 * sw.f("overall"));
    let jt: Vec<&[usize]> = std::iter::repeat([&[0, 1][..], &[2, 3][..]]).take(320).flatten().collect();
    assert_eq!(sc(&row(&jt), 4).f("patterntech"), 0.);
}

#[test]
fn hand_descriptors() {
    // 1H Trill: one hand carrying the rows alternating two fingers; two-hand alternation is not
    let one: Vec<Note> = (0..600).map(|i| (1000 + 90 * i, (i % 2) as usize, None)).collect();
    let two: Vec<Note> = (0..600).map(|i| (1000 + 90 * i, [0, 3, 1, 2][i as usize % 4], None)).collect();
    assert!(sc(&one, 4).f("trill1h") > 0.9 * sc(&one, 4).f("overall"));
    assert_eq!(sc(&two, 4).f("trill1h"), 0.);
    // Anchor: one column every other row around the other three; an even roll is not
    let anchor: Vec<Note> = (0..600usize).map(|i| (1000 + 80 * i as i64, if i % 2 == 0 { 1 } else { [0, 2, 3][(i / 2) % 3] }, None)).collect();
    let roll: Vec<Note> = (0..600).map(|i| (1000 + 80 * i, (i % 4) as usize, None)).collect();
    assert!(sc(&anchor, 4).f("anchor") > 0.9 * sc(&anchor, 4).f("overall"));
    assert_eq!(sc(&roll, 4).f("anchor"), 0.);
}

#[test]
fn tech_card() {
    // tech inside the dump passage: Tech Dump replaces Dump; Technical, Pattern/Rhythm Tech never bare; no part twice;
    // a partial Tech X never sits next to its own X
    let r = res(&dump(&[40, 40, 40, 60, 60], 900, &DUMP_COLS), 4);
    let entries = card(&r);
    let n: Vec<String> = entries.iter().map(name_of).collect();
    assert!(n[0] == "Tech Dump" && !n.iter().any(|x| x == "Dump"), "{n:?}");
    assert!(!n.iter().any(|x| ["Technical", "Pattern Tech", "Rhythm Tech"].contains(&x.as_str())));
    let mut parts: Vec<String> = entries.iter().flat_map(parts_of).collect();
    let all = parts.len();
    parts.sort();
    parts.dedup();
    assert_eq!(parts.len(), all);
    let Some(Val::List(places)) = r.get("tech_where") else { panic!() };
    let Val::Dict(first) = &places[0] else { panic!() };
    assert!(first.f("rhythm") > 0.9);
    // tech in one half, harder plain dump in the other: Dump keeps its entry, no Tech Dump beside it
    let mut half = dump(&[60], 400, &DUMP_COLS);
    half.extend(dump(&[40, 40, 40, 60, 60], 300, &DUMP_COLS).into_iter().map(|n| (n.0 + 30000, n.1, n.2)));
    let n = names(&half, 4);
    assert!(!(n.iter().any(|x| x == "Dump") && n.iter().any(|x| x == "Tech Dump")), "{n:?}");
}

#[test]
fn interleaved_bracket_is_harder_than_a_plain_bracket() {
    // user 2026-10-06: "they get rated the same as regular brackets but they constantly alternate fingers"
    let inter = rows(&[&[1, 5], &[0, 2, 4, 6]], 110, 800);
    let plain = rows(&[&[2, 4], &[0, 1, 5, 6]], 110, 800);
    assert!(sc(&inter, 7).f("overall") > 1.1 * sc(&plain, 7).f("overall"));
}

// ---- whole-chart Tech qualification ----

fn profile_of(c: &Chart, rate: f64) -> [f64; 3] {
    let prm = standard_params();
    let r = compute(c, rate, true, &prm).unwrap();
    let t = r.dict("tech_profile").unwrap();
    [t.f("technical"), t.f("patterntech"), t.f("rhythmtech")]
}

#[test]
fn constant_global_pulse_with_skipped_hand_strokes_is_not_rhythm_tech() {
    for rate in [0.83, 1., 1.24] {
        for offset in [1000., 61037.] {
            let c = rice((0..400).map(|i| (offset + 80. * i as f64, [0, 4, 5, 1, 6][i % 5])), 7, 8.);
            assert_eq!(profile_of(&c, rate)[2], 0.);
        }
    }
}

#[test]
fn irregular_five_gap_control_remains_technical() {
    let mut t = 1000.;
    let notes: Vec<(f64, u8)> = (0..400usize).map(|i| {
        t += [60., 93., 47., 111., 75.][i % 5];
        (t, (i % 4) as u8)
    }).collect();
    assert!(profile_of(&rice(notes, 4, 8.), 1.)[2] > 0.9);
}

#[test]
fn regular_inserted_jacks_retain_pattern_evidence() {
    let pattern: [[u8; 2]; 5] = [[0, 1], [4, 5], [4, 6], [1, 2], [5, 6]];
    let c = rice((0..400).flat_map(|i| pattern[i % 5].map(|c| (1000. + 40. * i as f64, c))), 7, 8.);
    assert_eq!(profile_of(&c, 1.), [1., 1., 0.]);
}

fn with_archetype(scores_of: &[(&str, f64)], name: &str, rating: f64, parts: &[&str], of: Option<&str>, technical: Option<f64>) -> Dict {
    let mut sc = Dict::new();
    for s in skills(7) {
        sc.set(s.name(), 0.);
    }
    sc.set("overall", 0.);
    for (k, v) in scores_of {
        sc.set(k, *v);
    }
    let mut a = Dict::new();
    a.set("name", name);
    a.set("rating", rating);
    a.set("parts", Val::strs(parts));
    if let Some(of) = of {
        a.set("of", of);
    }
    let mut res = Dict::new();
    res.set("keys", 7i64);
    res.set("scores", sc);
    res.set("archetypes", Val::List(vec![Val::Dict(a)]));
    if let Some(t) = technical {
        let mut p = Dict::new();
        p.set("technical", t);
        res.set("tech_profile", p);
    }
    res
}

#[test]
fn preexisting_tech_archetype_obeys_same_profile_qualification() {
    let of = |t| card(&with_archetype(&[("overall", 10.), ("delay", 9.), ("stamina", 8.8), ("technical", 8.7)], "Tech Delay", 9., &["delay", "technical"], Some("delay"), t));
    let legacy = of(None);
    assert_eq!(name_of(&legacy[0]), "Tech Delay");
    let qualified = of(Some(0.3));
    assert_eq!(name_of(&qualified[0]), "Delay");
    let rating = |e: &Val| if let Val::Dict(d) = e { d.f("rating") } else { panic!() };
    assert_eq!(qualified.iter().map(rating).collect::<Vec<_>>(), legacy.iter().map(rating).collect::<Vec<_>>());
    assert_eq!(parts_of(&qualified[0]), ["delay"]);
    assert_eq!(of(Some(0.8)), legacy);
}

#[test]
fn unqualified_technical_only_archetype_does_not_consume_a_card_slot() {
    let res = with_archetype(&[("overall", 10.), ("ln", 9.), ("delay", 8.), ("stamina", 7.8), ("technical", 9.5), ("patterntech", 9.)],
                             "Tech Control", 9.5, &["technical", "patterntech"], None, Some(0.3));
    let entries = card(&res);
    assert_eq!((name_of(&entries[0]), parts_of(&entries[0])), ("LN".to_string(), vec!["ln".to_string()]));
    assert!(entries.iter().all(|e| name_of(e) != "Control"));
}

#[test]
fn existing_tech_description_can_be_requalified_and_legacy_is_preserved() {
    use ms_calc::pattern::technical_title;
    assert_eq!(technical_title("Tech LN", 0.8, Some(&[0.3, 0., 0.])), "LN");
    assert_eq!(technical_title("Tech Delay", 0.8, None), "Tech Delay");
    assert_eq!(technical_title("Delay", 0.8, Some(&[0.8, 0., 0.])), "Tech Delay");
}

#[test]
fn delay_keeps_its_name_without_ln_tech_prevalence() {
    let res = with_archetype(&[("overall", 10.), ("delay", 8.54), ("ln", 8.27), ("release", 7.18), ("technical", 5.17), ("bracket", 7.49)], "x", 0., &[], None, None);
    let mut res = res;
    res.set("archetypes", Val::List(Vec::new()));
    res.set("notes", 100i64);
    res.set("ln_notes", 83i64);
    let mut p = Dict::new();
    p.set("technical", 0.43);
    res.set("tech_profile", p);
    assert_eq!(name_of(&card(&res)[0]), "Delay");
}
