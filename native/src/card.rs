//! Archetypes, Technical passages, card entries and descriptions (skill_calc + archetype_names).
use crate::chart::Chart;
use crate::demand::{Parts, P};
use crate::params::Knobs;
use crate::pattern::technical_title;
use crate::py;
use crate::skills::{combine, horizon_levels, skills, stamina, Series, Sk, LN_KINDS, SV_KINDS, TECH_KEYS};
use crate::val::{Dict, Key, Val};

const TECH_NAMES: [(Sk, &str); 19] = [
    (Sk::stream, "Tech Stream"), (Sk::delay, "Tech Delay"), (Sk::chordstream, "Tech Chordstream"),
    (Sk::dump, "Tech Dump"), (Sk::jumptrill, "Tech Jumptrill"), (Sk::splittrill, "Tech Split Trill"),
    (Sk::jumpstream, "Tech Jumpstream"), (Sk::handstream, "Tech Handstream"), (Sk::jack, "Tech Jacks"),
    (Sk::jackspeed, "Tech Jacks"), (Sk::chordjack, "Tech Chordjack"), (Sk::minijack, "Tech Minijack"),
    (Sk::longjack, "Tech Longjack"), (Sk::bracket, "Tech Bracket"), (Sk::hybrid, "Tech Hybrid LN"),
    (Sk::inverse, "Tech Inverse"), (Sk::release, "Tech LN Release"), (Sk::ln, "Tech LN"),
    (Sk::trill1h, "Tech 1H Trill"),
];
const TECH_ANCHOR: (Sk, &str) = (Sk::anchor, "Tech Anchor");
const STAMINA_NAMES: [(Sk, &str); 17] = [
    (Sk::stream, "Stream Stamina"), (Sk::delay, "Delay Stamina"), (Sk::chordstream, "Chordstream Stam"),
    (Sk::dump, "Dump Stamina"), (Sk::jumptrill, "JT Stamina"), (Sk::splittrill, "Split Stamina"),
    (Sk::jumpstream, "JS Stamina"), (Sk::handstream, "HS Stamina"), (Sk::jack, "Jack Stamina"),
    (Sk::jackspeed, "Jack Stamina"), (Sk::longjack, "Jack Stamina"), (Sk::chordjack, "CJ Stamina"),
    (Sk::bracket, "Bracket Stamina"), (Sk::release, "LN Stamina"), (Sk::inverse, "LN Stamina"),
    (Sk::hybrid, "LN Stamina"), (Sk::ln, "LN Stamina"),
];
const TECH_CONTROL: &str = "Tech Control";
const REFINES: [(Sk, &[Sk]); 8] = [
    (Sk::bracket, &[Sk::chordstream]),
    (Sk::quadstream, &[Sk::handstream, Sk::minijack]),
    (Sk::longjack, &[Sk::jack, Sk::jackspeed, Sk::chordjack]),
    (Sk::vibro, &[Sk::jack, Sk::jackspeed, Sk::chordjack, Sk::longjack]),
    (Sk::release, &[Sk::ln]),
    (Sk::inverse, &[Sk::ln]),
    (Sk::hybrid, &[Sk::ln]),
    (Sk::shield, &[Sk::ln]),
];
fn describes(s: &str) -> &'static [&'static str] {
    match s {
        "dump" => &["stream"],
        "trill1h" | "anchor" => &["stream", "delay"],
        _ => &[],
    }
}

pub fn display_name(s: Sk) -> &'static str {
    match s {
        Sk::stream => "Stream", Sk::jumpstream => "Jumpstream", Sk::handstream => "Handstream",
        Sk::quadstream => "Quadstream", Sk::sv_fast => "Fast SV", Sk::sv_slow => "Slowjam SV",
        Sk::sv_accel => "Accel SV", Sk::sv_stutter => "Stutter SV", Sk::sv_brake => "Brakes",
        Sk::jackspeed => "Jackspeed", Sk::chordjack => "Chordjack", Sk::delay => "Delay",
        Sk::chordstream => "Chordstream", Sk::bracket => "Bracket", Sk::jack => "Jack", Sk::dump => "Dump",
        Sk::jumptrill => "Jumptrill", Sk::splittrill => "Split Trill", Sk::technical => "Technical",
        Sk::patterntech => "Pattern Tech", Sk::rhythmtech => "Rhythm Tech", Sk::trill1h => "1H Trill",
        Sk::anchor => "Anchor", Sk::ln => "LN", Sk::sv => "SV", Sk::stamina => "Stamina", Sk::minijack => "Minijack",
        Sk::longjack => "Longjack", Sk::release => "LN Release", Sk::inverse => "Inverse", Sk::hybrid => "Hybrid LN",
        Sk::shield => "Shield", Sk::vibro => "Vibro", Sk::mash => "Mash", Sk::overall | Sk::switch => "",
    }
}

fn arch(name: &str, rating: f64, parts: &[&str], of: Option<&str>) -> Val {
    let mut d = Dict::new();
    d.set("name", name);
    d.set("rating", rating);
    d.set("parts", Val::strs(parts));
    if let Some(o) = of {
        d.set("of", o);
    }
    Val::Dict(d)
}

/// Scores dict lookup by skill.
fn s(sc: &Dict, k: Sk) -> f64 {
    sc.f(k.name())
}

fn min_series(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x.min(*y)).collect()
}

/// skill_calc._archetypes.
pub fn archetypes(series: &Series, sc: &Dict, notes: usize, switch: &[f64], k: &Knobs) -> Vec<Val> {
    let mut out: Vec<Val> = Vec::new();
    let mut top = f64::NEG_INFINITY;
    for (key, v) in &sc.0 {
        if let Key::S(key) = key {
            if let Some(sk) = Sk::from_name(key) {
                if series.has(sk) && !TECH_KEYS.contains(&sk) && sk != Sk::ln && v.f() > top {
                    top = v.f();
                }
            }
        }
    }
    if series.has(Sk::quadstream) {
        let (q, mj) = (s(sc, Sk::quadstream), s(sc, Sk::minijack));
        if q >= 0.8 * mj && 0.8 * mj > 0.0 && q >= k.ARCH_MATERIAL * s(sc, Sk::overall) {
            out.push(arch("Quadstream", mj, &["quadstream", "minijack"], None));
        }
    }
    let tech = series.get(Sk::technical);
    let mut best = 0.0f64;
    for (key, name) in TECH_NAMES.iter().chain(std::iter::once(&TECH_ANCHOR)) {
        if !series.has(*key) || s(sc, *key) < k.ARCH_MATERIAL * top {
            continue;
        }
        let r = combine(&horizon_levels(&min_series(series.get(*key), tech), k), notes, k);
        let sk = s(sc, *key);
        if r >= k.ARCH_TECH * sk && r >= 0.80 * s(sc, Sk::overall) {
            let whole = r >= k.TECH_WHOLE * sk;
            let rating = if whole { sk } else { r };
            if whole {
                out.push(arch(name, rating, &[key.name(), "technical"], Some(key.name())));
            } else {
                out.push(arch(name, rating, &["technical"], Some(key.name())));
            }
            best = best.max(rating);
        }
    }
    if !switch.is_empty() {
        let r = combine(&horizon_levels(&min_series(switch, tech), k), notes, k);
        if r >= k.ARCH_MATERIAL * top && best < k.REFINE_Q * r {
            out.push(arch(TECH_CONTROL, r, &["technical", "patterntech"], None));
        }
    }
    for (key, parents) in REFINES {
        if !series.has(key) {
            continue;
        }
        let sk = s(sc, key);
        let ps: Vec<Sk> = parents.iter().copied().filter(|&p| series.has(p) && sk >= k.REFINE_Q * s(sc, p) && k.REFINE_Q * s(sc, p) > 0.0).collect();
        if !ps.is_empty() {
            let mut bp = ps[0];
            for &p in &ps[1..] {
                if s(sc, p) > s(sc, bp) {
                    bp = p;
                }
            }
            out.push(arch(display_name(key), sk, &[key.name(), bp.name()], None));
        }
    }
    let teched: Vec<String> = out
        .iter()
        .filter_map(|a| {
            let Val::Dict(d) = a else { return None };
            let Some(Val::Str(of)) = d.get("of") else { return None };
            let Some(Val::Tuple(parts)) = d.get("parts") else { return None };
            parts.iter().any(|p| matches!(p, Val::Str(x) if x == of)).then(|| of.clone())
        })
        .collect();
    for (key, name) in STAMINA_NAMES {
        if !series.has(key) || s(sc, key) < k.ARCH_MATERIAL * top || teched.iter().any(|t| t == key.name()) {
            continue;
        }
        let lv = horizon_levels(series.get(key), k);
        let mx = lv.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sk = s(sc, key);
        if mx > 0.0 && stamina(series.get(key), k) * sk / mx >= k.STAMINA_SUSTAIN * sk {
            out.push(arch(name, sk, &[key.name(), "stamina"], None));
        }
    }
    out
}

const TECH_WHY: [(P, &str); 3] = [
    (P::rodd, "rhythm: gaps off the pulse (e.g. 1/4 <-> 1/6)"),
    (P::pins, "minijacks inserted in rice"),
    (P::psw, "pattern transitions with repeated-column or irregular-hand pressure"),
];

fn window_sum(v: &[f64], i: usize, w: usize) -> f64 {
    let end = (i + w).min(v.len());
    if i >= end {
        return 0.0;
    }
    py::sum(v[i..end].iter().copied())
}

/// skill_calc._tech_where.
pub fn tech_where(series: &Series, parts: &Parts, start: f64, rate: f64, k: &Knobs) -> Vec<Val> {
    let n = 3;
    let tech = series.get(Sk::technical);
    let w = 4usize;
    let ns = 1i64.max(tech.len() as i64 - w as i64 + 1) as usize;
    let sums: Vec<f64> = (0..ns).map(|i| window_sum(tech, i, w)).collect();
    let mut order: Vec<usize> = (0..sums.len()).collect();
    order.sort_by(|&a, &b| (-sums[a]).partial_cmp(&-sums[b]).unwrap());
    let mut out = Vec::new();
    let mut taken: Vec<usize> = Vec::new();
    for i in order {
        let refv = sums[if out.is_empty() { 0 } else { taken[0] }];
        if out.len() == n || sums[i] == 0.0 || sums[i] < 0.5 * refv {
            break;
        }
        if taken.iter().any(|&j| (i as i64 - j as i64).abs() < w as i64) {
            continue;
        }
        taken.push(i);
        let pt = window_sum(series.get(Sk::patterntech), i, w);
        let rt = window_sum(series.get(Sk::rhythmtech), i, w);
        let ev: Vec<f64> = TECH_WHY.iter().map(|(p, _)| window_sum(parts.get(*p), i, w) / if *p == P::rodd { k.RT_FULL } else { k.PT_FULL }).collect();
        let denom = if pt + rt != 0.0 { pt + rt } else { 1.0 };
        let why = if ev.iter().any(|&x| x != 0.0) {
            let mut bi = 0;
            for j in 1..3 {
                if ev[j] > ev[bi] {
                    bi = j;
                }
            }
            TECH_WHY[bi].1
        } else {
            ""
        };
        let mut d = Dict::new();
        d.set("t", start + i as f64 * k.BIN * rate);
        d.set("pattern", pt / denom);
        d.set("rhythm", rt / denom);
        d.set("why", why);
        out.push(Val::Dict(d));
    }
    out
}

fn parts_of(e: &Dict) -> Vec<String> {
    match e.get("parts") {
        Some(Val::Tuple(v)) | Some(Val::List(v)) => v.iter().map(|x| if let Val::Str(s) = x { s.clone() } else { String::new() }).collect(),
        _ => Vec::new(),
    }
}
fn str_of(e: &Dict, k: &str) -> Option<String> {
    match e.get(k) {
        Some(Val::Str(s)) => Some(s.clone()),
        _ => None,
    }
}

fn names_for(res: &Dict) -> impl Fn(&str) -> String + '_ {
    move |k: &str| {
        if k == "sv" {
            str_of(res, "sv_name").unwrap_or_else(|| "SV".into())
        } else {
            Sk::from_name(k).map(|s| display_name(s).to_string()).unwrap_or_default()
        }
    }
}

/// The keymode's skills, strongest first (skill_calc.ranked).
pub fn ranked(res: &Dict) -> Vec<Sk> {
    let sc = res.dict("scores").unwrap();
    let mut v = skills(res.f("keys") as usize);
    v.sort_by(|a, b| (-py::round_digits(s(sc, *a), 2)).partial_cmp(&-py::round_digits(s(sc, *b), 2)).unwrap());
    v
}

fn tech_profile_of(res: &Dict) -> Option<[f64; 3]> {
    res.dict("tech_profile").map(|p| [p.f("technical"), p.f("patterntech"), p.f("rhythmtech")])
}

/// skill_calc.card.
pub fn card(res: &Dict) -> Vec<Val> {
    if let Some(Val::List(uc)) = res.get("unit_card") {
        return uc.clone();
    }
    let sc = res.dict("scores").unwrap();
    if sc.f("overall") < 0.1 {
        return Vec::new();
    }
    let technical_sk = py::round_digits(sc.f("technical") / 0.05f64.max(sc.f("overall")), 4);
    let profile = tech_profile_of(res);
    let names = names_for(res);
    let scf = |p: &str| sc.f(p);
    // (score, order, entry)
    let mut cands: Vec<(f64, u8, Dict)> = Vec::new();
    if let Some(Val::List(archs)) = res.get("archetypes") {
        for a in archs {
            let Val::Dict(a) = a else { continue };
            let name = str_of(a, "name").unwrap();
            if name.starts_with("Tech ") && technical_title(&name, technical_sk, profile.as_ref()) != name {
                continue;
            }
            let parts = parts_of(a);
            let mut mx = a.f("rating");
            for p in &parts {
                if !TECH_KEYS.iter().any(|t| t.name() == p) {
                    let v = scf(p);
                    if v > mx {
                        mx = v;
                    }
                }
            }
            let of = str_of(a, "of");
            let first = of.as_deref().is_some_and(|o| matches!(o, "dump" | "trill1h" | "anchor"))
                || parts.iter().any(|p| p == "ln") && parts.iter().any(|p| LN_KINDS.iter().any(|l| l.name() == p));
            cands.push((mx, if first { 0 } else { 1 }, a.clone()));
        }
    }
    for kk in ranked(res) {
        if TECH_KEYS.contains(&kk) || SV_KINDS.contains(&kk) || s(sc, kk) <= 0.0 {
            continue;
        }
        let mut d = Dict::new();
        d.set("name", names(kk.name()));
        d.set("rating", s(sc, kk));
        d.set("parts", Val::strs(&[kk.name()]));
        cands.push((s(sc, kk), 2, d));
    }
    cands.sort_by(|a, b| (-py::round_digits(a.0, 6), a.1).partial_cmp(&(-py::round_digits(b.0, 6), b.1)).unwrap());
    let mut picked: Vec<Dict> = Vec::new();
    let mut used: Vec<String> = Vec::new();
    for (_, _, e) in cands {
        if picked.len() == 3 || (!picked.is_empty() && e.f("rating") < 0.3 * picked[0].f("rating")) {
            break;
        }
        let parts = parts_of(&e);
        let of = str_of(&e, "of");
        if parts.iter().any(|p| used.contains(p)) || of.as_ref().is_some_and(|o| used.contains(o)) {
            continue;
        }
        let mut d = e.clone();
        d.pop("of");
        picked.push(d);
        for p in &parts {
            if !used.contains(p) {
                used.push(p.clone());
            }
        }
        for p in parts.iter().map(|x| x.as_str()).chain(std::iter::once(of.as_deref().unwrap_or(""))) {
            for m in describes(p) {
                if !used.iter().any(|u| u == m) {
                    used.push(m.to_string());
                }
            }
        }
    }
    let lnv = sc.f("ln");
    let key = |e: &Dict| {
        let parts = parts_of(e);
        let lnk = parts.iter().any(|p| p == "ln") && parts.iter().any(|p| LN_KINDS.iter().any(|l| l.name() == p));
        -e.f("rating").max(if lnk { lnv } else { 0.0 })
    };
    picked.sort_by(|a, b| key(a).partial_cmp(&key(b)).unwrap());
    if let Some(first) = picked.first_mut() {
        let name = str_of(first, "name").unwrap();
        let t = technical_title(&name, technical_sk, profile.as_ref());
        if t != name {
            first.set("name", t);
            let mut parts = parts_of(first);
            if !parts.iter().any(|p| p == "technical") {
                parts.push("technical".into());
            }
            first.set("parts", Val::Tuple(parts.into_iter().map(Val::Str).collect()));
        }
    }
    let ln_cs = res.dict("archetype_profile").map(|p| p.f("ln_chordstream"));
    let sk_wo: Vec<(String, f64)> = sc.0.iter().filter_map(|(k, v)| match k {
        Key::S(k) if k != "overall" => Some((k.clone(), v.f())),
        _ => None,
    }).collect();
    names_card(picked, &sk_wo, ln_cs, res.f("keys") as usize).into_iter().map(Val::Dict).collect()
}

// ---- archetype_names ----

pub fn profile(chart: &Chart, rate: f64) -> f64 {
    if chart.len() == 0 {
        return 0.0;
    }
    let mut rows: Vec<(f64, u32, Vec<(u8, f64)>)> = Vec::new();
    for i in 0..chart.len() {
        let (t, e, c) = (chart.t[i], chart.end[i], chart.col[i]);
        match rows.last_mut() {
            Some(r) if t - r.0 <= 3.0 => {
                r.1 |= 1 << c;
                r.2.push((c, e));
            }
            _ => rows.push((t, 1 << c, vec![(c, e)])),
        }
    }
    let mut bins: Vec<(i64, [f64; 2])> = Vec::new();
    let mut held: Vec<(u8, f64)> = Vec::new();
    let origin = rows[0].0;
    for (t, cols, ends) in &rows {
        let bucket = ((t - origin) / 1000.0 / rate) as i64;
        let n = cols.count_ones() as f64;
        let taps = ends.iter().filter(|(_, e)| e - t <= 3.0).count();
        let locked = held.iter().any(|&(c, e)| e > t + 3.0 && cols >> c & 1 == 0);
        let bi = match bins.iter().position(|b| b.0 == bucket) {
            Some(i) => i,
            None => {
                bins.push((bucket, [0.0, 0.0]));
                bins.len() - 1
            }
        };
        bins[bi].1[0] += n;
        if locked && n >= 2.0 && taps > 0 {
            bins[bi].1[1] += n;
        }
        for &(c, e) in ends {
            if e > t + 3.0 {
                match held.iter_mut().find(|x| x.0 == c) {
                    Some(x) => x.1 = e,
                    None => held.push((c, e)),
                }
            }
        }
    }
    let get = |j: i64| bins.iter().find(|b| b.0 == j).map_or([0.0, 0.0], |b| b.1);
    let den = py::sum(bins.iter().map(|b| b.1[0]));
    let mut total = 0.0f64;
    for &(index, b) in &bins {
        let nb = [get(index - 1), get(index), get(index + 1)];
        let mass = py::sum(nb.iter().map(|v| v[0]));
        total += b[0] * py::sum(nb.iter().map(|v| v[1])) / 1f64.max(mass);
    }
    py::round_digits(total / 1f64.max(den), 6)
}

fn descriptor(sk: &[(String, f64)], ln_cs: Option<f64>, keys: usize) -> Option<(&'static str, [&'static str; 3])> {
    let ln_cs = ln_cs?;
    let g = |k: &str| sk.iter().find(|(kk, _)| kk == k).map_or(0.0, |x| x.1);
    let flow: [&str; 4] = if keys == 4 { ["stream", "dump", "jumpstream", "handstream"] } else { ["stream", "delay", "chordstream", "bracket"] };
    let mut primary = flow[0];
    for f in &flow[1..] {
        if g(f) > g(primary) {
            primary = f;
        }
    }
    let top = sk.iter().map(|x| x.1).fold(None, |m: Option<f64>, x| Some(match m { Some(m) if m >= x => m, _ => x })).unwrap_or(0.0);
    if top <= 0.0 || g(primary) < 0.6 * top {
        return None;
    }
    if ln_cs >= 0.25 && g("hybrid") >= 0.6 * top && g("ln") >= 0.65 * top && g("chordstream").max(g("handstream")) >= 0.65 * top {
        return Some(("LN Chordstream", ["ln", "hybrid", if keys == 4 { "handstream" } else { "chordstream" }]));
    }
    None
}

fn an_name(k: &str) -> &str {
    match k {
        "stream" => "Stream", "dump" => "Dump", "jumpstream" => "Jumpstream", "handstream" => "Handstream",
        "delay" => "Delay", "chordstream" => "Chordstream", "bracket" => "Bracket", "minijack" => "Minijack",
        "anchor" => "Anchor", "ln" => "LN", "hybrid" => "Hybrid LN", "release" => "LN Release", _ => k,
    }
}

fn rename(name: &str, desc: Option<(&str, [&str; 3])>) -> String {
    let Some((dname, parts)) = desc else { return name.to_string() };
    let prefix = if name.starts_with("Tech ") { "Tech " } else { "" };
    let text = &name[prefix.len()..];
    let suffix = if text.ends_with(" Stamina") { " Stamina" } else { "" };
    let plain = &text[..text.len() - suffix.len()];
    if !parts.iter().any(|p| an_name(p) == plain) {
        return name.to_string();
    }
    format!("{prefix}{dname}{suffix}")
}

fn names_card(entries: Vec<Dict>, sk: &[(String, f64)], ln_cs: Option<f64>, keys: usize) -> Vec<Dict> {
    let desc = descriptor(sk, ln_cs, keys);
    if entries.is_empty() || desc.is_none() {
        return entries;
    }
    let first_name = str_of(&entries[0], "name").unwrap();
    let name = rename(&first_name, desc);
    if name == first_name {
        return entries;
    }
    let (_, dparts) = desc.unwrap();
    let mut first = entries[0].clone();
    first.set("name", name);
    let mut parts = parts_of(&entries[0]);
    for p in dparts {
        if !parts.iter().any(|x| x == p) {
            parts.push(p.to_string());
        }
    }
    first.set("parts", Val::Tuple(parts.into_iter().map(Val::Str).collect()));
    let mut out = vec![first];
    for e in &entries[1..] {
        if !parts_of(e).iter().any(|p| dparts.contains(&p.as_str())) {
            out.push(e.clone());
        }
    }
    out
}

/// skill_calc.describe → (dominant text, tags).
pub fn describe(res: &Dict, k: &Knobs) -> (String, Vec<String>) {
    let sc = res.dict("scores").unwrap();
    let entries = card(res);
    if entries.is_empty() {
        return (String::new(), Vec::new());
    }
    let Val::Dict(a) = &entries[0] else { unreachable!() };
    let mut dom = str_of(a, "name").unwrap();
    if entries.len() > 1 {
        if let Val::Dict(b) = &entries[1] {
            if b.f("rating") >= 0.85 * a.f("rating") {
                dom = format!("{dom} / {}", str_of(b, "name").unwrap());
            }
        }
    }
    if let Some(pd) = str_of(res, "preunit_description") {
        dom = pd;
    }
    let mut tags = Vec::new();
    let lv = res.dict("levels").unwrap();
    let l30 = lv.0.iter().find(|(kk, _)| matches!(kk, Key::F(x) if *x == 30.0)).map(|(_, v)| v.f()).unwrap_or(0.0);
    if res.get("horizon").is_some_and(|h| matches!(h, Val::F(x) if *x == 2.0)) && l30 < 0.75 * sc.f("overall") {
        tags.push("short burst".to_string());
    }
    let shown: Vec<String> = entries.iter().flat_map(|e| if let Val::Dict(d) = e { parts_of(d) } else { Vec::new() }).collect();
    let names = names_for(res);
    let mut extra = Vec::new();
    let groups: [(&[&str], f64); 4] = [
        (&["sv"], k.ALSO_SPECIAL),
        (&["release", "inverse", "hybrid", "shield", "ln"], k.ALSO_SHARE),
        (&["vibro"], k.ALSO_SPECIAL),
        (&["mash"], 0.25),
    ];
    for (group, share) in groups {
        if shown.iter().any(|p| group.contains(&p.as_str())) {
            continue;
        }
        let mut best: Option<&str> = None;
        for g in group {
            if sc.has(g) && best.is_none_or(|b| sc.f(g) > sc.f(b)) {
                best = Some(g);
            }
        }
        let Some(kk) = best else { continue };
        if kk == "mash"
            && res.dict("execution").map_or(0.0, |e| e.fget("mash", 0.0)) < 0.05
            && sc.f(kk) < 0.8 * sc.f("overall")
        {
            continue;
        }
        if sc.f(kk) >= share * a.f("rating") {
            extra.push(format!("{} {:.1}", names(kk), sc.f(kk)));
        }
    }
    if !extra.is_empty() {
        tags.push(format!("also {}", extra.join(" · ")));
    }
    (dom, tags)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sk(v: &[(&str, f64)]) -> Vec<(String, f64)> {
        v.iter().map(|&(k, x)| (k.to_string(), x)).collect()
    }

    #[test]
    fn old_jackstream_evidence_does_not_rename_patterns() {
        let mut a = Dict::new();
        a.set("name", "Tech Stream");
        a.set("parts", Val::strs(&["stream", "technical"]));
        let entries = vec![a];
        let sk = sk(&[("stream", 1.0), ("minijack", 0.8), ("technical", 0.8)]);
        assert_eq!(names_card(entries.clone(), &sk, None, 4), entries);
    }

    #[test]
    fn ln_chordstream_names_the_texture_with_valid_parts() {
        let s = sk(&[("chordstream", 1.0), ("ln", 0.9), ("hybrid", 0.85)]);
        assert_eq!(rename("Chordstream", descriptor(&s, Some(0.95), 7)), "LN Chordstream");
        for (keys, texture) in [(4, "handstream"), (7, "chordstream")] {
            let s = sk(&[(texture, 1.0), ("ln", 0.9), ("hybrid", 0.85)]);
            let (_, parts) = descriptor(&s, Some(0.8), keys).unwrap();
            let names: Vec<&str> = skills(keys).iter().map(|k| k.name()).collect();
            assert!(parts.iter().all(|p| names.contains(p)), "{keys}K {parts:?}");
        }
    }

    #[test]
    fn card_names_fit_the_card() {
        let names = TECH_NAMES.iter().chain(STAMINA_NAMES.iter()).chain([TECH_ANCHOR].iter()).map(|&(_, n)| n);
        for n in names.chain([TECH_CONTROL]) {
            assert!(n.len() <= 16 && n.split_whitespace().count() <= 3, "{n}");
        }
    }
}
