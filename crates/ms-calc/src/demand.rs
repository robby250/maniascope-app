//! Per-action effort → per-bin demand (skill_calc._demand_geometry/_demand and their row evidence).
use crate::chart::Chart;
use crate::kern::{inner, off_grid, smooth, v};
use crate::params::Knobs;
use crate::py;

/// Demand parts: one row of the [P::N × nbins] matrix each (order = skill_calc's all_keys).
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum P {
    stream, jumpstream, handstream, jumptrill, splittrill, jackspeed, chordjack, ln, sv, delay,
    chordstream, jack, technical, bracket, dump, odd, rodd, pins, psw, oht, anc, rows, sv_raw,
    ln_rel, mj, sus, lj, shield, inv, hyb, vib, sv_fast, sv_slow, sv_accel, sv_stutter, sv_brake,
}
pub const NP: usize = 36;
pub const P_NAMES: [&str; NP] = [
    "stream", "jumpstream", "handstream", "jumptrill", "splittrill", "jackspeed", "chordjack", "ln", "sv", "delay",
    "chordstream", "jack", "technical", "bracket", "dump", "odd", "rodd", "pins", "psw", "oht", "anc", "rows",
    "sv_raw", "ln_rel", "mj", "sus", "lj", "shield", "inv", "hyb", "vib", "sv_fast", "sv_slow", "sv_accel",
    "sv_stutter", "sv_brake",
];
pub const PARTS_4K: [P; 9] = [P::stream, P::jumpstream, P::handstream, P::jumptrill, P::splittrill, P::jackspeed, P::chordjack, P::ln, P::sv];
pub const PARTS_5K: [P; 8] = [P::delay, P::stream, P::chordstream, P::jumptrill, P::jack, P::chordjack, P::ln, P::sv];
pub const SV_KINDS: [P; 5] = [P::sv_fast, P::sv_slow, P::sv_accel, P::sv_stutter, P::sv_brake];
const TAIL: [P; 26] = [
    P::technical, P::bracket, P::dump, P::odd, P::rodd, P::pins, P::psw, P::oht, P::anc, P::rows, P::sv_raw, P::ln_rel, P::mj, P::sus, P::lj, P::shield, P::inv, P::hyb,
    P::vib, P::sv_fast, P::sv_slow, P::sv_accel, P::sv_stutter, P::sv_brake,
    // padding never read (TAIL_LEN)
    P::stream, P::stream,
];
const TAIL_LEN: usize = 24;

/// Output order of the parts dict for a keymode.
pub fn part_keys(wide: bool) -> Vec<P> {
    let mut out: Vec<P> = if wide { PARTS_5K.to_vec() } else { PARTS_4K.to_vec() };
    out.extend_from_slice(&TAIL[..TAIL_LEN]);
    out
}

#[inline]
fn non_demand(p: P) -> bool {
    matches!(p, P::odd | P::rodd | P::pins | P::psw | P::oht | P::anc | P::rows | P::sv_raw | P::mj | P::sus | P::lj | P::shield | P::inv | P::hyb | P::vib)
}

/// Part series, keyed like the Python dict (`keys` = its order).
#[derive(Clone, Debug)]
pub struct Parts {
    pub nbins: usize,
    pub data: Vec<f64>,
    pub keys: Vec<P>,
}
impl Parts {
    #[inline]
    pub fn get(&self, p: P) -> &[f64] {
        let o = p as usize * self.nbins;
        &self.data[o..o + self.nbins]
    }
    #[inline]
    pub fn get_mut(&mut self, p: P) -> &mut [f64] {
        let o = p as usize * self.nbins;
        &mut self.data[o..o + self.nbins]
    }
    pub fn has(&self, p: P) -> bool {
        self.keys.contains(&p)
    }
}

/// One `_demand` run: (parts, totals, fine per side).
#[derive(Clone, Debug)]
pub struct Run {
    pub parts: Parts,
    pub totals: Vec<f64>,
    pub fine: [Vec<f64>; 3],
}

/// Per-note motor action (rolled_execution's plans).
#[derive(Clone, Copy, Debug, Default)]
pub struct Action {
    pub hand: u8,
    pub rice: f64,
    pub cross: f64,
    pub protected: bool,
    pub fb: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct Row {
    pub t: f64,
    pub mask: u32,
    pub start: usize,
    pub end: usize,
}
impl Row {
    #[inline]
    pub fn size(&self) -> u32 {
        self.mask.count_ones()
    }
}

/// Hand-independent rows and repeat evidence for one chart/rate calculation.
pub struct Geometry {
    pub times: Vec<f64>,
    pub ends: Vec<f64>,
    pub columns: Vec<u8>,
    pub vib: Vec<bool>,
    pub rows: Vec<Row>,
    pub roll: Vec<bool>,
}

pub fn geometry(chart: &Chart, rate: f64, k: &Knobs) -> Geometry {
    let inv = 0.001 / rate;
    let n = chart.len();
    let times: Vec<f64> = chart.t.iter().map(|&t| t * inv).collect();
    let ends: Vec<f64> = chart.end.iter().map(|&e| e * inv).collect();
    let columns = chart.col.clone();
    let mut vib = vec![false; n];
    let mut bycol: Vec<Vec<usize>> = vec![Vec::new(); chart.keys];
    for (i, &c) in columns.iter().enumerate() {
        bycol[c as usize].push(i);
    }
    for ks in &bycol {
        if ks.is_empty() {
            continue;
        }
        let mut run = vec![ks[0]];
        for (x, &a) in ks.iter().enumerate() {
            let next = ks.get(x + 1).copied();
            if let Some(kk) = next {
                if times[kk] - times[a] < k.VIBRO_GAP && ends[a] - times[a] <= 0.003 {
                    run.push(kk);
                    continue;
                }
            }
            if run.len() >= k.VIBRO_MIN {
                for &i in &run {
                    vib[i] = true;
                }
            }
            run = next.into_iter().collect();
        }
    }
    let mut rows = Vec::new();
    let mut i = 0;
    while i < n {
        let mut m = i + 1;
        while m < n && times[m] - times[i] <= 0.003 {
            m += 1;
        }
        let mask = columns[i..m].iter().fold(0u32, |a, &c| a | 1 << c);
        rows.push(Row { t: times[i], mask, start: i, end: m });
        i = m;
    }
    let nr = rows.len();
    let mut roll = vec![false; nr];
    let mut run: Vec<usize> = Vec::new();
    let mut run_step: Option<i64> = None;
    let keys = chart.keys as i64;
    for r in 0..=nr {
        let cs = if r < nr { rows[r].mask } else { 0 };
        let single = cs.count_ones() == 1;
        let step = if single && !run.is_empty() && rows[r - 1].mask.count_ones() == 1 {
            Some((cs.trailing_zeros() as i64 - rows[r - 1].mask.trailing_zeros() as i64).rem_euclid(keys))
        } else {
            None
        };
        if let Some(s) = step {
            if (s == 1 || s == keys - 1) && (run.len() < 2 || Some(s) == run_step) {
                run.push(r);
                run_step = step;
                continue;
            }
        }
        if run.len() >= k.ROLL_MIN {
            for &x in &run {
                roll[x] = true;
            }
        }
        run = if single { vec![r] } else { Vec::new() };
        run_step = None;
    }
    Geometry { times, ends, columns, vib, rows, roll }
}

/// Row families for Pattern Tech switches.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Trill {
    None,
    Jumptrill,
    Splittrill,
}

/// Pattern Tech evidence per row (inserted repeats, chained switches).
pub fn pattern_evidence(rows: &[Row], trill: &[Trill], k: &Knobs) -> (Vec<f64>, Vec<f64>) {
    let n = rows.len();
    let rep: Vec<bool> =
        (0..n).map(|r| r > 0 && rows[r].mask & rows[r - 1].mask != 0 && rows[r].t - rows[r - 1].t < 0.25).collect();
    let mut ins = vec![0.0; n];
    let mut sws = vec![0.0; n];
    let mut last: Option<f64> = None;
    for r in 0..n {
        let t = rows[r].t;
        let g = if r > 0 { t - rows[r - 1].t } else { 1.0 };
        if rep[r] && r >= 3 && !(rep[r - 1] || rep[r - 2] || r + 1 < n && rep[r + 1]) {
            ins[r] = smooth(1.0 / g, 6.0, 12.0) * smooth(1.0 / (rows[r - 1].t - rows[r - 2].t), 6.0, 12.0);
        }
        if r >= 3
            && r + 2 < n
            && trill[r] != trill[r - 1]
            && trill[r - 1] == trill[r - 2]
            && trill[r - 2] == trill[r - 3]
            && trill[r] == trill[r + 1]
            && trill[r + 1] == trill[r + 2]
        {
            let sw = smooth(1.0 / g, 5.0, 10.0);
            if let Some(l) = last {
                if t - l < k.PT_CHAIN {
                    sws[r] = sw;
                }
            }
            if sw > 0.3 {
                last = Some(t);
            }
        }
    }
    (ins, sws)
}

/// Row membership of 1H Trill and Anchor.
fn hand_patterns(rows: &[Row], hand_mask: [u32; 2], keys: usize) -> (Vec<bool>, Vec<bool>) {
    let n = rows.len();
    let mut oht = vec![false; n];
    let mut anc = vec![false; n];
    for hm in hand_mask {
        for r in 3..n {
            let (a, b) = (rows[r - 3].mask & hm, rows[r - 2].mask & hm);
            if a != 0
                && b != 0
                && a & (a - 1) == 0
                && b & (b - 1) == 0
                && a == rows[r - 1].mask & hm
                && b == rows[r].mask & hm
                && a != b
                && ((rows[r - 3].mask != a) as u32
                    + (rows[r - 2].mask != b) as u32
                    + (rows[r - 1].mask != a) as u32
                    + (rows[r].mask != b) as u32)
                    <= 1
            {
                let g = [rows[r - 1].t - rows[r - 2].t, rows[r].t - rows[r - 1].t, 0.0];
                let g0 = rows[r - 2].t - rows[r - 3].t;
                let gaps = [g0, g[0], g[1]];
                let mx = gaps.iter().cloned().fold(f64::MIN, f64::max);
                let mn = gaps.iter().cloned().fold(f64::MAX, f64::min);
                if mx <= 0.2 && mx <= 1.6 * mn {
                    for x in &mut oht[r - 3..=r] {
                        *x = true;
                    }
                }
            }
        }
    }
    let mut others = vec![0usize; keys];
    for c in 0..keys {
        let bit = 1u32 << c;
        let occ: Vec<usize> = (0..n).filter(|&r| rows[r].mask & bit != 0).collect();
        let mut run: Vec<usize> = occ.first().copied().into_iter().collect();
        for (x, &a) in occ.iter().enumerate() {
            let z = occ.get(x + 1).copied();
            if let Some(z) = z {
                if (2..=4).contains(&(z - a)) && rows[z].t - rows[a].t < 0.5 {
                    run.push(z);
                    continue;
                }
            }
            if run.len() >= 5 && run.len() * 3 >= run[run.len() - 1] - run[0] + 1 {
                others.iter_mut().for_each(|o| *o = 0);
                for r in run[0]..=run[run.len() - 1] {
                    let m = rows[r].mask & !bit;
                    for x in 0..keys {
                        if m >> x & 1 == 1 {
                            others[x] += 1;
                        }
                    }
                }
                let distinct = others.iter().filter(|&&o| o > 0).count();
                let mx = others.iter().copied().max().unwrap_or(0);
                if distinct >= 3 && mx as f64 <= 0.6 * run.len() as f64 {
                    for x in &mut anc[run[0]..=run[run.len() - 1]] {
                        *x = true;
                    }
                }
            }
            run = z.into_iter().collect();
        }
    }
    (oht, anc)
}

pub fn hand_of(keys: usize, centre_left: bool) -> Vec<u8> {
    let half = keys / 2;
    (0..keys).map(|c| if c < half || (keys % 2 == 1 && c == half && centre_left) { 0 } else { 1 }).collect()
}

/// Extra outputs of a run on an SV chart: the same physical work without the reading part.
pub struct Plain {
    pub run: Run,
}

/// One `_demand` run (see skill_calc._demand).
#[allow(clippy::too_many_arguments)]
pub fn demand(
    chart: &Chart,
    g: &Geometry,
    centre_left: bool,
    strain: Option<&[f64]>,
    kinds: Option<&[[f64; 5]]>,
    want_plain: bool,
    mut motor: Option<&mut Vec<Action>>,
    k: &Knobs,
) -> (Run, Option<Run>) {
    let keys = chart.keys;
    let wide = keys >= 5;
    let v_fast = k.V_FAST * ((keys.max(7) as f64) / 7.0).powf(k.V_FAST_KEYS);
    let w_jack = if keys == 7 { k.W_JACK_7K } else { k.W_JACK };
    let w_group = k.W_GROUP * (keys.min(7) as f64 / 7.0).powf(k.GROUP_KEYS);
    let jack_share = k.JACK_SHARE * (7.0 / keys.max(7) as f64).powf(k.JACK_SHARE_KEYS);
    let hand_of = hand_of(keys, centre_left);
    let n = g.times.len();
    let times = &g.times;
    let ends = &g.ends;
    let columns = &g.columns;
    let t0 = times[0];
    let max_end = ends.iter().cloned().fold(f64::MIN, f64::max);
    let bin = k.BIN;
    let phases = k.PHASES;
    let nbins = ((max_end - t0) / bin) as usize + 1;
    let nfine = nbins * phases;
    let zero = vec![0.0; if strain.is_none() { n } else { 0 }];
    let reading: &[f64] = strain.unwrap_or(&zero);
    let mut order = vec![0usize; keys];
    let mut seen = vec![false; keys];
    let mut last_head = vec![0.0f64; keys];
    let mut last_end = vec![0.0f64; keys];
    let mut last_size = vec![0usize; keys];
    let mut hold_weight = vec![0.0f64; keys];
    let mut nseen = 0usize;
    let mut previous_hand = [0.0f64; 2];
    let mut previous_mask = [0u32; 2];
    let mut hand_mask = [0u32; 2];
    for c in 0..keys {
        hand_mask[hand_of[c] as usize] |= 1 << c;
    }
    let mut jrun = vec![0usize; keys];
    let mut run_time = vec![0.0f64; keys];
    let mut runs: Vec<Vec<usize>> = vec![Vec::new(); keys];
    let mut burst: Vec<usize> = Vec::new();
    let rows = &g.rows;
    let nr = rows.len();
    let mut trill = vec![Trill::None; nr];
    for r in 3..nr {
        let (a, bb, cc, dd) = (rows[r - 3].mask, rows[r - 2].mask, rows[r - 1].mask, rows[r].mask);
        if a.count_ones().min(bb.count_ones()).min(cc.count_ones()).min(dd.count_ones()) >= 2
            && rows[r].t - rows[r - 3].t < 3.0 * k.TRILL_GAP
        {
            let hands = |m: u32| -> u8 {
                let mut s = 0u8;
                for c in 0..keys {
                    if m >> c & 1 == 1 {
                        s |= 1 << hand_of[c];
                    }
                }
                s
            };
            let (ha, hb, hc, hd) = (hands(a), hands(bb), hands(cc), hands(dd));
            let kind = if ha.count_ones() == 1 && hb.count_ones() == 1 && ha != hb && ha == hc && hb == hd {
                Trill::Jumptrill
            } else if !wide && cc == a && dd == bb && a & bb == 0 && ha.count_ones() == 2 && hb.count_ones() == 2 {
                Trill::Splittrill
            } else {
                Trill::None
            };
            if kind != Trill::None {
                for x in &mut trill[r - 3..=r] {
                    *x = kind;
                }
            }
        }
    }
    let (p_ins, p_sw) = pattern_evidence(rows, &trill, k);
    let roll = &g.roll;
    let (oht, anc) = hand_patterns(rows, hand_mask, keys);
    let keys_out = part_keys(wide);
    let mut comp = vec![0.0f64; NP * nbins];
    let mut hands = vec![0.0f64; 3 * nbins];
    let mut fine = vec![0.0f64; 3 * nfine];
    let mut plain_x = if want_plain { vec![0.0f64; nbins] } else { Vec::new() };
    let mut plain_fine_x = if want_plain { vec![0.0f64; nfine] } else { Vec::new() };
    let mut rows_by_size = [vec![0usize; nbins], vec![0usize; nbins], vec![0usize; nbins]];
    let o = |p: P| p as usize * nbins;
    let mut prev_row_t = 0.0f64;
    let mut prev_gap = 0.0f64;
    let mut pp_gap = 0.0f64;
    let mut grid = 0.0f64;
    let mut ev_t = 0.0f64;
    let mut ev_gap = 0.0f64;
    let mut ev_grid = 0.0f64;
    let mut ev_seen = false;
    let mut after_roll = false;
    let mut prev_row_mask = 0u32;
    let mut pending: Vec<(f64, usize)> = Vec::with_capacity(16);
    let mut shapes_seen: [Vec<(f64, u32)>; 2] = [Vec::new(), Vec::new()];
    let w_cross = if wide { k.W_CROSS_WIDE } else { k.W_CROSS };
    let flam = k.FLAM;
    for (ri, row) in rows.iter().enumerate() {
        let (i, j) = (row.start, row.end);
        let t = times[i];
        let size = j - i;
        let share = (size as f64).powf(-k.ROW_SHARE);
        let b = ((t - t0) / bin) as usize;
        let fb = (nfine - 1).min(((t - t0) * phases as f64 / bin) as usize);
        let tr: Option<usize> = match trill[ri] {
            Trill::None => None,
            Trill::Jumptrill => Some(o(P::jumptrill)),
            Trill::Splittrill => Some(o(P::splittrill)),
        };
        rows_by_size[size.min(3) - 1][b] += 1;
        let gap = if ri > 0 { t - prev_row_t } else { 0.0 };
        let cross = if gap != 0.0 { w_cross * v(gap, flam, 2.0) } else { 0.0 };
        let mut tech = 0.0;
        if gap != 0.0 && gap < 0.5 {
            let irr = if gap < 0.04 {
                1.0
            } else if prev_gap != 0.0 && prev_gap < 0.5 {
                let x = (gap / prev_gap).log2().abs();
                let frac = (x - py::round(x)).abs();
                if x < 0.08 {
                    0.0
                } else {
                    1f64.min(2.2 * 0f64.max(frac - 0.06) + 0.25 * x.min(1.0))
                }
            } else {
                0.0
            };
            tech = k.W_TECH * v(gap, flam, 2.0) * irr;
        }
        comp[o(P::technical) + b] += tech;
        let mut read = 0.0;
        let mut top = i;
        for ni in i..j {
            if reading[ni] > reading[top] {
                top = ni;
            }
        }
        if gap != 0.0 {
            read = k.W_SV * reading[top] * (6.0 * v(gap, flam, 2.0)).sqrt();
        }
        comp[o(P::sv) + b] += read;
        comp[o(P::sv_raw) + b] += read;
        if read != 0.0 {
            if let Some(kinds) = kinds.filter(|x| !x.is_empty()) {
                for (q, &f) in SV_KINDS.iter().zip(kinds[top].iter()) {
                    comp[o(*q) + b] += read * f;
                }
            }
        }
        hands[2 * nbins + b] += cross + tech + read;
        fine[2 * nfine + fb] += cross + tech + read;
        if want_plain {
            plain_x[b] += cross + tech + 0.0;
            plain_fine_x[fb] += cross + tech + 0.0;
        }
        let mut odd_v = 0.0;
        if gap != 0.0 && gap < 0.04 {
            let nxt = if j < n { times[j] - t } else { 1.0 };
            odd_v = if !(prev_gap != 0.0 && prev_gap < 0.06 || nxt < 0.06) { 1.0 } else { 0.0 };
        } else if gap != 0.0
            && (0.06..0.5).contains(&gap)
            && (grid != 0.0 || (prev_gap != 0.0 && (0.06..0.5).contains(&prev_gap) && !(pp_gap != 0.0 && pp_gap < 0.06)))
            && !(prev_gap != 0.0 && prev_gap < 0.06 && pp_gap != 0.0 && pp_gap < 0.06)
        {
            let x = (gap / if grid != 0.0 { grid } else { prev_gap }).log2().abs();
            odd_v = 1f64.min(2.2 * 0f64.max((x - py::round(x)).abs() - 0.06));
        }
        if gap != 0.0 && (0.06..0.5).contains(&gap) && prev_gap != 0.0 && (gap / prev_gap).log2().abs() < 0.06 {
            grid = gap;
        }
        let nxt = if j < n { times[j] - t } else { 0.0 };
        let rolling = roll[ri]
            && gap != 0.0
            && (prev_gap != 0.0 && (gap / prev_gap).log2().abs() < 0.06 || (nxt != 0.0 && (gap / nxt).log2().abs() < 0.06));
        if ri == 0 || gap >= k.R_EVENT {
            let eg = if ev_seen { t - ev_t } else { 0.0 };
            let judged = !(rolling || after_roll);
            let evref = if ev_grid != 0.0 { ev_grid } else { ev_gap };
            if judged && eg != 0.0 && eg < 0.5 && (ev_grid != 0.0 || (ev_gap != 0.0 && ev_gap < 0.5)) {
                comp[o(P::rodd) + b] +=
                    (if gap != 0.0 && gap < 0.04 { odd_v } else { 0.0 }).max(off_grid(eg, evref, ev_grid != 0.0));
            } else if judged && gap != 0.0 && gap < 0.04 {
                comp[o(P::rodd) + b] += odd_v;
            }
            if rolling {
            } else if eg != 0.0 && eg < 0.5 && ev_gap != 0.0 && (eg / ev_gap).log2().abs() < 0.06 {
                ev_grid = eg;
            } else if eg == 0.0 || eg >= 0.5 {
                ev_grid = 0.0;
            }
            ev_t = t;
            ev_gap = eg;
            after_roll = rolling;
            ev_seen = true;
        } else {
            comp[o(P::rodd) + b] += odd_v;
        }
        comp[o(P::rows) + b] += 1.0;
        comp[o(P::pins) + b] += p_ins[ri];
        comp[o(P::psw) + b] += p_sw[ri];
        comp[o(P::oht) + b] += oht[ri] as u8 as f64;
        comp[o(P::anc) + b] += anc[ri] as u8 as f64;
        let row_mask = row.mask;
        let any_vib = g.vib[i..j].iter().any(|&x| x);
        comp[o(P::vib) + b] += any_vib as u8 as f64;
        if gap != 0.0 && gap < 0.25 && row_mask & prev_row_mask != 0 {
            burst.push(b);
        } else {
            let part = if burst.len() <= k.MJ_BURST { P::mj } else { P::sus };
            for &rb in &burst {
                comp[o(part) + rb] += 1.0;
            }
            burst.clear();
        }
        let shared_count = (row_mask & prev_row_mask).count_ones();
        let mut n_hand = [0usize; 2];
        let mut n_jack = [0usize; 2];
        let row_hand = [row_mask & hand_mask[0], row_mask & hand_mask[1]];
        let mut mirror = [1.0f64; 2];
        let mut eh = [0.0f64; 2];
        for ni in i..j {
            let c = columns[ni] as usize;
            let h = hand_of[c] as usize;
            n_hand[h] += 1;
            n_jack[h] += (previous_mask[h] >> c & 1) as usize;
        }
        if n_jack[0] != 0 && n_jack[1] != 0 {
            let mh = if n_jack[0] <= n_jack[1] { 0 } else { 1 };
            mirror[mh] = k.MIRROR;
            if keys == 7 {
                let mut partial = false;
                for h in 0..2 {
                    let old_width = previous_mask[h].count_ones() as usize;
                    partial = partial || !(n_jack[h] == n_hand[h] && n_hand[h] == old_width);
                }
                if partial {
                    let speed = v(t - previous_hand[0], 0.07, 0.5).min(v(t - previous_hand[1], 0.07, 0.5));
                    mirror[mh] += (1.0 - k.MIRROR) * smooth(speed, 7.0, 9.0);
                }
            }
        }
        let mut jack_effort = [0.0f64; 2];
        let mut inserted = [0.0f64; 2];
        for h in 0..2 {
            if n_jack[h] != 0 {
                let vj = v(t - previous_hand[h], 0.07, 0.5);
                jack_effort[h] =
                    w_jack * vj * (vj / 12.0).powf(k.JACK_P) * (n_jack[h] as f64).powf(-jack_share) * mirror[h];
                inserted[h] = k.W_IJACK + (1.0 - k.W_IJACK) * smooth(vj, 7.0, 11.0);
            }
        }
        // How many chords the hand must have ready: user's duplicated-4K map is 44 NPS with few shapes and plays far easier than a
        // generated map at the same speed. Grows past what a 7K hand can reach, so wider keymodes pay it on their own.
        let mut many = [0.0f64; 2];
        if k.W_SHAPES != 0.0 {
            for h in 0..2 {
                if n_hand[h] >= 2 {
                    shapes_seen[h].push((t, row_hand[h]));
                }
                shapes_seen[h].retain(|x| x.0 >= t - 2.0);
                if n_hand[h] >= 2 {
                    let mut count = 0.0f64;
                    let mut wide_all: Vec<u32> = Vec::new();
                    for x in &shapes_seen[h] {
                        if !wide_all.contains(&x.1) {
                            wide_all.push(x.1);
                            count += 1.0;
                        }
                    }
                    many[h] = k.W_SHAPES * (count / (k.SHAPES_T * (10.0 / keys as f64).powf(k.SHAPES_KEYS))).max(1.0).log2();
                }
            }
        }
        let mut mix = [0.0f64; 2];
        for h in 0..2 {
            if k.W_MIX != 0.0 && row_hand[h] & previous_mask[h] != 0 && row_hand[h] & !previous_mask[h] != 0 {
                let edge = k.MIX_V * (7.0 / keys as f64).powf(k.MIX_KEYS);
                mix[h] = k.W_MIX * smooth(v(t - previous_hand[h], 0.07, 0.5), edge, edge * 1.3);
            }
        }
        pending.clear();
        let mut jack_w = 0.0f64;
        for ni in i..j {
            let e = ends[ni];
            let c = columns[ni] as usize;
            let h = hand_of[c] as usize;
            let jacked = previous_mask[h] >> c & 1 == 1;
            let mut dh = 0.0f64;
            for oc in 0..keys {
                if seen[oc]
                    && hand_of[oc] as usize == h
                    && oc != c
                    && !(jacked && (row_mask | previous_mask[h]) >> oc & 1 == 1)
                {
                    let gg = t - last_head[oc];
                    if dh == 0.0 || gg < dh {
                        dh = gg;
                    }
                }
            }
            let dj = if seen[c] { (t - last_head[c]).min(1.0) } else { 1.0 };
            let vdj = v(dj, flam, 2.0);
            let vdh = if dh != 0.0 { v(dh, flam, 2.0) } else { 0.0 };
            let e_hand = if dh != 0.0 {
                k.W_HAND * (vdh * vdj).sqrt() * share * 1f64.max(vdh / v_fast).powf(k.SPEED_P)
            } else {
                0.0
            };
            eh[h] += e_hand;
            let mut e_jack = if jacked { jack_effort[h] } else { 0.0 };
            if jacked && prev_row_mask >> c & 1 == 0 {
                e_jack *= inserted[h];
            }
            // User: "when chords switch, it gets much harder, the higher the keycount is". Rejected on replays: entropy per row,
            // neighbour left up (W_ISO, overlaps the groups), fast jack premium, reviving V_FAST on 10K.
            let groups = (row_hand[h] & !(row_hand[h] << 1)).count_ones() as f64;
            let cost = 1. + mix[h] + many[h] + if size as f64 >= k.GROUP_SIZE { w_group * (groups - 1.) } else { 0. };
            let rice = (e_hand + e_jack) * 1f64.max(vdj / k.V_FINGER) * cost;
            let mut held_same = 0.0f64;
            let mut held_other = 0.0f64;
            for &oc in &order[..nseen] {
                if oc != c && last_head[oc] < t - 0.003 && last_end[oc] > t + 0.003 {
                    if hand_of[oc] as usize == h {
                        held_same += hold_weight[oc];
                    } else {
                        held_other += hold_weight[oc];
                    }
                }
            }
            let mut e_ln = (rice + k.LN_BASE * share)
                * (k.W_HOLD_SAME * held_same.powf(0.7) + k.W_HOLD_OTHER * held_other.min(1.0));
            if e - t <= 0.003 && held_same + held_other >= 0.5 && gap != 0.0 && gap < 0.15 {
                comp[o(P::hyb) + b] += 1.0;
            }
            if seen[c] && last_end[c] - last_head[c] <= 0.003 && e - t > 0.003 && t - last_head[c] < 0.12 {
                comp[o(P::shield) + b] += 1.0;
            }
            if seen[c] && last_end[c] - last_head[c] > 0.003 {
                let gg = t - last_end[c];
                if gg < 0.25 {
                    let rep = k.W_REPRESS
                        * v(gg, 0.04, 0.5)
                        * (1.0 - smooth(gg, 0.12, 0.25))
                        * smooth(last_end[c] - last_head[c], 0.06, 0.25)
                        * share;
                    e_ln += rep;
                    if gg < 0.12 && last_end[c] - last_head[c] >= 0.1 {
                        comp[o(P::inv) + b] += 1.0;
                    }
                }
            }
            hands[h * nbins + b] += rice + e_ln;
            fine[h * nfine + fb] += rice + e_ln;
            comp[o(P::ln) + b] += e_ln;
            if let Some(m) = motor.as_deref_mut() {
                m[ni] = Action {
                    hand: h as u8,
                    rice,
                    cross: (cross + tech) / size as f64,
                    protected: jacked || e_ln > 0.0 || e - t > 0.003,
                    fb,
                };
            }
            jrun[c] = if jacked { jrun[c] + 1 } else { 0 };
            if jacked
                && prev_row_mask >> c & 1 == 1
                && shared_count == 1
                && e - t <= 0.003
                && last_end[c] - last_head[c] <= 0.003
                && t - run_time[c] < 0.25
            {
                runs[c].push(b);
            } else {
                if runs[c].len() >= 3 {
                    for &rb in &runs[c] {
                        comp[o(P::lj) + rb] += 1.0;
                    }
                }
                runs[c].clear();
            }
            run_time[c] = t;
            if wide && e_jack != 0.0 && (jrun[c] >= 2 || n_hand[h] >= 2) {
                jack_w += 1.0;
                let part = if n_jack[h] >= 2 || n_hand[h] >= 2 { P::chordjack } else { P::jack };
                comp[tr.unwrap_or(o(part)) + b] += rice;
            } else if wide && e_jack != 0.0 {
                pending.push((rice, h));
            } else if !wide && e_jack != 0.0 && prev_row_mask >> c & 1 == 1 {
                jack_w += 1.0;
                let part = if size >= 2 && seen[c] && last_size[c] >= 2 { P::chordjack } else { P::jackspeed };
                comp[tr.unwrap_or(o(part)) + b] += rice;
            } else {
                pending.push((rice, h));
            }
        }
        comp[o(P::odd) + b] += odd_v;
        let split = k.W_SPLIT * eh[0].min(eh[1]);
        hands[2 * nbins + b] += split;
        fine[2 * nfine + fb] += split;
        if want_plain {
            plain_x[b] += split;
            plain_fine_x[fb] += split;
        }
        let fast = 1.0 - smooth(if gap != 0.0 { gap } else { 1.0 }, k.DELAY_GAP - 0.01, k.DELAY_GAP + 0.01);
        if wide {
            if k.W_ALT != 0.0 {
                for h in 0..2 {
                    if row_hand[h] != 0
                        && previous_mask[h] != 0
                        && n_jack[h] == 0
                        && previous_hand[h] == prev_row_t
                        && (inner(previous_mask[h], keys) & row_hand[h] != 0
                            || inner(row_hand[h], keys) & previous_mask[h] != 0)
                    {
                        let mut alt = 0.0;
                        for &(amount, hh) in &pending {
                            if hh == h {
                                alt += amount;
                            }
                        }
                        alt *= k.W_ALT;
                        hands[h * nbins + b] += alt;
                        fine[h * nfine + fb] += alt;
                        pending.push((alt, h));
                    }
                }
            }
            let jf = jack_w / size as f64;
            let mut cs_sum = 0.0;
            let mut ds_sum = 0.0;
            for &(amount, h) in &pending {
                if n_hand[h] >= 2 {
                    cs_sum += amount;
                } else {
                    ds_sum += amount;
                }
            }
            cs_sum += if size >= 3 { cross + split } else { 0.0 };
            ds_sum += if size < 3 { cross + split } else { 0.0 };
            comp[tr.unwrap_or(o(P::chordstream)) + b] += cs_sum * (1.0 - jf);
            comp[tr.unwrap_or(o(P::delay)) + b] += ds_sum * (1.0 - jf) * fast;
            comp[tr.unwrap_or(o(P::stream)) + b] += ds_sum * (1.0 - jf) * (1.0 - fast);
            let part = if size >= 2 { P::chordjack } else { P::jack };
            comp[tr.unwrap_or(o(part)) + b] += (cs_sum + ds_sum) * jf;
            for h in 0..2 {
                let old_width = previous_mask[h].count_ones() as usize;
                if n_hand[h] != 0
                    && n_jack[h] == 0
                    && previous_mask[h] != 0
                    && previous_hand[h] == prev_row_t
                    && n_hand[h].max(old_width) >= 2
                {
                    let mut other = 0.0;
                    for &(amount, hh) in &pending {
                        if hh == h {
                            other += amount;
                        }
                    }
                    comp[o(P::bracket) + b] += other * (1.0 - jf);
                }
            }
        } else {
            let jf = if size >= 2 { 1f64.min(2.0 * jack_w / size as f64) } else { jack_w };
            let mut other = 0.0;
            for &(amount, _) in &pending {
                other += amount;
            }
            other = other + cross + split;
            comp[tr.unwrap_or(o(P::stream)) + b] += other * (1.0 - jf);
            let part = if size >= 2 { P::chordjack } else { P::jackspeed };
            comp[tr.unwrap_or(o(part)) + b] += other * jf;
            if tr.is_none() {
                let mut ds_sum = 0.0;
                for &(amount, h) in &pending {
                    if n_hand[h] < 2 {
                        ds_sum += amount;
                    }
                }
                ds_sum += if size < 3 { cross + split } else { 0.0 };
                comp[o(P::dump) + b] += ds_sum * (1.0 - jf) * fast;
            }
        }
        for ni in i..j {
            let c = columns[ni] as usize;
            let e = ends[ni];
            if !seen[c] {
                order[nseen] = c;
                nseen += 1;
                seen[c] = true;
            }
            last_head[c] = t;
            last_end[c] = e;
            last_size[c] = size;
            hold_weight[c] = smooth(e - t, 0.06, 0.25);
        }
        for h in 0..2 {
            if row_hand[h] != 0 {
                previous_hand[h] = t;
                previous_mask[h] = row_hand[h];
            }
        }
        pp_gap = prev_gap;
        prev_gap = gap;
        prev_row_t = t;
        prev_row_mask = row_mask;
    }
    // releases and staggered LN actions, per hand
    let mut acts: [Vec<(f64, u8)>; 2] = [Vec::new(), Vec::new()];
    for ni in 0..n {
        let (t, e, c) = (times[ni], ends[ni], columns[ni]);
        let h = hand_of[c as usize] as usize;
        acts[h].push((t, c));
        if e - t > 0.003 {
            acts[h].push((e, c));
        }
    }
    for a in &mut acts {
        a.sort_by(|x, y| x.partial_cmp(y).unwrap());
    }
    let ats: [Vec<f64>; 2] = [acts[0].iter().map(|x| x.0).collect(), acts[1].iter().map(|x| x.0).collect()];
    for ni in 0..n {
        let (t, e, c) = (times[ni], ends[ni], columns[ni]);
        let ln_len = e - t;
        if ln_len <= 0.003 {
            continue;
        }
        let h = hand_of[c as usize] as usize;
        let at = &ats[h];
        let lo = at.partition_point(|&x| x < e - 0.18);
        let hi = at.partition_point(|&x| x <= e + 0.18);
        let mut sync = 1usize;
        let mut stag = 0.0f64;
        let mut prev_d = 0.0f64;
        for oi in lo..hi {
            if acts[h][oi].1 == c {
                continue;
            }
            let ot = at[oi];
            let d = (ot - e).abs();
            if d <= 0.005 {
                sync += 1;
            } else {
                stag += smooth(d, 0.005, 0.03) * (1.0 - smooth(d, 0.08, 0.18));
            }
            if ot < e - 0.005 {
                prev_d = if prev_d == 0.0 { e - ot } else { prev_d.min(e - ot) };
            }
        }
        let base = k.W_REL * (k.LN_BASE + if prev_d != 0.0 { v(prev_d, flam, 2.0) } else { 0.0 });
        let eff = (base * (sync as f64).powf(-k.ROW_SHARE) + k.W_STAG * 10.0 * stag.min(2.0)) * smooth(ln_len, 0.06, 0.25);
        let b = (nbins - 1).min(((e - t0) / bin) as usize);
        let fb = (nfine - 1).min(((e - t0) * phases as f64 / bin) as usize);
        hands[h * nbins + b] += eff;
        fine[h * nfine + fb] += eff;
        comp[o(P::ln) + b] += eff;
        comp[o(P::ln_rel) + b] += eff;
    }
    for c in 0..keys {
        if runs[c].len() >= 3 {
            for &rb in &runs[c] {
                comp[o(P::lj) + rb] += 1.0;
            }
        }
    }
    let part = if burst.len() <= k.MJ_BURST { P::mj } else { P::sus };
    for &rb in &burst {
        comp[o(part) + rb] += 1.0;
    }
    if !wide {
        for b in 0..nbins {
            let low = b.saturating_sub(2);
            let high = nbins.min(b + 3);
            let singles: usize = rows_by_size[0][low..high].iter().sum();
            let jumps: usize = rows_by_size[1][low..high].iter().sum();
            let chords: usize = rows_by_size[2][low..high].iter().sum();
            let count = singles + jumps + chords;
            if count != 0 && comp[o(P::stream) + b] != 0.0 {
                let hs = 1f64.min(3.0 * chords as f64 / count as f64);
                let js = (1.0 - hs) * 1f64.min(3.0 * jumps as f64 / count as f64);
                let work = comp[o(P::stream) + b];
                comp[o(P::handstream) + b] = work * hs;
                comp[o(P::jumpstream) + b] = work * js;
                comp[o(P::stream) + b] = work * (1.0 - hs - js);
            }
        }
    }
    let offsets: Vec<usize> = keys_out.iter().filter(|&&p| !non_demand(p)).map(|&p| o(p)).collect();
    let plain = if want_plain {
        let mut pc = comp.clone();
        for p in [P::sv, P::sv_raw].iter().chain(SV_KINDS.iter()) {
            for x in &mut pc[o(*p)..o(*p) + nbins] {
                *x = 0.0;
            }
        }
        let mut pt = vec![0.0f64; nbins];
        for b in 0..nbins {
            let (left, right, cv) = (hands[b], hands[nbins + b], plain_x[b]);
            let raw = left + right + cv;
            if raw > 0.0 {
                pt[b] = ((left * left + right * right).sqrt() + cv) / bin;
                let factor = pt[b] / raw;
                for &off in &offsets {
                    pc[off + b] *= factor;
                }
            }
        }
        Some(Run {
            parts: Parts { nbins, data: pc, keys: keys_out.clone() },
            totals: pt,
            fine: [fine[..nfine].to_vec(), fine[nfine..2 * nfine].to_vec(), plain_fine_x],
        })
    } else {
        None
    };
    let mut totals = vec![0.0f64; nbins];
    for b in 0..nbins {
        let (left, right, cv) = (hands[b], hands[nbins + b], hands[2 * nbins + b]);
        let raw = left + right + cv;
        if raw > 0.0 {
            totals[b] = ((left * left + right * right).sqrt() + cv) / bin;
            let factor = totals[b] / raw;
            for &off in &offsets {
                comp[off + b] *= factor;
            }
        }
    }
    let fine = [fine[..nfine].to_vec(), fine[nfine..2 * nfine].to_vec(), fine[2 * nfine..].to_vec()];
    (Run { parts: Parts { nbins, data: comp, keys: keys_out }, totals, fine }, plain)
}
