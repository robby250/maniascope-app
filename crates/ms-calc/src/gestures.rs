//! Timing-feasible hand gestures (gestures.py).
use crate::chart::Chart;
use crate::kern::smooth;
use crate::py;

pub const FEATURES: [&str; 8] = [
    "vibro_easy", "vibro_control", "vibro_sustain", "jumptrill_easy", "roll_control", "roll_anchor",
    "jumptrill_sustain", "vibro_locked",
];
pub const SERIES: [&str; 13] = [
    "rolled", "jumptrill", "control", "anchor", "vibro", "vibro_easy", "vibro_control", "vibro_sustain",
    "jumptrill_easy", "roll_control", "roll_anchor", "jumptrill_sustain", "vibro_locked",
];
pub mod s {
    pub const ROLLED: usize = 0;
    pub const JUMPTRILL: usize = 1;
    pub const CONTROL: usize = 2;
    pub const ANCHOR: usize = 3;
    pub const VIBRO: usize = 4;
    pub const VIBRO_EASY: usize = 5;
    pub const VIBRO_CONTROL: usize = 6;
    pub const VIBRO_SUSTAIN: usize = 7;
    pub const JUMPTRILL_EASY: usize = 8;
    pub const ROLL_CONTROL: usize = 9;
    pub const ROLL_ANCHOR: usize = 10;
    pub const JUMPTRILL_SUSTAIN: usize = 11;
    pub const VIBRO_LOCKED: usize = 12;
}

pub fn great_window(od: f64) -> f64 {
    (64.0 - 3.0 * 10f64.min(0f64.max(od))) / 1000.0
}

#[derive(Default)]
pub struct Notes {
    pub t: Vec<f64>,
    pub e: Vec<f64>,
    pub c: Vec<u8>,
}

/// (row time, first note, end note).
pub type Rows = Vec<(f64, usize, usize)>;

#[derive(Clone, Debug)]
pub struct Packet {
    pub center: f64,
    pub hand: u8,
    pub ids: Vec<usize>,
    pub cols: u32,
    pub span: f64,
    pub direction: f64,
    pub gap: f64,
    pub held: f64,
}

fn packets(notes: &Notes, rows: &Rows, hand_of: &[u8], window: f64) -> Vec<Packet> {
    let n = notes.t.len();
    let mut out: Vec<Packet> = Vec::new();
    let mut segments = vec![0usize; n];
    let mut segment = 0usize;
    for i in 0..n {
        if i > 0 && notes.t[i] - notes.t[i - 1] >= 0.045 {
            segment += 1;
        }
        segments[i] = segment;
    }
    for h in 0..2u8 {
        let mut group: Vec<usize> = Vec::new();
        let mut used = 0u32;
        let mut holds: Vec<(u8, f64)> = Vec::new(); // dict col -> end, insertion order
        let mut group_hold = 0.0f64;
        let close = |group: &mut Vec<usize>, used: &mut u32, group_hold: &mut f64, out: &mut Vec<Packet>| {
            if group.is_empty() {
                return;
            }
            let t0 = notes.t[group[0]];
            let tl = notes.t[*group.last().unwrap()];
            if t0 == tl {
                let mean = py::sum(std::iter::repeat(t0).take(group.len())) / group.len() as f64;
                let cols = group.iter().fold(0u32, |m, &i| m | 1 << notes.c[i]);
                out.push(Packet {
                    center: mean, hand: h, ids: group.clone(), cols, span: 0.0, direction: 0.0, gap: 0.0,
                    held: *group_hold,
                });
            } else {
                let ts: Vec<f64> = group.iter().map(|&i| notes.t[i]).collect();
                let order: Vec<u8> = group.iter().map(|&i| notes.c[i]).collect();
                let dirs: Vec<i64> = order.windows(2).map(|w| if w[1] > w[0] { 1 } else { -1 }).collect();
                let direction = if !dirs.is_empty() && ts[ts.len() - 1] - ts[0] > 0.003 {
                    dirs.iter().sum::<i64>() as f64 / dirs.len() as f64
                } else {
                    0.0
                };
                let mut gap: Option<f64> = None;
                for i in group[0] + 1..=*group.last().unwrap() {
                    let g = notes.t[i] - notes.t[i - 1];
                    gap = Some(match gap {
                        Some(m) if m >= g => m,
                        _ => g,
                    });
                }
                out.push(Packet {
                    center: py::sum(ts.iter().copied()) / ts.len() as f64,
                    hand: h,
                    ids: group.clone(),
                    cols: order.iter().fold(0u32, |m, &c| m | 1 << c),
                    span: ts[ts.len() - 1] - ts[0],
                    direction,
                    gap: gap.unwrap_or(0.0),
                    held: *group_hold,
                });
            }
            group.clear();
            *used = 0;
            *group_hold = 0.0;
        };
        for &(t, a, z) in rows {
            let mut local: Vec<usize> = (a..z).filter(|&i| hand_of[notes.c[i] as usize] == h).collect();
            if local.is_empty() {
                continue;
            }
            let long: Vec<usize> = local.iter().copied().filter(|&i| notes.e[i] - t > 0.003).collect();
            if !long.is_empty() {
                close(&mut group, &mut used, &mut group_hold, &mut out);
                for &i in &long {
                    let (c, e) = (notes.c[i], notes.e[i]);
                    match holds.iter_mut().find(|x| x.0 == c) {
                        Some(x) => x.1 = e,
                        None => holds.push((c, e)),
                    }
                }
                local.retain(|i| !long.contains(i));
                if local.is_empty() {
                    continue;
                }
            }
            let cols = local.iter().fold(0u32, |m, &i| m | 1 << notes.c[i]);
            if !group.is_empty()
                && (used & cols != 0
                    || t - notes.t[group[0]] > 2.0 * window
                    || segments[local[0]] != segments[*group.last().unwrap()]
                    || local.len() > 1)
            {
                close(&mut group, &mut used, &mut group_hold, &mut out);
            }
            group.extend_from_slice(&local);
            used |= cols;
            let held = holds.iter().filter(|x| x.1 > t + 0.003).count() as f64;
            group_hold = group_hold.max(held);
            if local.len() > 1 {
                close(&mut group, &mut used, &mut group_hold, &mut out);
            }
        }
        close(&mut group, &mut used, &mut group_hold, &mut out);
    }
    out.sort_by(|a, b| (a.center, a.hand).partial_cmp(&(b.center, b.hand)).unwrap());
    out
}

#[derive(Clone, Debug)]
pub struct Plan {
    pub ids: Vec<usize>,
    pub time: f64,
    pub strength: f64,
    pub control: f64,
    pub anchor: f64,
    pub span: f64,
    pub offsets: Vec<f64>,
}

/// [rolled, jumptrill, control, anchor, jumptrill_easy] per note.
fn jumptrills(notes: &Notes, rows: &Rows, hand_of: &[u8], window: f64, mut plans: Option<&mut Vec<Plan>>) -> [Vec<f64>; 5] {
    let n = notes.t.len();
    let mut res: [Vec<f64>; 5] = std::array::from_fn(|_| vec![0.0; n]);
    let packets = packets(notes, rows, hand_of, window);
    let np = packets.len();
    let want = plans.is_some();
    let mut support = vec![0.0f64; if want { np } else { 0 }];
    let mut positions: Vec<Vec<(f64, f64)>> = vec![Vec::new(); if want { np } else { 0 }];
    let mut cross = vec![false; n];
    for &(_, a, z) in rows {
        let hs = (a..z).fold(0u8, |m, i| m | 1 << hand_of[notes.c[i] as usize]);
        if hs.count_ones() > 1 {
            for x in &mut cross[a..z] {
                *x = true;
            }
        }
    }
    let mut previous: [Option<usize>; 2] = [None, None];
    let mut previous_packets: Vec<Option<usize>> = Vec::with_capacity(np);
    for (j, p) in packets.iter().enumerate() {
        previous_packets.push(previous[p.hand as usize]);
        previous[p.hand as usize] = Some(j);
    }
    let mut controls: Vec<Option<f64>> = vec![None; np];
    let mut anchors = vec![0.0f64; np];
    let control_at = |j: usize, controls: &mut Vec<Option<f64>>, anchors: &mut Vec<f64>| {
        if controls[j].is_some() {
            return;
        }
        let p = &packets[j];
        let mut control = 0.18 * 1f64.min(p.span / (2.0 * window)) + 0.25 * 1f64.min(p.held);
        let mut anchor = 0.0f64;
        if let Some(pj) = previous_packets[j] {
            let prev = &packets[pj];
            if 0.0 < p.center - prev.center && p.center - prev.center < 0.4 {
                let cycle = p.center - prev.center;
                let change = 1.0
                    - (p.cols & prev.cols).count_ones() as f64 / 1u32.max((p.cols | prev.cols).count_ones()) as f64;
                let turn = 0f64.max(-p.direction * prev.direction);
                let mut best: Option<f64> = None;
                for &i in &p.ids {
                    let c = notes.c[i];
                    if let Some(&li) = prev.ids.iter().rev().find(|&&x| notes.c[x] == c) {
                        let v = 0f64.max(1.0 - (notes.t[i] - notes.t[li]) / cycle);
                        best = Some(match best {
                            Some(m) if m >= v => m,
                            _ => v,
                        });
                    }
                }
                anchor = best.unwrap_or(0.0);
                control += 0.25 * change + 0.3 * turn + 0.5 * anchor;
            }
        }
        controls[j] = Some(1f64.min(control));
        anchors[j] = anchor;
    };
    for r in 3..np {
        let seg = &packets[r - 3..=r];
        if seg.iter().any(|p| p.cols.count_ones() < 2) {
            continue;
        }
        if seg.windows(2).any(|w| w[0].hand == w[1].hand) {
            continue;
        }
        let gap = ((seg[2].center - seg[0].center) + (seg[3].center - seg[1].center)) / 4.0;
        if !(0.0 < gap && gap < 0.2) || seg.windows(2).any(|w| w[1].center - w[0].center > 0.25) {
            continue;
        }
        let nids: usize = seg.iter().map(|p| p.ids.len()).sum();
        let ncross: usize = seg.iter().flat_map(|p| p.ids.iter()).filter(|&&i| cross[i]).count();
        if ncross as f64 > 0.25 * nids as f64 {
            continue;
        }
        let rolled = seg.iter().any(|p| p.span > 0.003);
        let speed_of = || {
            seg.iter().filter(|p| p.span > 0.003).map(|p| 1.0 - smooth(p.gap, 0.025, 0.045)).fold(f64::INFINITY, |m, x| if x < m { x } else { m })
        };
        if want && rolled {
            let pl = seg.iter().enumerate().map(|(j, p)| p.center - window - j as f64 * gap).fold(f64::NEG_INFINITY, |m, x| if x > m { x } else { m });
            let ph = seg.iter().enumerate().map(|(j, p)| p.center + window - j as f64 * gap).fold(f64::INFINITY, |m, x| if x < m { x } else { m });
            let mut ps = speed_of();
            ps *= smooth((ph - pl) / (2.0 * window), 0.0, 0.35);
            if ph > pl && ps > 0.0 {
                for (jj, p) in seg.iter().enumerate() {
                    let j = r - 3 + jj;
                    if p.span > 0.003 {
                        support[j] = support[j].max(ps);
                        positions[j].push(((pl + ph) * 0.5 + jj as f64 * gap, ps));
                    }
                }
            }
        }
        let strength = if rolled {
            let mut lo = f64::NEG_INFINITY;
            let mut hi = f64::INFINITY;
            for (j, p) in seg.iter().enumerate() {
                for &i in &p.ids {
                    let a = notes.t[i] - window - j as f64 * gap;
                    let b = notes.t[i] + window - j as f64 * gap;
                    if a > lo {
                        lo = a;
                    }
                    if b < hi {
                        hi = b;
                    }
                }
            }
            if lo >= hi {
                continue;
            }
            let speed = speed_of();
            let margin = (hi - lo) / (2.0 * window);
            speed * smooth(margin, 0.0, 0.35)
        } else {
            1.0
        };
        let quick = smooth(1.0 / (2.0 * gap), 6.0, 11.0);
        for jj in 0..4 {
            let j = r - 3 + jj;
            control_at(j, &mut controls, &mut anchors);
            let p = &packets[j];
            let cj = controls[j].unwrap();
            for &i in &p.ids {
                res[1][i] = res[1][i].max(strength);
                if p.span > 0.003 {
                    res[0][i] = res[0][i].max(strength);
                }
                res[2][i] = res[2][i].max(strength * cj);
                res[3][i] = res[3][i].max(strength * anchors[j]);
                res[4][i] = res[4][i].max(strength * quick * (1.0 - cj));
            }
        }
    }
    if let Some(plans) = plans.as_deref_mut() {
        let mut row_sizes = vec![0usize; n];
        for &(_, a, z) in rows {
            for x in &mut row_sizes[a..z] {
                *x = z - a;
            }
        }
        for j in 0..np {
            let p = &packets[j];
            if support[j] == 0.0 || p.span <= 0.003 || p.held != 0.0 || p.ids.iter().any(|&i| row_sizes[i] != 1) {
                continue;
            }
            let target = py::sum(positions[j].iter().map(|&(t, w)| t * w)) / py::sum(positions[j].iter().map(|&(_, w)| w));
            control_at(j, &mut controls, &mut anchors);
            plans.push(Plan {
                ids: p.ids.clone(),
                time: target,
                strength: support[j],
                control: controls[j].unwrap(),
                anchor: anchors[j],
                span: p.span,
                offsets: p.ids.iter().map(|&i| notes.t[i] - p.center).collect(),
            });
        }
    }
    res
}

fn repetition_lock(rows: &Rows, masks: &[u32], clean: &[bool]) -> Vec<f64> {
    let n = rows.len();
    let mut left = vec![0.0f64; n];
    let mut right = vec![0.0f64; n];
    for r in 1..n {
        if clean[r] && clean[r - 1] && masks[r] == masks[r - 1] {
            left[r] = (-(rows[r].0 - rows[r - 1].0) / 0.5).exp() * (1.0 + left[r - 1]);
        }
    }
    for r in (0..n.saturating_sub(1)).rev() {
        if clean[r] && clean[r + 1] && masks[r] == masks[r + 1] {
            right[r] = (-(rows[r + 1].0 - rows[r].0) / 0.5).exp() * (1.0 + right[r + 1]);
        }
    }
    left.iter().zip(&right).map(|(a, b)| smooth(1.0 + a + b, 6.0, 12.0)).collect()
}

/// [vibro, vibro_easy, vibro_control, vibro_sustain, vibro_locked] per note.
fn vibro(notes: &Notes, rows: &Rows) -> [Vec<f64>; 5] {
    let n = rows.len();
    let nn = notes.t.len();
    let mut res: [Vec<f64>; 5] = std::array::from_fn(|_| vec![0.0; nn]);
    let masks: Vec<u32> = rows.iter().map(|&(_, a, z)| (a..z).fold(0u32, |m, i| m | 1 << notes.c[i])).collect();
    let clean: Vec<bool> = rows
        .iter()
        .zip(&masks)
        .map(|(&(_, a, z), m)| m.count_ones() as usize == z - a && (a..z).all(|i| notes.e[i] - notes.t[i] <= 0.003))
        .collect();
    let mut speed = vec![0.0f64; n];
    let mut similarity = vec![0.0f64; n];
    let mut connected = vec![false; n];
    for r in 1..n {
        let gap = rows[r].0 - rows[r - 1].0;
        if gap <= 0.0 || !(clean[r] && clean[r - 1]) {
            continue;
        }
        let shared = (masks[r] & masks[r - 1]).count_ones();
        if shared == 0 {
            continue;
        }
        connected[r] = true;
        speed[r] = smooth(1.0 / gap, 9.0, 13.0);
        similarity[r] = shared as f64 / (masks[r] | masks[r - 1]).count_ones() as f64;
    }
    if !speed.iter().any(|&x| x != 0.0) {
        return res;
    }
    let locked = repetition_lock(rows, &masks, &clean);
    let mut pressure = vec![0.0f64; n];
    let mut start = 1usize;
    while start < n {
        if !connected[start] {
            start += 1;
            continue;
        }
        let mut end = start + 1;
        while end < n && connected[end] {
            end += 1;
        }
        for r in start..end {
            if speed[r] == 0.0 {
                continue;
            }
            let mass = py::sum((start.max(r.saturating_sub(4))..end.min(r + 5)).map(|j| (-(rows[j].0 - rows[r].0).abs() / 0.3).exp()));
            let support = smooth(mass, 1.6, 4.0);
            let local = &similarity[start.max(r.saturating_sub(3))..end.min(r + 4)];
            let stable = py::sum(local.iter().map(|&x| x.powf(4.0))) / local.len() as f64;
            let mut gaps: Vec<f64> = (start.max(r.saturating_sub(3))..end.min(r + 4)).map(|j| rows[j].0 - rows[j - 1].0).collect();
            let unsorted = gaps.clone();
            gaps.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let pulse = gaps[gaps.len() / 2];
            let active: Vec<f64> = unsorted.into_iter().filter(|&g| g <= 2.0 * pulse).collect();
            let mn = active.iter().cloned().fold(f64::INFINITY, |m, x| if x < m { x } else { m });
            let mx = active.iter().cloned().fold(f64::NEG_INFINITY, |m, x| if x > m { x } else { m });
            let rhythm = mn / mx;
            let easy = stable * rhythm;
            let v = speed[r] * support;
            pressure[r] = v;
            let (_, a, z) = rows[r];
            for i in a..z {
                res[0][i] = v * (0.25 + 0.75 * easy);
                res[1][i] = v * easy;
                res[2][i] = v * (1.0 - easy);
                res[4][i] = v * easy * locked[r];
            }
        }
        start = end;
    }
    let mut state = 0.0f64;
    for r in 1..n {
        let v = pressure[r];
        let decay = (-(rows[r].0 - rows[r - 1].0) / if v > state { 20.0 } else { 4.0 }).exp();
        state = v + (state - v) * decay;
        let (_, a, z) = rows[r];
        for i in a..z {
            res[3][i] = v * state;
        }
    }
    res
}

#[derive(Default)]
pub struct Analysis {
    pub series: [Vec<f64>; 13],
    pub features: [f64; 8],
    /// centre_left (false, true) → plans
    pub plans: Option<Vec<(bool, Vec<Plan>)>>,
    pub notes: Notes,
    pub rows: Rows,
}

pub fn analyse(chart: &Chart, rate: f64, bin_seconds: f64, with_plans: bool) -> Analysis {
    let origin = chart.t.first().copied().unwrap_or(0.0);
    let conv = |x: f64| (x - origin) * 0.001 / rate;
    let notes = Notes { t: chart.t.iter().map(|&x| conv(x)).collect(), e: chart.end.iter().map(|&x| conv(x)).collect(), c: chart.col.clone() };
    let nn = notes.t.len();
    if nn == 0 {
        return Analysis { series: std::array::from_fn(|_| Vec::new()), features: [0.0; 8], plans: None, notes, rows: Vec::new() };
    }
    let first = notes.t[0];
    let max_e = notes.e.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let bins = ((max_e - first) / bin_seconds + 1e-9) as usize + 1;
    let mut rows: Rows = Vec::new();
    let mut at = 0.0f64;
    let mut start = 0usize;
    for i in 0..nn {
        let t = notes.t[i];
        if i > start && t - at > 0.003 {
            rows.push((at, start, i));
            start = i;
        }
        if i == start {
            at = t;
        }
    }
    rows.push((at, start, nn));
    let half = chart.keys / 2;
    let assignments: &[bool] = if chart.keys % 2 == 1 { &[false, true] } else { &[false] };
    let mut js: Vec<[Vec<f64>; 5]> = Vec::new();
    let mut plans_out: Vec<(bool, Vec<Plan>)> = Vec::new();
    let window = great_window(chart.od);
    for &cl in assignments {
        let hand_of: Vec<u8> = (0..chart.keys).map(|c| if c < half || (cl && c == half) { 0 } else { 1 }).collect();
        let mut selected: Vec<Plan> = Vec::new();
        js.push(jumptrills(&notes, &rows, &hand_of, window, if with_plans { Some(&mut selected) } else { None }));
        if with_plans {
            plans_out.push((cl, selected));
        }
    }
    let point: [Vec<f64>; 5] = if js.len() == 2 {
        std::array::from_fn(|k| js[0][k].iter().zip(&js[1][k]).map(|(l, r)| (l + r) / 2.0).collect())
    } else {
        js.pop().unwrap()
    };
    let vib = vibro(&notes, &rows);
    let zeros = vec![0.0f64; nn];
    // per-note values in SERIES order
    let per_note: [&Vec<f64>; 13] = [
        &point[0], &point[1], &point[2], &point[3], &vib[0], &vib[1], &vib[2], &vib[3], &point[4], &point[2],
        &point[3], &zeros, &vib[4],
    ];
    let mut counts = vec![0usize; bins];
    let note_bins: Vec<usize> = notes.t.iter().map(|&t| (bins - 1).min(((t - first) / bin_seconds + 1e-9) as usize)).collect();
    for &b in &note_bins {
        counts[b] += 1;
    }
    let mut series: [Vec<f64>; 13] = std::array::from_fn(|_| vec![0.0; bins]);
    for (k, values) in series.iter_mut().enumerate() {
        for (&b, &v) in note_bins.iter().zip(per_note[k].iter()) {
            values[b] += v;
        }
        for (v, &c) in values.iter_mut().zip(&counts) {
            *v /= 1usize.max(c) as f64;
        }
    }
    let mut state = 0.0f64;
    for b in 0..bins {
        let value = series[s::JUMPTRILL][b];
        let decay = (-bin_seconds / if value > state { 15.0 } else { 4.0 }).exp();
        state = value + (state - value) * decay;
        series[s::JUMPTRILL_SUSTAIN][b] = value * state;
    }
    let weights: Vec<usize> = counts.iter().map(|&c| c * c).collect();
    let denominator = 1usize.max(weights.iter().sum()) as f64;
    let feat_idx = [s::VIBRO_EASY, s::VIBRO_CONTROL, s::VIBRO_SUSTAIN, s::JUMPTRILL_EASY, s::ROLL_CONTROL, s::ROLL_ANCHOR, s::JUMPTRILL_SUSTAIN, s::VIBRO_LOCKED];
    let features: [f64; 8] =
        std::array::from_fn(|f| py::sum(series[feat_idx[f]].iter().zip(&weights).map(|(v, &w)| v * w as f64)) / denominator);
    Analysis { series, features, plans: if with_plans { Some(plans_out) } else { None }, notes, rows }
}
