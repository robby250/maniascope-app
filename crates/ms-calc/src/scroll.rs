//! Scroll-reading geometry in played time (scroll_reading.py).
use crate::chart::Chart;
use crate::params::Params;
use crate::py;
use std::collections::HashMap;

pub const FEATURES: [&str; 4] = ["preview", "crowding", "motion", "stutter"];
pub const DEFAULT_WEIGHTS: [f64; 4] = [0.8, 0.7, 0.8, 1.6];
const VISIBILITY_MS: f64 = 450.0;

pub struct Geometry {
    pub vectors: Vec<[f64; 4]>,
    pub kinds: Vec<[f64; 5]>,
    pub summary: [f64; 4],
}

#[inline]
fn bisect_right(a: &[f64], x: f64) -> usize {
    a.partition_point(|&v| v <= x)
}
#[inline]
fn bisect_left(a: &[f64], x: f64) -> usize {
    a.partition_point(|&v| v < x)
}

pub fn geometry(chart: &Chart, rate: f64) -> Geometry {
    let n = chart.len();
    let empty = || Geometry { vectors: vec![[0.0; 4]; n], kinds: vec![[0.0; 5]; n], summary: [0.0; 4] };
    if chart.sv.is_empty() || n == 0 {
        return empty();
    }
    let mut ts: Vec<f64> = Vec::new();
    let mut velocities: Vec<f64> = Vec::new();
    for &(t, v) in &chart.sv {
        if velocities.last() != Some(&v) {
            ts.push(t);
            velocities.push(v);
        }
    }
    let start = chart.t[0];
    let end = chart.last_end();
    let vmax = velocities.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let vmin = velocities.iter().cloned().fold(f64::INFINITY, f64::min);
    if vmax - vmin < 1e-6 || end <= start {
        return empty();
    }
    let mut raw_positions = vec![0.0f64];
    for i in 1..ts.len() {
        let last = *raw_positions.last().unwrap();
        raw_positions.push(last + (ts[i] - ts[i - 1]) * velocities[i - 1]);
    }
    let raw_pos = |t: f64| {
        let i = bisect_right(&ts, t).saturating_sub(1);
        raw_positions[i] + (t - ts[i]) * velocities[i]
    };
    let sample_count = 2048usize.min(64usize.max(n));
    let half_approach = VISIBILITY_MS * rate / 2.0;
    let mut averages: Vec<f64> = (0..sample_count)
        .map(|i| {
            let t = start + (end - start) * (i as f64 + 0.5) / sample_count as f64;
            let (left, right) = (start.max(t - half_approach), end.min(t + half_approach));
            (raw_pos(right) - raw_pos(left)) / (right - left)
        })
        .collect();
    let reference = 1e-6f64.max(py::median(&mut averages));
    let vs: Vec<f64> = velocities.iter().map(|&v| 1e-6f64.max(v / reference)).collect();
    let xs: Vec<f64> = raw_positions.iter().map(|&p| p / reference).collect();
    let pos = |t: f64| {
        let i = bisect_right(&ts, t).saturating_sub(1);
        xs[i] + (t - ts[i]) * vs[i]
    };
    let at = |position: f64| {
        let i = bisect_right(&xs, position).saturating_sub(1);
        ts[i] + (position - xs[i]) / vs[i]
    };
    let nominal = VISIBILITY_MS * rate;
    let positions: Vec<f64> = chart.t.iter().map(|&t| pos(t)).collect();
    let mut neighbours: Vec<[usize; 2]> = vec![[usize::MAX; 2]; n]; // [previous, next] in column
    let mut previous: Vec<Option<usize>> = vec![None; chart.keys];
    for i in 0..n {
        let c = chart.col[i] as usize;
        if let Some(pi) = previous[c] {
            if chart.t[i] > chart.t[pi] {
                neighbours[i][0] = pi;
                neighbours[pi][1] = i;
            }
        }
        previous[c] = Some(i);
    }
    let mut by_time: HashMap<u64, (f64, f64, f64, bool, f64)> = HashMap::new();
    let mut history: std::collections::VecDeque<(f64, [f64; 4])> = std::collections::VecDeque::with_capacity(12);
    let mut vectors = Vec::with_capacity(n);
    let mut kinds = Vec::with_capacity(n);
    for i in 0..n {
        let t = chart.t[i];
        let ti = (t + 0.0).to_bits();
        let current = match by_time.get(&ti) {
            Some(&c) => c,
            None => {
                let p = positions[i];
                let entry = at(p - nominal);
                let seen = 0.001f64.max(t - entry);
                let preview = 0f64.max(1.0 - seen / nominal);
                let look = seen.min(2.0 * nominal);
                let a = t - look;
                let distance = 1e-6f64.max(p - pos(a));
                let mean = distance / look;
                let (lo, hi) = (bisect_right(&ts, a), bisect_left(&ts, t));
                let fingerprint: [f64; 4];
                let cur;
                if lo >= hi {
                    fingerprint = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1f64.min(seen / nominal)];
                    cur = (preview, 0.0, 0.0, false, 1f64.min(nominal / seen));
                } else {
                    let mut boundaries = Vec::with_capacity(hi - lo + 2);
                    boundaries.push(a);
                    boundaries.extend_from_slice(&ts[lo..hi]);
                    boundaries.push(t);
                    let mut deviation = 0.0f64;
                    let mut turns: Vec<f64> = Vec::new();
                    let mut last_direction = 0i32;
                    let mut previous_v: Option<f64> = None;
                    for w in boundaries.windows(2) {
                        let (left, right) = (w[0], w[1]);
                        let v = vs[bisect_right(&ts, (left + right) * 0.5).saturating_sub(1)];
                        deviation += (v - mean).abs() * (right - left);
                        let direction = match previous_v {
                            Some(pv) if pv != 0.0 => {
                                if v > pv * 1.05 {
                                    1
                                } else if v < pv / 1.05 {
                                    -1
                                } else {
                                    0
                                }
                            }
                            _ => 0,
                        };
                        if direction != 0 && last_direction != 0 && direction != last_direction {
                            turns.push(left);
                        }
                        if direction != 0 {
                            last_direction = direction;
                        }
                        previous_v = Some(v);
                    }
                    let visible_fraction = 1f64.min(distance / nominal);
                    let total_variation = 2f64.min(deviation / distance) * visible_fraction;
                    let thirds: [f64; 3] = std::array::from_fn(|j| {
                        0f64.max(pos(a + (j + 1) as f64 * look / 3.0) - pos(a + j as f64 * look / 3.0)) / distance
                    });
                    let mut broad =
                        total_variation.min(py::sum(thirds.iter().map(|&v| (v - 1.0 / 3.0).abs())) * visible_fraction);
                    let fast_cycle = turns.len() > 2 && turns.windows(3).any(|w| w[2] - w[0] < nominal / 3.0);
                    let mut fine = if fast_cycle { 0f64.max(total_variation - broad) } else { 0.0 };
                    if fine != 0.0 {
                        let edges: [f64; 4] = std::array::from_fn(|j| a + j as f64 * look / 3.0);
                        let edge_positions: [f64; 4] = std::array::from_fn(|j| pos(edges[j]));
                        let mut peak = 0.0f64;
                        for &point in &boundaries[1..boundaries.len() - 1] {
                            let j = 2usize.min(((point - a) * 3.0 / look) as usize);
                            let expected = edge_positions[j]
                                + (edge_positions[j + 1] - edge_positions[j]) * (point - edges[j]) / (look / 3.0);
                            peak = peak.max((pos(point) - expected).abs());
                        }
                        fine = fine.min(6.0 * peak / nominal);
                    } else if !fast_cycle {
                        broad = total_variation;
                    }
                    fingerprint = [thirds[0], thirds[1], thirds[2], 1f64.min(seen / nominal)];
                    let recent: Vec<&[f64; 4]> =
                        history.iter().filter(|(when, _)| t - when <= 1500.0 * rate).map(|(_, v)| v).collect();
                    let novelty = if recent.iter().any(|v| **v == fingerprint) {
                        0.0
                    } else {
                        recent
                            .iter()
                            .map(|old| py::sum(fingerprint.iter().zip(old.iter()).map(|(x, y)| (x - y).abs())))
                            .fold(None, |m: Option<f64>, x| Some(match m {
                                Some(m) if m <= x => m,
                                _ => x,
                            }))
                            .unwrap_or(1.0)
                    };
                    let novelty = 1f64.min(novelty / 0.5);
                    let motion = broad * (0.4 + 0.6 * novelty);
                    let accel = thirds[2] > thirds[0];
                    cur = (preview, motion, fine, accel, 1f64.min(nominal / seen));
                }
                by_time.insert(ti, cur);
                if history.len() == 12 {
                    history.pop_front();
                }
                history.push_back((t, fingerprint));
                cur
            }
        };
        let (preview, motion, fine, accel, preview_pressure) = current;
        let mut crowd = 0.0f64;
        for &j in &neighbours[i] {
            if j == usize::MAX {
                continue;
            }
            let normal = (chart.t[j] - t).abs() / nominal;
            let actual = (positions[j] - positions[i]).abs() / nominal;
            let extra = 0f64.max(1.0 - actual / 0.10) - 0f64.max(1.0 - normal / 0.10);
            crowd = crowd.max(0f64.max(extra));
        }
        crowd *= preview_pressure;
        vectors.push([preview, crowd, motion, fine]);
        kinds.push([preview, crowd, if accel { motion } else { 0.0 }, fine, if !accel { motion } else { 0.0 }]);
    }
    let summary: [f64; 4] = std::array::from_fn(|j| py::sum(vectors.iter().map(|v| v[j])) / n as f64);
    Geometry { vectors, kinds, summary }
}

/// Per-note reading strain, SV kind shares, and the diagnostic summary (scroll_reading.reading).
pub fn reading(chart: &Chart, rate: f64, prm: &Params) -> Result<(Vec<f64>, Vec<[f64; 5]>, [f64; 4]), String> {
    let info = geometry(chart, rate);
    let cfg = &prm.model["scroll_reading"];
    let mode = &cfg["modes"][chart.keys.to_string()];
    let cfg = if mode.is_null() { cfg } else { mode };
    let weights: [f64; 4] = match cfg.get("weights").and_then(|w| w.as_array()) {
        None => DEFAULT_WEIGHTS,
        Some(w) => {
            let ws: Vec<f64> = w.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect();
            if ws.len() != 4 || ws.iter().any(|w| !w.is_finite() || *w < 0.0) {
                return Err("Invalid scroll-reading coefficients".into());
            }
            [ws[0], ws[1], ws[2], ws[3]]
        }
    };
    let n = info.vectors.len();
    let mut out = Vec::with_capacity(n);
    let mut kinds = Vec::with_capacity(n);
    for (vector, kind) in info.vectors.iter().zip(&info.kinds) {
        let values: [f64; 4] = std::array::from_fn(|j| vector[j] * weights[j]);
        let total = py::sum(values);
        out.push(total);
        let ks = [values[0], values[1], if kind[2] != 0.0 { values[2] } else { 0.0 }, values[3], if kind[4] != 0.0 { values[2] } else { 0.0 }];
        kinds.push(if total > 0.0 { ks.map(|v| v / total) } else { [0.0; 5] });
    }
    Ok((out, kinds, info.summary))
}
