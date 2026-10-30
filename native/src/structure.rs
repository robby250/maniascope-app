//! Chart-only geometry descriptors (chart_structure.py), numpy semantics within 1e-9.
use crate::chart::Chart;
use crate::params::Knobs;
use crate::py;
use crate::val::{Dict, Val};
use sha2::{Digest, Sha256};

pub const STRUCTURE_VERSION: i64 = 2;

/// numpy pairwise summation (np.add.reduce on a contiguous float64 array).
pub fn np_sum(a: &[f64]) -> f64 {
    const BLOCK: usize = 128;
    let n = a.len();
    if n < 8 {
        let mut res = -0.0f64;
        for &x in a {
            res += x;
        }
        return res;
    }
    if n <= BLOCK {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        while i < n - (n % 8) {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        return res;
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    np_sum(&a[..n2]) + np_sum(&a[n2..])
}

pub fn np_mean(a: &[f64]) -> f64 {
    np_sum(a) / a.len() as f64
}

pub fn np_std(a: &[f64]) -> f64 {
    let m = np_mean(a);
    let sq: Vec<f64> = a.iter().map(|x| (x - m) * (x - m)).collect();
    (np_sum(&sq) / a.len() as f64).sqrt()
}

/// np.percentile(a, q), method 'linear'.
pub fn np_percentile(a: &[f64], q: f64) -> f64 {
    let mut s = a.to_vec();
    s.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let n = s.len();
    let virt = (n - 1) as f64 * (q / 100.0);
    let prev = virt.floor();
    let pi = (prev as usize).min(n - 1);
    let ni = (pi + 1).min(n - 1);
    let t = virt - prev;
    let (a0, b0) = (s[pi], s[ni]);
    let d = b0 - a0;
    if t >= 0.5 {
        b0 - d * (1.0 - t)
    } else {
        a0 + d * t
    }
}

pub fn structure(chart: &Chart, rate: f64, k: &Knobs) -> Dict {
    let mut d = Dict::new();
    d.set("version", STRUCTURE_VERSION);
    let n = chart.len();
    if n < 2 {
        d.set("nps", Val::None);
        return d;
    }
    let heads = &chart.t;
    let span = heads[n - 1] - heads[0];
    let max_end = chart.end.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let play_span = max_end - heads[0];
    if span <= 0.0 {
        d.set("nps", Val::None);
        return d;
    }
    let mut row_times: Vec<f64> = Vec::new();
    let mut masks: Vec<u32> = Vec::new();
    let mut cols: Vec<Vec<f64>> = vec![Vec::new(); chart.keys];
    for i in 0..n {
        let t = heads[i];
        // defaultdict keyed by time: notes are time-sorted, so a new key is a new last row
        match row_times.last() {
            Some(&x) if x == t => *masks.last_mut().unwrap() |= 1 << chart.col[i],
            _ => {
                row_times.push(t);
                masks.push(1 << chart.col[i]);
            }
        }
        cols[chart.col[i] as usize].push(t);
    }
    let sizes: Vec<f64> = masks.iter().map(|m| m.count_ones() as f64).collect();
    let overlap: Vec<f64> = masks
        .windows(2)
        .map(|w| (w[0] & w[1]).count_ones() as f64 / w[0].count_ones().max(w[1].count_ones()) as f64)
        .collect();
    let repeated = masks.windows(2).filter(|w| w[0] == w[1]).count() as f64 / 1usize.max(masks.len() - 1) as f64;
    let mut fast_count = 0usize;
    let mut longest = 0.0f64;
    for times in &cols {
        let mut start = 0usize;
        for i in 1..=times.len() {
            if i < times.len() && times[i] - times[i - 1] < 1000.0 * k.VIBRO_GAP * rate {
                continue;
            }
            if i - start >= k.VIBRO_MIN {
                fast_count += i - start;
                longest = longest.max((times[i - 1] - times[start]) / 1000.0 / rate);
            }
            start = i;
        }
    }
    // Layout plus coarse normalized phrase timing (variety only, never accuracy merging).
    let num = 64usize.min(n);
    let step = (n - 1) as f64 / (num - 1) as f64;
    let mut sample: Vec<usize> = (0..num).map(|i| (i as f64 * step + 0.0) as usize).collect();
    *sample.last_mut().unwrap() = n - 1;
    let mut h = Sha256::new();
    h.update([chart.keys as u8]);
    h.update(&chart.col);
    h.update(chart.t.iter().zip(&chart.end).map(|(t, e)| (e > t) as u8).collect::<Vec<u8>>());
    for &i in &sample {
        let v = py::round((heads[i] - heads[0]) / span * 200.0) as i16;
        h.update(v.to_le_bytes());
    }
    let family: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect::<String>()[..24].to_string();
    let idx: Vec<usize> = heads.iter().map(|&x| ((x - heads[0]) / 5000.0) as usize).collect();
    let nb = idx.iter().copied().max().unwrap() + 1;
    let mut bin_counts = vec![0.0f64; nb];
    for &i in &idx {
        bin_counts[i] += 1.0;
    }
    for b in &mut bin_counts {
        *b /= 5.0;
    }
    let mean = n as f64 / (span / 1000.0);
    let cycles = [1usize, 2, 4, 8]
        .iter()
        .map(|&lag| {
            let c = (lag..masks.len()).filter(|&i| masks[i] == masks[i - lag]).count();
            c as f64 / 1usize.max(masks.len().saturating_sub(lag)) as f64
        })
        .fold(None, |m: Option<f64>, x| Some(match m { Some(m) if m >= x => m, _ => x }))
        .unwrap_or(0.0);
    let flat = 0f64.max(1.0 - np_std(&bin_counts) / 1f64.max(mean) / 0.35);
    let size_mean = np_mean(&sizes);
    let wide = 1f64.min(0f64.max(size_mean - 1.5) / 1.5);
    let mut drill = 0f64.max((cycles - 0.65) / 0.35) * flat * wide;
    let ge_keys: Vec<f64> = sizes.iter().map(|&s| (s >= chart.keys as f64) as u8 as f64).collect();
    let all_keys = np_mean(&ge_keys);
    drill = drill.max(0f64.max((all_keys - 0.45) / 0.55) * flat);
    let ge3: Vec<f64> = sizes.iter().map(|&s| (s >= 3.0) as u8 as f64).collect();
    let active_v: Vec<f64> = bin_counts.iter().map(|&b| (b > 0.4 * mean) as u8 as f64).collect();
    let rhythm = if row_times.len() > 1 {
        let diff: Vec<f64> = row_times.windows(2).map(|w| w[1] - w[0]).collect();
        np_std(&diff) / 1f64.max(np_mean(&diff))
    } else {
        0.0
    };
    d.set("nps", mean * rate);
    d.set("head_span", span / 1000.0);
    d.set("play_span", play_span / 1000.0);
    d.set("family", family);
    d.set("chord", size_mean);
    d.set("wide", np_mean(&ge3));
    d.set("repeat", repeated);
    d.set("overlap", if overlap.is_empty() { 0.0 } else { np_mean(&overlap) });
    d.set("burst", np_percentile(&bin_counts, 95.0) / mean.max(1e-6));
    d.set("active", np_mean(&active_v));
    d.set("rhythm", rhythm);
    d.set("vibro_share", fast_count as f64 / n as f64);
    d.set("vibro_longest", longest);
    d.set("vibro_focused", fast_count as f64 / n as f64 > 0.6 && repeated > 0.65 && longest > 3.0);
    d.set("periodicity", cycles);
    d.set("all_keys", all_keys);
    d.set("drill", 1f64.min(drill));
    d
}
