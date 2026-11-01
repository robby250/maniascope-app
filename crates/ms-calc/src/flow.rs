//! Per-hand flow and pattern change (user 2026-11-01: "on 10k per hand patterns and how good they flow matterns A LOT more than on
//! 7k", "mental and physical endurance are different"). Descriptors only: what they weigh is fitted per keymode.
use crate::chart::Chart;

const LINK: f64 = 0.25;     // s: a hand's next note further than this starts a new phrase
const FLOOR: f64 = 0.04;    // s: faster than this counts the same
const MEMORY: usize = 32;   // rows: a pair of rows seen this recently is no new pattern
const FAST_LN: f64 = 0.05;  // s: LN heads this close on adjacent columns roll

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Flow {
    pub broken: f64,  // speed-weighted share of a hand's notes out of finger order: not a roll, not a trill
    pub skip: f64,    // speed-weighted fingers jumped over per note
    pub change: f64,  // share of rows making a pair the last MEMORY rows didn't have
    pub fast_ln: f64, // share of LN heads rolling onto a neighbour
}

impl Flow {
    /// Kept with the chart's other execution numbers: the next calibration needn't read the file again.
    pub fn store(&self, execution: &mut crate::val::Dict) {
        for (name, v) in [("flow_broken", self.broken), ("flow_skip", self.skip), ("flow_change", self.change), ("fast_ln", self.fast_ln)] {
            execution.set(name, v);
        }
    }
}

pub const NAMES: [&str; 4] = ["flow_broken", "flow_skip", "mental", "fast_ln"];
pub const DIRECT: usize = crate::model::SR_NAMES_LEN + NAMES.len();

/// Under about 5–7 stars processing is the limit, on physically hard maps the arms are (user 2026-11-01).
fn mental_weight(stars: f64) -> f64 {
    ((9. - stars) / 3.).clamp(0., 1.)
}

/// The joint stage's inputs: the residual stage's, the new descriptors, then each over log rate and log stars. `raw` is the demand
/// before any fitted stage, so taking a stage out doesn't move the inputs.
pub fn joint_row(residual: &[f64], length_ms: f64, raw: f64, rate: f64, f: Flow) -> Vec<f64> {
    let mut v = residual.to_vec();
    v.resize(crate::model::SR_NAMES_LEN, 0.);
    let stars = raw.max(0.05);
    v.extend([f.broken, f.skip, f.change * (1. - (-length_ms / 1000. / rate / 120.).exp()) * mental_weight(stars), f.fast_ln]);
    let (r, l) = (rate.ln(), stars.ln() - 1.8);
    v.extend((0..DIRECT).map(|i| v[i] * r).collect::<Vec<_>>());
    v.extend((0..DIRECT).map(|i| v[i] * l).collect::<Vec<_>>());
    v
}

/// Which hand plays a column and where on it; the middle key of an odd keymode is a thumb, outside both hands' order.
fn hand(col: usize, keys: usize) -> Option<(usize, f64)> {
    if keys % 2 == 1 && col == keys / 2 {
        return None;
    }
    Some(if col < keys / 2 { (0, col as f64) } else { (1, col as f64) })
}

pub fn flow(chart: &Chart, rate: f64) -> Flow {
    let scale = 0.001 / rate;
    let (mut rows, mut hands): (Vec<(f64, u32)>, [Vec<(f64, f64)>; 2]) = Default::default();
    let mut i = 0;
    while i < chart.len() {
        let t = chart.t[i];
        let (mut mask, mut sum, mut count) = (0u32, [0.; 2], [0.; 2]);
        while i < chart.len() && chart.t[i] == t {
            mask |= 1 << chart.col[i];
            if let Some((h, at)) = hand(chart.col[i] as usize, chart.keys) {
                sum[h] += at;
                count[h] += 1.;
            }
            i += 1;
        }
        rows.push((t * scale, mask));
        for h in 0..2 {
            if count[h] > 0. {
                hands[h].push((t * scale, sum[h] / count[h]));
            }
        }
    }
    let (mut total, mut broken, mut skip) = (0., 0., 0.);
    for notes in &hands {
        for w in notes.windows(3) {
            let (gap1, gap2) = (w[1].0 - w[0].0, w[2].0 - w[1].0);
            let (s1, s2) = (w[1].1 - w[0].1, w[2].1 - w[1].1);
            if gap1 > LINK || gap2 > LINK || s1 == 0. || s2 == 0. {
                continue; // a jack has no order; its cost is elsewhere
            }
            let weight = 1. / gap2.max(FLOOR);
            total += weight;
            skip += weight * (s2.abs() - 1.).max(0.);
            if s1 * s2 < 0. && (w[2].1 - w[0].1).abs() > 1e-9 {
                broken += weight;
            }
        }
    }
    let share = |v: f64| if total > 0. { v / total } else { 0. };
    let (mut changed, mut seen) = (0usize, Vec::with_capacity(MEMORY));
    for pair in rows.windows(2) {
        let p = (pair[0].1, pair[1].1);
        changed += usize::from(!seen.contains(&p));
        if seen.len() == MEMORY {
            seen.remove(0);
        }
        seen.push(p);
    }
    let heads: Vec<(f64, u8)> = (0..chart.len()).filter(|&n| chart.end[n] > chart.t[n]).map(|n| (chart.t[n] * scale, chart.col[n])).collect();
    let rolled = heads.windows(2).filter(|w| w[1].0 - w[0].0 <= FAST_LN && w[1].0 > w[0].0 && w[1].1.abs_diff(w[0].1) == 1).count();
    Flow { broken: share(broken), skip: share(skip), change: changed as f64 / rows.len().max(2).saturating_sub(1) as f64, fast_ln: rolled as f64 / heads.len().max(1) as f64 }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chart(keys: usize, cols: &[u8], step: f64) -> Chart {
        let t: Vec<f64> = (0..cols.len()).map(|i| i as f64 * step).collect();
        Chart { keys, end: t.clone(), t, col: cols.to_vec(), ..Chart::default() }
    }

    fn repeated(pattern: &[u8], n: usize) -> Vec<u8> {
        pattern.iter().copied().cycle().take(n).collect()
    }

    #[test]
    fn the_same_notes_out_of_order_read_broken() {
        let rolled = flow(&chart(10, &repeated(&[0, 1, 2, 3, 4], 400), 100.), 1.);
        let broken = flow(&chart(10, &repeated(&[0, 2, 1, 4, 3], 400), 100.), 1.);
        assert!(broken.broken > rolled.broken + 0.3 && broken.skip > rolled.skip, "{rolled:?} {broken:?}");
    }

    #[test]
    fn a_trill_flows() {
        assert_eq!(flow(&chart(7, &repeated(&[0, 1], 200), 100.), 1.).broken, 0.);
    }

    #[test]
    fn two_columns_a_hand_have_no_order_to_break() {
        assert_eq!(flow(&chart(4, &repeated(&[0, 1, 0, 0, 1, 1], 300), 100.), 1.).broken, 0.);
    }

    #[test]
    fn repeating_one_pattern_stops_being_new() {
        let same = flow(&chart(7, &repeated(&[0, 1, 2], 600), 100.), 1.);
        let cols: Vec<u8> = (0..600u32).scan(1u32, |x, _| { *x = x.wrapping_mul(1664525).wrapping_add(1013904223); Some((*x >> 24) as u8 % 7) }).collect();
        assert!(same.change < 0.02 && flow(&chart(7, &cols, 100.), 1.).change > same.change + 0.1);
    }

    #[test]
    fn long_notes_rolling_onto_a_neighbour_are_counted() {
        let mut c = chart(7, &repeated(&[0, 1, 2], 90), 30.);
        c.end = c.t.iter().map(|t| t + 200.).collect();
        assert!(flow(&c, 1.).fast_ln > 0.5);
        assert_eq!(flow(&chart(7, &repeated(&[0, 1, 2], 90), 30.), 1.).fast_ln, 0.);
    }
}
