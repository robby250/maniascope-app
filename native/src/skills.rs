//! Skill keys and the per-skill series (skill_calc.skills / _attribute).
use crate::demand::{Parts, P, PARTS_4K, PARTS_5K};
use crate::params::Knobs;
use crate::py;

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum Sk {
    stream, jumpstream, handstream, dump, jumptrill, splittrill, jackspeed, chordjack, ln, sv, delay,
    chordstream, bracket, jack, quadstream, minijack, longjack, vibro, release, inverse, hybrid, shield,
    sv_fast, sv_slow, sv_accel, sv_stutter, sv_brake, trill1h, anchor, patterntech, rhythmtech, technical,
    mash, stamina, overall, switch,
}
pub const NSK: usize = 36;
pub const SK_NAMES: [&str; NSK] = [
    "stream", "jumpstream", "handstream", "dump", "jumptrill", "splittrill", "jackspeed", "chordjack", "ln", "sv",
    "delay", "chordstream", "bracket", "jack", "quadstream", "minijack", "longjack", "vibro", "release", "inverse",
    "hybrid", "shield", "sv_fast", "sv_slow", "sv_accel", "sv_stutter", "sv_brake", "trill1h", "anchor",
    "patterntech", "rhythmtech", "technical", "mash", "stamina", "overall", "_switch",
];
pub const ALL_SK: [Sk; NSK] = {
    use Sk::*;
    [
        stream, jumpstream, handstream, dump, jumptrill, splittrill, jackspeed, chordjack, ln, sv, delay,
        chordstream, bracket, jack, quadstream, minijack, longjack, vibro, release, inverse, hybrid, shield,
        sv_fast, sv_slow, sv_accel, sv_stutter, sv_brake, trill1h, anchor, patterntech, rhythmtech, technical,
        mash, stamina, overall, switch,
    ]
};
impl Sk {
    pub fn name(self) -> &'static str {
        SK_NAMES[self as usize]
    }
    pub fn from_name(s: &str) -> Option<Sk> {
        SK_NAMES.iter().position(|&n| n == s).map(|i| ALL_SK[i])
    }
}

/// The skill with the same key as a demand part.
pub fn sk_of(p: P) -> Option<Sk> {
    Sk::from_name(crate::demand::P_NAMES[p as usize])
}

pub const JACK_PARTS: [Sk; 3] = [Sk::chordjack, Sk::jack, Sk::jackspeed];
pub const LN_KINDS: [Sk; 4] = [Sk::release, Sk::inverse, Sk::hybrid, Sk::shield];
pub const SV_KINDS: [Sk; 5] = [Sk::sv_fast, Sk::sv_slow, Sk::sv_accel, Sk::sv_stutter, Sk::sv_brake];
pub const TECH_KEYS: [Sk; 3] = [Sk::patterntech, Sk::rhythmtech, Sk::technical];

/// Skill keys measured for a keymode, in display order for ties.
pub fn skills(keys: usize) -> Vec<Sk> {
    use Sk::*;
    let base: Vec<Sk> = if keys < 5 {
        vec![stream, jumpstream, handstream, dump, jumptrill, splittrill, jackspeed, chordjack, ln, sv]
    } else {
        vec![delay, stream, chordstream, bracket, jumptrill, jack, chordjack, ln, sv]
    };
    let i = base.iter().position(|&s| s == ln).unwrap();
    let mut out = base[..i].to_vec();
    if keys == 4 {
        out.push(quadstream);
    }
    out.extend([minijack, longjack, vibro, ln]);
    out.extend(LN_KINDS);
    out.extend_from_slice(&base[i + 1..]);
    out.extend(SV_KINDS);
    out.extend([trill1h, anchor, patterntech, rhythmtech, technical, mash, stamina]);
    out
}

/// Ordered per-skill series (a Python dict of lists).
#[derive(Clone, Debug)]
pub struct Series {
    pub nbins: usize,
    pub keys: Vec<Sk>,
    pub data: Vec<f64>, // [NSK × nbins], rows of absent keys stay zero
    /// A key's list may be shorter than nbins (Python zip() truncation in execution.attribute).
    pub lens: [usize; NSK],
}
impl Series {
    pub fn new(nbins: usize, keys: Vec<Sk>) -> Series {
        Series { nbins, keys, data: vec![0.0; NSK * nbins], lens: [nbins; NSK] }
    }
    #[inline]
    pub fn get(&self, s: Sk) -> &[f64] {
        let o = s as usize * self.nbins;
        &self.data[o..o + self.lens[s as usize]]
    }
    #[inline]
    pub fn get_mut(&mut self, s: Sk) -> &mut [f64] {
        let o = s as usize * self.nbins;
        &mut self.data[o..o + self.lens[s as usize]]
    }
    /// Replace a key's list (len ≤ nbins).
    pub fn set(&mut self, s: Sk, v: &[f64]) {
        let o = s as usize * self.nbins;
        assert!(v.len() <= self.nbins);
        self.data[o..o + v.len()].copy_from_slice(v);
        self.data[o + v.len()..o + self.nbins].iter_mut().for_each(|x| *x = 0.0);
        self.lens[s as usize] = v.len();
    }
    pub fn has(&self, s: Sk) -> bool {
        self.keys.contains(&s)
    }
    pub fn remove(&mut self, s: Sk) -> Vec<f64> {
        self.keys.retain(|&k| k != s);
        let v = self.get(s).to_vec();
        self.get_mut(s).iter_mut().for_each(|x| *x = 0.0);
        v
    }
}

/// ±h bins sums.
pub fn win(v: &[f64], h: usize) -> Vec<f64> {
    let nb = v.len();
    let mut out = vec![0.0; nb];
    let mut acc = 0.0f64;
    for i in 0..nb + h {
        if i < nb {
            acc += v[i];
        }
        if i >= 2 * h + 1 {
            acc -= v[i - 2 * h - 1];
        }
        if i >= h {
            out[i - h] = acc;
        }
    }
    out
}

/// Per-skill demand series (skill_calc._attribute).
pub fn attribute(parts: &Parts, total: &[f64], keys: usize, technical_only: bool, k: &Knobs) -> Series {
    let nb = total.len();
    let pk: &[P] = if keys < 5 { &PARTS_4K } else { &PARTS_5K };
    let w = |p: P| win(parts.get(p), 2);
    let pw: Vec<Vec<f64>> = pk.iter().map(|&p| w(p)).collect();
    let p_tech = w(P::technical);
    let p_rows = w(P::rows);
    let rodd = win(parts.get(P::rodd), k.RT_WIN);
    let rrows = win(parts.get(P::rows), k.RT_WIN);
    let pp: Vec<f64> = parts.get(P::pins).iter().zip(parts.get(P::psw)).map(|(a, c)| a + c).collect();
    let podd = win(&pp, k.RT_WIN);
    let psw = win(parts.get(P::psw), k.RT_WIN);
    let nosv: Vec<f64> = total.iter().zip(parts.get(P::sv_raw)).map(|(t, s)| 0f64.max(t - s / k.BIN)).collect();
    let mut names: Vec<Sk> =
        if technical_only { vec![Sk::technical, Sk::patterntech, Sk::rhythmtech] } else { skills(keys) };
    names.push(Sk::switch);
    names.retain(|&s| s != Sk::stamina);
    let mut se = Series::new(nb, names);
    let extra = |p: P| if technical_only { Vec::new() } else { w(p) };
    let (mj, lj, vib, shield, inv, hyb, oht, anc, sus, ln_rel, dump, bracket) = (
        extra(P::mj), extra(P::lj), extra(P::vib), extra(P::shield), extra(P::inv), extra(P::hyb), extra(P::oht),
        extra(P::anc), extra(P::sus), extra(P::ln_rel), extra(P::dump), extra(P::bracket),
    );
    let ln_i = pk.iter().position(|&p| p == P::ln).unwrap();
    let pk_sk: Vec<Sk> = pk.iter().map(|&p| sk_of(p).unwrap()).collect();
    let has_dump = se.has(Sk::dump);
    let has_bracket = se.has(Sk::bracket);
    let nbn = se.nbins;
    let at = |s: Sk, b: usize| s as usize * nbn + b;
    for b in 0..nb {
        let tot = py::sum(pw.iter().map(|x| x[b])) + p_tech[b];
        if tot == 0.0 || total[b] == 0.0 {
            continue;
        }
        if !technical_only {
            let top = pw.iter().map(|x| x[b]).fold(f64::NEG_INFINITY, f64::max);
            for (x, &s) in pw.iter().zip(&pk_sk) {
                se.data[at(s, b)] = total[b] * x[b] / top;
            }
        }
        let rows = p_rows[b];
        let mut rt = 0.0;
        let mut pt = 0.0;
        if rows != 0.0 {
            if rrows[b] > 0.5 {
                rt = 1f64.min(0f64.max(rodd[b]) / (k.RT_FULL * rrows[b]));
                pt = 1f64.min(0f64.max(podd[b]) / (k.PT_FULL * rrows[b]));
            }
            se.data[at(Sk::rhythmtech, b)] = nosv[b] * rt;
            se.data[at(Sk::patterntech, b)] = nosv[b] * pt;
            se.data[at(Sk::switch, b)] =
                if rrows[b] > 0.5 { nosv[b] * 1f64.min(0f64.max(psw[b]) / (k.PT_FULL * rrows[b])) } else { 0.0 };
            se.data[at(Sk::technical, b)] = nosv[b] * (1.0 - (1.0 - rt) * (1.0 - pt));
        }
        if technical_only {
            continue;
        }
        let tb = total[b];
        if rows != 0.0 {
            se.data[at(Sk::minijack, b)] = tb * 1f64.min(mj[b] / (k.MJ_FULL * rows));
            se.data[at(Sk::longjack, b)] = tb * 1f64.min(lj[b] / (k.LJ_FULL * rows));
            se.data[at(Sk::vibro, b)] = tb * 1f64.min(vib[b] / (k.VIB_FULL * rows));
            se.data[at(Sk::shield, b)] = tb * 1f64.min(shield[b] / (k.SHIELD_FULL * rows));
            se.data[at(Sk::inverse, b)] = tb * 1f64.min(inv[b] / (k.INV_FULL * rows));
            se.data[at(Sk::hybrid, b)] = tb * 1f64.min(hyb[b] / (k.HYB_FULL * rows));
            se.data[at(Sk::trill1h, b)] = tb * 1f64.min(oht[b] / (k.OHT_FULL * rows));
            se.data[at(Sk::anchor, b)] = tb * 1f64.min(anc[b] / (k.ANC_FULL * rows));
        }
        let jr = mj[b] + sus[b];
        if jr != 0.0 {
            let burst = mj[b] / jr;
            for s in JACK_PARTS {
                if se.keys.contains(&s) {
                    let moved = se.data[at(s, b)] * burst;
                    se.data[at(s, b)] -= moved;
                    let m = &mut se.data[at(Sk::minijack, b)];
                    *m = m.max(moved);
                }
            }
        }
        if pw[ln_i][b] > 0.0 {
            se.data[at(Sk::release, b)] = se.data[at(Sk::ln, b)] * 1f64.min(ln_rel[b] / (k.REL_FULL * pw[ln_i][b]));
        }
        if has_dump {
            se.data[at(Sk::dump, b)] = tb * 1f64.min(dump[b] / (k.DUMP_FULL * tot));
        }
        if has_bracket {
            se.data[at(Sk::bracket, b)] = tb * 1f64.min(bracket[b] / (k.BRACKET_FULL * tot));
        }
    }
    se
}

pub fn star(level: f64, k: &Knobs) -> f64 {
    k.STAR_A * level.powf(k.STAR_B)
}

/// Hardest window of each horizon → required level (skill_calc._horizon_levels).
pub fn horizon_levels(series: &[f64], k: &Knobs) -> [f64; 4] {
    let mut out = [0.0; 4];
    for (o, &(w, tol)) in out.iter_mut().zip(&k.HORIZONS) {
        let n = 1usize.max((w / k.BIN) as usize).min(series.len());
        let mut acc = 0.0f64;
        let mut peak = 0.0f64;
        for (i, &d) in series.iter().enumerate() {
            let previous = if i >= n { series[i - n] } else { 0.0 };
            acc += d - previous;
            if acc > peak {
                peak = acc;
            }
        }
        *o = star(k.SCALE * (peak / n as f64).sqrt() / tol, k);
    }
    out
}

pub fn window_level(series: &[f64], w: f64, k: &Knobs) -> f64 {
    let n = 1usize.max((w / k.BIN) as usize).min(series.len());
    let mut acc = 0.0f64;
    let mut peak = 0.0f64;
    for (i, &d) in series.iter().enumerate() {
        acc += d - if i >= n { series[i - n] } else { 0.0 };
        if acc > peak {
            peak = acc;
        }
    }
    star(k.SCALE * (peak / n as f64).sqrt(), k)
}

pub fn combine(levels: &[f64; 4], notes: usize, k: &Knobs) -> f64 {
    let top = levels.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if top <= 0.0 {
        return 0.0;
    }
    let mut f = 1.0f64;
    for (&lv, &b) in levels.iter().zip(&k.H_WEIGHTS) {
        f *= (lv.max(1e-9) / top).powf(b);
    }
    top * f * ((notes.max(1) as f64) / 2000.0).powf(k.G_NOTES)
}

pub fn stamina(total: &[f64], k: &Knobs) -> f64 {
    let n = (k.STAMINA_WINDOW / k.BIN) as usize;
    let mut acc = 0.0f64;
    let mut peak = 0.0f64;
    for (i, &d) in total.iter().enumerate() {
        let previous = if i >= n { total[i - n] } else { 0.0 };
        acc += d - previous;
        peak = peak.max(acc);
    }
    star(k.SCALE * (peak / n as f64).sqrt() / k.HORIZONS[3].1, k)
}

/// Demand totals for a bin-grid offset (skill_calc._phase_total).
pub fn phase_total(fine: &[Vec<f64>; 3], p: usize, k: &Knobs) -> Vec<f64> {
    let phases = k.PHASES;
    let n = (fine[0].len() + p + phases - 1) / phases;
    let mut acc = vec![0.0f64; 3 * n];
    for side in 0..3 {
        for (i, &v) in fine[side].iter().enumerate() {
            if v != 0.0 {
                acc[side * n + (i + p) / phases] += v;
            }
        }
    }
    (0..n).map(|b| ((acc[b] * acc[b] + acc[n + b] * acc[n + b]).sqrt() + acc[2 * n + b]) / k.BIN).collect()
}
