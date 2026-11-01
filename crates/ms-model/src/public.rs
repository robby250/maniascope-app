//! The public bundle as the model reads it.
use crate::predict::{Consts, Feats, Pop, Slopes};
use ms_calc::val::{Dict, Val};
use ms_data::public::{self, Meta, Population, Public};
use std::collections::HashMap;
use std::path::Path;

/// warmup._rate_elasticities: rate response from the same chart at multiple analysed rates. Chart derivatives, not
/// player outcomes or SR targets; sparse families borrow their keymode response instead of a universal accuracy tax.
pub fn rate_elasticities(records: impl Iterator<Item = (i64, f64, Val)>, c: &Consts) -> Vec<(String, f64)> {
    let mut by: HashMap<String, Vec<f64>> = HashMap::new();
    let mut last: Option<(i64, f64, f64, Dict)> = None; // chart, rate, overall, features of the previous rate kept
    for (bid, rate, f) in records {
        let Val::Dict(f) = f else { continue };
        let overall = f.fget("overall", 0.);
        if rate <= 0. || overall < 2. {
            continue;
        }
        if let Some((b, left, a_overall, a)) = &last {
            if *b == bid {
                let derivative = (overall / a_overall).ln() / (rate / left).ln();
                if (0.2..=3.).contains(&derivative) {
                    if let Ok(features) = Feats::of(a, c) {
                        by.entry(features.keys.to_string()).or_default().push(derivative);
                        by.entry(format!("{}:{}", features.keys, c.families[c.dominant(&features)])).or_default().push(derivative);
                    }
                }
            }
        }
        last = Some((bid, rate, overall, f));
    }
    let mut out: Vec<(String, f64)> = by.into_iter().filter(|(_, v)| v.len() >= 20).map(|(k, mut v)| {
        v.sort_by(f64::total_cmp);
        let n = v.len();
        (k, if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2. })
    }).collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// Reads public.pkl into the mapped form.
pub fn convert(from: &Path, to: &Path) -> Result<(), String> {
    let raw = std::fs::read(from).map_err(|e| e.to_string())?;
    let (mut meta, records) = public::from_pickle(&raw)?;
    drop(raw);
    let c = Consts::standard();
    let decoded = records.iter().filter_map(|r| Some((r.0, r.1, ms_data::codec::decode(&r.2).ok()?)));
    meta.elasticities = rate_elasticities(decoded, &c);
    public::write(to, &meta, &records)
}

/// recommend._native_pop: the fitted population as the predictor reads it.
pub fn population(p: &Population, c: &Consts) -> Pop {
    let keyed = |t: &[(i64, f64)]| t.iter().copied().collect();
    let parts = &c.gap_part_names;
    let mut mbr: crate::predict::HashMap<i64, Vec<(f64, f64, f64)>> = Default::default();
    let mut rows = p.mbr.clone();
    rows.sort_by(|a, b| a.1.total_cmp(&b.1)); // VARIANTS order: rising rates, as base_of's anchors
    for (b, r, v, n) in rows {
        mbr.entry(b).or_default().push((r.ln(), v, n));
    }
    Pop {
        slope: keyed(&p.slope),
        window: keyed(&p.window),
        curve: keyed(&p.curve),
        knee: p.curve_knee,
        mb: p.mb.iter().map(|&(b, v, n)| (b, (v, n))).collect(),
        mbr,
        tau2: p.tau2,
        kappa: p.kappa,
        gap: p.gap.iter().map(|(k, g)| (*k, g.iter().filter_map(|(name, cs)| Some((parts.iter().position(|x| x == name)?, cs.clone()))).collect())).collect(),
        share_mean: p.share_mean.iter().map(|(k, m)| (*k, parts.iter().map(|name| m.iter().find(|x| x.0 == *name).map_or(0., |x| x.1)).collect())).collect(),
    }
}

/// The rate response of a fit's warm-up models.
pub fn slopes(p: &Population, meta: &Meta, c: &Consts) -> Slopes {
    let keyed = |t: &[(i64, f64)]| t.iter().copied().collect();
    let flat: std::collections::HashMap<String, f64> = meta.elasticities.iter().cloned().collect();
    Slopes::new(keyed(&p.slope), keyed(&p.curve), p.curve_knee, &flat, &c.families)
}

/// The public evidence of a data folder, mapped.
pub fn open(path: &Path) -> Result<Public, String> {
    Public::open(path)
}
