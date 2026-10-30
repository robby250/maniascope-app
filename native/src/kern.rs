//! Small shared kernels of the calculator.
use crate::py;

/// Speed (Hz) of a gap; below the knee it tapers to 0 (flam ≈ chord).
#[inline]
pub fn v(d: f64, knee: f64, taper: f64) -> f64 {
    if d >= knee {
        return 1.0 / d;
    }
    (d.max(0.0) / knee).powf(taper) / knee
}

#[inline]
pub fn smooth(x: f64, lo: f64, hi: f64) -> f64 {
    let x = (x - lo) / (hi - lo);
    if !(x > 0.0) {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    x * x * (3.0 - 2.0 * x)
}

/// Columns strictly between the outermost fingers of `mask` (13 → 2).
#[inline]
pub fn inner(mask: u32, keys: usize) -> u32 {
    let m = mask & ((1u64 << keys) - 1) as u32;
    if m == 0 {
        return 0;
    }
    let lo = m.trailing_zeros();
    let hi = 31 - m.leading_zeros();
    if hi <= lo + 1 {
        return 0;
    }
    (((1u64 << hi) - 1) as u32) & !(((1u64 << (lo + 1)) - 1) as u32)
}

/// 0..1: how far an event gap is off a reference (see skill_calc._off_grid).
#[inline]
pub fn off_grid(g: f64, reference: f64, pulse: bool) -> f64 {
    let mut r = g / reference;
    if r < 1.0 && !pulse {
        r = 1.0 / r;
    }
    let x = if r >= 1.0 { (r / 1f64.max(py::round(r))).log2().abs() } else { r.log2().abs() };
    1f64.min(2.2 * 0f64.max((x - py::round(x)).abs() - 0.06))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn inner_mask() {
        assert_eq!(inner(0b101, 7), 0b010);
        assert_eq!(inner(0b1000001, 7), 0b0111110);
        assert_eq!(inner(0b11, 7), 0);
        assert_eq!(inner(0, 7), 0);
    }
}
