//! Python numeric semantics the calculator was written against.

/// `round(x)`: half to even.
#[inline]
pub fn round(x: f64) -> f64 {
    x.round_ties_even()
}

/// `round(x, nd)`: correctly rounded decimal (half to even on the exact binary value), back to the nearest double.
pub fn round_digits(x: f64, nd: usize) -> f64 {
    if !x.is_finite() {
        return x;
    }
    format!("{:.*}", nd, x).parse().unwrap()
}

/// `float(s)`: surrounding whitespace, inf/nan, single underscores between digits.
pub fn float(s: &str) -> Option<f64> {
    let s = s.trim();
    if s.contains('_') {
        let b = s.as_bytes();
        for (i, &c) in b.iter().enumerate() {
            if c == b'_' && !(i > 0 && i + 1 < b.len() && b[i - 1].is_ascii_digit() && b[i + 1].is_ascii_digit()) {
                return None;
            }
        }
        return s.replace('_', "").parse().ok();
    }
    s.parse().ok()
}

/// `int(s)` for a decimal integer string.
pub fn int(s: &str) -> Option<i64> {
    let s = s.trim();
    if s.contains('_') {
        return float(s).filter(|_| !s.contains(['.', 'e', 'E'])).map(|v| v as i64);
    }
    s.parse().ok()
}

/// `int(float(s))`: NaN is a ValueError (Err(false)), ±inf an OverflowError (Err(true)).
pub fn int_float(s: &str) -> Result<i64, bool> {
    match float(s) {
        None => Err(false),
        Some(v) if v.is_nan() => Err(false),
        Some(v) if v.is_infinite() => Err(true),
        Some(v) => Ok(v.trunc() as i64),
    }
}

/// `builtins.sum` of floats (CPython ≥3.12: Neumaier-compensated).
pub fn sum<I: IntoIterator<Item = f64>>(values: I) -> f64 {
    let mut total = 0.0f64;
    let mut c = 0.0f64;
    for v in values {
        let t = total + v;
        if total.abs() >= v.abs() {
            c += (total - t) + v;
        } else {
            c += (v - t) + total;
        }
        total = t;
    }
    if c != 0.0 && c.is_finite() {
        total + c
    } else {
        total
    }
}

/// `statistics.median`.
pub fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    }
}

/// Floor division of Python ints, b > 0.
#[inline]
pub fn floordiv(a: i64, b: i64) -> i64 {
    a.div_euclid(b)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rounding() {
        assert_eq!(round(2.5), 2.0);
        assert_eq!(round(3.5), 4.0);
        assert_eq!(round(-2.5), -2.0);
        assert_eq!(round_digits(0.0625, 3), 0.062);
        assert_eq!(round_digits(0.0635, 3), 0.064);
        assert_eq!(round_digits(333.3335, 3), 333.334);
        assert_eq!(round_digits(461.53846153846155, 3), 461.538);
    }
    #[test]
    fn floats() {
        assert_eq!(float(" 1_000 "), Some(1000.0));
        assert_eq!(float("1__0"), None);
        assert_eq!(int_float("12.9"), Ok(12));
        assert_eq!(int_float("-12.9"), Ok(-12));
        assert_eq!(int_float("nan"), Err(false));
        assert_eq!(floordiv(-1, 512), -1);
        assert_eq!(floordiv(511, 512), 0);
    }
    #[test]
    fn neumaier() {
        assert_eq!(sum([0.1; 10]), 1.0);
        assert_eq!(sum([1e100, 1.0, -1e100]), 1.0);
    }
}
