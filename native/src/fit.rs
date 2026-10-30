//! Fitting kernels for the population and warm-up fits (outside the calculator identity:
//! native_backend.identity_files leaves this file out).
use pyo3::prelude::*;

fn f64s(b: &[u8]) -> Vec<f64> {
    b.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect()
}

/// Projected Gauss-Seidel on A·x = rhs within [lower, upper] (a convex bounded ridge problem):
/// sweeps until no coordinate moves more than `tol`, at most `sweeps` times. A is n×n row-major f64
/// bytes (numpy `.tobytes()`); x0 the start. Same update order as the Python loops it replaces.
#[pyfunction]
pub fn box_cd(a: &[u8], rhs: Vec<f64>, x0: Vec<f64>, lower: Vec<f64>, upper: Vec<f64>, sweeps: usize, tol: f64) -> Vec<f64> {
    box_cd_on(&f64s(a), &rhs, x0, &lower, &upper, sweeps, tol)
}

fn box_cd_on(a: &[f64], rhs: &[f64], x0: Vec<f64>, lower: &[f64], upper: &[f64], sweeps: usize, tol: f64) -> Vec<f64> {
    let n = rhs.len();
    assert!(a.len() == n * n && x0.len() == n && lower.len() == n && upper.len() == n, "box_cd: shape mismatch");
    let mut x = x0;
    for _ in 0..sweeps {
        let mut moved = 0f64;
        for j in 0..n {
            let row = &a[j * n..(j + 1) * n];
            let dot: f64 = row.iter().zip(&x).map(|(p, q)| p * q).sum();
            let value = (x[j] + (rhs[j] - dot) / row[j]).max(lower[j]).min(upper[j]);
            moved = moved.max((value - x[j]).abs());
            x[j] = value;
        }
        if moved < tol {
            break;
        }
    }
    x
}

fn i64s(b: &[u8]) -> Vec<usize> {
    b.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap()) as usize).collect()
}

/// A·x = b by Gaussian elimination with partial pivoting (A n×n row-major).
pub(crate) fn solve(a: &[f64], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut m: Vec<f64> = a.to_vec();
    let mut x = b.to_vec();
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| m[i * n + c].abs().total_cmp(&m[j * n + c].abs())).unwrap();
        if p != c {
            for k in 0..n {
                m.swap(c * n + k, p * n + k);
            }
            x.swap(c, p);
        }
        for r in c + 1..n {
            let f = m[r * n + c] / m[c * n + c];
            for k in c..n {
                m[r * n + k] -= f * m[c * n + k];
            }
            x[r] -= f * x[c];
        }
    }
    for c in (0..n).rev() {
        let tail: f64 = (c + 1..n).map(|k| m[c * n + k] * x[k]).sum();
        x[c] = (x[c] - tail) / m[c * n + c];
    }
    x
}

/// warmup._fit's alternation: box-constrained ridge coefficients (A·coeff = Xᵀ·w·(y − effects) + prior, by
/// box_cd from the unconstrained solve), then each random effect group in turn as the ridge-shrunk weighted
/// mean of what the rest leaves. X is n×p row-major f64 bytes, y and weights f64 bytes; groups are
/// (member group index per row as i64 bytes, group count, ridge). → (coeff, effects per group).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn warmup_ridge(x: &[u8], y: &[u8], weights: &[u8], a: &[u8], lower: Vec<f64>, upper: Vec<f64>, prior: Vec<f64>,
                    groups: Vec<(Vec<u8>, usize, f64)>, iterations: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let groups: Vec<(Vec<usize>, usize, f64)> = groups.into_iter().map(|(ix, k, r)| (i64s(&ix), k, r)).collect();
    ridge_alternation(&f64s(x), &f64s(y), &f64s(weights), &f64s(a), &lower, &upper, &prior, &groups, iterations)
}

/// warmup_ridge on slices.
#[allow(clippy::too_many_arguments)]
pub(crate) fn ridge_alternation(x: &[f64], y: &[f64], w: &[f64], a_m: &[f64], lower: &[f64], upper: &[f64], prior: &[f64],
                                groups: &[(Vec<usize>, usize, f64)], iterations: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let (n, p) = (y.len(), prior.len());
    assert!(x.len() == n * p && w.len() == n && a_m.len() == p * p, "warmup_ridge: shape mismatch");
    let mut effects: Vec<Vec<f64>> = groups.iter().map(|(_, k, _)| vec![0.; *k]).collect();
    let mut mass: Vec<Vec<f64>> = Vec::new();
    for (ix, k, _) in groups {
        let mut m = vec![0.; *k];
        ix.iter().zip(w).for_each(|(&g, &wi)| m[g] += wi);
        mass.push(m);
    }
    let mut coeff = vec![0.; p];
    let mut fitted = vec![0.; n];
    for _ in 0..iterations {
        let mut rhs = vec![0.; p];
        for i in 0..n {
            let offset: f64 = groups.iter().zip(&effects).map(|((ix, _, _), e)| e[ix[i]]).sum();
            let v = w[i] * (y[i] - offset);
            rhs.iter_mut().zip(&x[i * p..(i + 1) * p]).for_each(|(r, xi)| *r += xi * v);
        }
        rhs.iter_mut().zip(prior).for_each(|(r, q)| *r += q);
        let start = solve(a_m, &rhs);
        coeff = box_cd_on(a_m, &rhs, start, lower, upper, 40, 1e-7);
        for (i, f) in fitted.iter_mut().enumerate() {
            *f = x[i * p..(i + 1) * p].iter().zip(&coeff).map(|(xi, c)| xi * c).sum();
        }
        for g in 0..groups.len() {
            let mut sum = vec![0.; groups[g].1];
            for i in 0..n {
                let others: f64 = (0..groups.len()).filter(|&h| h != g).map(|h| effects[h][groups[h].0[i]]).sum();
                sum[groups[g].0[i]] += w[i] * (y[i] - fitted[i] - others);
            }
            let ridge = groups[g].2;
            effects[g] = sum.iter().zip(&mass[g]).map(|(s, m)| s / (m + ridge)).collect();
        }
    }
    (coeff, effects)
}

/// warmup.timeline's state machine over completions in end order → (session, readiness minutes, exposure
/// minutes d) per completion. `transfer[target][played]` is warmup.transfer between FAMILIES indices.
#[allow(clippy::type_complexity)]
pub(crate) fn warmup_timeline(end: &[f64], seconds: &[f64], keys: &[i64], family: &[usize], transfer: &[Vec<f64>],
                              gap: f64, cooling: f64) -> (Vec<i64>, Vec<f64>, Vec<f64>) {
    let nf = transfer.len();
    // Readiness per (keymode, family), in first-touched keymode order; absent = 0.
    let mut state: Vec<(i64, Vec<f64>)> = Vec::new();
    let (mut sessions, mut minutes, mut ds) = (Vec::new(), Vec::new(), Vec::new());
    let (mut session, mut last_end): (i64, Option<f64>) = (-1, None);
    for i in 0..end.len() {
        let start = end[i] - seconds[i];
        match last_end {
            Some(last) if start - last <= gap => {
                let decay = (-(0f64.max(start - last)) / cooling).exp();
                state.iter_mut().for_each(|(_, v)| v.iter_mut().for_each(|x| *x *= decay));
            }
            _ => {
                state.clear();
                session += 1;
            }
        }
        let k = keys[i];
        let at = match state.iter().position(|(key, _)| *key == k) {
            Some(at) => at,
            None => {
                state.push((k, vec![0.; nf]));
                state.len() - 1
            }
        };
        minutes.push(state[at].1[family[i]]);
        let s = match last_end {
            None => seconds[i],
            Some(last) => seconds[i].min(0f64.max(end[i] - last)),
        };
        let d = 10f64.min(s / 60.);
        sessions.push(session);
        ds.push(d);
        let decay = (-s / cooling).exp();
        state.iter_mut().for_each(|(_, v)| v.iter_mut().for_each(|x| *x *= decay));
        for (target, x) in state[at].1.iter_mut().enumerate() {
            *x += d * transfer[target][family[i]];
        }
        last_end = Some(end[i]);
    }
    (sessions, minutes, ds)
}
