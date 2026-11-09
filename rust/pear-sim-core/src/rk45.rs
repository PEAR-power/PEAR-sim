// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 PEAR Power and the PEARSim contributors.
// PEARSim provenance mark: fac45c9d-2490-41e0-8e53-bbd84e280889
//! Dormand-Prince 5(4) on an affine system, mirroring SciPy's `solve_ivp(method="RK45")`.
//!
//! Every step follows scipy/integrate/_ivp/rk.py and ivp.py operation for
//! operation: the same tableau, initial-step selection, error norm, step
//! control, dense output and terminal-event handling (Brent's method with
//! xtol = rtol = 4 eps). The reference stays SciPy; agreement is checked in
//! tests/test_rust_core.py, in tolerances, because BLAS may sum in a different
//! order than these loops.

use crate::brent::brentq;

const SAFETY: f64 = 0.9;
const MIN_FACTOR: f64 = 0.2;
const MAX_FACTOR: f64 = 10.0;
const ERROR_EXPONENT: f64 = -1.0 / 5.0; // -1 / (error_estimator_order + 1)
const N_STAGES: usize = 6;

const C: [f64; 6] = [0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0];
const A: [[f64; 5]; 6] = [
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0 / 5.0, 0.0, 0.0, 0.0, 0.0],
    [3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0],
    [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0],
    [
        19372.0 / 6561.0,
        -25360.0 / 2187.0,
        64448.0 / 6561.0,
        -212.0 / 729.0,
        0.0,
    ],
    [
        9017.0 / 3168.0,
        -355.0 / 33.0,
        46732.0 / 5247.0,
        49.0 / 176.0,
        -5103.0 / 18656.0,
    ],
];
const B: [f64; 6] = [
    35.0 / 384.0,
    0.0,
    500.0 / 1113.0,
    125.0 / 192.0,
    -2187.0 / 6784.0,
    11.0 / 84.0,
];
const E: [f64; 7] = [
    -71.0 / 57600.0,
    0.0,
    71.0 / 16695.0,
    -71.0 / 1920.0,
    17253.0 / 339200.0,
    -22.0 / 525.0,
    1.0 / 40.0,
];
const P: [[f64; 4]; 7] = [
    [
        1.0,
        -8048581381.0 / 2820520608.0,
        8663915743.0 / 2820520608.0,
        -12715105075.0 / 11282082432.0,
    ],
    [0.0, 0.0, 0.0, 0.0],
    [
        0.0,
        131558114200.0 / 32700410799.0,
        -68118460800.0 / 10900136933.0,
        87487479700.0 / 32700410799.0,
    ],
    [
        0.0,
        -1754552775.0 / 470086768.0,
        14199869525.0 / 1410260304.0,
        -10690763975.0 / 1880347072.0,
    ],
    [
        0.0,
        127303824393.0 / 49829197408.0,
        -318862633887.0 / 49829197408.0,
        701980252875.0 / 199316789632.0,
    ],
    [
        0.0,
        -282668133.0 / 205662961.0,
        2019193451.0 / 616988883.0,
        -1453857185.0 / 822651844.0,
    ],
    [
        0.0,
        40617522.0 / 29380423.0,
        -110615467.0 / 29380423.0,
        69997945.0 / 29380423.0,
    ],
];

/// x' = a x + c, with `a` row-major n x n.
pub struct Affine {
    pub n: usize,
    pub a: Vec<f64>,
    pub c: Vec<f64>,
}

impl Affine {
    fn eval(&self, y: &[f64], out: &mut [f64]) {
        for i in 0..self.n {
            let row = &self.a[i * self.n..(i + 1) * self.n];
            let mut sum = 0.0;
            for j in 0..self.n {
                sum += row[j] * y[j];
            }
            out[i] = sum + self.c[i];
        }
    }
}

/// One terminal event: value(t, y) crosses zero in `direction` (+1, -1, 0).
#[derive(Clone)]
pub struct Event {
    /// Constant value for the whole segment (never crosses).
    pub constant: Option<f64>,
    /// value = coef . y + offset, after `early_until`.
    pub coef: Vec<f64>,
    pub offset: f64,
    /// Up to and including this time the event reports `early_value`.
    pub early_until: f64,
    pub early_value: f64,
    pub direction: f64,
}

impl Event {
    fn value(&self, t: f64, y: &[f64]) -> f64 {
        if let Some(v) = self.constant {
            return v;
        }
        if t <= self.early_until {
            return self.early_value;
        }
        let mut sum = 0.0;
        for (c, v) in self.coef.iter().zip(y) {
            sum += c * v;
        }
        sum + self.offset
    }
}

/// Dense output of one step, SciPy's RkDenseOutput.
#[derive(Clone)]
pub struct Dense {
    pub t_old: f64,
    pub t: f64,
    pub y_old: Vec<f64>,
    /// n x 4, row-major.
    pub q: Vec<f64>,
}

impl Dense {
    pub fn eval(&self, t: f64, out: &mut [f64]) {
        let h = self.t - self.t_old;
        let x = (t - self.t_old) / h;
        let p0 = x;
        let p1 = p0 * x;
        let p2 = p1 * x;
        let p3 = p2 * x;
        let n = self.y_old.len();
        for i in 0..n {
            let q = &self.q[i * 4..i * 4 + 4];
            let dot = q[0] * p0 + q[1] * p1 + q[2] * p2 + q[3] * p3;
            out[i] = h * dot + self.y_old[i];
        }
    }
}

pub struct Solution {
    pub ts: Vec<f64>,
    /// one state per entry of `ts`, concatenated.
    pub ys: Vec<f64>,
    pub dense: Vec<Dense>,
    /// (event index, root) of the terminal event, if one fired.
    pub event: Option<(usize, f64)>,
    pub status: i32,
    pub message: String,
    pub nfev: usize,
}

fn rms_norm(x: &[f64]) -> f64 {
    let mut sq = 0.0;
    for v in x {
        sq += v * v;
    }
    sq.sqrt() / (x.len() as f64).sqrt()
}

fn next_after_up(t: f64) -> f64 {
    if t.is_nan() || t == f64::INFINITY {
        return t;
    }
    if t == 0.0 {
        return f64::from_bits(1);
    }
    let bits = t.to_bits();
    if t > 0.0 {
        f64::from_bits(bits + 1)
    } else {
        f64::from_bits(bits - 1)
    }
}

#[allow(clippy::too_many_arguments)]
fn select_initial_step(
    sys: &Affine,
    t0: f64,
    y0: &[f64],
    t_bound: f64,
    max_step: f64,
    f0: &[f64],
    rtol: f64,
    atol: f64,
    nfev: &mut usize,
) -> f64 {
    let n = y0.len();
    if n == 0 {
        return f64::INFINITY;
    }
    let interval_length = (t_bound - t0).abs();
    if interval_length == 0.0 {
        return 0.0;
    }
    let scale: Vec<f64> = y0.iter().map(|v| atol + v.abs() * rtol).collect();
    let d0 = rms_norm(&y0.iter().zip(&scale).map(|(v, s)| v / s).collect::<Vec<_>>());
    let d1 = rms_norm(&f0.iter().zip(&scale).map(|(v, s)| v / s).collect::<Vec<_>>());
    let mut h0 = if d0 < 1e-5 || d1 < 1e-5 { 1e-6 } else { 0.01 * d0 / d1 };
    h0 = h0.min(interval_length);
    let y1: Vec<f64> = y0.iter().zip(f0).map(|(y, f)| y + h0 * f).collect();
    let mut f1 = vec![0.0; n];
    sys.eval(&y1, &mut f1);
    *nfev += 1;
    let d2 = rms_norm(
        &f1.iter()
            .zip(f0)
            .zip(&scale)
            .map(|((a, b), s)| (a - b) / s)
            .collect::<Vec<_>>(),
    ) / h0;
    let h1 = if d1 <= 1e-15 && d2 <= 1e-15 {
        (1e-6f64).max(h0 * 1e-3)
    } else {
        (0.01 / d1.max(d2)).powf(1.0 / 5.0)
    };
    (100.0 * h0).min(h1).min(interval_length).min(max_step)
}

/// Integrate x' = a x + c from t0 to t_bound with terminal events.
#[allow(clippy::too_many_arguments)]
pub fn integrate(
    sys: &Affine,
    y0: &[f64],
    t0: f64,
    t_bound: f64,
    rtol_in: f64,
    atol: f64,
    max_step: f64,
    events: &[Event],
) -> Solution {
    let n = sys.n;
    let eps = f64::EPSILON;
    let rtol = if rtol_in < 100.0 * eps { 100.0 * eps } else { rtol_in };
    let mut nfev = 0usize;

    let mut t = t0;
    let mut y = y0.to_vec();
    let mut f = vec![0.0; n];
    sys.eval(&y, &mut f);
    nfev += 1;
    let mut h_abs = select_initial_step(sys, t, &y, t_bound, max_step, &f, rtol, atol, &mut nfev);

    let mut ts = vec![t0];
    let mut ys = y0.to_vec();
    let mut dense: Vec<Dense> = Vec::new();
    let mut g: Vec<f64> = events.iter().map(|e| e.value(t0, y0)).collect();
    let mut fired: Option<(usize, f64)> = None;

    let mut k = vec![0.0; (N_STAGES + 1) * n];
    let mut stage_y = vec![0.0; n];
    let mut y_new = vec![0.0; n];
    let mut f_new = vec![0.0; n];
    let mut status: Option<i32> = None;
    let mut message = String::new();
    let mut finished = n == 0 || t == t_bound;
    if finished {
        status = Some(0);
    }

    while status.is_none() {
        // ---- solver.step() -------------------------------------------------
        let t_start = t;
        let min_step = 10.0 * (next_after_up(t) - t).abs();
        h_abs = if h_abs > max_step {
            max_step
        } else if h_abs < min_step {
            min_step
        } else {
            h_abs
        };
        let mut step_accepted = false;
        let mut step_rejected = false;
        let mut h = 0.0;
        let mut t_new = t;
        while !step_accepted {
            if h_abs < min_step {
                status = Some(-1);
                message = "Required step size is less than spacing between numbers.".into();
                break;
            }
            h = h_abs;
            t_new = t + h;
            if t_new - t_bound > 0.0 {
                t_new = t_bound;
            }
            h = t_new - t;
            h_abs = h.abs();

            // rk_step
            k[..n].copy_from_slice(&f);
            for s in 1..N_STAGES {
                for i in 0..n {
                    let mut dot = 0.0;
                    for j in 0..s {
                        dot += k[j * n + i] * A[s][j];
                    }
                    stage_y[i] = y[i] + dot * h;
                }
                let (_, rest) = k.split_at_mut(s * n);
                sys.eval(&stage_y, &mut rest[..n]);
                nfev += 1;
            }
            for i in 0..n {
                let mut dot = 0.0;
                for j in 0..N_STAGES {
                    dot += k[j * n + i] * B[j];
                }
                y_new[i] = y[i] + h * dot;
            }
            sys.eval(&y_new, &mut f_new);
            nfev += 1;
            k[N_STAGES * n..].copy_from_slice(&f_new);

            // error norm
            let mut sq = 0.0;
            for i in 0..n {
                let mut dot = 0.0;
                for j in 0..=N_STAGES {
                    dot += k[j * n + i] * E[j];
                }
                let scale = atol + y[i].abs().max(y_new[i].abs()) * rtol;
                let e = dot * h / scale;
                sq += e * e;
            }
            let error_norm = sq.sqrt() / (n as f64).sqrt();
            if error_norm < 1.0 {
                let mut factor = if error_norm == 0.0 {
                    MAX_FACTOR
                } else {
                    MAX_FACTOR.min(SAFETY * error_norm.powf(ERROR_EXPONENT))
                };
                if step_rejected {
                    factor = factor.min(1.0);
                }
                h_abs *= factor;
                step_accepted = true;
            } else {
                h_abs *= MIN_FACTOR.max(SAFETY * error_norm.powf(ERROR_EXPONENT));
                step_rejected = true;
            }
        }
        if status == Some(-1) {
            break;
        }
        let _ = h;
        let y_old = y.clone();
        y.copy_from_slice(&y_new);
        f.copy_from_slice(&f_new);
        t = t_new;
        let t_old = t_start;
        if t - t_bound >= 0.0 {
            finished = true;
        }
        if finished {
            status = Some(0);
        }

        // ---- dense output ---------------------------------------------------
        let mut q = vec![0.0; n * 4];
        for i in 0..n {
            for col in 0..4 {
                let mut dot = 0.0;
                for j in 0..=N_STAGES {
                    dot += k[j * n + i] * P[j][col];
                }
                q[i * 4 + col] = dot;
            }
        }
        let sol = Dense { t_old, t, y_old, q };

        // ---- events (all terminal: the earliest root ends the segment) --------
        let mut t_report = t;
        let mut y_report = y.clone();
        if !events.is_empty() {
            let g_new: Vec<f64> = events.iter().map(|e| e.value(t, &y)).collect();
            let mut roots: Vec<(usize, f64)> = Vec::new();
            for (index, event) in events.iter().enumerate() {
                let (g0, g1) = (g[index], g_new[index]);
                let up = g0 <= 0.0 && g1 >= 0.0;
                let down = g0 >= 0.0 && g1 <= 0.0;
                let active = (up && event.direction > 0.0)
                    || (down && event.direction < 0.0)
                    || ((up || down) && event.direction == 0.0);
                if active {
                    let mut buffer = vec![0.0; n];
                    let root = brentq(
                        |time| {
                            sol.eval(time, &mut buffer);
                            event.value(time, &buffer)
                        },
                        t_old,
                        t,
                        4.0 * eps,
                        4.0 * eps,
                        100,
                    );
                    roots.push((index, root));
                }
            }
            if !roots.is_empty() {
                // np.argsort on <= a few entries: stable for ties.
                roots.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                let (index, root) = roots[0];
                fired = Some((index, root));
                status = Some(1);
                t_report = root;
                sol.eval(root, &mut y_report);
            }
            g = g_new;
        }

        dense.push(sol);
        let do_not_append = ts.len() > 1 && *ts.last().unwrap() == t_report;
        if !do_not_append {
            ts.push(t_report);
            ys.extend_from_slice(&y_report);
        } else {
            dense.pop();
        }
    }

    Solution {
        ts,
        ys,
        dense,
        event: fired,
        status: status.unwrap_or(-1),
        message,
        nfev,
    }
}

/// The compiled tableau, for an exact comparison with SciPy's arrays.
#[allow(clippy::type_complexity)]
pub fn tableau() -> (Vec<f64>, Vec<Vec<f64>>, Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
    (
        C.to_vec(),
        A.iter().map(|row| row.to_vec()).collect(),
        B.to_vec(),
        E.to_vec(),
        P.iter().map(|row| row.to_vec()).collect(),
    )
}
