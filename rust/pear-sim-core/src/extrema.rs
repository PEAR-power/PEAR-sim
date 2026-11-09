// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 PEAR Power and the PEARSim contributors.
//! State extrema over a window, from each step's dense polynomial.
//!
//! Mirrors pear_sim.core.metrics._dense_state_extrema for RK45 steps: on every
//! step the candidates are 17 evenly spaced probe points plus every real root
//! of the polynomial's derivative inside the window (clamped to it); the
//! extremes among the candidates win, earliest first on ties.
//!
//! One deliberate difference: NumPy finds the derivative's roots as companion-
//! matrix eigenvalues. Here the cubic is split into monotone pieces at its own
//! critical points and each sign change is polished with Brent's method. Both
//! give the roots to machine precision; at an extremum the value is flat, so a
//! root displaced by 1e-15 changes the reported value by about 1e-30.

use crate::brent::brentq;
use crate::rk45::Dense;

const PROBE_SUBINTERVALS: usize = 16;
const BOUNDARY_TOLERANCE: f64 = 1e-9;

#[derive(Clone, Copy)]
pub struct Extremum {
    pub present: bool,
    pub minimum: f64,
    pub minimum_time: f64,
    pub maximum: f64,
    pub maximum_time: f64,
}

impl Extremum {
    fn consider(&mut self, value: f64, time: f64) {
        if !self.present {
            *self = Extremum {
                present: true,
                minimum: value,
                minimum_time: time,
                maximum: value,
                maximum_time: time,
            };
            return;
        }
        if value < self.minimum {
            self.minimum = value;
            self.minimum_time = time;
        }
        if value > self.maximum {
            self.maximum = value;
            self.maximum_time = time;
        }
    }
}

/// Horner, lowest coefficient first (NumPy polyval order).
fn polyval(c: &[f64], x: f64) -> f64 {
    let mut acc = c[c.len() - 1];
    for k in (0..c.len() - 1).rev() {
        acc = c[k] + acc * x;
    }
    acc
}

/// Real roots of the polynomial `c` (lowest first) inside [lo, hi].
fn real_roots_in(c: &[f64], lo: f64, hi: f64, out: &mut Vec<f64>) {
    let mut degree = c.len() - 1;
    while degree > 0 && c[degree] == 0.0 {
        degree -= 1;
    }
    let c = &c[..=degree];
    match degree {
        0 => {}
        1 => {
            let root = -c[0] / c[1];
            if root >= lo && root <= hi {
                out.push(root);
            }
        }
        _ => {
            // Split at the critical points of c (roots of its derivative),
            // then each monotone piece holds at most one root.
            let derivative: Vec<f64> = (1..=degree).map(|k| c[k] * k as f64).collect();
            let mut cuts = vec![lo];
            let mut critical = Vec::new();
            real_roots_in(&derivative, lo, hi, &mut critical);
            critical.sort_by(|a, b| a.partial_cmp(b).unwrap());
            for point in critical {
                if point > *cuts.last().unwrap() && point < hi {
                    cuts.push(point);
                }
            }
            cuts.push(hi);
            for pair in cuts.windows(2) {
                let (a, b) = (pair[0], pair[1]);
                let (fa, fb) = (polyval(c, a), polyval(c, b));
                if fa == 0.0 {
                    out.push(a);
                } else if fb == 0.0 {
                    // the next piece (or the end) reports it as its `a`
                    if b == hi {
                        out.push(b);
                    }
                } else if fa.is_sign_negative() != fb.is_sign_negative() {
                    let eps = f64::EPSILON;
                    out.push(brentq(|x| polyval(c, x), a, b, 4.0 * eps, 4.0 * eps, 200));
                }
            }
        }
    }
}

/// Extrema of each state over [window_start, window_end] across the steps.
///
/// `order` lists state indices in the order the reference visits them (by
/// component id), so ties resolve identically.
pub fn state_extrema(
    steps: &[Dense],
    ts: &[f64],
    window_start: f64,
    window_end: f64,
    order: &[usize],
    n: usize,
) -> Vec<Extremum> {
    let empty = Extremum {
        present: false,
        minimum: 0.0,
        minimum_time: 0.0,
        maximum: 0.0,
        maximum_time: 0.0,
    };
    let mut result = vec![empty; n];
    let mut roots = Vec::new();
    let mut candidates: Vec<f64> = Vec::new();
    for i in 0..ts.len().saturating_sub(1) {
        let start = ts[i].max(window_start);
        let end = ts[i + 1].min(window_end);
        if end <= start {
            continue;
        }
        let step = &steps[i];
        let dense_start = step.t_old;
        let dense_step = step.t - step.t_old;
        let normalized_start = (start - dense_start) / dense_step;
        let normalized_end = (end - dense_start) / dense_step;
        // np.linspace(a, b, 17): a + k * (b - a) / 16, last point exactly b.
        let span = normalized_end - normalized_start;
        let delta = span / PROBE_SUBINTERVALS as f64;
        for &state in order {
            let q = &step.q[state * 4..state * 4 + 4];
            let coefficients = [
                step.y_old[state],
                dense_step * q[0],
                dense_step * q[1],
                dense_step * q[2],
                dense_step * q[3],
            ];
            let derivative = [
                coefficients[1],
                coefficients[2] * 2.0,
                coefficients[3] * 3.0,
                coefficients[4] * 4.0,
            ];
            candidates.clear();
            for k in 0..=PROBE_SUBINTERVALS {
                candidates.push(if k == PROBE_SUBINTERVALS {
                    normalized_end
                } else {
                    normalized_start + k as f64 * delta
                });
            }
            roots.clear();
            real_roots_in(
                &derivative,
                normalized_start - BOUNDARY_TOLERANCE,
                normalized_end + BOUNDARY_TOLERANCE,
                &mut roots,
            );
            for &root in &roots {
                candidates.push(root.max(normalized_start).min(normalized_end));
            }
            candidates.sort_by(|a, b| a.partial_cmp(b).unwrap());
            candidates.dedup();
            for &point in &candidates {
                let value = polyval(&coefficients, point);
                let time = dense_start + dense_step * point;
                result[state].consider(value, time);
            }
        }
    }
    result
}
