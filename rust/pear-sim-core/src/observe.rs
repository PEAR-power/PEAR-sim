// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 PEAR Power and the PEARSim contributors.
//! What the simulator reads from one integrated segment, computed in place.
//!
//! Mirrors pear_sim.core.metrics._MetricAccumulator.add_solution and
//! _integrate_dense_interval (Gauss-Legendre quadrature over every step, with
//! the last-half window split at its start) and the solver's output sampling
//! (OdeSolution lookup with side="left"). Outputs and component voltages and
//! currents are affine in the state: row . y + offset, rows supplied by the
//! compiled mode.

use crate::rk45::Dense;

pub struct Rows {
    pub count: usize,
    pub n: usize,
    /// count x n, row-major.
    pub coef: Vec<f64>,
    pub offset: Vec<f64>,
}

impl Rows {
    fn eval(&self, y: &[f64], out: &mut [f64]) {
        for r in 0..self.count {
            let row = &self.coef[r * self.n..(r + 1) * self.n];
            let mut sum = 0.0;
            for (c, v) in row.iter().zip(y) {
                sum += c * v;
            }
            out[r] = sum + self.offset[r];
        }
    }
}

#[derive(Default, Clone)]
pub struct Window {
    pub duration: f64,
    pub intervals: usize,
    pub integrals: Vec<f64>,
    pub squares: Vec<f64>,
    pub energy: Vec<f64>,
}

impl Window {
    pub fn new(outputs: usize, components: usize) -> Self {
        Window {
            duration: 0.0,
            intervals: 0,
            integrals: vec![0.0; outputs],
            squares: vec![0.0; outputs],
            energy: vec![0.0; components],
        }
    }

    fn add(&mut self, other: &Window) {
        self.duration += other.duration;
        self.intervals += 1;
        for (a, b) in self.integrals.iter_mut().zip(&other.integrals) {
            *a += b;
        }
        for (a, b) in self.squares.iter_mut().zip(&other.squares) {
            *a += b;
        }
        for (a, b) in self.energy.iter_mut().zip(&other.energy) {
            *a += b;
        }
    }
}

pub struct Energy {
    pub voltage: Rows,
    pub current: Rows,
}

pub struct Observed {
    pub full: Window,
    pub last_half: Window,
    /// samples x outputs, row-major.
    pub samples: Vec<f64>,
}

#[allow(clippy::too_many_arguments)]
fn integrate_interval(
    step: &Dense,
    start: f64,
    end: f64,
    nodes: &[f64],
    weights: &[f64],
    outputs: &Rows,
    energy: Option<&Energy>,
    state: &mut [f64],
    values: &mut [f64],
    volts: &mut [f64],
    amps: &mut [f64],
) -> Window {
    let components = energy.map_or(0, |e| e.voltage.count);
    let mut window = Window::new(outputs.count, components);
    window.duration = end - start;
    let half_width = 0.5 * (end - start);
    let midpoint = 0.5 * (start + end);
    for (node, base_weight) in nodes.iter().zip(weights) {
        let time = midpoint + half_width * node;
        let weight = half_width * base_weight;
        step.eval(time, state);
        outputs.eval(state, values);
        for (k, value) in values.iter().enumerate() {
            window.integrals[k] += weight * value;
            window.squares[k] += weight * value * value;
        }
        if let Some(rows) = energy {
            rows.voltage.eval(state, volts);
            rows.current.eval(state, amps);
            for c in 0..components {
                window.energy[c] += weight * volts[c] * amps[c];
            }
        }
    }
    window
}

/// Index of the step whose dense output `solution(t)` uses (SciPy OdeSolution,
/// ascending, side="left").
fn segment_for(ts: &[f64], t: f64) -> usize {
    let index = ts.partition_point(|&boundary| boundary < t);
    let segments = ts.len() - 1;
    (index.max(1) - 1).min(segments - 1)
}

#[allow(clippy::too_many_arguments)]
pub fn observe(
    steps: &[Dense],
    ts: &[f64],
    nodes: &[f64],
    weights: &[f64],
    midpoint: f64,
    outputs: &Rows,
    energy: Option<&Energy>,
    sample_times: &[f64],
    full_start: Window,
    last_half_start: Window,
) -> Observed {
    // The running window totals come in and go back out, so contributions are
    // summed in the same order as the reference accumulator sums them.
    let n = outputs.n;
    let components = energy.map_or(0, |e| e.voltage.count);
    let mut full = full_start;
    let mut last_half = last_half_start;
    let mut state = vec![0.0; n];
    let mut values = vec![0.0; outputs.count];
    let mut volts = vec![0.0; components];
    let mut amps = vec![0.0; components];

    for i in 0..ts.len().saturating_sub(1) {
        let (start, end) = (ts[i], ts[i + 1]);
        if end <= start {
            continue;
        }
        let step = &steps[i];
        let contribution = integrate_interval(
            step, start, end, nodes, weights, outputs, energy,
            &mut state, &mut values, &mut volts, &mut amps,
        );
        full.add(&contribution);
        if end <= midpoint {
            continue;
        }
        if start >= midpoint {
            last_half.add(&contribution);
        } else {
            let tail = integrate_interval(
                step, midpoint, end, nodes, weights, outputs, energy,
                &mut state, &mut values, &mut volts, &mut amps,
            );
            last_half.add(&tail);
        }
    }

    let mut samples = Vec::with_capacity(sample_times.len() * outputs.count);
    for &time in sample_times {
        steps[segment_for(ts, time)].eval(time, &mut state);
        outputs.eval(&state, &mut values);
        samples.extend_from_slice(&values);
    }
    Observed { full, last_half, samples }
}
