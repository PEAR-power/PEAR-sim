// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 PEAR Power and the PEARSim contributors.
// PEARSim provenance mark: fac45c9d-2490-41e0-8e53-bbd84e280889
//! Optional compiled kernels for PEARSim.
//!
//! The Python package stays the reference implementation. This crate only
//! takes over work whose input is plain arrays (a compiled mode's matrices and
//! its event rows) and returns plain arrays, so the Python side keeps every
//! decision about circuits, modes and evidence.

mod brent;
mod extrema;
mod observe;
mod rk45;

use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Integrate x' = a x + c over [t0, t_bound] with terminal events.
///
/// Events are given row-wise: `ev_is_constant[i]` with `ev_constant[i]`, or
/// the affine row `ev_coef[i] . y + ev_offset[i]` reported as
/// `ev_early_value[i]` while t <= `ev_early_until[i]`; `ev_direction[i]` is
/// +1, -1 or 0.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn integrate_affine_rk45<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    c: PyReadonlyArray1<'py, f64>,
    y0: PyReadonlyArray1<'py, f64>,
    t0: f64,
    t_bound: f64,
    rtol: f64,
    atol: f64,
    max_step: f64,
    ev_is_constant: Vec<bool>,
    ev_constant: Vec<f64>,
    ev_coef: PyReadonlyArray2<'py, f64>,
    ev_offset: Vec<f64>,
    ev_early_until: Vec<f64>,
    ev_early_value: Vec<f64>,
    ev_direction: Vec<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let a = a.as_array();
    let n = a.nrows();
    if a.ncols() != n || c.len()? != n || y0.len()? != n {
        return Err(PyValueError::new_err("a must be n x n and c, y0 of length n"));
    }
    if !(t_bound > t0) {
        return Err(PyValueError::new_err("t_bound must exceed t0"));
    }
    let m = ev_is_constant.len();
    let coef = ev_coef.as_array();
    if [ev_constant.len(), ev_offset.len(), ev_early_until.len(), ev_early_value.len(), ev_direction.len()]
        .iter()
        .any(|&len| len != m)
        || (m > 0 && (coef.nrows() != m || coef.ncols() != n))
    {
        return Err(PyValueError::new_err("event rows have inconsistent lengths"));
    }
    let system = rk45::Affine {
        n,
        a: a.iter().copied().collect(),
        c: c.as_array().to_vec(),
    };
    let events: Vec<rk45::Event> = (0..m)
        .map(|i| rk45::Event {
            constant: if ev_is_constant[i] { Some(ev_constant[i]) } else { None },
            coef: coef.row(i).to_vec(),
            offset: ev_offset[i],
            early_until: ev_early_until[i],
            early_value: ev_early_value[i],
            direction: ev_direction[i],
        })
        .collect();
    let start = y0.as_array().to_vec();

    let solution = py.allow_threads(|| {
        rk45::integrate(&system, &start, t0, t_bound, rtol, atol, max_step, &events)
    });

    let k = solution.dense.len();
    let mut t_old = Vec::with_capacity(k);
    let mut t_new = Vec::with_capacity(k);
    let mut y_old = Vec::with_capacity(k * n);
    let mut q = Vec::with_capacity(k * n * 4);
    for step in &solution.dense {
        t_old.push(step.t_old);
        t_new.push(step.t);
        y_old.extend_from_slice(&step.y_old);
        q.extend_from_slice(&step.q);
    }
    let out = PyDict::new(py);
    out.set_item("ts", solution.ts.into_pyarray(py))?;
    out.set_item("ys", solution.ys.into_pyarray(py))?;
    out.set_item("dense_t_old", t_old.into_pyarray(py))?;
    out.set_item("dense_t", t_new.into_pyarray(py))?;
    out.set_item("dense_y_old", y_old.into_pyarray(py))?;
    out.set_item("dense_q", q.into_pyarray(py))?;
    match solution.event {
        Some((index, root)) => {
            out.set_item("event_index", index as i64)?;
            out.set_item("event_root", root)?;
        }
        None => {
            out.set_item("event_index", -1i64)?;
            out.set_item("event_root", f64::NAN)?;
        }
    }
    out.set_item("status", solution.status)?;
    out.set_item("message", solution.message)?;
    out.set_item("nfev", solution.nfev)?;
    Ok(out)
}

/// Root of the polynomial sum(coeffs[i] x^i) (Horner, highest first) in
/// [xa, xb] by the ported Brent method; for checking the port against SciPy.
#[pyfunction]
fn brentq_polynomial(coeffs: Vec<f64>, xa: f64, xb: f64, xtol: f64, rtol: f64) -> f64 {
    brent::brentq(
        |x| coeffs.iter().rev().fold(0.0, |acc, c| acc * x + c),
        xa,
        xb,
        xtol,
        rtol,
        100,
    )
}

#[pymodule]
fn pear_sim_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(integrate_affine_rk45, m)?)?;
    m.add_function(wrap_pyfunction!(brentq_polynomial, m)?)?;
    m.add_function(wrap_pyfunction!(rk45_tableau, m)?)?;
    m.add_function(wrap_pyfunction!(observe_segment, m)?)?;
    m.add_function(wrap_pyfunction!(state_extrema, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

/// The Dormand-Prince coefficients as compiled, for an exact comparison with SciPy.
#[pyfunction]
fn rk45_tableau(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let out = PyDict::new(py);
    let (c, a, b, e, p) = rk45::tableau();
    out.set_item("C", c)?;
    out.set_item("A", a)?;
    out.set_item("B", b)?;
    out.set_item("E", e)?;
    out.set_item("P", p)?;
    Ok(out)
}

fn rows(coef: &PyReadonlyArray2<'_, f64>, offset: Vec<f64>, n: usize) -> PyResult<observe::Rows> {
    let view = coef.as_array();
    let count = offset.len();
    if count > 0 && (view.nrows() != count || view.ncols() != n) {
        return Err(PyValueError::new_err("row matrix shape does not match its offsets"));
    }
    Ok(observe::Rows {
        count,
        n,
        coef: if count > 0 { view.iter().copied().collect() } else { Vec::new() },
        offset,
    })
}

fn window(duration: f64, intervals: usize, integrals: Vec<f64>, squares: Vec<f64>, energy: Vec<f64>) -> observe::Window {
    observe::Window { duration, intervals, integrals, squares, energy }
}

fn window_dict<'py>(py: Python<'py>, w: observe::Window) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("duration", w.duration)?;
    out.set_item("intervals", w.intervals)?;
    out.set_item("integrals", w.integrals)?;
    out.set_item("squares", w.squares)?;
    out.set_item("energy", w.energy)?;
    Ok(out)
}

/// Sample outputs and add the segment's quadrature to the running window totals.
///
/// `window_state` is (duration, intervals, integrals, squares, energy) for the
/// full window and then for the last-half window.
#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn observe_segment<'py>(
    py: Python<'py>,
    ts: Vec<f64>,
    dense_t_old: Vec<f64>,
    dense_t: Vec<f64>,
    dense_y_old: PyReadonlyArray1<'py, f64>,
    dense_q: PyReadonlyArray1<'py, f64>,
    nodes: Vec<f64>,
    weights: Vec<f64>,
    midpoint: f64,
    output_coef: PyReadonlyArray2<'py, f64>,
    output_offset: Vec<f64>,
    voltage_coef: PyReadonlyArray2<'py, f64>,
    voltage_offset: Vec<f64>,
    current_coef: PyReadonlyArray2<'py, f64>,
    current_offset: Vec<f64>,
    sample_times: Vec<f64>,
    full_state: (f64, usize, Vec<f64>, Vec<f64>, Vec<f64>),
    last_state: (f64, usize, Vec<f64>, Vec<f64>, Vec<f64>),
) -> PyResult<Bound<'py, PyDict>> {
    let count = dense_t.len();
    if dense_t_old.len() != count || ts.len() != count + 1 || count == 0 {
        return Err(PyValueError::new_err("ts must have one more entry than the dense steps"));
    }
    let y_old = dense_y_old.as_array();
    let q = dense_q.as_array();
    let n = y_old.len() / count;
    if y_old.len() != n * count || q.len() != n * count * 4 {
        return Err(PyValueError::new_err("dense arrays do not match the step count"));
    }
    let y_old = y_old.to_vec();
    let q = q.to_vec();
    let steps: Vec<rk45::Dense> = (0..count)
        .map(|i| rk45::Dense {
            t_old: dense_t_old[i],
            t: dense_t[i],
            y_old: y_old[i * n..(i + 1) * n].to_vec(),
            q: q[i * n * 4..(i + 1) * n * 4].to_vec(),
        })
        .collect();
    let outputs = rows(&output_coef, output_offset, n)?;
    let voltage = rows(&voltage_coef, voltage_offset, n)?;
    let current = rows(&current_coef, current_offset, n)?;
    if voltage.count != current.count {
        return Err(PyValueError::new_err("voltage and current rows differ in count"));
    }
    let energy = if voltage.count > 0 { Some(observe::Energy { voltage, current }) } else { None };
    let full = window(full_state.0, full_state.1, full_state.2, full_state.3, full_state.4);
    let last = window(last_state.0, last_state.1, last_state.2, last_state.3, last_state.4);
    let observed = py.allow_threads(|| {
        observe::observe(&steps, &ts, &nodes, &weights, midpoint, &outputs, energy.as_ref(), &sample_times, full, last)
    });
    let out = PyDict::new(py);
    out.set_item("full", window_dict(py, observed.full)?)?;
    out.set_item("last_half", window_dict(py, observed.last_half)?)?;
    out.set_item("samples", observed.samples.into_pyarray(py))?;
    Ok(out)
}

fn dense_steps(
    dense_t_old: &[f64],
    dense_t: &[f64],
    dense_y_old: &PyReadonlyArray1<'_, f64>,
    dense_q: &PyReadonlyArray1<'_, f64>,
) -> PyResult<(Vec<rk45::Dense>, usize)> {
    let count = dense_t.len();
    let y_old = dense_y_old.as_array();
    let q = dense_q.as_array();
    if count == 0 || dense_t_old.len() != count {
        return Err(PyValueError::new_err("dense step arrays are empty or inconsistent"));
    }
    let n = y_old.len() / count;
    if y_old.len() != n * count || q.len() != n * count * 4 {
        return Err(PyValueError::new_err("dense arrays do not match the step count"));
    }
    let y_old = y_old.to_vec();
    let q = q.to_vec();
    Ok((
        (0..count)
            .map(|i| rk45::Dense {
                t_old: dense_t_old[i],
                t: dense_t[i],
                y_old: y_old[i * n..(i + 1) * n].to_vec(),
                q: q[i * n * 4..(i + 1) * n * 4].to_vec(),
            })
            .collect(),
        n,
    ))
}

/// Per-state (present, minimum, minimum_time, maximum, maximum_time) over a window.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn state_extrema<'py>(
    py: Python<'py>,
    ts: Vec<f64>,
    dense_t_old: Vec<f64>,
    dense_t: Vec<f64>,
    dense_y_old: PyReadonlyArray1<'py, f64>,
    dense_q: PyReadonlyArray1<'py, f64>,
    window_start: f64,
    window_end: f64,
    order: Vec<usize>,
) -> PyResult<Vec<(bool, f64, f64, f64, f64)>> {
    let (steps, n) = dense_steps(&dense_t_old, &dense_t, &dense_y_old, &dense_q)?;
    if ts.len() != steps.len() + 1 || order.iter().any(|&i| i >= n) {
        return Err(PyValueError::new_err("ts or state order does not match the steps"));
    }
    let result = py.allow_threads(|| {
        extrema::state_extrema(&steps, &ts, window_start, window_end, &order, n)
    });
    Ok(result
        .into_iter()
        .map(|e| (e.present, e.minimum, e.minimum_time, e.maximum, e.maximum_time))
        .collect())
}
