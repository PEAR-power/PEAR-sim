//! Optional compiled kernels for PEARSim.
//!
//! The Python package stays the reference implementation. This crate only
//! takes over work whose input is plain arrays (a compiled mode's matrices and
//! its event rows) and returns plain arrays, so the Python side keeps every
//! decision about circuits, modes and evidence.

mod brent;
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
