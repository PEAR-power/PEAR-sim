// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 PEAR Power and the PEARSim contributors.
// PEARSim provenance mark: fac45c9d-2490-41e0-8e53-bbd84e280889
//! Brent's root finder, a line-for-line port of SciPy's `brentq`
//! (scipy/optimize/Zeros/brentq.c). Used to place terminal events inside a
//! step exactly where `solve_ivp` places them.

/// Root of `f` in [xa, xb]; `f(xa)` and `f(xb)` must differ in sign (the
/// caller guarantees it: the bracket comes from a detected sign change).
pub fn brentq<F: FnMut(f64) -> f64>(
    mut f: F,
    xa: f64,
    xb: f64,
    xtol: f64,
    rtol: f64,
    iter: usize,
) -> f64 {
    let mut xpre = xa;
    let mut xcur = xb;
    let mut xblk = 0.0;
    let mut fblk = 0.0;
    let mut spre = 0.0;
    let mut scur = 0.0;

    let mut fpre = f(xpre);
    let mut fcur = f(xcur);
    if fpre == 0.0 {
        return xpre;
    }
    if fcur == 0.0 {
        return xcur;
    }
    for _ in 0..iter {
        if fpre != 0.0 && fcur != 0.0 && (fpre.is_sign_negative() != fcur.is_sign_negative()) {
            xblk = xpre;
            fblk = fpre;
            spre = xcur - xpre;
            scur = xcur - xpre;
        }
        if fblk.abs() < fcur.abs() {
            xpre = xcur;
            xcur = xblk;
            xblk = xpre;

            fpre = fcur;
            fcur = fblk;
            fblk = fpre;
        }

        let delta = (xtol + rtol * xcur.abs()) / 2.0;
        let sbis = (xblk - xcur) / 2.0;
        if fcur == 0.0 || sbis.abs() < delta {
            return xcur;
        }

        if spre.abs() > delta && fcur.abs() < fpre.abs() {
            let stry = if xpre == xblk {
                // interpolate
                -fcur * (xcur - xpre) / (fcur - fpre)
            } else {
                // extrapolate
                let dpre = (fpre - fcur) / (xpre - xcur);
                let dblk = (fblk - fcur) / (xblk - xcur);
                -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))
            };
            if 2.0 * stry.abs() < spre.abs().min(3.0 * sbis.abs() - delta) {
                // good short step
                spre = scur;
                scur = stry;
            } else {
                // bisect
                spre = sbis;
                scur = sbis;
            }
        } else {
            // bisect
            spre = sbis;
            scur = sbis;
        }

        xpre = xcur;
        fpre = fcur;
        if scur.abs() > delta {
            xcur += scur;
        } else {
            xcur += if sbis > 0.0 { delta } else { -delta };
        }
        fcur = f(xcur);
    }
    xcur
}
