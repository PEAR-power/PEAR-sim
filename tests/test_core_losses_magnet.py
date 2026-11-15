# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""VALIDATION of the iGSE producer against measured core loss (Princeton MagNet).

This is a comparison with MEASUREMENT, not with another piece of code, so it
can catch an assumption the producer and its closed forms share.

Design, chosen so the test is not circular:

* Steinmetz coefficients are fitted here, by least squares in log space, on
  the MagNet SINUSOIDAL points only (per temperature, two frequency bands
  split at 150 kHz). Published coefficients (e.g. the MAS database) are NOT
  used: MAS refits its ferrite coefficients preferring MagNet data, so they
  would compare MagNet with itself.
* The claim under test is what iGSE adds over Steinmetz: predicting
  NON-SINUSOIDAL waveforms (triangles and trapezoids) that the fit never saw,
  through the production function ``igse_volumetric_loss`` on the exact
  piecewise-linear B(t) MagNet describes.

The gate discriminates: it is calibrated so that iGSE passes and the naive
answer — the sinusoidal Steinmetz equation at f = 1/T, which ignores the
waveform shape — FAILS, and the test asserts both. Thresholds were set from
the measurement on the full MagNet N87 zero-bias data (all four temperatures:
iGSE median |error| 7-15 %, 95th percentile 31-51 %; naive median 19-21 %,
95th percentile 51-73 %), not chosen to make a run pass.

Declared scope of the evidence: one ferrite (TDK N87), zero DC bias, 25 and
90 degC, 50-450 kHz, B up to ~0.3 T. Nothing here says anything about other
materials, DC bias or the MAS coefficients themselves.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pear_sim.analysis.core_losses import igse_volumetric_loss

DATA = Path(__file__).with_name("data") / "magnet_n87_dc0_25c_90c.csv"
BAND_SPLIT_HZ = 150e3
MEDIAN_LIMIT = 0.20
P95_LIMIT = 0.60


_COLUMNS = {
    "frequency_hz": "frequency",
    "b_peak_t": "b_peak",
    "duty_p": "duty_p",
    "duty_n": "duty_n",
    "temperature_c": "temperature",
    "loss_w_per_m3": "loss",
}


def _load():
    lines = [line for line in DATA.read_text().splitlines() if not line.startswith("#")]
    header = lines[0].split(",")
    missing = sorted(set(_COLUMNS) - set(header))
    if missing:
        raise KeyError(f"{DATA.name} lacks column(s) {missing}; header is {header}")
    table = np.array([[float(cell) for cell in line.split(",")] for line in lines[1:]])
    return {_COLUMNS[name]: table[:, header.index(name)] for name in _COLUMNS}


def _corners(duty_p: float, duty_n: float, peak: float):
    """MagNet's piecewise-linear B(t) over one normalised period.

    Triangle when duty_p + duty_n = 1; otherwise a trapezoid whose two
    'flat' parts last (1 - duty_p - duty_n)/2 each and whose levels follow
    from equal volt-seconds (MagNet src/magnet/core.py, bdata_generation).
    """

    if abs(duty_p + duty_n - 1.0) < 1e-9:
        return [0.0, duty_p, 1.0], [-peak, peak, -peak]
    duty_0 = (1.0 - duty_p - duty_n) / 2.0
    if duty_p > duty_n:
        b_p = peak
        b_n = -b_p * ((-1.0 - duty_p + duty_n) * duty_n) / ((1.0 - duty_p + duty_n) * duty_p)
    else:
        b_n = peak
        b_p = -b_n * ((1.0 - duty_p + duty_n) * duty_p) / ((-1.0 - duty_p + duty_n) * duty_n)
    return (
        [0.0, duty_p, duty_p + duty_0, 1.0 - duty_0, 1.0],
        [-b_p, b_p, b_n, -b_n, -b_p],
    )


def _fit_sinusoids(data, temperature):
    """k, alpha, beta per band from the sinusoidal points at one temperature."""

    sine = (data["duty_p"] < 0) & (data["temperature"] == temperature)
    fits = []
    for low, high in ((0.0, BAND_SPLIT_HZ), (BAND_SPLIT_HZ, np.inf)):
        rows = sine & (data["frequency"] >= low) & (data["frequency"] < high)
        assert rows.sum() >= 30, "too few sinusoidal points to fit a band"
        design = np.column_stack(
            [np.ones(rows.sum()), np.log(data["frequency"][rows]), np.log(data["b_peak"][rows])]
        )
        coefficients, *_ = np.linalg.lstsq(design, np.log(data["loss"][rows]), rcond=None)
        fits.append((low, high, float(np.exp(coefficients[0])), *map(float, coefficients[1:])))
    return fits


@pytest.fixture(scope="module")
def errors():
    data = _load()
    results = {}
    for temperature in (25.0, 90.0):
        fits = _fit_sinusoids(data, temperature)
        rows = np.nonzero((data["duty_p"] >= 0) & (data["temperature"] == temperature))[0]
        for index in rows:
            frequency = data["frequency"][index]
            k, alpha, beta = next(fit[2:] for fit in fits if fit[0] <= frequency < fit[1])
            fractions, flux = _corners(data["duty_p"][index], data["duty_n"][index], data["b_peak"][index])
            time = np.asarray(fractions) / frequency
            predicted = igse_volumetric_loss(time, np.asarray(flux), k=k, alpha=alpha, beta=beta)[
                "volumetric_loss_w_per_m3"
            ]
            naive = k * frequency**alpha * data["b_peak"][index] ** beta
            measured = data["loss"][index]
            shape = (
                "triangle"
                if abs(data["duty_p"][index] + data["duty_n"][index] - 1.0) < 1e-9
                else "trapezoid"
            )
            bucket = results.setdefault((temperature, shape), {"igse": [], "naive": []})
            bucket["igse"].append(abs(predicted - measured) / measured)
            bucket["naive"].append(abs(naive - measured) / measured)
    return {key: {name: np.asarray(values) for name, values in value.items()} for key, value in results.items()}


def test_every_group_has_enough_points(errors):
    assert set(errors) == {(t, s) for t in (25.0, 90.0) for s in ("triangle", "trapezoid")}
    for key, bucket in errors.items():
        assert bucket["igse"].size >= 400, key


@pytest.mark.parametrize("temperature", [25.0, 90.0])
@pytest.mark.parametrize("shape", ["triangle", "trapezoid"])
def test_igse_predicts_measured_nonsinusoidal_loss(errors, temperature, shape):
    igse = errors[(temperature, shape)]["igse"]
    assert np.median(igse) <= MEDIAN_LIMIT, (temperature, shape, np.median(igse))
    assert np.quantile(igse, 0.95) <= P95_LIMIT, (temperature, shape, np.quantile(igse, 0.95))


@pytest.mark.parametrize("temperature", [25.0, 90.0])
@pytest.mark.parametrize("shape", ["triangle", "trapezoid"])
def test_igse_beats_shape_blind_steinmetz(errors, temperature, shape):
    bucket = errors[(temperature, shape)]
    assert np.median(bucket["igse"]) < np.median(bucket["naive"])
    assert np.quantile(bucket["igse"], 0.95) < np.quantile(bucket["naive"], 0.95)


@pytest.mark.parametrize("temperature", [25.0, 90.0])
def test_gate_rejects_the_shape_blind_answer(errors, temperature):
    """The gate must be able to fail: the naive Steinmetz answer does."""

    naive = errors[(temperature, "triangle")]["naive"]
    assert np.median(naive) > MEDIAN_LIMIT or np.quantile(naive, 0.95) > P95_LIMIT
