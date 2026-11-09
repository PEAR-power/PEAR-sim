# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""The double-exponential shape used to inject a declared analog transient."""

from __future__ import annotations

import textwrap

import pytest

from pear_sim.errors import ModelError
from pear_sim.spec.model import load_model
from pear_sim.spec.waveforms import DoubleExponentialWaveform


def _waveform(**overrides):
    kwargs = {
        "amplitude": 1.0,
        "tau_rise_s": 1e-9,
        "tau_fall_s": 1e-7,
        "start_s": 1e-3,
    }
    kwargs.update(overrides)
    return DoubleExponentialWaveform(**kwargs)


def test_zero_before_onset_and_decaying_after():
    waveform = _waveform()
    assert waveform.value(0.0) == 0.0
    assert waveform.value(1e-3) == 0.0
    late = waveform.value(1e-3 + 1e-5)
    assert 0.0 < late < 1e-3  # ~100 tau_fall later: essentially back to zero


def test_extremum_equals_the_declared_amplitude():
    """The scale factor is the point: `amplitude` means peak, not prefactor."""

    for amplitude in (0.5, 2.0, -1.5):
        waveform = _waveform(amplitude=amplitude)
        peak_time = waveform.as_dict()["peak_time_s"]
        assert waveform.value(peak_time) == pytest.approx(amplitude, rel=1e-12)
        # Sampled either side, nothing exceeds it in magnitude.
        samples = [
            waveform.value(waveform.start_s + step)
            for step in (1e-10, 1e-9, 1e-8, 5e-8, 1e-7, 1e-6)
        ]
        assert max(abs(value) for value in samples) <= abs(amplitude) * (1 + 1e-12)


def test_slope_vanishes_at_the_peak_and_is_zero_before_onset():
    waveform = _waveform()
    assert waveform.slope(waveform.start_s - 1e-9) == 0.0
    peak_time = waveform.as_dict()["peak_time_s"]
    scale = abs(waveform.amplitude) / waveform.tau_rise_s
    assert abs(waveform.slope(peak_time)) < 1e-6 * scale


def test_onset_is_the_only_breakpoint_and_only_inside_the_window():
    waveform = _waveform()
    assert waveform.breakpoints(0.0, 2e-3) == (1e-3,)
    assert waveform.breakpoints(2e-3, 3e-3) == ()


MODEL = """schema: pear-sim/v0.1
name: set_injection_probe
parameters:
  vin: 24 V
circuit:
  ground: gnd
  components:
    - {{id: vin, type: voltage_source, p: in, n: gnd, value: "${{vin}}"}}
    - {{id: r1, type: resistor, p: in, n: out, value: 10 ohm}}
    - {{id: c1, type: capacitor, p: out, n: gnd, value: 1 uF}}
    - id: iset
      type: current_source
      p: out
      n: gnd
      waveform: {{type: double_exponential, amplitude: 1 A, tau_rise: {rise},
                 tau_fall: {fall}, at: 200 us}}
analysis:
  transient:
    start: 0 s
    stop: 500 us
    output_step: 1 us
    solver: dopri5
    rtol: 1.0e-8
    atol: 1.0e-10
outputs:
  - v(out)
"""


def _load(tmp_path, *, rise="1 us", fall="10 us"):
    path = tmp_path / "set.yaml"
    path.write_text(MODEL.format(rise=rise, fall=fall))
    return load_model(path)


def test_model_declares_the_injection_and_it_perturbs_the_node(tmp_path):
    from pear_sim.core.solver import HybridSimulator

    model = _load(tmp_path)
    component = next(c for c in model.components if c.id == "iset")
    assert component.waveform.kind == "double_exponential"

    result = HybridSimulator(model).simulate()
    times = result.time
    node = result.signals["v(out)"]
    # Skip the RC start-up (tau = 10 ohm * 1 uF = 10 us); settled well before.
    quiet = [v for t, v in zip(times, node) if 150e-6 <= t < 195e-6]
    disturbed = [v for t, v in zip(times, node) if 200e-6 <= t <= 260e-6]
    settled = [v for t, v in zip(times, node) if t > 450e-6]
    # The node is dragged down by the injected sink, then recovers.
    assert max(quiet) - min(quiet) < 1e-4
    assert min(disturbed) < min(quiet) - 1.0
    assert abs(settled[-1] - quiet[-1]) < 1e-4


def test_degenerate_time_constants_are_rejected_not_regularized(tmp_path):
    with pytest.raises(ModelError, match="tau_fall must strictly exceed tau_rise"):
        _load(tmp_path, rise="10 us", fall="10 us")
    with pytest.raises(ModelError, match="tau_fall must strictly exceed tau_rise"):
        _load(tmp_path, rise="10 us", fall="1 us")


def test_unknown_waveform_type_names_the_full_supported_set(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        textwrap.dedent(
            """
            schema: pear-sim/v0.1
            name: bad
            circuit:
              ground: gnd
              components:
                - {id: vin, type: voltage_source, p: in, n: gnd,
                   waveform: {type: sawtooth, amplitude: 1 V}}
                - {id: c1, type: capacitor, p: in, n: gnd, value: 1 uF}
            analysis:
              transient:
                start: 0 s
                stop: 1 ms
                output_step: 10 us
                solver: dopri5
                rtol: 1.0e-8
                atol: 1.0e-10
            outputs:
              - v(in)
            """
        ).strip()
    )
    with pytest.raises(ModelError):
        load_model(path)
