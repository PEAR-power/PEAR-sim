"""Machine-readable, fail-closed capability declarations for agent clients."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITIES_SCHEMA = "pear-sim/capabilities/v0.1"
CAPABILITY_STATUSES = ("supported", "experimental", "unsupported")
MAX_CAPABILITY_ID_LENGTH = 256


_CAPABILITY_RECORDS: tuple[dict[str, Any], ...] = (
    {
        "id": "analysis.emi_conducted",
        "status": "unsupported",
        "scope": "conducted or radiated EMI, LISN spectra, compliance limits (CISPR, EN)",
        "statement": (
            "Not modelled. Switches change state instantaneously (infinite dv/dt), there are "
            "no parasitic capacitances, no common-mode paths and no layout, so a spectrum "
            "computed from these waveforms says nothing reliable about emissions or "
            "compliance — at high frequency it is dominated by the idealization itself. "
            "Do not derive dBuV levels or pass/fail verdicts from runs of this tool."
        ),
        "evidence": [],
    },
    {
        "id": "analysis.gradient_backend_step1",
        "status": "experimental",
        "scope": (
            "declared-sequence DC-source converters (buck-class AND "
            "transformer-coupled: DAB, half-bridge LLC as a vertical slice), "
            "scheduled edges AND natural commutation (saltation); integral "
            "metrics (mean, rms), device-card loss power (open thermal loop) "
            "and their theta-sensitivities; NOT covered: mode-count changes "
            "across theta (a Jacobian discontinuity, refused via the "
            "declared-sequence flag, not smoothed)"
        ),
        "statement": (
            "Direction, not verdict: the affine period map of a DECLARED mode "
            "sequence is evaluated exactly (matrix exponentials, one linear solve "
            "for the fixed point) and differentiated by central differences on the "
            "compilation step. A violated declaration returns as an assumption "
            "flag with a signed margin, never an exception; the oracle remains "
            "the only issuer of pass. Natural commutation, losses and thermal "
            "terms are separate engine extensions and are refused here, not "
            "approximated."
        ),
        "evidence": [
            "T1: steady state vs the production shooting solver (2.2e-10 "
            "normalized) and vs the closed-form buck oracle (mean exact, ripple "
            "3.7e-16 relative)",
            "T2: Jacobian vs central finite differences on the oracle at "
            "periodic convergence, all five theta, 1e-3 gate; headline "
            "derivatives match closed forms (d mean(v_out)/d duty = Vin to "
            "3.4e-11, dead-time slope = -fsw*Vin)",
            "T3: DCM model under a declared CCM sequence returns "
            "declared_sequence_held=False with a negative margin, no exception",
            "saltation extension: all nine DCM derivatives vs oracle finite "
            "differences within 1e-8 of the gate; realized diode conduction "
            "0.8477 us matches the oracle event log; T4 dead-time slope "
            "-fsw*Vin to 7.3e-9; BCM coincidence deterministic and flagged; "
            "a declaration the physics never fires returns a flag, not a "
            "raise",
            "LLC/DAB vertical slice: DAB steady state vs the "
            "shooting solver 1.4e-9, d/d{phi,load,vin} within 3.6e-8 of the "
            "T2 criterion (the vin column regained its oracle when the "
            "constant-guard livelock was fixed); LLC solved "
            "on the invariant subspace of the structural star-invariant unit "
            "eigenvalue (the production shooting solver cannot run the model "
            "at all without invariants), d/d(lr) vs settled-run reference 7.3e-6 (mean)",
            "losses extension: P_loss vs the oracle's one-cycle artifact to "
            "1.4e-4 relative; dP_loss/d{duty,load,fsw} vs oracle finite "
            "differences within the 1e-3 gate (duty at the grid-commensurate "
            "step the artifact's sample anchoring requires); clamped table "
            "lookups surface as flags with signed edge distances; PCHIP "
            "smoothing engine-only and declared; Jacobian labelled "
            "'no d/dT_j term'",
        ],
    },
    {
        "id": "analysis.harmonics.event_aligned_fourier",
        "status": "experimental",
        "scope": (
            "explicit integer-period signal requests through harmonic order 256 on the "
            "reference transient backend"
        ),
        "statement": (
            "Fourier coefficients and bandwidth-limited THD are integrated from dense "
            "output on event-aligned topology intervals; no independent reference "
            "acceptance is implied. Three-phase semantics add sequence components; the "
            "`generic` semantic accepts any signal (e.g. DC-side ripple) and reports "
            "amplitudes only, without THD."
        ),
        "evidence": ["synthetic Fourier oracle and output-grid-independence tests"],
    },
    {
        "id": "analysis.resonance.modal_audit",
        "status": "experimental",
        "scope": (
            "oscillatory eigenvalues of every accepted hybrid mode, clustered "
            "across modes and placed against the declared modulation lines"
        ),
        "statement": (
            "Reports each network resonance with its damping ratio, the share "
            "of the run whose modes contain it, and its separation from the "
            "nearest carrier-band or fundamental line. Damping comes only from "
            "modelled resistances and is therefore a LOWER bound; baseband "
            "harmonics injected by dead time are not enumerated. A flag marks "
            "a design question, never a proven instability."
        ),
        "evidence": ["closed-form LC filter poles and a carrier-coincident regression"],
    },
    {
        "id": "analysis.small_signal.ac_sweep_switched",
        "status": "experimental",
        "scope": (
            "duty-to-output response of the SWITCHED model: a sinusoidal "
            "duty perturbation integrated by the production solver and "
            "recovered by single-bin correlation over an integer number of "
            "perturbation periods, below the carrier's Nyquist rate"
        ),
        "statement": (
            "Answers what the actual switching circuit does when its duty is "
            "wiggled, as distinct from what its averaged model predicts. The "
            "two are checked AGAINST EACH OTHER: agreement inside the "
            "averaging's validity band is a gate, and the divergence above it "
            "is the measurement — the switched result carries the modulator's "
            "transport lag, which no averaged model contains. Duty-to-output "
            "only; the perturbation enters as duty, so plant-to-output and "
            "load-to-output paths are NOT covered. An amplitude sweep is "
            "offered so a caller can see whether the answer depends on the "
            "excitation, which is the only honest test of small-signal "
            "validity."
        ),
        "evidence": [
            "canonical buck at 20 kHz: magnitude agrees with the averaged "
            "model to 0.000 % at 100 Hz and 300 Hz and 0.004 % at 1 kHz — two "
            "independent derivations, one number",
            "the phase residual tracks a transport lag: 0.77-0.83 of half a "
            "carrier period across two decades, always lagging",
            "divergence grows where averaging expires (0.13 % at f_sw/10, "
            "4.4 % at f_sw/4), which is the averaged model failing, not this one",
            "the correlator recovers a synthetic tone's amplitude to 1e-4 and "
            "its phase to 0.05 deg, and returns zero for a pure DC input",
            "NOT compared against an independent engine's AC sweep: no independent-engine comparison exists, so this carries no compatibility claim",
        ],
    },
    {
        "id": "analysis.small_signal.loop_gain_declared_compensator",
        "status": "experimental",
        "scope": (
            "CCM cycles of one PWM leg (a pwm gate, or a complementary pair whose "
            "dead-time intervals are averaged in) at a converged shooting operating point, "
            "frequencies below switching_frequency/10, caller-declared "
            "compensator and sensor gain"
        ),
        "statement": (
            "State-space-averaged control-to-output response with loop-gain "
            "phase/gain margins retained as blocking evidence gates; margins "
            "attest the declared design only and are not a closed-loop "
            "simulation or a stability proof outside the averaged model."
        ),
        "evidence": [
            "closed-form ideal-buck transfer-function comparison",
            "complementary leg with dead time (three topologies per cycle) against the "
            "closed-form ideal-buck G_vd(s) to 1e-9 relative",
            "stability-evidence gate tests, including an unbounded gain margin when the "
            "sweep covers the validity band and the phase never reaches -180 deg",
            "a loop crossing 0 dB twice (buck LC resonance lifted above 0 dB) reports the "
            "worse crossover's margin (~53 deg), not the first one's (~152 deg)",
        ],
    },
    {
        "id": "analysis.steady_state.shooting",
        "status": "experimental",
        "scope": (
            "PWM models with one shared switching frequency on the reference "
            "transient backend, including natural-commutation cycles and "
            "inductor-star topologies (series chains, floating stars): the "
            "solve is confined to the kernel's star invariants with levels "
            "pinned from the declared initial state"
        ),
        "statement": (
            "Newton shooting on the production one-period cycle map with "
            "retained convergence status; a non-converged result is retained "
            "evidence, not a silent fallback. Star-invariant levels are "
            "pinned, never projected, and recorded in the result; a cycle "
            "that does not conserve an initial-topology invariant is refused "
            "with a named error."
        ),
        "evidence": [
            "buck/boost CCM and DCM fixed points vs the independent periodic oracles"
        ],
    },
    {
        "id": "analysis.transient.reference_python",
        "status": "supported",
        "scope": "small independent-state RLC networks within documented topology limits",
        "statement": "Adaptive transient integration with explicit requested output grids.",
        "evidence": ["VAL-001 partial", "cross-platform CI contract"],
    },
    {
        "id": "artifacts.gate_drive_observation",
        "status": "supported",
        "scope": "explicit PWM gate-to-device mapping and bounded switching timeline",
        "statement": (
            "Records atomic scheduled transitions and intrinsic-interlock observations "
            "without claiming physical gate-driver timing or safety."
        ),
        "evidence": ["gate-drive artifact, manifest, and report regression suite"],
    },
    {
        "id": "artifacts.offline_report_and_review",
        "status": "supported",
        "scope": "deterministic SVG, offline HTML, and hash-bound review packets",
        "statement": "Human-facing projections are generated from immutable run artifacts.",
        "evidence": ["report and review regression suites"],
    },
    {
        "id": "artifacts.reproducible_run_bundle",
        "status": "supported",
        "scope": "reference-python transient runs",
        "statement": "Run inputs, outputs, warnings, provenance, and artifact hashes are recorded.",
        "evidence": ["run bundle integrity regression suite"],
    },
    {
        "id": "artifacts.run_comparison",
        "status": "supported",
        "scope": "two manifest-verified PEARSim v0.1 run bundles with identical physics semantics",
        "statement": (
            "Produces a fail-closed, event-aware numerical comparison without signal "
            "remapping, sign correction, gain fitting, or unconstrained time alignment."
        ),
        "evidence": ["run-comparison contract and regression suite"],
    },
    {
        "id": "artifacts.three_phase_modulation_observation",
        "status": "supported",
        "scope": (
            "explicit SPWM phase-to-gate-to-device mapping, bounded command projection, "
            "and source-bound switching evidence"
        ),
        "statement": (
            "Projects producer-recorded carrier cells and realized gate events for human "
            "review without recomputing modulation or validation verdicts in the report."
        ),
        "evidence": ["three-phase modulation artifact, manifest, and report regression suite"],
    },
    {
        "id": "component.machine.driven_shaft_speed",
        "status": "experimental",
        "scope": (
            "mechanical speed imposed from outside (constant or linear ramp) "
            "instead of integrated, so the torque is the measurement"
        ),
        "statement": (
            "The declared profile is READ at each segment boundary and HELD "
            "across the segment, the same scheme the integrated speed uses, so "
            "the back-EMF stays a pure sinusoid. Two consequences are measured "
            "rather than asserted: the speed lags a ramp by at most the "
            "profile's drift across one segment, and the angle is a left "
            "Riemann sum of the ramp. Independence from torque holds up to the "
            "segment grid — natural-commutation instants depend on the "
            "currents, so a different torque history changes WHEN the profile "
            "is sampled, never WHAT it returns. Fields that would have no "
            "effect (inertia, friction, load torque, initial speed) are "
            "refused, not ignored."
        ),
        "evidence": [
            "speed tracks the declared ramp to 5.24e-2 rad/s, equal to the "
            "profile drift over one 10 us segment (derived bound, not tuned)",
            "final angle 15.7050 rad vs 15.7080 rad analytic, the predicted "
            "left-Riemann undershoot",
            "a 27 N*m change of torque moves the speed by 1.96e-2 rad/s "
            "(1.25e-4 relative)",
            "NOT compared against an independent engine: no independent-engine comparison exists, so this carries no compatibility claim",
        ],
    },
    {
        "id": "component.machine.pmsm_non_salient",
        "status": "experimental",
        "scope": (
            "surface-magnet synchronous machines with sinusoidal back-EMF and "
            "an isolated neutral, expanded into three auditable stator branches "
            "plus one mechanical state"
        ),
        "statement": (
            "The declaration expands into ordinary components; the mechanical "
            "speed is HELD within one integration segment while the rotor angle "
            "advances exactly, so the electrical side stays strictly linear and "
            "the splitting error shrinks with the segment length (asserted by a "
            "convergence test). Saliency, saturation, iron loss, cogging and "
            "non-sinusoidal back-EMF are out of scope and fail closed where "
            "declarable."
        ),
        "evidence": [
            "closed-form terminal current and braking torque",
            "power identity sum(e*i) == T*omega",
            "sequence-sensitive motoring regression",
            "NOT compared against an independent engine: no independent-engine comparison exists, so this carries no compatibility claim — "
            "one reason this stays experimental",
        ],
    },
    {
        "id": "component.machine.salient_rotor",
        "status": "experimental",
        "scope": (
            "three-phase synchronous machine with unequal d- and q-axis "
            "inductances, including the zero-magnet (synchronous reluctance) "
            "case; linear axis inductances only — no saturation, no flux maps"
        ),
        "statement": (
            "The windings expand into ONE magnetically coupled group whose "
            "inductance matrix depends on rotor position and is HELD across "
            "each integration segment, the same scheme the speed uses, so the "
            "electrical side stays strictly linear inside a segment. Torque "
            "carries both terms, T = 1.5*p*[psi_f*i_q + (Ld-Lq)*i_d*i_q]; a "
            "machine with no magnets produces the second alone. The "
            "zero-sequence entry of the matrix is singular by construction "
            "and is set explicitly because the kernel inverts the matrix; "
            "with a floating star it is unobservable and that is TESTED, not "
            "asserted. Saturation and flux maps are NOT modelled, so a real "
            "reluctance machine — which works deep in saturation — is "
            "represented only to the extent a linear model can."
        ),
        "evidence": [
            "a Park transform reads back the declared Ld and Lq at every "
            "rotor angle to 2.6e-18 H, in both saliency directions, and "
            "collapses to the constant matrix when the axes are equal",
            "a machine with psi_f = 0 produces torque matching the closed "
            "form 1.5*p*(Ld-Lq)*i_d*i_q to 1e-9 relative; under the previous "
            "model that machine produced exactly zero",
            "scaling the unobservable zero-sequence entry by 10 moves current "
            "and torque by 1e-15 relative",
            "the non-salient machine is byte-identical to before the change "
            "(waveforms, metrics and machine-trace), so nothing shipped moved",
            "cost measured BEFORE building: mode recompilation 0.102 ms x "
            "7537 segments = 0.77 s on a 9.3 s run, an 8 % overhead",
            "the segment-boundary inductance-matrix swap preserves FLUX, not "
            "current (the discarded term would be the reluctance voltage "
            "omega*dL/dtheta*i); regression: flux linkage continuous across "
            "every swap",
            "NOT compared against an independent engine: no independent-engine comparison exists, so this carries no compatibility claim",
        ],
    },
    {
        "id": "component.machine.saturating_flux_map",
        "status": "experimental",
        "scope": (
            "machine flux linkage declared as psi_d(i_d,i_q) and "
            "psi_q(i_d,i_q) on a current grid, with the inductance matrix "
            "rebuilt from the map's LOCAL SLOPES at each segment's operating "
            "point and torque taken from the general form"
        ),
        "statement": (
            "Saturation and cross-saturation, which constant axis "
            "inductances cannot express at all. The kernel uses d(psi)/d(i), "
            "not the chord psi/i, because that is what multiplies di/dt in "
            "the circuit equation — the chord gives the right flux and the "
            "wrong dynamics, which looks correct in steady state. Flux must "
            "increase with its own axis current, checked at load time. Maps "
            "are DECLARED, never derived from geometry: a real one has to "
            "come from field or bench identification, and the shipped "
            "example is explicitly synthetic."
        ),
        "evidence": [
            "a LINEAR map reproduces the constant-Ld/Lq machine exactly: "
            "inductance matrix to 6.3e-18 H at every rotor angle, and the "
            "whole closed-loop drive run to 8.9e-13 (0.0000 %)",
            "mapped torque collapses to psi_f*i_q + (Ld-Lq)*i_d*i_q when the "
            "flux is linear, to 1e-9 relative",
            "saturation measurably changes the answer: at the SAME current "
            "operating point the shipped map yields 7.67 N*m against the "
            "linear twin's 7.85 N*m (-2.3 %)",
            "NOT compared against an independent engine: no independent-engine comparison exists, so this carries no compatibility claim",
        ],
    },
    {
        "id": "control.complementary_pwm_dead_time",
        "status": "experimental",
        "scope": (
            "fixed-frequency two-output PWM with intrinsic interlock and bounded dead time"
        ),
        "statement": (
            "The outgoing gate turns off immediately and the incoming gate turns on after "
            "the declared dead time; coincident zero-dead-time swaps are atomic."
        ),
        "evidence": ["complementary-PWM independent schedule oracle and regression suite"],
    },
    {
        "id": "control.field_oriented_torque_controller",
        "status": "experimental",
        "scope": (
            "field-oriented torque control expressed entirely in the declared "
            "signal layer: Park transform on the rotor angle, dq regulators "
            "with back-calculation anti-windup, flagged decoupling, circular "
            "voltage limit with dead-time margin, one-tick angle prediction, "
            "inverse Park into the space-vector bridge"
        ),
        "statement": (
            "The signal layer's acceptance criterion met on a REAL controller: "
            "the control law is data, with no line in src/. The transforms "
            "share the machine's frame convention by construction and by "
            "test, because a controller on a different convention produces a "
            "plausible wrong torque no aggregate metric reveals. NOT "
            "implemented and not implied: field weakening (the references are "
            "constant) and flux maps INSIDE the regulation loop — lookup_2d "
            "exists, but a real psi(id,iq) map must come from machine "
            "identification and inventing one would be worse than its "
            "absence, so the regulators work on currents rather than flux. "
            "That is the one structural difference from the source material."
        ),
        "evidence": [
            "reaches its references from zero current: id -6 A and iq +12 A "
            "within 0.05 A",
            "the integral term is what closes the error, as theory requires: "
            "ki 0.02 -> -0.70 A, 0.10 -> -0.14 A, 0.30 -> -0.0014 A",
            "torque recomputed from the currents the CONTROLLER measured "
            "equals what the MACHINE reported (8.144 N*m both ways)",
            "the circular voltage limit is never exceeded; a per-axis clamp "
            "would have admitted sqrt(2) times the bound",
            "turning the decoupling flag off changes the q-axis command by "
            "more than 1 V, so the term is genuinely wired in",
            "NOT compared against an independent engine: no independent-engine comparison exists, so this carries no compatibility claim",
        ],
    },
    {
        "id": "control.signal_layer_declared_blocks",
        "status": "experimental",
        "scope": (
            "one periodic task of declared discrete blocks (constant, probe, "
            "time_source, gain, sum, product, saturate, unit_delay, switch, "
            "lookup_1d, compare, logic, select) driving a signal_pwm gate "
            "bridge"
        ),
        "statement": (
            "A control law expressed AS DATA rather than as a compiled type. "
            "Execution order is derived from the graph and retained in the "
            "bundle; a cycle through combinational blocks is refused with the "
            "cycle named, while a cycle through unit_delay is a difference "
            "equation and is legal. Units are declared per block and a probe "
            "may not relabel what it reads. The graph enters the physics hash, "
            "so two control laws cannot share one. Conditions carry the "
            "declared unit 'bool' and nothing else may reach a condition "
            "input, which is what makes conditional-integration anti-windup "
            "expressible without a dimensionless signal silently becoming a "
            "control decision; the check runs one way only, since a truth "
            "value feeding arithmetic is a legitimate 0/1 gate. Equality is "
            "not offered: two floats reached by different arithmetic are "
            "equal only by luck. NOT expressible and named rather than "
            "implied: multi-rate tasks, and any block that would need its own "
            "integrator."
        ),
        "evidence": [
            "the compiled sampled_pi_pwm law rebuilt entirely from declared "
            "blocks on the same circuit and gains reproduces it to 1.0e-7 V on "
            "12.4 V (8.2e-9 relative) and 1.4e-7 A, i.e. the integrator's own "
            "noise floor at rtol 1e-9 — including the 25 ms load step",
            "execution order proven independent of file order",
            "a delay returning its own previous value instead of its input's "
            "was caught end-to-end (duty pinned to its lower limit for a whole "
            "run) and is now pinned by a unit test",
            "NO comparison against an external block-diagram tool: no such "
            "oracle record exists, so this carries no compatibility claim",
        ],
    },
    {
        "id": "control.signal_space_vector_bridge",
        "status": "experimental",
        "scope": (
            "three-phase bridge whose commanded VOLTAGE VECTOR comes from the "
            "declared signal layer; six gates with intrinsic interlock and "
            "dead time, one task tick per carrier cycle, linear range only"
        ),
        "statement": (
            "The modulator a field-oriented controller needs: it takes the "
            "vector an inverse Park produces, not a duty, so the modulator "
            "can be swapped without rewriting the control law. Sector "
            "and dwell times are recorded per tick. Overmodulation is refused "
            "rather than clipped. A shoot-through is reported as a NAMED "
            "interlock violation rather than as a singular circuit, which is "
            "the lesson of the bridge rail-to-rail short regression as much as "
            "the fix is."
        ),
        "evidence": [
            "commanding alpha = 200 V yields a mean phase voltage across the "
            "load of 200.0000 V (5.7e-14 error, continuous-time quadrature) — "
            "the bridge reproduces what it was asked for, not merely 'switches'",
            "pole voltage matches the commanded duty exactly: 150.000 V for "
            "duty 0.75 on a +/-300 V rail",
            "the sector duties themselves agree with min-max injection to "
            "8.9e-16 over 14 440 points (see the modulator's own entry)",
            "rail-to-rail short regression shipped as "
            "examples/inverter_signal_svpwm.yaml",
            "NOT compared against an independent engine's SVPWM, and no THD "
            "criterion is declared or measured, so this carries no "
            "compatibility claim and does not lift the inverter capability",
        ],
    },
    {
        "id": "control.three_phase_npc_spwm_level_shifted_pd",
        "status": "experimental",
        "scope": (
            "synchronous offline-midpoint-sampled three- and five-level "
            "diode-clamped modulation with level-shifted in-phase (PD) carrier "
            "bands (12 or 24 interlocked gate outputs), intrinsic dead time and "
            "declared minimum-pulse suppression"
        ),
        "statement": (
            "The [-1, 1] reference range is split into levels-1 stacked carrier "
            "bands driving one complementary pair per band and leg; "
            "band-transition hand-overs and dead-time-swallowed band-edge pulses "
            "are recorded per carrier cell, and the staircase gating invariant "
            "is a runtime interlock, without claiming a causal "
            "digital-modulator implementation. The band law is level-generic; "
            "v0.1 validates three and five levels."
        ),
        "evidence": ["NPC modulation-law recompute contract and regression suite"],
    },
    {
        "id": "control.three_phase_spwm_offline_midpoint_sampled",
        "status": "experimental",
        "scope": (
            "synchronous offline-midpoint-sampled, center-aligned three-phase sinusoidal PWM in "
            "the linear range with explicit abc/acb sequence and intrinsic dead time"
        ),
        "statement": (
            "A bounded integer carrier ratio is expanded to six interlocked gate outputs; "
            "commanded carrier cells and realized transitions are retained separately, "
            "without claiming a causal digital-modulator implementation."
        ),
        "evidence": ["SPWM independent schedule oracle and regression suite"],
    },
    {
        "id": "converter.boost.bcm.canonical_ideal",
        "status": "experimental",
        "scope": "exact canonical single-switch ideal boost boundary point",
        "statement": (
            "A periodic affine oracle and retained-run sweep exercise the numerical "
            "boundary between continuous and discontinuous conduction."
        ),
        "evidence": ["VAL-003 partial", "canonical boost boundary sweep v0.1"],
    },
    {
        "id": "converter.boost.ccm.canonical_ideal",
        "status": "experimental",
        "scope": "exact canonical single-switch ideal boost at declared CCM load ratios",
        "statement": (
            "Periodic boundary states, mean output, state ripple, modes, events, and "
            "energy balance are checked against an independent affine oracle."
        ),
        "evidence": ["VAL-002 partial", "VAL-003 partial", "canonical boost boundary sweep v0.1"],
    },
    {
        "id": "converter.boost.dcm.canonical_ideal",
        "status": "experimental",
        "scope": "exact canonical single-switch ideal boost at declared DCM load ratios",
        "statement": (
            "Natural diode turn-off, zero-current idle behavior, state ripple, and event "
            "semantics are checked in retained one-period runs."
        ),
        "evidence": ["VAL-003 partial", "canonical boost boundary sweep v0.1"],
    },
    {
        "id": "converter.buck.bcm.canonical_ideal",
        "status": "experimental",
        "scope": "exact canonical single-switch ideal buck boundary point",
        "statement": (
            "A periodic affine oracle and a retained-run sweep exercise coincident "
            "natural and scheduled commutation at the independently found boundary."
        ),
        "evidence": ["VAL-003 partial", "canonical buck boundary sweep v0.1"],
    },
    {
        "id": "converter.buck.ccm.ideal",
        "status": "experimental",
        "scope": "canonical single-switch buck in strict CCM",
        "statement": "Analytic self-checks cover selected steady-state relations only.",
        "evidence": ["VAL-002 partial"],
    },
    {
        "id": "converter.buck.dcm.canonical_ideal",
        "status": "experimental",
        "scope": "canonical single-switch buck across declared DCM load ratios",
        "statement": (
            "Natural diode turn-off, zero-current reduction, and periodic DCM states "
            "are checked across the retained-run boundary sweep."
        ),
        "evidence": ["VAL-003 partial", "canonical buck boundary sweep v0.1"],
    },
    {
        "id": "converter.buck_boost",
        "status": "unsupported",
        "scope": "general buck-boost schematics and operating regimes",
        "statement": (
            "Only the separately declared exact canonical ideal inverting slice is "
            "implemented; no general buck-boost capability is claimed."
        ),
        "evidence": [],
    },
    {
        "id": "converter.buck_boost.inverting.bcm.canonical_ideal",
        "status": "experimental",
        "scope": "exact canonical single-switch ideal inverting buck-boost boundary point",
        "statement": (
            "A periodic affine oracle and retained-run sweep exercise the numerical "
            "boundary between continuous and discontinuous conduction."
        ),
        "evidence": ["VAL-002 partial", "VAL-003 partial", "canonical boundary sweep v0.1"],
    },
    {
        "id": "converter.buck_boost.inverting.ccm.canonical_ideal",
        "status": "experimental",
        "scope": "exact canonical single-switch ideal inverting buck-boost in declared CCM",
        "statement": (
            "Signed output polarity, periodic states, ripple, modes, and events are checked "
            "against an independent affine oracle; energy uses the full-cycle physics audit."
        ),
        "evidence": ["VAL-002 partial", "VAL-003 partial", "canonical boundary sweep v0.1"],
    },
    {
        "id": "converter.buck_boost.inverting.dcm.canonical_ideal",
        "status": "experimental",
        "scope": "exact canonical single-switch ideal inverting buck-boost in declared DCM",
        "statement": (
            "Natural diode turn-off, constrained zero-current idle behavior, negative "
            "output polarity, and event semantics are checked in retained one-period runs."
        ),
        "evidence": ["VAL-003 partial", "canonical boundary sweep v0.1"],
    },
    {
        "id": "converter.dc_dc.bcm_general",
        "status": "unsupported",
        "scope": "general boundary-conduction operation",
        "statement": (
            "No generalized BCM semantics or accepted sweep exists outside the exact "
            "canonical buck, boost, and inverting buck-boost boundary slices."
        ),
        "evidence": [],
    },
    {
        "id": "converter.dc_dc.dcm_general",
        "status": "unsupported",
        "scope": (
            "topologies outside the canonical buck, boost, and inverting buck-boost slices"
        ),
        "statement": "General descriptor/DAE reduction and complementarity are not implemented.",
        "evidence": [],
    },
    {
        "id": "converter.inverter_three_phase.six_step_ideal",
        "status": "experimental",
        "scope": "two-level six-step bridge with the executable example load",
        "statement": "Fundamental amplitude and phase-current balance have partial checks.",
        "evidence": ["VAL-005 partial"],
    },
    {
        "id": "converter.inverter_three_phase.spwm_midpoint_neutral_rl",
        "status": "experimental",
        "scope": (
            "the executable two-level, split-DC-bus balanced RL example with its neutral "
            "tied to the DC midpoint "
            "using offline-midpoint-sampled SPWM and explicit antiparallel diodes"
        ),
        "statement": (
            "Fundamental amplitude, phase sequence, intrinsic gate interlock, dense-output "
            "Fourier observation, and energy balance have narrow regression checks."
        ),
        "evidence": ["VAL-005 partial", "SPWM vertical slice"],
    },
    {
        "id": "converter.inverter_three_phase.spwm_svpwm_dead_time",
        "status": "unsupported",
        "scope": "complete SPWM/SVPWM inverter, dead-time sweep, and THD acceptance",
        "statement": (
            "Modulation itself exists — offline-midpoint-sampled SPWM and "
            "min-max (SVPWM-equivalent) reference injection over the linear "
            "range, with dead time. A SECTOR modulator now exists too "
            "(`pear_sim.core.svpwm`): explicit vector geometry, dwell times and "
            "sector observation, refusing overmodulation rather than clipping "
            "it. What is STILL not implemented is the capability this id "
            "claims. The modulator IS now wired to the kernel "
            "(`control.signal_space_vector_bridge`), but there "
            "is still no retained modulation/dead-time/load matrix and no "
            "independent THD acceptance. A working modulator without an "
            "acceptance matrix is not validated inverter support, so this "
            "stays unsupported."
        ),
        "evidence": [
            "sector duties agree with min-max injection to 8.9e-16 over "
            "14 440 points spanning the whole linear range and all angles — "
            "two independent derivations, one answer",
            "end-to-end since 2026-08-16: a shipped example drives a bridge "
            "through it and reproduces the commanded vector to 5.7e-14; what "
            "is still missing is the acceptance matrix and a THD criterion",
        ],
    },
    {
        "id": "electrical.general_descriptor_dae",
        "status": "unsupported",
        "scope": "dependent storage states, capacitor loops, and inductor cutsets",
        "statement": "Only narrow topology-specific converter zero-current paths are covered.",
        "evidence": [],
    },
    {
        "id": "electrothermal.loss_and_temperature",
        "status": "experimental",
        "scope": (
            "declared device cards (conduction LUT over current and "
            "temperature, switching energies at a declared DPT condition) and "
            "cooling paths (TIM, heatsink, fan operating point, liquid loop) "
            "evaluated as POST-PROCESSING at a declared junction temperature"
        ),
        "statement": (
            "Losses and junction temperatures are computed from recorded, "
            "auditable signals after the run — the kernel itself stays ideal, "
            "so device dissipation never feeds back into the switching "
            "solution. The electrothermal LOOP is therefore OPEN by decision: "
            "`tj_evaluation_c` is a declared constant and the Jacobian "
            "carries an explicit 'no d/dT_j term' label. Table lookups outside "
            "their declared axes surface as flags with signed edge distances, "
            "never as silent plateaus. Cards shipped in the repo are "
            "explicitly SYNTHETIC — there is no vendor part catalogue, and a "
            "loss number is only as good as the card it came from."
        ),
        "evidence": [
            "DC conduction golden to 1e-9; Foster network vs the closed-form "
            "exponential sum to 1e-9; fan operating point checked by hand",
            "ASTM E1049 rainflow counting against the canonical example "
            "(counting, deliberately not a life model)",
            "buck CCM loss totals and the liquid-cooling path "
            "(tests/test_losses_thermal.py, tests/test_liquid_cooling.py)",
        ],
    },
    {
        "id": "experiments.parameter_sweep.one_dimension",
        "status": "supported",
        "scope": "one declared model parameter, one to sixty-four retained transient runs",
        "statement": (
            "Executes a bounded parameter sweep in caller order, retains every point, and "
            "keeps execution status separate from optional expectation-based acceptance."
        ),
        "evidence": ["parameter-sweep contract and regression suite"],
    },
    {
        "id": "interchange.vendor_native_models",
        "status": "unsupported",
        "scope": "native model import or export for commercial simulators",
        "statement": (
            "Reading or writing the native model files of commercial simulators is a "
            "deliberate non-goal."
        ),
        "evidence": [],
    },
    {
        "id": "library.topology_comparison",
        "status": "experimental",
        "scope": (
            "per-finalist optimisation on the gradient engine with an oracle "
            "verdict at the landing point, a mode-structure trust-region "
            "gate, and a Pareto front over declared objectives"
        ),
        "statement": (
            "Each finalist is pulled to ITS OWN optimum — evaluating a field "
            "at one shared operating point silently crowns whichever "
            "topology that point suited. The gradient engine supplies "
            "direction only; every reported number comes from a full "
            "transient run at the landing point, with the gradient's own "
            "values kept beside them. The trust region is the mode "
            "STRUCTURE, never the step durations (moving those is what "
            "optimising theta does), and an unchecked region is never "
            "reported as a satisfied one. A candidate this kernel cannot run "
            "returns `unsupported` — a statement about the engine, not a "
            "defect of the candidate — and is listed with its reason rather "
            "than dropped. Objectives are never summed into a score; "
            "dominated points stay in the record. On an unevenly "
            "loss-instrumented field, candidates without device cards are "
            "flagged OPTIMISTIC, because a model with no cards reports no "
            "dissipation at all. A front position is not a recommendation "
            "and implies no hardware claim. NOT implemented: snapping "
            "transistor choices to real parts — there is no part catalogue, "
            "and snapping to an empty one would produce unbacked numbers."
        ),
        "evidence": [
            "buck optimised over L moves the inductance up and the ripple "
            "down with both tiers agreeing on the move to ~4e-7 relative "
            "(tests/test_topology_comparison.py)",
            "trust-region gate holds across a legitimate duration change and "
            "reports a real structure change; unsupported, dominated-point "
            "retention and the OPTIMISTIC rule each pinned by a test",
        ],
    },
    {
        "id": "library.topology_entries",
        "status": "experimental",
        "scope": (
            "declarative per-topology library entries (prerequisites, mode "
            "sequences with run provenance, mandatory known limitations) "
            "loaded fail-closed from data files"
        ),
        "statement": (
            "A topology entry is data, never code: its mode sequence "
            "deserializes into the exact ModeSequence the gradient engine "
            "consumes (one canonical serialization, no parallel notation), "
            "its provenance names the source run, and a stored cycle is "
            "audited by re-extraction from a fresh run — content agreement, "
            "not a hash, so a version bump cannot silently invalidate "
            "entries while a changed cycle still does. An entry without "
            "known limitations is refused: an empty list is a claim, not "
            "an absence of data."
        ),
        "evidence": [
            "buck entry drives the gradient engine directly and lands on "
            "the oracle physics (tests/test_topology_library.py)",
            "boost entry added purely as data; refusal matrix for the "
            "format's fail-closed rules",
        ],
    },
    {
        "id": "model.coupled_winding_zero_current_reduction",
        "status": "supported",
        "scope": "transformer secondaries opened by non-conducting switches",
        "statement": (
            "A coupled winding whose branch is opened is frozen out of the differential "
            "state and carries its INDUCED voltage, not 0 V; a star of coupled windings "
            "reduces with the mutual terms in the invariant's derivative. The twelve-pulse "
            "stack still needs declared winding shunts and is NOT covered."
        ),
        "evidence": [
            "six-pulse bridge on a floating transformer secondary with no shunts, "
            "agreeing with the shunted variant to 1e-5 on load current",
            "open-secondary induced voltage against the turns ratio to 1e-3",
        ],
    },
    {
        "id": "model.latching_thyristor",
        "status": "supported",
        "scope": "ideal gate-fired, zero-current-commutated latching branch",
        "statement": (
            "A thyristor closes only on a gate edge under forward bias and opens only at "
            "zero current, so conduction outlives the gate pulse. Ideal: no forward drop, "
            "no holding or latching current, no dv/dt firing and no blocking-recovery time, "
            "so forced-commutation timing margins are OUT of scope."
        ),
        "evidence": [
            "half-wave phase-controlled rectifier against the R-L closed form "
            "(extinction angle within 0.5 deg, trajectory within 0.018 % of peak)",
            "held-gate equivalence to ideal_switch + diode within 1e-9 A",
        ],
    },
    {
        "id": "model.piecewise_linear_conduction",
        "status": "experimental",
        "scope": (
            "on_resistance on ideal_switch and diode, forward_voltage on diode: a "
            "conducting device obeys v = V_f + R_on * i"
        ),
        "statement": (
            "Conduction drops are stamped into the circuit equations, so they shape the "
            "waveforms, the operating point and the damping, and the run's energy audit "
            "includes their dissipation. A blocking diode turns on when its voltage "
            "reaches V_f. The drop is constant: no temperature dependence, no curved "
            "V-I characteristic beyond R_on, no reverse recovery and no switching "
            "transitions. A component with a drop cannot also carry a loss card, which "
            "would count the same conduction loss twice."
        ),
        "evidence": [
            "RL step through a diode with V_f and R_d against the closed-form "
            "exponential (max error < 1e-6 A), diode voltage = V_f + R_d i sample by sample",
            "half-wave rectifier: first turn-on at 10 sin(wt) = V_f within 2 us of "
            "asin(V_f/V_pk)/w, peak (V_pk - V_f) R/(R + R_s)",
            "synchronous buck with equal R_on in periodic steady state: "
            "V_out = D V_in R/(R + R_on) to 1e-6 relative",
            "diode buck in CCM: V_out = D V_in - (1 - D) V_f to 1e-6 relative",
            "switch with R_on in parallel with a diode: the diode clamps the node at V_f "
            "and the currents split as the closed form says",
            "ideal devices keep their model hash; an explicit zero is the ideal device",
            "NOT compared against an independent engine: an ngspice diode is exponential, "
            "not piecewise-linear, so it cannot check this model exactly",
        ],
    },
    {
        "id": "model.rlc_dc_sources_ideal_switch_diode",
        "status": "supported",
        "scope": "documented v0.1 primitives and topology restrictions",
        "statement": "The reference model supports R, L, C, DC sources, ideal switches, and diodes.",
        "evidence": ["component and converter regression suites"],
    },
    {
        "id": "model.subcircuit.flatten",
        "status": "supported",
        "scope": (
            "declared reusable subcircuits with explicit ports and parameter "
            "bindings, expanded deterministically at load time"
        ),
        "statement": (
            "A subcircuit instance flattens into ordinary namespaced components "
            "(`<instance>.<inner>`) before validation, so every derived branch "
            "passes the same rules as a hand-written one and the canonical IR "
            "hash is a function of the flattened model only — hierarchy is "
            "presentation, never physics. Unbound ports, unknown parameters, "
            "id collisions and recursive references fail closed naming the "
            "offending instance."
        ),
        "evidence": [
            "bit-identity regression: examples/rectifier_12pulse_yyd.yaml "
            "rewritten around a three-phase diode-bridge subcircuit produces "
            "waveforms exactly equal to the flat original (tests/test_subcircuits.py)",
        ],
    },
    {
        "id": "verification.agent.run_compare",
        "status": "supported",
        "scope": "manifest-verified PEARSim numerical-equivalence observations",
        "statement": (
            "Binds both source runs and a versioned comparison profile into retained JSON, "
            "offline HTML, and a recursive artifact manifest."
        ),
        "evidence": ["run-comparison producer contract"],
    },
    {
        "id": "verification.canonical_boost.boundary_sweep",
        "status": "supported",
        "scope": (
            "retained v0.1 evidence producer for the exact recognized canonical ideal boost"
        ),
        "statement": (
            "Produces a bounded, hash-manifested CCM/BCM/DCM sweep with thirteen "
            "independent-oracle gates per point; this is verification evidence only."
        ),
        "evidence": ["canonical boost boundary evidence contract and regression suite"],
    },
    {
        "id": "verification.canonical_buck.numerical_matrix",
        "status": "supported",
        "scope": (
            "full-v1 retained numerical evidence for the exact recognized canonical ideal buck"
        ),
        "statement": (
            "Produces source-bound tolerance, maximum-step, output-grid, solver, and native "
            "cross-platform evidence; acceptance is decided independently for every run."
        ),
        "evidence": ["canonical-buck numerical evidence contract and regression suite"],
    },
    {
        "id": "verification.canonical_inverting_buck_boost.boundary_sweep",
        "status": "supported",
        "scope": (
            "retained v0.1 evidence producer for the exact recognized canonical ideal "
            "inverting buck-boost over a bounded 0.25..200 R/R_BCM request containing "
            "CCM, exact BCM, and DCM points"
        ),
        "statement": (
            "Produces a bounded, hash-manifested CCM/BCM/DCM sweep with fourteen blocking "
            "comparison and invariant gates per point after a derived minimum regime-"
            "separation check; this is verification evidence only."
        ),
        "evidence": ["canonical inverting buck-boost evidence contract"],
    },
)


def _validate_catalog() -> None:
    ids = [record["id"] for record in _CAPABILITY_RECORDS]
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise RuntimeError("PEARSim capability declarations must have unique, sorted IDs")
    for record in _CAPABILITY_RECORDS:
        if record["status"] not in CAPABILITY_STATUSES:
            raise RuntimeError(f"Invalid status for capability {record['id']!r}")


_validate_catalog()


def build_capabilities() -> dict[str, Any]:
    """Return a fresh deterministic capability catalog.

    ``supported`` means an implemented contract within the stated alpha scope. It does
    not imply physical validation. Anything absent from the catalog is unsupported.
    """

    return {
        "schema": CAPABILITIES_SCHEMA,
        "maturity": "early_alpha",
        "default_policy": {
            "unlisted_capability": "unsupported",
            "silent_fallback": False,
            "reason": "Capabilities are fail-closed: absence is never evidence of support.",
        },
        "status_definitions": {
            "supported": (
                "Implemented in the reference path within the stated scope; this is not a "
                "physical-validation or production-suitability claim."
            ),
            "experimental": (
                "Executable, but its acceptance evidence or supported envelope is incomplete."
            ),
            "unsupported": "No accepted semantic and evidence contract is available.",
        },
        "claim_boundary": {
            "classification": "exploratory_early_alpha",
            "validation_claim": False,
            "compatibility_claim": False,
            "allowed_uses": [
                "inspect and reproduce small system-level idealized simulations",
                "develop exploratory hypotheses with independent engineering checks",
            ],
            "prohibited_uses": [
                "hardware or production design sign-off",
                "protection or safety settings",
                "regulatory evidence",
                "semiconductor loss, temperature, EMI, or reliability decisions",
            ],
        },
        "capabilities": deepcopy(list(_CAPABILITY_RECORDS)),
    }


def get_capability(capability_id: str) -> dict[str, Any]:
    """Return one declaration, treating every unknown identifier as unsupported."""

    if not isinstance(capability_id, str) or not capability_id.strip():
        raise ValueError("capability_id must be a non-empty string")
    if len(capability_id) > MAX_CAPABILITY_ID_LENGTH:
        raise ValueError(f"capability_id exceeds the {MAX_CAPABILITY_ID_LENGTH}-character limit")
    for record in _CAPABILITY_RECORDS:
        if record["id"] == capability_id:
            return deepcopy(record)
    return {
        "id": capability_id,
        "status": "unsupported",
        "scope": "unlisted capability",
        "statement": "This capability is absent from the versioned catalog and is unsupported.",
        "evidence": [],
        "reason": "fail_closed_unlisted_capability",
    }


__all__ = [
    "CAPABILITIES_SCHEMA",
    "CAPABILITY_STATUSES",
    "build_capabilities",
    "get_capability",
]
