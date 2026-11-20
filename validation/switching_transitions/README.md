# Switching transitions — verification vs validation

The `switching_transitions` section (docs/model-format.md §7c) is a post-processed
estimate. This folder keeps the two kinds of evidence apart, by design.

## Verification (does the code solve its own equations?)

Two methods that are not PEARSim, both in CI:

- **Closed forms**, `tests/test_switching_transitions.py`: exact piecewise integration of
  literal Coss tables written in the test (never the producer's parser) — swing time
  Q_node/I, the LC energy condition for ZVS, Qoss(V)·V for a full hard turn-on, the
  linear-C overshoot I·√(L/C) and the ramp response.
- **ngspice**, `tests/test_switching_transitions_ngspice.py` (required job
  `ngspice-oracle`, where a skip is a failure): behavioural nonlinear capacitors
  `C='pwl(V, table)'`, its own integrator and its own initial operating point —
  constant current, lumped LC, the full-bridge pair, and the overshoot.

Each gate is shown both ways: silent on the correct code, firing on Coss × 1.2 and on
an inverted commutating current.

## Validation (does the model agree with hardware?)

`validate.py` compares PEARSim with **published bench measurements** of other people's
hardware and writes `results.json` and `RESULTS.md` (generated — do not edit). MEASURED
always means one of those third-party values, cited to page and figure/table.

| Part | Source | What is compared | Kind of number |
|---|---|---|---|
| A | Barba, Musumeci, Stella, Mandrile, Palma, *Energies* 17(15) 3855, 2024, Fig. 4 and text p. 5 | ZVS yes/no boundary of an EPC2065 leg at 48 V, 150 ns dead time; fall times at 1–2 A; residual voltage at 0.5 A | boundary: stated in text; times and residual: read off a plot (±5 ns, ±1.5 V; the fall onset is ambiguous by 10–20 ns) |
| B | onsemi AND90238/D Rev. 1, Table 7 p. 6 (conditions p. 5, loops Table 1 p. 2) | turn-off V_peak of a SiC module at 400 V / 600 A for Lσ = 23, 30, 37 nH | table values |

Card inputs: EPC2065 Coss(V) read from the datasheet plot (Fig. 5a, ±20 pF); the
datasheet TABLE gives QOSS(40 V) = 33 nC typ, 50 nC max, which sets the Coss band used
in part A and in the sweep (×1.00 … ×1.52). NVXR17S90M2SPC: Coss = 3.3 nF typ at 400 V
(datasheet table); the curve above 400 V is not published and is held (flagged).

**What this supports, and no more** (numbers in `RESULTS.md`):

- ZVS boundary (A): reproduced at ONE operating point of ONE device. At 0.5 A the
  agreement is marginal with typical Coss (a residual just above the 1 % threshold) and
  robust only toward the max end of the datasheet band. Absolute fall times are not
  validated: the measurements sit inside the typ–max band, slower than typical at
  1–1.5 A (board and probe capacitance, Coss hysteresis — the paper does not say).
- Overshoot (B): with the MEASURED di/dt, V + Lσ·di/dt under-estimates slightly and the
  lossless ramp over-estimates; every measured peak lies between them. The model was
  not given the gate drive, so di/dt itself is not predicted.
- Not validated at all: the lumped-LC and bridge-pair regimes (DAB, LLC), other devices
  and voltages, and anything the cards of the DAB/LLC examples say (they are synthetic).

The label of the feature therefore stays **"verified; validated only at the cited
points"** — never "measured", never "validated" without that qualifier.

Looked at and not usable as measurements: EPC WP012 (dead-time numbers are
calculated), Reusch & Strydom APEC 2013 / EPC WP010 (loop inductances are simulated, no
di/dt), the EPC LLC article (calculated), Costinett et al. TPEL 2015 (paywalled).

## Sensitivity (part C of validate.py)

Each example is simulated ONCE; the transition is re-post-processed with Coss scaled
over the datasheet band and Lσ over a declared band (exact, because the transition never
feeds back). `RESULTS.md` marks which ZVS verdicts are robust and which are only a range.
