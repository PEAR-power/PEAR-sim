# Sensitivity: which parameters move which outputs

`pear-sim sensitivity` measures how much each model parameter moves each output when
all parameters vary over ranges you declare. Use it before a batch to learn which
parameters are worth searching over, and after one to see whether the winner sits on
a parameter the result is fragile to.

```bash
pear-sim sensitivity examples/buck_with_loss_cards.yaml \
  --request examples/sensitivity_buck_morris.json --out out/morris
```

```text
Sensitivity (morris) pass: 80 runs, 0 failed
  v(out).mean_last_half
    duty             mu*=10 [10, 10]  sigma=3.54e-05
    fsw              mu*=5e-05 [2.83e-05, 7.24e-05]  sigma=5.61e-05
    load             mu*=3.895e-05 [2.16e-05, 5.69e-05]  sigma=4.42e-05
  v(out).peak_to_peak_last_half
    fsw              mu*=0.2759 [0.228, 0.322]  sigma=0.116
    ...
  losses.total_mean_power_w
    fsw              mu*=21.12 [17.2, 25.4]  sigma=9.75
    load             mu*=20.81 [17.7, 24.4]  sigma=7.76
    duty             mu*=8.923 [7.2, 11]  sigma=4.39
```

Read: the mean output moves 10 V across the duty range (100 V × 10 % points, the
ideal buck law) and not at all with load or frequency. Ripple is a frequency matter.
Losses depend on frequency and load about equally, with a large `sigma`: the effect of
one depends on the other (switching loss grows with frequency times current).

## Two methods

| | Morris | Sobol |
|---|---|---|
| Question | which parameters matter, roughly how much | what share of the output's variation each one explains |
| Runs | `trajectories × (k + 1)`; 20 × 4 = 80 for 3 parameters | `base_samples × (k + 2)`; 64 × 5 = 320 for 3 parameters |
| Result | `mu_star`: typical output change across the whole range, in the output's unit; `mu`: the same with sign; `sigma`: how much that change depends on the other parameters | `S1`: share explained by the parameter alone; `ST`: share including its interactions; `ST − S1` is the interaction part |
| Use | screening many parameters cheaply | a quantitative split once you know which parameters matter |

Every index has a bootstrap 95 % interval (resampling trajectories or base samples). A
wide interval means the study needs more runs, not that the parameter is uncertain.

## What the numbers mean, and do not

- Parameters are **independent and uniform** over their ranges (log-uniform on a log
  scale). The indices describe the model **inside this box**. They are a statement
  about real component spread only if the ranges are that spread.
- The intervals measure the **sampling error of this study**, not model error.
- Sobol indices split the output's variance; for an output that does not vary at all
  they are undefined, and the summary says so instead of reporting zeros.
- A run that fails removes its whole trajectory (Morris) or base sample (Sobol) from
  the estimate; the summary reports how many were used.

## Checked against closed forms

The estimators are tested on functions whose indices are known on paper, so the
tests measure the truth, not the code's agreement with itself:

- Morris on a linear function returns its coefficients exactly with `sigma = 0`, and a
  product of two parameters gives a large `sigma`.
- Sobol on the Ishigami function converges to the analytic indices (S1 = 0.314,
  0.442, 0; ST = 0.558, 0.442, 0.244) within 0.02 at 4 096 base samples, and the
  95 % intervals cover them at 256.
- Sobol indices do not change when a constant is added to the output. (The textbook
  first-order estimator is unbiased but its noise grows with the output's mean; the
  outputs are centred first.)
- On the loss-card buck, the mean output's Morris effect of the duty is 10 V and of
  load and frequency below 1 mV, as the ideal buck law requires.

## Request

```json
{
  "schema": "pear-sim/sensitivity-request/v0.1",
  "method": "sobol",
  "parameters": [
    {"name": "duty", "from": "35 %", "to": "45 %"},
    {"name": "fsw", "from": "10 kHz", "to": "40 kHz", "scale": "log"}
  ],
  "outputs": [
    {"signal": "v(out)", "metric": "peak_to_peak_last_half"},
    {"losses": "total_mean_power_w"}
  ],
  "base_samples": 64,
  "seed": 1
}
```

- `parameters`: at least two, at most 16; same form as a [batch](batch.md) range.
- `outputs`: 1 to 16 quantities, same selectors as a batch objective (signal metrics,
  `losses.total_mean_power_w`, `losses.tj_max_c`, per-device `mean_power_w`).
- Morris: `trajectories` (default 10, 2..200) and `levels` (even, default 4).
- Sobol: `base_samples`, a power of two (default 64, 8..1024).
- `seed` (default 0) makes the study reproducible; `bootstrap` (default 1000) sets the
  number of resamples for the intervals.
- At most 2 000 runs per study. Runs are lean and go in parallel (`--jobs`).

## Output

```
out/morris/
  sensitivity-summary.json  claim, settings, per output: indices with intervals, ranking
  runs.csv                  every run's parameters and outputs
  request.source.json
  source-model/             the model, byte for byte, every run used
  runs/d0001/run/           each run, pruned to the files the reading needs
```

`status` is `pass`, `pass_with_failed_runs` (some runs failed, estimates still made) or
`incomplete` (an output could not be estimated). The command exits 0 on the first two.
