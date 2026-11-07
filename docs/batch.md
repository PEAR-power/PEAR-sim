# Batch: many designs, one ranking

`pear-sim batch` evaluates many designs of one model and ranks them against a goal you
declare. It answers "which of these designs is best for my objective, under my limits?"
for up to 2 000 designs per request.

```bash
pear-sim batch examples/buck_with_loss_cards.yaml \
  --request examples/batch_buck_losses.json --out out/batch
```

```text
Batch pass: 40 designs, 12 feasible, 0 failed
  #1 d0034  duty=0.4034, fsw=9136  ->  losses.total_mean_power_w=18.8459, ...
  ...
  (minimize total_mean_power_w; best of the sample, not an optimum)
```

The same request runs through the Python API (`api.evaluate_designs(model, request,
out)`) and the MCP tool `evaluate_designs`.

## What you get, and what you do not

- **Best design of the sample, not an optimum.** Nothing searches between the sampled
  designs, so a better design may exist. Sample more densely, or narrow the ranges
  around the winner, to look for one.
- **Screening runs are lean.** Every design runs with `--lean`: the same metrics and
  losses as a full run, less evidence (checked on all 38 shipped examples: every metric
  and loss field identical). A screening run keeps only the files the ranking
  reads (`run.json`, its artifact manifest, `metrics.json`, `losses-thermal.json`,
  `warnings.json`). Waveforms and the rest are deleted: a 2 000-design batch of the
  loss-card buck takes about 170 MB instead of 4.6 GB. Runs are deterministic, so any design can
  be re-run bit for bit.
- **Finalists are full runs and are checked.** The top `finalists` feasible designs
  (default 3) are re-run with full evidence under `finalists/<design_id>/run`. Each
  must reproduce its screening values exactly. If one does not, the batch status is
  `finalist_check_failed` and the mismatch is listed, because the ranking would then
  rest on a number the full run does not confirm.
- Designs run in parallel (`--jobs N`, default all cores but one); the results do not
  depend on the job count.

## Request

```json
{
  "schema": "pear-sim/batch-request/v0.1",
  "label": "free text",
  "designs": { ... },
  "objective": {"minimize": { quantity }},
  "constraints": [ { quantity, "min": 38, "max": 42 } ],
  "finalists": 3
}
```

### Designs

Three ways to choose designs. Every parameter must be declared in the model's
`parameters:` section and must change the simulation.

**Latin hypercube** (`"method": "lhs"`, the default). Each parameter range is cut into
`count` equal slices and every slice gets exactly one design, so even a small sample
covers every range evenly. `seed` (default 0) makes the sample reproducible; the
expanded designs are written to `request.normalized.json`.

```json
{"method": "lhs", "count": 200, "seed": 1, "parameters": [
  {"name": "fsw", "from": "5 kHz", "to": "50 kHz", "scale": "log"},
  {"name": "duty", "from": "35 %", "to": "45 %"}
]}
```

**Grid** (`"method": "grid"`): every combination of `points` values per parameter.
The product must stay within 2 000.

```json
{"method": "grid", "parameters": [
  {"name": "fsw", "from": "10 kHz", "to": "40 kHz", "points": 7, "scale": "log"}
]}
```

**List** (`"method": "list"`): designs you already have, e.g. catalogue parts.

```json
{"method": "list", "rows": [{"fsw": "10 kHz", "duty": "40 %"}, {"fsw": "20 kHz"}]}
```

`scale` is `linear` (default) or `log`; a log range must be positive.

### Quantities

The objective and every constraint read one number from a run:

| Quantity | Meaning |
|---|---|
| `{"signal": "v(out)", "metric": "mean_last_half"}` | a recorded output's metric (same metric names as `sweep parameter`, e.g. `rms_last_half`, `peak_to_peak_last_half`, `max`) |
| `{"losses": "total_mean_power_w"}` | estimated mean loss of all devices with cards, W |
| `{"losses": "tj_max_c"}` | hottest junction temperature of any device, °C (add `"device": "q1"` for one device) |
| `{"losses": "mean_power_w", "device": "q1"}` | one device's mean loss, W |

Loss quantities need `device_cards` and `loss_analysis` in the model. Constraint bounds
are plain numbers in the quantity's unit (V, A, W, °C).

### Constraints and ranking

A design is **feasible** when its run succeeded and every constraint holds (`min <=
value <= max`). Feasible designs are ranked by the objective; ties go to the lower
design id. Designs that fail to run are kept with their error and never ranked.

## Output

```
out/batch/
  batch-summary.json       status, claim, ranking (top 20), finalists, failures, every design
  designs.csv              one row per design: parameters, every quantity, feasibility, rank
  request.source.json      the request as given
  request.normalized.json  the request with every design expanded
  source-model/            the model, byte for byte, that every design ran from
  designs/d0001/run/       screening run (ranking inputs only)
  finalists/d0034/run/     full evidence bundle
```

`status` is `pass`, `no_feasible_design` or `finalist_check_failed`. The command exits
0 only on `pass`.

## Limits

2 000 designs per request, 16 parameters, 16 constraints, 10 finalists, 100 grid
points per parameter. Each design has the same per-run time limit as a sweep point;
the batch as a whole has no time limit.
