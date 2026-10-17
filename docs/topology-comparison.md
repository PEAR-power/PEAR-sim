# Fair comparison of finalists `pear-sim/topology-comparison/v0.1`

Status: early-alpha contract
Code: `pear_sim.topology.comparison` · consumes: the [topology library](topology-library.md)
and the gradient engine

This is the place where it is **easiest to fool yourself**, so the rules are
structural, not advisory.

## Four principles

**1. Every finalist is driven to ITS OWN optimum.**
Evaluating all of them at one shared operating point silently crowns the
topology that point happened to suit. Every finalist gets its own
optimization run, its own trajectory and its own landing point.

**2. Only the oracle delivers the verdict.**
The gradient engine drives the search because it is ~20× cheaper (3 ms vs 78 ms
on the buck entry), but what it produces is a **direction**. Reported numbers
come from a full transient run at the landing point; gradient numbers
sit **beside** them, never in their place.

**3. The trust region is the STRUCTURE of the mode sequence.**
The Jacobian describes this circuit as long as the declared sequence holds. When a mode
appears, disappears or starts ending with a different event, the Jacobian describes
a system that no longer exists, and **nothing in the numbers will reveal it**. That is why
the landing point is re-extracted from the oracle run and compared
structurally.

> **The trust region is NOT the mode durations.** Shifting them is
> exactly what optimizing `duty` or `fsw` does. Checking them here
> would flag every successful run as an escape from the region — the gate would cry
> "wolf" until nobody read it any more. Duration drift is reported
> beside the verdict as information.

**4. A candidate the kernel cannot compute is REFUSED, not skipped.**
Entries may declare `kernel_support: not_simulated`. Ranking them
would be fabrication; skipping them would let the engine's coverage silently decide the
shortlist. They come back with the status `unsupported` — a statement about the **engine**,
not a defect of the candidate.

## Statuses

`ok` · `infeasible` · `not_converged` · `invalid_model` · **`unsupported`**

The fifth is required by the `not_simulated` declaration and **must not be merged**
with `invalid_model`: that would be the same mistake ("rejecting for absence")
one level further down.

## What is rankable

Only a finalist with status `ok` **whose landing point has been checked
and preserved the mode structure**. Rationale: the promise of the comparison is "each at its
own optimum". If the optimizer left the trust region, we do not have an optimum —
we have a point (measured correctly by the oracle, but reached by descending along
a wrong Jacobian). `checked: false` **never means** `held: true`:
an unchecked region is not a satisfied region.

Everyone else goes to `not_ranked` **with a reason**, because a candidate the
kernel cannot compute, a candidate that left the trust region and a candidate
that simply lost are **three different facts**.

## Tier-pair health metric

For each metric: **the change predicted by the gradient** (landing − start)
vs **the change measured by the oracle** (the same two points). Stricter than
rank correlation, because it carries magnitude and sign; free from day one.
Divergence of this pair is the earliest signal that the trust region is too
large.

When a metric **did not move** (e.g. the buck's mean output voltage does not depend on
L), the relative error is `None`, not "100%" — dividing two floating-point
crumbs would yield a confident message about noise.

## Pareto front — with the dominated points

Objectives **are not summed** into a single number: efficiency, ripple and temperature
do not add up to anything an engineer would defend. The result is a front, not a
ranking. **Dominated points stay in the record** — a front showing only the
winners cannot be audited.

## "Without full loss cards you do not win silently"

A model without `device_cards`/`loss_analysis` is not just less well documented —
it **looks lossless**, so on every efficiency-flavored axis it beats an
instrumented rival for free. When the field is **unevenly** instrumented,
every uninstrumented finalist receives the `OPTIMISTIC` flag, and the record carries
`mixed_loss_instrumentation: true`. An evenly uninstrumented field **is not**
flagged — flagging everyone would make the flag worthless.

## Usage

```bash
uv run pear-sim topology compare --dir library/topologies \
  --request examples/comparison_requests/buck_vs_boost.json --out comparison.json
```

The request is a file, not a stack of flags, because a fair comparison needs
**per-finalist bounds** — and a silently defaulted bound is a bound
nobody chose. Exit code `1` when nothing could be ranked: the question
"which one is better?" remained unanswered.

## What is NOT here (deliberately)

- **Transistor selection by snapping to a real part** (relaxation of
  `Rds_on`/`Qg` → optimization → snap → re-evaluation by the oracle)
  **is not implemented**: the `components` store contains no device
  catalog, and the cards in the repository are explicitly **synthetic** (see the capability
  `electrothermal.loss_and_temperature`). Building a snap to a non-existent
  catalog would produce numbers without backing. It returns once such a catalog exists.
- **Recording the engineer's decision** — that is a human action, and the
  comparison does not settle anything on the engineer's behalf.

## What this is NOT

A position on the front **is not a recommendation** and carries no claim
about hardware behavior. It means only this: for the declared objectives, at the measured
landing point, this candidate is not dominated by any other
**among those that could be ranked**.
