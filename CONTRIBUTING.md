# Contributing to PEARSim

Thank you for considering a contribution. PEARSim values **correct, honestly
bounded results** over breadth, so please read this before opening a pull
request.

## Before you start

- For anything larger than a small fix, open an issue first and describe the
  change, the physics or equations behind it, and how it will be verified.
- Read [AGENTS.md](AGENTS.md): it lists the source map and the rules every
  change must follow (no hidden regularization, deterministic events,
  fail-closed behaviour). It applies to human contributors too.

## Development setup

```bash
uv sync --locked --extra dev
uv run pytest            # full suite
uv run ruff check .
```

Optional: install [ngspice](https://ngspice.sourceforge.io/) to run the
independent-engine differential tests (`tests/test_ngspice_oracle.py`); they
skip without it, and CI runs them.

## What a good pull request contains

1. **A test that fails before the change.** Every solver or model bugfix starts
   with a regression test. New components need a positive test and an
   invalid-input test.
2. **The source of the physics.** Record the equation reference (textbook,
   paper, standard) and expected behaviour for every new component, analysis or
   benchmark.
3. **Honest claim boundaries.** If a capability is new or unverified, register
   it in `src/pear_sim/api/capabilities.py` as `experimental` and say what it
   does *not* cover. Never widen a claim to make output look better, and never
   loosen a validation threshold just to make a test pass — find the cause.
4. **Schema discipline.** A model-format change updates the JSON Schema,
   `docs/model-format.md`, the examples and the tests in the same pull request.

## Clean-room rules

Contributions must rest on public mathematics, standards, openly licensed
sources, original engineering work, or material you are authorized to use. Do
**not** submit proprietary or decompiled code, copied icons, manuals, component
libraries or demo models, native model files from commercial tools, golden data
of unknown provenance, or wording implying vendor endorsement.

`tests/test_public_hygiene.py` rejects a small set of terms that must never
enter this repository; if it fires on your change, please ask in the pull
request rather than working around it.

## License

By contributing you agree that your contribution is licensed under the
[Apache License 2.0](LICENSE), the license of this project.
