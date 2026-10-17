# ADR-0002: Native macOS-first with Windows support without Docker

- Status: Accepted
- Date: 2026-07-18
- Deciders: the PEARSim project

## Context

The first development and usage environment is macOS, including Apple Silicon. However, the project must also be usable by agents and CI on Windows. Docker cannot be a requirement, because:

- not every Windows workstation has Docker Desktop or organizational approval to install it;
- integration with local files, manufacturer tools and licensed solvers is often native;
- the goal is a small CLI/Python tool, not a service requiring a separate environment;
- container start-up cost degrades short, repeated agent invocations.

At the same time, the target Python -> C++20 migration must not break the contract or require a typical user to compile manually.

## Decision

PEARSim is designed as a native cross-platform tool:

- macOS arm64 is the first supported target;
- macOS x86_64 and Windows x86_64 are distribution targets;
- Linux x86_64 is a CI/server target, but does not mandate containerization;
- Docker MAY be used to reproduce Linux benchmarks, never as the only way to run;
- the reference Python kernel supports a modern CPython, initially at least Python 3.11;
- the target kernel is built in C++20 with CMake and shipped as a wheel and a native executable;
- the CLI, schema and artifacts are logically identical on all platforms.

## Platform contract

### Paths and files

- Code uses path APIs (`pathlib` in Python, `std::filesystem` in C++), not concatenation with `/`.
- The model stores relative paths with `/` as the portable format; the loader maps them onto the native filesystem.
- After normalization, a resource path must remain under the project root.
- Artifact names contain no `:`, `*`, `?`, control characters or Windows device names.
- Text is UTF-8; the parser accepts CRLF and LF, canonical output uses LF.
- Atomic writes use a temporary file in the same directory and a platform-appropriate replace.

### External processes

- A process is launched with an argument array, without `shell=True`, `sh -c` or `cmd /c`.
- Backend lookup order: explicit option, user configuration, environment variable, `PATH`.
- Spaces in paths are handled by the process API, not by manual quoting.
- Timeout and cancellation kill the whole process tree in a platform-appropriate way.
- A missing ngspice/Xyce yields `backend_unavailable`; the native backend still works.

### Time and determinism

- The simulation scheduler uses model time, not wall-clock time.
- Wall-clock time is used only for timeouts and statistics.
- Sample hits are based on integer ticks/rational periods.
- The reference regression mode is single-threaded and has a stable reduction order.
- `run.json` records the OS, architecture, Python or compiler version and the numerical backend.

### CLI

- The command name is `pear-sim`; Windows gets `pear-sim.exe` via a console-script shim (see [ADR-0003](0003-pear-sim-naming.md)).
- In `--json` mode stdout is exclusively a JSON document.
- Logs and progress go to stderr.
- Exit codes are common across platforms; details are in `model-format.md`.
- Basic operations do not open a GUI or a browser.

## Packaging

### Python stage

- the package is installable as a wheel;
- binary dependencies must have prebuilt wheels for the supported platforms or be optional;
- no runtime dependency on Homebrew, WSL, MSYS2 or Docker;
- development extras are separated from the minimal runtime;
- schema data and the component catalog are package resources, not paths relative to the repository.

### C++20 stage

- CMake as the only required project generator;
- pybind11 or an equivalent thin binding layer;
- `scikit-build-core`/`cibuildwheel` or an equivalent wheel pipeline;
- macOS universal2 MAY be added, but separate arm64/x86_64 wheels are acceptable;
- Windows uses the MSVC runtime in line with wheel distribution rules;
- symbols and assertions may be available in a separate diagnostic build.

## CI

Minimum matrix once the MVP is stabilized:

| OS | Architecture | Role |
|---|---|---|
| macOS | arm64 | first target, full tests and benchmark smoke |
| macOS | x86_64 or emulation | wheel compatibility |
| Windows | x86_64 | full contract tests without Docker |
| Linux | x86_64 | full tests, sanitizers and benchmark reference |

Each target runs:

1. parse and canonical hash of the same corpus;
2. analytical tests of the native kernel;
3. the CLI/JSON contract and exit codes;
4. artifact round-trip;
5. optional backend tests only when the backend is installed.

Tests comparing `float64` use physical tolerances, not byte-identical waveforms. The canonical model and event ordering remain identical.

## User configuration

Configuration is not required for the native backend. If it exists, it uses the standard application directory:

- macOS: `~/Library/Application Support/pear-sim/`;
- Windows: `%APPDATA%\pear-sim\`;
- Linux: `$XDG_CONFIG_HOME/pear-sim/` or `~/.config/pear-sim/`.

The CLI MAY accept `--config`, which takes precedence. Models must not rely on global configuration for physical values; configuration concerns only backends, cache and limits.

## Positive consequences

- a Windows user needs neither WSL nor Docker;
- an agent can run short simulations with a plain subprocess;
- easier access to local vendor tools and licenses;
- path and process-handling bugs are caught from the start;
- the migration to C++ has a defined distribution channel.

## Costs and negative consequences

- CI and releases cover more platforms;
- native wheels increase the complexity of the C++ release;
- the optional SPICE backend has different executable names and locations;
- process signals, file locking and atomic replace require separate implementations;
- `float64` results may differ slightly due to BLAS/compiler, so tolerances are needed.

## Rejected alternatives

### Docker as the only environment

Rejected because it breaks the Windows-without-Docker requirement, hampers integration with local solvers and increases the cost of agent invocations.

### Python only, without a native plan

Rejected as the target solution. Python is a good reference kernel, but event-heavy integration, sparse stamping and numerous factorizations should move to C++ once correctness is stabilized.

### C++ from the first commit

Rejected because it slows down iteration on semantics, format and tests. A readable Python oracle and a compatibility corpus are needed first.

### macOS-only

Rejected because the contract of an agent tool must be portable, and removing POSIX assumptions later would be costly.

## Acceptance criteria

1. The same model produces the same canonical hash on macOS and Windows.
2. `pear-sim lint` and `pear-sim simulate` do not require a POSIX shell.
3. The native backend works without any external executable.
4. A missing optional SPICE backend is a controlled, machine-readable error.
5. Paths with spaces and non-ASCII characters pass the tests.
6. An output directory created on one platform is readable on the other.
7. Installing a release does not require a compiler for a user on a typical supported platform.

## Signals for reconsideration

The ADR should be reopened if:

- required numerical dependencies have no stable native distribution path;
- the product becomes exclusively a remote service, without a local CLI;
- organizational security requirements mandate container isolation as an additional mode;
- support for a new architecture would require changing canonical results or the ABI.
