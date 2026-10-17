# Security policy

## Reporting a vulnerability

Please report security issues **privately** through GitHub's
[private vulnerability reporting](https://github.com/PEAR-power/pear-sim/security/advisories/new)
rather than a public issue. Include the version (`pear-sim --version`), the
command or MCP call, and a minimal model that reproduces the problem.

We aim to acknowledge a report within five working days.

## Scope

PEARSim reads model files and writes run bundles on the local machine. Reports
we especially want:

- a model, request or MCP tool call that reads or writes outside the paths it
  was given;
- a crafted input that hangs the process or exhausts memory despite the
  enforced limits (`pear-sim doctor --json` lists them);
- anything that executes code from a model file (the format has no expression
  language by design).

Numerical inaccuracy is a bug, not a security issue — please open a normal
issue for it.

## Supported versions

During 0.x only the latest release receives fixes.
