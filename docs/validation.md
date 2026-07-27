# MicroColossus Validation Ledger

This document records executable validation results and the exact boundary of what each run establishes.

## Validation policy

A run must identify the tested commit, environment, commands, generated artifacts, failures, and untested capabilities.

A planning result is not a training result. A resident result is not evidence of storage-backed or out-of-core execution. A CPU result is not evidence that MPS works on an Apple Silicon system.

## First independent M0 diagnostic. July 27, 2026

### Scope

The diagnostic tested commit `37fb45b189d13bab2b6f4084af0c21993f65ceff` on `main`.

It attempted to validate:

- isolated installation;
- Ruff, pytest, compileall, and mypy;
- static planning;
- resident CPU training;
- CPU reproducibility;
- fixed-batch learning behavior;
- CUDA execution when available;
- artifact integrity;
- source-tree cleanliness.

### Reported result

Overall result: **FAIL**.

Confirmed successes:

- installation completed after a network retry;
- the static planner completed for CPU and automatic-device configurations;
- `compileall` passed;
- the repository remained unchanged;
- CUDA was unavailable and was correctly skipped.

Blocking failures:

1. Ruff reported an import-format issue in `microcolossus/planner.py`.
2. pytest failed because checksum telemetry called `Tensor.numpy()` in an environment where NumPy was unavailable.
3. Every resident run stopped while calculating the model checksum.
4. CPU reproducibility and fixed-batch results could not complete because checksum telemetry aborted execution.
5. mypy reported missing third-party stubs and typing issues.

The run did not test or establish any claim about:

- MPS execution;
- NVMe-backed state;
- activation offloading;
- asynchronous transfers;
- intra-layer tiling;
- training state larger than resident memory.

## Corrective implementation

The next implementation removes the implicit NumPy dependency and changes the primary accelerator direction to Apple MPS.

Changes include:

- hashing raw CPU tensor storage without calling NumPy;
- tests that force `Tensor.numpy()` to fail;
- checksum tests for stability, state changes, and bfloat16 tensors;
- explicit `mps` device support;
- `auto` device selection that prefers MPS;
- synchronized MPS step timing;
- MPS current tensor allocation, Metal driver allocation, and recommended working-set telemetry;
- `microcolossus doctor` environment reporting;
- a dedicated `examples/tiny-mps.yaml` configuration;
- unified-memory warnings in the static planner;
- planner import and typing corrections;
- third-party type stubs;
- mypy and a CPU smoke run in CI.

### Maintainer-side verification before publication

The candidate implementation completed in a CPU-only environment:

- 19 pytest tests;
- bytecode compilation;
- a three-step resident CPU run;
- generation of all expected JSON and JSONL artifacts;
- static planning for the MPS-oriented configuration;
- environment reporting on a machine without MPS.

The earlier corrective candidate also completed two identical CPU runs and a 100-step fixed-batch diagnostic whose loss moved from `5.566842079162598` to `0.414573609828949`.

Ruff and mypy could not be installed in the maintainer environment because its package index did not provide those packages. They remain required CI checks.

No real Mac M2 was available in the maintainer environment. MPS execution and MPS memory values remain unverified until the next independent run.

## Required Mac M2 follow-up

The next independent diagnostic should run on the exact corrective commit and include:

1. `microcolossus doctor`;
2. Ruff, mypy, pytest, and compileall;
3. the CPU baseline twice for exact reproducibility;
4. the MPS baseline twice;
5. CPU-versus-MPS loss, gradient, and parameter comparisons with documented tolerances;
6. all MPS allocation fields at every step;
7. macOS memory-pressure observations;
8. confirmation that no unsupported operation silently used CPU fallback;
9. a fixed-batch MPS overfit diagnostic;
10. the complete diagnostic archive and report.

The run must not claim NVMe streaming, activation offloading, intra-layer tiling, or out-of-core behavior because those capabilities are not implemented.
