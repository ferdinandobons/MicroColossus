# MicroColossus Validation Ledger

This document records executable validation results and the exact boundary of what each run establishes.

## Validation policy

A run must identify the tested commit, environment, commands, generated artifacts, failures, and untested capabilities.

A planning result is not a training result. A resident result is not evidence of storage-backed or out-of-core execution. A CPU result is not evidence that MPS works on an Apple Silicon system.

Strict protocol status and functional status are reported separately when a diagnostic applies local fixes without committing them. A functional pass does not become a clean repository pass until the same result is reproduced from a clean checkout of the integrated commit.

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

## Corrective implementation after the first diagnostic

The next implementation removed the implicit NumPy dependency and changed the primary accelerator direction to Apple MPS.

Changes included:

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
- planner typing corrections;
- third-party type stubs;
- mypy and a CPU smoke run in CI.

Maintainer-side CPU verification completed tests, bytecode compilation, resident runs, expected artifacts, static MPS-oriented planning, and environment reporting. It did not verify real Metal execution.

## First real Mac M2 diagnostic. July 27, 2026

### Tested source and machine

The diagnostic started from commit `29e7a1c4b3d012b3dc1e223a97f35e1bd865e22e` on `main`.

Environment reported by the independent run:

- MacBook Air;
- Apple M2;
- 8 GB unified memory;
- native `arm64` execution;
- no Rosetta translation;
- PyTorch `2.13.0`;
- MPS built: `true`;
- MPS available: `true`;
- MPS fallback environment variable unset.

### Strict and functional results

Strict protocol result: **FAIL**.

The strict result was caused by the final source-tree check. Two fixes were intentionally applied to tracked files during the post-baseline rerun, so the working tree was not clean.

Functional post-fix result: **PASS** for the implemented resident MPS scope.

Post-fix checks:

- Ruff: passed;
- pytest: passed, 20 tests;
- compileall: passed;
- mypy: passed;
- explicit MPS training: passed;
- automatic device selection: passed and resolved to `mps`;
- larger resident MPS smoke run: passed;
- no post-fix command failed.

### Fixes identified by the Mac M2 run

1. `microcolossus/planner.py` contained an extra blank line that caused Ruff import formatting to fail.
2. `microcolossus/training.py` calculated the gradient norm through float64 tensors. MPS does not support that path. The corrected implementation calculates float32 squared sums on the active device, transfers scalar values to CPU, and combines them with Python floating-point accumulation.

Both fixes are now integrated in `main`, together with a regression test that rejects a return to `Tensor.double()` in the gradient-norm path.

### Numerical results

MPS reproducibility was not bitwise exact. The final parameter checksum differed at step 2, while loss and gradient norm were identical at the first differing step.

The run therefore classified repeated MPS execution as divergent by checksum, but the available scalar trajectory did not show a corresponding numerical instability.

CPU versus MPS one-step comparison reported:

- worst parameter absolute difference: `9.277835488319397e-06`;
- worst parameter relative difference: `0.0005274261347949505`;
- worst optimizer-state absolute difference: `3.259629011154175e-09`;
- worst optimizer-state relative difference: `0.004142380319535732`.

These measurements are evidence of close behavior for the tested tiny workload. They are not a general tolerance standard for every architecture, dtype, operator, or training duration.

### Learning behavior

The fixed-batch MPS diagnostic reduced loss from:

```text
5.566842079162598
```

to:

```text
0.4145740866661072
```

This establishes that the tested resident model and AdamW path can learn the repeated synthetic batch on MPS. It does not establish language-model quality or out-of-core correctness.

### Memory observations

Maximum values reported across the diagnostic were:

- process RSS: `414744576` bytes;
- current MPS tensor allocation: `394506496` bytes;
- Metal driver allocation: `1156251648` bytes.

These counters overlap under unified memory and must not be summed as physical memory use.

Swap change reported by the individual experiments:

- tiny MPS runs: `0` bytes;
- tiny 10-step run: `-16777216` bytes;
- larger resident MPS run: `669054402` bytes.

The larger run's swap growth is a signal that future scale tests must include memory pressure, compression, swap, thermal state, and elapsed time. It is not by itself proof of a leak.

### Fallback and operator support

`PYTORCH_ENABLE_MPS_FALLBACK` was unset. The filtered diagnostic scan reported no evidence of:

- CPU fallback;
- unimplemented operators;
- unsupported operators;
- MPS out-of-memory failures;
- high-watermark failures;
- allocation failures.

Absence of evidence in this run does not prove that all future operators and model sizes are supported.

### Exact boundary of the result

The run validated only:

- resident full-parameter training;
- the controlled Transformer used by the prototype;
- the tested MPS operators;
- current diagnostic and telemetry paths;
- the tested tiny and small resident configurations.

It did not test:

- NVMe-backed parameters or optimizer states;
- out-of-core training;
- storage-to-MPS streaming;
- activation offloading;
- planned recomputation;
- asynchronous prefetch or writeback;
- intra-layer tiling;
- training state larger than safe resident unified memory.

## Competitive validation gate

MicroColossus must remain competitive with the strongest relevant alternatives on the same target hardware.

The mandatory direct baseline set for Apple Silicon begins with:

1. resident PyTorch MPS;
2. PyTorch MPS with activation checkpointing when applicable;
3. native MLX with an equivalent controlled Transformer;
4. MLX-LM full-model fine-tuning when the workload can be matched;
5. MicroColossus reference mode;
6. MicroColossus compact and adapter modes, reported separately.

Storage-offload systems that cannot run on MPS, including ZeRO-Infinity and LoHan, remain architectural and algorithmic references rather than direct hardware benchmarks.

Every comparison must document:

- exact model architecture and parameter count;
- initialization, input batches, and random seeds;
- dtype, optimizer, clipping, and update semantics;
- sequence length, batch configuration, and step count;
- warm-up and synchronization policy;
- macOS, framework, and package versions;
- machine power, thermal, and memory-pressure conditions where measurable.

Required outputs include:

- numerical distance from the reference;
- first-step and steady-state latency;
- tokens per second;
- process RSS;
- current MPS allocation;
- Metal driver allocation;
- swap growth and memory pressure;
- NVMe bytes and SSD writes when storage is involved;
- checkpoint and recovery cost when implemented.

A proposed optimization is accepted only if it improves a declared objective while respecting correctness, memory, endurance, and reproducibility constraints.

MLX is a mandatory baseline because it is designed specifically for Apple Silicon, exposes lazy graph execution, and uses shared unified-memory arrays across CPU and GPU operations. If an MLX implementation materially outperforms PyTorch MPS for a required path, MicroColossus should add an MLX backend or adopt MLX for that path. PyTorch remains the portable numerical reference.

Full-parameter training results must not be presented as equivalent to LoRA, QLoRA, low-rank optimizer methods, quantized optimizer state, or other approximate modes.

## Next required validation

The next diagnostic must start from a clean checkout containing the integrated fixes and must leave the working tree clean.

It should include:

1. all existing M2 checks;
2. two repeated MPS runs with tensor-level numerical comparison;
3. an equivalent controlled Transformer implemented in MLX;
4. PyTorch MPS versus MLX throughput and memory comparison;
5. macOS memory-pressure and swap observations;
6. warm-up-separated latency statistics;
7. complete artifacts and exact framework versions.

Only after that clean rerun should the resident M2 milestone receive an unqualified protocol `PASS`.
