# MicroColossus Validation Ledger

This ledger records executable evidence, engineering decisions, and the exact boundary of each result.

## Validation policy

A run must identify the tested commit, machine, software versions, commands, artifacts, failures, and capabilities that were not exercised.

Mandatory distinctions:

- a static plan is not a training result;
- resident training is not out-of-core training;
- a CPU result is not evidence for MPS or MLX;
- a functional pass with local source changes is not a clean protocol pass;
- checksum equality is stronger than numerical agreement;
- framework memory counters are not assumed to represent equivalent physical memory;
- full-parameter, compact, and adapter methods are reported separately;
- a performance result is not accepted unless initialization, inputs, optimizer semantics, synchronization, and measured phases are documented;
- a larger parameter count is not a successful result when correctness, throughput, recovery, or storage endurance are unacceptable.

## First M0 diagnostic. July 27, 2026

Tested commit:

```text
37fb45b189d13bab2b6f4084af0c21993f65ceff
```

Overall result: **FAIL**.

Blocking findings:

1. Ruff found an import-format problem in `microcolossus/planner.py`.
2. Checksum telemetry called `Tensor.numpy()` without NumPy available.
3. Resident runs stopped while calculating the checksum.
4. Reproducibility and fixed-batch diagnostics could not complete.
5. mypy found third-party stub and local typing problems.

This run did not validate MPS or any storage-backed capability.

## Corrective MPS implementation

The next implementation:

- removed the implicit NumPy dependency from model checksums;
- added checksum regression tests, including bfloat16 state;
- introduced explicit `mps` support and MPS-first automatic selection;
- synchronized MPS timing;
- added MPS tensor, Metal driver, and recommended working-set telemetry;
- added `microcolossus doctor`;
- added an M2-oriented configuration;
- made the planner aware of unified-memory accounting;
- corrected typing and CI coverage.

## First real Mac M2 diagnostic. July 27, 2026

The first hardware diagnostic started from:

```text
29e7a1c4b3d012b3dc1e223a97f35e1bd865e22e
```

Environment:

- MacBook Air;
- Apple M2;
- 8 GB unified memory;
- native arm64 execution;
- no Rosetta translation;
- PyTorch 2.13.0;
- MPS built and available;
- deliberate MPS-to-CPU fallback disabled.

Strict protocol result: **FAIL** because two source fixes were intentionally left in the working tree.

Functional result after those fixes: **PASS** for the resident MPS scope.

The fixes were:

1. remove an extra blank line that failed Ruff formatting;
2. calculate the global gradient norm without constructing float64 tensors on MPS.

Reported CPU-versus-MPS one-step differences for the tiny workload:

- worst parameter absolute difference: `9.277835488319397e-06`;
- worst parameter relative difference: `0.0005274261347949505`;
- worst optimizer-state absolute difference: `3.259629011154175e-09`;
- worst optimizer-state relative difference: `0.004142380319535732`.

The fixed-batch MPS diagnostic reduced loss from `5.566842079162598` to `0.4145740866661072`.

Repeated MPS runs were not bitwise identical by final checksum. Loss and gradient norm were identical at the first checksum divergence. MPS reproducibility is evaluated numerically as well as bitwise.

## Clean resident Mac M2 rerun. July 27, 2026

Tested commit:

```text
a56fc514f2f8e705654034f3c2f02e3a441c61f3
```

Overall result: **PASS**.

Protocol integrity:

- `git status --short` was empty before the run;
- no source fix was required during the run;
- `git status --short` was empty after the run.

Passed checks:

- Ruff;
- pytest, 21 tests;
- compileall;
- mypy;
- MPS preflight;
- explicit MPS resident training;
- automatic device selection, resolved to `mps`;
- larger resident MPS smoke run;
- CPU-versus-MPS diagnostic;
- fixed-batch MPS overfit diagnostic.

Observed learning result:

```text
5.566842079162598 -> 0.4145740866661072
```

The diagnostic reported no evidence of deliberate CPU fallback, unsupported MPS operators in the tested path, NaN, infinity, or MPS out-of-memory failure.

This pass validates resident training, device selection, telemetry, and the tested model paths. It does not validate NVMe, streaming, tiling, or out-of-core execution.

## Competitive benchmark harness

Version 0.3 introduced a backend-neutral resident benchmark with:

- one portable FP32 parameter state generated outside either framework;
- one deterministic portable token stream;
- matching controlled Transformer architecture in PyTorch and MLX;
- synchronized warm-up and measured phases;
- uncheckpointed PyTorch resident execution;
- PyTorch activation-checkpointing execution;
- an MLX resident implementation;
- matching AdamW coefficients, epsilon, bias correction, clipping, batch, and sequence settings;
- first-step and steady-state latency statistics;
- tokens per second;
- process RSS, framework allocator values, available system memory, memory percentage, and swap usage;
- machine-readable benchmark and comparison schemas;
- initial-state and input-batch checksums;
- atomic final-state `.state.npz` artifacts;
- tensor-level numerical comparison.

## First competitive target attempt

Commit `2f7963a5d0af3eabb5a31eab4013f422725c71c0` stopped at the project-check gate.

Blocking findings:

1. Ruff import ordering in benchmark modules.
2. NumPy 2.5 typing syntax incompatible with the configured Python 3.11 mypy target.

No performance benchmark or backend decision was accepted from that attempt.

## Competitive target run. Version 0.3.2

Tested commit:

```text
785183a1ff87df0c22df9619d1ab7bf53968bc79
```

Environment:

- MacBook Air M2;
- 8 GB unified memory;
- native arm64 without Rosetta;
- PyTorch `2.13.0`;
- MLX `0.32.0`;
- NumPy `2.4.6`;
- MPS fallback disabled.

Runtime result: **PASS**.

Repository-quality result at that commit: **PARTIAL PASS**, because mypy reported three static typing errors while every runtime variant completed.

Completed:

- PyTorch MPS preflight;
- MLX preflight;
- all three tiny variants;
- three counterbalanced competitive rounds for each backend variant;
- all nine competitive runs;
- equal portable initial-state checksums;
- equal token-batch checksums;
- valid final-state artifacts;
- clean repository state;
- no detected hidden fallback, unsupported operator, non-finite value, or MPS OOM.

The competitive workload used 23,213,056 parameters, 6 blocks, hidden size 512, 8 heads, vocabulary 8,192, sequence length 128, microbatch 1, and full-parameter FP32 AdamW.

Median measured throughput:

| Variant | Tokens/s | Relative to PyTorch |
|---|---:|---:|
| PyTorch MPS | 1,379.23 | 1.000x |
| PyTorch MPS checkpointed | 1,337.37 | 0.970x |
| MLX | 2,195.69 | 1.592x |

Numerical result:

- PyTorch versus checkpointed PyTorch: exact equality in the competitive runs;
- PyTorch versus MLX maximum loss difference: `1.9073486328125e-06`;
- PyTorch versus MLX maximum final-parameter absolute difference: `3.8036610931158066e-05`;
- PyTorch versus MLX mean final-parameter absolute difference: `7.378751249633382e-09`;
- all values finite and stable across all three rounds.

Backend decision: **DUAL BACKEND**.

- MLX is the preferred optimized Apple Silicon execution candidate.
- PyTorch MPS remains the portable numerical oracle and recovery/debugging reference.
- The storage layer must remain backend-neutral.

Activation checkpointing was classified as **INCONCLUSIVE** for this model because it was numerically equivalent but did not produce a decisive memory benefit and reduced throughput by about 3%.

## Typing corrections. Version 0.3.3

The competitive run found three static typing issues:

1. dynamic NumPy `savez` archive-member typing;
2. MLX `tree_flatten` union narrowing;
3. MLX AdamW `betas` annotation.

Commit:

```text
b75d2f646da4ca4dce5acdee567a1f17adcc503c
```

Version `0.3.3` applied runtime-preserving fixes for all three findings.

## Final 0.3.3 verification

A fresh Mac M2 clone of commit `b75d2f646da4ca4dce5acdee567a1f17adcc503c` completed the final release-quality gate.

Environment:

- MicroColossus `0.3.3`;
- NumPy `2.4.6`;
- PyTorch `2.13.0`;
- MLX import successful;
- MPS built and available;
- MPS fallback disabled.

Project checks:

- Ruff: passed, all checks passed;
- mypy: passed, no issues in 17 source files;
- pytest: passed, 28 tests;
- compileall: passed;
- doctor: passed and detected Apple M2 MPS.

Tiny smoke results:

| Variant | Tokens/s | Final loss |
|---|---:|---:|
| PyTorch MPS | 10,339.74 | 5.601982116699219 |
| PyTorch MPS checkpointed | 8,083.50 | 5.601982593536377 |
| MLX | 13,927.55 | 5.601983070373535 |

Equivalence and artifact checks:

- portable-state checksum identical: `c375bea95d4d37da897cf852d824098775ef1552530cc2936876167ae53cdc40`;
- batch checksum identical: `a2eacb6299cacfdc41d19863545606365c2b4c793c0dc0336b19f4cb3b4eacce`;
- every JSON document parsed;
- every `.state.npz` artifact existed and matched its recorded checksum;
- all final parameters were finite;
- no fallback, unsupported operator, OOM, allocation failure, NaN, or infinity evidence was found;
- `git status --short` was empty;
- `git diff --check` was clean.

Numerical comparison:

- PyTorch versus checkpointed maximum loss difference: `4.768e-07`;
- PyTorch versus checkpointed maximum final-state absolute difference: `5.960e-08`;
- PyTorch versus MLX maximum loss difference: `9.536e-07`;
- PyTorch versus MLX maximum final-state absolute difference: `3.4948e-05`;
- PyTorch versus MLX mean final-state absolute difference: `1.20696e-08`.

The tiny results did not materially differ from the previous baseline. The nine larger performance runs were not repeated because version `0.3.3` changed typing contracts without changing the numerical algorithm or benchmark schedule.

Overall result: **PASS**.

## Milestone status

### M0. Resident foundation

Completed.

### M1. Clean Mac M2 validation

Completed.

### M2. Competitive Apple Silicon baseline

Completed.

Established:

- both resident backends execute correctly on the target;
- the benchmark provides equal initial state and batches;
- artifacts are verifiable;
- numerical distance is measured;
- resident performance is characterized;
- the dual-backend decision is evidence-backed;
- package quality checks pass on the target environment.

### M3. Versioned NVMe tensor store

Next.

Required validation:

- deterministic tensor and chunk manifests;
- immutable or copy-on-write chunks;
- per-chunk checksums;
- atomic manifest publication;
- write-ahead journal;
- recovery after partial writes and interruption;
- bounded staging memory;
- read and write telemetry;
- PyTorch and MLX export and restore equivalence;
- clean source tree and reproducible artifacts.

## Current boundary

The completed resident and competitive milestones do not validate:

- NVMe-backed parameters or optimizer state;
- storage-to-accelerator streaming;
- managed activation recomputation;
- asynchronous prefetch or writeback;
- intra-layer tiling;
- transactional optimizer publication;
- crash recovery;
- training state larger than safe resident unified memory.
