# MicroColossus Validation Ledger

This ledger records executable evidence and the exact boundary of each result.

## Validation policy

A run must identify the tested commit, machine, software versions, commands, artifacts, failures, and capabilities that were not exercised.

Mandatory distinctions:

- a static plan is not a training result;
- resident training is not out-of-core training;
- a CPU result is not evidence for MPS;
- a functional pass with local source changes is not a clean protocol pass;
- checksum equality is stronger than numerical agreement;
- framework memory counters are not assumed to represent equivalent physical memory;
- full-parameter, compact, and adapter methods are reported separately.

## First M0 diagnostic. July 27, 2026

Tested commit:

```text
37fb45b189d13bab2b6f4084af0c21993f65ceff
```

Overall result: **FAIL**.

Confirmed successes:

- isolated installation completed after a network retry;
- the static planner completed;
- bytecode compilation passed;
- CUDA was unavailable and was skipped correctly;
- the source tree remained clean.

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

The MPS-safe path accumulates float32 squared-gradient sums on the device, transfers scalar values to CPU, and combines them with Python floating-point summation. A regression test protects this behavior.

Reported CPU-versus-MPS one-step differences for the tiny workload:

- worst parameter absolute difference: `9.277835488319397e-06`;
- worst parameter relative difference: `0.0005274261347949505`;
- worst optimizer-state absolute difference: `3.259629011154175e-09`;
- worst optimizer-state relative difference: `0.004142380319535732`.

The fixed-batch MPS diagnostic reduced loss from `5.566842079162598` to `0.4145740866661072`.

Repeated MPS runs were not bitwise identical by final checksum. Loss and gradient norm were identical at the first checksum divergence. MPS reproducibility is therefore evaluated numerically as well as bitwise.

## Clean Mac M2 rerun. July 27, 2026

Tested commit:

```text
a56fc514f2f8e705654034f3c2f02e3a441c61f3
```

Environment:

- MacBook Air;
- Apple M2;
- 8 GB unified memory;
- native arm64 execution;
- no Rosetta translation;
- PyTorch 2.13.0;
- MPS built and available.

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

One warning reported that NumPy was not installed. It did not block resident training because core checksums no longer require NumPy. Version 0.3.0 declares NumPy explicitly for development and competitive benchmarks.

### Exact boundary of the clean pass

The clean pass validates:

- package installation and project checks;
- the controlled decoder-only Transformer;
- full-parameter resident AdamW training;
- the tested MPS operator path;
- device selection and diagnostics;
- resident telemetry and artifact generation;
- the tested tiny and small configurations.

It does not validate:

- NVMe-backed parameters or optimizer state;
- storage-to-accelerator streaming;
- out-of-core execution;
- runtime-managed activation recomputation;
- asynchronous prefetch or writeback;
- intra-layer tiling;
- crash recovery;
- training state larger than safe resident unified memory.

## Competitive benchmark harness. Version 0.3.0

After the clean M2 pass, the repository added a backend-neutral resident benchmark harness.

Implemented benchmark features:

- one portable FP32 parameter state generated outside either framework;
- one portable deterministic token-batch stream;
- matching controlled Transformer architecture in PyTorch and MLX;
- synchronized warm-up and measured phases;
- uncheckpointed PyTorch resident execution;
- PyTorch activation-checkpointing execution;
- an uncompiled MLX resident implementation;
- matching AdamW coefficients, epsilon, bias correction, clipping, batch, and sequence settings;
- first-step and steady-state latency statistics;
- tokens per second;
- process RSS, framework allocator values, available system memory, memory percentage, and swap usage;
- machine-readable benchmark and comparison schemas;
- checksums for the portable initial state and input batches;
- warnings that prevent allocator counters from being treated as physical-memory equivalents.

The PyTorch CPU benchmark and checkpointed variant have automated test coverage and a CI smoke command.

The MLX backend is implemented against documented MLX APIs, but it is not considered validated until it runs on the target M2 and its model, optimizer, loss, parameter count, and artifacts are checked.

## Competitive validation gate

Direct Apple Silicon baselines begin with:

1. resident PyTorch MPS;
2. PyTorch MPS with activation checkpointing;
3. native MLX with the equivalent controlled Transformer;
4. compiled MLX as a later optimized variant;
5. MLX-LM full-model fine-tuning when semantically comparable;
6. MicroColossus reference execution;
7. future compact and adapter modes, reported separately.

Storage-offload systems that cannot run on MPS remain architectural references rather than direct hardware benchmarks.

Every accepted comparison records:

- architecture and unique parameter count;
- initial state and batch checksums;
- precision, optimizer, clipping, and update semantics;
- sequence length, microbatch, warm-up, and measured steps;
- synchronization policy;
- exact framework, Python, macOS, and repository versions;
- first-step and steady-state latency;
- tokens per second;
- process footprint, framework memory counters, available memory, pressure, and swap;
- storage bytes and SSD writes when storage is involved;
- numerical distance from the reference.

An optimization is accepted only when it improves a declared objective without violating correctness, memory, endurance, stability, recovery, or reproducibility constraints.

## Next required validation

The next target-hardware run should:

1. install the `benchmark` extra on the clean M2 checkout;
2. run PyTorch MPS without activation checkpointing;
3. run PyTorch MPS with activation checkpointing;
4. run the uncompiled MLX backend with the same configuration;
5. compare initial-state and batch checksums;
6. compare loss trajectories and throughput;
7. inspect framework memory values together with macOS pressure and swap;
8. preserve all JSON artifacts and exact versions;
9. leave the source tree clean.

No backend decision should be made before those results exist.
