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

One warning reported that NumPy was not installed. It did not block resident training because core checksums no longer require NumPy. NumPy is now an explicit dependency of the development and benchmark extras, not of the core runtime.

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

Implemented features:

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

The PyTorch CPU benchmark and checkpointed variant received automated test coverage and a CI smoke command.

The MLX backend was implemented against documented MLX APIs. It was not considered validated because it had not run on the target M2.

## Benchmark hardening. Version 0.3.1

A design review found that the version 0.3.0 benchmark kept the portable NumPy initialization state alive while the framework model was being measured. On an 8 GB unified-memory system, that duplicate full-model copy would inflate process memory and could distort swap and pressure results.

Version 0.3.1 changes the benchmark contract:

- initial swap is sampled before constructing portable state;
- the backend loads the portable state and clears the NumPy state dictionary before warm-up;
- garbage collection runs before measured execution;
- the result records whether the initialization state was released;
- the result records the byte size of portable state and batches;
- final parameters are exported only after the timed region;
- final parameters are written atomically to a sibling `.state.npz` artifact;
- the final-state artifact is protected by a checksum;
- benchmark comparison loads and verifies both state artifacts;
- every final parameter tensor is compared for maximum absolute, mean absolute, and maximum relative difference;
- comparisons record finite-value status and the worst absolute and relative tensors;
- the implementation was split into schema, data, runner, comparison, and backend modules.

Development-environment verification for the hardened path:

- five dedicated benchmark tests passed;
- identical PyTorch CPU runs produced zero final-state tensor difference;
- the activation-checkpointing variant completed;
- state artifacts and checksums were validated;
- bytecode compilation passed;
- no new benchmark source line exceeded the configured 100-character limit.

This verification does not establish that the MLX backend works correctly on the target M2. It establishes the CPU-tested benchmark infrastructure that the target run will use.

## M2 competitive preset

`examples/m2-competitive.yaml` defines a controlled 23,213,056-parameter resident workload:

- 6 Transformer blocks;
- hidden size 512;
- 8 attention heads;
- vocabulary 8,192;
- sequence length 128;
- microbatch 1;
- dropout disabled;
- MPS target;
- 8 GB unified-memory system model.

The tiny configuration remains the first smoke test. The competitive preset is intended to reduce the fraction of time dominated by framework startup and compilation overhead.

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
- final parameter tensor differences;
- storage bytes and SSD writes when storage is involved;
- numerical distance from the reference.

An optimization is accepted only when it improves a declared objective without violating correctness, memory, endurance, stability, recovery, or reproducibility constraints.

## Next required validation

The next target-hardware run should:

1. start from a clean checkout of version 0.3.1 or later;
2. install the `benchmark` extra;
3. run the tiny PyTorch MPS, checkpointed PyTorch MPS, and MLX smoke tests;
4. compare initial-state, batch, and final-state artifacts;
5. run the same three variants with `examples/m2-competitive.yaml`;
6. compare loss trajectories, final tensors, throughput, memory pressure, and swap;
7. inspect framework memory values together with macOS pressure and swap;
8. preserve JSON and `.state.npz` artifacts with exact versions;
9. leave the source tree clean.

No backend decision should be made before those results exist. No out-of-core claim should be made before storage-backed execution exists.
