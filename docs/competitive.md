# Competitive Engineering Log

This document records the evidence used to select optimization techniques and execution backends for MicroColossus.

## Decision rule

MicroColossus must remain competitive by measurement. A technique is accepted only when it improves a declared objective without violating numerical correctness, memory limits, storage endurance, stability, recovery, or reproducibility.

Full-parameter training, compact optimizer methods, and adapter methods are reported separately.

## Validated resident foundation

A clean MacBook Air M2 run of commit `a56fc514f2f8e705654034f3c2f02e3a441c61f3` passed the resident validation protocol:

- native arm64 execution with 8 GB unified memory;
- MPS built and available;
- project checks passed;
- explicit MPS training and automatic MPS selection passed;
- CPU-versus-MPS numerical comparison passed for the tested workload;
- fixed-batch loss moved from `5.566842079162598` to `0.4145740866661072`;
- no detected fallback, unsupported operator, non-finite value, or MPS out-of-memory error.

This result validates resident training only.

## Competitive benchmark harness

Version 0.3 introduced a backend-neutral resident benchmark with:

- identical portable FP32 parameters and token batches;
- PyTorch MPS, checkpointed PyTorch MPS, and MLX variants;
- synchronized warm-up and measured phases;
- latency, throughput, process memory, allocator counters, available memory, and swap;
- atomic final-state artifacts;
- tensor-level numerical comparison.

## First competitive target attempt

Commit `2f7963a5d0af3eabb5a31eab4013f422725c71c0` stopped at the project-check gate because of Ruff import ordering and an incompatible NumPy typing version. No runtime benchmark or backend decision was made from that attempt.

Release `0.3.2` corrected import ordering, constrained NumPy to `>=2.4,<2.5`, added multi-version CI, and added packaging regression tests.

## Competitive M2 run on version 0.3.2

A clean MacBook Air M2 run tested commit `785183a1ff87df0c22df9619d1ab7bf53968bc79`.

Environment:

- Apple M2;
- 8 GB unified memory;
- native arm64 without Rosetta;
- PyTorch `2.13.0` with MPS built and available;
- MLX `0.32.0`;
- NumPy `2.4.6`;
- deliberate MPS-to-CPU fallback disabled.

Runtime result: **PASS**.

Repository-quality result at that commit: **PARTIAL PASS**, because mypy reported three static typing errors while all runtime variants completed.

Passed runtime and integrity checks:

- installation;
- Ruff;
- 26 pytest tests;
- compileall;
- PyTorch MPS preflight;
- MLX preflight;
- all three tiny variants;
- all nine competitive runs;
- equal portable initial-state checksums;
- equal token-batch checksums;
- valid final-state `.state.npz` artifacts;
- clean source tree before and after execution;
- no detected hidden fallback, unsupported operator, NaN, infinity, or MPS out-of-memory failure.

### Workload

The competitive preset used:

- 23,213,056 unique parameters;
- 6 Transformer blocks;
- hidden size 512;
- 8 attention heads;
- vocabulary size 8,192;
- sequence length 128;
- microbatch 1;
- full-parameter FP32 AdamW;
- 3 warm-up steps and 10 measured steps per process;
- three counterbalanced rounds per variant.

### Performance

Median measured throughput:

| Variant | Tokens/s | Relative to PyTorch |
|---|---:|---:|
| PyTorch MPS | 1,379.23 | 1.000x |
| PyTorch MPS checkpointed | 1,337.37 | 0.970x |
| MLX | 2,195.69 | 1.592x |

Median first-step duration:

| Variant | Seconds |
|---|---:|
| PyTorch MPS | 0.27484 |
| PyTorch MPS checkpointed | 0.40946 |
| MLX | 0.09846 |

The result was not classified as noisy. Relative throughput spread across rounds was approximately:

- PyTorch: `0.38%`;
- checkpointed PyTorch: `0.26%`;
- MLX: `1.53%`.

No material order or thermal bias was reported.

### Numerical behavior

PyTorch versus checkpointed PyTorch:

- classification: **GREEN**;
- maximum loss difference: `0.0`;
- maximum final-parameter absolute difference: `0.0`;
- maximum final-parameter relative difference: `0.0`.

PyTorch versus MLX:

- classification: **YELLOW**;
- maximum loss difference: `1.9073486328125e-06`;
- maximum final-parameter absolute difference: `3.8036610931158066e-05`;
- mean final-parameter absolute difference: `7.378751249633382e-09`;
- all values finite;
- identical aggregate values across all three competitive rounds.

The maximum relative difference was large for a parameter near zero. It is interpreted together with the corresponding absolute and mean differences.

### Memory and swap

The frameworks expose different allocator scopes. Their counters are not added or treated as directly equivalent physical memory.

Maximum observed process RSS:

| Variant | Bytes |
|---|---:|
| PyTorch MPS | 400,556,032 |
| PyTorch MPS checkpointed | 383,205,376 |
| MLX | 467,320,832 |

Maximum framework allocator values:

| Variant | Bytes |
|---|---:|
| PyTorch MPS | 393,752,832 |
| PyTorch MPS checkpointed | 393,922,816 |
| MLX | 659,408,088 |

Additional reported counters:

- PyTorch Metal driver allocation: up to `1,156,251,648` bytes;
- checkpointed PyTorch Metal driver allocation: up to `1,147,863,040` bytes;
- MLX cache: `447,390,448` bytes;
- MLX reported no swap increase in the three competitive runs;
- PyTorch reported one first-round swap increase of `826,540,032` bytes and zero in the other two rounds;
- checkpointed PyTorch reported `655,360` bytes in one round and zero in the other two.

The isolated PyTorch swap event is retained as evidence. It is not treated as a proven framework property without additional controlled runs.

### Activation checkpointing

Decision: **INCONCLUSIVE for the tested resident model**.

Checkpointing was numerically identical, but:

- throughput was approximately 3% lower;
- allocator and driver counters were nearly unchanged;
- process RSS was only modestly lower;
- the tested activation footprint was too small to establish a decisive capacity benefit.

Checkpointing remains a required baseline for larger contexts and activation-heavy workloads.

## Backend decision

Decision: **DUAL BACKEND**.

Rationale:

- MLX was approximately `1.592x` faster in median steady-state throughput.
- MLX also had lower median first-step latency in the competitive runs.
- PyTorch remains the portable numerical oracle.
- PyTorch provides the clearest current reference for debugging, exact state comparison, validation policy, and recovery semantics.
- The storage-backed path has not been benchmarked and may have different bottlenecks from resident execution.

Current responsibility split:

- **MLX**: preferred optimized Apple Silicon execution path.
- **PyTorch MPS**: numerical oracle, reference backend, debugging path, and recovery-contract baseline.
- **Backend-neutral storage layer**: required so the tensor store does not depend on one framework.

This is not a permanent promise that every operation will use MLX. Each future optimization remains subject to measurement.

## Static typing findings and release 0.3.3

The version 0.3.2 runtime suite identified three mypy findings:

1. NumPy's dynamic `np.savez` keyword archive interface conflicted with the typed `allow_pickle` keyword.
2. MLX `tree_flatten` returns a list-or-dictionary union, so iteration required explicit narrowing.
3. MLX AdamW stubs require `betas` as a list rather than a tuple.

Version `0.3.3`, commit `b75d2f646da4ca4dce5acdee567a1f17adcc503c`, applied runtime-preserving corrections:

- locally cast the public `np.savez` callable to its dynamic callable contract;
- verify that `tree_flatten` returned a list before tuple unpacking;
- pass AdamW betas as `[0.9, 0.999]`.

## Final 0.3.3 verification

A fresh Mac M2 clone of commit `b75d2f646da4ca4dce5acdee567a1f17adcc503c` completed the final release-quality gate.

Environment:

- MicroColossus `0.3.3`;
- NumPy `2.4.6`;
- PyTorch `2.13.0`;
- MLX `0.32.0`;
- MPS built and available;
- MPS fallback disabled.

Project checks:

- Ruff: passed;
- mypy: passed with no issues in 17 source files;
- pytest: passed, 28 tests;
- compileall: passed;
- doctor: passed and detected Apple M2 MPS.

Tiny smoke results:

| Variant | Tokens/s | Final loss |
|---|---:|---:|
| PyTorch MPS | 10,339.74 | 5.601982116699219 |
| PyTorch MPS checkpointed | 8,083.50 | 5.601982593536377 |
| MLX | 13,927.55 | 5.601983070373535 |

Artifact and equivalence checks:

- portable-state checksum identical across all variants: `c375bea95d4d37da897cf852d824098775ef1552530cc2936876167ae53cdc40`;
- batch checksum identical across all variants: `a2eacb6299cacfdc41d19863545606365c2b4c793c0dc0336b19f4cb3b4eacce`;
- every JSON document parsed;
- every sibling `.state.npz` artifact existed and matched its recorded checksum;
- all final parameters were finite;
- no fallback, unsupported operator, OOM, allocation failure, NaN, or infinity evidence was reported;
- `git status --short` was empty and `git diff --check` was clean.

Numerical comparison:

- PyTorch versus checkpointed PyTorch maximum loss difference: `4.768e-07`;
- PyTorch versus checkpointed PyTorch maximum final-state absolute difference: `5.960e-08`;
- PyTorch versus MLX maximum loss difference: `9.536e-07`;
- PyTorch versus MLX maximum final-state absolute difference: `3.4948e-05`;
- PyTorch versus MLX mean final-state absolute difference: `1.20696e-08`.

The tiny results did not materially differ from the prior baseline, so the nine larger performance rounds were not repeated.

## Competitive milestone status

The resident competitive milestone is **complete**.

Established:

- package quality checks pass on the target environment;
- PyTorch MPS and MLX both execute the controlled full-parameter workload;
- the benchmark delivers identical initial state and batches;
- output artifacts are verifiable;
- numerical differences are measured rather than hidden;
- MLX is materially faster in the tested resident workload;
- the dual-backend decision is supported by evidence.

Not established:

- NVMe-backed parameters or optimizer state;
- storage-to-accelerator streaming;
- managed activation recomputation;
- asynchronous prefetch or writeback;
- intra-layer tiling;
- transactional optimizer publication;
- crash recovery;
- training state larger than safe resident unified memory.

## Next engineering gate

The next milestone is the backend-neutral versioned NVMe tensor store:

1. define stable tensor, chunk, version, manifest, and transaction schemas;
2. implement immutable or copy-on-write chunk storage;
3. add per-chunk checksums;
4. add write-ahead journaling;
5. publish manifests atomically;
6. recover the last committed state after interruption;
7. enforce staging and storage budgets;
8. report read bytes, write bytes, latency, fsync time, and cumulative SSD writes;
9. validate identical export and restore through PyTorch and MLX adapters.

Synchronous storage-to-accelerator training begins only after the store passes integrity, recovery, and budget tests.

Operational tracking continues in GitHub issues.
