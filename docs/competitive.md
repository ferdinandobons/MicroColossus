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

Repository-quality result: **PARTIAL PASS**, because mypy reported three static typing errors while all runtime variants completed.

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
- identical values across all three competitive rounds.

The maximum relative difference was large for a parameter near zero. It is not interpreted without the corresponding absolute and mean differences.

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
- PyTorch currently provides the clearest reference for debugging, exact state comparison, validation policy, and future recovery semantics.
- The storage-backed path has not been benchmarked. It may have different bottlenecks from resident execution.

Current responsibility split:

- **MLX**: preferred optimized Apple Silicon execution path.
- **PyTorch MPS**: numerical oracle, reference backend, debugging path, and recovery-contract baseline.
- **Backend-neutral storage layer**: required so the tensor store does not depend on one framework.

This is not a permanent promise that every operation will use MLX. Each future optimization remains subject to measurement.

## Static typing findings

The runtime suite identified three mypy findings:

1. NumPy's dynamic `np.savez` keyword archive interface conflicted with the typed `allow_pickle` keyword.
2. MLX `tree_flatten` returns a list-or-dictionary union, so iteration required explicit narrowing.
3. MLX AdamW stubs require `betas` as a list rather than a tuple.

Version `0.3.3` applies runtime-preserving corrections:

- locally casts the public `np.savez` callable to its dynamic callable contract;
- verifies that `tree_flatten` returned a list before tuple unpacking;
- passes AdamW betas as `[0.9, 0.999]`.

A clean mypy and tiny-runtime verification on the target M2 is required before the typing gate is marked complete. Repeating the nine performance runs is not required unless the smoke comparison changes, because the fixes do not alter the numerical algorithm or benchmark schedule.

## Next engineering gate

1. Verify version `0.3.3` from a clean M2 checkout.
2. Require Ruff, mypy, pytest, and compileall to pass.
3. Repeat the tiny PyTorch MPS, checkpointed PyTorch MPS, and MLX smoke tests.
4. Confirm unchanged initial-state and batch checksums.
5. Confirm numerical behavior remains within the recorded tiny baseline.
6. Define a backend-neutral tensor manifest and transaction model.
7. Implement the versioned NVMe tensor store.
8. Build synchronous bounded storage-to-MLX execution.
9. Compare it against storage-to-PyTorch execution and the resident oracle.

## Current boundary

The competitive result establishes a resident dual-backend direction. It does not establish:

- NVMe-backed parameters or optimizer state;
- storage-to-accelerator streaming;
- managed activation recomputation;
- asynchronous prefetch or writeback;
- intra-layer tiling;
- transactional optimizer publication;
- crash recovery;
- training state larger than safe resident unified memory.

Operational tracking remains in [GitHub issue #1](https://github.com/ferdinandobons/MicroColossus/issues/1).
