# Competitive Engineering Log

This document records the evidence used to select execution backends, memory techniques, storage strategies, and future optimizations for MicroColossus.

## 1. Decision rule

MicroColossus must remain competitive by measurement.

An optimization is accepted only when it improves a declared objective without violating:

- numerical correctness;
- working-set or memory limits;
- storage endurance;
- stability;
- recovery semantics;
- reproducibility;
- observability.

A technique can still be accepted when it is slower if it enables a declared capacity or durability objective that the faster baseline cannot satisfy. The tradeoff must be measured and documented.

Full-parameter training, compact optimizer methods, quantized methods, and adapter methods are reported separately.

## 2. Comparison contract

A valid comparison should hold constant, as far as the frameworks permit:

- model architecture;
- unique parameter count;
- initial parameter arrays;
- token batches;
- dtype;
- loss definition;
- AdamW hyperparameters and semantics;
- gradient clipping;
- sequence length;
- microbatch;
- warm-up and measured steps;
- synchronization points;
- target machine and power state;
- framework and dependency versions.

Required result categories include:

- tensor-level numerical distance;
- loss trajectory;
- first-step latency;
- steady-state throughput;
- process RSS;
- framework allocator counters;
- driver or cache counters;
- available memory and pressure;
- swap;
- storage reads and writes;
- `fsync` and publication cost;
- recovery behavior;
- cumulative application-managed writes.

Framework allocator counters are never added as separate physical-memory capacities.

## 3. Validated resident foundation

A clean MacBook Air M2 run of commit:

```text
a56fc514f2f8e705654034f3c2f02e3a441c61f3
```

validated:

- native arm64 execution with 8 GB unified memory;
- MPS built and available;
- explicit and automatic MPS execution;
- CPU-versus-MPS numerical comparison;
- fixed-batch learning from `5.566842079162598` to `0.4145740866661072`;
- no detected fallback, unsupported operator, non-finite value, or MPS out-of-memory error.

This result established the resident PyTorch MPS oracle. It did not establish storage-backed or bounded execution.

## 4. Competitive resident benchmark harness

Version 0.3 introduced a controlled resident benchmark with:

- portable identical FP32 parameters;
- portable identical token batches;
- PyTorch MPS, checkpointed PyTorch MPS, and MLX variants;
- synchronized warm-up and measured phases;
- latency and throughput statistics;
- process and framework memory counters;
- system memory, pressure, and swap snapshots;
- atomic final-state artifacts;
- tensor-level state comparison.

The initial competitive attempt stopped at static checks. Ruff import ordering and an incompatible NumPy typing version were corrected before any backend conclusion was accepted.

## 5. Competitive M2 result

The complete target benchmark tested commit:

```text
785183a1ff87df0c22df9619d1ab7bf53968bc79
```

Environment:

- MacBook Air with Apple M2;
- 8 GB unified memory;
- native arm64 without Rosetta;
- PyTorch 2.13.0 with MPS;
- MLX 0.32.0;
- NumPy 2.4.6;
- MPS-to-CPU fallback disabled.

Runtime result: **PASS**.

It completed all tiny variants and nine competitive runs, with three counterbalanced rounds for each backend variant.

## 6. Competitive workload

```text
unique parameters:   23,213,056
Transformer blocks:  6
hidden size:         512
attention heads:     8
vocabulary:          8,192
sequence length:     128
microbatch:          1
precision:           FP32
optimizer:           full-parameter AdamW
warm-up steps:       3 per process
measured steps:      10 per process
rounds:              3 counterbalanced rounds
```

The initial-state and batch checksums were identical across all variants and rounds.

## 7. Resident performance

Median steady-state throughput:

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

Relative throughput spread across the three rounds was approximately:

- PyTorch MPS: `0.38%`;
- checkpointed PyTorch MPS: `0.26%`;
- MLX: `1.53%`.

The result was not classified as noisy. No material order or thermal bias was reported.

## 8. Resident numerical behavior

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
- identical aggregate results across all three rounds.

The maximum relative difference was dominated by values near zero. It is interpreted together with the much smaller absolute and mean differences.

## 9. Resident memory observations

Maximum process RSS:

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

Additional counters:

- PyTorch Metal driver allocation reached `1,156,251,648` bytes;
- checkpointed PyTorch Metal driver allocation reached `1,147,863,040` bytes;
- MLX cache reached `447,390,448` bytes;
- MLX reported zero swap increase in the three competitive runs;
- PyTorch reported one first-round swap increase of `826,540,032` bytes and zero in the other two rounds;
- checkpointed PyTorch reported `655,360` bytes in one round and zero in the other two.

The isolated PyTorch swap event remains evidence. It is not treated as a proven framework property without further controlled runs.

## 10. Activation checkpointing decision

Decision: **INCONCLUSIVE for the tested resident workload**.

Checkpointing was numerically equivalent, but:

- throughput was about 3% lower;
- allocator and driver counters were nearly unchanged;
- process RSS was only modestly lower;
- the tested activation footprint was too small to demonstrate a decisive capacity benefit.

Checkpointing remains a required baseline for longer contexts, larger microbatches, and activation-heavy workloads.

## 11. Backend decision

Decision: **DUAL BACKEND**.

### MLX responsibility

MLX is the preferred optimized Apple Silicon execution candidate because it was materially faster in the accepted resident benchmark.

### PyTorch MPS responsibility

PyTorch MPS remains:

- the numerical oracle;
- the reference backend;
- the debugging path;
- the exact state-comparison path;
- the baseline for transaction and recovery semantics.

### Backend-neutral responsibility

The following remain independent of either framework:

- canonical tensor identity;
- chunk and manifest formats;
- versioning;
- checksums;
- transaction state;
- root step bundles;
- execution plans;
- recovery contracts.

This is not a promise that every future operation will use MLX. Every storage-backed and bounded path must earn its backend choice through measurement.

## 12. Static typing and release-quality findings

The competitive target run exposed three static typing issues:

1. NumPy's dynamic `np.savez` keyword interface conflicted with its static stub;
2. MLX `tree_flatten` required explicit list-or-mapping narrowing;
3. MLX AdamW stubs required `betas` as a list rather than a tuple.

Version 0.3.3 corrected those issues without changing the benchmark algorithm.

A fresh Mac M2 verification of commit:

```text
b75d2f646da4ca4dce5acdee567a1f17adcc503c
```

passed Ruff, mypy, 28 tests, compileall, doctor, all three tiny backend smoke paths, artifact checks, numerical comparisons, and clean source-tree checks.

## 13. Competitive implications for storage-backed execution

The resident benchmark does not automatically determine the best storage-backed backend.

Storage-backed execution introduces additional dimensions:

- canonical conversion overhead;
- tensor materialization overhead;
- child-store reads and writes;
- chunk reuse;
- `fsync` and manifest publication;
- page-cache behavior;
- root-bundle atomicity;
- recovery cost;
- group size and scheduling;
- opportunities for overlap.

Therefore:

- PyTorch MPS remains the first bounded reference implementation;
- MLX bounded execution remains a future optimization candidate;
- no MLX storage-backed performance advantage is claimed yet.

## 14. Competitive status of implemented storage milestones

### 14.1 Version 0.5 storage lifecycle

Validated on the target M2:

- exact PyTorch storage lifecycle for the tested paths;
- exact MLX model and optimizer round trips;
- cross-backend canonical state;
- five failure-injection points;
- detailed read, write, reuse, `fsync`, publication, memory, and swap telemetry.

### 14.2 Version 0.6 bounded forward

Validated on the target M2:

- one parameter group materialized at a time;
- exact boundary activations, logits, and loss;
- tied token-embedding reload;
- parameter-budget rejection;
- unchanged parameter manifest;
- zero swap growth.

### 14.3 Version 0.7 bounded backward

Validated on the target M2:

- reverse group execution;
- exact final gradients for the tested paths;
- tied-gradient accumulation and versioning;
- streamed global norm;
- parameter and gradient budget rejection;
- separate parameter, oracle-gradient, and bounded-gradient stores;
- zero swap growth.

### 14.4 Version 0.8 bounded optimizer

Implemented and validated in CPU CI:

- canonical global clipping;
- unique tied-weight update;
- group-local parameter, gradient, and Adam state;
- exact candidate state versus resident oracle;
- exact candidate restore;
- optimizer working-set rejection;
- root bundle checksum and atomic publication;
- interruption before bundle manifest or root pointer publication.

Target MPS validation remains pending.

## 15. Optimization sequence

The accepted optimization sequence is deliberately incremental.

1. correct synchronous storage and transactions;
2. bounded forward;
3. bounded backward and gradient storage;
4. streamed global norm;
5. group-bounded AdamW and atomic root publication;
6. multiple consecutive steps and resume;
7. activation recomputation or offload;
8. cache policy and chunk scheduling;
9. asynchronous prefetch and writeback;
10. intra-layer tiling;
11. bounded MLX execution where measurements justify it;
12. larger capacity demonstrations.

Prefetch, writeback, recomputation, and tiling are not introduced together. Each must demonstrate numerical equivalence and a measured resource benefit relative to the synchronous baseline.

## 16. Future competitive gates

### Gate A. Version 0.8 target MPS

Compare one complete bounded optimizer step with the resident PyTorch oracle. Measure:

- parameter and optimizer-state distance;
- clipping coefficient;
- tied-weight update count;
- group working sets;
- storage traffic;
- bundle publication;
- recovery;
- RSS, MPS, driver allocation, pressure, and swap.

### Gate B. Multi-step persistence

Compare uninterrupted and resumed bounded runs over many steps. Measure:

- loss trajectory;
- state continuity;
- optimizer step continuity;
- deterministic batch cursor and RNG restoration;
- storage growth;
- chunk reuse;
- stale-state retention and garbage collection;
- recovery time.

### Gate C. Activation policy

Compare:

- CPU-retained boundary activations;
- deterministic recomputation;
- storage-offloaded activations;
- checkpointed resident PyTorch.

Evaluate memory reduction, extra compute, storage traffic, and numerical distance.

### Gate D. Synchronous versus overlapped I/O

Compare:

- synchronous reads and writes;
- double-buffered prefetch;
- asynchronous writeback;
- different chunk sizes and group orders.

Report accelerator stalls and overlap efficiency.

### Gate E. PyTorch bounded versus MLX bounded

Only after equivalent bounded MLX forward, backward, and optimizer execution exist, compare identical state, data, budgets, group plans, and storage contracts.

### Gate F. Capacity demonstration

Require:

```text
complete managed state > safe resident capacity

every active working set < declared runtime budget
```

Measure correctness, throughput, storage traffic, writes, recovery, and training usefulness. A parameter-count-only demonstration is insufficient.

## 17. Risks that remain competitive blockers

### I/O overhead

Many small reads, writes, checksums, and `fsync` operations can dominate step time.

### SSD endurance

Full AdamW state includes parameters and two moments. Write amplification and checkpoint retention must remain explicit.

### Activation dominance

After parameter and optimizer residency are reduced, activations can become the largest remaining memory consumer.

### Minimum group size

If one layer exceeds the working-set budget, layer-level streaming is insufficient and intra-layer tiling is required.

### Validation overhead

Current milestone paths sometimes materialize complete oracle and candidate states after bounded execution. Production execution must not depend on this validation-only behavior.

### Backend duplication

Maintaining equivalent PyTorch and MLX implementations increases engineering cost. The dual-backend split is retained only while it provides measurable correctness and performance value.

## 18. Current competitive conclusion

MicroColossus has established a credible and measurable foundation:

- resident MPS and MLX execution are controlled and comparable;
- MLX is materially faster for the accepted resident workload;
- PyTorch remains a reliable numerical oracle;
- storage and recovery are backend-neutral;
- forward and backward parameter groups are bounded on the target M2;
- gradients match the resident oracle for the tested paths;
- one complete atomic bounded optimizer update is implemented and CPU-validated.

The project has not yet established:

- multi-step bounded training;
- real-corpus training;
- activation-bounded execution;
- asynchronous overlap;
- tiled operators;
- bounded MLX performance;
- a training state larger than safe resident memory.

The next competitive decision must be based on the 0.8 target MPS result and then on persistent multi-step execution, not on the resident benchmark alone.
