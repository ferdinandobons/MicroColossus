# MicroColossus Project Specification

> **Trade memory for time.**

This document defines the project, architecture, evidence, competitive rules, and development path.

## 1. Purpose

MicroColossus explores full-parameter training when the complete training state cannot remain resident in available memory.

The central question is:

> How can a runtime execute a valid update while active computation remains inside explicit memory, storage, time, and endurance budgets?

The primary target is Apple Silicon, beginning with a Mac M2 with 8 GB of unified memory and local NVMe storage.

MicroColossus does not promise infinite models, datacenter throughput, or elimination of total compute. Storage capacity, memory pressure, minimum useful tile size, compute, bandwidth, SSD endurance, and elapsed time remain physical constraints.

## 2. Why the project exists

Training requires more than model weights. A full step may require:

- parameter gradients;
- optimizer states;
- optional master-weight copies;
- saved activations or recomputation metadata;
- logits and loss intermediates;
- temporary tensors and operator workspaces;
- allocator, framework, driver, and operating-system overhead.

A machine may have enough storage for the complete state but insufficient safe resident memory. MicroColossus treats that mismatch as a scheduling problem whose complete cost must be measured.

## 3. Apple Silicon design rules

Apple Silicon CPU and GPU workloads share unified physical memory.

### Placement is not capacity offload

Moving a tensor from CPU execution to MPS changes placement and execution. It does not create a second physical memory pool. Keeping CPU and accelerator copies can increase pressure on the same memory.

### Memory counters overlap

Process RSS, framework allocator values, Metal driver allocation, compressed memory, and swap are different views of memory behavior. They must not be added as separate physical pools.

### NVMe is the capacity tier

```text
MLX or MPS execution and active tensor working set
                         |
bounded unified-memory staging and runtime state
                         |
versioned NVMe tensor store
```

### Memory pressure is a planning input

The planner must consider active allocations, process footprint, macOS pressure, compression, swap, storage traffic, SSD writes, and step time together.

## 4. Central hypothesis

For a useful class of Transformer models and training recipes, part of resident-memory demand can be exchanged for:

- versioned tensor storage;
- bounded staging and caches;
- activation recomputation;
- layer-wise execution;
- intra-layer tiling;
- asynchronous prefetch and writeback;
- additional elapsed time.

Minimum capacity relationships are:

```text
model state + checkpoints + metadata <= allocated NVMe capacity

active parameter tiles + activations + workspaces + runtime buffers
    <= safe accelerator working set

process state + staging + operating overhead
    remains below safe unified-memory pressure
```

These conditions are necessary but not sufficient. A viable schedule must also satisfy numerical, bandwidth, endurance, recovery, and usefulness constraints.

## 5. Goals

MicroColossus aims to:

1. train models whose complete state cannot remain safely resident;
2. keep only the active working set available to accelerator execution;
3. keep inactive parameter and optimizer chunks on NVMe;
4. apply explicit memory, storage, write, and time budgets;
5. recompute selected activations;
6. tile an operation when it cannot fit as a whole;
7. preserve a full-parameter reference path;
8. track identity, version, location, and integrity of managed chunks;
9. publish optimizer steps atomically;
10. recover the last committed state after interruption;
11. report numerical distance, time, memory pressure, storage traffic, and writes.

## 6. Boundaries

The project does not claim:

- literally infinite models;
- throughput equal to a well-provisioned cluster;
- support for arbitrary PyTorch or MLX graphs in the initial runtime;
- feasibility for every architecture;
- that CPU-to-MPS placement increases capacity;
- that offloading, checkpointing, quantization, or tiling are new ideas;
- success at a scale target before a reproducible demonstration.

Parameter count alone is not a sufficient success metric.

## 7. Training modes

Modes remain separate.

- **`reference`**: all parameters are trainable. No silent adapters, low-rank gradient projections, or quantized optimizer states.
- **`compact`**: future declared approximations such as quantized optimizer state or low-rank optimizer methods.
- **`adapter`**: future LoRA or QLoRA path, reported separately from full-parameter execution.

Only `reference` is currently implemented.

## 8. Architecture

```text
Dataset stream
      |
Controlled Transformer frontend
      |
Tensor and dependency analysis
      |
Unified-memory-aware planner
      |
Execution schedule
      |
+----------------------------------+
| Runtime                          |
| - backend executor               |
| - storage transfer engine        |
| - activation manager             |
| - optimizer engine               |
| - checkpoint coordinator         |
| - telemetry                      |
+----------------------------------+
      |
bounded unified-memory working set <-> versioned NVMe tensor store
```

The intended components are:

- a controlled frontend with a small validated operator set;
- a backend-neutral tensor manifest;
- a planner for placement, tiling, prefetch, eviction, and recomputation;
- bounded staging and accelerator working-set policies;
- a chunked, versioned, recoverable tensor store;
- layer-wise and tile-wise forward, backward, and optimizer execution;
- telemetry for memory, pressure, storage traffic, stalls, throughput, and writes.

## 9. Backend strategy

The current decision is **DUAL BACKEND**.

- **MLX** is the preferred native Apple Silicon execution candidate.
- **PyTorch MPS** remains the portable numerical oracle and reference implementation.
- **Storage and transaction logic** must remain backend-neutral.

The decision is based on resident measurements, not storage-backed execution. The storage milestone must compare both adapters before assigning all future execution responsibilities.

## 10. Current implementation

Implemented:

- typed YAML configuration;
- controlled decoder-only Transformer;
- deterministic synthetic training data;
- resident full-parameter AdamW training;
- CPU, MPS, CUDA, and automatic device selection;
- real MPS diagnostics and synchronized telemetry;
- NumPy-independent training checksums;
- static memory estimation with unified-memory warnings;
- atomic JSON and JSONL run artifacts;
- backend-neutral competitive benchmark schemas;
- portable identical benchmark weights and batches;
- synchronized PyTorch resident benchmark;
- PyTorch activation-checkpointing benchmark;
- equivalent MLX resident benchmark;
- atomic final-state `.state.npz` artifacts;
- tensor-level final-state comparison;
- tests, linting, type checking, compilation, and CI smoke runs.

Not implemented:

- storage-backed model or optimizer state;
- NVMe-to-accelerator execution;
- runtime-managed activation recomputation;
- asynchronous storage overlap;
- hard runtime budget enforcement;
- intra-layer tiling;
- tensor journal and crash recovery;
- training state larger than safe resident unified memory.

## 11. Verified evidence

### Resident MPS foundation

A clean independent run on commit `a56fc514f2f8e705654034f3c2f02e3a441c61f3` passed on a MacBook Air M2 with 8 GB unified memory.

It validated:

- native arm64 execution;
- project checks;
- MPS preflight;
- explicit MPS training;
- automatic MPS selection;
- CPU-versus-MPS comparison;
- fixed-batch learning from `5.566842079162598` to `0.4145740866661072`;
- no detected fallback, unsupported operator, non-finite value, or MPS OOM;
- a clean source tree.

MPS was not bitwise reproducible by final checksum. Numerical reproducibility is evaluated with tensor-level tolerances and trajectories.

### Competitive resident result

A clean version `0.3.2` run completed all resident PyTorch MPS, checkpointed PyTorch MPS, and MLX variants on a 23,213,056-parameter workload.

Median throughput:

| Variant | Tokens/s |
|---|---:|
| PyTorch MPS | 1,379.23 |
| PyTorch MPS checkpointed | 1,337.37 |
| MLX | 2,195.69 |

MLX was approximately `1.592x` faster than PyTorch MPS in the measured resident workload.

PyTorch and MLX remained numerically close:

- maximum loss difference: about `1.91e-06`;
- maximum final-parameter absolute difference: about `3.80e-05`;
- mean final-parameter absolute difference: about `7.38e-09`;
- all values finite and stable across three rounds.

### Final release-quality gate

Version `0.3.3`, commit `b75d2f646da4ca4dce5acdee567a1f17adcc503c`, passed a fresh target-machine verification:

- Ruff passed;
- mypy passed with no issues in 17 source files;
- pytest passed with 28 tests;
- compileall passed;
- doctor detected Apple M2 MPS;
- all three tiny backend variants passed;
- state and batch checksums were equal;
- JSON and `.state.npz` artifacts were valid;
- no fallback or unsupported operator evidence was reported;
- the working tree remained clean.

This evidence validates resident execution and the benchmark contract. It does not validate out-of-core execution.

## 12. Competitive engineering policy

MicroColossus must remain competitive by measurement.

Direct Apple Silicon baselines include:

1. resident PyTorch MPS;
2. PyTorch MPS with activation checkpointing;
3. native MLX with an equivalent controlled Transformer;
4. compiled MLX when justified;
5. MLX-LM full-model fine-tuning when semantically comparable;
6. MicroColossus reference execution;
7. compact and adapter modes, reported separately.

An apples-to-apples comparison should hold constant:

- architecture and unique parameter count;
- initial parameter arrays and input batches;
- dtype and optimizer semantics;
- clipping, sequence length, and microbatch;
- warm-up and measured update count;
- synchronization points;
- machine, power state, macOS, and framework versions.

Required metrics include:

- numerical distance;
- first-step and steady-state latency;
- tokens per second;
- process RSS;
- framework and driver allocator counters;
- available memory, pressure, compression, and swap;
- storage reads and writes;
- SSD write amplification;
- checkpoint and recovery cost.

An optimization is accepted only when it improves a declared objective without violating correctness, memory, endurance, stability, recovery, or reproducibility constraints.

## 13. Tensor manifest and transaction contract

The next implementation uses a backend-neutral logical model.

Each tensor record must identify:

```text
tensor_id
logical_name
kind
shape
dtype
version
chunk_ids
byte_length
checksum
committed_step
```

Initial tensor kinds include:

```text
parameter
gradient
adam_first_moment
adam_second_moment
master_weight
metadata
```

Each chunk record must identify:

```text
chunk_id
storage_path
byte_offset
byte_length
checksum
compression
creation_transaction
```

Each transaction must identify:

```text
transaction_id
parent_manifest
candidate_manifest
created_chunks
expected_checksums
state
```

Transaction states begin with:

```text
prepared
writing
validated
committed
aborted
```

A transaction cannot publish a new manifest until every required chunk is durable and checksum-valid.

## 14. Storage and recovery requirements

The tensor store must provide:

- immutable or copy-on-write chunks;
- logical tensor versions;
- per-chunk checksums;
- write-ahead journaling;
- atomic manifest publication;
- incremental checkpoints;
- bounded staging buffers;
- read and write telemetry;
- failure-injection tests.

A step is committed only after all required chunks and the new manifest are durable. Until then, the previous committed version remains authoritative.

Recovery must distinguish:

- the last committed manifest;
- complete but unpublished chunks;
- partial chunk writes;
- checksum failures;
- aborted transactions;
- stale cache entries.

## 15. Roadmap

### M0. Resident foundation

Status: completed.

### M1. Clean Mac M2 validation

Status: completed.

### M2. Competitive Apple Silicon baseline

Status: completed.

Established:

- equivalent PyTorch and MLX workloads;
- target-hardware correctness and artifact validation;
- resident performance comparison;
- dual-backend decision;
- clean release-quality verification.

### M3. Versioned NVMe tensor store

Status: next.

- stable tensor, chunk, manifest, and transaction schemas;
- checksums and copy-on-write versions;
- write-ahead journal;
- atomic manifest publication;
- recovery and failure injection;
- bounded staging;
- storage telemetry;
- PyTorch and MLX export and restore adapters.

### M4. Synchronous storage-to-accelerator execution

- authoritative state on NVMe;
- layer-wise loading into a bounded working set;
- forward, backward, and optimizer execution;
- atomic step publication;
- resident-versus-streamed comparison.

### M5. Recomputation and strict budgets

- activation checkpoint planning;
- deterministic recomputation;
- memory, pressure, storage, write, and time rejection rules.

### M6. Asynchronous overlap

- double buffering;
- prefetch and writeback;
- stall measurement;
- storage and accelerator overlap.

### M7. Intra-layer tiling

- linear and MLP tiles;
- partitioned embeddings;
- tiled output projection and loss;
- normalization and attention paths.

### M8. Constrained-hardware demonstration

- validate around 124 million parameters;
- attempt full-parameter training around 350 million parameters on the 8 GB M2 target;
- investigate larger targets only after correctness and usefulness are measured.

Scale targets are research goals, not guaranteed outcomes.

## 16. Success criteria

The first meaningful storage-backed milestone requires:

1. the active working set remains inside configured limits;
2. the complete state exceeds safe resident capacity;
3. updates are compared with the resident reference;
4. every managed tensor version is traceable;
5. storage traffic and writes are reported;
6. interruption recovers the last committed state;
7. the result is reproducible;
8. throughput and endurance remain explicit, not hidden behind parameter count.
