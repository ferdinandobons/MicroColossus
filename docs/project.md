# MicroColossus Project Specification

> **Trade memory for time.**

This document defines the purpose, constraints, architecture, evidence, and development path of MicroColossus.

## 1. Purpose

MicroColossus explores full-parameter training when the complete training state cannot remain safely resident in available memory.

The central question is:

> How can a runtime execute valid model updates while active computation remains inside explicit memory, storage, time, correctness, recovery, and endurance budgets?

The primary target is Apple Silicon, beginning with a MacBook Air M2 with 8 GB unified memory and local NVMe storage.

MicroColossus does not promise infinite models, datacenter throughput, or elimination of compute cost. Storage capacity, memory pressure, bandwidth, minimum useful tile size, SSD endurance, and elapsed time remain physical constraints.

## 2. Why the project exists

Training requires more than model weights. A full update may require:

- parameter gradients;
- Adam first and second moments;
- optional master-weight copies;
- activations or recomputation metadata;
- logits and loss intermediates;
- operator workspaces;
- framework, driver, and operating-system overhead.

A machine can have enough storage for the complete state but insufficient safe resident memory. MicroColossus treats that mismatch as a scheduling and transaction problem whose complete cost must be measured.

## 3. Apple Silicon design rules

Apple Silicon CPU and GPU workloads share one physical memory pool.

### Placement is not capacity offload

Moving a tensor from CPU execution to MPS changes placement and execution. It does not create another physical memory pool. Keeping CPU and accelerator copies can increase pressure on the same unified memory.

### Memory counters overlap

Process RSS, PyTorch MPS allocation, Metal driver allocation, MLX allocator values, compressed memory, and swap describe different or overlapping scopes. They are never added as separate physical capacities.

### NVMe is the first external capacity tier

```text
MLX or MPS active working set
              |
bounded unified-memory staging and activations
              |
versioned NVMe tensor store
```

### Memory pressure is a planning input

The planner must consider active tensors, process footprint, macOS pressure, compression, swap, storage traffic, SSD writes, and step time together.

## 4. Central hypothesis

For a useful class of Transformer models and training recipes, part of resident-memory demand can be exchanged for:

- versioned tensor storage;
- bounded staging and caches;
- parameter-group execution;
- activation recomputation;
- intra-layer tiling;
- asynchronous prefetch and writeback;
- additional elapsed time.

Necessary capacity relationships include:

```text
model state + checkpoints + metadata <= allocated storage capacity

active parameter group or tile + activations + workspaces + runtime buffers
    <= safe execution working set

process state + staging + operating overhead
    remains below safe unified-memory pressure
```

These conditions are necessary but not sufficient. A viable schedule must also satisfy numerical, bandwidth, endurance, recovery, and usefulness constraints.

## 5. Goals

MicroColossus aims to:

1. train models whose complete state cannot remain safely resident;
2. keep only the active parameter group or tile available to accelerator execution;
3. keep inactive parameter and optimizer chunks on NVMe;
4. apply explicit memory, storage, write, and time budgets;
5. recompute or offload selected activations;
6. tile an operation when it cannot fit as a whole;
7. preserve a full-parameter numerical reference path;
8. track identity, version, location, and integrity of every managed tensor;
9. publish optimizer steps atomically;
10. recover the last committed state after interruption;
11. report numerical distance, memory pressure, storage traffic, elapsed time, and writes.

## 6. Boundaries

The project does not claim:

- literally infinite models;
- throughput equal to a well-provisioned cluster;
- support for arbitrary PyTorch or MLX graphs in the first runtime;
- feasibility for every architecture;
- that CPU-to-MPS placement increases capacity;
- that offloading, checkpointing, tiling, or quantization are new ideas;
- success at a scale target before a reproducible demonstration.

Parameter count alone is not a sufficient success metric.

## 7. Training modes

Modes remain separate.

- **`reference`**: all parameters are trainable. No silent adapters, low-rank gradient projection, or quantized optimizer state.
- **`compact`**: future declared approximations such as quantized optimizer state or low-rank optimizer methods.
- **`adapter`**: future LoRA or QLoRA path, reported separately from full-parameter execution.

Only `reference` is currently implemented.

## 8. Architecture

```text
Dataset or deterministic batch stream
                 |
Controlled Transformer frontend
                 |
Tensor and dependency analysis
                 |
Unified-memory-aware execution plan
                 |
+------------------------------------------------+
| Runtime                                        |
| - backend executor                             |
| - versioned tensor store                       |
| - transfer and staging coordinator             |
| - activation manager                           |
| - optimizer coordinator                        |
| - transaction and recovery coordinator         |
| - telemetry                                    |
+------------------------------------------------+
                 |
bounded active working set <-> versioned NVMe tensor store
```

The intended components are:

- a controlled frontend with a small validated operator set;
- backend-neutral tensor and execution-plan schemas;
- placement, tiling, prefetch, eviction, and recomputation planning;
- bounded staging and accelerator working-set policies;
- a chunked, versioned, recoverable tensor store;
- group-wise and tile-wise forward, backward, and optimizer execution;
- telemetry for memory, pressure, storage traffic, stalls, throughput, and writes.

## 9. Backend strategy

The current decision is **DUAL BACKEND**.

- **MLX** is the preferred optimized Apple Silicon execution candidate.
- **PyTorch MPS** is the portable numerical oracle and reference implementation.
- **Storage, transactions, and execution plans** remain backend-neutral.

The decision is based on resident measurements. Every storage-backed path must be benchmarked independently before assigning all runtime responsibilities to one backend.

## 10. Current implementation

### Resident and competitive execution

Implemented:

- typed YAML configuration;
- controlled decoder-only Transformer;
- deterministic synthetic training data;
- resident full-parameter AdamW training;
- CPU, MPS, CUDA, and automatic device selection;
- MPS diagnostics and synchronized telemetry;
- static memory estimation with unified-memory warnings;
- resident PyTorch MPS and MLX benchmark paths;
- activation-checkpointed PyTorch baseline;
- portable identical FP32 weights and batches;
- tensor-level numerical comparison;
- machine-readable run artifacts.

### Versioned tensor store

Implemented:

- tensor, chunk, manifest, journal, and telemetry schemas;
- canonical little-endian tensor bytes;
- content-addressed immutable chunks;
- copy-on-write tensor versions;
- per-chunk and whole-tensor checksums;
- immutable manifests and atomic `CURRENT` publication;
- write-ahead transaction journals;
- storage and staging-budget enforcement;
- corruption detection;
- conservative recovery and failure injection;
- read, write, range-read, fsync, publication, recovery, and cumulative-write telemetry;
- PyTorch model and AdamW export and restore;
- MLX model and optimizer-tree export and restore.

### Observable storage-backed optimizer lifecycle

Implemented:

- deterministic canonical bootstrap state;
- initial storage commit;
- destruction of bootstrap framework objects;
- independently restored resident reference step;
- storage restore, forward, backward, clipping, and AdamW step;
- tensor-level resident-versus-storage comparison;
- atomic publication of updated model and optimizer state;
- post-commit restoration and exact verification;
- detailed compute, memory, I/O, checksum, fsync, and publication telemetry.

This lifecycle still materializes the complete micro model and optimizer during compute.

### Bounded parameter-group forward

Implemented and validated on the target M2:

- parameter-only canonical store;
- execution groups for embeddings, each Transformer block, and the final head;
- one parameter group materialized at a time;
- tied token-embedding reload for the output projection;
- logical parameter working-set budget rejection;
- resident boundary, logits, and loss comparison;
- per-group read, materialization, compute, release, activation, RSS, accelerator, and driver telemetry;
- immutable parameter-store verification.

The first bounded forward executor retains hidden activations.

### Bounded backward and gradient storage

Implemented and validated on the target M2 in version 0.7:

- detached boundary activations retained on CPU;
- reverse execution-group order;
- one parameter group reloaded at a time;
- local forward recomputation with autograd enabled;
- one incoming activation gradient propagated at a time;
- parameter gradients committed into a separate versioned gradient store;
- tied token-embedding contribution accumulation with explicit gradient versioning;
- streamed global gradient-norm calculation;
- final gradient comparison with the resident PyTorch oracle;
- immutable parameter manifest;
- parameter and gradient working-set budgets;
- per-group storage, compute, release, checksum, RSS, accelerator, and driver telemetry.

The complete final gradient state is materialized after bounded execution only for validation.

### Streamed AdamW and atomic step publication

Implemented in version 0.8, pending target validation:

- canonical global clipping coefficient shared by resident and bounded paths;
- unique parameter-group execution that updates tied weights exactly once;
- parameter, gradient, first-moment, second-moment, and step-state streaming;
- candidate parameter and optimizer stores built group by group;
- exact CPU comparison with a resident PyTorch AdamW oracle;
- candidate restore and re-export verification;
- atomic root step-bundle manifest and `CURRENT` pointer;
- failure injection before root manifest and pointer publication;
- logical optimizer working-set budget and per-group telemetry.

### Not implemented

- multiple consecutive bounded optimizer steps;
- checkpoint and resume for the bounded runtime;
- activation recomputation from storage or activation offload;
- asynchronous storage overlap;
- intra-layer tiling;
- bounded MLX backward;
- direct I/O or compression;
- real-corpus training frontend;
- training state larger than safe resident unified memory.

## 11. Verified evidence

### Resident MPS foundation

A clean MacBook Air M2 run with 8 GB unified memory validated native arm64 MPS training, device selection, telemetry, CPU-versus-MPS comparison, and fixed-batch learning from `5.566842079162598` to `0.4145740866661072`.

### Competitive resident result

A controlled 23,213,056-parameter workload produced:

| Variant | Median tokens/s | Relative to PyTorch |
|---|---:|---:|
| PyTorch MPS | 1,379.23 | 1.000x |
| PyTorch MPS checkpointed | 1,337.37 | 0.970x |
| MLX | 2,195.69 | 1.592x |

PyTorch and MLX remained numerically close. Maximum loss difference was about `1.91e-06`, maximum final-parameter absolute difference was about `3.80e-05`, and all values remained finite across three rounds.

### Clean target storage validation

Version 0.5.0, commit `82e53c671848d231c2361443882b97dbe4e3a408`, passed on a clean MacBook Air M2. It validated exact storage lifecycle, recovery, failure injection, MLX round trips, and cross-backend canonical state for the tested micro and tiny paths.

### Clean target bounded-forward validation

Version 0.6.0, commit `1feea9f9eef28e551ad4ae4944614083effa804f`, passed on the same 8 GB M2 target:

- Ruff, mypy, 56 tests, and compileall passed;
- two micro runs were GREEN and bitwise exact;
- the 443,648-parameter tiny run was GREEN;
- group order and tied-embedding reload were correct;
- maximum group sizes were `33,280` bytes for micro and `788,480` bytes for tiny under a `1,048,576` byte budget;
- boundary, logits, and loss differences were zero;
- intentional budget rejection passed;
- parameter manifests remained unchanged;
- stores verified and recovered;
- no fallback, unsupported operation, non-finite value, allocation failure, or swap growth was detected;
- the source tree remained clean.

### Clean target bounded-backward validation

Version 0.7.0, commit `c72dcc2f8d8a7bd783ae263cf14476d0681b664b`, passed on the same target. Two micro bounded-backward runs were GREEN and bitwise exact; the tiny run was GREEN; maximum loss and tensor-gradient differences were zero; maximum norm difference was about `9.57e-08`; tied-gradient accumulation, budget rejection, store recovery, no-fallback checks, and clean source integrity all passed.

## 12. Competitive engineering policy

MicroColossus remains competitive by measurement.

Direct Apple Silicon baselines include:

1. resident PyTorch MPS;
2. checkpointed PyTorch MPS;
3. equivalent MLX execution;
4. compiled MLX when justified;
5. semantically comparable MLX-LM full-model fine-tuning;
6. MicroColossus reference execution;
7. future compact and adapter modes, reported separately.

An accepted comparison holds constant architecture, parameter count, initial arrays, input batches, dtype, optimizer semantics, clipping, sequence length, microbatch, synchronization, machine, and framework versions.

Required metrics include numerical distance, latency, throughput, RSS, allocator counters, available memory, pressure, swap, storage reads and writes, estimated endurance, and recovery cost.

An optimization is accepted only when it improves a declared objective without violating correctness, memory, endurance, stability, recovery, or reproducibility.

## 13. Versioned transaction contract

A tensor record identifies:

```text
tensor_id
logical_name
kind
shape
dtype
byte_order
version
chunk_ids
byte_length
checksum
committed_step
```

A chunk record identifies:

```text
chunk_id
storage_path
byte_offset
byte_length
checksum
compression
creation_transaction
```

A transaction identifies:

```text
transaction_id
parent_manifest
candidate_manifest
created_chunks
expected_checksums
state
```

Transaction states are:

```text
prepared
writing
validated
committed
aborted
```

A transaction cannot publish a new manifest until all required chunks are durable and checksum-valid. Until publication, the previous committed manifest remains authoritative.

## 14. Development scale ladder

1. **Unit scale**. Bytes, arrays, operators, corruption, and failure injection.
2. **Micro model**. A sub-million-parameter Transformer for every complete runtime path.
3. **Small real training**. A few-million-parameter model on a small text corpus with training and validation loss, checkpoint and resume, and sample generation.
4. **Milestone scale**. Larger runs only after the same path is correct at smaller scales.
5. **Capacity demonstration**. 124M, 350M, and larger targets after bounded storage-backed training exists.

A future external training project may become the real-training frontend. Storage, scheduling, transactions, and backend interfaces remain independent of it.

## 15. Roadmap

### M0. Resident foundation

Status: completed.

### M1. Clean Mac M2 validation

Status: completed.

### M2. Competitive Apple Silicon baseline

Status: completed.

### M3. Versioned tensor store

Status: completed.

### M4A. Observable storage-backed optimizer lifecycle

Status: completed.

### M4B1. Bounded parameter-group forward

Status: completed, including target M2 validation.

### M4B2. Bounded backward and gradient store

Status: completed, including target M2 validation.

This phase stores final parameter gradients separately, validates tied-gradient accumulation, and calculates the global gradient norm without changing parameters.

### M4B3. Streamed AdamW and atomic step publication

Status: implemented in version 0.8. Target M2 validation pending.

The reference implementation:

1. read each parameter, final gradient, Adam first moment, Adam second moment, and step state group by group;
2. apply the global clipping coefficient;
3. calculate AdamW updates without retaining the full optimizer state;
4. write candidate parameter and optimizer versions;
5. publish one complete atomic training-step manifest only after every group is durable and valid.

### M5. Activation recomputation and strict budgets

- deterministic activation recomputation;
- activation retention, offload, and rejection policies;
- memory, pressure, storage, write, and time budgets.

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

### M8. Small real-corpus training

- tokenizer and deterministic dataset cursor;
- train and validation loss;
- checkpoint and resume;
- periodic sample generation;
- resident-versus-storage comparison over many steps.

### M9. Constrained-hardware demonstrations

- validate around 124 million parameters;
- attempt full-parameter training around 350 million parameters on the 8 GB M2 target;
- investigate larger targets only after correctness and usefulness are measured.

Scale targets are research goals, not guaranteed outcomes.

## 16. Success criteria

The first meaningful out-of-core training demonstration requires:

1. the complete managed training state exceeds safe resident capacity;
2. the active parameter, activation, gradient, optimizer, and workspace set remains inside configured limits;
3. updates are compared with the resident reference where feasible;
4. every managed tensor version is traceable;
5. storage traffic and writes are reported;
6. interruption recovers the last committed state;
7. the result is reproducible;
8. throughput and endurance remain explicit, not hidden behind parameter count.
