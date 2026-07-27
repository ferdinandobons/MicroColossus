# MicroColossus Project Specification

> **Trade memory for time.**

This document is the authoritative overview of the purpose, architecture, design decisions, implemented milestones, validation boundary, and development path of MicroColossus.

## 1. Purpose

MicroColossus is an experimental runtime for full-parameter training when the complete managed training state cannot remain safely resident in available memory.

The central engineering question is:

> Can a correct and recoverable training update be executed while only a bounded working set is materialized, with inactive parameters, gradients, and optimizer state stored on NVMe?

The primary target is Apple Silicon, beginning with a MacBook Air M2 with 8 GB of unified memory and local NVMe storage.

The project does not promise infinite models, datacenter throughput, or the removal of physical constraints. Storage capacity, unified-memory pressure, I/O bandwidth, minimum useful tile size, SSD endurance, elapsed time, and numerical behavior remain explicit constraints.

## 2. Why the project exists

Training requires substantially more state than model weights alone. A full AdamW update can include:

- model parameters;
- parameter gradients;
- Adam first moments;
- Adam second moments;
- optimizer step tensors and parameter-group metadata;
- optional master-weight copies;
- activations or recomputation metadata;
- logits, loss intermediates, and operator workspaces;
- framework, driver, filesystem-cache, and operating-system overhead.

A machine can have enough storage for this complete state while lacking enough safe resident memory. MicroColossus treats that mismatch as a scheduling, storage, transaction, and observability problem.

## 3. Apple Silicon design rules

Apple Silicon uses one physical unified-memory pool. CPU execution, GPU execution through Metal, framework allocators, the Metal driver, and macOS compete for the same capacity.

### 3.1 Placement is not capacity offload

Moving a tensor between CPU execution and MPS changes placement and execution. It does not create an independent physical memory tier. Holding CPU and accelerator copies can increase pressure on the same unified memory.

### 3.2 Memory counters are not additive capacities

Process RSS, PyTorch MPS allocation, Metal driver allocation, MLX allocator counters, compressed memory, and swap describe different or overlapping scopes. MicroColossus reports them separately and never adds them as separate physical memories.

### 3.3 NVMe is the first external capacity tier

```text
MLX or MPS active working set
              |
bounded unified-memory staging and activations
              |
versioned NVMe tensor store
```

### 3.4 Memory pressure is part of the execution plan

A valid plan must consider more than tensor byte counts. It must also consider process footprint, macOS memory pressure, compression, swap, filesystem cache, storage traffic, cumulative writes, recovery cost, and elapsed time.

## 4. Central hypothesis

For a useful class of controlled Transformer workloads, part of resident-memory demand can be exchanged for:

- versioned storage of inactive tensors;
- bounded parameter, gradient, and optimizer staging;
- group-wise forward, backward, and optimizer execution;
- activation retention, recomputation, or offload;
- intra-layer tiling when a complete operator does not fit;
- asynchronous prefetch and writeback;
- additional elapsed time.

Necessary relationships include:

```text
complete managed state + checkpoints + metadata
    <= allocated storage capacity

active parameter or tile + local activations + gradients + optimizer state
    + workspaces + runtime buffers
    <= safe working-set budget

process state + staging + operating-system overhead
    remains below safe unified-memory pressure
```

These conditions are necessary but not sufficient. A useful schedule must also satisfy numerical, bandwidth, endurance, recovery, and throughput constraints.

## 5. Goals and non-goals

### 5.1 Goals

MicroColossus aims to:

1. train models whose complete managed state cannot remain safely resident;
2. keep only the active parameter group or tile available to accelerator execution;
3. keep inactive parameter and optimizer chunks on NVMe;
4. enforce explicit parameter, gradient, optimizer, activation, storage, write, and time budgets;
5. preserve a full-parameter numerical reference path;
6. track identity, version, location, lineage, and integrity of every managed tensor;
7. publish complete optimizer steps atomically;
8. recover the last committed state after interruption;
9. measure numerical distance, memory pressure, storage traffic, stalls, elapsed time, and cumulative writes;
10. remain competitive with relevant resident and offload baselines by measurement.

### 5.2 Non-goals

The project does not claim:

- literally unlimited model size;
- throughput equal to a well-provisioned cluster;
- support for arbitrary PyTorch or MLX graphs in the first runtime;
- feasibility for every architecture or optimizer;
- that CPU-to-MPS placement increases capacity;
- that offloading, checkpointing, tiling, or quantization are new concepts;
- success at a scale target before a reproducible demonstration.

Parameter count alone is not a success metric.

## 6. Training modes

Training modes remain explicitly separate.

- **`reference`**: all parameters are trainable. No silent adapters, low-rank gradient projection, or quantized optimizer state.
- **`compact`**: future declared approximations such as quantized optimizer state or low-rank optimizer methods.
- **`adapter`**: future LoRA or QLoRA paths, reported separately from full-parameter execution.

Only `reference` is currently implemented.

## 7. High-level architecture

```text
Dataset or deterministic batch stream
                 |
Controlled Transformer frontend
                 |
Tensor identity and dependency model
                 |
Unified-memory-aware execution plan
                 |
+------------------------------------------------------+
| Runtime                                              |
| - backend executor                                   |
| - execution-group coordinator                        |
| - versioned tensor store                             |
| - parameter, gradient, and optimizer coordinators    |
| - activation manager                                 |
| - transaction and recovery coordinator               |
| - telemetry and validation                           |
+------------------------------------------------------+
                 |
bounded active working set <-> versioned NVMe state
```

The current implementation intentionally uses a controlled decoder-only Transformer. A smaller validated operator set makes exact state comparison, group decomposition, failure injection, and recovery tractable.

## 8. Backend strategy

The current decision is **DUAL BACKEND**.

- **MLX** is the preferred optimized Apple Silicon execution candidate.
- **PyTorch MPS** is the numerical oracle and reference implementation for debugging, state comparison, and recovery semantics.
- **Storage, transactions, tensor identity, and execution plans** remain backend-neutral.

This decision is based on resident measurements. Storage-backed paths must be benchmarked independently because their bottlenecks may differ from resident training.

## 9. Controlled workloads and development scale

MicroColossus uses several workload sizes for different purposes.

| Workload | Parameters | Purpose |
|---|---:|---|
| Micro storage model | 11,456 | Fast end-to-end storage, recovery, bounded-execution, and failure tests |
| Tiny model | 443,648 | Target-hardware numerical and telemetry gate |
| Competitive resident model | 23,213,056 | PyTorch MPS, checkpointed PyTorch, and MLX performance comparison |
| Future small real model | About 1M to 5M | Multi-step real-corpus training, checkpoint, resume, and samples |
| Future capacity models | 124M, 350M, and later targets | Demonstrate state larger than safe resident memory |

The current deterministic synthetic batches are used to isolate runtime correctness. They are not a substitute for real language-model training.

The scale ladder is:

1. unit tensors and operators;
2. micro end-to-end paths;
3. tiny target-hardware validation;
4. small real training;
5. larger milestone benchmarks;
6. capacity demonstrations.

## 10. Implemented architecture by release

| Release | Main capability | Validation status |
|---|---|---|
| 0.2.x | Native Apple MPS resident training, diagnostics, telemetry, and numerical checks | Validated on an 8 GB M2 |
| 0.3.x | Identical portable state and batches across PyTorch MPS and MLX, competitive benchmark harness | Validated on an 8 GB M2 |
| 0.4.x | Backend-neutral versioned tensor store, transactions, checksums, recovery, and adapters | Validated in CI and later target storage tests |
| 0.5.0 | Observable storage-backed optimizer lifecycle with exact restore and failure injection | Validated on an 8 GB M2 |
| 0.6.0 | Parameter-group bounded forward | Validated on an 8 GB M2 |
| 0.7.0 | Reverse group-bounded backward and versioned gradient store | Validated on an 8 GB M2 |
| 0.8.0 | Group-bounded AdamW and atomic root step-bundle publication | CPU CI passed. Target MPS validation pending |

Current `main` implementation commit for 0.8.0:

```text
ef88198d66f1d1795ffa14dcb6db388ae1715e85
```

## 11. Versioned state architecture

### 11.1 Canonical tensor state

Every managed tensor has a backend-neutral representation containing:

```text
tensor_id
logical_name
kind
shape
dtype
byte_order
version
ordered chunk IDs
byte_length
whole-tensor checksum
committed_step
adapter metadata
```

Initial tensor kinds include parameters, gradients, Adam first moments, Adam second moments, optional master weights, and metadata.

### 11.2 Immutable chunks and copy-on-write versions

Tensor bytes are split into content-addressed immutable chunks. New tensor versions reuse unchanged chunks and write only new content. Published manifests are immutable.

### 11.3 Store transactions

A store transaction follows:

```text
prepared -> writing -> validated -> committed
                            \
                             -> aborted
```

The previous `CURRENT` manifest remains authoritative until all candidate chunks and tensors verify, the candidate manifest is durable, and `CURRENT` is replaced atomically.

### 11.4 Root step bundles

Version 0.8 adds a second atomicity level above child tensor stores. A root step bundle references:

- a parameter-store manifest;
- an optimizer-store manifest;
- an optional gradient-store manifest;
- the batch checksum;
- the committed step;
- the parent bundle;
- a root checksum.

Candidate parameter and optimizer stores can advance while being built. They are not authoritative training state until the root bundle is published.

## 12. Bounded execution design

### 12.1 Execution groups

The controlled Transformer is divided into:

```text
embedding
block-0
block-1
...
final-head
```

Each group has explicit tensor membership and logical byte size.

### 12.2 Tied token embedding

The token embedding is shared with the output projection.

The runtime handles this in three different phases:

- forward: the weight is loaded for embeddings, released, and loaded again for the output projection;
- backward: the output-head contribution is written first, then the embedding contribution is accumulated into gradient version 1;
- optimizer: the final accumulated gradient is used to update the shared parameter exactly once.

### 12.3 Bounded forward

`bounded-forward` materializes one parameter group at a time, records each boundary activation, and compares the boundaries, logits, and loss with a resident oracle.

### 12.4 Bounded backward

`bounded-backward` retains detached boundary activations on CPU, processes groups in reverse order, recomputes each local forward with autograd enabled, propagates one activation gradient, and commits final parameter gradients into a separate store.

### 12.5 Streamed global gradient norm

Final gradients are read from storage one tensor at a time to compute the global norm. The clipping coefficient is canonicalized as:

```text
min(1, max_norm / (global_norm + 1e-6))
```

The same coefficient is used by the resident oracle and the group-bounded optimizer path.

### 12.6 Group-bounded AdamW

`bounded-step` reads, for one unique parameter group at a time:

- parameters;
- final gradients;
- first moments;
- second moments;
- optimizer step tensors.

It applies the canonical clipping coefficient, runs AdamW, writes candidate parameter and optimizer versions, releases the group, validates the complete candidate state against a resident oracle, restores it, and publishes the root step bundle atomically.

## 13. Working-set budgets

The runtime currently enforces separate logical budgets for:

- parameter bytes in one execution group;
- gradient bytes in one backward group;
- combined parameter, gradient, first-moment, second-moment, and step bytes in one optimizer group;
- tensor-store staging and total managed storage.

A group is rejected before compute when it exceeds its declared budget.

These logical budgets do not represent total physical memory. Activations, operator workspaces, framework caches, RSS, page cache, and operating-system pressure remain separately measured.

## 14. Correctness and validation policy

MicroColossus uses different standards for different environments.

### CPU oracle

CPU tests require exact canonical bytes when the same operation order and optimizer semantics are used.

### MPS target

MPS tests report both:

- bitwise equality through checksums;
- tensor-level numerical distance.

A checksum difference alone is not treated as numerical failure.

### Validation-only full materialization

Some current paths materialize complete oracle and candidate states after bounded execution to compare every tensor. This is explicitly reported and excluded from the bounded execution claim.

## 15. Telemetry

The runtime reports:

- per-group parameter, gradient, moment, and step bytes;
- tensor and chunk reads;
- logical and physical application writes;
- chunk creation and reuse;
- read, materialization, compute, export, commit, release, checksum, `fsync`, and publication times;
- process RSS;
- MPS or CUDA allocation;
- Metal driver allocation;
- MLX allocator and cache values where applicable;
- system memory pressure and swap in target diagnostics;
- manifest IDs, tensor versions, lineage, and checksums;
- recovery actions and unpublished state.

Application byte counters are not presented as NAND-level SSD writes.

## 16. Important design decisions

### 16.1 Backend-neutral storage before optimized execution

Storage identity, transactions, and recovery were implemented independently from PyTorch and MLX. This prevents backend choice from controlling state durability.

### 16.2 Correct synchronous path before overlap

The project first implements synchronous read, compute, write, verify, and publish. Prefetch and asynchronous writeback are deferred until the synchronous result is correct and observable.

### 16.3 One new source of complexity per milestone

Forward, backward, gradient storage, global clipping, optimizer execution, and root publication were separated into distinct milestones. This keeps numerical or durability failures localizable.

### 16.4 Explicit tied-weight semantics

Shared parameters are never treated as accidental duplicate tensors. Their read reuse, gradient accumulation, and unique optimizer update are explicit and tested.

### 16.5 Transaction payload lifetime

Committed or aborted transactions release staged payload references so a live transaction object cannot retain a hidden complete copy of canonical state in RAM.

### 16.6 Small models for fast iteration

Micro and tiny models are the routine development gates. Larger models are reserved for performance and capacity milestones after correctness has been established.

## 17. Accepted target evidence

### Resident MPS foundation

A clean MacBook Air M2 run validated native MPS training, automatic device selection, telemetry, CPU-versus-MPS comparison, and fixed-batch learning from `5.566842079162598` to `0.4145740866661072`.

### Competitive resident result

For the 23,213,056-parameter workload:

| Variant | Median tokens/s | Relative to PyTorch |
|---|---:|---:|
| PyTorch MPS | 1,379.23 | 1.000x |
| PyTorch MPS checkpointed | 1,337.37 | 0.970x |
| MLX | 2,195.69 | 1.592x |

The backend decision remains dual backend.

### Version 0.5 storage lifecycle

A clean M2 run validated exact storage round trips, five failure-injection points, PyTorch storage-backed equality, MLX export and restore, and zero swap growth.

### Version 0.6 bounded forward

A clean M2 run validated micro and tiny bounded forward, exact boundaries, logits, and loss, tied-weight reload, budget rejection, and unchanged parameter manifests.

### Version 0.7 bounded backward

A clean M2 run validated reverse group order, exact gradient tensors for the tested paths, tied-gradient versioning, streamed norm, parameter and gradient budget rejection, three-store recovery, and zero swap growth.

### Version 0.8 bounded optimizer

CPU CI on Python 3.11 and 3.13 validates exact resident-versus-candidate state, exact candidate restore, optimizer budget rejection, root-bundle checksum verification, and failure recovery. Target MPS validation is the current hardware gate.

## 18. Current boundary

MicroColossus does not yet establish:

- multiple consecutive bounded optimizer steps;
- checkpoint and resume for a long bounded run;
- deterministic dataset cursor and RNG restoration across process restart;
- activation recomputation from storage or activation offload;
- strict total-memory-pressure enforcement;
- asynchronous prefetch or writeback;
- intra-layer tiling;
- bounded MLX backward or optimizer execution;
- a real-corpus training frontend;
- training state larger than safe resident unified memory;
- 124M or 350M full-parameter training on the target machine.

No full out-of-core training, throughput-at-scale, or model-capacity claim is made yet.

## 19. Roadmap

| Milestone | Status |
|---|---|
| M0. Resident foundation | Completed |
| M1. Clean Mac M2 validation | Completed |
| M2. Competitive Apple Silicon baseline | Completed |
| M3. Versioned tensor store | Completed |
| M4A. Observable storage-backed optimizer lifecycle | Completed |
| M4B1. Bounded parameter-group forward | Completed and validated on M2 |
| M4B2. Bounded backward and gradient store | Completed and validated on M2 |
| M4B3. Streamed AdamW and atomic step publication | Implemented. Target MPS validation pending |
| M4C. Consecutive steps, checkpoint, and resume | Next |
| M5. Activation recomputation, offload, and strict budgets | Planned |
| M6. Asynchronous prefetch and writeback | Planned |
| M7. Intra-layer tiling | Planned |
| M8. Small real-corpus training | Planned |
| M9. 124M, 350M, and larger capacity demonstrations | Planned |

## 20. Criteria for the first meaningful out-of-core result

A meaningful demonstration requires all of the following:

1. the complete managed training state exceeds safe resident capacity;
2. every active parameter, activation, gradient, optimizer, and workspace set remains inside declared limits;
3. multiple optimizer steps complete from prior committed state;
4. interruption recovers the last committed step;
5. resumed execution matches uninterrupted execution within declared tolerances;
6. every managed tensor version remains traceable;
7. storage traffic and cumulative writes are reported;
8. numerical behavior is compared with a reference where feasible;
9. throughput and endurance costs remain explicit;
10. the training trajectory is useful on a real dataset.

## 21. Documentation map

- [`project.md`](project.md): purpose, architecture, decisions, current state, and roadmap.
- [`storage.md`](storage.md): tensor-store, transaction, bundle, and bounded-execution design.
- [`validation.md`](validation.md): accepted evidence and exact validation boundaries.
- [`competitive.md`](competitive.md): backend and optimization decisions supported by measurements.
