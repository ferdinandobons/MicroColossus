# MicroColossus Project Specification

> **Trade memory for time.**

This document is the authoritative overview of the purpose, architecture, design decisions, implemented milestones, validation boundary, and development path of MicroColossus.

## 1. Purpose

MicroColossus is an experimental runtime for full-parameter training when the complete managed training state cannot remain safely resident in available memory.

The central engineering question is:

> Can a correct and recoverable training trajectory be executed while only a bounded working set is materialized, with inactive parameters, gradients, and optimizer state stored on NVMe?

The primary target is Apple Silicon, beginning with a MacBook Air M2 with 8 GB of unified memory and local NVMe storage.

The project does not promise unlimited model size, datacenter throughput, or removal of physical constraints. Storage capacity, unified-memory pressure, bandwidth, minimum useful group or tile size, SSD endurance, elapsed time, and numerical behavior remain explicit constraints.

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

A machine can have enough storage for the complete state while lacking enough safe resident memory. MicroColossus treats that mismatch as a scheduling, storage, transaction, and observability problem.

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
versioned NVMe tensor state
```

### 3.4 Memory pressure is part of the execution plan

A valid plan must consider more than logical tensor bytes. It must also consider process footprint, macOS pressure, compression, swap, filesystem cache, storage traffic, cumulative writes, recovery cost, and elapsed time.

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

These relationships are necessary but not sufficient. A useful schedule must also satisfy numerical, bandwidth, endurance, recovery, and throughput constraints.

## 5. Goals and non-goals

### 5.1 Goals

MicroColossus aims to:

1. train models whose complete managed state cannot remain safely resident;
2. keep only the active parameter group or tile available to accelerator execution;
3. keep inactive parameter and optimizer chunks on NVMe;
4. enforce parameter, gradient, optimizer, activation, storage, write, and time budgets;
5. preserve a full-parameter numerical reference path;
6. track identity, version, location, lineage, and integrity of every managed tensor;
7. publish complete optimizer steps atomically;
8. recover the last committed state after interruption;
9. resume a multi-step trajectory from persistent checkpoint state;
10. measure numerical distance, memory pressure, storage traffic, stalls, elapsed time, and cumulative writes;
11. remain competitive with relevant resident and offload baselines by measurement.

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
| - root checkpoint and resume coordinator             |
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
- **Storage, transactions, tensor identity, root checkpoints, and execution plans** remain backend-neutral.

This decision is based on resident measurements. Storage-backed paths must be benchmarked independently because their bottlenecks may differ from resident training.

## 9. Controlled workloads and development scale

| Workload | Parameters | Purpose |
|---|---:|---|
| Micro storage model | 11,456 | Fast end-to-end storage, recovery, bounded execution, and failure tests |
| Tiny model | 443,648 | Target-hardware numerical and telemetry gate |
| Competitive resident model | 23,213,056 | PyTorch MPS, checkpointed PyTorch, and MLX performance comparison |
| Future small real model | About 1M to 5M | Text training, validation, checkpoint, resume, and samples |
| Future capacity models | 124M, 350M, and later targets | Demonstrate state larger than safe resident memory |

Deterministic synthetic batches isolate runtime correctness. They are not a substitute for real language-model training.

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
| 0.3.x | Portable state and batches across PyTorch MPS and MLX, competitive benchmark harness | Validated on an 8 GB M2 |
| 0.4.x | Backend-neutral tensor store, transactions, checksums, recovery, and adapters | Validated in CI and target storage tests |
| 0.5.0 | Observable storage-backed optimizer lifecycle with exact restore and failure injection | Validated on an 8 GB M2 |
| 0.6.0 | Parameter-group bounded forward | Validated on an 8 GB M2 |
| 0.7.0 | Reverse group-bounded backward and versioned gradient store | Validated on an 8 GB M2 |
| 0.8.0 | Group-bounded AdamW and atomic root step-bundle publication | Validated on an 8 GB M2 |
| 0.9.0 | Persistent consecutive bounded steps, root lineage, checkpoint, and process resume | Validated on an 8 GB M2 |

Accepted 0.9 runtime commit:

```text
4b1ffb20857dd948d7737484e62b007f24bf69b9
```

## 11. Versioned state architecture

### 11.1 Canonical tensor state

Every managed tensor has a backend-neutral representation:

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

```text
prepared -> writing -> validated -> committed
                            \
                             -> aborted
```

The previous store `CURRENT` remains authoritative until candidate chunks and tensors verify, the candidate manifest is durable, and `CURRENT` is replaced atomically.

### 11.4 Root step bundles

A root bundle references:

- parameter-store manifest;
- optimizer-store manifest;
- optional gradient-store manifest;
- consumed batch checksum;
- committed training step;
- parent bundle;
- root checksum.

Candidate child stores can advance while being built. They are not authoritative training state until the root bundle is published.

### 11.5 Persistent training metadata

Version 0.9 adds checksummed `TRAINING.json` containing:

- semantic configuration digest;
- runtime version;
- training seed;
- batch-stream version;
- schedule kind.

A resumed run rejects incompatible model, optimizer, clipping, sequence, microbatch, seed, or schedule semantics.

## 12. Bounded execution design

### 12.1 Execution groups

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

- forward: loaded for embeddings, released, and loaded again for the output projection;
- backward: output-head contribution written first, then embedding contribution accumulated into the final gradient;
- optimizer: final accumulated gradient updates the shared parameter exactly once.

### 12.3 Bounded forward

`bounded-forward` materializes one parameter group at a time, records boundary activations, and compares boundaries, logits, and loss with a resident oracle.

### 12.4 Bounded backward

`bounded-backward` retains detached boundary activations on CPU, processes groups in reverse order, recomputes each local forward with autograd enabled, propagates one activation gradient, and commits final parameter gradients into a separate store.

### 12.5 Streamed global gradient norm

Final gradients are read one tensor at a time. The clipping coefficient is:

```text
min(1, max_norm / (global_norm + 1e-6))
```

The resident oracle and bounded optimizer use the same coefficient.

### 12.6 Group-bounded AdamW

For one unique group at a time, the runtime reads parameters, final gradients, first moments, second moments, and optimizer step tensors. It executes AdamW, writes candidate state, validates it, restores it, and publishes a root bundle atomically.

## 13. Persistent consecutive steps

Version 0.9 opens the current root and advances:

```text
step N -> step N+1
```

The next deterministic synthetic batch is derived from the current committed step. The new root parent is the prior current bundle. A later process can reopen the same root and continue without recreating parameter or Adam state.

The initial cursor contract is:

```text
batch seed = training seed + 1 + committed step
next cursor = committed step
```

The accepted target run demonstrated:

- an uninterrupted micro trajectory from step 0 to step 5;
- a micro trajectory to step 2 followed by process exit and resume to step 5;
- bitwise-exact final state between uninterrupted and resumed paths;
- a tiny trajectory from step 0 to step 3;
- correct root parent chains and batch cursors;
- final optimizer step tensors equal to the committed root step;
- atomic preservation of step 2 when publication of step 3 was interrupted.

Historical bundles remain available. Pruning and compaction are not implemented in 0.9.

## 14. Working-set budgets

The runtime enforces separate logical budgets for:

- parameter bytes in one group;
- gradient bytes in one backward group;
- parameter, gradient, first-moment, second-moment, and step bytes in one optimizer group;
- tensor-store staging and total managed storage.

A group is rejected before compute when it exceeds its declared budget.

Logical budgets do not represent total physical memory. Activations, workspaces, caches, RSS, page cache, and operating-system pressure remain separately measured.

## 15. Correctness and validation policy

### 15.1 CPU oracle

CPU tests require exact canonical state when operation order and optimizer semantics are equivalent.

### 15.2 MPS target

MPS tests report bitwise equality and tensor-level numerical distance separately. A checksum difference alone is not numerical failure.

### 15.3 Validation-only full materialization

Some paths materialize complete oracle and candidate states after bounded execution to compare every tensor. This is explicitly reported and excluded from bounded claims.

Version 0.9 also replays a resident reference from step zero through the target step. This replay is validation-only and not part of the intended production runtime.

## 16. Telemetry

The runtime reports:

- per-group parameter, gradient, moment, and step bytes;
- tensor and chunk reads;
- logical and physical application writes;
- chunk creation and reuse;
- read, materialization, compute, export, commit, release, checksum, `fsync`, and publication times;
- process RSS;
- accelerator and driver allocation;
- system memory pressure and swap in target diagnostics;
- manifest IDs, tensor versions, lineage, cursor, and checksums;
- recovery actions and unpublished state;
- cumulative storage growth across steps.

Application byte counters are not presented as NAND-level SSD writes.

## 17. Important design decisions

### 17.1 Backend-neutral durability

Storage identity, root checkpoints, transactions, and recovery remain independent from PyTorch and MLX.

### 17.2 Correct synchronous path before overlap

The project implements synchronous read, compute, write, verify, and publish before prefetch or asynchronous writeback.

### 17.3 One source of complexity per milestone

Forward, backward, gradient storage, clipping, optimizer execution, atomic publication, and multi-step resume are isolated so failures remain localizable.

### 17.4 Explicit shared-weight semantics

Shared parameters have explicit forward reads, gradient accumulation, and unique optimizer update rules.

### 17.5 Transaction payload lifetime

Committed or aborted transactions release staged payload references so a live transaction cannot retain a hidden full canonical copy.

### 17.6 Small models for fast iteration

Micro and tiny models are routine gates. Larger workloads are reserved for performance and capacity milestones.

### 17.7 Root step is authoritative

Child tensor-store transaction steps are internal publication sequence numbers. The root bundle committed step is the authoritative training-step number. Candidate tensor versions equal the root step being published.

### 17.8 Cursor and checkpoint authority are unified

The committed root step is the deterministic next-batch cursor. A separate mutable cursor file is intentionally avoided because it could diverge from authoritative model and optimizer state after interruption.

## 18. Accepted target evidence

### 18.1 Resident MPS foundation

A clean M2 run validated native MPS training, automatic device selection, telemetry, CPU-versus-MPS comparison, and fixed-batch learning from `5.566842079162598` to `0.4145740866661072`.

### 18.2 Competitive resident result

| Variant | Median tokens/s | Relative to PyTorch |
|---|---:|---:|
| PyTorch MPS | 1,379.23 | 1.000x |
| PyTorch MPS checkpointed | 1,337.37 | 0.970x |
| MLX | 2,195.69 | 1.592x |

### 18.3 Storage through bounded AdamW

Versions 0.5 through 0.8 validated exact storage lifecycle, failure injection, PyTorch and MLX canonical adapters, bounded forward, bounded backward, streamed gradient norm, group-bounded AdamW, exact candidate restore, and atomic step publication.

### 18.4 Version 0.9 persistent training

The clean M2 validation used commit `4b1ffb20857dd948d7737484e62b007f24bf69b9` and version 0.9.0.

Quality and environment:

- MacBook Air M2 with 8 GB unified memory;
- native arm64 without Rosetta;
- MPS built and available;
- Ruff, mypy, pytest, and compileall passed;
- 80 tests passed with one skip;
- fallback remained disabled;
- final source tree was clean.

Correctness and durability:

- micro uninterrupted step 0 to 5: GREEN;
- micro process restart resume step 2 to 5: GREEN;
- uninterrupted versus resumed final state: BITWISE_EXACT;
- tiny step 0 to 3: GREEN;
- maximum per-step loss difference: `0.0`;
- maximum per-step gradient-norm difference: `1.6985336648289717e-07`;
- maximum final-state absolute difference: `7.450580596923828e-09`;
- mean final-state absolute difference: `1.0105799101017887e-10`;
- candidate restore exactness: true;
- configuration mismatch rejection: PASS;
- corrupt-child detection: PASS;
- parameter, gradient, and optimizer budget rejection: PASS;
- both later-step root-publication failures preserved step 2;
- root and referenced child-store verification and recovery: PASS;
- swap delta: zero;
- no hidden fallback or unsupported operator evidence.

Resource and storage observations:

- maximum RSS: `410,943,488` bytes;
- maximum Metal driver allocation: `25,935,872` bytes;
- sampled MPS current allocation: `0` bytes, retained as reported;
- total successful-root storage size across the accepted scenarios: `59,901,060` bytes;
- parameter bytes read and written: `5,782,016` and `5,782,016`;
- gradient bytes read: `5,782,016`;
- optimizer bytes read and written: `11,564,752` and `11,572,248`;
- reused chunks: `167`;
- publication `fsync` total: `0.0074204159` seconds;
- rename and publication total: `0.0025823780` seconds.

## 19. Current boundary

MicroColossus does not yet establish:

- real tokenizer and corpus training;
- persisted real-dataset shard, epoch, shuffle, and sample state;
- activation recomputation from storage or activation offload;
- strict total-memory-pressure enforcement;
- asynchronous prefetch or writeback;
- intra-layer tiling;
- bounded MLX backward or optimizer execution;
- storage pruning or compaction;
- training state larger than safe resident unified memory;
- 124M or 350M full-parameter training on the target machine.

No full out-of-core, throughput-at-scale, model-quality, or model-capacity claim is made yet.

## 20. Roadmap

| Milestone | Status |
|---|---|
| M0. Resident foundation | Completed |
| M1. Clean Mac M2 validation | Completed |
| M2. Competitive Apple Silicon baseline | Completed |
| M3. Versioned tensor store | Completed |
| M4A. Observable storage-backed optimizer lifecycle | Completed |
| M4B1. Bounded parameter-group forward | Completed and validated on M2 |
| M4B2. Bounded backward and gradient store | Completed and validated on M2 |
| M4B3. Streamed AdamW and atomic step publication | Completed and validated on M2 |
| M4C. Consecutive steps, checkpoint, and resume | Completed and validated on M2 |
| M5. Small real-corpus training frontend | Next |
| M6. Activation recomputation, offload, and strict budgets | Planned |
| M7. Asynchronous prefetch and writeback | Planned |
| M8. Intra-layer tiling | Planned |
| M9. 124M, 350M, and larger capacity demonstrations | Planned |

## 21. Criteria for the first meaningful out-of-core result

A meaningful demonstration requires:

1. complete managed state larger than safe resident capacity;
2. every active parameter, activation, gradient, optimizer, and workspace set within declared limits;
3. multiple optimizer steps from prior committed state;
4. interruption recovery to the last committed step;
5. resumed execution matching uninterrupted execution within declared tolerances;
6. traceable managed tensor versions and root lineage;
7. reported storage traffic and cumulative writes;
8. numerical comparison with a reference where feasible;
9. explicit throughput and endurance costs;
10. a useful training trajectory on real data.

M4C establishes criteria 3 through 6 for the tested micro and tiny synthetic paths. It does not establish criteria 1, 2 for activations and workspaces, or criterion 10.

## 22. Documentation map

- [`project.md`](project.md): purpose, architecture, decisions, current state, and roadmap.
- [`storage.md`](storage.md): tensor-store, transaction, bundle, and bounded-execution design.
- [`multistep.md`](multistep.md): persistent checkpoint, cursor, lineage, resume design, and accepted target evidence.
- [`validation.md`](validation.md): accepted evidence and exact validation boundaries.
- [`competitive.md`](competitive.md): backend and optimization decisions supported by measurements.
