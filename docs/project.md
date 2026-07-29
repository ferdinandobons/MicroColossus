# MicroColossus Project Specification

> **Trade memory for time.**

This document is the authoritative overview of why MicroColossus exists, how the runtime is designed, what has been implemented, what has been validated, and what remains outside the current claim boundary.

## 1. Purpose

MicroColossus is an experimental open-source runtime for full-parameter generative-model training when the complete managed training state cannot remain safely resident in available memory.

The primary engineering question is:

> Can a correct, observable, recoverable, and storage-bounded training trajectory be executed while only a declared working set is materialized, with inactive parameters, gradients, and optimizer state stored outside the active compute set?

The primary target is Apple Silicon, beginning with an 8 GB MacBook Air M2.

The project does not promise unlimited model size. Capacity is constrained by storage, memory pressure, minimum execution-group or tile size, bandwidth, SSD endurance, elapsed time, numerical behavior, checkpoint retention, and operating-system overhead.

## 2. Why the project exists

Training requires substantially more state than model parameters alone:

- model parameters;
- gradients;
- Adam first moments;
- Adam second moments;
- optimizer step tensors and metadata;
- activations or recomputation state;
- logits, loss intermediates, and workspaces;
- framework, driver, filesystem-cache, and operating-system overhead.

A machine can have enough storage for the complete managed state while lacking enough safe resident memory. MicroColossus treats this mismatch as a scheduling, storage, transaction, retention, data-provenance, activation-management, and observability problem.

## 3. Apple Silicon rules

Apple Silicon uses one physical unified-memory pool. CPU execution, Metal execution, framework allocators, the Metal driver, filesystem cache, and macOS compete for the same capacity.

### 3.1 CPU placement is not capacity offload

Moving a tensor between CPU execution and MPS changes placement and execution. It does not create a separate physical memory tier.

### 3.2 Memory counters are not additive

Process RSS, PyTorch MPS allocation, Metal-driver allocation, MLX allocator counters, compressed memory, and swap describe overlapping or different scopes. They are recorded separately and are never added as independent physical capacities.

### 3.3 Storage is the external capacity tier

```text
MLX or MPS active working set
              |
bounded unified-memory staging and activations
              |
versioned storage-backed tensor state
```

The current implementation uses normal filesystem I/O. Direct I/O, device-specific NVMe behavior, and physical NAND-write guarantees are not claimed.

### 3.4 Logical budgets are not physical-memory proof

Parameter, gradient, optimizer, activation, and workspace budgets are logical guardrails. A policy can reduce logical retained tensors while still increasing sampled RSS because allocator behavior, replay workspaces, caches, compressed memory, and validation-only materialization affect the physical process footprint.

## 4. Goals

MicroColossus aims to:

1. execute full-parameter training when complete managed state cannot remain safely resident;
2. keep only the active parameter group or tile available to accelerator execution;
3. keep inactive parameter and optimizer chunks in versioned storage;
4. enforce parameter, gradient, optimizer, activation, workspace, storage, and time budgets;
5. preserve a full-parameter numerical reference path;
6. track identity, version, location, lineage, integrity, and retention of managed state;
7. publish complete optimizer steps atomically;
8. recover the last committed state after interruption;
9. resume multi-step trajectories from persistent checkpoints;
10. bind dataset identity and cursor progression to the authoritative checkpoint;
11. reclaim historical state without changing `CURRENT` or retained checkpoints;
12. trade activation memory for deterministic recomputation when requested;
13. report numerical, memory, storage, pruning, replay, and recovery behavior;
14. remain competitive with relevant resident and offload baselines by measurement.

## 5. Non-goals

The project does not claim:

- literally unlimited model size;
- datacenter-cluster throughput on a laptop;
- support for arbitrary PyTorch or MLX graphs;
- feasibility for every architecture or optimizer;
- that CPU-to-MPS movement increases capacity;
- that offloading, checkpointing, recomputation, tiling, pruning, or quantization are new concepts;
- success at a scale target before a reproducible target-hardware demonstration.

Parameter count alone is not a success metric.

## 6. Training modes and backends

### 6.1 Training modes

- **`reference`**. Full-parameter FP32 AdamW without a silent approximation. Implemented.
- **`compact`**. Future declared approximations such as quantized optimizer state.
- **`adapter`**. Future LoRA or QLoRA paths, reported separately from full-parameter training.

### 6.2 Backend strategy

The current decision is **DUAL BACKEND**.

- **MLX** is the preferred optimized Apple Silicon execution candidate.
- **PyTorch MPS** is the numerical oracle and the reference for state comparison, debugging, and recovery semantics.
- Tensor storage, checkpoint lineage, data identity, retention, and execution policy remain backend-neutral where practical.

The decision is based on resident measurements. Storage-backed paths require independent measurements.

## 7. High-level architecture

```text
Dataset or deterministic batch source
                 |
Controlled decoder-only Transformer
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
| - activation policy and recomputation coordinator    |
| - root checkpoint and resume coordinator             |
| - data identity and cursor coordinator               |
| - retention and pruning coordinator                  |
| - transaction and recovery coordinator               |
| - telemetry                                          |
+------------------------------------------------------+
                 |
bounded active working set <-> versioned storage state
```

A controlled decoder-only Transformer keeps operator decomposition, canonical-state comparison, failure injection, and recovery tractable.

## 8. Versioned tensor state

Every managed tensor records:

```text
tensor ID
logical name
kind
shape
dtype
byte order
version
ordered chunk IDs
byte length
checksum
committed step
adapter metadata
```

Tensor bytes are split into content-addressed immutable chunks. New versions use copy-on-write manifests. Transactions move through prepared, writing, validated, and committed states. Failed transactions are aborted.

The previous `CURRENT` remains authoritative until candidate data verifies and a new pointer is atomically published.

## 9. Root step bundles

A root bundle references:

- parameter-store state;
- optimizer-store state;
- optional gradient-store state;
- consumed batch checksum;
- committed training step;
- parent bundle;
- root checksum.

Candidate child stores are not authoritative until the root bundle is published.

```text
build candidate stores
        |
verify candidate stores
        |
write and fsync root manifest
        |
write and fsync candidate CURRENT
        |
atomic CURRENT rename
```

Failure before the final pointer replacement leaves the previous root authoritative.

## 10. Bounded execution

Execution groups are:

```text
embedding
block-0
block-1
...
final-head
```

### 10.1 Forward

One parameter group is read, materialized, executed, and released at a time. The tied token embedding is read again for the output projection.

### 10.2 Backward

Backward executes groups in reverse order and publishes final gradients into a versioned gradient store. The tied token embedding receives both output-head and embedding contributions.

### 10.3 Global clipping

Final gradients are streamed one tensor at a time. The canonical clipping coefficient is:

```text
min(1, max_norm / (global_norm + 1e-6))
```

### 10.4 AdamW

For one unique group at a time, the runtime reads parameters, gradients, first moments, second moments, and step tensors. The shared embedding is updated exactly once with the accumulated gradient.

New parameter and optimizer versions are written to candidate stores, restored, compared with an oracle, and then referenced by the next root bundle.

## 11. Persistent training and data authority

Every step consumes the parameter and Adam state referenced by the current root bundle. A later process can reopen the same directory and continue from `CURRENT`.

The committed root step is the authoritative next-batch cursor. Model state, optimizer state, consumed batch checksum, parent lineage, and data progress advance together.

The current real-text frontend provides:

- local UTF-8 corpus files;
- tokenizer `utf8-bytes-v1` with vocabulary size 256;
- checksummed train and validation identity;
- deterministic train and validation split;
- deterministic random-access text windows;
- byte offsets, seeds, and batch checksums;
- validation loss at committed checkpoints;
- deterministic greedy sample generation;
- resume rejection when corpus bytes or declared data semantics change.

The byte tokenizer is an engineering reference. It is not a production tokenizer claim.

## 12. Retention and pruning

Version 0.11 introduced an explicit two-stage pruning workflow:

```text
dry-run plan
    -> verify CURRENT and retained checkpoints
    -> inventory exact deletion targets
    -> checksum the plan

explicit apply
    -> publish operation journal
    -> delete only proven-unreachable state
    -> verify retained state
    -> preserve CURRENT byte-for-byte
```

Retention can preserve `CURRENT`, a declared number of previous checkpoints, and optional milestone checkpoints.

Root manifests remain as lightweight lineage metadata. Only retained checkpoints keep restorable parameter, optimizer, and gradient child stores.

Pruning is manual and conservative. It is not a background operation.

The accepted M2/APFS gate reclaimed `10,739,392` bytes in the micro scenario and `289,540,389` selected bytes in the 1.85M-parameter small scenario while preserving `CURRENT`, retained state, recovery, and resume.

## 13. Activation policies

Persistent training supports:

```yaml
training:
  activation_policy: retain_all
```

or:

```yaml
training:
  activation_policy: recompute
```

or:

```yaml
training:
  activation_policy: hybrid
```

### 13.1 `retain_all`

Every non-final forward boundary is retained on CPU until the matching reverse group executes. This minimizes replay but grows activation residency with depth, sequence length, microbatch size, and hidden width.

### 13.2 `recompute`

No forward boundary is retained for later backward use. Reverse groups reconstruct their local input from token IDs and the authoritative parameter store.

```text
final-head backward
    replay embedding and every Transformer block

block K backward
    replay embedding through block K-1

embedding backward
    use token IDs directly
```

The first schedule is synchronous and intentionally favors correctness and observability over throughput. Prefix replay can be quadratic in execution-group count.

The recomputed gradient store uses the same clipping, group-bounded AdamW, candidate verification, atomic publication, progress, resume, and pruning contracts as `retain_all`.

The activation policy is part of checkpoint identity. A root cannot silently change policy at resume.

### 13.3 `hybrid`

The M6C profile and planning layer accepts:

```yaml
training:
  activation_policy: hybrid
  activation_anchor_policy:
    kind: measured_budget_v1
```

The implemented M6C increment builds deterministic, checksummed activation
profiles and plans. It compares `retain_all`, full-prefix `recompute`,
fixed-interval anchors, and measured-budget anchor schedules, then records
selected anchors, replay segments, retained bytes, replayed groups, parameter
rereads, workspace status, and feasibility.

Persistent nearest-anchor hybrid backward execution is integrated into
`microcolossus-bounded-train`. The forward path retains only plan-selected
anchors; backward reconstructs each target input from the nearest retained
preceding anchor and replays only intervening groups. Hybrid roots bind profile
checksum, plan checksum, planner version, selected anchors, activation budget,
workspace budget, and replay-depth constraint.

## 14. Budget model

The runtime enforces separate logical budgets for:

- parameter bytes in one execution group;
- gradient bytes in one backward group;
- parameter, gradient, first-moment, second-moment, and step bytes in one optimizer group;
- retained activation bytes;
- local forward or backward workspace bytes;
- tensor-store staging and total managed storage.

These budgets are logical guardrails. They are not a direct measurement of total physical unified-memory use.

## 15. Development workloads

| Workload | Parameters | Purpose |
|---|---:|---|
| Synthetic micro | 11,456 | Fast numerical, storage, failure, and pruning tests |
| Real-text micro | 18,624 | Data identity, validation, samples, resume, pruning, and activation-policy diagnostics |
| Tiny | 443,648 | Target-hardware numerical and telemetry gate |
| Real-text small | 1,846,656 | Meaningful learning, reclamation, and activation-memory comparison |
| Competitive resident | 23,213,056 | PyTorch MPS, checkpointed PyTorch, and MLX comparison |
| Future capacity targets | 124M, 350M, and later | Demonstrate state larger than safe resident capacity |

The two micro counts are intentionally different. The real-text model uses vocabulary size 256 and a larger positional table.

## 16. Implemented releases

| Release | Main capability | Accepted status |
|---|---|---|
| 0.2.x | Native Apple MPS resident training and telemetry | Validated on an 8 GB M2 |
| 0.3.x | Portable state and PyTorch MPS or MLX benchmark harness | Validated on an 8 GB M2 |
| 0.4.x | Backend-neutral tensor store, transactions, checksums, and recovery | Validated in CI and target tests |
| 0.5.0 | Observable storage-backed optimizer lifecycle | Validated on an 8 GB M2 |
| 0.6.0 | Parameter-group bounded forward | Validated on an 8 GB M2 |
| 0.7.0 | Reverse group-bounded backward and gradient store | Validated on an 8 GB M2 |
| 0.8.0 | Group-bounded AdamW and atomic root publication | Validated on an 8 GB M2 |
| 0.9.0 | Consecutive bounded steps, lineage, checkpoint, and process resume | Validated on an 8 GB M2 |
| 0.10.0 | Deterministic local real-text trajectory, validation, samples, and resume | Validated on an 8 GB M2 after a protocol correction |
| 0.11.0 | Safe retention, pruning journal, reclamation, and resume after pruning | Validated on an 8 GB M2 and APFS |
| 0.12.0 | Persistent activation policies, recomputation, activation and workspace budgets | Validated on an 8 GB M2 |
| 0.13.0 | Measured hybrid activation-anchor planner and nearest-anchor execution | Validated on an 8 GB M2 |

Important accepted commits:

```text
0.9.0 persistent resume: 4b1ffb20857dd948d7737484e62b007f24bf69b9
0.10.0 real text:       8bc277123267c3d3f15bf60cd640819fa823d2e3
0.11.0 pruning fix:     1fedf611e7a090dad218be64811e0a4e007fbd77
0.12.0 target gate:     4742f8a7f57a46edb075159275fb66c83c78ced7
0.13.0 M6C target gate: 8e9b0f8e58fdaa288ba551d994d9b8b81adbea12
```

## 17. Accepted target evidence

### 17.1 Competitive resident baseline

A controlled 23,213,056-parameter resident workload produced:

| Variant | Median tokens/s | Relative to PyTorch |
|---|---:|---:|
| PyTorch MPS | 1,379.23 | 1.000x |
| PyTorch MPS checkpointed | 1,337.37 | 0.970x |
| MLX | 2,195.69 | 1.592x |

### 17.2 Persistent and real-text training

Accepted target runs demonstrated uninterrupted and resumed multi-step training, deterministic provenance, real-text learning signals, exact candidate restore, and corpus-mutation rejection.

### 17.3 Pruning

The corrected 0.11 M2/APFS run demonstrated deterministic planning, material reclamation, interruption continuation, idempotence, and process resume after pruning.

### 17.4 Activation recomputation

The 0.12 M2 gate demonstrated:

- numerically stable `retain_all` and `recompute` parameter plus optimizer state;
- zero retained forward-boundary bytes under `recompute`;
- expected replay totals of `15` for micro, `18` for tiny, and `15` for small;
- process restart and resume under `recompute`;
- activation-policy mismatch rejection;
- activation and workspace budget rejection;
- pruning followed by recompute resume;
- preservation of the previous root after injected publication failures;
- tied-gradient accumulation and unique AdamW update semantics;
- clean source state and no detected hidden fallback or non-finite value.

Small logical activation evidence:

```text
retain_all forward-boundary bytes: 491,520
recompute forward-boundary bytes:        0
retain_all maximum retained activation:  491,520
recompute maximum retained activation:   196,608
maximum local workspace:                 393,216
```

Small replay evidence:

```text
logical parameter bytes reread: 19,200,000
recorded prefix recomputation:  about 0.102379 seconds
```

Small physical-memory observation:

```text
retain_all sampled peak RSS: 444,071,936 bytes
recompute sampled peak RSS:  533,528,576 bytes
ratio:                        1.2014462809917354x
```

The logical memory reduction is accepted. The physical result shows that full-prefix recomputation is not yet the optimal policy for the measured workload.

## 18. Important design decisions

### 18.1 Correct synchronous path before overlap

The project implements synchronous read, compute, write, verify, publish, prune, and replay before prefetch or asynchronous writeback.

### 18.2 One source of complexity per milestone

Forward, backward, gradient storage, clipping, optimizer execution, atomic publication, multi-step resume, real-data provenance, retention, and activation scheduling were isolated so failures remain localizable.

### 18.3 Explicit shared-weight semantics

Shared parameters have explicit forward reads, gradient accumulation, and unique optimizer update rules.

### 18.4 Small models for fast iteration

Synthetic micro, real-text micro, tiny, and small models are routine gates. Larger workloads are reserved for performance and capacity milestones.

### 18.5 Checkpoint and data authority are unified

Model state, optimizer state, parent lineage, consumed batch checksum, and next data cursor advance through one authoritative root checkpoint.

### 18.6 Derived evidence is not authority

Metrics and samples may be recreated from a committed root. They do not replace the root bundle as training state.

### 18.7 Lineage metadata and restorable history are separate

Root manifests preserve the committed trajectory. Retention policy decides which historical checkpoints still own restorable child state.

### 18.8 Destructive maintenance is explicit

Training never silently deletes history. Planning is dry-run. Apply requires a checksummed plan and a separate explicit command.

### 18.9 Logical and physical activation optimization are separate

Zero forward-boundary retention is a logical property. A useful policy must also be evaluated against physical RSS, allocator counters, replay cost, and memory pressure on the target machine.

## 19. Current boundary

MicroColossus does not yet establish:

- activation tensors stored on disk;
- asynchronous activation prefetch or writeback;
- strict enforcement of total physical memory pressure;
- live chunk repacking across retained stores;
- deduplication across independent store directories;
- automatic storage-pressure pruning;
- a representative tokenizer or production corpus;
- production model quality;
- large sharded dataset state, epochs, and shuffle semantics;
- direct-I/O or NVMe-specific performance behavior;
- intra-layer tiling;
- bounded MLX backward or optimizer execution;
- training state larger than safe resident unified memory;
- 124M or 350M full-parameter training on the target machine.

No complete out-of-core, production-quality, throughput-at-scale, or model-capacity claim is made yet.

## 20. Roadmap

| Milestone | Status |
|---|---|
| M0 through M5. Resident, storage, bounded execution, resume, and real text | Completed and validated on M2 |
| M6A. Historical-state pruning and compaction | Completed and validated on M2/APFS |
| M6B. Persistent activation recomputation and strict activation/workspace budgets | Completed and validated on M2 |
| M6C. Measured hybrid activation-anchor planner | Completed and validated on M2 |
| M7. Asynchronous prefetch and writeback | Planned |
| M8. Intra-layer tiling | Planned |
| M9. Bounded MLX backward and optimizer execution | Planned |
| M10. 124M, 350M, and larger capacity demonstrations | Planned |

## 21. Criteria for a meaningful out-of-core result

A meaningful capacity demonstration requires:

1. complete managed state larger than safe resident capacity;
2. every active parameter, activation, gradient, optimizer, and workspace set within declared limits;
3. multiple optimizer steps from prior committed state;
4. interruption recovery to the last committed step;
5. resumed execution matching uninterrupted execution within a declared numerical band;
6. traceable tensor versions, data provenance, root lineage, and retention;
7. reported storage traffic, cumulative writes, and historical-state reclamation;
8. numerical comparison with a reference where feasible;
9. explicit throughput, replay, and endurance costs;
10. a useful training trajectory on real data.

## 22. Documentation map

- [`project.md`](project.md). Purpose, architecture, decisions, releases, and roadmap.
- [`storage.md`](storage.md). Tensor store, transactions, bundles, and bounded execution.
- [`multistep.md`](multistep.md). Persistent checkpoints, lineage, cursor, and resume.
- [`real-text.md`](real-text.md). Data identity, tokenizer, validation, samples, and real-text evidence.
- [`pruning.md`](pruning.md). Retention, planning, apply, journal, recovery, and M2 pruning evidence.
- [`activations.md`](activations.md). Activation policies, recomputation, budgets, telemetry, and accepted M2 evidence.
- [`validation.md`](validation.md). Accepted executable evidence and exact claim boundaries.
- [`competitive.md`](competitive.md). Backend and optimization decisions supported by measurements.
