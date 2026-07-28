# MicroColossus Project Specification

> **Trade memory for time.**

This document is the authoritative overview of the purpose, architecture, design decisions, implemented milestones, validation boundary, and development path of MicroColossus.

## 1. Purpose

MicroColossus is an experimental runtime for full-parameter training when the complete managed training state cannot remain safely resident in available memory.

The central engineering question is:

> Can a correct, observable, recoverable, and storage-bounded training trajectory be executed while only a bounded working set is materialized, with inactive parameters, gradients, and optimizer state stored outside unified memory?

The primary target is Apple Silicon, beginning with an 8 GB MacBook Air M2.

The project does not promise unlimited model size or datacenter throughput. Storage capacity, unified-memory pressure, bandwidth, minimum group or tile size, SSD endurance, elapsed time, numerical behavior, and retained-checkpoint policy remain explicit constraints.

## 2. Why the project exists

Training requires more state than model weights alone:

- parameters;
- gradients;
- Adam first moments;
- Adam second moments;
- optimizer step tensors and metadata;
- activations or recomputation state;
- logits, loss intermediates, and workspaces;
- framework, driver, filesystem-cache, and operating-system overhead.

A machine can have enough storage for the complete state while lacking enough safe resident memory. MicroColossus treats that mismatch as a scheduling, storage, transaction, retention, data-provenance, and observability problem.

## 3. Apple Silicon rules

Apple Silicon uses one physical unified-memory pool. CPU execution, Metal execution, framework allocators, the Metal driver, and macOS compete for the same capacity.

### 3.1 Placement is not capacity offload

Moving a tensor between CPU execution and MPS changes placement and execution. It does not create a separate physical memory tier.

### 3.2 Counters are not additive capacities

RSS, PyTorch MPS allocation, Metal driver allocation, MLX allocator counters, compressed memory, and swap describe overlapping or different scopes. They are reported separately and never summed as physical memory.

### 3.3 NVMe is the external capacity tier

```text
MLX or MPS active working set
              |
bounded unified-memory staging and activations
              |
versioned external tensor state
```

The reference implementation currently uses normal filesystem I/O. Direct-I/O or device-specific NVMe behavior is not yet claimed.

### 3.4 Memory and storage pressure are part of the plan

A valid plan must consider logical tensor bytes, process footprint, macOS pressure, compression, swap, filesystem cache, storage traffic, cumulative writes, retained history, recovery cost, and elapsed time.

## 4. Goals

MicroColossus aims to:

1. train models whose complete managed state cannot remain safely resident;
2. keep only the active parameter group or tile available to accelerator execution;
3. keep inactive parameter and optimizer chunks on storage;
4. enforce parameter, gradient, optimizer, activation, storage, write, and time budgets;
5. preserve a full-parameter numerical reference path;
6. track identity, version, location, lineage, integrity, and retention of managed state;
7. publish complete optimizer steps atomically;
8. recover the last committed state after interruption;
9. resume a multi-step trajectory from persistent checkpoint state;
10. bind dataset identity and cursor progression to the same authoritative checkpoint;
11. reclaim historical state without changing `CURRENT` or retained checkpoints;
12. report training, validation, memory, storage, pruning, numerical, and recovery behavior;
13. remain competitive with relevant resident and offload baselines by measurement.

## 5. Non-goals

The project does not claim:

- literally unlimited model size;
- throughput equal to a provisioned cluster;
- support for arbitrary PyTorch or MLX graphs;
- feasibility for every architecture or optimizer;
- that CPU-to-MPS placement increases capacity;
- that offloading, checkpointing, tiling, pruning, or quantization are new concepts;
- success at a scale target before a reproducible demonstration.

Parameter count alone is not a success metric.

## 6. Training modes

- **`reference`**: full-parameter FP32 AdamW with no silent approximation. Implemented.
- **`compact`**: future declared approximations such as quantized optimizer state.
- **`adapter`**: future LoRA or QLoRA paths, reported separately.

## 7. High-level architecture

```text
Dataset or deterministic batch source
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
| - data identity and cursor coordinator               |
| - validation and sample coordinator                  |
| - retention and pruning coordinator                  |
| - transaction and recovery coordinator               |
| - telemetry                                          |
+------------------------------------------------------+
                 |
bounded active working set <-> versioned storage state
```

A controlled decoder-only Transformer keeps operator decomposition, state comparison, failure injection, and recovery tractable.

## 8. Backend strategy

The current decision is **DUAL BACKEND**.

- **MLX** is the preferred optimized Apple Silicon execution candidate.
- **PyTorch MPS** is the numerical oracle and reference for state comparison, debugging, and recovery semantics.
- **Storage, tensor identity, root checkpoints, retention, data provenance, and execution plans** remain backend-neutral.

This decision came from resident measurements. Storage-backed paths must be benchmarked independently.

## 9. Development workloads

| Workload | Parameters | Purpose |
|---|---:|---|
| Synthetic micro model | 11,456 | Fast storage, recovery, numerical, failure, and pruning tests |
| Real-text micro model | 18,624 | Byte tokenizer, data identity, validation, samples, resume, and pruning diagnostics |
| Tiny model | 443,648 | Target-hardware numerical and telemetry gate |
| Real-text small model | 1,846,656 | First meaningful learning and filesystem-reclamation experiment |
| Competitive resident model | 23,213,056 | PyTorch MPS, checkpointed PyTorch, and MLX comparison |
| Future capacity models | 124M, 350M, and later | Demonstrate state larger than safe resident memory |

The two micro counts are intentionally different. The real-text model needs vocabulary size 256 and a 64-position embedding table. The older 11,456 count belongs to the synthetic model with vocabulary size 64 and a shorter positional table.

Synthetic batches isolate runtime correctness. The byte-text frontend adds a real training trajectory without external tokenizer dependencies.

## 10. Implemented architecture by release

| Release | Main capability | Validation status |
|---|---|---|
| 0.2.x | Native Apple MPS resident training and telemetry | Validated on an 8 GB M2 |
| 0.3.x | Portable state and competitive PyTorch MPS or MLX harness | Validated on an 8 GB M2 |
| 0.4.x | Backend-neutral tensor store, transactions, checksums, and recovery | Validated in CI and target tests |
| 0.5.0 | Observable storage-backed optimizer lifecycle | Validated on an 8 GB M2 |
| 0.6.0 | Parameter-group bounded forward | Validated on an 8 GB M2 |
| 0.7.0 | Reverse group-bounded backward and gradient store | Validated on an 8 GB M2 |
| 0.8.0 | Group-bounded AdamW and atomic root publication | Validated on an 8 GB M2 |
| 0.9.0 | Consecutive bounded steps, lineage, checkpoint, and process resume | Validated on an 8 GB M2 |
| 0.10.0 | Local real-text identity, deterministic windows, validation, samples, and resume | Validated on an 8 GB M2 with a documented protocol correction |
| 0.11.0 | Explicit safe retention, pruning journal, compaction, and resume after pruning | CPU CI passed. Target filesystem gate pending |

Accepted runtime commits:

```text
0.9.0:  4b1ffb20857dd948d7737484e62b007f24bf69b9
0.10.0: 8bc277123267c3d3f15bf60cd640819fa823d2e3
```

## 11. Versioned state

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

Tensor bytes are split into content-addressed immutable chunks. New versions use copy-on-write manifests. A store transaction follows:

```text
prepared -> writing -> validated -> committed
                            \
                             -> aborted
```

The previous `CURRENT` remains authoritative until all candidate data verifies and the new pointer is atomically published.

Committed or aborted transactions release references to staged payloads. A live transaction object therefore cannot silently retain a full canonical copy after completion.

## 12. Root step bundles

A root bundle references:

- parameter-store manifest;
- optimizer-store manifest;
- optional gradient-store manifest;
- consumed batch checksum;
- committed training step;
- parent bundle;
- root checksum.

Candidate child stores are not authoritative training state until the root bundle is published.

The publication sequence is:

```text
build candidate child stores
        ↓
verify child stores
        ↓
write and fsync root manifest
        ↓
write and fsync candidate CURRENT
        ↓
atomic CURRENT rename
```

Failure before the final pointer replacement leaves the prior root authoritative.

## 13. Bounded execution

Execution groups are:

```text
embedding
block-0
block-1
...
final-head
```

### 13.1 Forward

One parameter group is read, materialized, executed, and released at a time. The tied token embedding is read again for the output projection.

### 13.2 Backward

Boundary activations are currently retained on CPU. Groups execute in reverse order, locally recompute forward, propagate one incoming activation gradient, and publish final parameter gradients.

The tied token embedding receives two gradient contributions. The output-head contribution is written first, then the embedding contribution is accumulated into a newer gradient version.

### 13.3 Global clipping

Final gradients are streamed one tensor at a time. The clipping coefficient is:

```text
min(1, max_norm / (global_norm + 1e-6))
```

The bounded path and resident oracle use the same coefficient.

### 13.4 AdamW

For one unique group at a time, the runtime reads parameters, gradients, first moments, second moments, and step tensors. The shared embedding is updated exactly once with its final accumulated gradient.

New parameter and optimizer versions are written to candidate stores, restored, compared with the oracle, and then referenced by a new root bundle.

## 14. Persistent training and resume

Version 0.9 advances:

```text
step N -> step N+1
```

The new root points to the prior root as parent. A later process can reopen `CURRENT` and continue without recreating parameters or Adam state.

The root committed step is the authoritative next-batch cursor. A separate mutable cursor file is avoided because it could diverge from model and optimizer state after interruption.

The accepted M2 gate demonstrated:

- uninterrupted micro step 0 to 5;
- process exit at step 2 and resume to step 5;
- bitwise-exact resumed versus uninterrupted final state;
- tiny step 0 to 3;
- correct optimizer step tensors and contiguous parent lineage;
- preservation of step 2 after an interrupted attempt to publish step 3;
- rejection of incompatible configuration;
- detection of a corrupted authoritative child store.

## 15. Real-text data contract

Version 0.10 adds a backend-neutral `PreparedDataSource` interface and the first local text implementation.

### 15.1 Tokenizer

```text
version: utf8-bytes-v1
vocabulary: 256 bytes
```

This is an engineering tokenizer, not a model-quality claim.

### 15.2 Data identity

Persistent metadata includes:

- source kind;
- tokenizer version;
- sampler version;
- split policy;
- train and validation SHA-256;
- train and validation byte counts;
- identity checksum.

Changed corpus bytes or policies reject resume before another step is published.

### 15.3 Cursor and windows

For text training, committed cursor `N` deterministically selects byte windows using:

```text
seed = training seed + 1 + N
```

Each microbatch row records its selected byte offset. The root committed step remains the authoritative next-batch cursor.

### 15.4 Evaluation and samples

Validation loss and greedy samples are produced from the parameter store referenced by a committed bundle. Per-step progress files include bundle ID, offsets, seed, batch checksum, training loss, gradient norm, clipping coefficient, validation evidence, and sample tokens.

Evaluation materializes a resident model and is excluded from bounded execution claims.

Progress records are atomically written derived evidence. The root bundle remains authoritative. A missing progress record for an already committed current root can be reconstructed on resume.

## 16. Retention and pruning contract

Version 0.11 introduces explicit safe compaction of persistent training roots.

### 16.1 Retention policy

```yaml
retention:
  keep_previous: 2
  milestone_interval: 5
```

The materialized retained set includes:

- the current checkpoint;
- the configured number of immediately previous checkpoints;
- optional milestone checkpoints divisible by the interval.

Retention is operational policy. It is intentionally excluded from the semantic training digest so future pruning choices do not alter model, optimizer, data, or schedule identity.

### 16.2 Complete lineage, selected restorable history

Every root bundle manifest in the committed parent chain remains available as lightweight metadata. This preserves:

- complete committed step lineage;
- parent bundle IDs;
- batch checksums;
- progress-record alignment;
- root checksums.

Only retained checkpoints keep their referenced parameter, optimizer, and gradient child stores. A historical root whose child stores were reclaimed remains visible in lineage but is no longer restorable.

### 16.3 Dry-run plan

Planning is non-mutating and records:

- exact `CURRENT` bundle, checksum, step, and pointer-file digest;
- complete lineage;
- retained checkpoints and child-store paths;
- checkpoints losing materialized state;
- exact deletion paths;
- recursive file counts, byte counts, and content digests;
- managed bytes before pruning;
- selected bytes;
- plan checksum.

Apply refuses stale or modified plans.

### 16.4 Apply and journal

Apply is explicit. It:

1. verifies training metadata and the plan checksum;
2. confirms `CURRENT` is byte-identical to the planned pointer;
3. verifies every retained root and child store;
4. publishes an atomic pruning operation journal;
5. deletes only paths proven unreachable from retained checkpoints;
6. records progress after each path deletion;
7. verifies retained state again;
8. runs root recovery;
9. confirms `CURRENT` never changed;
10. publishes a final report.

The journal states are:

```text
prepared -> deleting -> completed
```

An interrupted operation can resume. Reapplying a completed plan is idempotent.

### 16.5 Allowed reclamation

The initial implementation may remove:

- unretained candidate parameter and optimizer stores;
- unretained gradient stores;
- oracle and validation-only work state;
- interrupted candidate state;
- unreferenced corrupt orphan state;
- unpublished root manifests;
- safe temporary files.

Protected root metadata, retained child stores, `CURRENT`, `TRAINING.json`, progress records, and committed lineage manifests are never selected.

### 16.6 Concurrency

Pruning owns an exclusive root lock. Persistent bounded training refuses to resume while the pruning lock is active.

The implementation assumes one training or pruning coordinator per root. Distributed writers are out of scope.

## 17. Working-set and storage budgets

The runtime enforces separate logical budgets for:

- parameter bytes in one group;
- gradient bytes in one backward group;
- parameter, gradient, first-moment, second-moment, and step bytes in one optimizer group;
- tensor-store staging and total managed storage.

Pruning adds explicit retained-history policy and application-level reclamation reporting.

These are not total physical-memory or physical SSD-block measurements. Activations, workspaces, caches, RSS, page cache, filesystem metadata, snapshots, and operating-system pressure are reported separately.

## 18. Correctness policy

### CPU

Exact canonical state is required when operation order and semantics are equivalent.

### MPS

Bitwise equality and numerical distance are reported separately. A checksum difference alone is not numerical failure.

### Validation-only materialization

Complete oracle or candidate states, resident replay, validation, and generation may materialize full state only for declared verification or evaluation. They are excluded from bounded claims.

### Pruning

`CURRENT` must remain byte-identical. Every retained checkpoint must verify before and after apply. Resume after pruning must match an equivalent unpruned trajectory.

## 19. Telemetry

The runtime and target harnesses report:

- per-group parameter, gradient, moment, and step bytes;
- tensor and chunk reads;
- logical and physical application writes;
- chunk creation and reuse;
- read, materialization, compute, export, commit, release, checksum, `fsync`, and publication times;
- process RSS;
- accelerator and driver allocation;
- system memory pressure and swap;
- manifest IDs, tensor versions, lineage, cursor, offsets, seeds, and checksums;
- validation loss and sample provenance;
- recovery actions and unpublished state;
- storage growth;
- pruning paths, selected bytes, cumulative reclaimed bytes, newly reclaimed bytes, and remaining managed bytes.

Application byte counters are not presented as NAND-level SSD writes or guaranteed filesystem block reclamation.

## 20. Accepted evidence through 0.10

### 20.1 Resident and competitive foundation

- fixed-batch MPS loss decreased from `5.566842079162598` to `0.4145740866661072`;
- MLX reached `1.592x` PyTorch MPS throughput in the tested resident competitive workload;
- the dual-backend decision was retained.

### 20.2 Storage and bounded execution

- exact storage lifecycle and failure recovery;
- zero tested bounded-forward boundary, logits, and loss differences;
- exact tested bounded-backward gradients;
- streamed global gradient norm;
- complete bounded AdamW and atomic publication;
- parameter, gradient, and optimizer budget rejection;
- exact candidate restore.

### 20.3 Persistent multi-step 0.9

- uninterrupted step 0 to 5 and resumed step 2 to 5 were bitwise exact;
- maximum bounded-versus-resident absolute state difference was `7.450580596923828e-09`;
- later-step atomicity, configuration rejection, and corruption detection passed;
- swap growth was zero in the accepted scenarios.

### 20.4 Real-text 0.10

The target report was formally labeled `FAIL` only because the external prompt expected the obsolete synthetic count of 11,456 parameters for `real-text-micro.yaml`. The checked configuration correctly plans 18,624 parameters. That protocol mismatch did not indicate a runtime failure.

Accepted target evidence after correcting the expectation:

- native MPS execution on an 8 GB M2 with fallback disabled;
- 88 tests passed with one skip, and Ruff, mypy, compileall, and doctor passed;
- independent-process data identity matched;
- byte-tokenizer round trip passed;
- micro training step 0 to 20 was GREEN;
- micro validation loss decreased from `5.548418998718262` to `3.302267074584961`;
- process restart from step 5 to step 20 was GREEN;
- uninterrupted versus resumed state was numerically stable with maximum absolute difference `1.1920928955078125e-07` and mean absolute difference `8.844825718731097e-10`;
- data provenance and generated sample tokens matched;
- corpus mutation was rejected before another step became authoritative;
- the 1,846,656-parameter small run reached step 10 and was GREEN;
- small validation loss decreased from `5.687370777130127` to `4.083975553512573`;
- candidate restore was exact;
- root and child stores verified and recovered;
- no runtime fallback, unsupported operation, non-finite value, or unexpected command failure remained after audit;
- final source state was clean.

The ten-step small root occupied about 632 MB while retaining all historical candidates and work stores. This established the need for M6A.

## 21. Version 0.11 CPU evidence

The safe pruning implementation is required to pass on Python 3.11 and 3.13:

- explicit Ruff rule selection and clean lint;
- mypy over the complete source tree;
- 100 tests with one Apple-only skip;
- compileall;
- deterministic non-mutating planning;
- exact deletion-path inventories;
- explicit apply;
- byte-identical `CURRENT`;
- retained-checkpoint verification;
- interrupted apply and resume;
- idempotent repeated apply;
- corruption rejection before deletion;
- safe orphan removal;
- training exclusion while pruning is active;
- exact post-pruning training resume against an unpruned reference;
- end-to-end CLI smoke through training, plan, apply, and another optimizer step.

Target Apple M2 filesystem validation remains required before M6A is accepted as target evidence.

## 22. Important design decisions

### 22.1 Correct synchronous path before overlap

The project implements synchronous read, compute, write, verify, publish, and prune before prefetch or asynchronous writeback.

### 22.2 One source of complexity per milestone

Forward, backward, gradient storage, clipping, optimizer execution, atomic publication, multi-step resume, real-data provenance, and retention were isolated so failures remain localizable.

### 22.3 Explicit shared-weight semantics

Shared parameters have explicit forward reads, gradient accumulation, and unique optimizer update rules.

### 22.4 Small models for fast iteration

Synthetic micro, real-text micro, tiny, and small models are routine gates. Larger workloads are reserved for performance and capacity milestones.

### 22.5 Checkpoint and data authority are unified

Model state, optimizer state, parent lineage, consumed batch checksum, and next data cursor advance through one authoritative root checkpoint.

### 22.6 Derived evidence is not authority

Metrics and samples may be recreated from a committed root. They do not replace the root bundle as training state.

### 22.7 Lineage metadata and restorable history are separate

Root manifests remain to preserve the committed trajectory. Retention policy decides which historical checkpoints still own restorable child state.

### 22.8 Destructive maintenance is explicit

Training never silently deletes history in version 0.11. Planning is dry-run. Apply requires a checksummed plan and a separate explicit command.

## 23. Current boundary

MicroColossus does not yet establish:

- accepted Apple M2 validation of pruning and compaction;
- live chunk repacking across retained stores;
- content deduplication across independent store directories;
- automatic periodic or storage-pressure pruning;
- a representative tokenizer or production corpus;
- production model quality;
- large sharded dataset state, epochs, and shuffle semantics;
- activation recomputation from storage or activation offload;
- bounded activation and workspace residency;
- strict total-memory-pressure enforcement;
- asynchronous prefetch or writeback;
- intra-layer tiling;
- bounded MLX backward or optimizer execution;
- training state larger than safe resident unified memory;
- 124M or 350M full-parameter training on the target machine.

No full out-of-core, throughput-at-scale, production model-quality, or model-capacity claim is made yet.

## 24. Roadmap

| Milestone | Status |
|---|---|
| M0 through M4C. Resident, storage, bounded step, and persistent resume | Completed and validated on M2 |
| M5. Deterministic small real-corpus frontend | Completed and validated on M2 |
| M6A. Historical-state pruning and compaction | Implemented. CPU CI passed. M2 filesystem gate pending |
| M6B. Activation recomputation, offload, and strict budgets | Planned |
| M7. Asynchronous prefetch and writeback | Planned |
| M8. Intra-layer tiling | Planned |
| M9. Bounded MLX optimizer execution | Planned |
| M10. 124M, 350M, and larger capacity demonstrations | Planned |

Pruning was promoted before longer real-data trajectories because the accepted 1.85M-parameter ten-step root occupied about 632 MB while retaining all historical candidates and work stores.

## 25. Criteria for the first meaningful out-of-core result

A meaningful demonstration requires:

1. complete managed state larger than safe resident capacity;
2. every active parameter, activation, gradient, optimizer, and workspace set within limits;
3. multiple optimizer steps from prior committed state;
4. interruption recovery to the last committed step;
5. resumed execution matching uninterrupted execution;
6. traceable tensor versions, data provenance, root lineage, and retention;
7. reported storage traffic, cumulative writes, and historical-state reclamation;
8. numerical comparison with a reference where feasible;
9. explicit throughput and endurance costs;
10. a useful training trajectory on real data.

## 26. Documentation map

- [`project.md`](project.md): purpose, architecture, decisions, and roadmap.
- [`storage.md`](storage.md): tensor store, transaction, bundle, and bounded execution design.
- [`multistep.md`](multistep.md): persistent checkpoint, lineage, cursor, and resume design.
- [`real-text.md`](real-text.md): corpus identity, tokenizer, validation, samples, and accepted real-text gate.
- [`pruning.md`](pruning.md): retention policy, dry-run planning, apply, journal, recovery, and limitations.
- [`validation.md`](validation.md): accepted evidence and exact validation boundaries.
- [`competitive.md`](competitive.md): backend and optimization decisions supported by measurements.
