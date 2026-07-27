# Versioned Tensor Store and Bounded Execution

This document is the technical design record for the storage, transaction, bounded-execution, gradient, optimizer, and atomic step-publication systems implemented through MicroColossus 0.8.

## 1. Scope

The storage subsystem keeps canonical training state outside framework-owned memory. Its internal representation does not depend on `torch.Tensor` or `mlx.core.array`.

The current runtime exposes four related execution paths:

- `storage-step`: validates a complete storage-backed optimizer lifecycle while materializing the full micro model and optimizer during compute;
- `bounded-forward`: materializes one parameter execution group at a time and retains hidden boundary activations;
- `bounded-backward`: processes groups in reverse, persists final gradients, and streams the global gradient norm;
- `bounded-step`: consumes the final gradient store, executes AdamW group by group, and atomically publishes a complete root step bundle.

Version 0.8 executes one isolated optimizer step. Consecutive-step scheduling, checkpoint and resume, activation offload, asynchronous I/O, and intra-layer tiling remain later milestones.

## 2. Design invariants

The implementation is built around these invariants:

1. published manifests are immutable;
2. chunks are content-addressed and immutable;
3. a candidate tensor-store manifest is not authoritative until `CURRENT` is replaced atomically;
4. a candidate training step is not authoritative until the root step-bundle `CURRENT` is replaced atomically;
5. interrupted work never silently replaces the last committed state;
6. every managed tensor has explicit identity, version, shape, dtype, byte order, kind, checksum, and lineage;
7. tied parameters have explicit read, gradient-accumulation, and optimizer-update semantics;
8. every bounded phase rejects an oversized group before compute;
9. storage and framework memory counters are reported separately;
10. full-state materialization used only for validation is declared explicitly and excluded from bounded-execution claims.

## 3. Canonical tensor representation

A `TensorPayload` is the framework-neutral in-memory form used to stage canonical tensor data. A published tensor record contains:

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

Supported initial tensor kinds are:

- `parameter`;
- `gradient`;
- `adam_first_moment`;
- `adam_second_moment`;
- `master_weight`;
- `metadata`.

Multi-byte numeric values use canonical little-endian storage. Non-contiguous framework tensors are normalized to contiguous logical bytes before checksumming and chunking.

### 3.1 Logical names

Logical names remain stable across storage versions and are independent of content hashes. Examples include:

```text
model.token_embedding.weight
model.blocks.0.attention.qkv.weight
gradient.model.blocks.0.attention.qkv.weight
optimizer.blocks.0.attention.qkv.weight.exp_avg
optimizer.blocks.0.attention.qkv.weight.exp_avg_sq
optimizer.blocks.0.attention.qkv.weight.step
optimizer.param_groups
```

The exact adapter metadata records backend-specific model or optimizer paths without making them the primary tensor identity.

## 4. Tensor-store layout

```text
store/
  store.json
  CURRENT
  CUMULATIVE_WRITES
  chunks/
    <prefix>/<sha256>.chunk
  manifests/
    <manifest-id>.json
  transactions/
    <transaction-id>/journal.jsonl
  telemetry/
    events.jsonl
```

### 4.1 Store metadata

`store.json` defines schema version and limits, including:

- chunk size;
- maximum managed storage bytes;
- maximum transaction staging bytes.

### 4.2 Content-addressed chunks

Chunk IDs are SHA-256 digests of immutable chunk contents. Identical content can be reused across tensor versions and manifests.

### 4.3 Tensor manifests

A manifest defines one complete logical view of the tensor store. It contains ordered tensor records and the committed step. Published manifests are immutable.

### 4.4 `CURRENT`

`CURRENT` identifies the authoritative manifest and its checksum. It is written to a temporary file, flushed, synchronized, and atomically renamed only after the candidate manifest and all referenced chunks validate.

### 4.5 Cumulative writes

`CUMULATIVE_WRITES` tracks application-managed state bytes written through the store. It is not an estimate of physical NAND writes.

## 5. Tensor-store transaction protocol

A transaction follows:

```text
prepared -> writing -> validated -> committed
                            \
                             -> aborted
```

The synchronous commit sequence is:

1. read the current parent manifest;
2. build a copy-on-write candidate manifest;
3. canonicalize payload bytes;
4. compute whole-tensor and chunk checksums;
5. enforce staging and storage budgets;
6. write every new chunk through a transaction-local partial file;
7. flush and `fsync` the chunk file;
8. atomically rename the chunk into the content-addressed directory;
9. validate every candidate tensor from referenced chunks;
10. write and `fsync` the candidate manifest;
11. atomically publish the manifest file;
12. write and `fsync` a candidate `CURRENT` pointer;
13. atomically replace `CURRENT`;
14. append the committed journal state;
15. release staged payload references held by the transaction.

Until step 13 completes, the previous manifest remains authoritative.

### 5.1 Copy-on-write behavior

A transaction copies unchanged tensor records from the parent manifest and replaces only records explicitly updated in the candidate. Unchanged chunks remain referenced by content hash.

### 5.2 Transaction payload lifetime

A committed or aborted transaction clears staged payload and explicit-version maps. This prevents a transaction object that remains alive from retaining a hidden full copy of canonical state in RAM.

Regression tests use weak references to verify that staged payloads can be collected after commit or abort.

## 6. Recovery and corruption handling

`store-recover` verifies the current manifest and marks incomplete transactions aborted without publishing candidate state.

Recovery reports, but does not silently delete:

- partial chunk files;
- complete chunks not referenced by a manifest;
- unpublished manifests;
- temporary manifest files;
- temporary pointer files;
- incomplete and aborted transactions.

Validated tensor-store failure points include:

- before a chunk write;
- during a partial chunk write;
- before chunk `fsync`;
- before manifest rename;
- before `CURRENT` rename.

For every accepted failure test, the previous committed manifest remains authoritative.

Checksum validation exists at:

- chunk level;
- whole-tensor level;
- tensor-store manifest level;
- root step-bundle level.

## 7. Framework adapters

### 7.1 PyTorch

The PyTorch adapter exports and restores:

- unique model parameters;
- model buffers;
- AdamW first moments;
- AdamW second moments;
- optimizer step tensors;
- scalar parameter-group metadata.

Shared PyTorch parameters are exported once through `named_parameters(remove_duplicate=True)`.

### 7.2 MLX

The MLX adapter flattens model and optimizer trees into canonical NumPy-compatible arrays. It restores model weights and optimizer state through stable flattened paths.

### 7.3 Adapter boundary

The storage layer owns canonical bytes, versions, checksums, and transactions. Framework adapters own conversion to and from framework arrays.

## 8. Step-bundle store

Version 0.8 introduces a root transaction layer above child tensor stores.

### 8.1 Layout

```text
bundle-root/
  CURRENT
  manifests/
    <bundle-id>.json
  work/
    parameters/
    oracle-gradients/
    bounded-gradients/
    initial-optimizer/
    oracle-state/
  candidates/
    step-1-parameters/
    step-1-optimizer/
```

The exact work and candidate directories are implementation details. The authoritative contract is the root bundle manifest.

### 8.2 Bundle manifest

A `StepBundleManifest` contains:

```text
schema_version
bundle_id
parent_bundle_id
committed_step
created_at_utc
parameter store reference
optimizer store reference
optional gradient store reference
batch checksum
bundle checksum
```

Each child-store reference contains:

```text
relative path
manifest ID
manifest checksum
tensor count
chunk count
logical bytes
physical bytes
```

All referenced child stores must remain inside the bundle root.

### 8.3 Publication protocol

The root publication sequence is:

1. verify candidate parameter, optimizer, and optional gradient stores;
2. capture their current manifest IDs, checksums, counts, and byte totals;
3. require the new committed step to advance exactly by one;
4. write and `fsync` a candidate bundle manifest;
5. optionally inject failure before manifest rename;
6. atomically publish the bundle manifest;
7. write and `fsync` a candidate root `CURRENT` pointer;
8. optionally inject failure before root `CURRENT` rename;
9. atomically replace root `CURRENT`;
10. `fsync` the root directory.

Candidate child stores can contain valid state before step 9. They are not the authoritative training state until the root pointer changes.

### 8.4 Root recovery

`StepBundleStore.recover()` verifies the current root bundle, identifies unpublished bundle manifests and temporary paths, and preserves the current authoritative bundle.

Validated root failure points are:

- before bundle-manifest rename;
- before root `CURRENT` rename.

Both cases must leave the previous committed bundle authoritative.

## 9. Execution groups

The controlled Transformer is decomposed into:

```text
embedding
block-0
block-1
...
final-head
```

An execution group contains an ordered tuple of unique logical parameter names.

### 9.1 Forward group membership

The forward plan includes the tied token embedding in both:

- `embedding`, to map input token IDs to hidden states;
- `final-head`, to project hidden states to vocabulary logits.

The parameter is read twice but is not retained between those groups.

### 9.2 Optimizer group membership

The optimizer plan deduplicates parameter names across forward groups. The tied token embedding is therefore updated exactly once.

## 10. Observable full-state lifecycle

`storage-step` validates storage and transaction behavior before bounded execution.

It performs:

1. deterministic model and AdamW initialization;
2. canonical export and initial commit;
3. destruction of bootstrap objects;
4. one independently restored resident reference update;
5. restoration of the same initial state from storage;
6. forward, backward, clipping, and AdamW;
7. tensor-level resident-versus-storage comparison;
8. atomic publication of the updated state;
9. restoration of the committed update;
10. exact comparison with the published state.

The compute phase materializes the full micro model and optimizer. This path validates lifecycle, not bounded compute.

## 11. Bounded forward

`bounded-forward` creates a parameter-only store, releases bootstrap and resident models, and executes:

```text
embedding
  read -> materialize -> compute -> release

block 0
  read -> materialize -> compute -> release

block N
  read -> materialize -> compute -> release

final-head
  read -> materialize -> compute -> release
```

The resident oracle records:

- embedding output;
- every block output;
- final logits;
- loss.

The bounded executor compares each corresponding boundary.

The current forward path retains hidden boundary activations. Its bounded claim applies to managed parameter residency.

## 12. Bounded backward and gradient storage

`bounded-backward` isolates reverse-mode differentiation from optimizer execution.

### 12.1 Forward preparation

The executor runs the bounded forward path and retains detached group-boundary activations in CPU memory. The parameter manifest remains immutable.

### 12.2 Reverse execution

Groups execute in reverse:

```text
final-head
  read parameters
  materialize saved input activation
  recompute local output
  backward from cross-entropy
  extract parameter gradients and upstream activation gradient
  commit gradients
  release

block N ... block 0
  read parameters
  materialize saved input and incoming gradient
  recompute local output
  backward
  extract gradients and next upstream gradient
  commit gradients
  release

embedding
  read parameters
  recompute embedding output
  backward with incoming gradient
  commit embedding gradients
  release
```

The intended accelerator residency is one parameter group, one local activation, and one incoming activation gradient.

### 12.3 Oracle and bounded gradient stores

The path uses separate stores:

- immutable parameter store;
- resident-oracle gradient store;
- bounded gradient store.

The resident gradient payloads are released before bounded execution.

### 12.4 Tied-gradient versioning

The tied token embedding receives two contributions:

1. the final-head contribution is stored at gradient version 0;
2. the embedding contribution is added and stored at gradient version 1.

The final bounded gradient store contains one final gradient record for each unique parameter.

### 12.5 Global norm

Oracle and bounded gradient stores are streamed independently to calculate their global norms without materializing all gradients together.

The canonical clipping coefficient is:

```text
clip = min(1, max_norm / (global_norm + 1e-6))
```

The complete oracle and bounded gradient states are loaded together only after bounded execution for tensor-level validation.

## 13. Group-bounded AdamW

`bounded-step` first runs the validated bounded-backward path. It then creates zero-initialized AdamW state and publishes root bundle step 0.

For every unique optimizer group it reads:

- parameter tensors;
- final gradient tensors;
- first-moment tensors;
- second-moment tensors;
- optimizer step tensors.

It then:

1. materializes the group on the selected device;
2. multiplies gradients by the canonical clipping coefficient;
3. restores group-local AdamW state;
4. executes AdamW with deterministic reference semantics;
5. exports updated parameters and optimizer state;
6. commits version 1 tensors to candidate stores;
7. releases group payloads and device objects;
8. records memory, timing, checksum, and storage telemetry.

After every group completes:

1. candidate parameter and optimizer stores verify;
2. the complete candidate state is compared with a resident oracle;
3. the candidate state is restored into a fresh model and optimizer;
4. the restored state is re-exported and compared exactly;
5. the root step-1 bundle is published;
6. root and child stores verify.

Complete candidate and oracle states are materialized together only for final validation.

## 14. Working-set budgets

### 14.1 Tensor-store budgets

`StoreLimits` defines:

- chunk size;
- maximum managed storage bytes;
- maximum internal staging bytes.

### 14.2 Parameter budget

Maximum logical parameter bytes materialized for one forward, backward, or optimizer group.

### 14.3 Gradient budget

Maximum logical final-gradient bytes materialized for one backward or optimizer group.

### 14.4 Optimizer budget

Maximum combined logical bytes for:

```text
parameter
+ gradient
+ first moment
+ second moment
+ step tensor
```

A group is rejected before optimizer compute when this sum exceeds the configured limit.

### 14.5 Budget interpretation

Logical working-set budgets do not include all physical memory. Activations, temporary kernels, allocator caches, RSS, page cache, and operating-system overhead remain separate measurements.

## 15. Correctness model

### 15.1 CPU exactness

CPU reference tests require exact canonical state for:

- parameters;
- first moments;
- second moments;
- optimizer step tensors;
- optimizer parameter-group metadata;
- candidate restore.

The resident oracle and group-bounded path use the same canonical clipping coefficient and reference AdamW mode.

### 15.2 MPS numerical validation

MPS target tests report raw loss, gradient-norm, parameter, and optimizer-state differences. Bitwise equality and numerical agreement remain distinct properties.

### 15.3 Validation-only materialization

Result fields explicitly report when complete gradient or candidate state is materialized after bounded execution for validation.

## 16. Telemetry

### 16.1 Tensor-store telemetry

- logical and physical bytes read or written;
- chunk reads and writes;
- chunk reuse;
- checksum time;
- read and write time;
- `fsync` time;
- manifest publication time;
- recovery time and actions;
- current store size;
- cumulative managed-state writes.

### 16.2 Bounded forward telemetry

- group tensor names and parameter bytes;
- referenced chunks;
- read, materialization, compute, and release time;
- input and output activation bytes;
- boundary checksums and numerical differences;
- RSS, accelerator allocation, and driver allocation.

### 16.3 Bounded backward telemetry

- reverse group order;
- local activation and activation-gradient bytes;
- parameter read, recomputation, backward, extraction, commit, and release time;
- gradient logical and physical writes;
- chunk reuse;
- local and upstream-gradient checksums;
- RSS, accelerator allocation, and driver allocation.

### 16.4 Optimizer telemetry

- parameter, gradient, first-moment, second-moment, and step bytes;
- complete group working-set bytes;
- state read, materialization, AdamW, export, commit, and release time;
- parameter and optimizer logical and physical writes;
- chunk writes and reuse;
- updated parameter and optimizer checksums;
- RSS, accelerator allocation, and driver allocation before and after release.

### 16.5 Root publication telemetry

- bundle-manifest `fsync` and rename time;
- root `CURRENT` `fsync` and rename time;
- metadata bytes written;
- parent and child bundle lineage;
- verification and recovery results.

Filesystem metadata, APFS copy-on-write amplification, controller behavior, and NAND-level writes are not inferred from these application counters.

## 17. Validated target results

### 17.1 Version 0.5

A clean MacBook Air M2 run validated exact storage lifecycle, exact MLX round trips, all five tensor-store failure points, exact PyTorch storage-backed state for the tested paths, and zero swap growth.

### 17.2 Version 0.6

A clean M2 run validated micro and tiny bounded forward. Maximum boundary, logits, and loss differences were zero. The largest tiny parameter group was `788,480` bytes under a `1,048,576` byte budget.

### 17.3 Version 0.7

A clean M2 run validated bounded backward, exact final gradients for the tested paths, tied-gradient version 1, streamed global norm, parameter and gradient budget rejection, three-store recovery, and zero swap growth.

### 17.4 Version 0.8

CPU CI on Python 3.11 and 3.13 validates exact resident-versus-candidate state, exact candidate restore, optimizer working-set rejection, bundle checksum validation, and failure recovery. Target MPS validation is pending.

## 18. CLI

Create and inspect a tensor store:

```bash
microcolossus store-init \
  --path runs/store \
  --chunk-size-mib 4 \
  --max-staging-mib 16 \
  --max-storage-gib 100

microcolossus store-verify --path runs/store
microcolossus store-recover --path runs/store
```

Run the observable full-state lifecycle:

```bash
microcolossus storage-step \
  --config examples/micro-storage.yaml \
  --store runs/micro-storage-store \
  --output runs/micro-storage-result.json \
  --device cpu
```

Run bounded forward:

```bash
microcolossus bounded-forward \
  --config examples/micro-storage.yaml \
  --store runs/micro-bounded-forward-store \
  --output runs/micro-bounded-forward.json \
  --device cpu \
  --parameter-working-set-mib 1
```

Run bounded backward:

```bash
microcolossus bounded-backward \
  --config examples/micro-storage.yaml \
  --parameter-store runs/micro-backward-parameters \
  --oracle-gradient-store runs/micro-backward-oracle-gradients \
  --gradient-store runs/micro-backward-gradients \
  --output runs/micro-bounded-backward.json \
  --device cpu \
  --parameter-working-set-mib 1 \
  --gradient-working-set-mib 1
```

Run one complete bounded optimizer step:

```bash
microcolossus bounded-step \
  --config examples/micro-storage.yaml \
  --bundle-store runs/micro-bounded-step \
  --output runs/micro-bounded-step.json \
  --device cpu \
  --parameter-working-set-mib 1 \
  --gradient-working-set-mib 1 \
  --optimizer-working-set-mib 4
```

Every command that creates a store or bundle requires a new destination directory.

## 19. Current limitations

The current implementation does not yet provide:

- multiple consecutive bounded optimizer steps;
- reuse of a prior committed root bundle as the next step input;
- checkpoint and resume across process restart;
- dataset cursor and RNG restoration;
- activation recomputation from storage;
- activation offload;
- strict total-memory-pressure enforcement;
- asynchronous prefetch or writeback;
- intra-layer tiling;
- bounded MLX backward or optimizer execution;
- direct I/O, compression, or storage-specific tuning;
- real-corpus language-model training;
- a demonstrated state larger than safe resident unified memory.

## 20. Next design milestone

The next runtime milestone is consecutive bounded training steps.

The intended sequence is:

```text
open authoritative root bundle N
        |
read its parameter and optimizer child manifests
        |
execute bounded forward, backward, norm, clipping, and AdamW
        |
build candidate child stores for N + 1
        |
verify and publish root bundle N + 1
        |
persist batch cursor, RNG state, and schedule state
        |
allow process exit and exact resume
```

This milestone must also define retention and garbage-collection rules for obsolete manifests, candidate stores, gradients, and chunks without deleting state required for recovery.
