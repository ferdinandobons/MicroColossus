# Versioned Tensor Store and Bounded Execution

This document describes the storage foundation introduced in MicroColossus 0.4, the observable optimizer lifecycle introduced in 0.5, bounded forward in 0.6, bounded backward and gradient storage in 0.7, and group-bounded AdamW with atomic step-bundle publication in 0.8.

## Scope

The tensor store keeps canonical training state outside framework-owned memory. Its internal representation does not depend on `torch.Tensor` or `mlx.core.array`.

The current implementation has three distinct compute paths:

- `storage-step` validates an entire optimizer lifecycle, but materializes the complete micro model and optimizer;
- `bounded-forward` materializes one parameter execution group at a time and retains hidden activations;
- `bounded-backward` recomputes one local group at a time, propagates one activation gradient at a time, and persists final parameter gradients in a separate versioned store;
- `bounded-step` consumes those gradients, streams one unique parameter and Adam state group at a time, and atomically publishes a complete root step bundle.

Version 0.8 still implements one isolated step. Multiple-step scheduling and resume remain separate milestones.

## Store layout

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

Chunks are content-addressed by SHA-256. Published manifests are immutable. `CURRENT` is the authoritative pointer and is replaced atomically only after all candidate chunks and the candidate manifest have been written, synchronized, and validated.

## Canonical tensor representation

Each tensor record contains:

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

Initial tensor kinds are:

- `parameter`;
- `gradient`;
- `adam_first_moment`;
- `adam_second_moment`;
- `master_weight`;
- `metadata`.

Multi-byte values use canonical little-endian storage. Non-contiguous inputs are normalized to contiguous logical bytes before checksumming and chunking.

## Transaction protocol

A transaction moves through:

```text
prepared -> writing -> validated -> committed
                            \
                             -> aborted
```

The synchronous commit path is:

1. build a copy-on-write candidate manifest;
2. calculate tensor and chunk checksums;
3. reject the transaction when storage or staging limits would be exceeded;
4. write each new chunk through a transaction-local partial file;
5. flush and `fsync` the chunk;
6. atomically rename the chunk into the content-addressed directory;
7. read and validate all candidate chunks and tensors;
8. write and `fsync` the candidate manifest;
9. atomically publish the manifest file;
10. write and `fsync` a candidate `CURRENT` pointer;
11. atomically replace `CURRENT`;
12. append the committed journal state.

Until step 11 completes, the previous manifest remains authoritative.

## Recovery and failure injection

`store-recover` verifies the current state and marks incomplete transactions aborted without publishing their candidate manifest.

Recovery reports, but does not silently delete:

- partial chunk files;
- complete chunks not referenced by a manifest;
- unpublished manifests;
- temporary pointer or manifest files.

Tests and target-hardware diagnostics inject interruptions:

- before a chunk write;
- during a partial chunk write;
- before chunk `fsync`;
- before manifest rename;
- before `CURRENT` rename.

Every validated interruption leaves the prior committed manifest authoritative.

## Budgets

`StoreLimits` defines:

- chunk size;
- maximum managed storage content;
- maximum internal staging buffer.

Bounded execution adds logical working-set budgets:

- parameter bytes for one execution group;
- gradient bytes for one backward group;
- parameter, gradient, first-moment, second-moment, and step bytes for one optimizer group.

A group is rejected before compute when it exceeds its declared budget. These logical limits do not represent total physical memory. RSS, framework allocations, driver allocations, activations, page cache, compression, and operating-system pressure remain separate measurements.

## Store telemetry

Store operations report:

- logical and physical bytes read or written;
- chunk reads, writes, and reuse;
- checksum time;
- read and write time;
- `fsync` time;
- manifest publication time;
- recovery time and actions;
- current store size;
- cumulative managed-state bytes written.

Filesystem metadata, APFS copy-on-write behavior, controller behavior, and NAND-level writes are not inferred from application counters.

## Framework adapters

The PyTorch adapter exports and restores:

- unique model parameters;
- buffers;
- AdamW first and second moments;
- optimizer step tensors;
- scalar parameter-group settings.

The MLX adapter exports and restores flattened model and optimizer trees through canonical NumPy-compatible arrays.

Storage records and manifests remain independent of both frameworks.

## Observable storage-backed optimizer lifecycle

`storage-step` performs:

1. deterministic model and AdamW initialization;
2. canonical export and initial commit;
3. destruction of the bootstrap objects;
4. one independently restored resident reference update;
5. restoration of the same initial state from storage;
6. the same forward, backward, clipping, and AdamW update;
7. tensor-level comparison with the resident reference;
8. atomic publication of the updated state;
9. restoration of the committed update;
10. exact comparison with the state that was published.

CPU validation requires exact bytes. MPS validation reports bitwise equality and tensor-level numerical distance separately.

## Bounded parameter-group forward

`bounded-forward` creates a parameter-only store, releases the bootstrap and resident models, and executes:

```text
embedding weights
  read -> materialize -> compute -> release

Transformer block 0
  read -> materialize -> compute -> release

Transformer block N
  read -> materialize -> compute -> release

final normalization and output projection
  read -> materialize -> compute -> release
```

When token embedding and output projection weights are tied, the token embedding is read again for the final group. It is not retained from the embedding group.

The resident oracle records the embedding output, every block output, final logits, and loss. The bounded executor compares every corresponding boundary.

The first implementation retains hidden activations between groups. Its bounded claim applies to managed parameter residency only.

## Bounded backward and the gradient store

`bounded-backward` isolates reverse-mode differentiation from optimizer execution.

### Forward preparation

The executor first performs the same parameter-group forward path. It stores detached group-boundary activations in CPU memory. The parameter manifest remains immutable.

### Reverse execution

Groups are processed in reverse order:

```text
final normalization and output projection
  read parameters
  materialize saved input activation
  recompute local output
  backward from cross-entropy
  extract parameter gradients and upstream activation gradient
  commit gradients
  release

Transformer block N
  read parameters
  materialize saved input and incoming gradient
  recompute local output
  backward
  extract gradients and next upstream gradient
  commit gradients
  release

embedding weights
  read parameters
  recompute embedding output
  backward with incoming gradient
  commit embedding gradients
  release
```

Only one parameter group, one local activation, and one incoming activation gradient are intended to be accelerator-resident during each backward group.

### Separate versioned gradient state

Resident-oracle gradients are first written to a dedicated versioned store. The resident gradient payloads and bootstrap parameter payloads are released before bounded execution begins. Bounded gradients are written to a second `VersionedTensorStore`. The parameter store is never modified.

Each unique parameter has one final gradient record. Tied token embeddings receive two contributions:

1. output-head contribution, stored at gradient version 0;
2. embedding contribution, added to the existing value and stored at version 1.

This version history makes tied-gradient accumulation explicit and traceable.

### Global gradient norm

After all groups complete, the oracle and bounded gradient stores are streamed independently to calculate their global norms. The complete states are loaded together only for the final tensor-level validation comparison. The result includes the clipping coefficient that a later optimizer phase would apply. Version 0.7 does not update parameters.

The complete final gradient state is materialized after bounded execution only for tensor-level validation against the resident oracle. The result reports this validation-only materialization explicitly.

### Bounded backward telemetry

Every backward group records:

- reverse ordinal and group name;
- parameter tensor names and bytes;
- referenced chunk count;
- input and output activation bytes;
- incoming and outgoing activation-gradient bytes;
- parameter read and materialization time;
- local recomputation and backward time;
- gradient extraction and commit time;
- logical and physical gradient bytes written;
- chunk writes and reuse;
- release time;
- local gradient and upstream-gradient checksums;
- RSS, accelerator allocation, and driver allocation.

The complete run reports:

- parameter, oracle-gradient, and bounded-gradient manifest IDs;
- batch checksum;
- resident and bounded loss;
- resident and bounded global gradient norm;
- retained CPU activation bytes;
- maximum parameter and gradient group bytes;
- total parameter reads and gradient writes;
- tied-gradient accumulation count and version;
- final tensor-level gradient comparison;
- parameter-manifest immutability;
- store verification results.

## Bounded AdamW and atomic step bundles

`bounded-step` first executes the validated bounded backward path. It then publishes an initial root bundle that references immutable parameter and zero-initialized AdamW stores. For each unique execution group it reads parameters, final gradients, first moments, second moments, and step tensors; applies the shared clipping coefficient; executes AdamW; writes candidate parameter and optimizer versions; and releases the group.

The tied token embedding appears in both the embedding and final-head execution plans, but it is updated only once because optimizer groups deduplicate parameter names. Candidate child stores may advance while they are built, but the authoritative training state remains the root bundle at step 0 until every candidate store verifies and the step-1 bundle `CURRENT` pointer is replaced atomically.

CPU validation compares all parameters, moments, steps, and optimizer metadata exactly with a resident PyTorch oracle. Complete candidate and oracle states are materialized together only after bounded execution for validation.

## Development scale ladder

1. **Unit scale**. Bytes, arrays, operators, corruption, and failure injection.
2. **Micro model**. A sub-million-parameter Transformer for each complete runtime path.
3. **Small real training**. A few-million-parameter model on a real text corpus with validation, checkpoint, resume, and sample generation.
4. **Milestone scale**. Larger runs after the same path is correct at smaller scales.
5. **Capacity demonstrations**. 124M, 350M, and larger targets after bounded storage-backed training exists.

Small models accelerate iteration. They do not replace real training or larger-than-resident demonstrations.

## CLI

Create and inspect a store:

```bash
microcolossus store-init \
  --path runs/store \
  --chunk-size-mib 4 \
  --max-staging-mib 16 \
  --max-storage-gib 100

microcolossus store-verify --path runs/store
microcolossus store-recover --path runs/store
```

Run the observable optimizer lifecycle:

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

## Validated target-hardware results

### Version 0.5

A clean MacBook Air M2 run validated the storage-backed optimizer lifecycle, exact MLX round trips, all five failure-injection points, and zero resident-versus-storage differences for the tested PyTorch paths.

### Version 0.6

A clean MacBook Air M2 run with 8 GB unified memory validated commit `1feea9f9eef28e551ad4ae4944614083effa804f`:

- Ruff, mypy, 56 tests, and compileall passed;
- two micro bounded-forward runs were GREEN and bitwise exact;
- the 443,648-parameter tiny bounded-forward run was GREEN;
- group order and tied-embedding reload behavior were correct;
- the largest micro group used `33,280` bytes;
- the largest tiny group used `788,480` bytes;
- both stayed below a `1,048,576` byte parameter budget;
- every boundary, final logits, and loss matched the resident oracle exactly;
- budget rejection passed;
- parameter manifests remained unchanged;
- stores verified and recovered;
- no fallback, unsupported operation, non-finite value, allocation failure, or swap growth was detected;
- the repository remained clean.

Maximum observed RSS was `322,273,280` bytes. Maximum MPS allocation was `1,181,696` bytes. Maximum Metal driver allocation was `19,300,352` bytes.

### Version 0.7

A clean target run validated two bitwise-exact micro bounded-backward executions and one GREEN tiny execution. Loss and tensor-gradient differences were zero, the maximum global-norm difference was approximately `9.57e-08`, tied-gradient versioning and both budget rejections passed, all three stores verified and recovered, swap growth was zero, and the repository remained clean.

## Current boundary

MicroColossus does not yet establish:

- multiple consecutive bounded optimizer steps;
- checkpoint and resume for the bounded runtime;
- activation recomputation from storage or activation offload;
- asynchronous prefetch or writeback;
- intra-layer tiling;
- MLX bounded backward;
- direct I/O or compression;
- real-corpus language-model training;
- training state larger than safe resident unified memory.

The next hardware gate validates the 0.8 bounded-step path on MPS. After that, development moves to repeated bounded steps, checkpoint/resume, and then activation recomputation or offload.
