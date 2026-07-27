# Versioned Tensor Store and Bounded Execution

This document describes the storage foundation introduced in MicroColossus 0.4, the observable storage-backed optimizer lifecycle introduced in 0.5, and the first bounded parameter-group forward executor introduced in 0.6.

## Scope

The tensor store keeps canonical model and optimizer state outside framework-owned memory. Its internal representation does not depend on `torch.Tensor` or `mlx.core.array`.

The 0.5 optimizer lifecycle still materializes the complete micro model and optimizer for compute. The 0.6 bounded forward path instead materializes one declared parameter execution group at a time. It retains hidden activations and does not yet implement bounded backward or optimizer execution.

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

The bounded forward command adds a separate logical parameter working-set budget. An execution group is rejected before compute when its referenced parameter bytes exceed that budget.

The current budget is a logical managed-parameter limit. It is not a claim that process RSS, framework allocation, driver allocation, activations, and operating-system pressure are all captured by one number.

## Telemetry

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

The observable optimizer lifecycle also reports:

- state-read and materialization time;
- forward, backward, clipping, and optimizer time;
- state-export time;
- loss and global gradient norm;
- process RSS and accelerator samples;
- canonical state digests and tensor versions;
- numerical distance from the resident oracle.

The bounded forward executor reports for every group:

- group ordinal and logical tensor names;
- tensor and referenced-chunk count;
- logical parameter bytes;
- read, materialization, compute, and release time;
- input and output activation bytes;
- activation checksum;
- numerical distance from the matching resident boundary;
- process RSS;
- accelerator allocation after materialization, compute, and release;
- Metal driver allocation when available.

Filesystem metadata, APFS copy-on-write behavior, controller behavior, and NAND-level writes are not inferred from application counters.

## Framework adapters

The PyTorch adapter exports and restores:

- unique model parameters;
- buffers;
- AdamW first and second moments;
- optimizer step tensors;
- scalar parameter-group settings.

The MLX adapter exports and restores flattened model and optimizer trees through canonical NumPy-compatible arrays.

The store and manifest format remain independent of both frameworks.

## Observable storage-backed optimizer step

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

CPU validation requires exact bytes. MPS validation reports both bitwise equality and tensor-level numerical distance.

## Bounded parameter-group forward

`bounded-forward` creates a parameter-only store, releases the bootstrap and resident models, and executes this plan:

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

Run the bounded forward reference:

```bash
microcolossus bounded-forward \
  --config examples/micro-storage.yaml \
  --store runs/micro-bounded-forward-store \
  --output runs/micro-bounded-forward.json \
  --device cpu \
  --parameter-working-set-mib 1
```

## Validated 0.5 target-hardware result

A clean MacBook Air M2 run with 8 GB unified memory validated commit `82e53c671848d231c2361443882b97dbe4e3a408`:

- Ruff, mypy, 52 tests, and compileall passed;
- two MPS micro storage-step runs were GREEN and bitwise repeatable;
- the 443,648-parameter tiny MPS storage-step was GREEN;
- resident-versus-storage loss, gradient norm, and final canonical state differences were zero;
- storage-versus-restored state was exact;
- all five interruption points preserved the previous committed manifest;
- MLX micro and tiny round trips were exact;
- PyTorch-versus-MLX canonical model state was GREEN;
- no fallback, unsupported operation, non-finite value, allocation failure, or swap growth was detected;
- the repository remained clean.

The run reported `7,468,738` bytes read across `226` referenced chunks and `7,641,680` bytes written across `166` chunk writes, with `60` reused chunks. Maximum observed RSS was `432,472,064` bytes. Maximum MPS allocation was `8,709,376` bytes. Maximum Metal driver allocation was `28,049,408` bytes.

## Current boundary

MicroColossus does not yet establish:

- bounded backward propagation;
- stored or streamed gradients;
- bounded AdamW execution;
- activation recomputation or offload;
- asynchronous prefetch or writeback;
- intra-layer tiling;
- direct I/O or compression;
- real-corpus language-model training;
- training state larger than safe resident unified memory.

The next step after target validation of bounded forward is a reverse-order bounded backward path, followed by a second streamed pass for global clipping and AdamW publication.
