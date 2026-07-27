# Versioned Tensor Store

This document describes the synchronous storage foundation introduced in MicroColossus 0.4.0 and the observable micro training lifecycle introduced in 0.5.0.

## Scope

The tensor store is the first component that can keep canonical model and optimizer state outside framework-owned memory. It is backend-neutral and does not depend on `torch.Tensor` or `mlx.core.array` internally.

Version 0.4.0 implements storage, integrity, versioning, transactions, recovery, budgets, telemetry, and framework adapters. Version 0.5.0 connects that foundation to one fully observed micro optimizer step.

The compute phase still materializes the complete micro model and optimizer. Bounded layer-wise execution is a later milestone.

## Layout

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

Chunks are content-addressed by SHA-256. Published manifests are immutable. `CURRENT` is the only authoritative pointer and is replaced atomically after a candidate manifest and all referenced chunks have been written, synchronized, and validated.

## Canonical tensor representation

Each tensor records:

```text
tensor_id
logical_name
kind
shape
dtype
byte_order
version
ordered chunk IDs
byte length
whole-tensor checksum
committed step
adapter metadata
```

Initial tensor kinds are:

- `parameter`
- `gradient`
- `adam_first_moment`
- `adam_second_moment`
- `master_weight`
- `metadata`

Multi-byte values use canonical little-endian storage. Non-contiguous NumPy inputs are normalized to contiguous logical bytes before checksumming and chunking.

## Transaction protocol

A transaction moves through:

```text
prepared -> writing -> validated -> committed
                            \
                             -> aborted
```

The commit sequence is synchronous:

1. build a copy-on-write candidate manifest;
2. calculate tensor and chunk checksums;
3. reject the transaction if storage or staging limits would be exceeded;
4. write each new chunk through a transaction-local partial file;
5. flush and `fsync` the chunk;
6. atomically rename the chunk into the content-addressed chunk directory;
7. read and validate every candidate chunk and tensor;
8. write and `fsync` the candidate manifest;
9. atomically publish the manifest file;
10. write and `fsync` a candidate `CURRENT` pointer;
11. atomically replace `CURRENT`;
12. append the committed journal state.

Until step 11 completes, the previous manifest remains authoritative.

## Recovery

`store-recover` verifies the current committed state and scans transaction journals. An incomplete transaction is marked aborted without publishing its candidate manifest.

Recovery reports, but does not silently delete:

- partial chunk files;
- complete chunks not referenced by a manifest;
- unpublished manifests;
- temporary pointer or manifest files.

This conservative policy preserves forensic evidence and avoids making cleanup decisions during correctness validation.

## Failure injection

The test suite can interrupt commits at:

- before a chunk write;
- during a partial chunk write;
- before chunk `fsync`;
- before manifest rename;
- before `CURRENT` rename.

Every tested interruption must leave the previously committed manifest authoritative.

## Budgets

`StoreLimits` configures:

- chunk size;
- maximum storage content;
- maximum internal staging buffer.

The first implementation is intentionally conservative. A transaction is rejected before publication when its projected content exceeds the configured storage limit. Each chunk must fit inside the staging limit.

## Telemetry

Commit, read, range-read, chunk-read, and recovery operations report:

- logical bytes read and written;
- physical bytes read and written;
- chunk reads and writes;
- reused chunks;
- checksum time;
- read and write time;
- `fsync` time;
- manifest publication time;
- recovery time and actions;
- current store size;
- cumulative managed-state bytes written.

The observable storage step adds:

- state-read time;
- model and optimizer materialization time;
- forward time;
- backward time;
- gradient-clipping time;
- optimizer-update time;
- state-export time;
- loss and global gradient norm;
- process RSS and accelerator allocator samples;
- initial, updated, and restored canonical state digests;
- tensor-level numerical comparison with the resident oracle;
- committed tensor versions.

Filesystem metadata, APFS copy-on-write behavior, write amplification, and device-level NAND writes are not inferred from these counters. Those require target-hardware measurement.

## Framework adapters

The PyTorch adapter exports and restores:

- unique model parameters;
- buffers;
- AdamW first moments;
- AdamW second moments;
- optimizer step tensors;
- scalar parameter-group settings.

The MLX adapter exports and restores flattened model and optimizer trees through canonical NumPy-compatible arrays. The MLX path requires validation on the target Mac M2.

The store itself remains independent of both frameworks.

## Development scale ladder

Routine development must not require a long 18M or 23M parameter run. MicroColossus uses multiple validation scales:

1. **Unit scale**. Byte payloads, small arrays, isolated operators, transaction states, corruption, and failure injection.
2. **Micro model**. A sub-million-parameter Transformer for every end-to-end storage-backed path. It exposes detailed loss, gradient, tensor-version, memory, I/O, and timing telemetry.
3. **Small real-training model**. A few-million-parameter model on a small real text corpus. It will validate training and validation loss, checkpoint and resume, and sample generation.
4. **Milestone scale**. Larger resident or storage-backed models run only after the same path passes at smaller scales.
5. **Capacity demonstration**. The 124M, 350M, and larger targets remain later research gates.

Small models accelerate iteration. They do not replace real training, resident-versus-storage numerical comparison, or larger-than-resident capacity demonstrations.

A future external training project may become a real-training frontend. The tensor store, planner, transaction protocol, telemetry, and backend interfaces must remain independent of that frontend.

## Observable micro storage step

The `storage-step` experiment performs this sequence:

1. create deterministic model and AdamW state on CPU;
2. export it into the canonical representation;
3. commit the initial state to a new store;
4. destroy the original resident objects;
5. restore one independent resident reference instance;
6. execute one observed full-parameter update;
7. release that model before materializing the storage-backed instance;
8. restore the same initial state from the store;
9. execute the same batch and optimizer update;
10. compare final model and optimizer state with the resident oracle;
11. publish all updated tensor versions atomically;
12. restore the committed update into a third instance;
13. compare the third instance with the state that was published.

CPU validation requires exact bytes. MPS validation reports both bitwise equality and tensor-level numerical distance because Metal reductions may not be bitwise reproducible.

This experiment validates the storage lifecycle around compute. It does not reduce the compute working set yet.

## CLI

Create an empty store:

```bash
microcolossus store-init \
  --path runs/store \
  --chunk-size-mib 4 \
  --max-staging-mib 16 \
  --max-storage-gib 100
```

Verify the current manifest and all data:

```bash
microcolossus store-verify --path runs/store
```

Recover incomplete transactions conservatively:

```bash
microcolossus store-recover --path runs/store
```

Run the fast CPU micro lifecycle:

```bash
microcolossus storage-step \
  --config examples/micro-storage.yaml \
  --store runs/micro-storage-store \
  --output runs/micro-storage-result.json \
  --device cpu
```

On the target Mac, use `--device mps` with the same micro preset first, then repeat with `examples/tiny-mps.yaml` only after the micro run passes.

## Current validation

The isolated storage validation completed:

- PyTorch model and AdamW state round-tripped exactly;
- corruption was detected;
- copy-on-write versions reused unchanged chunks;
- scalar, zero-length, non-contiguous, and big-endian inputs were covered;
- every injected interruption preserved the previous manifest;
- CLI initialization, verification, and recovery completed.

The integrated GitHub Actions matrix passes on Python 3.11 and 3.13:

- installation;
- Ruff;
- mypy;
- the complete pytest suite;
- bytecode compilation;
- resident CPU training and benchmark smoke tests;
- tensor-store initialization, verification, and recovery smoke tests;
- one complete observable CPU storage-step smoke test;
- final store verification after the update.

The CPU integration tests also require:

- exact resident-versus-storage model and AdamW state;
- exact committed-versus-restored state;
- deterministic repeated state and batch digests;
- previous-state recovery after an interrupted update;
- NumPy-independent batch checksums;
- explicit detection of a modified tensor.

The MLX adapter and the MPS storage-step still require clean target-M2 validation.

## Explicitly not validated

Version 0.5.0 does not establish:

- bounded layer-wise storage-to-accelerator execution;
- optimizer execution without materializing the complete micro state;
- activation offloading;
- asynchronous prefetch or writeback;
- direct I/O;
- compression;
- intra-layer tiling;
- real-corpus language-model training;
- training state larger than safe resident unified memory.

The next bounded-execution milestone will keep authoritative state in this store while loading and updating only a declared execution group at a time.
