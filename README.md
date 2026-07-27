# MicroColossus

> **Trade memory for time.**

MicroColossus is an experimental open-source runtime for full-parameter training when the complete training state cannot remain resident in available memory.

The primary target is **Apple Silicon, beginning with an 8 GB Mac M2**.

- **MLX** is the preferred native Apple Silicon execution candidate.
- **PyTorch MPS** is the portable numerical oracle and the reference for validation, debugging, state comparison, and recovery semantics.
- The tensor store and transaction protocol are backend-neutral.

The project does not yet claim out-of-core training. It now has the storage and micro-integration foundations required to build that path.

## Why Apple Silicon changes the design

Apple Silicon uses one unified physical memory pool. CPU tensors, accelerator tensors, framework allocations, the Metal driver, and macOS compete for the same capacity.

Project rules:

- CPU-to-MPS placement is not capacity offload.
- RSS, MPS, driver, and MLX allocator counters overlap or describe different scopes.
- Those counters are never added as separate physical memories.
- NVMe is the first capacity tier outside unified memory.
- Memory pressure, swap, storage traffic, elapsed time, recovery, and SSD writes are first-class metrics.

```text
MLX or MPS active working set
              |
bounded unified-memory staging
              |
versioned NVMe tensor store
```

## Current capabilities

### Resident and competitive foundation

- controlled decoder-only Transformer;
- full-parameter AdamW training;
- CPU, MPS, CUDA, and automatic device selection;
- MPS diagnostics and synchronized telemetry;
- resident PyTorch MPS and MLX backends;
- PyTorch activation checkpointing;
- deterministic portable FP32 weights and token batches;
- tensor-level numerical comparison;
- latency, throughput, RSS, allocator, available-memory, and swap reporting.

### Versioned tensor store

- backend-neutral canonical tensor bytes;
- explicit shape, dtype, byte order, kind, version, and checksum;
- content-addressed immutable chunks;
- copy-on-write tensor versions;
- immutable manifests;
- atomic `CURRENT` publication;
- transaction journals;
- conservative crash recovery;
- corruption detection;
- storage and staging budgets;
- read, write, checksum, fsync, publication, recovery, and cumulative-write telemetry;
- PyTorch model and AdamW export and restore;
- MLX model and optimizer-tree adapters;
- `store-init`, `store-verify`, and `store-recover` commands.

### Observable micro storage step

Version 0.5 adds a fast integration experiment that:

1. creates deterministic model and AdamW state;
2. commits the canonical state to the tensor store;
3. destroys the original resident objects;
4. executes an independently restored resident reference step;
5. restores the same state from storage;
6. executes the same optimizer step;
7. publishes the updated state atomically;
8. restores the committed result again;
9. compares model and optimizer state tensor-by-tensor.

It reports forward, backward, clipping, optimizer, materialization, export, read, checksum, fsync, and publication costs.

The complete micro model is still materialized for compute. This validates storage lifecycle and observability, not bounded layer-wise execution.

## Validated resident result

A clean MacBook Air M2 run with 8 GB unified memory validated the resident paths. The fixed-batch loss moved from `5.566842079162598` to `0.4145740866661072`, with no detected fallback, unsupported operator, non-finite value, or MPS out-of-memory failure in the tested path.

The competitive 23,213,056-parameter resident workload produced:

| Variant | Median tokens/s | Relative to PyTorch |
|---|---:|---:|
| PyTorch MPS | 1,379.23 | 1.000x |
| PyTorch MPS checkpointed | 1,337.37 | 0.970x |
| MLX | 2,195.69 | 1.592x |

PyTorch versus MLX had a maximum loss difference of about `1.91e-06` and a maximum final-parameter absolute difference of about `3.80e-05`. The decision is **dual backend**.

## Fast development scale ladder

Long 18M or 23M parameter runs are not required for routine development.

1. **Unit scale**. Bytes, arrays, isolated operators, corruption, and failure injection.
2. **Micro model**. A sub-million-parameter Transformer for every end-to-end runtime path with detailed telemetry.
3. **Small real training**. A few-million-parameter model on a small text corpus with training loss, validation loss, checkpoint and resume, and sample generation.
4. **Milestone scale**. Larger runs only after the same path is correct at smaller scales.
5. **Capacity demonstrations**. 124M, 350M, and larger targets only after bounded storage-backed execution exists.

A future external training project may become the real-training frontend. Storage, scheduling, transactions, and backend interfaces will remain independent of it.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For MLX benchmarks on native Apple Silicon:

```bash
python -m pip install -e ".[dev,benchmark]"
```

## Run

Resident reference:

```bash
microcolossus doctor
microcolossus plan --config examples/tiny-mps.yaml
microcolossus train --config examples/tiny-mps.yaml
```

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

Run the fast observable CPU micro step:

```bash
microcolossus storage-step \
  --config examples/micro-storage.yaml \
  --store runs/micro-storage-store \
  --output runs/micro-storage-result.json \
  --device cpu
```

The same lifecycle can be requested on the target Mac with `--device mps` after installing the project in a clean environment.

## Not yet implemented

- layer-wise bounded storage-to-accelerator execution;
- optimizer execution without full micro-state materialization;
- runtime-managed activation recomputation;
- asynchronous prefetch and writeback;
- intra-layer tiling;
- direct I/O and storage-specific tuning;
- full real-corpus pretraining frontend;
- training state larger than safe resident unified memory.

No out-of-core performance or model-scale claim is made yet.

## Engineering policy

An optimization is accepted only when it improves a declared objective without violating:

- numerical correctness;
- memory limits;
- storage endurance;
- stability;
- recovery semantics;
- reproducibility.

Full-parameter training is reported separately from LoRA, QLoRA, low-rank optimizer methods, quantized optimizer state, and adapter training.

## Documentation

- [`docs/project.md`](docs/project.md)
- [`docs/validation.md`](docs/validation.md)
- [`docs/competitive.md`](docs/competitive.md)
- [`docs/storage.md`](docs/storage.md)

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
