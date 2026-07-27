# MicroColossus

> **Trade memory for time.**

MicroColossus is an experimental open-source runtime for full-parameter training when the complete training state cannot remain resident in available memory.

The primary target is **Apple Silicon, beginning with an 8 GB Mac M2**.

- **MLX** is the preferred optimized Apple Silicon execution candidate.
- **PyTorch MPS** is the numerical oracle and the reference for validation, debugging, state comparison, and recovery semantics.
- Tensor storage, transactions, and execution plans remain backend-neutral.

The project does not yet claim full out-of-core training. It has validated resident execution, recoverable storage-backed state, a complete observable micro optimizer lifecycle, and bounded parameter-group forward execution. Version 0.7 adds bounded backward propagation and versioned gradient storage for target-hardware validation.

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
- immutable manifests and atomic `CURRENT` publication;
- transaction journals and conservative crash recovery;
- corruption detection and storage or staging budgets;
- read, write, checksum, fsync, publication, recovery, and cumulative-write telemetry;
- PyTorch model and AdamW export and restore;
- MLX model and optimizer-tree export and restore;
- `store-init`, `store-verify`, and `store-recover` commands.

### Observable storage-backed optimizer lifecycle

Version 0.5 performs a complete micro lifecycle:

1. create deterministic model and AdamW state;
2. commit the canonical state to storage;
3. destroy the bootstrap objects;
4. execute an independently restored resident reference step;
5. restore the same state from storage;
6. execute the same optimizer step;
7. publish the updated state atomically;
8. restore the committed result;
9. compare model and optimizer state tensor by tensor.

The compute phase still materializes the complete micro model and optimizer. This validates storage lifecycle, transactions, recovery, and observability. It is not a bounded training result.

### Bounded parameter-group forward

Version 0.6 introduced the first bounded compute path:

- load and release token and positional embeddings;
- load and release one Transformer block at a time;
- load final normalization and the output projection last;
- reload tied token embeddings for the output projection;
- reject a group before compute when it exceeds the parameter budget;
- compare every group boundary, final logits, and loss with the resident oracle;
- report reads, chunks, timings, activation bytes, RSS, accelerator allocation, and driver allocation.

The executor retains hidden activations. Its bounded claim applies to managed parameter residency.

### Bounded backward and gradient storage

Version 0.7 adds the next isolated phase:

- retain detached forward boundary activations on CPU;
- process execution groups in reverse order;
- reload one parameter group at a time;
- recompute the local group with autograd enabled;
- propagate one upstream activation gradient at a time;
- commit parameter gradients into a separate versioned gradient store;
- combine both tied token-embedding gradient contributions;
- stream the gradient store to calculate the global gradient norm;
- compare all final gradients with the resident PyTorch oracle;
- keep the parameter manifest immutable.

The complete final gradient state is materialized only after bounded execution for validation. AdamW updates are intentionally deferred to the next milestone.

## Validated Apple M2 evidence

A clean MacBook Air M2 run with 8 GB unified memory validated version 0.6.0 at commit `1feea9f9eef28e551ad4ae4944614083effa804f`:

- native arm64 execution without Rosetta;
- Ruff, mypy, 56 tests, and compileall;
- two GREEN micro bounded-forward runs;
- bitwise-exact micro repeatability;
- one GREEN 443,648-parameter tiny bounded-forward run;
- zero boundary, logits, and loss differences from the resident oracle;
- correct tied-embedding reload counts;
- parameter groups limited to `33,280` bytes for micro and `788,480` bytes for tiny under a `1,048,576` byte budget;
- intentional budget rejection;
- unchanged parameter manifest;
- successful store verification and recovery;
- no detected fallback, unsupported operation, non-finite value, allocation failure, or swap growth;
- a clean source tree.

Maximum observed RSS was `322,273,280` bytes. Maximum MPS allocation was `1,181,696` bytes. Maximum Metal driver allocation was `19,300,352` bytes. These counters are not summed as physical-memory equivalents.

The earlier resident 23,213,056-parameter benchmark produced:

| Variant | Median tokens/s | Relative to PyTorch |
|---|---:|---:|
| PyTorch MPS | 1,379.23 | 1.000x |
| PyTorch MPS checkpointed | 1,337.37 | 0.970x |
| MLX | 2,195.69 | 1.592x |

The backend decision remains **dual backend**. MLX is the preferred optimized Apple Silicon candidate. PyTorch remains the numerical oracle.

## Fast development scale ladder

Long 18M or 23M parameter runs are not required for routine development.

1. **Unit scale**. Bytes, arrays, isolated operators, corruption, and failure injection.
2. **Micro model**. A sub-million-parameter Transformer for each end-to-end runtime path with detailed telemetry.
3. **Small real training**. A few-million-parameter model on a small text corpus with training loss, validation loss, checkpoint and resume, and sample generation.
4. **Milestone scale**. Larger runs only after the same path is correct at smaller scales.
5. **Capacity demonstrations**. 124M, 350M, and larger targets only after bounded storage-backed training exists.

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

Observable storage-backed optimizer lifecycle:

```bash
microcolossus storage-step \
  --config examples/micro-storage.yaml \
  --store runs/micro-storage-store \
  --output runs/micro-storage-result.json \
  --device cpu
```

Bounded parameter-group forward:

```bash
microcolossus bounded-forward \
  --config examples/micro-storage.yaml \
  --store runs/micro-bounded-forward-store \
  --output runs/micro-bounded-forward.json \
  --device cpu \
  --parameter-working-set-mib 1
```

Bounded backward with separate oracle and bounded gradient stores:

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

## Not yet implemented

- streamed AdamW parameter and optimizer-state updates;
- atomic publication of a complete bounded optimizer step;
- activation recomputation from storage or activation offload;
- asynchronous prefetch and writeback;
- intra-layer tiling;
- bounded MLX backward;
- direct I/O and storage-specific tuning;
- full real-corpus pretraining frontend;
- training state larger than safe resident unified memory.

No full out-of-core training, performance, or model-scale claim is made yet.

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
