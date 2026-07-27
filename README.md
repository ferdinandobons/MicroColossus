# MicroColossus

> **Trade memory for time.**

MicroColossus is an experimental open-source runtime for full-parameter training when the complete training state cannot remain resident in available memory.

The primary target is **Apple Silicon, beginning with an 8 GB Mac M2**.

- **MLX** is the preferred native Apple Silicon execution candidate.
- **PyTorch MPS** is the numerical oracle and the reference for validation, debugging, state comparison, and recovery semantics.
- The tensor store, transaction protocol, and execution plans remain backend-neutral.

The project does not yet claim out-of-core training. It has validated resident execution, durable storage-backed state, a complete micro optimizer lifecycle, and the target-hardware prerequisites for bounded execution.

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

### Observable storage-backed optimizer step

Version 0.5 performs this complete micro lifecycle:

1. create deterministic model and AdamW state;
2. commit the canonical state to the tensor store;
3. destroy the original resident objects;
4. execute an independently restored resident reference step;
5. restore the same state from storage;
6. execute the same optimizer step;
7. publish the updated state atomically;
8. restore the committed result again;
9. compare model and optimizer state tensor-by-tensor.

The compute phase still materializes the complete micro model and optimizer. This validates storage lifecycle, transactions, recovery, and observability. It is not a bounded training result.

### Bounded parameter-group forward

Version 0.6 introduces the first bounded compute path:

- token and positional embeddings are loaded from the store, executed, and released;
- one complete Transformer block is loaded and released at a time;
- final normalization and output projection are loaded last;
- tied token embeddings are reloaded for the output projection instead of being retained;
- every execution group is rejected when its logical parameter bytes exceed the configured budget;
- each group boundary, final logits, and loss are compared with the resident PyTorch oracle;
- reads, chunk references, materialization, compute, release, activation bytes, RSS, accelerator allocation, and driver allocation are reported.

The initial bounded executor retains hidden activations and implements forward propagation only. Bounded backward propagation and streamed optimizer execution remain future work.

## Validated Apple M2 evidence

A clean MacBook Air M2 run with 8 GB unified memory validated version 0.5.0 at commit `82e53c671848d231c2361443882b97dbe4e3a408`:

- native arm64 execution without Rosetta;
- Ruff, mypy, 52 tests, and compileall;
- two GREEN MPS micro storage-step runs with bitwise repeatability;
- one GREEN 443,648-parameter tiny MPS storage-step;
- zero resident-versus-storage loss, gradient-norm, and final-state difference;
- exact storage-versus-restored state;
- all five injected interruption points preserving the prior committed manifest;
- exact MLX micro and tiny round trips;
- GREEN PyTorch-versus-MLX canonical model-state comparisons;
- no detected fallback, unsupported operation, non-finite value, MPS allocation failure, or swap growth;
- a clean source tree before and after validation.

Aggregate application-level activity included `7,468,738` bytes read across `226` referenced chunks and `7,641,680` bytes written across `166` chunk writes, with `60` reused chunks. These counters do not represent NAND-level writes or APFS write amplification.

The resident competitive 23,213,056-parameter workload previously produced:

| Variant | Median tokens/s | Relative to PyTorch |
|---|---:|---:|
| PyTorch MPS | 1,379.23 | 1.000x |
| PyTorch MPS checkpointed | 1,337.37 | 0.970x |
| MLX | 2,195.69 | 1.592x |

The current backend decision is **dual backend**. MLX is the preferred optimized Apple Silicon candidate. PyTorch remains the numerical oracle.

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

Observable storage-backed micro optimizer step:

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

On the target Mac, repeat the bounded-forward command with `--device mps`. Use the micro preset first, then `examples/tiny-mps.yaml` after the micro path passes.

## Not yet implemented

- bounded backward propagation;
- stored or streamed gradients;
- optimizer execution without full-state materialization;
- runtime-managed activation recomputation or offload;
- asynchronous prefetch and writeback;
- intra-layer tiling;
- direct I/O and storage-specific tuning;
- full real-corpus pretraining frontend;
- training state larger than safe resident unified memory.

No out-of-core training, performance, or model-scale claim is made yet.

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
