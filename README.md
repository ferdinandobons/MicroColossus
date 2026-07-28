# MicroColossus

> **Trade memory for time.**

MicroColossus is an experimental open-source runtime for full-parameter training when the complete managed training state cannot remain safely resident in available memory.

The primary target is **Apple Silicon, beginning with an 8 GB MacBook Air M2**.

- **MLX** is the preferred optimized Apple Silicon execution candidate.
- **PyTorch MPS** is the numerical oracle and reference for validation, debugging, state comparison, and recovery semantics.
- Tensor storage, transactions, root checkpoints, data provenance, and execution plans remain backend-neutral.

MicroColossus does not yet claim full out-of-core training or training state larger than unified memory. It has validated recoverable storage-backed state, group-bounded forward and backward, streamed gradient norm, group-bounded AdamW, atomic step publication, consecutive optimizer steps, and process-level checkpoint resume on the target M2.

Version **0.10.0** adds the first deterministic local real-text frontend. Its CPU and Apple M2 gates are kept separate from the accepted 0.9 evidence.

## Why Apple Silicon changes the design

Apple Silicon uses one unified physical memory pool. CPU tensors, accelerator tensors, framework allocations, the Metal driver, filesystem cache, and macOS compete for the same capacity.

Project rules:

- CPU-to-MPS placement is not capacity offload.
- RSS, MPS, driver, and MLX allocator counters overlap or describe different scopes.
- Those counters are never added as separate physical memories.
- NVMe is the first capacity tier outside unified memory.
- Memory pressure, swap, storage traffic, elapsed time, recovery, and cumulative writes are first-class metrics.

```text
MLX or MPS active working set
              |
bounded unified-memory staging and activations
              |
versioned NVMe tensor state
```

## Current architecture

### Versioned state

- backend-neutral canonical tensor bytes;
- explicit shape, dtype, byte order, kind, version, lineage, and checksum;
- content-addressed immutable chunks;
- copy-on-write tensor versions;
- immutable manifests and atomic `CURRENT` publication;
- transaction journals and conservative recovery;
- corruption detection and storage or staging budgets;
- PyTorch model and AdamW export and restore;
- MLX model and optimizer-tree export and restore.

### Bounded training step

```text
parameter-group forward
        ↓
reverse group backward
        ↓
versioned final gradients
        ↓
streamed global norm and clipping
        ↓
group-bounded AdamW
        ↓
candidate parameter and optimizer stores
        ↓
exact restore validation
        ↓
atomic root bundle publication
```

The tied token embedding is:

- read once for embeddings and again for the final projection;
- given both backward contributions;
- updated exactly once by AdamW.

### Persistent multi-step training

```text
bundle step 0
    -> step 1
    -> step 2
    -> ...
    -> step N
```

Every step consumes parameter and Adam state referenced by the current root bundle. A later process can reopen the same directory and continue from `CURRENT` without rebuilding state from initialization.

The committed root step is the authoritative next-batch cursor. Configuration provenance, root lineage, batch checksums, tensor versions, and optimizer step values are verified on resume.

### Deterministic real-text frontend

Version 0.10 introduces:

- local UTF-8 corpus files;
- a fixed byte tokenizer with vocabulary size 256;
- checksummed train and validation data identity;
- deterministic train/validation split;
- deterministic random-access text windows;
- byte offsets, seeds, and checksums for consumed batches;
- validation loss at committed checkpoints;
- deterministic greedy sample generation;
- atomic progress records tied to root bundle IDs;
- resume rejection when corpus bytes or declared data semantics change.

The first tokenizer is deliberately simple. It removes external vocabulary and download dependencies while the storage and resume semantics are validated. It is not presented as the final tokenizer for model quality.

## Accepted Apple M2 evidence through 0.9

### Competitive resident baseline

A controlled 23,213,056-parameter resident workload produced:

| Variant | Median tokens/s | Relative to PyTorch |
|---|---:|---:|
| PyTorch MPS | 1,379.23 | 1.000x |
| PyTorch MPS checkpointed | 1,337.37 | 0.970x |
| MLX | 2,195.69 | 1.592x |

The decision remains **dual backend**. MLX is the optimized execution candidate. PyTorch remains the numerical oracle.

### Storage and bounded execution

Clean target runs on the 8 GB M2 validated:

- exact storage-backed model and optimizer round trips;
- PyTorch and MLX canonical adapters;
- tensor-store failure injection and recovery;
- bounded forward with zero tested boundary, logits, and loss differences;
- bounded backward with exact tested gradients;
- streamed global gradient norm;
- group-bounded AdamW;
- exact candidate restore;
- atomic root publication;
- parameter, gradient, and optimizer budget rejection;
- no detected hidden CPU fallback or non-finite values.

### Persistent resume in 0.9

Accepted runtime commit:

```text
4b1ffb20857dd948d7737484e62b007f24bf69b9
```

Validated results:

- micro uninterrupted training from step 0 to step 5: GREEN;
- a new process resumed the same root from step 2 to step 5: GREEN;
- uninterrupted and resumed final canonical states: BITWISE_EXACT;
- tiny training from step 0 to step 3: GREEN;
- maximum per-step loss difference: `0.0`;
- maximum gradient-norm difference: `1.6985336648289717e-07`;
- maximum final bounded-versus-resident absolute difference: `7.450580596923828e-09`;
- candidate restore exactness: true;
- later-step interruption preserved the previous bundle;
- configuration mismatch was rejected;
- corrupted authoritative state was detected;
- swap delta was zero;
- final source tree was clean.

## Development scale ladder

1. **Unit scale**. Bytes, tensors, operators, corruption, and failure injection.
2. **Micro model**. 11,456 parameters for rapid complete runtime paths.
3. **Tiny model**. 443,648 parameters for target-hardware numerical gates.
4. **Small real model**. Approximately 1M to 5M parameters for a real learning trajectory.
5. **Milestone scale**. Larger workloads after the same path is correct at smaller scales.
6. **Capacity demonstrations**. 124M, 350M, and later targets after activation and tiling work.

Included real-text examples:

| Configuration | Parameters | Purpose |
|---|---:|---|
| `examples/real-text-micro.yaml` | 11,456 | CI, deterministic resume, validation, and fast MPS diagnostics |
| `examples/real-text-small.yaml` | about 1.85M | first meaningful Apple M2 real-text trajectory after the micro gate |

The included corpus is original project text used as an engineering fixture. It is not a representative language-model dataset.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For native MLX benchmarks:

```bash
python -m pip install -e ".[dev,benchmark]"
```

## Run

Environment and static plan:

```bash
microcolossus doctor
microcolossus plan --config examples/real-text-micro.yaml
microcolossus plan --config examples/real-text-small.yaml
```

Resident real-text baseline:

```bash
microcolossus train \
  --config examples/real-text-micro.yaml \
  --steps 20 \
  --device cpu
```

Persistent bounded real-text training:

```bash
microcolossus-bounded-train \
  --config examples/real-text-micro.yaml \
  --bundle-store runs/real-text-training \
  --target-step 10 \
  --output runs/real-text-step-10.json \
  --device cpu
```

Resume the same root in a later process:

```bash
microcolossus-bounded-train \
  --config examples/real-text-micro.yaml \
  --bundle-store runs/real-text-training \
  --target-step 20 \
  --output runs/real-text-step-20.json \
  --device cpu
```

Progress records are written under:

```text
runs/real-text-training/metrics/
```

## Version 0.10 validation status

Implemented:

- byte tokenizer and local corpus identity;
- deterministic split and training windows;
- real-text bounded training and process resume;
- validation loss and deterministic samples;
- progress records linked to root lineage;
- corpus mutation rejection;
- CPU unit and end-to-end gates;
- Python 3.11 and 3.13 CI coverage.

The release is not accepted as target evidence until the clean Apple M2 protocol passes. The roughly 1.85M-parameter experiment follows only after the real-text micro gate is GREEN.

## Current boundary

Not yet established:

- accepted Apple M2 validation of version 0.10;
- representative tokenizer or corpus quality;
- persisted large-dataset shards, epochs, and shuffle state;
- activation recomputation from storage or activation offload;
- strict total-memory-pressure enforcement;
- asynchronous prefetch and writeback;
- intra-layer tiling;
- bounded MLX backward and optimizer execution;
- storage pruning and compaction;
- training state larger than safe resident unified memory;
- 124M or 350M full-parameter training on the target Mac.

No full out-of-core, performance-at-scale, or model-quality claim is made yet.

## Engineering policy

An optimization is accepted only when it improves a declared objective without violating:

- numerical correctness;
- memory limits;
- storage endurance;
- stability;
- recovery semantics;
- reproducibility;
- observability.

Full-parameter training is reported separately from LoRA, QLoRA, low-rank optimizer methods, quantized optimizer state, and adapter training.

## Documentation

- [`docs/project.md`](docs/project.md)
- [`docs/storage.md`](docs/storage.md)
- [`docs/multistep.md`](docs/multistep.md)
- [`docs/real-text.md`](docs/real-text.md)
- [`docs/validation.md`](docs/validation.md)
- [`docs/competitive.md`](docs/competitive.md)

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
