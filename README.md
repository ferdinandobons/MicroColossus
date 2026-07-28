# MicroColossus

> **Trade memory for time.**

MicroColossus is an experimental open-source runtime for full-parameter training when the complete managed training state cannot remain safely resident in available memory.

The primary target is **Apple Silicon, beginning with an 8 GB MacBook Air M2**.

- **MLX** is the preferred optimized Apple Silicon execution candidate.
- **PyTorch MPS** is the numerical oracle and reference for validation, debugging, state comparison, and recovery semantics.
- Tensor storage, transactions, root checkpoints, and execution plans remain backend-neutral.

MicroColossus does not yet claim full out-of-core training or training state larger than unified memory. The project has now validated recoverable storage-backed state, group-bounded forward and backward, streamed gradient norm, group-bounded AdamW, atomic step publication, consecutive optimizer steps, and process-level checkpoint resume on the target M2.

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

### Resident and competitive foundation

- controlled decoder-only Transformer;
- full-parameter FP32 AdamW training;
- CPU, MPS, CUDA, and automatic device selection;
- MPS diagnostics and synchronized telemetry;
- resident PyTorch MPS and MLX backends;
- PyTorch activation-checkpointing baseline;
- deterministic portable weights and token batches;
- tensor-level numerical comparison;
- latency, throughput, RSS, allocator, available-memory, and swap reporting.

### Versioned tensor store

- backend-neutral canonical tensor bytes;
- explicit shape, dtype, byte order, kind, version, lineage, and checksum;
- content-addressed immutable chunks;
- copy-on-write tensor versions;
- immutable manifests and atomic `CURRENT` publication;
- transaction journals and conservative recovery;
- corruption detection and storage or staging budgets;
- read, write, checksum, `fsync`, publication, recovery, and cumulative-write telemetry;
- PyTorch model and AdamW export and restore;
- MLX model and optimizer-tree export and restore.

### Bounded parameter-group forward

```text
embedding      read -> compute -> release
block 0        read -> compute -> release
block N        read -> compute -> release
final head     read -> compute -> release
```

The tied token embedding is read again for the final output projection instead of being retained for the whole forward.

### Bounded backward and gradient storage

```text
final head
    -> block N
    -> block 0
    -> embedding
```

The runtime reloads one parameter group, recomputes the local forward, propagates one incoming activation gradient, commits final parameter gradients into a versioned store, and releases the group. The global gradient norm is calculated by streaming the gradient store.

### Group-bounded AdamW and atomic step publication

For each unique optimizer group, the runtime reads:

```text
parameter
+ final gradient
+ Adam first moment
+ Adam second moment
+ optimizer step tensor
```

It applies one canonical clipping coefficient, executes AdamW, writes candidate parameter and optimizer stores, verifies the candidate, restores it, and publishes a root step bundle only after every child store is valid.

The tied token embedding is updated exactly once using its final accumulated gradient.

### Persistent multi-step training and resume

Version 0.9 advances one authoritative root checkpoint through consecutive steps:

```text
bundle step 0
    -> step 1
    -> step 2
    -> ...
    -> step N
```

Each step consumes the parameter and optimizer stores referenced by the current bundle. It does not recreate model or Adam state from initialization.

The persistent root records checksummed training metadata, deterministic batch provenance, parent lineage, child-store manifests, and the committed training step. A new Python process can reopen the root and continue from `CURRENT`.

The first batch-stream contract is:

```text
batch seed = training seed + 1 + committed step
next batch cursor = current committed step
```

A changed model, optimizer recipe, sequence length, microbatch, seed, clipping policy, or schedule is rejected on resume.

## Accepted Apple M2 evidence

### Resident training

A clean native M2 run validated forward, backward, AdamW, automatic MPS selection, telemetry, CPU-versus-MPS numerical comparison, and fixed-batch learning:

```text
5.566842079162598 -> 0.4145740866661072
```

### Competitive resident benchmark

A controlled 23,213,056-parameter workload produced:

| Variant | Median tokens/s | Relative to PyTorch |
|---|---:|---:|
| PyTorch MPS | 1,379.23 | 1.000x |
| PyTorch MPS checkpointed | 1,337.37 | 0.970x |
| MLX | 2,195.69 | 1.592x |

The backend decision remains **dual backend**. MLX is the preferred optimized Apple Silicon candidate. PyTorch remains the numerical oracle.

### Storage and bounded execution through 0.8

Clean target runs on the 8 GB M2 validated:

- exact storage-backed model and optimizer round trips;
- PyTorch and MLX canonical state adapters;
- tensor-store failure injection and recovery;
- bounded forward with zero boundary, logits, and loss differences;
- bounded backward with exact tested gradients;
- streamed global gradient norm;
- group-bounded AdamW;
- exact candidate-versus-resident and candidate-versus-restored state;
- atomic bundle step 0 to step 1;
- failure before manifest or root pointer publication preserving the prior step;
- parameter, gradient, and optimizer working-set rejection;
- no detected hidden CPU fallback or non-finite values.

### Persistent multi-step validation in 0.9

A clean target run validated version 0.9.0 at commit:

```text
4b1ffb20857dd948d7737484e62b007f24bf69b9
```

Environment and quality gate:

- MacBook Air with Apple M2 and 8 GB unified memory;
- native arm64 without Rosetta;
- PyTorch 2.13.0 with MPS built and available;
- Ruff passed;
- mypy passed over 33 source files;
- 80 tests passed with one skip;
- compileall passed;
- fallback remained disabled;
- the source tree remained clean.

Accepted runtime results:

- micro uninterrupted training from step 0 to step 5: GREEN;
- micro training from step 0 to step 2: GREEN;
- a new process resumed the same root from step 2 to step 5: GREEN;
- uninterrupted and resumed final canonical states: BITWISE_EXACT;
- tiny training from step 0 to step 3: GREEN;
- micro lineage: `[0, 1, 2, 3, 4, 5]`;
- tiny lineage: `[0, 1, 2, 3]`;
- deterministic cursor, seed, and batch-checksum equivalence: true;
- all final micro optimizer step tensors: `5.0`;
- all final tiny optimizer step tensors: `3.0`;
- maximum per-step loss difference: `0.0`;
- maximum per-step gradient-norm difference: `1.6985336648289717e-07`;
- maximum final bounded-versus-resident absolute difference: `7.450580596923828e-09`;
- mean final bounded-versus-resident absolute difference: `1.0105799101017887e-10`;
- candidate restore exactness: true;
- both later-step publication failures preserved authoritative step 2;
- configuration mismatch was rejected;
- corrupt current child state was detected;
- parameter, gradient, and optimizer budget rejections passed;
- root and referenced child stores verified and recovered;
- swap delta was zero;
- no hidden fallback or unsupported operation was detected.

Observed maxima were `410,943,488` bytes RSS and `25,935,872` bytes Metal driver allocation. The sampled MPS current-allocation counter reported zero and is retained as reported. It is not treated as evidence that MPS execution did not occur.

## Development scale ladder

Large runs are not required for routine development.

1. **Unit scale**. Bytes, arrays, isolated operators, corruption, and failure injection.
2. **Micro model**. 11,456 parameters for rapid complete runtime paths.
3. **Tiny model**. 443,648 parameters for target-hardware numerical and telemetry gates.
4. **Small real training**. About 1M to 5M parameters on text, with validation, checkpoint, resume, and samples.
5. **Milestone scale**. Larger runs after the same path is correct at smaller scales.
6. **Capacity demonstrations**. 124M, 350M, and later targets after bounded storage-backed training is complete.

Synthetic batches isolate runtime correctness. They are not a substitute for real language-model training.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For native MLX benchmarks on Apple Silicon:

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

Inspect a tensor store:

```bash
microcolossus store-init \
  --path runs/store \
  --chunk-size-mib 4 \
  --max-staging-mib 16 \
  --max-storage-gib 100

microcolossus store-verify --path runs/store
microcolossus store-recover --path runs/store
```

One complete bounded optimizer step:

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

Persistent training to step 2:

```bash
microcolossus-bounded-train \
  --config examples/micro-storage.yaml \
  --bundle-store runs/micro-bounded-training \
  --target-step 2 \
  --output runs/micro-step-2.json \
  --device cpu
```

Resume the same root in a later process to step 5:

```bash
microcolossus-bounded-train \
  --config examples/micro-storage.yaml \
  --bundle-store runs/micro-bounded-training \
  --target-step 5 \
  --output runs/micro-step-5.json \
  --device cpu
```

## Current boundary

Not yet established:

- real tokenizer and corpus training;
- persisted real-dataset shard, sample, epoch, shuffle, and tokenizer state;
- activation recomputation from storage or activation offload;
- strict total-memory-pressure enforcement;
- asynchronous prefetch and writeback;
- intra-layer tiling;
- bounded MLX backward and optimizer execution;
- direct I/O and storage-specific tuning;
- storage pruning and compaction;
- training state larger than safe resident unified memory;
- 124M or 350M full-parameter training on the target Mac.

No full out-of-core, performance-at-scale, or model-quality claim is made yet.

## Next engineering milestone

The next functional gate is a small real-text training frontend. It must preserve the established transactional checkpoint contract while adding tokenizer provenance, dataset identity, deterministic shard and sample position, validation loss, checkpoint resume, and sample generation.

Activation-memory management and larger-than-resident capacity demonstrations remain later milestones.

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
- [`docs/validation.md`](docs/validation.md)
- [`docs/competitive.md`](docs/competitive.md)

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
