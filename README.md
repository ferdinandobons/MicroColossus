# MicroColossus

> **Trade memory for time.**

MicroColossus is an experimental open-source runtime for full-parameter training when the complete managed training state cannot remain safely resident in available memory.

The primary target is **Apple Silicon, beginning with an 8 GB MacBook Air M2**.

- **MLX** is the preferred optimized Apple Silicon execution candidate.
- **PyTorch MPS** is the numerical oracle and reference for validation, debugging, state comparison, and recovery semantics.
- Tensor storage, transactions, root checkpoints, data provenance, and execution plans remain backend-neutral.

MicroColossus does not yet claim full out-of-core training or training state larger than unified memory. It has validated recoverable storage-backed state, group-bounded forward and backward, streamed gradient norm, group-bounded AdamW, atomic step publication, consecutive optimizer steps, process-level checkpoint resume, and a deterministic real-text learning trajectory on the target M2.

Version **0.10.0** adds the first deterministic local real-text frontend and is accepted on Apple M2 with a documented validation-protocol correction.

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
- deterministic train and validation split;
- deterministic random-access text windows;
- byte offsets, seeds, and checksums for consumed batches;
- validation loss at committed checkpoints;
- deterministic greedy sample generation;
- atomic progress records tied to root bundle IDs;
- resume rejection when corpus bytes or declared data semantics change.

The first tokenizer is deliberately simple. It removes external vocabulary and download dependencies while the storage and resume semantics are validated. It is not presented as the final tokenizer for model quality.

## Accepted Apple M2 evidence

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

### Real-text training in 0.10

Accepted runtime commit:

```text
8bc277123267c3d3f15bf60cd640819fa823d2e3
```

The external report initially labeled the release `FAIL` because the validation prompt expected 11,456 parameters for `real-text-micro.yaml`. The checked configuration correctly contains 18,624 parameters because the byte tokenizer requires a 256-token embedding and the model has a 64-position table. The stale value belonged to the older synthetic micro configuration. No runtime command or integrity gate failed because of this difference.

Accepted target results after correcting that protocol expectation:

- native Apple M2 MPS execution with fallback unset;
- Ruff, mypy, 88 tests with one skip, compileall, and doctor passed;
- data identity matched between independent processes;
- UTF-8 byte-tokenizer round trip passed;
- micro uninterrupted training reached step 20: GREEN;
- micro validation loss decreased from `5.548418998718262` to `3.302267074584961`;
- a new process resumed the micro root from step 5 to step 20: GREEN;
- uninterrupted-versus-resumed state was `NUMERICALLY_STABLE`;
- maximum resumed-state absolute difference was `1.1920928955078125e-07`;
- mean resumed-state absolute difference was `8.844825718731097e-10`;
- cursors, seeds, offsets, batch checksums, sample token IDs, and sample completion matched;
- corpus mutation was rejected before step 3 became authoritative;
- the 1,846,656-parameter small model reached step 10: GREEN;
- small validation loss decreased from `5.687370777130127` to `4.083975553512573`;
- candidate restore was exact for micro and small runs;
- root and child-store verification and recovery passed;
- no runtime fallback, unsupported operation, non-finite value, or unexpected command failure remained after scanner audit;
- final source tree was clean.

The small training root occupied about 632 MB because historical candidates and work stores are retained. Pruning and compaction remain a required engineering milestone.

## Development scale ladder

1. **Unit scale**. Bytes, tensors, operators, corruption, and failure injection.
2. **Synthetic micro model**. 11,456 parameters for fast storage and numerical paths.
3. **Real-text micro model**. 18,624 parameters for tokenizer, data provenance, validation, samples, and resume.
4. **Tiny model**. 443,648 parameters for target-hardware numerical gates.
5. **Small real model**. Approximately 1M to 5M parameters for a real learning trajectory.
6. **Milestone scale**. Larger workloads after the same path is correct at smaller scales.
7. **Capacity demonstrations**. 124M, 350M, and later targets after activation and tiling work.

Included real-text examples:

| Configuration | Parameters | Purpose |
|---|---:|---|
| `examples/real-text-micro.yaml` | 18,624 | deterministic resume, validation, samples, and fast MPS diagnostics |
| `examples/real-text-small.yaml` | 1,846,656 | first meaningful Apple M2 real-text trajectory |

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

Completed and accepted on the target M2:

- byte tokenizer and local corpus identity;
- deterministic split and training windows;
- real-text bounded training and process resume;
- validation loss and deterministic samples;
- progress records linked to root lineage;
- corpus mutation rejection;
- CPU unit and end-to-end gates;
- Python 3.11 and 3.13 CI coverage;
- micro step 0 to 20;
- process restart and resume from step 5 to step 20;
- 1.85M-parameter small step 0 to 10;
- target store verification and recovery;
- numerical comparison with resident replay;
- clean source-tree verification.

## Current boundary

Not yet established:

- representative tokenizer or corpus quality;
- production model quality;
- persisted large-dataset shards, epochs, and shuffle state;
- activation recomputation from storage or activation offload;
- strict total-memory-pressure enforcement;
- asynchronous prefetch and writeback;
- intra-layer tiling;
- bounded MLX backward and optimizer execution;
- storage pruning and compaction;
- training state larger than safe resident unified memory;
- 124M or 350M full-parameter training on the target Mac.

No full out-of-core, performance-at-scale, or production model-quality claim is made yet.

## Next engineering milestone

The next work should prioritize:

1. historical-state pruning and compaction so longer trajectories do not consume storage linearly;
2. activation recomputation, optional activation offload, and strict total-memory-pressure budgets;
3. a larger and more representative corpus and tokenizer adapter after runtime semantics remain stable;
4. performance work only after the synchronous reference path stays numerically and transactionally correct.

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
