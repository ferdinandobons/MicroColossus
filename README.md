# MicroColossus

> **Trade memory for time.**

MicroColossus is an experimental open-source runtime for full-parameter training when the complete managed training state cannot remain safely resident in available memory.

The primary target is **Apple Silicon, beginning with an 8 GB MacBook Air M2**.

- **MLX** is the preferred optimized Apple Silicon execution candidate.
- **PyTorch MPS** is the numerical oracle and reference for validation, debugging, state comparison, and recovery semantics.
- Tensor storage, transactions, root checkpoints, data provenance, execution plans, and activation policies remain backend-neutral where practical.

MicroColossus does not yet claim complete out-of-core training or training state larger than unified memory. It has validated recoverable storage-backed state, group-bounded forward and backward, streamed gradient norm, group-bounded AdamW, atomic step publication, consecutive optimizer steps, process-level resume, deterministic real-text learning, and safe checkpoint pruning on the target M2.

Version **0.12.0** adds persistent multi-step activation recomputation. The CPU implementation and regression suite are the release gate before Apple M2 validation.

## Why Apple Silicon changes the design

Apple Silicon uses one unified physical memory pool. CPU tensors, accelerator tensors, framework allocations, the Metal driver, filesystem cache, and macOS compete for the same capacity.

Project rules:

- CPU-to-MPS placement is not capacity offload.
- RSS, MPS, Metal-driver, and MLX allocator counters overlap or describe different scopes.
- Those counters are never added as separate physical memories.
- NVMe is the first capacity tier outside unified memory.
- Memory pressure, swap, storage traffic, elapsed time, recovery, and cumulative writes are first-class metrics.

```text
MLX or MPS active working set
              |
bounded unified-memory staging and activations
              |
versioned storage-backed tensor state
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

### One bounded optimizer step

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

The committed root step is the authoritative next-batch cursor. Configuration provenance, root lineage, batch checksums, tensor versions, optimizer steps, and data identity are verified on resume.

### Activation policies

Version 0.12 introduces:

```text
training.activation_policy: retain_all
training.activation_policy: recompute
```

`retain_all` keeps every non-final forward boundary on CPU until its reverse group executes.

`recompute` keeps zero forward boundaries for later backward use. Each reverse group reconstructs its input by replaying the deterministic prefix from token IDs and the authoritative parameter store.

```text
final-head backward
    replay embedding and all Transformer blocks

block K backward
    replay embedding through block K-1

embedding backward
    use token IDs directly
```

The first recomputation algorithm is synchronous and intentionally favors numerical isolation over throughput. It can replay a quadratic number of groups. Future hybrid policies will retain selected anchors based on measured memory and replay cost.

Separate logical budgets now cover:

- parameters;
- gradients;
- optimizer state;
- retained activations;
- local forward or backward workspace.

The activation policy is part of checkpoint identity. A `retain_all` root cannot silently resume as `recompute`, or the reverse.

### Deterministic real-text frontend

The current data frontend provides:

- local UTF-8 corpus files;
- a fixed byte tokenizer with vocabulary size 256;
- checksummed train and validation identity;
- deterministic split and random-access text windows;
- byte offsets, seeds, and batch checksums;
- validation loss at committed checkpoints;
- deterministic greedy sample generation;
- atomic progress records tied to root bundle IDs;
- resume rejection when corpus bytes or declared data semantics change.

The byte tokenizer is an engineering reference, not a final tokenizer for model quality.

### Safe checkpoint pruning

Version 0.11 established:

```text
dry-run plan
    -> verify CURRENT and retained checkpoints
    -> inventory exact deletion targets
    -> checksum the plan

explicit apply
    -> publish operation journal
    -> delete only proven-unreachable state
    -> verify retained state
    -> keep CURRENT byte-identical
```

The retention policy can preserve:

- `CURRENT`;
- a declared number of previous checkpoints;
- optional milestone checkpoints.

All root bundle manifests remain as lightweight lineage metadata. Only selected checkpoints keep materialized parameter, optimizer, and gradient child stores.

The corrected Apple M2/APFS validation demonstrated:

- deterministic plans;
- byte-identical `CURRENT`;
- safe interruption continuation;
- idempotent repeated apply;
- MPS resume after pruning;
- micro reclamation of `10,739,392` managed bytes;
- small-model reclamation of `289,540,389` selected bytes;
- no hidden CPU fallback or non-finite values;
- clean source state.

## Accepted Apple M2 evidence

| Capability | Result |
|---|---|
| Resident MPS forward, backward, and AdamW | PASS |
| PyTorch MPS versus MLX resident benchmark | PASS, dual backend selected |
| Versioned tensor store and recovery | PASS |
| Group-bounded forward | PASS |
| Group-bounded backward and gradient store | PASS |
| Group-bounded AdamW and atomic root publication | PASS |
| Persistent multi-step and process resume | PASS |
| Deterministic real-text micro and 1.85M training | PASS |
| Safe pruning and post-pruning resume | PASS |
| Persistent activation recomputation | CPU gate in progress, M2 gate pending |

The controlled resident 23,213,056-parameter benchmark produced:

| Variant | Median tokens/s | Relative to PyTorch |
|---|---:|---:|
| PyTorch MPS | 1,379.23 | 1.000x |
| PyTorch MPS checkpointed | 1,337.37 | 0.970x |
| MLX | 2,195.69 | 1.592x |

MLX remains the optimized execution candidate. PyTorch remains the numerical and recovery oracle.

## Development scale ladder

1. **Unit scale**. Bytes, tensors, operators, corruption, and failure injection.
2. **Synthetic micro**. 11,456 parameters for rapid numerical and storage tests.
3. **Real-text micro**. 18,624 parameters for data provenance, validation, samples, resume, pruning, and activation-policy diagnostics.
4. **Tiny**. 443,648 parameters for target-hardware numerical gates.
5. **Small real model**. 1,846,656 parameters for meaningful M2 learning, reclamation, and activation-memory comparisons.
6. **Milestone scale**. Larger workloads after the same path is correct at smaller scales.
7. **Capacity demonstrations**. 124M and 350M after activation scheduling and intra-layer tiling.

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

Inspect the target and static plan:

```bash
microcolossus doctor
microcolossus plan --config examples/real-text-micro-recompute.yaml
```

Persistent `retain_all` training:

```bash
microcolossus-bounded-train \
  --config examples/real-text-micro.yaml \
  --bundle-store runs/retain-all \
  --target-step 5 \
  --output runs/retain-all-step-5.json \
  --device mps \
  --parameter-working-set-mib 1 \
  --gradient-working-set-mib 1 \
  --optimizer-working-set-mib 4 \
  --activation-working-set-mib 1 \
  --workspace-working-set-mib 4
```

Persistent recomputation:

```bash
microcolossus-bounded-train \
  --config examples/real-text-micro-recompute.yaml \
  --bundle-store runs/recompute \
  --target-step 5 \
  --output runs/recompute-step-5.json \
  --device mps \
  --parameter-working-set-mib 1 \
  --gradient-working-set-mib 1 \
  --optimizer-working-set-mib 4 \
  --activation-working-set-mib 1 \
  --workspace-working-set-mib 4
```

Resume the same recomputation root:

```bash
microcolossus-bounded-train \
  --config examples/real-text-micro-recompute.yaml \
  --bundle-store runs/recompute \
  --target-step 10 \
  --output runs/recompute-step-10.json \
  --device mps \
  --parameter-working-set-mib 1 \
  --gradient-working-set-mib 1 \
  --optimizer-working-set-mib 4 \
  --activation-working-set-mib 1 \
  --workspace-working-set-mib 4
```

Create and apply an explicit pruning plan:

```bash
microcolossus-prune plan \
  --config examples/real-text-micro-recompute.yaml \
  --bundle-store runs/recompute \
  --output runs/recompute-pruning-plan.json

microcolossus-prune apply \
  --config examples/real-text-micro-recompute.yaml \
  --bundle-store runs/recompute \
  --plan runs/recompute-pruning-plan.json \
  --output runs/recompute-pruning-report.json
```

## Current boundary

Not yet established:

- accepted Apple M2 comparison of persistent `retain_all` and `recompute`;
- hybrid activation-anchor scheduling;
- activation tensors stored on disk;
- asynchronous activation prefetch or writeback;
- direct-I/O or NVMe-specific performance behavior;
- strict physical total-memory-pressure enforcement;
- intra-layer tiling;
- bounded MLX backward and optimizer execution;
- representative tokenizer or production corpus quality;
- training state larger than safe resident unified memory;
- 124M or 350M full-parameter training on the target Mac.

No complete out-of-core, performance-at-scale, or production model-quality claim is made.

## Next engineering milestones

1. validate persistent `retain_all` versus `recompute` on Apple M2;
2. use measured replay and memory curves to implement hybrid activation anchors;
3. add optional activation storage only after the synchronous reference remains correct;
4. add intra-layer tiling for groups that individually exceed memory budgets;
5. add performance overlap after correctness, recovery, and endurance remain stable.

## Engineering policy

An optimization is accepted only when it improves a declared objective without violating:

- numerical correctness;
- memory limits;
- storage endurance;
- stability;
- recovery semantics;
- reproducibility;
- observability.

Full-parameter training is reported separately from LoRA, QLoRA, quantized optimizer state, low-rank optimizers, and adapter training.

## Documentation

- [`docs/project.md`](docs/project.md)
- [`docs/storage.md`](docs/storage.md)
- [`docs/multistep.md`](docs/multistep.md)
- [`docs/real-text.md`](docs/real-text.md)
- [`docs/pruning.md`](docs/pruning.md)
- [`docs/activations.md`](docs/activations.md)
- [`docs/validation.md`](docs/validation.md)
- [`docs/competitive.md`](docs/competitive.md)

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
