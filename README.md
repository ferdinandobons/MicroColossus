# MicroColossus

> **Trade memory for time.**

MicroColossus is an experimental open-source runtime for full-parameter generative-model training when the complete managed training state cannot remain safely resident in available memory.

The primary target is **Apple Silicon, beginning with an 8 GB MacBook Air M2**.

- **MLX** is the preferred optimized Apple Silicon execution candidate.
- **PyTorch MPS** is the numerical oracle and the reference for validation, debugging, state comparison, and recovery semantics.
- Tensor storage, transactions, root checkpoints, data provenance, retention, and activation policies remain backend-neutral where practical.

MicroColossus does not yet claim complete out-of-core training or training state larger than unified memory. It has validated recoverable storage-backed state, group-bounded forward and backward, streamed global clipping, group-bounded AdamW, atomic step publication, consecutive optimizer steps, process-level resume, deterministic real-text learning, safe checkpoint pruning, and persistent activation recomputation on the target M2.

Version **0.13.0** adds the measured hybrid activation-anchor runtime on top
of the accepted 0.12.0 recomputation baseline. Native Apple M2 validation for
0.12.0 passed at commit `4742f8a7f57a46edb075159275fb66c83c78ced7`. Native
Apple M2 validation for 0.13.0 M6C passed on PR `#36` at commit
`8e9b0f8e58fdaa288ba551d994d9b8b81adbea12`.

## Current status

As of 2026-07-29, the current development head is PR `#36`
(`agent/m6c-profile-planner`) at commit
`e3de605119dc8bf481a03ddc38283879fcc350ee`.

Current state:

- PR `#36` is open and still marked draft, with a clean merge state.
- GitHub Actions run `30465100374` passed on Python 3.11 and 3.13, including
  lint, type check, pytest, compile, and CPU smoke.
- Local gates also passed on the Apple M2 development machine: Ruff, mypy,
  pytest, compileall, and `microcolossus doctor`.
- The M6C Apple M2 target validation passed on the runtime commit
  `8e9b0f8e58fdaa288ba551d994d9b8b81adbea12` before this documentation commit.
- The validated M6C scope covers micro, tiny, and small MPS policy comparisons,
  deterministic profiles and plans, checksum/provenance equivalence, numerical
  state comparison, process resume, plan identity rejection, pruning followed
  by resume, and simulated publication-failure recovery.

Still incomplete:

- PR `#36` has not been merged into `main`.
- No final MicroColossus 0.13.0 tag or GitHub Release exists.
- The project still does not claim larger-than-memory training, activation
  tensor offload, asynchronous I/O, direct NVMe behavior, intra-layer tiling,
  or bounded MLX backward and optimizer execution.
- The next engineering phase is to remove or reduce validation-only full-state
  materialization and then produce the first larger-than-memory proof.

## Why Apple Silicon changes the design

Apple Silicon uses one unified physical memory pool. CPU tensors, accelerator tensors, framework allocations, the Metal driver, filesystem cache, and macOS compete for the same capacity.

Project rules:

- CPU-to-MPS placement is not capacity offload.
- RSS, MPS, Metal-driver, and MLX counters describe overlapping or different scopes.
- Those counters are never added as independent physical memories.
- Storage is the first capacity tier outside unified memory.
- Memory pressure, swap, storage traffic, elapsed time, recovery, replay, and cumulative writes are first-class metrics.

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
- PyTorch model and AdamW export or restore;
- MLX model and optimizer-tree export or restore.

### Persistent bounded optimizer steps

```text
parameter-group forward
        |
reverse group backward
        |
versioned final gradients
        |
streamed global norm and clipping
        |
group-bounded AdamW
        |
candidate parameter and optimizer stores
        |
restore and oracle comparison
        |
atomic root bundle publication
```

Every step consumes the parameter and Adam state referenced by the current root bundle. A later process can reopen the same directory and continue from `CURRENT`.

The committed root step is the authoritative next-batch cursor. Configuration provenance, data identity, batch checksums, parent lineage, tensor versions, and optimizer steps are verified on resume.

### Activation policies

MicroColossus supports:

```text
training.activation_policy: retain_all
training.activation_policy: recompute
training.activation_policy: hybrid
```

`retain_all` keeps each non-final forward boundary on CPU until the matching reverse group executes.

`recompute` keeps zero forward boundaries for later backward use. Each reverse group reconstructs its input by replaying the deterministic prefix from token IDs and the authoritative parameter store.

```text
final-head backward
    replay embedding and all Transformer blocks

block K backward
    replay embedding through block K-1

embedding backward
    use token IDs directly
```

The first recomputation schedule is synchronous and favors correctness and
observability over throughput. Prefix replay can be quadratic in group count.

`hybrid` uses the M6C measured-budget planner to retain selected forward
anchors, then reconstructs each backward input from its nearest retained
preceding anchor:

```bash
microcolossus-activation-plan \
  --config examples/real-text-micro-hybrid.yaml \
  --profile-output runs/hybrid-profile.json \
  --plan-output runs/hybrid-plan.json
```

This produces checksummed profile and plan JSON for `hybrid` activation
policy. During persistent training, the root also stores hybrid profile and
plan identity so resume rejects changed anchors, budgets, or plan checksums.

Separate logical budgets cover:

- parameters;
- gradients;
- optimizer state;
- retained activations;
- local forward or backward workspace.

The activation policy is part of checkpoint identity. A root cannot silently change policy at resume.

The accepted M2 gate demonstrated numerically stable `retain_all` and `recompute` trajectories on micro, tiny, and the 1.85M-parameter small workload. `recompute` retained zero forward-boundary bytes and reduced the small logical boundary footprint from `491,520` bytes to `0`.

The same small gate measured a higher sampled peak RSS under full-prefix recomputation:

```text
retain_all: 444,071,936 bytes
recompute:  533,528,576 bytes
```

This is a deliberate result, not a hidden failure. The logical zero-boundary
contract passed, while the synchronous schedule did not reduce measured
physical RSS for that workload. M6C therefore uses a measured hybrid anchor
planner rather than assuming that maximal recomputation is optimal.

### Deterministic real-text frontend

The current frontend provides:

- local UTF-8 corpus files;
- tokenizer `utf8-bytes-v1` with vocabulary size 256;
- checksummed train and validation identity;
- deterministic split and random-access text windows;
- byte offsets, seeds, and batch checksums;
- validation loss at committed checkpoints;
- deterministic greedy sample generation;
- resume rejection when corpus bytes or declared data semantics change.

The byte tokenizer is an engineering reference, not a production tokenizer claim.

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

The corrected Apple M2/APFS validation demonstrated:

- deterministic non-mutating plans;
- byte-identical `CURRENT`;
- interruption continuation and pre-journal retry;
- idempotent repeated apply;
- MPS resume after pruning;
- micro reclamation of `10,739,392` bytes;
- small-model reclamation of `289,540,389` selected bytes;
- no detected hidden CPU fallback or non-finite values;
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
| Persistent activation recomputation | PASS |
| Measured hybrid activation-anchor execution | PASS |

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

## Environment and static plan

```bash
microcolossus doctor
microcolossus plan --config examples/real-text-micro-recompute.yaml
microcolossus plan --config examples/real-text-small-recompute.yaml
```

## Persistent recomputation training

```bash
microcolossus-bounded-train \
  --config examples/real-text-micro-recompute.yaml \
  --bundle-store runs/recompute-training \
  --target-step 5 \
  --output runs/recompute-step-5.json \
  --device mps \
  --parameter-working-set-mib 1 \
  --gradient-working-set-mib 1 \
  --optimizer-working-set-mib 4 \
  --activation-working-set-mib 1 \
  --workspace-working-set-mib 4
```

Resume the same root in a later process:

```bash
microcolossus-bounded-train \
  --config examples/real-text-micro-recompute.yaml \
  --bundle-store runs/recompute-training \
  --target-step 10 \
  --output runs/recompute-step-10.json \
  --device mps \
  --parameter-working-set-mib 1 \
  --gradient-working-set-mib 1 \
  --optimizer-working-set-mib 4 \
  --activation-working-set-mib 1 \
  --workspace-working-set-mib 4
```

## Pruning

Create a non-mutating plan:

```bash
microcolossus-prune plan \
  --config examples/real-text-micro-recompute.yaml \
  --bundle-store runs/recompute-training \
  --output runs/pruning-plan.json
```

Inspect the plan, then apply it explicitly:

```bash
microcolossus-prune apply \
  --config examples/real-text-micro-recompute.yaml \
  --bundle-store runs/recompute-training \
  --plan runs/pruning-plan.json \
  --output runs/pruning-report.json
```

## Current boundary

Not yet established:

- activation tensors stored on disk;
- asynchronous activation prefetch or writeback;
- strict total physical-memory-pressure enforcement;
- direct-I/O or NVMe-specific performance behavior;
- intra-layer tiling;
- bounded MLX backward or optimizer execution;
- representative tokenizer or production corpus quality;
- training state larger than safe resident unified memory;
- 124M or 350M full-parameter training on the target Mac.

No complete out-of-core, production-model-quality, throughput-at-scale, or larger-than-memory claim is made yet.

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
