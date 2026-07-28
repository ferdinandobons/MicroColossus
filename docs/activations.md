# Activation Policies and Recomputation

This document defines MicroColossus M6B, the activation-memory layer that sits beside the existing bounded parameter, gradient, optimizer, checkpoint, and pruning runtimes.

## 1. Motivation

MicroColossus already limits managed parameters, gradients, and Adam state by execution group. Before M6B, bounded backward retained every forward boundary activation on CPU until the matching reverse group executed.

For deeper models, larger microbatches, or longer sequences, those boundaries can become the dominant unified-memory consumer even when every parameter and optimizer group fits its declared limit.

M6B therefore introduces an explicit activation policy. The first optimized policy is synchronous prefix recomputation. It trades extra reads and compute for a zero-forward-boundary-retention contract.

## 2. Policies

`training.activation_policy` accepts:

```text
retain_all
recompute
```

### `retain_all`

The established bounded-backward path stores each non-final forward boundary on CPU. Reverse groups consume those saved inputs directly.

Advantages:

- minimum recomputation;
- lower parameter-read amplification;
- established numerical baseline.

Cost:

- retained activation bytes grow with depth, microbatch, sequence length, and hidden width.

### `recompute`

The forward retains no boundary for later backward use. During reverse execution, each group reconstructs its input by replaying the deterministic prefix from token IDs.

```text
final-head backward
    replay embedding and every block
    reconstruct final-head input
    execute local backward

block K backward
    replay embedding through block K-1
    reconstruct block-K input
    execute local backward

embedding backward
    use token IDs directly
```

The first algorithm deliberately favors correctness and observability over throughput. Prefix replay is quadratic in group count. Later hybrid policies can retain selected anchors to reduce replay depth.

## 3. Persistent training

MicroColossus 0.12 routes the persistent multi-step trainer through the configured activation policy:

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

The recomputed gradient store then flows through the same clipping, group-bounded AdamW, candidate-state verification, atomic bundle publication, progress records, checkpoint lineage, and pruning contracts used by `retain_all`.

The policy is part of checkpoint identity. A root created with `retain_all` cannot be resumed as `recompute`, or vice versa. Existing pre-0.12 `retain_all` roots keep their previous semantic digest and runtime identifier for backward-compatible resume.

## 4. Budget model

Five logical budgets are enforced independently:

- parameter working set;
- gradient working set;
- optimizer working set;
- retained activation working set;
- local forward or backward workspace.

The activation budget covers CPU activations or activation gradients retained across group boundaries. Under `recompute`, the forward-boundary count and bytes are zero, but one reconstructed input and one adjacent activation gradient can coexist during reverse execution.

The workspace budget covers the logical input, output, incoming activation gradient, and outgoing activation gradient used during one local replay or backward operation.

These are logical guardrails. On Apple Silicon they overlap within one unified-memory pool and must not be added to RSS, MPS current allocation, or Metal-driver counters.

## 5. Checkpoint and resume semantics

The authoritative root bundle still advances only after:

1. the configured backward policy produces a complete verified gradient store;
2. global norm and clipping are calculated;
3. AdamW updates every unique parameter exactly once;
4. candidate parameter and optimizer stores verify;
5. candidate state restores exactly;
6. the root bundle is atomically published.

Process restart uses the activation policy stored in the training metadata digest. Batch cursor, seed, data checksum, parent lineage, parameters, Adam moments, and optimizer steps continue from the previous committed bundle.

## 6. Pruning compatibility

Pruning remains policy-neutral at the storage graph level. It retains or removes root-referenced parameter, optimizer, and gradient stores according to the explicit retention policy.

A recompute root can be pruned and then resumed because the next backward reconstructs its boundaries from:

- token IDs from the deterministic data cursor;
- the current authoritative parameter store;
- the configured activation policy and budgets.

No historical activation checkpoint is required.

## 7. Numerical oracle

For the current validation runtime, the same authoritative parameter state and deterministic batch are also executed through a resident PyTorch oracle. Oracle gradients and the resident post-AdamW state are stored separately for validation.

The recomputation path records and compares:

- loss;
- global gradient norm;
- tensor-level gradient differences;
- final parameter and Adam state;
- tied-token-embedding accumulation and unique update counts;
- candidate-versus-restored exactness;
- parameter-manifest immutability.

Full oracle and candidate materialization are validation-only behavior and are declared in result metadata. They are not part of the eventual larger-than-memory claim.

## 8. Telemetry

Every persistent step records:

- activation policy;
- activation and workspace budgets;
- maximum retained activation bytes;
- maximum local workspace bytes;
- retained forward-boundary count and bytes;
- total prefix groups replayed;
- prefix recomputation time;
- per-group parameter, gradient, and optimizer traffic;
- loss, gradient norm, and clipping coefficient;
- RSS, accelerator, and Metal-driver samples;
- candidate and bundle verification.

The detailed recomputation JSON also records replayed group names, prefix parameter tensor and chunk reads, prefix logical bytes, local forward and backward time, and gradient-store traffic.

## 9. Accepted Apple M2 evidence

The native Apple M2 validation tested commit:

```text
4742f8a7f57a46edb075159275fb66c83c78ced7
```

Package version: `0.12.0`.

Overall result: **PASS**.

The quality gate passed Ruff, mypy, pytest, compileall, doctor, help, and static plan commands. All runtime scenarios passed with MPS built and available and `PYTORCH_ENABLE_MPS_FALLBACK` unset.

### 9.1 Numerical comparisons

All principal policy comparisons were classified `NUMERICALLY_STABLE`:

| Comparison | Maximum absolute state difference | Mean absolute state difference | Maximum loss difference | Maximum gradient-norm difference |
|---|---:|---:|---:|---:|
| Micro round 1, retain versus recompute | `1.862645149230957e-08` | `2.6070487856787704e-10` | `0.0` | `5.061370145220678e-08` |
| Micro round 2, retain versus recompute | `2.2351741790771484e-08` | `2.6154205006840354e-10` | `0.0` | `5.061370145220678e-08` |
| Resume recompute versus uninterrupted | `1.6763806343078613e-08` | `1.9436587520528433e-10` | `0.0` | `5.74115421869692e-09` |
| Tiny retain versus recompute | `2.3283064365386963e-10` | `8.079835528479944e-16` | `0.0` | `0.0` |
| Small retain versus recompute | `4.656612873077393e-10` | `4.673514187992496e-15` | `0.0` | `9.893160068941143e-08` |
| Pruned resume versus unpruned | `2.2351741790771484e-08` | `2.1037602991500962e-10` | `0.0` | `6.147735875927651e-08` |

Both injected publication-failure retry paths remained numerically stable. Their maximum state difference was `5.960464477539063e-08`, and the maximum observed loss difference was `4.76837158203125e-07`.

Batch provenance, cursor sequence, checksums, and generated sample sequences matched in the compared trajectories. Candidate restore remained exact.

### 9.2 Replay and logical activation memory

Expected replay totals were observed:

```text
real-text micro: 15 groups over five optimizer steps
tiny:            18 groups over three optimizer steps
real-text small: 15 groups over one optimizer step
```

For the 1,846,656-parameter small workload:

```text
retain_all forward-boundary bytes: 491,520
recompute forward-boundary bytes:        0

retain_all maximum retained activation bytes: 491,520
recompute maximum retained activation bytes:  196,608
maximum local workspace for both policies:    393,216
```

The small recompute path read `19,200,000` logical parameter bytes during prefix replay and recorded approximately `0.102379` seconds of prefix recomputation.

This is accepted evidence for a material logical activation reduction and a zero-forward-boundary contract.

### 9.3 Physical-memory observation

The small physical-memory classification was `HIGHER`:

```text
retain_all sampled peak RSS: 444,071,936 bytes
recompute sampled peak RSS:  533,528,576 bytes
ratio:                        1.2014462809917354x
```

This does not invalidate the logical activation contract. It demonstrates that the synchronous full-prefix policy is not yet a physical-memory optimization for the measured small workload. RSS, MPS allocator counters, Metal-driver allocation, filesystem cache, compressed memory, and swap overlap or describe different scopes and must not be added as independent memory pools.

### 9.4 Durability and safety

The target gate also passed:

- process restart and recompute resume;
- activation-policy mismatch rejection;
- activation and workspace budget rejection;
- pruning followed by recompute resume;
- preservation of the previous root at both publication failure points;
- tied-gradient accumulation count `2`;
- unique tied-parameter AdamW update count `1`;
- fallback, unsupported-operator, and non-finite scans;
- clean final Git state.

## 10. Current claim boundary

MicroColossus 0.12 establishes persistent multi-step training with either full boundary retention or synchronous zero-boundary prefix recomputation on Apple M2, including process resume, pruning compatibility, budget rejection, and atomic failure recovery.

It does not yet establish:

- hybrid activation anchors;
- activation tensors stored on disk;
- asynchronous activation prefetch or writeback;
- optimal replay scheduling;
- lower physical RSS for every workload;
- strict total physical-memory-pressure enforcement;
- direct-I/O or NVMe-specific behavior;
- bounded MLX backward or optimizer execution;
- intra-layer activation tiling;
- training state larger than unified memory.

The next milestone is a measured hybrid anchor planner. It must use the accepted replay, logical-memory, and physical-memory observations to search for a Pareto improvement instead of choosing an arbitrary fixed checkpoint interval.