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

## 9. Current claim boundary

MicroColossus 0.12 establishes persistent multi-step training with either full boundary retention or synchronous zero-boundary prefix recomputation, including process resume and pruning compatibility.

It does not yet establish:

- hybrid activation anchors;
- activation tensors stored on disk;
- asynchronous activation prefetch or writeback;
- optimal replay scheduling;
- reduced physical RSS on every workload;
- direct-I/O or NVMe-specific behavior;
- bounded MLX backward or optimizer execution;
- intra-layer activation tiling;
- training state larger than unified memory.

The next optimization should compare `retain_all` and `recompute` on Apple M2, then use the measured replay and memory curves to choose hybrid anchor intervals rather than guessing them.
