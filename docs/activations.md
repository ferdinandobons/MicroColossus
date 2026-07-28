# Activation Recomputation Reference

This document defines the first MicroColossus M6B vertical slice for reducing activation residency without changing parameter, gradient, optimizer, checkpoint, or pruning semantics.

## 1. Motivation

MicroColossus already bounds managed parameters, gradients, and Adam state by execution group. The current bounded backward path still retains every forward boundary activation on CPU until reverse execution reaches the corresponding group.

For deeper models, larger microbatches, or longer sequences, those retained activations can become the dominant unified-memory consumer even when all parameter and optimizer groups fit their declared limits.

The first M6B implementation therefore establishes a synchronous recomputation reference before activation storage, asynchronous overlap, or intra-layer tiling are introduced.

## 2. Current vertical slice

The diagnostic entry point is:

```bash
python -m microcolossus.activation_recompute_cli \
  --config examples/micro-storage.yaml \
  --parameter-store runs/recompute-parameters \
  --oracle-gradient-store runs/recompute-oracle-gradients \
  --gradient-store runs/recompute-gradients \
  --output runs/recompute-result.json \
  --device cpu \
  --parameter-working-set-mib 1 \
  --gradient-working-set-mib 1 \
  --activation-working-set-mib 1 \
  --workspace-working-set-mib 4
```

The path is a validation runtime. It is not yet the default persistent multi-step trainer.

## 3. Execution model

The reference forward executes one parameter group at a time:

```text
embedding -> block 0 -> ... -> block N -> final head
```

Only the current hidden state remains live long enough to feed the next group. No forward boundary activation is retained for later backward use.

During reverse execution, each group reconstructs its input by replaying the deterministic prefix from the token IDs:

```text
final head backward
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

This first algorithm deliberately favors clarity and numerical isolation over throughput. Prefix replay can be quadratic in the number of groups. Later hybrid policies can retain selected anchors to trade activation memory for recomputation time.

## 4. Budget model

Four logical budgets remain separate:

- parameter working set;
- gradient working set;
- retained activation working set;
- local recomputation workspace.

The activation budget covers the reconstructed CPU input and the activation gradient retained between adjacent reverse groups.

The workspace budget covers the local logical input, output, incoming activation gradient, and outgoing activation gradient used during one replay or local backward operation.

These are logical guardrails. On Apple Silicon they overlap within one physical unified-memory pool and must not be summed with RSS, MPS allocation, or Metal-driver counters.

## 5. Numerical oracle

The same initial parameter state and deterministic batch are executed through a resident PyTorch reference. Resident gradients are committed to a separate versioned oracle store.

The recomputation path then commits its final gradients group by group. Validation compares the complete canonical gradient states and records:

- loss difference;
- global gradient-norm difference;
- tensor-level absolute and relative differences;
- tied-token-embedding accumulation count and version;
- parameter-manifest immutability;
- exact store verification.

The token embedding remains shared between the initial embedding and final projection. Both gradient contributions are accumulated, while later AdamW execution must still update the parameter once.

## 6. Telemetry

The result records:

- zero retained forward boundary count and bytes;
- maximum retained recomputed activation or activation-gradient bytes;
- maximum local workspace bytes;
- forward group reads and timings;
- every reverse group's prefix replay;
- replayed group names;
- replay parameter tensor, chunk, and logical-byte reads;
- prefix recomputation time;
- local forward and backward time;
- gradient commit traffic;
- process RSS;
- accelerator and driver counters;
- full resident-versus-recomputed gradient comparison.

## 7. Current claim boundary

This vertical slice establishes only a deterministic synchronous recomputation reference for one bounded backward validation.

It does not yet establish:

- persistent multi-step training with recomputation enabled;
- checkpoint resume under a recomputation policy;
- pruning followed by recomputation resume;
- hybrid anchor retention;
- activation tensors stored on disk;
- asynchronous activation prefetch or writeback;
- reduced physical RSS on every workload;
- direct-I/O or NVMe-specific behavior;
- bounded MLX backward or optimizer execution;
- training state larger than unified memory.

The next integration step is to route persistent bounded training through the same activation policy and budgets, then compare retain-all and recompute trajectories on CPU and Apple M2.
