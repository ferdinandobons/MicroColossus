# Persistent Multi-Step Bounded Training

This document records the MicroColossus 0.9 design for consecutive bounded optimizer steps, persistent checkpoint state, process restart, and deterministic resume.

## 1. Scope

Version 0.8 validated one complete bounded optimizer step:

```text
bounded forward
    -> bounded backward
    -> versioned gradients
    -> global clipping
    -> group-bounded AdamW
    -> candidate parameter and optimizer stores
    -> atomic root bundle publication
```

Version 0.9 turns that isolated update into a persistent sequence:

```text
bundle step 0
    -> step 1
    -> step 2
    -> ...
    -> step N
```

Every new step consumes the parameter and optimizer stores referenced by the current authoritative root bundle. State is not recreated from initialization after step 0.

The first implementation remains:

- PyTorch reference backend;
- FP32 full-parameter AdamW;
- deterministic synthetic batches;
- CPU-retained boundary activations;
- synchronous tensor-store operations;
- group-bounded parameter, gradient, and optimizer execution;
- full-state materialization only for explicit validation comparisons.

## 2. Persistent root layout

```text
training-root/
  TRAINING.json
  CURRENT
  manifests/
    bundle-0.json
    bundle-1.json
    ...
  work/
    step-<N>-<attempt-id>/
      oracle-gradients/
      bounded-gradients/
      oracle-state/
      bounded-backward.json
  candidates/
    step-0-parameters/
    step-0-optimizer/
    step-<N>-<attempt-id>/
      parameters/
      optimizer/
```

Historical bundles and child stores are retained in 0.9. Automatic pruning and compaction are intentionally separate milestones.

## 3. Training metadata

`TRAINING.json` is written during initialization and contains a checksum over:

```text
schema version
semantic configuration digest
runtime version
training seed
batch-stream version
schedule kind
```

The semantic configuration digest includes:

- model architecture;
- microbatch;
- sequence length;
- learning rate;
- weight decay;
- gradient clipping;
- seed;
- full-parameter training mode;
- batch-stream policy;
- schedule policy.

Output paths, target step, and execution device are not part of the semantic digest. A resumed run may request a later target step or choose a compatible execution device, but it may not silently change model or optimizer semantics.

Resume rejects a mismatch before another step is executed.

## 4. Deterministic batch cursor

The first multi-step runtime uses a random-access synthetic stream:

```text
batch seed = training seed + 1 + cursor
cursor = current committed root step
```

Therefore:

```text
step 0 -> 1 consumes cursor 0
step 1 -> 2 consumes cursor 1
step 2 -> 3 consumes cursor 2
```

The new root bundle records the checksum of the batch consumed by that step. The committed root step is the authoritative next-batch cursor.

A future real-corpus frontend will replace the synthetic cursor with persisted dataset shard, sample, epoch, shuffle, and tokenizer state while preserving the same transactional rule.

## 5. Initialization

When the root does not exist, the runtime:

1. seeds the controlled model deterministically;
2. writes checksummed `TRAINING.json`;
3. exports canonical step-0 parameters;
4. creates zero-initialized AdamW first moments, second moments, and step tensors;
5. verifies both child stores;
6. publishes root bundle step 0;
7. verifies the root bundle.

The step-0 bundle has no parent and no gradient-store reference.

## 6. Resume

When the root already exists, the runtime:

1. opens `StepBundleStore`;
2. validates `TRAINING.json` and its checksum;
3. compares training metadata with the requested configuration;
4. verifies the current root bundle and referenced child stores;
5. verifies contiguous parent lineage;
6. reads the current committed step;
7. continues only when the requested target is not behind the current step.

The process can exit after any committed step. A new Python process can reopen the same directory and continue from `CURRENT`.

## 7. Step N to N+1

For each new step the runtime:

1. loads the current root bundle;
2. opens the referenced parameter and optimizer stores;
3. generates the deterministic batch for cursor N;
4. executes bounded forward from current parameters;
5. executes bounded backward in reverse group order;
6. publishes final gradients into a new versioned gradient store;
7. streams the gradient store to calculate the global norm;
8. calculates the canonical clipping coefficient;
9. reads each unique parameter group with its gradient and current Adam state;
10. executes group-local AdamW;
11. writes version N+1 parameter and optimizer tensors into fresh candidate stores;
12. verifies candidate stores;
13. restores and re-exports the candidate state for exact validation;
14. compares the candidate against an independently restored resident oracle for that step;
15. publishes root bundle N+1 only after validation;
16. verifies the new root and child references.

The source bundle remains authoritative until the final root `CURRENT` replacement.

## 8. Tied token embedding

The tied token embedding has three distinct rules:

```text
forward:
read for embedding and read again for final projection

backward:
accumulate final-head and embedding contributions

optimizer:
deduplicate the logical parameter and update it exactly once
```

The optimizer step tensor, first moment, and second moment advance once per committed training step.

## 9. Root lineage

Every root bundle contains:

- bundle ID;
- parent bundle ID;
- committed training step;
- parameter-store reference;
- optimizer-store reference;
- optional gradient-store reference;
- consumed batch checksum;
- bundle checksum.

The accepted lineage must be contiguous:

```text
step 0 parent = null
step 1 parent = step 0 bundle
step 2 parent = step 1 bundle
...
```

A missing parent, cycle, skipped step, or checksum mismatch is an integrity failure.

Child tensor-store transaction steps are internal publication sequence numbers. The root bundle committed step is the authoritative training-step number. Tensor versions in candidate parameter and optimizer stores equal the root training step being published.

## 10. Atomicity and later-step failure

Candidate gradient, parameter, and optimizer stores can become internally valid before the root publication. They do not become the current checkpoint until the new root pointer is atomically replaced.

For an interruption while attempting step N+1:

```text
CURRENT before failure = bundle N
CURRENT after recovery = bundle N
```

Validated failure points include:

- before root bundle-manifest rename;
- before root `CURRENT` rename.

Unpublished manifests and temporary files are reported by recovery and are not silently treated as committed training state.

## 11. Correctness model

CPU validation requires exact canonical equality for:

- parameters;
- Adam first moments;
- Adam second moments;
- optimizer step tensors;
- parameter-group metadata;
- candidate restore;
- uninterrupted versus resumed final state;
- bounded versus resident final state.

MPS validation reports raw numerical differences and keeps bitwise and numerical reproducibility separate.

Each step explicitly reports that the complete candidate and resident-oracle states are materialized only after bounded execution for validation. The operational forward, backward, gradient-norm, and optimizer phases remain group-bounded.

The final multi-step result also explicitly reports:

- full final-state materialization for validation;
- resident reference replay from step zero;
- cursor derivation from committed step;
- retention of historical bundles.

## 12. Resident validation oracle

During development and validation, MicroColossus performs two reference checks.

### Per-step oracle

The current parameter and Adam stores are restored into a resident model and optimizer. The same batch and canonical clipping coefficient are used for one update. The candidate bounded state is compared with this result.

### Final replay oracle

After reaching the target step, a resident model is initialized from step zero and replays every deterministic batch through the target step. The final bounded checkpoint is compared with this uninterrupted resident result.

Final replay is validation-only. It is not part of the intended production runtime and its full-state memory usage is excluded from bounded claims.

## 13. Working-set budgets

All existing budgets remain enforced at every step:

- parameter working-set budget;
- gradient working-set budget;
- optimizer working-set budget;
- tensor-store staging and capacity budgets.

The optimizer budget includes:

```text
parameter
+ final gradient
+ Adam first moment
+ Adam second moment
+ optimizer step tensor
```

A rejected step never replaces the current authoritative bundle.

## 14. Result schema and telemetry

A multi-step result records:

- requested, starting, and final steps;
- whether the invocation resumed an existing root;
- training metadata and digest;
- root lineage;
- cursor, batch seed, and checksum for each executed step;
- per-step loss, global norm, and clipping coefficient;
- parameter, gradient, and optimizer working sets;
- optimizer group order;
- tied-weight update count;
- candidate tensor versions;
- state comparisons;
- publication and verification results;
- logical and physical application bytes read or written;
- per-group timings and memory counters;
- final optimizer step values.

Application-level write counters are not interpreted as physical NAND writes.

## 15. Initial validation gates

CPU CI requires:

- exact 3-step micro training;
- exact 5-step uninterrupted training;
- exact 2-step then resumed-to-5 training;
- uninterrupted and resumed final-state equality;
- exact final resident replay equality;
- correct optimizer step values;
- contiguous bundle lineage;
- configuration mismatch rejection;
- later-step publication failure preserving the previous step;
- working-set rejection preserving step 0;
- separate Python 3.11 and 3.13 CLI resume smoke runs.

The target Apple Silicon gate will run:

- micro 5-step uninterrupted;
- micro 2-step process, exit, then resume to step 5;
- tiny 3-step uninterrupted;
- later-step failure injection;
- configuration mismatch rejection;
- root and child-store verification and recovery;
- memory, swap, storage-growth, and cumulative-write telemetry;
- fallback-disabled source-tree-clean validation.

## 16. Current boundary

Version 0.9 does not yet establish:

- real-corpus training;
- persisted tokenizer or dataset-shard state;
- activation recomputation from storage;
- activation offload;
- asynchronous prefetch or writeback;
- intra-layer tiling;
- bounded MLX backward or optimizer execution;
- storage pruning or compaction;
- training state larger than unified memory;
- 124M or 350M full-parameter training on the target Mac.

The next functional milestone after target validation is a small real-text training frontend with validation loss, checkpoint, resume, and sample generation. Activation-memory management and larger-than-resident capacity demonstrations remain subsequent gates.
