# Persistent Multi-Step Bounded Training

This document records the MicroColossus 0.9 design and accepted Apple M2 evidence for consecutive bounded optimizer steps, persistent checkpoint state, process restart, and deterministic resume.

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

The accepted implementation remains:

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
13. restores and re-exports the candidate state for validation;
14. compares the candidate against a resident oracle for that step;
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

Candidate gradient, parameter, and optimizer stores can become internally valid before root publication. They do not become the current checkpoint until the new root pointer is atomically replaced.

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

Each step explicitly reports that complete candidate and resident-oracle states are materialized only after bounded execution for validation. The operational forward, backward, gradient-norm, and optimizer phases remain group-bounded.

The final multi-step result also explicitly reports:

- full final-state materialization for validation;
- resident reference replay from step zero;
- cursor derivation from committed step;
- retention of historical bundles.

## 12. Resident validation oracle

During development and validation, MicroColossus performs two reference checks.

### 12.1 Per-step oracle

The current parameter and Adam stores are restored into a resident model and optimizer. The same batch and canonical clipping coefficient are used for one update. The candidate bounded state is compared with this result.

### 12.2 Final replay oracle

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

## 15. Accepted Apple M2 validation

Tested runtime commit:

```text
4b1ffb20857dd948d7737484e62b007f24bf69b9
```

Package version:

```text
0.9.0
```

Overall result: **PASS**.

### 15.1 Environment and quality

- MacBook Air `Mac14,2`;
- Apple M2;
- 8 GB unified memory;
- native arm64 without Rosetta;
- Python 3.13.12;
- PyTorch 2.13.0;
- MPS built and available;
- fallback unset;
- Ruff passed;
- mypy passed over 33 source files;
- pytest passed, 80 passed and 1 skipped;
- compileall passed;
- final source tree was clean.

### 15.2 Successful trajectories

```text
micro uninterrupted: step 0 -> 5, GREEN
micro phase 1:       step 0 -> 2, GREEN
micro resumed:       step 2 -> 5, GREEN, resumed=true
tiny uninterrupted:  step 0 -> 3, GREEN
```

Lineage:

```text
micro uninterrupted: [0, 1, 2, 3, 4, 5]
micro resumed:       [0, 1, 2, 3, 4, 5]
tiny:                [0, 1, 2, 3]
```

Final optimizer step values:

```text
micro: all 5.0
tiny:  all 3.0
```

### 15.3 Process-restart equivalence

Uninterrupted micro step 0 to 5 and resumed micro step 0 to 2 plus step 2 to 5 were:

```text
BITWISE_EXACT
```

The following were equal:

- canonical final-state bytes;
- parameter and optimizer tensor names and structures;
- root lineage;
- batch cursor sequence;
- batch seed sequence;
- batch checksum sequence;
- loss trajectory;
- gradient-norm trajectory;
- clipping trajectory;
- final optimizer step values.

### 15.4 Numerical results

- maximum per-step loss difference: `0.0`;
- maximum per-step gradient-norm difference: `1.6985336648289717e-07`;
- maximum final bounded-versus-resident absolute difference: `7.450580596923828e-09`;
- mean final bounded-versus-resident absolute difference: `1.0105799101017887e-10`;
- candidate-versus-restored exactness: true;
- all values finite.

Tiny final-state comparison:

- maximum absolute difference: `2.3283064365386963e-10`;
- mean absolute difference: `3.718845572899834e-16`.

### 15.5 Working sets

| Configuration | Maximum parameter group | Maximum gradient group | Maximum optimizer group |
|---|---:|---:|---:|
| Micro | 33,280 bytes | 33,280 bytes | 133,152 bytes |
| Tiny | 788,480 bytes | 788,480 bytes | 3,153,952 bytes |

Budgets:

```text
parameter: 1,048,576 bytes
gradient:  1,048,576 bytes
optimizer: 4,194,304 bytes
```

All successful steps respected the three budgets.

### 15.6 Later-step failure recovery

The failure harness first committed step 2, then interrupted the attempted publication of step 3.

Before root manifest rename:

```text
PASS_PREVIOUS_STEP_2_PRESERVED
```

Before root `CURRENT` rename:

```text
PASS_PREVIOUS_STEP_2_PRESERVED
```

The second case left an unpublished step-3 manifest, which recovery reported without making authoritative.

### 15.7 Rejection and integrity results

- configuration mismatch: PASS_CONFIGURATION_MISMATCH_REJECTED;
- corrupt referenced child: PASS_CORRUPT_CHILD_DETECTED;
- parameter budget: PASS_PARAMETER_BUDGET_REJECTION;
- gradient budget: PASS_GRADIENT_BUDGET_REJECTION;
- optimizer budget: PASS_OPTIMIZER_BUDGET_REJECTION;
- root and referenced child verification and recovery: PASS.

A changed learning rate was rejected through the semantic configuration digest. A deliberately shortened referenced parameter chunk was detected before another step was published.

### 15.8 Storage and publication telemetry

Across successful validation roots:

- total directory size: `59,901,060` bytes;
- parameter bytes read and written: `5,782,016` and `5,782,016`;
- gradient bytes read: `5,782,016`;
- optimizer bytes read and written: `11,564,752` and `11,572,248`;
- reused chunks: `167`;
- publication `fsync` total: `0.0074204159` seconds;
- rename and publication total: `0.0025823780` seconds.

Historical bundles remain retained. No pruning or compaction behavior was validated.

### 15.9 Memory and fallback

- maximum RSS: `410,943,488` bytes;
- sampled maximum MPS current allocation: `0` bytes;
- maximum Metal driver allocation: `25,935,872` bytes;
- swap delta: `0` bytes;
- hidden fallback evidence: none;
- unsupported operator evidence: none;
- unexpected failed commands: none.

The zero sampled MPS allocation is recorded as reported. MPS availability, successful MPS computation, and Metal driver activity were independently observed.

## 16. Accepted conclusion

M4C is complete for the tested PyTorch MPS reference path.

The accepted result demonstrates that MicroColossus can:

1. initialize state once;
2. advance several bounded optimizer steps from prior committed state;
3. stop the process after a committed checkpoint;
4. reopen the same root in a new process;
5. continue the deterministic trajectory;
6. preserve Adam moments and optimizer step tensors;
7. match uninterrupted execution bitwise in the tested micro path;
8. preserve the previous committed step after an interrupted later publication;
9. reject incompatible or corrupt state before continuing.

## 17. Current boundary

Version 0.9 does not establish:

- real-corpus training;
- persisted tokenizer or real dataset-shard state;
- activation recomputation from storage;
- activation offload;
- bounded activation and operator-workspace residency;
- asynchronous prefetch or writeback;
- intra-layer tiling;
- bounded MLX backward or optimizer execution;
- storage pruning or compaction;
- training state larger than unified memory;
- 124M or 350M full-parameter training on the target Mac.

The next functional milestone is a small real-text training frontend with validation loss, checkpoint resume, and sample generation. Activation-memory management and larger-than-resident capacity demonstrations remain subsequent gates.
