# MicroColossus Validation Ledger

This ledger records accepted executable evidence, corrected failures, exact tested commits, engineering conclusions, and the boundary of every result.

## 1. Validation policy

Every accepted target run must identify:

- repository and exact commit;
- package version;
- machine and native architecture;
- framework and dependency versions;
- commands, exit codes, stdout, stderr, and elapsed time;
- numerical results and raw tolerances;
- memory, swap, storage, and publication telemetry;
- generated artifacts;
- source-tree state before and after execution;
- capabilities that were not exercised.

Mandatory distinctions:

- a static plan is not a training result;
- resident training is not storage-backed training;
- storage-backed state is not necessarily bounded compute;
- one bounded step is not a multi-step training run;
- a multi-step synthetic run is not real-corpus training;
- a CPU result is not evidence for MPS or MLX;
- a functional pass with tracked local changes is not a clean protocol pass;
- a protocol expectation can be wrong even when the checked runtime is correct;
- checksum equality is stronger than numerical agreement;
- framework memory counters are not equivalent physical-memory counters;
- application storage bytes are not NAND-level writes;
- full-state materialization used only for validation must be declared;
- full-parameter, compact, quantized, and adapter methods are reported separately;
- a larger parameter count is not success when correctness, recovery, throughput, or endurance are unacceptable.

## 2. Accepted evidence summary

| Area | Version or commit | Target result | Main conclusion |
|---|---|---|---|
| Resident MPS foundation | `a56fc514f2f8e705654034f3c2f02e3a441c61f3` | PASS on 8 GB M2 | Native MPS forward, backward, AdamW, telemetry, and fixed-batch learning work |
| Competitive PyTorch and MLX | `785183a1ff87df0c22df9619d1ab7bf53968bc79`, finalized by 0.3.3 | PASS runtime and clean release verification | MLX was 1.592x faster in the tested resident workload. Dual backend selected |
| Versioned tensor store and lifecycle | `82e53c671848d231c2361443882b97dbe4e3a408`, 0.5.0 | PASS on 8 GB M2 | Canonical state, recovery, failure injection, PyTorch and MLX round trips work |
| Bounded forward | `1feea9f9eef28e551ad4ae4944614083effa804f`, 0.6.0 | PASS on 8 GB M2 | One parameter group at a time matched resident boundaries, logits, and loss |
| Bounded backward | `c72dcc2f8d8a7bd783ae263cf14476d0681b664b`, 0.7.0 | PASS on 8 GB M2 | Reverse group execution produced exact tested gradients and valid gradient stores |
| Bounded AdamW and root bundles | `ef88198d66f1d1795ffa14dcb6db388ae1715e85`, 0.8.0 | PASS on 8 GB M2 | One complete group-bounded optimizer step and atomic root publication work on MPS |
| Persistent multi-step and resume | `4b1ffb20857dd948d7737484e62b007f24bf69b9`, 0.9.0 | PASS on 8 GB M2 | Consecutive bounded steps, process restart, exact resume, lineage, and later-step atomicity work on MPS |
| Deterministic real-text training | `8bc277123267c3d3f15bf60cd640819fa823d2e3`, 0.10.0 | PASS on 8 GB M2 after protocol correction | Real-text micro and 1.85M trajectories, validation, samples, provenance, mutation rejection, and resume work on MPS |

## 3. Resident Apple M2 foundation

The first executable diagnostic exposed:

- Ruff import formatting problems;
- an unintended NumPy checksum dependency;
- static typing problems;
- a float64 gradient-norm path incompatible with MPS.

The corrected clean run tested:

```text
a56fc514f2f8e705654034f3c2f02e3a441c61f3
```

Result: **PASS**.

Validated:

- MacBook Air M2 with 8 GB unified memory;
- native arm64 execution without Rosetta;
- PyTorch MPS built and available;
- explicit MPS training;
- automatic MPS selection;
- CPU-versus-MPS numerical comparison;
- fixed-batch learning;
- synchronized memory and timing telemetry;
- clean source tree.

Observed fixed-batch learning:

```text
5.566842079162598 -> 0.4145740866661072
```

No fallback, unsupported operator, non-finite value, or MPS out-of-memory failure was detected in the tested path.

MPS bitwise and numerical reproducibility remain separate properties.

## 4. Competitive PyTorch MPS and MLX validation

Accepted runtime commit:

```text
785183a1ff87df0c22df9619d1ab7bf53968bc79
```

Environment:

- MacBook Air with Apple M2;
- 8 GB unified memory;
- native arm64 without Rosetta;
- PyTorch 2.13.0 with MPS;
- MLX 0.32.0;
- NumPy 2.4.6;
- MPS-to-CPU fallback disabled.

The run completed all tiny smoke paths and nine 23,213,056-parameter competitive runs with identical initial portable-state and batch checksums.

Performance:

| Variant | Median tokens/s | Relative to PyTorch |
|---|---:|---:|
| PyTorch MPS | 1,379.23 | 1.000x |
| PyTorch MPS checkpointed | 1,337.37 | 0.970x |
| MLX | 2,195.69 | 1.592x |

PyTorch versus checkpointed PyTorch was GREEN with zero final-state difference.

PyTorch versus MLX was YELLOW:

- maximum loss difference: `1.9073486328125e-06`;
- maximum final-parameter absolute difference: `3.8036610931158066e-05`;
- mean final-parameter absolute difference: `7.378751249633382e-09`;
- all values finite;
- result stable across all three rounds.

Decision: **DUAL BACKEND**.

- MLX is the preferred optimized Apple Silicon execution candidate.
- PyTorch MPS remains the numerical oracle and recovery or debugging reference.
- Storage, transaction, tensor identity, and execution-plan formats remain backend-neutral.

## 5. Versioned tensor store and observable optimizer lifecycle

MicroColossus 0.4 introduced canonical tensor, chunk, manifest, journal, and telemetry schemas, content-addressed immutable chunks, copy-on-write versions, atomic `CURRENT`, corruption detection, conservative recovery, failure injection, and PyTorch and MLX adapters.

MicroColossus 0.5 added one fully observed storage-backed optimizer lifecycle.

Accepted commit:

```text
82e53c671848d231c2361443882b97dbe4e3a408
```

Overall result: **PASS**.

Key results:

- two GREEN micro runs;
- bitwise-exact micro repeatability;
- one GREEN 443,648-parameter tiny run;
- zero resident-versus-storage loss difference;
- zero gradient-norm difference;
- zero final-state absolute difference;
- exact storage-versus-restored state;
- all five tensor-store failure points preserved the prior manifest;
- MLX micro and tiny model and optimizer round trips passed;
- cross-backend canonical model state was GREEN.

Application-level totals:

- bytes read: `7,468,738`;
- referenced chunk reads: `226`;
- bytes written: `7,641,680`;
- chunk writes: `166`;
- reused chunks: `60`;
- reuse ratio: `26.55%`;
- `fsync` time: `0.010095799` seconds;
- publication time: `0.001221206` seconds.

Maximum RSS was `432,472,064` bytes. Maximum MPS allocation was `8,709,376` bytes. Maximum Metal driver allocation was `28,049,408` bytes. Swap delta was zero.

This result validated storage lifecycle and recovery. It did not validate bounded compute.

## 6. Bounded parameter-group forward

Version 0.6 executes and releases embeddings, one Transformer block at a time, and the final head. It reloads tied token embeddings for the output projection and rejects groups that exceed the parameter budget.

Accepted commit:

```text
1feea9f9eef28e551ad4ae4944614083effa804f
```

Overall result: **PASS**.

Key results:

- Ruff, mypy, 56 tests, and compileall passed;
- micro run 1 and run 2 were GREEN;
- micro repeatability was BITWISE_EXACT;
- tiny was GREEN;
- intentional parameter-budget rejection passed;
- tied token embedding reload count was correct;
- largest micro group: `33,280` bytes;
- largest tiny group: `788,480` bytes;
- configured budget: `1,048,576` bytes;
- maximum boundary, logits, and loss differences: `0.0`;
- parameter manifests remained unchanged;
- store verification and recovery passed.

Maximum RSS was `322,273,280` bytes. Maximum MPS allocation was `1,181,696` bytes. Maximum Metal driver allocation was `19,300,352` bytes. Swap delta was zero.

Hidden boundary activations remained resident. The bounded claim applied to managed parameter residency.

## 7. Bounded backward and versioned gradients

Version 0.7 retains detached boundary activations on CPU, processes groups in reverse, recomputes each local forward, propagates one upstream activation gradient, and commits final parameter gradients into a separate store.

Accepted commit:

```text
c72dcc2f8d8a7bd783ae263cf14476d0681b664b
```

Overall result: **PASS**.

Key results:

- Ruff and mypy passed;
- 62 tests passed with one skip;
- compileall passed;
- two micro runs were GREEN and BITWISE_EXACT;
- tiny was GREEN;
- reverse group order was correct;
- parameter manifest remained immutable;
- micro and tiny gradient tensor counts were 12 and 20;
- tied-gradient accumulation count was 2;
- tied-gradient final version was 1;
- maximum loss difference was `0.0`;
- maximum tensor-gradient difference was `0.0`;
- maximum global-norm difference was about `9.57e-08`;
- parameter and gradient budget rejection passed;
- parameter, oracle-gradient, and bounded-gradient stores verified and recovered;
- swap delta was zero;
- no fallback, unsupported operation, non-finite value, or unexpected command failure was detected.

This phase intentionally did not update parameters or AdamW state.

## 8. Bounded AdamW and atomic root step bundles

Version 0.8 consumes the final-gradient store, computes one canonical clipping coefficient, reads matching parameters and Adam state group by group, updates the tied token embedding once, writes candidate state, verifies it, and atomically publishes a root bundle.

Accepted runtime commit:

```text
ef88198d66f1d1795ffa14dcb6db388ae1715e85
```

Package version: `0.8.0`.

Overall result: **PASS**.

Successful runs:

- micro run 1: GREEN;
- micro run 2: GREEN;
- micro repeatability: BITWISE_EXACT;
- tiny: GREEN.

Working sets:

| Configuration | Maximum parameter group | Maximum gradient group | Maximum optimizer group |
|---|---:|---:|---:|
| Micro | 33,280 bytes | 33,280 bytes | 133,152 bytes |
| Tiny | 788,480 bytes | 788,480 bytes | 3,153,952 bytes |

Configured budgets were 1 MiB for parameters, 1 MiB for gradients, and 4 MiB for optimizer groups.

Correctness:

- tied parameter update count: `1`;
- initial bundle step: `0`;
- final bundle step: `1`;
- maximum loss difference: `0.0`;
- maximum gradient-norm difference: `9.567381065167524e-08`;
- resident-versus-candidate maximum and mean differences: `0.0`;
- candidate-versus-restored exact bytes: true;
- all candidate tensor versions: `1`.

Durability:

- optimizer working-set rejection passed;
- interruption before root bundle-manifest rename preserved step 0;
- interruption before root `CURRENT` rename preserved step 0;
- root and child-store verification and recovery passed.

Observed resources:

- maximum RSS: `393,052,160` bytes;
- sampled MPS current allocation: `0` bytes;
- maximum Metal driver allocation: `25,935,872` bytes;
- swap delta: `8,388,608` bytes.

The zero sampled MPS allocation is recorded as reported and is not treated as evidence that the MPS path did not execute.

## 9. Persistent multi-step training and process resume

Version 0.9 advances the authoritative root from step N to step N+1, preserving parameter state, Adam moments, optimizer step tensors, parent lineage, batch provenance, and configuration identity.

Accepted runtime commit:

```text
4b1ffb20857dd948d7737484e62b007f24bf69b9
```

Package version: `0.9.0`.

Overall result: **PASS**.

Environment and quality:

- MacBook Air `Mac14,2` with Apple M2 and 8 GB unified memory;
- native arm64 without Rosetta;
- Python 3.13.12;
- PyTorch 2.13.0;
- MPS built and available;
- `PYTORCH_ENABLE_MPS_FALLBACK` unset;
- Ruff, mypy, 80 tests with one skip, and compileall passed;
- source tree clean before and after execution.

Successful trajectories:

| Scenario | Start | Final | Resume | Classification |
|---|---:|---:|---|---|
| Micro uninterrupted | 0 | 5 | false | GREEN |
| Micro phase 1 | 0 | 2 | false | GREEN |
| Micro process restart | 2 | 5 | true | GREEN |
| Tiny uninterrupted | 0 | 3 | false | GREEN |

Uninterrupted step 0 to 5 and process-restarted step 0 to 2 plus step 2 to 5 produced:

```text
classification: BITWISE_EXACT
canonical bytes equal: true
maximum absolute difference: 0.0
mean absolute difference: 0.0
lineage equal: true
batch cursor equal: true
batch seed equal: true
batch checksum equal: true
loss trajectory equal: true
gradient-norm trajectory equal: true
clipping trajectory equal: true
```

Numerical comparison with resident replay:

- maximum per-step loss difference: `0.0`;
- maximum per-step gradient-norm difference: `1.6985336648289717e-07`;
- maximum final-state absolute difference: `7.450580596923828e-09`;
- mean final-state absolute difference: `1.0105799101017887e-10`;
- final candidate restore exactness: true;
- all values finite.

Later-step atomicity, configuration mismatch rejection, corrupt-child detection, and all three working-set rejections passed. Swap delta was zero.

Historical root bundles and child stores were retained. No automatic pruning or compaction was tested.

## 10. Deterministic real-text training

Version 0.10 extends the persistent bounded runtime with a local UTF-8 byte-tokenizer frontend, corpus identity, deterministic train and validation windows, validation loss, greedy samples, and atomic progress records.

Accepted runtime commit:

```text
8bc277123267c3d3f15bf60cd640819fa823d2e3
```

Package version: `0.10.0`.

### 10.1 Protocol correction

The external report classified the release as formal `FAIL` because its prompt expected:

```text
real-text micro parameter_count = 11,456
```

The checked commit correctly reported:

```text
real-text micro parameter_count = 18,624
```

The 18,624 count follows directly from the checked configuration:

```text
vocabulary size:              256
maximum positional length:     64
hidden size:                   32
Transformer blocks:             1
```

The older 11,456 count belongs to the synthetic micro configuration with vocabulary size 64 and a shorter positional table. The validation mismatch was therefore a stale protocol expectation, not a runtime or planner defect.

No training, integrity, numerical, fallback, or source-cleanliness gate failed because of this count. The target result is accepted as **PASS with a documented protocol correction**.

### 10.2 Environment and project quality

- MacBook Air with Apple M2;
- 8 GB unified memory;
- native arm64;
- Rosetta translation value `0`;
- Python 3.13.12;
- PyTorch 2.13.0;
- MPS built and available;
- `PYTORCH_ENABLE_MPS_FALLBACK` unset;
- exact commit and package version matched;
- Ruff: PASS;
- mypy: PASS;
- pytest: PASS, 88 passed and 1 skipped;
- compileall: PASS;
- doctor: PASS;
- bounded training CLI help: PASS;
- real-text micro and small plans parsed successfully;
- final `git status --short`: empty;
- final `git diff --check`: clean.

### 10.3 Data identity and tokenizer preflight

- independent-process data identity matched exactly;
- tokenizer version was `utf8-bytes-v1`;
- token range was valid;
- UTF-8 encode and decode round trip passed;
- configured train and validation split identity was stable.

### 10.4 Micro uninterrupted trajectory

Configuration:

```text
parameters:          18,624
Transformer blocks: 1
hidden size:         32
vocabulary:          256
sequence length:     32
microbatch:          2
```

Result:

```text
classification: GREEN
final step:      20
learning signal: LEARNING_SIGNAL_GREEN
```

Validation loss:

```text
step 0:  5.548418998718262
step 20: 3.302267074584961
```

Lineage and progress records were contiguous for steps 0 through 20. Bundle IDs matched the corresponding progress records.

### 10.5 Process restart and resume

A separate root executed:

```text
process 1: step 0 -> step 5
process exit
process 2: step 5 -> step 20
```

Both phases were GREEN. The second phase reported `resumed=true` and executed 15 new optimizer steps.

Uninterrupted versus resumed comparison:

```text
classification: NUMERICALLY_STABLE
names equal: true
structures equal: true
all values finite: true
maximum absolute difference: 1.1920928955078125e-07
mean absolute difference:    8.844825718731097e-10
```

The relative worst case occurred in a near-zero Adam first-moment value and is reported with the absolute metrics.

Equivalent properties:

- steps;
- batch cursors;
- seeds;
- byte offsets;
- batch checksums;
- training-loss trajectory within tolerance;
- validation-loss trajectory within tolerance;
- gradient norms within tolerance;
- clipping coefficients within tolerance;
- sample token IDs;
- decoded sample completion.

Candidate-versus-restored state was exact.

### 10.6 Corpus mutation rejection

The harness copied the corpus and configuration outside the repository, completed step 2, changed the copied corpus bytes, and attempted resume.

Result:

```text
PASS_CORPUS_MUTATION_REJECTED
exception: ResumeConfigurationError
mismatch: data_identity
current authoritative step: 2
step 3 published: false
```

The root verified and recovered after rejection.

### 10.7 Small real-text trajectory

Configuration:

```text
parameters:          1,846,656
Transformer blocks: 4
hidden size:         192
vocabulary:          256
sequence length:     128
microbatch:          1
```

Result:

```text
classification: GREEN
final step:      10
learning signal: LEARNING_SIGNAL_GREEN
```

Validation loss:

```text
step 0:  5.687370777130127
step 10: 4.083975553512573
```

Numerical comparison with resident replay:

- maximum loss difference: `0.0`;
- maximum gradient-norm difference: `1.639226772098823e-07`;
- maximum final-state absolute difference: `1.1920928955078125e-07`;
- mean final-state absolute difference: `1.2758380908789405e-10`;
- candidate-versus-restored exact bytes: true;
- all values finite.

Lineage and progress records were contiguous for steps 0 through 10.

### 10.8 Storage and resource observations

Micro root:

- directory size: `25,805,954` bytes;
- current parameter tensors: `12`;
- current gradient tensors: `12`;
- current optimizer tensors: `37`;
- maximum process RSS: `318,963,712` bytes;
- maximum recorded accelerator allocation: `20,152,320` bytes;
- validation time total: about `11.86` seconds.

Small root:

- directory size: `631,729,632` bytes;
- current parameter tensors: `36`;
- current gradient tensors: `36`;
- current optimizer tensors: `109`;
- maximum process RSS: `393,314,304` bytes;
- maximum recorded accelerator allocation: `62,586,880` bytes;
- validation time total: about `4.30` seconds.

The small scenario consumed substantial storage because every historical candidate and work store was retained. Free filesystem storage fell from about 7.20 GiB before installation to about 3.67 GiB after the small run.

This is evidence that pruning and compaction should precede much longer real-data trajectories.

Application byte counters are not NAND-level SSD-write measurements. APFS amplification, page cache, and controller behavior remain outside those counters.

### 10.9 Fallback and source integrity

The first scanner pass produced textual false positives because substrings such as `nan` appeared inside ordinary words and because the environment key `PYTORCH_ENABLE_MPS_FALLBACK` was present with a null value.

After audit:

- no actual runtime fallback evidence remained;
- no unsupported operator evidence remained;
- no non-finite tensor or loss evidence remained;
- no unexpected command failed;
- the MPS fallback environment variable remained unset;
- final source state was clean.

### 10.10 Accepted conclusion

M5 is complete for the tested PyTorch MPS reference path.

The accepted evidence establishes:

```text
local UTF-8 corpus
    -> checksummed data identity
    -> deterministic windows and offsets
    -> bounded full-parameter training
    -> validation loss and samples
    -> process exit
    -> deterministic resume
    -> numerically stable final state
    -> corpus mutation rejection
```

It demonstrates a real-text learning signal for the micro and 1.85M-parameter workloads. It does not establish production model quality, activation-bounded execution, direct-I/O or NVMe-specific behavior, bounded MLX optimization, pruning, or training state larger than unified memory.

## 11. Milestone status

| Milestone | Status |
|---|---|
| M0. Resident foundation | Completed |
| M1. Clean Mac M2 validation | Completed |
| M2. Competitive Apple Silicon baseline | Completed |
| M3. Versioned tensor store | Completed |
| M4A. Observable storage-backed optimizer lifecycle | Completed |
| M4B1. Bounded parameter-group forward | Completed and validated on M2 |
| M4B2. Bounded backward and gradient store | Completed and validated on M2 |
| M4B3. Streamed AdamW and atomic step publication | Completed and validated on M2 |
| M4C. Consecutive bounded steps, checkpoint, and resume | Completed and validated on M2 |
| M5. Deterministic small real-corpus frontend | Completed and validated on M2 |
| M6A. Historical-state pruning and compaction | Next |
| M6B. Activation recomputation and strict runtime budgets | Not started |
| Asynchronous prefetch and writeback | Not started |
| Intra-layer tiling | Not started |
| Bounded MLX optimizer execution | Not started |
| 124M and 350M capacity demonstrations | Not started |

## 12. Current evidence boundary

The accepted evidence does not establish:

- a representative tokenizer or production corpus;
- production model quality;
- large sharded dataset state, epochs, and shuffle semantics;
- activation recomputation from storage or activation offload;
- bounded activation and workspace residency;
- strict total-memory-pressure enforcement;
- asynchronous storage overlap;
- intra-layer tiling;
- bounded MLX backward or optimizer execution;
- direct-I/O or NVMe-specific performance behavior;
- storage pruning or compaction;
- training state larger than safe resident unified memory;
- 124M or 350M full-parameter training on the target machine.

No full out-of-core, production model-quality, throughput-at-scale, or model-capacity claim is made yet.

## 13. Next engineering gate

The next engineering gate should combine two closely related requirements:

1. **Historical-state pruning and compaction**
   - retain the current root and a declared number of recovery checkpoints;
   - identify reachable and unreachable child stores;
   - delete only state that is no longer referenced by retained roots;
   - preserve atomic recovery and integrity guarantees;
   - report reclaimed bytes and cumulative writes.

2. **Activation recomputation and strict runtime budgets**
   - stop retaining all boundary activations for the entire step;
   - choose which activations to retain, recompute, or store;
   - enforce activation and workspace budgets in addition to parameter, gradient, and optimizer budgets;
   - verify numerical behavior against the current synchronous reference path.

A larger or external training frontend can be integrated after it consumes the established tensor, checkpoint, lineage, data-identity, progress, and resume contracts rather than bypassing them.
