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

### 4.1 Accepted runtime benchmark

Tested commit:

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

Competitive workload:

```text
parameters:          23,213,056
Transformer blocks: 6
hidden size:         512
attention heads:     8
vocabulary:          8,192
sequence length:     128
microbatch:          1
precision:           FP32
optimizer:           full-parameter AdamW
rounds:              3 counterbalanced rounds
```

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

MicroColossus 0.5 added one fully observed storage-backed micro optimizer lifecycle.

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

Environment and quality:

- MacBook Air `Mac14,2`, Apple M2, 8 GB unified memory;
- native arm64 without Rosetta;
- Python 3.13.12;
- PyTorch 2.13.0;
- MPS built and available;
- fallback unset;
- Ruff, mypy, 72 tests with one skip, and compileall passed;
- source tree clean before and after execution.

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

M4B3 established one complete bounded optimizer step. It did not yet establish a persistent trajectory.

## 9. Persistent multi-step training and process resume

Version 0.9 advances the authoritative root from step N to step N+1, preserving parameter state, Adam moments, optimizer step tensors, parent lineage, batch provenance, and configuration identity.

Accepted runtime commit:

```text
4b1ffb20857dd948d7737484e62b007f24bf69b9
```

Package version: `0.9.0`.

Overall result: **PASS**.

### 9.1 Environment and source integrity

- MacBook Air `Mac14,2`;
- Apple M2 with 8 GPU cores;
- 8 GB unified memory;
- native arm64;
- Rosetta translation value `0`;
- Python 3.13.12;
- PyTorch 2.13.0;
- MPS built and available;
- `PYTORCH_ENABLE_MPS_FALLBACK` unset;
- free storage before installation: `9,346,940,928` bytes;
- free storage after validation: `8,295,481,344` bytes;
- initial and final `git status --short`: empty;
- final `git diff --check`: clean.

### 9.2 Project quality gate

- Ruff: PASS;
- mypy: PASS over 33 source files;
- pytest: PASS, 80 passed and 1 skipped;
- compileall: PASS;
- doctor: PASS;
- bounded training CLI help: PASS;
- micro and tiny static plans: PASS.

### 9.3 Successful trajectories

| Scenario | Start | Final | Resume | Classification |
|---|---:|---:|---|---|
| Micro uninterrupted | 0 | 5 | false | GREEN |
| Micro phase 1 | 0 | 2 | false | GREEN |
| Micro process restart | 2 | 5 | true | GREEN |
| Tiny uninterrupted | 0 | 3 | false | GREEN |

Lineage:

```text
micro uninterrupted: [0, 1, 2, 3, 4, 5]
micro resumed:       [0, 1, 2, 3, 4, 5]
tiny:                [0, 1, 2, 3]
```

Batch cursor, seed, and checksum sequences were equivalent between uninterrupted and resumed micro execution.

Final optimizer step tensors:

- every micro tensor: `5.0`;
- every tiny tensor: `3.0`.

### 9.4 Resume equivalence

Uninterrupted step 0 to 5 and process-restarted step 0 to 2 plus step 2 to 5 produced:

```text
classification: BITWISE_EXACT
names equal: true
structures equal: true
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

This is the accepted proof that process restart did not reinitialize model or Adam state in the tested path.

### 9.5 Numerical comparison with the resident reference

Across the accepted scenarios:

- maximum per-step loss difference: `0.0`;
- maximum per-step gradient-norm difference: `1.6985336648289717e-07`;
- maximum final-state absolute difference: `7.450580596923828e-09`;
- mean final-state absolute difference: `1.0105799101017887e-10`;
- final candidate restore exactness: true;
- all values finite.

Tiny final-state comparison was tighter:

- maximum absolute difference: `2.3283064365386963e-10`;
- mean absolute difference: `3.718845572899834e-16`.

The final resident replay is validation-only. It is not part of the bounded production path.

### 9.6 Working-set evidence

| Configuration | Maximum parameter group | Maximum gradient group | Maximum optimizer group |
|---|---:|---:|---:|
| Micro | 33,280 bytes | 33,280 bytes | 133,152 bytes |
| Tiny | 788,480 bytes | 788,480 bytes | 3,153,952 bytes |

Configured budgets:

```text
parameter: 1,048,576 bytes
gradient:  1,048,576 bytes
optimizer: 4,194,304 bytes
```

All successful groups respected their budgets.

Intentional rejection results:

- parameter: PASS_PARAMETER_BUDGET_REJECTION;
- gradient: PASS_GRADIENT_BUDGET_REJECTION;
- optimizer: PASS_OPTIMIZER_BUDGET_REJECTION.

Each rejected training root retained authoritative step 0 and remained verifiable and recoverable.

### 9.7 Later-step atomicity

The failure harness first completed step 2, then interrupted publication of step 3.

Before bundle-manifest rename:

```text
result: PASS_PREVIOUS_STEP_2_PRESERVED
CURRENT before: step 2
CURRENT after:  step 2
```

Before root `CURRENT` rename:

```text
result: PASS_PREVIOUS_STEP_2_PRESERVED
CURRENT before: step 2
CURRENT after:  step 2
unpublished step-3 bundle: reported by recovery
```

Candidate child stores may exist after interruption, but they do not become authoritative without root publication.

### 9.8 Resume and integrity rejection

Configuration mismatch:

```text
PASS_CONFIGURATION_MISMATCH_REJECTED
```

A changed learning rate produced `ResumeConfigurationError` identifying `config_digest`. Step 2 remained authoritative.

Corrupt child state:

```text
PASS_CORRUPT_CHILD_DETECTED
```

A referenced parameter chunk was shortened from 8,192 bytes to 24 bytes in an isolated copy. Resume failed with `IntegrityError` before publishing another step.

### 9.9 Storage and publication telemetry

Across successful roots in the accepted validation:

- total root storage size: `59,901,060` bytes;
- parameter bytes read: `5,782,016`;
- parameter bytes written: `5,782,016`;
- gradient bytes read: `5,782,016`;
- optimizer bytes read: `11,564,752`;
- optimizer bytes written: `11,572,248`;
- reused chunks: `167`;
- publication `fsync` total: `0.0074204159` seconds;
- rename and publication total: `0.0025823780` seconds.

Historical root bundles and child stores were retained. No automatic pruning or compaction was tested.

These are application-level counters. They are not physical NAND-write measurements.

### 9.10 Memory, swap, and fallback

- maximum RSS: `410,943,488` bytes;
- sampled maximum MPS current allocation: `0` bytes;
- maximum Metal driver allocation: `25,935,872` bytes;
- swap delta: `0` bytes;
- no unexpected thermal warning was reported;
- no hidden fallback evidence was found;
- no unsupported operator evidence was found;
- no unexpected command failed.

The zero sampled MPS current-allocation value is retained as reported. MPS availability, device selection, successful computation, and Metal driver activity were independently observed.

### 9.11 Accepted conclusion

M4C is complete for the tested PyTorch MPS reference path.

The accepted evidence establishes:

```text
persistent step 0
    -> bounded step 1
    -> bounded step 2
    -> process exit
    -> reopen CURRENT
    -> bounded step 3
    -> bounded step 4
    -> bounded step 5
```

with exact uninterrupted-versus-resumed final state, correct Adam continuity, deterministic batch provenance, contiguous root lineage, later-step atomicity, integrity rejection, and clean source state.

It does not establish real-corpus training, activation-bounded execution, bounded MLX optimization, or training state larger than unified memory.

## 10. Milestone status

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
| M5. Small real-corpus training frontend | Next milestone |
| Activation recomputation and strict runtime budgets | Not started |
| Asynchronous prefetch and writeback | Not started |
| Intra-layer tiling | Not started |
| Bounded MLX optimizer execution | Not started |
| 124M and 350M capacity demonstrations | Not started |

## 11. Current evidence boundary

The accepted evidence does not establish:

- real tokenizer and corpus training;
- persisted real-dataset shard, sample, epoch, shuffle, and tokenizer state;
- activation recomputation from storage or activation offload;
- bounded activation and workspace residency;
- strict total-memory-pressure enforcement;
- asynchronous storage overlap;
- intra-layer tiling;
- bounded MLX backward or optimizer execution;
- storage pruning or compaction;
- training state larger than safe resident unified memory;
- 124M or 350M full-parameter training on the target machine.

No full out-of-core, model-quality, throughput-at-scale, or model-capacity claim is made yet.

## 12. Next engineering gate

The next functional milestone is a small real-text training frontend.

It must:

1. introduce a tokenizer and immutable tokenizer identity;
2. identify the corpus and exact preprocessing recipe;
3. persist shard, sample, epoch, and shuffle position transactionally;
4. preserve deterministic resume across process restart;
5. report training loss and validation loss;
6. generate sample text from checkpoints;
7. keep the existing root-bundle authority and configuration-digest rules;
8. compare uninterrupted and resumed trajectories;
9. use micro or few-million-parameter workloads for fast iteration;
10. keep activation-memory and larger-than-resident claims explicitly out of scope until separately validated.

A future external training project may provide the frontend, but it must consume the established storage, checkpoint, lineage, and resume contracts rather than bypass them.
