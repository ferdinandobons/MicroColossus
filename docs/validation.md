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

### 4.1 Blocked attempt and corrections

Commit `2f7963a5d0af3eabb5a31eab4013f422725c71c0` stopped at the project-check gate because of Ruff import ordering and a NumPy typing-version incompatibility. No backend conclusion was accepted from that attempt.

Release 0.3.2 corrected import ordering, constrained NumPy to `>=2.4,<2.5`, added Python 3.11 and 3.13 CI, and added packaging regression checks.

### 4.2 Competitive target run

The accepted runtime benchmark tested:

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

Runtime result: **PASS**.

It completed:

- PyTorch MPS;
- checkpointed PyTorch MPS;
- MLX;
- all tiny smoke paths;
- nine 23,213,056-parameter competitive runs;
- identical portable initial-state checksums;
- identical batch checksums;
- valid final-state artifacts;
- tensor-level numerical comparisons;
- clean source-tree checks.

### 4.3 Competitive workload

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

### 4.4 Performance

| Variant | Median tokens/s | Relative to PyTorch |
|---|---:|---:|
| PyTorch MPS | 1,379.23 | 1.000x |
| PyTorch MPS checkpointed | 1,337.37 | 0.970x |
| MLX | 2,195.69 | 1.592x |

The result was not classified as noisy. No material order or thermal bias was reported.

### 4.5 Numerical comparison

PyTorch versus checkpointed PyTorch:

- classification: GREEN;
- maximum loss difference: `0.0`;
- maximum final-parameter absolute difference: `0.0`.

PyTorch versus MLX:

- classification: YELLOW;
- maximum loss difference: `1.9073486328125e-06`;
- maximum final-parameter absolute difference: `3.8036610931158066e-05`;
- mean final-parameter absolute difference: `7.378751249633382e-09`;
- all values finite;
- aggregate result stable across all three rounds.

### 4.6 Backend decision

Decision: **DUAL BACKEND**.

- MLX is the preferred optimized Apple Silicon execution candidate.
- PyTorch MPS remains the numerical oracle and recovery or debugging reference.
- Storage, transaction, tensor identity, and execution-plan formats remain backend-neutral.

### 4.7 Final 0.3.3 release verification

Commit `b75d2f646da4ca4dce5acdee567a1f17adcc503c` passed Ruff, mypy, 28 tests, compileall, doctor, PyTorch, checkpointed PyTorch, and MLX tiny smoke paths, artifact checks, numerical comparisons, and clean source-tree checks.

## 5. Versioned tensor store and observable optimizer lifecycle

MicroColossus 0.4 introduced:

- canonical tensor, chunk, manifest, journal, and telemetry schemas;
- content-addressed immutable chunks;
- copy-on-write tensor versions;
- atomic `CURRENT` publication;
- storage and staging budgets;
- corruption detection;
- conservative recovery;
- failure injection;
- PyTorch and MLX adapters.

MicroColossus 0.5 added one fully observed storage-backed micro optimizer lifecycle.

### 5.1 Clean 0.5 target validation

Tested commit:

```text
82e53c671848d231c2361443882b97dbe4e3a408
```

Overall result: **PASS**.

Environment and quality:

- MacBook Air `Mac14,2`, Apple M2, 8 GB unified memory;
- native arm64 without Rosetta;
- PyTorch 2.13.0 with MPS;
- MLX 0.32.0;
- fallback disabled;
- Ruff passed;
- mypy passed with no issues in 23 source files;
- 52 tests passed;
- compileall passed;
- final repository state was clean.

MPS lifecycle:

- two GREEN micro runs;
- bitwise-exact micro repeatability;
- one GREEN 443,648-parameter tiny run;
- zero resident-versus-storage loss difference;
- zero gradient-norm difference;
- zero final-state absolute difference;
- exact storage-versus-restored state.

Failure and adapter results:

- all five injected tensor-store interruption points preserved the prior committed manifest;
- MLX micro and tiny model and optimizer round trips passed;
- cross-backend canonical model state was GREEN.

Reported application-level totals:

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

Version 0.6:

1. creates a parameter-only canonical store;
2. releases bootstrap and resident models before bounded execution;
3. executes and releases embeddings;
4. executes and releases one Transformer block at a time;
5. reloads tied token embeddings for the output projection;
6. rejects a group that exceeds the parameter budget;
7. compares every group boundary, final logits, and loss;
8. verifies that the committed parameter manifest remains unchanged.

### 6.1 Clean 0.6 target validation

Tested commit:

```text
1feea9f9eef28e551ad4ae4944614083effa804f
```

Overall result: **PASS**.

- Ruff passed;
- mypy passed with no issues in 24 source files;
- 56 tests passed;
- compileall passed;
- micro run 1 and run 2 were GREEN;
- micro repeatability was BITWISE_EXACT;
- the tiny run was GREEN;
- intentional parameter-budget rejection passed;
- tied token embedding was reloaded exactly once;
- largest micro group was `33,280` bytes;
- largest tiny group was `788,480` bytes;
- configured budget was `1,048,576` bytes;
- maximum boundary, logits, and loss differences were `0.0`;
- parameter manifests remained unchanged;
- store verification and recovery passed.

Maximum RSS was `322,273,280` bytes. Maximum MPS allocation was `1,181,696` bytes. Maximum Metal driver allocation was `19,300,352` bytes. Swap delta was zero.

Hidden boundary activations remained resident. The bounded claim applied to managed parameter residency.

## 7. Bounded backward and versioned gradients

Version 0.7:

1. executes bounded forward and retains detached boundary activations on CPU;
2. processes execution groups in reverse;
3. reloads one parameter group at a time;
4. recomputes one local forward with autograd enabled;
5. propagates one upstream activation gradient at a time;
6. commits final parameter gradients into a separate versioned store;
7. combines both tied token-embedding contributions;
8. streams final gradients to calculate the global norm;
9. compares every gradient tensor with the resident oracle;
10. keeps the parameter manifest immutable.

The tied token-embedding gradient is published at version 0 and then updated to version 1.

### 7.1 Clean 0.7 target validation

Tested commit:

```text
c72dcc2f8d8a7bd783ae263cf14476d0681b664b
```

Overall result: **PASS**.

- Ruff passed;
- mypy passed with no issues in 25 source files;
- 62 tests passed with one skip;
- compileall passed;
- two micro runs were GREEN and BITWISE_EXACT;
- the tiny run was GREEN;
- reverse group order was correct;
- resident oracle and bootstrap payloads were released before bounded execution;
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
- no fallback, unsupported operation, non-finite value, or unexpected command failure was detected;
- final repository state was clean.

This phase intentionally did not update parameters or AdamW state.

## 8. Bounded AdamW and atomic root step bundles

Version 0.8 consumes the validated final-gradient store and completes one bounded optimizer update.

The path:

1. calculates one canonical global clipping coefficient;
2. reads one unique parameter group and its final gradients;
3. reads matching Adam first moments, second moments, and step tensors;
4. applies AdamW group by group;
5. updates the tied token embedding exactly once;
6. writes candidate parameter and optimizer stores;
7. compares the complete candidate state with a resident PyTorch oracle;
8. restores and re-exports the candidate state;
9. verifies every child store;
10. atomically publishes a root step bundle;
11. preserves the prior bundle when publication is interrupted.

### 8.1 CPU CI evidence

The 0.8 implementation passed Python 3.11 and 3.13 CI with:

- Ruff;
- mypy;
- pytest;
- compileall;
- exact CPU resident-versus-candidate state;
- exact candidate restore;
- deterministic repeats;
- optimizer working-set rejection;
- bundle checksum validation;
- failure recovery;
- CPU bounded-step smoke.

### 8.2 Clean 0.8 target validation

Tested runtime commit:

```text
ef88198d66f1d1795ffa14dcb6db388ae1715e85
```

Package version:

```text
0.8.0
```

Overall result: **PASS**.

Environment:

- MacBook Air `Mac14,2`;
- Apple M2 with 8 GPU cores;
- 8 GB unified memory;
- native arm64 without Rosetta;
- Python 3.13.12;
- PyTorch 2.13.0;
- MPS built and available;
- MPS recommended maximum memory reported as `5,726,633,984` bytes;
- `PYTORCH_ENABLE_MPS_FALLBACK` unset;
- source tree clean before and after execution.

Project quality:

- Ruff: PASS, all checks passed;
- mypy: PASS, no issues in 27 source files;
- pytest: PASS, 72 passed and 1 skipped;
- compileall: PASS;
- doctor and both static plans: PASS.

Successful bounded-step runs:

- micro run 1: GREEN;
- micro run 2: GREEN;
- micro repeatability: BITWISE_EXACT;
- 443,648-parameter tiny run: GREEN.

Group order:

```text
micro: embedding, block-0, final-head
tiny:  embedding, block-0, block-1, final-head
```

Working-set evidence:

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

Correctness:

- tied parameter update count: `1`;
- initial bundle step: `0`;
- final bundle step: `1`;
- initial bundle remained authoritative until final publication: true;
- final bundle became authoritative: true;
- maximum loss difference: `0.0`;
- maximum gradient-norm difference: `9.567381065167524e-08`;
- micro clipping coefficient: `0.34425213011375594`;
- tiny clipping coefficient: `0.3056236470631644`;
- resident-versus-candidate maximum difference: `0.0`;
- resident-versus-candidate mean difference: `0.0`;
- candidate-versus-restored exact bytes: true;
- all candidate tensor versions: `1`.

State counts:

| Configuration | Candidate parameters | Candidate optimizer tensors | Final gradients |
|---|---:|---:|---:|
| Micro | 12 | 37 | 12 |
| Tiny | 20 | 61 | 20 |

Durability and rejection tests:

- optimizer working-set rejection: PASS;
- interruption before root bundle-manifest rename: previous step preserved;
- interruption before root `CURRENT` rename: previous step preserved;
- root bundle verify and recover: PASS;
- child parameter, optimizer, and gradient stores verify and recover: PASS.

Application-level totals across the accepted runs:

- parameter bytes read: `1,866,240`;
- parameter bytes written: `1,866,240`;
- gradient bytes read: `1,866,240`;
- optimizer bytes read: `3,732,656`;
- optimizer bytes written: `3,734,460`;
- publication `fsync` total: `0.0022067919926485047` seconds;
- rename and publication total: `0.0010027890239143744` seconds.

Observed resources:

- maximum RSS: `393,052,160` bytes;
- maximum MPS current allocation reported by the sampled optimizer telemetry: `0` bytes;
- maximum Metal driver allocation: `25,935,872` bytes;
- swap delta: `8,388,608` bytes.

The zero sampled MPS allocation value is recorded as reported. It is not interpreted as evidence that the MPS path did not execute. The device, MPS preflight, driver allocation, and successful MPS computation were independently recorded.

No hidden fallback, unsupported operator, non-finite value, allocation failure, or unexpected command failure was detected. The first external diagnostic scan produced a false positive and was corrected outside the repository. Project source was not modified.

### 8.3 Accepted conclusion

M4B3 is complete for the tested PyTorch MPS reference path.

The accepted evidence establishes one complete bounded optimizer step containing:

```text
bounded forward
    -> bounded backward
    -> versioned gradient store
    -> streamed global clipping
    -> group-bounded AdamW
    -> candidate parameter and optimizer stores
    -> exact restore validation
    -> atomic root step publication
```

It does not establish consecutive training steps or larger-than-memory training.

## 9. Milestone status

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
| M4C. Consecutive bounded steps, checkpoint, and resume | Next milestone |
| Activation recomputation and strict runtime budgets | Not started |
| Asynchronous prefetch and writeback | Not started |
| Intra-layer tiling | Not started |
| Real-corpus training frontend | Not started |
| 124M and 350M capacity demonstrations | Not started |

## 10. Current evidence boundary

The accepted evidence does not establish:

- multiple consecutive bounded optimizer steps;
- checkpoint and resume across a process restart;
- deterministic dataset cursor restoration;
- activation recomputation from storage or activation offload;
- asynchronous storage overlap;
- intra-layer tiling;
- bounded MLX backward or optimizer execution;
- real-corpus language-model training;
- training state larger than safe resident unified memory;
- 124M or 350M full-parameter training on the target machine.

## 11. Next engineering gate

The next milestone is M4C, tracked in GitHub issue #13.

It must:

1. initialize step 0 only once;
2. consume the current authoritative bundle for every later step;
3. advance root bundle lineage from step N to step N+1;
4. persist deterministic batch cursor and configuration provenance;
5. preserve Adam moments and optimizer step tensors across updates;
6. reopen in a new process and resume from the current bundle;
7. match an uninterrupted bounded run;
8. match a resident PyTorch reference over the same batch sequence;
9. preserve the previous step after an interrupted later publication;
10. report per-step numerical, memory, storage, write, and recovery telemetry.

The first target gate will use short micro and tiny runs. Real text data and larger capacity demonstrations remain later milestones.
