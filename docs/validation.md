# MicroColossus Validation Ledger

This ledger records accepted executable evidence, corrected failures, tested commits, engineering conclusions, and the exact boundary of every result.

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
- full-parameter, compact, and adapter methods are reported separately;
- a larger parameter count is not success when correctness, recovery, throughput, or endurance are unacceptable.

## 2. Accepted evidence summary

| Area | Version or commit | Target result | Main conclusion |
|---|---|---|---|
| Resident MPS foundation | `a56fc514f2f8e705654034f3c2f02e3a441c61f3` | PASS on 8 GB M2 | Native MPS forward, backward, AdamW, telemetry, and learning work |
| Competitive PyTorch and MLX | `785183a1ff87df0c22df9619d1ab7bf53968bc79`, finalized by 0.3.3 | PASS runtime and clean release verification | MLX was 1.592x faster in the tested resident workload. Dual backend selected |
| Versioned tensor store and lifecycle | `82e53c671848d231c2361443882b97dbe4e3a408`, 0.5.0 | PASS on 8 GB M2 | Canonical state, recovery, failure injection, PyTorch and MLX round trips work |
| Bounded forward | `1feea9f9eef28e551ad4ae4944614083effa804f`, 0.6.0 | PASS on 8 GB M2 | One parameter group at a time matched resident boundaries, logits, and loss |
| Bounded backward | `c72dcc2f8d8a7bd783ae263cf14476d0681b664b`, 0.7.0 | PASS on 8 GB M2 | Reverse group execution produced exact tested gradients and valid gradient stores |
| Bounded AdamW and root bundles | `ef88198d66f1d1795ffa14dcb6db388ae1715e85`, 0.8.0 | CPU CI PASS. MPS gate pending | One complete group-bounded optimizer update and atomic root publication are implemented |

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

### 4.1 First blocked attempt

Commit `2f7963a5d0af3eabb5a31eab4013f422725c71c0` stopped at the project-check gate because of Ruff import ordering and a NumPy typing-version incompatibility. No backend decision was made from that attempt.

Release 0.3.2 corrected import ordering, constrained NumPy to `>=2.4,<2.5`, added Python 3.11 and 3.13 CI, and added packaging regression checks.

### 4.2 Competitive target run

The complete runtime benchmark tested:

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

The run completed:

- PyTorch MPS;
- checkpointed PyTorch MPS;
- MLX;
- tiny smoke runs;
- nine 23,213,056-parameter competitive runs;
- identical portable initial-state checksums;
- identical batch checksums;
- valid final `.state.npz` artifacts;
- tensor-level numerical comparisons;
- clean source-tree checks.

### 4.3 Competitive workload

```text
parameters:        23,213,056
Transformer blocks: 6
hidden size:       512
attention heads:   8
vocabulary:        8,192
sequence length:   128
microbatch:        1
precision:         FP32
optimizer:         full-parameter AdamW
rounds:            3 counterbalanced rounds
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
- Storage, transaction, and execution-plan formats remain backend-neutral.

### 4.7 Final 0.3.3 quality verification

Commit `b75d2f646da4ca4dce5acdee567a1f17adcc503c` passed:

- Ruff;
- mypy with no issues in 17 source files;
- 28 tests;
- compileall;
- doctor;
- PyTorch, checkpointed PyTorch, and MLX tiny smoke runs;
- artifact and checksum validation;
- clean source-tree checks.

The tiny results did not materially differ from the prior competitive baseline, so the nine larger rounds were not repeated.

## 5. Versioned tensor store and observable optimizer lifecycle

### 5.1 Implementation scope

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

### 5.2 Clean 0.5 target validation

Tested commit:

```text
82e53c671848d231c2361443882b97dbe4e3a408
```

Overall result: **PASS**.

Environment:

- MacBook Air `Mac14,2`;
- Apple M2;
- 8 GB unified memory;
- native arm64 without Rosetta;
- PyTorch 2.13.0 with MPS;
- MLX 0.32.0;
- fallback disabled.

Project quality:

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

Memory:

- maximum RSS: `432,472,064` bytes;
- maximum MPS allocation: `8,709,376` bytes;
- maximum Metal driver allocation: `28,049,408` bytes;
- swap delta: zero.

This result validated storage lifecycle and recovery. It did not validate bounded compute.

## 6. Bounded parameter-group forward

### 6.1 Implemented path

Version 0.6:

1. creates a parameter-only canonical store;
2. releases bootstrap and resident models before bounded execution;
3. executes and releases embeddings;
4. executes and releases one Transformer block at a time;
5. reloads tied token embeddings for the output projection;
6. rejects a group that exceeds the parameter budget;
7. compares every group boundary, final logits, and loss;
8. verifies that the committed parameter manifest remains unchanged.

### 6.2 Clean 0.6 target validation

Tested commit:

```text
1feea9f9eef28e551ad4ae4944614083effa804f
```

Overall result: **PASS**.

Project quality:

- Ruff passed;
- mypy passed with no issues in 24 source files;
- 56 tests passed;
- compileall passed;
- final repository state was clean.

Bounded-forward results:

- micro run 1: GREEN;
- micro run 2: GREEN;
- micro repeatability: BITWISE_EXACT;
- tiny run: GREEN;
- intentional parameter-budget rejection: passed;
- micro order: `embedding`, `block-0`, `final-head`;
- tiny order: `embedding`, `block-0`, `block-1`, `final-head`;
- micro reads: 13 total, 12 unique, 1 repeated;
- tiny reads: 21 total, 20 unique, 1 repeated;
- tied token embedding reloaded exactly once;
- largest micro group: `33,280` bytes;
- largest tiny group: `788,480` bytes;
- budget: `1,048,576` bytes;
- maximum boundary difference: `0.0`;
- maximum logits difference: `0.0`;
- loss difference: `0.0`;
- parameter manifest unchanged;
- store verification and recovery passed.

Memory:

- maximum RSS: `322,273,280` bytes;
- maximum MPS allocation: `1,181,696` bytes;
- maximum Metal driver allocation: `19,300,352` bytes;
- swap delta: zero.

No fallback, unsupported operation, non-finite value, allocation failure, or unexpected command failure was detected.

Hidden boundary activations remained resident. The bounded claim applied to managed parameter residency.

## 7. Bounded backward and versioned gradients

### 7.1 Implemented path

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

### 7.2 Clean 0.7 target validation

Tested commit:

```text
c72dcc2f8d8a7bd783ae263cf14476d0681b664b
```

Overall result: **PASS**.

Project quality:

- Ruff passed;
- mypy passed with no issues in 25 source files;
- 62 tests passed with one skip;
- compileall passed;
- final repository state was clean.

Bounded-backward results:

- micro run 1: GREEN;
- micro run 2: GREEN;
- repeatability: BITWISE_EXACT;
- tiny run: GREEN;
- micro reverse order: `final-head`, `block-0`, `embedding`;
- tiny reverse order: `final-head`, `block-1`, `block-0`, `embedding`;
- largest micro parameter and gradient group: `33,280` bytes;
- largest tiny parameter and gradient group: `788,480` bytes;
- parameter and gradient budgets: `1,048,576` bytes each;
- resident oracle released before bounded execution;
- bootstrap payloads released before bounded execution;
- parameter manifest remained unchanged;
- micro gradient count: 12;
- tiny gradient count: 20;
- tied-gradient accumulation count: 2;
- tied-gradient final version: 1;
- maximum loss difference: `0.0`;
- maximum tensor-gradient difference: `0.0`;
- maximum global-norm difference: approximately `9.57e-08`;
- parameter and gradient budget rejection passed;
- parameter, oracle-gradient, and bounded-gradient stores verified and recovered.

Memory and I/O:

- total parameter bytes read: `4,027,392`;
- total logical gradient bytes written: `2,013,696`;
- maximum RSS: `361,496,576` bytes;
- maximum MPS allocation: `2,132,480` bytes;
- maximum Metal driver allocation: `19,644,416` bytes;
- swap delta: zero.

No unexpected fallback, unsupported operation, non-finite value, or command failure was detected.

This result completed the bounded-backward and gradient-store milestone. Parameters and AdamW state were not updated in this phase.

## 8. Version 0.8 bounded optimizer and root bundle

### 8.1 Implemented path

Version 0.8:

1. consumes the validated final-gradient store;
2. applies a canonical global clipping coefficient;
3. deduplicates optimizer groups so tied parameters update once;
4. reads one parameter, gradient, first-moment, second-moment, and step group at a time;
5. executes reference AdamW;
6. writes candidate parameter and optimizer stores;
7. validates the complete candidate state against a resident oracle;
8. restores and re-exports the candidate state;
9. publishes a root step bundle only after child stores verify;
10. preserves the previous root bundle when publication is interrupted.

Current main implementation commit:

```text
ef88198d66f1d1795ffa14dcb6db388ae1715e85
```

### 8.2 Accepted CPU CI evidence

The 0.8 pull-request matrix passed on Python 3.11 and 3.13:

- installation;
- Ruff;
- mypy;
- pytest;
- compileall;
- bounded-step CPU smoke;
- exact resident-versus-candidate parameter and optimizer state;
- exact candidate-versus-restored state;
- deterministic repeated CPU execution;
- unique tied-weight update;
- optimizer working-set rejection;
- failure before bundle-manifest rename;
- failure before root `CURRENT` rename;
- bundle checksum verification.

### 8.3 Target MPS gate

Target MPS validation is pending.

The target protocol requires:

- two micro bounded-step runs;
- one tiny bounded-step run;
- raw loss, gradient-norm, parameter, and optimizer-state comparison;
- tied-parameter update count equal to one;
- candidate tensor versions equal to one;
- root step transition from 0 to 1;
- exact candidate restore;
- optimizer working-set rejection;
- both bundle failure points preserving step 0;
- child-store and root-bundle verification and recovery;
- fallback disabled;
- clean source tree.

No MPS claim for 0.8 is accepted until that run completes.

## 9. Milestone status

| Milestone | Status |
|---|---|
| M0. Resident foundation | Completed |
| M1. Clean Mac M2 validation | Completed |
| M2. Competitive Apple Silicon baseline | Completed |
| M3. Versioned tensor store | Completed |
| M4A. Observable storage-backed optimizer lifecycle | Completed |
| M4B1. Bounded parameter-group forward | Completed, including target M2 validation |
| M4B2. Bounded backward and gradient store | Completed, including target M2 validation |
| M4B3. Streamed AdamW and atomic step publication | Implemented. Target MPS validation pending |
| M4C. Consecutive bounded steps, checkpoint, and resume | Not started |
| M5. Activation recomputation, offload, and strict total budgets | Not started |
| M6. Asynchronous prefetch and writeback | Not started |
| M7. Intra-layer tiling | Not started |
| M8. Real-corpus training frontend | Not started |
| M9. 124M and 350M capacity demonstrations | Not started |

## 10. Current evidence boundary

The accepted evidence does not establish:

- multiple consecutive bounded optimizer steps;
- checkpoint and resume for the bounded runtime;
- dataset cursor or RNG recovery across process restart;
- activation recomputation from storage;
- activation offload;
- strict total-memory-pressure enforcement;
- asynchronous storage overlap;
- intra-layer tiling;
- bounded MLX backward or optimizer execution;
- real-corpus language-model training;
- training state larger than safe resident unified memory;
- 124M or 350M full-parameter training on the target machine.

## 11. Next accepted gate after 0.8 MPS

After a clean 0.8 MPS PASS, the next required evidence is a persistent multi-step run:

1. step N+1 starts from the authoritative bundle N;
2. parameter and Adam state continue across steps;
3. optimizer step tensors increment correctly;
4. batch cursor and RNG state are persisted;
5. process exit and resume reproduce uninterrupted execution;
6. interruption during a later step preserves the prior root bundle;
7. storage growth, chunk reuse, stale-version retention, and garbage collection are measured;
8. loss remains finite over many steps.
