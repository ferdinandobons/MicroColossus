# MicroColossus Validation Ledger

This ledger records executable evidence, engineering decisions, and the exact boundary of each accepted result.

## Validation policy

Every accepted run identifies the tested commit, machine, software versions, commands, artifacts, failures, and capabilities that were not exercised.

Mandatory distinctions:

- a static plan is not a training result;
- resident training is not out-of-core training;
- a CPU result is not evidence for MPS or MLX;
- a functional pass with local source changes is not a clean protocol pass;
- checksum equality is stronger than numerical agreement;
- framework memory counters are not equivalent physical-memory counters;
- application storage bytes are not NAND-level writes;
- full-parameter, compact, and adapter methods are reported separately;
- a larger parameter count is not success when correctness, recovery, throughput, or endurance are unacceptable.

## Resident Apple M2 foundation

The first executable diagnostic exposed Ruff formatting, an unintended NumPy checksum dependency, and an MPS-incompatible float64 gradient-norm path. After correction, a clean MacBook Air M2 rerun tested commit:

```text
a56fc514f2f8e705654034f3c2f02e3a441c61f3
```

Result: **PASS**.

The run validated native arm64 execution, MPS availability, resident full-parameter training, automatic MPS selection, CPU-versus-MPS comparison, fixed-batch learning, package checks, and a clean source tree.

Observed fixed-batch learning:

```text
5.566842079162598 -> 0.4145740866661072
```

No fallback, unsupported operator, non-finite value, or MPS out-of-memory failure was detected in the tested path. MPS bitwise and numerical reproducibility remain separate properties.

## Competitive PyTorch MPS and MLX validation

The competitive target run tested commit:

```text
785183a1ff87df0c22df9619d1ab7bf53968bc79
```

Runtime result: **PASS**.

It completed PyTorch MPS, checkpointed PyTorch MPS, and MLX on the tiny workload and in three counterbalanced rounds on a 23,213,056-parameter resident workload.

| Variant | Median tokens/s | Relative to PyTorch |
|---|---:|---:|
| PyTorch MPS | 1,379.23 | 1.000x |
| PyTorch MPS checkpointed | 1,337.37 | 0.970x |
| MLX | 2,195.69 | 1.592x |

PyTorch-versus-MLX maximum loss difference was `1.9073486328125e-06`. Maximum final-parameter absolute difference was `3.8036610931158066e-05`. Mean final-parameter absolute difference was `7.378751249633382e-09`.

Backend decision: **DUAL BACKEND**.

- MLX is the preferred optimized Apple Silicon execution candidate.
- PyTorch MPS remains the numerical oracle and recovery or debugging reference.
- Storage, transaction, and execution-plan formats remain backend-neutral.

A later clean verification of version 0.3.3 passed Ruff, mypy, 28 tests, compileall, doctor, and all three tiny backend smoke paths.

## Versioned tensor store and observable optimizer lifecycle

MicroColossus 0.4 introduced canonical tensor, chunk, manifest, journal, and telemetry schemas, content-addressed chunks, copy-on-write versions, atomic publication, budgets, corruption detection, recovery, failure injection, and framework adapters.

MicroColossus 0.5 added one fully observed storage-backed micro optimizer lifecycle. CPU CI required exact resident-versus-storage state, exact committed-versus-restored state, deterministic repeats, failure recovery, and clean checks on Python 3.11 and 3.13.

### Clean 0.5 target validation

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
- pytest passed, 52 tests;
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

All five injected interruption points preserved the prior committed manifest. MLX micro and tiny round trips passed exactly. Cross-backend canonical model state was GREEN.

Reported application-level totals:

- bytes read: `7,468,738`;
- referenced chunk reads: `226`;
- bytes written: `7,641,680`;
- chunk writes: `166`;
- reused chunks: `60`;
- reuse ratio: `26.55%`;
- fsync time: `0.010095799` seconds;
- publication time: `0.001221206` seconds.

Maximum RSS was `432,472,064` bytes. Maximum MPS allocation was `8,709,376` bytes. Maximum Metal driver allocation was `28,049,408` bytes. Swap delta was zero.

This result completed the versioned tensor-store and observable optimizer-lifecycle milestones. It did not validate bounded execution.

## Bounded parameter-group forward

MicroColossus 0.6 introduced a PyTorch reference executor that:

1. creates a parameter-only canonical store;
2. releases the bootstrap model;
3. executes and releases token and position embeddings;
4. executes and releases one Transformer block at a time;
5. reloads tied token embeddings for the output projection;
6. rejects an execution group that exceeds the parameter budget;
7. compares each boundary, final logits, and loss with the resident oracle;
8. verifies that the committed parameter manifest remains unchanged.

### Clean 0.6 target validation

Tested commit:

```text
1feea9f9eef28e551ad4ae4944614083effa804f
```

Overall result: **PASS**.

Environment:

- MacBook Air `Mac14,2`;
- Apple M2;
- 8 GB unified memory;
- native arm64 without Rosetta;
- fallback disabled.

Project quality:

- Ruff passed;
- mypy passed with no issues in 24 source files;
- pytest passed, 56 tests in 14.70 seconds;
- compileall passed;
- final repository state was clean.

Bounded-forward results:

- micro run 1: GREEN;
- micro run 2: GREEN;
- micro repeatability: BITWISE_EXACT;
- 443,648-parameter tiny run: GREEN;
- intentional budget rejection: passed;
- micro group order: `embedding`, `block-0`, `final-head`;
- tiny group order: `embedding`, `block-0`, `block-1`, `final-head`;
- micro reads: 13 total, 12 unique, 1 repeated;
- tiny reads: 21 total, 20 unique, 1 repeated;
- tied token embedding was reloaded exactly once;
- largest micro group: `33,280` bytes;
- largest tiny group: `788,480` bytes;
- configured budget: `1,048,576` bytes;
- maximum boundary difference: `0.0`;
- maximum logits difference: `0.0`;
- loss difference: `0.0`;
- parameter manifest unchanged in every run;
- store verification and recovery passed.

Maximum RSS was `322,273,280` bytes. Maximum MPS allocation was `1,181,696` bytes. Maximum Metal driver allocation was `19,300,352` bytes. Swap delta was zero.

No fallback, unsupported operation, non-finite value, allocation failure, or unexpected command failure was detected.

This result completed the bounded parameter-group forward milestone. Hidden activations remained resident between groups.

## Bounded backward and versioned gradients

MicroColossus 0.7 implements the next isolated phase:

1. execute the bounded forward path and retain detached boundary activations on CPU;
2. process groups in reverse order;
3. reload one parameter group at a time;
4. recompute one local forward with autograd enabled;
5. propagate one upstream activation gradient at a time;
6. commit parameter gradients into a separate versioned gradient store;
7. combine the output-head and embedding contributions for tied token embeddings;
8. stream final gradients to calculate the global norm;
9. compare every final gradient tensor with the resident PyTorch oracle;
10. keep the parameter manifest immutable.

The tied token-embedding gradient is first published at version 0 and then updated to version 1 when the embedding contribution is added.

CPU validation requires exact gradients and exact global norm. MPS target validation must report raw loss, norm, and tensor differences, plus group budgets, storage traffic, memory counters, store recovery, and repeatability.

This phase intentionally does not update parameters or AdamW state.

## Clean 0.7 target bounded-backward validation

Commit `c72dcc2f8d8a7bd783ae263cf14476d0681b664b` passed on a clean MacBook Air M2 with 8 GB unified memory. Ruff, mypy, 62 tests with one skip, compileall, two GREEN micro runs, one GREEN tiny run, repeatability, parameter and gradient budget rejection, three-store verification and recovery, tied-gradient versioning, and clean source-state checks all passed. Maximum loss and tensor-gradient differences were zero. Maximum global-norm difference was approximately `9.57e-08`. No swap growth, fallback, unsupported operation, or non-finite value was detected.

## Version 0.8 bounded optimizer implementation

Version 0.8 consumes the validated gradient store, applies the canonical clipping coefficient, streams unique parameter groups with matching Adam state, writes candidate parameter and optimizer stores, validates the complete result against a resident oracle, and publishes one atomic root step bundle. CPU CI requires exact state and failure recovery before target MPS validation.

## Milestone status

| Milestone | Status |
|---|---|
| M0. Resident foundation | Completed |
| M1. Clean Mac M2 validation | Completed |
| M2. Competitive Apple Silicon baseline | Completed |
| M3. Versioned tensor store | Completed |
| M4A. Observable storage-backed optimizer lifecycle | Completed |
| M4B1. Bounded parameter-group forward | Completed |
| M4B2. Bounded backward and gradient store | Completed, including target M2 validation |
| M4B3. Streamed AdamW and atomic step publication | Implemented, target validation pending |
| Activation recomputation and strict runtime budgets | Not started |
| Asynchronous prefetch and writeback | Not started |
| Intra-layer tiling | Not started |
| Real-corpus training frontend | Not started |
| 124M and 350M capacity demonstrations | Not started |

## Current boundary

The accepted evidence does not establish:

- multiple consecutive bounded optimizer steps;
- checkpoint and resume for the bounded runtime;
- activation recomputation from storage or activation offload;
- asynchronous storage overlap;
- intra-layer tiling;
- bounded MLX backward;
- real-corpus language-model training;
- training state larger than safe resident unified memory;
- 124M or 350M full-parameter training on the target machine.

The next hardware gate is the micro and tiny MPS validation of version 0.8: clipping, group-bounded AdamW, candidate-state equivalence, bundle failure recovery, optimizer working-set rejection, and atomic root publication.
