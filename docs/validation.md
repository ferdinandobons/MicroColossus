# MicroColossus Validation Ledger

This ledger records executable evidence, engineering decisions, and the exact boundary of each result.

## Validation policy

Every accepted run identifies the tested commit, machine, software versions, commands, artifacts, failures, and capabilities that were not exercised.

Mandatory distinctions:

- a static plan is not a training result;
- resident training is not out-of-core training;
- a CPU result is not evidence for MPS or MLX;
- a functional pass with local source changes is not a clean protocol pass;
- checksum equality is stronger than numerical agreement;
- framework memory counters are not assumed to represent equivalent physical memory;
- application storage bytes are not NAND-level writes;
- full-parameter, compact, and adapter methods are reported separately;
- a larger parameter count is not success when correctness, recovery, throughput, or endurance are unacceptable.

## M0 diagnostic and corrections

The first executable diagnostic tested commit:

```text
37fb45b189d13bab2b6f4084af0c21993f65ceff
```

Result: **FAIL**.

Findings:

1. Ruff import formatting failed.
2. Checksum telemetry required NumPy unexpectedly.
3. Resident training stopped during checksum calculation.
4. Reproducibility and fixed-batch diagnostics could not complete.
5. mypy reported third-party and local typing problems.

The following implementation removed the NumPy checksum dependency, added MPS support, synchronized MPS timing, added MPS memory telemetry, introduced `microcolossus doctor`, and made the planner aware of unified-memory accounting.

## Resident Apple M2 validation

The first real M2 diagnostic found a Ruff issue and an MPS-incompatible float64 gradient-norm path. After correction, a clean rerun tested:

```text
a56fc514f2f8e705654034f3c2f02e3a441c61f3
```

Result: **PASS**.

Environment:

- MacBook Air;
- Apple M2;
- 8 GB unified memory;
- native arm64;
- no Rosetta translation;
- PyTorch 2.13.0;
- MPS built and available;
- deliberate MPS-to-CPU fallback disabled.

Passed:

- Ruff, mypy, pytest, and compileall;
- MPS preflight;
- explicit MPS training;
- automatic MPS selection;
- larger resident smoke run;
- CPU-versus-MPS comparison;
- fixed-batch MPS overfit.

Observed fixed-batch learning:

```text
5.566842079162598 -> 0.4145740866661072
```

No fallback, unsupported operator, non-finite value, or MPS out-of-memory failure was detected in the tested path.

MPS was not always bitwise reproducible by final checksum. Numerical and bitwise reproducibility remain separate measurements.

## Competitive PyTorch MPS and MLX validation

The competitive target run tested commit:

```text
785183a1ff87df0c22df9619d1ab7bf53968bc79
```

Runtime result: **PASS**.

It completed PyTorch MPS, checkpointed PyTorch MPS, and MLX on the tiny workload and in three counterbalanced rounds on a 23,213,056-parameter resident workload.

Median throughput:

| Variant | Tokens/s | Relative to PyTorch |
|---|---:|---:|
| PyTorch MPS | 1,379.23 | 1.000x |
| PyTorch MPS checkpointed | 1,337.37 | 0.970x |
| MLX | 2,195.69 | 1.592x |

Numerical result:

- PyTorch versus checkpointed PyTorch was exact in the competitive runs;
- PyTorch versus MLX maximum loss difference was `1.9073486328125e-06`;
- maximum final-parameter absolute difference was `3.8036610931158066e-05`;
- mean final-parameter absolute difference was `7.378751249633382e-09`;
- all values were finite and stable across three rounds.

Backend decision: **DUAL BACKEND**.

- MLX is the preferred optimized Apple Silicon execution candidate.
- PyTorch MPS remains the numerical oracle and the recovery or debugging reference.
- Storage and transaction formats remain backend-neutral.

Version 0.3.3 then corrected three typing findings. A clean verification of commit `b75d2f646da4ca4dce5acdee567a1f17adcc503c` passed Ruff, mypy, 28 tests, compileall, doctor, and all three tiny backend smoke paths.

## Versioned tensor store and CPU storage lifecycle

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

MicroColossus 0.5 added one fully observed storage-backed micro optimizer lifecycle. CPU CI required exact resident-versus-storage state, exact committed-versus-restored state, deterministic repeats, failure recovery, and clean Ruff, mypy, pytest, and compileall on Python 3.11 and 3.13.

## Clean 0.5 target storage validation

Tested commit:

```text
82e53c671848d231c2361443882b97dbe4e3a408
```

Overall result: **PASS**.

Environment:

- MacBook Air `Mac14,2`;
- Apple M2;
- 8 GB unified memory;
- native arm64;
- `sysctl.proc_translated=0`;
- PyTorch 2.13.0 with MPS built and available;
- MLX 0.32.0;
- fallback disabled.

Project quality:

- Ruff passed;
- mypy passed with no issues in 23 source files;
- pytest passed, 52 tests in 13.63 seconds;
- compileall passed;
- final `git status --short` was empty;
- `git diff --check` was clean.

MPS storage lifecycle:

- micro run 1: GREEN;
- micro run 2: GREEN;
- repeated micro runs: bitwise exact;
- 443,648-parameter tiny run: GREEN;
- maximum resident-versus-storage loss difference: `0.0`;
- maximum gradient-norm difference: `0.0`;
- maximum final-state absolute difference: `0.0`;
- storage-versus-restored state: exact.

Durability:

- before chunk write: passed;
- during chunk write: passed;
- before chunk fsync: passed;
- before manifest rename: passed;
- before `CURRENT` rename: passed.

Every injected interruption preserved the prior committed manifest.

MLX and cross-backend validation:

- MLX micro round trip: passed;
- MLX tiny round trip: passed;
- PyTorch-versus-MLX canonical micro state: GREEN;
- PyTorch-versus-MLX canonical tiny state: GREEN;
- canonical model-state bytes were exact after loading the same portable state.

Application-level storage totals reported by the diagnostic:

- logical bytes read: `7,468,738`;
- referenced chunk reads: `226`;
- logical or managed bytes written: `7,641,680`;
- chunk writes: `166`;
- reused chunks: `60`;
- reuse ratio: `26.55%`;
- fsync time: `0.010095799` seconds;
- manifest publication time: `0.001221206` seconds.

Resource observations:

- maximum RSS: `432,472,064` bytes;
- maximum MPS allocation: `8,709,376` bytes;
- maximum Metal driver allocation: `28,049,408` bytes;
- maximum MLX active memory: `9,004,600` bytes;
- maximum MLX peak memory: `12,335,968` bytes;
- maximum MLX cache memory: `3,331,548` bytes;
- swap delta: `0`.

No hidden fallback, unsupported operation, non-finite value, MPS allocation failure, or unexpected command failure was detected.

This result completes the versioned tensor-store and observable micro-step milestones. It does not validate layer-wise bounded execution or true out-of-core training.

## Version 0.6 bounded forward implementation

Version 0.6 introduces a PyTorch reference executor that:

1. creates a parameter-only canonical store;
2. releases the bootstrap model;
3. executes and releases token and position embeddings;
4. executes and releases one Transformer block at a time;
5. reloads tied token embeddings for the output projection;
6. rejects an execution group that exceeds the declared logical parameter budget;
7. compares each boundary, final logits, and loss with the resident oracle;
8. verifies that the committed store manifest remains unchanged.

The implementation reports group reads, referenced chunks, logical parameter bytes, materialization, compute, release, activation bytes, RSS, accelerator allocation, driver allocation, checksums, and numerical distance.

The initial bounded executor retains hidden activations and implements forward propagation only. Target M2 validation is required before this milestone is completed.

## Milestone status

| Milestone | Status |
|---|---|
| M0. Resident foundation | Completed |
| M1. Clean Mac M2 validation | Completed |
| M2. Competitive Apple Silicon baseline | Completed |
| M3. Versioned tensor store | Completed |
| M4A. Observable storage-backed optimizer lifecycle | Completed |
| M4B1. Bounded parameter-group forward | Implemented, target validation pending |
| M4B2. Bounded backward and streamed optimizer update | Not started |
| Activation recomputation and strict runtime budgets | Not started |
| Asynchronous prefetch and writeback | Not started |
| Intra-layer tiling | Not started |
| Real-corpus training frontend | Not started |
| 124M and 350M capacity demonstrations | Not started |

## Current boundary

The accepted evidence does not establish:

- bounded backward propagation;
- stored or streamed gradients;
- bounded AdamW execution;
- activation recomputation or offload;
- asynchronous storage overlap;
- intra-layer tiling;
- real-corpus language-model training;
- training state larger than safe resident unified memory;
- 124M or 350M full-parameter training on the target machine.

The next accepted hardware gate is the micro and tiny MPS validation of the bounded forward executor. After that, development moves to reverse-order bounded backward and a second streamed pass for global clipping and AdamW publication.
