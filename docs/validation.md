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
- memory, swap, storage, replay, and publication telemetry;
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
- a logical working-set reduction is not automatically a physical RSS reduction;
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
| Versioned tensor store and lifecycle | `82e53c671848d231c2361443882b97dbe4e3a408`, 0.5.0 | PASS on 8 GB M2 | Canonical state, recovery, failure injection, and PyTorch or MLX round trips work |
| Bounded forward | `1feea9f9eef28e551ad4ae4944614083effa804f`, 0.6.0 | PASS on 8 GB M2 | One parameter group at a time matched resident boundaries, logits, and loss |
| Bounded backward | `c72dcc2f8d8a7bd783ae263cf14476d0681b664b`, 0.7.0 | PASS on 8 GB M2 | Reverse group execution produced exact tested gradients and valid gradient stores |
| Bounded AdamW and root bundles | `ef88198d66f1d1795ffa14dcb6db388ae1715e85`, 0.8.0 | PASS on 8 GB M2 | One complete group-bounded optimizer step and atomic root publication work on MPS |
| Persistent multi-step and resume | `4b1ffb20857dd948d7737484e62b007f24bf69b9`, 0.9.0 | PASS on 8 GB M2 | Consecutive bounded steps, process restart, exact resume, lineage, and later-step atomicity work on MPS |
| Deterministic real-text training | `8bc277123267c3d3f15bf60cd640819fa823d2e3`, 0.10.0 | PASS on 8 GB M2 after protocol correction | Real-text micro and 1.85M trajectories, validation, samples, provenance, mutation rejection, and resume work on MPS |
| Safe pruning and reclamation | `1fedf611e7a090dad218be64811e0a4e007fbd77`, 0.11.0 | PASS on 8 GB M2 and APFS | Deterministic pruning, interruption recovery, material reclamation, idempotence, and post-pruning resume work |
| Persistent activation recomputation | `4742f8a7f57a46edb075159275fb66c83c78ced7`, 0.12.0 | PASS on 8 GB M2 | Zero-forward-boundary recomputation, budgets, resume, pruning compatibility, and failure recovery work. Full-prefix recomputation increased sampled RSS in the small gate |

## 3. Resident Apple M2 foundation

Accepted commit:

```text
a56fc514f2f8e705654034f3c2f02e3a441c61f3
```

Result: **PASS**.

Validated native arm64 MPS execution, explicit and automatic MPS selection, CPU-versus-MPS numerical comparison, synchronized telemetry, and clean source state.

Fixed-batch loss:

```text
5.566842079162598 -> 0.4145740866661072
```

No fallback, unsupported operator, non-finite value, or MPS out-of-memory failure was detected in the tested path.

## 4. Competitive PyTorch MPS and MLX validation

Accepted commit:

```text
785183a1ff87df0c22df9619d1ab7bf53968bc79
```

A controlled 23,213,056-parameter workload produced:

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

## 5. Versioned storage and observable lifecycle

Accepted commit:

```text
82e53c671848d231c2361443882b97dbe4e3a408
```

Result: **PASS**.

Key evidence:

- two GREEN micro runs;
- bitwise-exact micro repeatability;
- one GREEN 443,648-parameter tiny run;
- zero resident-versus-storage loss, gradient-norm, and final-state differences;
- exact storage-versus-restored state;
- all five tensor-store failure points preserved the prior manifest;
- MLX micro and tiny round trips passed;
- cross-backend canonical model state was GREEN.

Application traffic:

```text
bytes read:       7,468,738
chunk reads:      226
bytes written:    7,641,680
chunk writes:     166
reused chunks:    60
reuse ratio:      26.55%
```

This result validated storage lifecycle and recovery. It did not validate bounded compute.

## 6. Bounded forward

Accepted commit:

```text
1feea9f9eef28e551ad4ae4944614083effa804f
```

Result: **PASS**.

- micro repeatability was BITWISE_EXACT;
- tiny was GREEN;
- parameter-budget rejection passed;
- largest micro group: `33,280` bytes;
- largest tiny group: `788,480` bytes;
- configured budget: `1,048,576` bytes;
- maximum boundary, logits, and loss differences: `0.0`;
- parameter manifests remained unchanged;
- store verification and recovery passed.

The claim applied to managed parameter residency. Hidden boundary activations were still resident.

## 7. Bounded backward and gradient storage

Accepted commit:

```text
c72dcc2f8d8a7bd783ae263cf14476d0681b664b
```

Result: **PASS**.

- two micro runs were GREEN and BITWISE_EXACT;
- tiny was GREEN;
- reverse group order was correct;
- parameter manifests remained immutable;
- tied-gradient accumulation count was `2`;
- maximum loss difference was `0.0`;
- maximum tensor-gradient difference was `0.0`;
- maximum global-norm difference was approximately `9.57e-08`;
- parameter and gradient budget rejection passed;
- parameter and gradient stores verified and recovered.

## 8. Bounded AdamW and atomic root publication

Accepted commit:

```text
ef88198d66f1d1795ffa14dcb6db388ae1715e85
```

Result: **PASS**.

| Configuration | Maximum parameter group | Maximum gradient group | Maximum optimizer group |
|---|---:|---:|---:|
| Micro | 33,280 bytes | 33,280 bytes | 133,152 bytes |
| Tiny | 788,480 bytes | 788,480 bytes | 3,153,952 bytes |

Correctness:

- tied parameter update count: `1`;
- maximum loss difference: `0.0`;
- maximum gradient-norm difference: `9.567381065167524e-08`;
- resident-versus-candidate maximum and mean differences: `0.0`;
- candidate-versus-restored exact bytes: true;
- all candidate tensor versions: `1`.

Failure before root-manifest or root-`CURRENT` publication preserved the previous authoritative bundle.

## 9. Persistent multi-step training and process resume

Accepted commit:

```text
4b1ffb20857dd948d7737484e62b007f24bf69b9
```

Result: **PASS**.

- micro uninterrupted step 0 to 5: GREEN;
- process exit at step 2 and resume to step 5: GREEN;
- uninterrupted versus resumed final state: BITWISE_EXACT;
- tiny step 0 to 3: GREEN;
- maximum per-step loss difference: `0.0`;
- maximum gradient-norm difference: `1.6985336648289717e-07`;
- maximum final bounded-versus-resident absolute difference: `7.450580596923828e-09`;
- candidate restore exactness: true;
- later-step interruption preserved the previous bundle;
- configuration mismatch was rejected;
- corrupted authoritative child state was detected;
- swap delta was zero;
- final source state was clean.

## 10. Deterministic real-text training

Accepted commit:

```text
8bc277123267c3d3f15bf60cd640819fa823d2e3
```

The external report initially labeled the release `FAIL` because the validation prompt expected 11,456 parameters for `real-text-micro.yaml`. The checked configuration correctly contains 18,624 parameters. The stale value belonged to the synthetic micro configuration.

Accepted result after correcting the protocol expectation: **PASS**.

### 10.1 Micro trajectory

```text
parameters:          18,624
Transformer blocks: 1
hidden size:         32
vocabulary:          256
sequence length:     32
microbatch:          2
```

Validation loss:

```text
step 0:  5.548418998718262
step 20: 3.302267074584961
```

A separate process resumed from step 5 to step 20. Uninterrupted and resumed states were numerically stable:

- maximum absolute difference: `1.1920928955078125e-07`;
- mean absolute difference: `8.844825718731097e-10`;
- cursors, seeds, offsets, batch checksums, sample tokens, and completions matched.

### 10.2 Corpus mutation rejection

Changing copied corpus bytes after step 2 produced `ResumeConfigurationError` for `data_identity`. Step 3 was not published.

### 10.3 Small trajectory

```text
parameters:          1,846,656
Transformer blocks: 4
hidden size:         192
vocabulary:          256
sequence length:     128
microbatch:          1
```

Validation loss:

```text
step 0:  5.687370777130127
step 10: 4.083975553512573
```

The final bounded-versus-resident maximum absolute difference was `1.1920928955078125e-07`. Candidate restore was exact.

The ten-step small root occupied approximately 632 MB because historical candidate and work stores were retained. This motivated M6A.

## 11. Safe pruning and APFS reclamation

Accepted corrected commit:

```text
1fedf611e7a090dad218be64811e0a4e007fbd77
```

Package version: `0.11.0`.

Overall result: **PASS**.

Quality gate:

- Ruff: PASS;
- mypy: PASS;
- pytest: 103 passed, 1 skipped;
- compileall: PASS;
- doctor, help, and static plan commands: PASS.

### 11.1 Micro pruning

```text
retained steps: [0, 5, 9, 10]
pruned steps:   [1, 2, 3, 4, 6, 7, 8]
selected bytes: 10,739,392

managed bytes: 13,037,759 -> 2,330,216
APFS du:       16,830,464 -> 3,235,840
reclaimed:     10,739,392 bytes
```

`CURRENT` remained byte-identical and state-identical. Retained checkpoints verified. Reapply was idempotent. Training resumed to step 12.

Pruned versus unpruned training was numerically stable:

- maximum state absolute difference: `1.1920928955078125e-07`;
- training-loss difference: `0.0`;
- validation-loss difference: `2.384185791015625e-07`;
- batch provenance and sample sequences matched.

### 11.2 Failure and drift scenarios

Passed:

- retry after failure before journal publication and later telemetry reads;
- continuation after one completed deletion;
- idempotent repeat after continuation;
- resume after interruption;
- rejection of a deletion target changed after planning.

### 11.3 Small-model reclamation

The 1,846,656-parameter real-text small run used budgets of 2 MiB for parameters, 2 MiB for gradients, and 32 MiB for optimizer state.

Result: `PASS_SMALL_MATERIAL_RECLAMATION`.

```text
retained steps: [5]
pruned steps:   [0, 1, 2, 3, 4]
selected and reclaimed: 289,540,389 bytes
managed-byte reduction: 289,445,109 bytes
APFS du reduction:      292,491,264 bytes
```

Reapply was idempotent and training resumed to step 6.

Scope boundary: synchronous ordinary-filesystem pruning on APFS. Direct NVMe I/O, activation offload, and larger-than-memory training were not validated.

## 12. Persistent activation recomputation

Accepted target commit:

```text
4742f8a7f57a46edb075159275fb66c83c78ced7
```

Package version: `0.12.0`.

Environment:

- MacBook Air with Apple M2 and 8 GB unified memory;
- native arm64;
- MPS built and available;
- MPS fallback unset;
- clean fresh clone and clean final source state.

Overall result: **PASS**.

### 12.1 Quality and runtime gates

Passed:

- Ruff;
- mypy;
- pytest;
- compileall;
- doctor, help, and static plan commands;
- micro retain-versus-recompute rounds;
- recompute process restart and resume;
- activation-policy mismatch rejection;
- activation and workspace budget rejection;
- tiny retain-versus-recompute;
- small activation-memory comparison;
- pruning followed by recompute resume;
- both root-publication failure-preservation scenarios;
- fallback, unsupported-operator, and non-finite scans.

The harness validated `1,645` JSON files and found no `.state.npz` dependency.

### 12.2 Numerical comparisons

All principal comparisons were `NUMERICALLY_STABLE`:

| Comparison | Maximum absolute state difference | Mean absolute state difference | Maximum loss difference | Maximum gradient-norm difference |
|---|---:|---:|---:|---:|
| Micro round 1, retain versus recompute | `1.862645149230957e-08` | `2.6070487856787704e-10` | `0.0` | `5.061370145220678e-08` |
| Micro round 2, retain versus recompute | `2.2351741790771484e-08` | `2.6154205006840354e-10` | `0.0` | `5.061370145220678e-08` |
| Resume recompute versus uninterrupted | `1.6763806343078613e-08` | `1.9436587520528433e-10` | `0.0` | `5.74115421869692e-09` |
| Tiny retain versus recompute | `2.3283064365386963e-10` | `8.079835528479944e-16` | `0.0` | `0.0` |
| Small retain versus recompute | `4.656612873077393e-10` | `4.673514187992496e-15` | `0.0` | `9.893160068941143e-08` |
| Pruned resume versus unpruned | `2.2351741790771484e-08` | `2.1037602991500962e-10` | `0.0` | `6.147735875927651e-08` |

Both injected publication-failure retry paths were numerically stable with maximum absolute state difference `5.960464477539063e-08`. The maximum observed loss difference in those paths was `4.76837158203125e-07`.

Batch provenance and sample sequences matched. Candidate restore remained exact.

### 12.3 Replay and activation working sets

Expected replay totals were observed:

```text
real-text micro: 15 groups over five steps
tiny:            18 groups over three steps
real-text small: 15 groups over one step
```

Small logical activation evidence:

```text
retain_all forward-boundary bytes: 491,520
recompute forward-boundary bytes:        0

retain_all maximum retained activation bytes: 491,520
recompute maximum retained activation bytes:  196,608
maximum local workspace for both policies:    393,216
```

The small recompute path read `19,200,000` logical parameter bytes during prefix replay and recorded approximately `0.102379` seconds of prefix recomputation.

This is accepted evidence for a material logical activation reduction and the zero-forward-boundary contract.

### 12.4 Physical-memory observation

The small physical-memory classification was `HIGHER`:

```text
retain_all sampled peak RSS: 444,071,936 bytes
recompute sampled peak RSS:  533,528,576 bytes
ratio:                        1.2014462809917354x
```

This does not invalidate M6B. It establishes that the synchronous full-prefix schedule is not a physical-memory optimization for the measured small workload. RSS, MPS allocation, Metal-driver allocation, filesystem cache, compressed memory, and swap remain non-additive and must be interpreted separately.

### 12.5 Safety and durability

Passed:

- activation-policy mismatch rejection;
- activation-budget and workspace-budget rejection without root publication;
- tied-gradient accumulation count `2`;
- tied-parameter AdamW update count `1`;
- pruning followed by recompute resume;
- previous-root preservation at both root-publication failure points;
- retry after the injected failures;
- hidden fallback, unsupported operator, and non-finite scans;
- clean final Git state.

### 12.6 Accepted conclusion

M6B is complete for the tested PyTorch MPS path.

The accepted result establishes persistent multi-step activation recomputation with zero forward-boundary retention, strict logical activation and workspace budgets, process resume, pruning compatibility, and atomic failure recovery.

It does not establish activation storage or offload, asynchronous overlap, direct NVMe behavior, bounded MLX optimization, intra-layer tiling, lower physical RSS on every workload, or larger-than-memory training.

## 13. Milestone status

| Milestone | Status |
|---|---|
| M0. Resident foundation | Completed |
| M1. Clean Mac M2 validation | Completed |
| M2. Competitive Apple Silicon baseline | Completed |
| M3. Versioned tensor store | Completed |
| M4A. Observable storage-backed optimizer lifecycle | Completed and validated on M2 |
| M4B1. Bounded parameter-group forward | Completed and validated on M2 |
| M4B2. Bounded backward and gradient store | Completed and validated on M2 |
| M4B3. Streamed AdamW and atomic step publication | Completed and validated on M2 |
| M4C. Consecutive bounded steps, checkpoint, and resume | Completed and validated on M2 |
| M5. Deterministic small real-corpus frontend | Completed and validated on M2 |
| M6A. Historical-state pruning and compaction | Completed and validated on M2/APFS |
| M6B. Persistent activation recomputation and strict budgets | Completed and validated on M2 |
| M6C. Measured hybrid activation-anchor planner | Open in issue #27 |
| Asynchronous prefetch and writeback | Not started |
| Intra-layer tiling | Not started |
| Bounded MLX backward and optimizer execution | Not started |
| 124M and 350M capacity demonstrations | Not started |

## 14. Current evidence boundary

The accepted evidence does not establish:

- hybrid activation anchors;
- activation tensors stored on disk;
- asynchronous activation prefetch or writeback;
- strict total physical-memory-pressure enforcement;
- direct-I/O or NVMe-specific performance behavior;
- live chunk repacking or cross-store deduplication;
- automatic pruning under storage pressure;
- a representative tokenizer or production corpus;
- production model quality;
- large sharded dataset state, epochs, and shuffle semantics;
- intra-layer tiling;
- bounded MLX backward or optimizer execution;
- training state larger than safe resident unified memory;
- 124M or 350M full-parameter training on the target machine.

No complete out-of-core, production-quality, throughput-at-scale, or model-capacity claim is made yet.

## 15. Next engineering gate

The next implementation gate is M6C, a measured hybrid activation-anchor planner.

The planner must use the accepted M2 observations rather than an arbitrary checkpoint interval. It should search for a Pareto improvement across:

- retained anchor bytes;
- local workspace;
- replayed group count;
- logical parameter rereads;
- recomputation time;
- end-to-end step time;
- RSS;
- MPS allocation;
- Metal-driver allocation;
- swap and memory pressure.

Correctness, resume, pruning, tied-weight semantics, and atomic publication must remain unchanged.