# History and validated evidence

This file records the development sequence and separates accepted evidence from abandoned or failed attempts.

## 1. Development method

The project deliberately added one major source of complexity at a time:

```text
resident training
    -> backend comparison
    -> versioned tensor storage
    -> observable storage lifecycle
    -> bounded forward
    -> bounded backward
    -> streamed clipping and bounded AdamW
    -> atomic root publication
    -> multi-step resume
    -> real-text data authority
    -> pruning
    -> activation recomputation
    -> measured hybrid activation planning
```

Small models were used to keep iteration fast. A capability moved to the next stage only after numerical, integrity, failure, and target-hardware checks were available.

## 2. Accepted milestone history

### M0 and M1. Resident reference and Apple MPS

The initial releases established:

- a controlled decoder-only Transformer;
- resident PyTorch training;
- Apple MPS selection and preflight;
- finite loss and gradients;
- telemetry for RSS, MPS allocation, Metal-driver allocation, swap, and fallback scans;
- deterministic fixed-batch overfit checks.

These gates established the target machine and reference numerical path. They did not establish out-of-core execution.

### M2. PyTorch MPS and MLX resident comparison

A controlled 23,213,056-parameter resident workload produced median throughput of approximately:

| Backend | Median tokens/s | Relative to PyTorch |
|---|---:|---:|
| PyTorch MPS | 1,379.23 | 1.000x |
| PyTorch MPS checkpointed | 1,337.37 | 0.970x |
| MLX | 2,195.69 | 1.592x |

Decision: **dual backend**.

- MLX is the optimized Apple Silicon candidate.
- PyTorch MPS remains the numerical, debugging, state-comparison, and recovery oracle.

This was a resident benchmark only. It did not validate bounded MLX backward, bounded MLX AdamW, or storage-backed MLX training.

### M3 and M4A. Versioned storage and observable lifecycle

The storage layer introduced:

- canonical tensor bytes;
- tensor identity, kind, shape, dtype, byte order, and version;
- immutable content-addressed chunks;
- copy-on-write manifests;
- checksums;
- transaction journals;
- atomic `CURRENT` publication;
- conservative recovery;
- corruption detection;
- PyTorch and MLX state adapters.

MicroColossus 0.5.0, commit `82e53c671848d231c2361443882b97dbe4e3a408`, validated an observable storage-backed optimizer lifecycle on the M2. It did not yet provide layer-wise bounded training.

### M4B1. Parameter-group bounded forward

MicroColossus 0.6.0, commit `1feea9f9eef28e551ad4ae4944614083effa804f`, validated forward execution in ordered groups:

```text
embedding
block-0
block-1
...
final-head
```

The M2 gate showed exact boundary and logits agreement for micro and tiny workloads and correct parameter-budget rejection.

### M4B2. Reverse group-bounded backward

MicroColossus 0.7.0, commit `c72dcc2f8d8a7bd783ae263cf14476d0681b664b`, validated:

- reverse group order;
- versioned gradient storage;
- exact tensor gradients against the resident oracle;
- streamed global gradient norm;
- tied token-embedding gradient accumulation;
- parameter and gradient budget rejection;
- store verification and recovery.

AdamW streaming was still outside this milestone.

### M4B3. Bounded AdamW and atomic optimizer step

MicroColossus 0.8.0, commit `ef88198d66f1d1795ffa14dcb6db388ae1715e85`, validated one complete optimizer step with:

- group-bounded parameter, gradient, and optimizer state;
- streamed clipping;
- one update for the tied parameter;
- exact candidate restore;
- atomic root publication;
- failure preservation before manifest and before `CURRENT` rename;
- optimizer-budget rejection.

### M4C. Consecutive steps and process resume

MicroColossus 0.9.0, commit `4b1ffb20857dd948d7737484e62b007f24bf69b9`, validated:

- micro step `0 -> 5`;
- process exit at step 2 and resume `2 -> 5`;
- bitwise-exact uninterrupted versus resumed final state in the accepted run;
- contiguous lineage;
- batch cursor, seed, and checksum continuity;
- optimizer step continuity;
- configuration mismatch rejection;
- corrupt-child detection;
- parameter, gradient, and optimizer budget rejection;
- root and child-store recovery.

### M5. Deterministic real-text training

MicroColossus 0.10.0, commit `8bc277123267c3d3f15bf60cd640819fa823d2e3`, introduced:

- `utf8-bytes-v1` tokenizer;
- checksummed corpus identity;
- deterministic train and validation split;
- deterministic random text windows;
- byte offsets, seeds, and batch checksums;
- validation loss;
- deterministic greedy samples;
- corpus-mutation rejection;
- real-text multi-step resume.

The real-text micro validation loss decreased from approximately `5.5484` to `3.3023` over 20 steps. The 1,846,656-parameter small workload decreased validation loss from approximately `5.6874` to `4.0840` over 10 steps.

A validation prompt initially expected the old synthetic-micro count of 11,456 parameters. The real-text micro count is 18,624. The protocol was corrected and a regression test was added. This was a protocol error, not a training failure.

### M6A. Safe historical-state pruning

MicroColossus 0.11.0 was first tested at commit `c4ee7e480b9a1f9aa4e061ddec7e61fb02af429f`. The apply path incorrectly rejected benign retained-state telemetry growth after planning.

The corrected commit is:

```text
1fedf611e7a090dad218be64811e0a4e007fbd77
```

The accepted M2/APFS gate validated:

- deterministic, non-mutating plans;
- byte-identical `CURRENT`;
- exact deletion-target inventories;
- retained-checkpoint verification;
- idempotent reapply;
- continuation after an interrupted deletion;
- retry after a pre-journal failure and later diagnostic reads;
- stale-plan rejection;
- retained-corruption rejection;
- corrupt orphan removal;
- training lock rejection;
- resume after pruning;
- micro reclamation of `10,739,392` bytes;
- small selected and reclaimed bytes of `289,540,389`;
- clean source state.

### M6B. Persistent activation recomputation

MicroColossus 0.12.0 was accepted on M2 at commit:

```text
4742f8a7f57a46edb075159275fb66c83c78ced7
```

The accepted gate validated `retain_all` against `recompute` on micro, tiny, and small workloads.

Key results:

- all parameter and optimizer comparisons were numerically stable;
- batch provenance and sample sequences matched;
- `recompute` retained zero forward boundaries;
- expected replay totals were 15 micro, 18 tiny, and 15 small;
- process restart and resume passed;
- pruning followed by recompute resume passed;
- policy mismatch and budget rejection passed;
- publication failure recovery passed;
- tied-gradient accumulation count remained 2;
- tied-parameter AdamW update count remained 1.

Small logical activation evidence:

```text
retain_all forward-boundary bytes: 491,520
recompute forward-boundary bytes:        0
```

Small physical observation:

```text
retain_all sampled peak RSS: 444,071,936 bytes
recompute sampled peak RSS:  533,528,576 bytes
```

Conclusion: full-prefix recomputation is correct and logically bounded, but it was not a physical RSS improvement for the measured small workload.

## 3. M6C attempt and failure

M6C is specified in issue `#27` as a measured hybrid activation-anchor planner.

A draft candidate was assembled in PR `#31` and tested on the M2 at commit:

```text
d98ac10a9861f16db305f416f3afa37cb905e5d6
```

The diagnostic correctly stopped at the quality gate.

Observed failures included:

- package import failure;
- `ImportError` for `run_bounded_backward_from_store` imported from the wrong module;
- package version still reporting `0.12.0`;
- planner API not exported;
- `microcolossus-activation-plan` not installed;
- four Ruff F401 errors;
- twelve mypy errors;
- twenty pytest collection errors;
- temporary integration workflow still tracked.

No M6C runtime, plan, numerical, resume, pruning, or target-memory result was validated from that candidate.

A second temporary branch, PR `#33`, contains checksummed patch payload parts and one-shot workflows. It is not product code and is not an accepted implementation.

## 4. Accepted baseline at handoff

The failed transfer attempts did not produce accepted M6C evidence. A later
clean branch, PR `#36`, validated M6C on Apple M2 at commit
`8e9b0f8e58fdaa288ba551d994d9b8b81adbea12`.

The accepted documentation baseline immediately before the failed M6C transfer experiments is:

```text
9f9365e597693e2cffa4f454180203e7219a7cde
```

Any successor should start from that accepted source behavior or from a cleaned `main` that is proven equivalent, then implement M6C through normal source commits and CI.
