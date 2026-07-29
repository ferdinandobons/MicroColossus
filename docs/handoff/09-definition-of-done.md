# Definition of done

MicroColossus has several completion levels. This file prevents an intermediate subsystem from being confused with the original project goal.

## 1. Done for a code change

A code change is complete when:

- the implementation is reviewable source code;
- focused tests exist;
- existing tests remain green;
- public APIs and CLI surfaces are intentional;
- documentation identifies the change and its boundary;
- no temporary payload or workflow is included;
- Ruff, mypy, pytest, and compileall pass;
- the pull request is clean and mergeable.

This level does not imply target-hardware acceptance.

## 2. Done for CPU acceptance

A milestone is CPU-accepted when:

- the code change definition is satisfied;
- Python 3.11 CI passes;
- Python 3.13 CI passes;
- deterministic tests pass in fresh environments;
- failure and rejection paths pass;
- package version and release contracts match;
- a CPU smoke demonstrates the complete intended path;
- the accepted commit is recorded.

This level permits the label:

```text
Implemented and accepted in CPU CI only
```

It does not permit Apple M2, MPS, APFS, physical-memory, or performance claims.

## 3. Done for target-hardware acceptance

A milestone is target-accepted when:

- CPU acceptance is complete;
- one exact immutable commit is tested;
- the target uses native arm64 without Rosetta;
- MPS is built and available;
- fallback is disabled;
- required target scenarios pass;
- numerical state remains in the declared band;
- provenance matches;
- candidate restore is exact;
- recovery and resume pass;
- target memory and storage metrics are recorded;
- fallback and non-finite scans are clean;
- final source state is clean;
- REPORT, RESULTS, command ledger, and ZIP are preserved;
- documentation is updated with exact evidence.

This level permits:

```text
Implemented and accepted on target hardware
```

Only the functionality exercised by the protocol is accepted.

## 4. M6C definition of done

M6C is complete only when all of the following are true.

### Repository

- normal CI restored;
- temporary M6C workflows removed;
- PR `#31` and PR `#33` closed as abandoned;
- clean 0.13.0 source PR merged;
- no patch payload files;
- no branch-specific self-modifying workflow.

### Profile and planner

- profile schema versioned;
- profile checksum validated;
- plan schema versioned;
- plan checksum validated;
- deterministic identical outputs for identical inputs;
- model and batch signature included;
- backend and device identity included;
- retain-all baseline included;
- recompute baseline included;
- fixed-interval diagnostic baseline included;
- measured-budget schedule included;
- infeasible plans include explicit reason;
- activation, workspace, and replay constraints enforced.

### Runtime

- selected anchors retained during forward;
- non-anchor boundaries released;
- nearest-anchor replay used during backward;
- replay groups and bytes measured;
- activation and workspace budgets enforced;
- tied gradients accumulate twice;
- tied parameter updates once;
- candidate restore exact;
- complete state within numerical band of both accepted policies.

### Persistence

- profile and plan identity bound to root metadata;
- process restart and resume pass;
- changed plan rejects resume;
- changed profile rejects resume;
- changed budget rejects resume;
- pruning preserves current plan identity;
- pruning followed by resume passes;
- both root publication failure points preserve previous authority.

### Apple M2

For micro, tiny, and small:

```text
0 < hybrid retained boundary bytes < retain_all
0 < hybrid replay groups < recompute
hybrid logical parameter rereads < recompute
state comparison GREEN or declared acceptable YELLOW
provenance equal
candidate restore exact
```

Raw RSS, MPS, Metal-driver, swap, memory pressure, timing, and storage results must be reported.

A physical-memory win is not required for M6C completion. A real logical Pareto intermediate point is required.

## 5. Done for the first larger-than-memory proof

The original idea reaches its first meaningful completion when one accepted target run demonstrates all of these properties.

### Capacity

- complete managed training state is larger than the declared safe resident-memory limit;
- the safe limit is justified from target measurements;
- no complete state is materialized inside the claimed capacity path;
- each active parameter group or tile fits its parameter budget;
- each gradient group or tile fits its gradient budget;
- each optimizer group or tile fits its optimizer budget;
- retained activations and local workspace fit their budgets;
- physical-pressure policy is respected or the run fails explicitly.

### Training

- full-parameter updates, not adapters or frozen weights;
- real text or a declared real dataset;
- more than one optimizer step;
- state at step N consumes committed state from step N-1;
- finite loss and gradients;
- useful training and validation trajectory;
- correct tied-weight semantics.

### Persistence

- authoritative root bundle at every committed step;
- exact candidate restore;
- interruption before publication preserves previous root;
- process restart resumes from the last committed root;
- resumed and uninterrupted trajectories match within a declared band;
- dataset cursor and batch identity remain aligned;
- pruning or retention policy is defined for the long run.

### Evidence

- exact commit and configuration;
- total managed state calculation;
- working-set maxima;
- RSS, MPS, Metal, pressure, and swap;
- storage reads and writes;
- chunk reuse and write amplification;
- elapsed time and per-step time;
- numerical comparison or sampled reference comparison;
- fallback scan;
- source integrity;
- packaged reproducible artifacts.

### Claim wording

After this gate, a defensible claim is:

> MicroColossus demonstrated multi-step, full-parameter, storage-backed training of a controlled model whose complete managed state exceeded the declared safe resident-memory limit on an 8 GB Apple Silicon Mac, while active working sets remained within declared budgets and interruption recovery and resume were validated.

Do not claim unlimited model size or production readiness.

## 6. Done for a practical M2 runtime

Beyond the first capacity proof, practical completion requires:

- acceptable elapsed time for the intended use;
- asynchronous overlap or other measured throughput improvements;
- storage growth controlled by retention;
- endurance costs characterized;
- stable physical-pressure behavior;
- automatic plan selection from target measurements;
- reduced validation-only overhead;
- clear operator and model support boundary;
- usable CLI and configuration diagnostics;
- repeatable runs over longer trajectories.

## 7. Done for production readiness

Production readiness is a separate, much larger goal. It would require:

- broader model and operator support;
- representative tokenizer and dataset pipeline;
- security review;
- compatibility and migration policy;
- long-run fault testing;
- observability suitable for users;
- documentation and support expectations;
- performance across multiple Apple Silicon generations;
- stable APIs;
- release packaging;
- data-loss and corruption risk assessment.

The current project is not close to this level and should not be described as production-ready.

## 8. Project status at handoff

| Completion level | Status |
|---|---|
| Controlled storage-backed multi-step reference | Complete and target-validated |
| Real-text learning and resume | Complete and target-validated |
| Safe pruning | Complete and target-validated |
| Full-prefix activation recomputation | Complete and target-validated |
| Measured hybrid anchors | Not accepted |
| Validation path without full-state materialization | Not started |
| Physical-memory controller | Not started |
| Intra-layer tiling | Not started |
| Larger-than-memory proof | Not demonstrated |
| Practical optimized M2 runtime | Not complete |
| Production readiness | Not a current claim |
