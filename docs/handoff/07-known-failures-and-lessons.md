# Known failures, protocol mistakes, and engineering lessons

This file records failures that should influence future design and validation.

## 1. Optional dependency used as a runtime requirement

### Failure

An early diagnostic failed because checksum telemetry required NumPy even though NumPy was not installed in the base runtime environment.

### Lesson

- base runtime code must not depend silently on an optional development package;
- optional fast paths need a tested standard-library or PyTorch fallback;
- dependency extras must match actual import boundaries;
- a checksum or telemetry feature must not prevent training from completing.

## 2. MPS float64 incompatibility

### Failure

An early gradient-norm implementation created float64 tensors, which are unsupported in important MPS paths.

### Correction

Gradient sums were accumulated in float32 and moved to CPU scalars for the final reduction.

### Lesson

- do not assume CPU dtype behavior is portable to MPS;
- test dtype creation, reductions, and scalar conversions directly on target hardware;
- avoid implicit float64 in metrics and control flow;
- record numerical differences after a dtype correction.

## 3. MPS is not always bitwise reproducible

### Observation

Several MPS comparisons produced identical loss trajectories and very small final-state distances while checksums differed.

### Lesson

- bitwise identity is ideal but not a universal MPS requirement;
- use declared numerical bands;
- require exact provenance and exact restore even when arithmetic state is only numerically stable;
- report maximum and mean absolute difference, not only a checksum mismatch.

## 4. PyTorch and MLX are numerically close, not identical

### Observation

Resident MLX was materially faster in the controlled benchmark, while final-state differences remained small but non-zero.

### Lesson

- maintain PyTorch as the reference oracle until bounded MLX state semantics are independently proven;
- compare canonical portable state, not framework-native object layouts;
- do not treat a throughput win as a correctness proof;
- keep backend allocator counters separate.

## 5. Static type failures can coexist with runtime PASS

### Failure

An early competitive benchmark completed all runtime rounds while mypy still failed in benchmark and MLX adapter code.

### Lesson

- release quality and runtime experiment status are separate dimensions;
- a report may say runtime PASS with type-check caveat, but the release gate remains incomplete;
- fix typing before accepting the release, because adapter mismatches often reveal real interface ambiguity.

## 6. Synthetic micro and real-text micro have different parameter counts

### Failure

A target prompt expected 11,456 parameters for the real-text micro configuration. The actual count was 18,624.

### Cause

The real-text model uses vocabulary size 256 and a larger positional table. The synthetic micro uses a smaller vocabulary and table.

### Lesson

- derive expected parameter counts from the exact configuration;
- include regression tests for release example counts;
- never reuse a count based only on the word “micro”;
- classify protocol errors separately from runtime errors.

## 7. Pruning plan invalidated by benign telemetry growth

### Failure

The first 0.11 M2 apply rebuilt and byte-compared the entire plan after retained checkpoint reads. Those reads appended telemetry, changing managed-byte snapshots and causing a false stale-plan error.

### Correction

The apply path was changed to verify immutable authority and the exact deletion-target inventory rather than time-dependent telemetry totals.

### Lesson

- distinguish authoritative immutable state from derived mutable evidence;
- a plan should bind the properties that matter for safe deletion;
- benign reads of retained state must not invalidate a deletion plan;
- modifications inside a deletion target must still reject apply.

## 8. Wrong working-set budget in a validation prompt

### Failure

The first small pruning validation used a 1 MiB parameter budget even though the largest small execution group was 1,772,544 bytes.

### Lesson

- read the static plan before writing a target protocol;
- budgets must be exact, explicit, and model-specific;
- expected budget rejection is not a runtime defect;
- a corrected rerun should preserve the original failure evidence.

## 9. Full-prefix recomputation reduced logical boundaries but increased RSS

### Observation

For the 1.85M small workload:

```text
forward-boundary bytes:
retain_all 491,520
recompute        0

sampled peak RSS:
retain_all 444,071,936
recompute  533,528,576
```

### Lesson

- logical and physical memory are different optimization targets;
- maximal recomputation is not automatically optimal;
- replay can increase allocator activity, workspaces, and cache pressure;
- select anchor schedules from measured trade-offs;
- a correct unfavorable result should guide the next design rather than be hidden.

## 10. M6C draft failed before runtime

### Failure

The M2 diagnostic of commit `d98ac10a9861f16db305f416f3afa37cb905e5d6` stopped at the quality gate.

Observed:

- wrong import source for `run_bounded_backward_from_store`;
- package import failure;
- version still `0.12.0`;
- planner API not exported;
- console script absent;
- Ruff F401 failures;
- mypy failures;
- pytest collection failures;
- temporary workflow present.

### Lesson

- never send target hardware a commit that has not passed fresh package import and CPU CI;
- package surface, CLI installation, types, and tests are part of the feature;
- compileall alone is insufficient;
- stop target experiments at the quality gate.

## 11. Self-modifying CI and patch-payload branches are an anti-pattern

### Failure

Attempts to transfer the local M6C working tree used:

- base64 or compressed patch payload parts;
- one-shot workflows;
- workflows with branch write permissions;
- force-push from CI;
- branch-specific replacement of the normal CI matrix.

The result was a polluted `main`, temporary PRs, and no accepted implementation.

### Lesson

- product code must arrive as ordinary reviewable source commits;
- CI validates code, it does not synthesize the feature branch;
- `pull_request_target` with write access requires exceptional justification and must not be used for feature transfer;
- never replace normal CI with a temporary branch-specific job;
- if the available tool cannot publish a large patch safely, split the source into normal commits or use an authenticated local Git workflow;
- stop rather than accumulating more transfer infrastructure.

## 12. Diagnostics can create false positives

### Examples

- matching `nan` inside an ordinary word;
- interpreting “No thermal warning” as an error;
- treating an unset fallback variable as fallback evidence;
- scanning expected injected tracebacks as unexpected failures.

### Lesson

- parse JSON numbers for non-finite detection;
- use word boundaries for textual scans;
- maintain an explicit list of expected failures;
- classify each finding by scenario and command;
- preserve raw evidence and the normalized classification.

## 13. Validation prompts must not silently fix code

### Reason

A target harness that edits source makes the tested commit ambiguous.

### Rule

- product source stays unchanged;
- only external scripts may be corrected;
- original harness failure is retained;
- the rerun uses a fresh root;
- final Git status must be empty.

## 14. Storage capacity on the target Mac is limited

### Observation

Recent target runs often began with approximately 5 to 9 GB free and substantial existing swap usage.

### Lesson

- check free storage before installation;
- remove only the diagnostic virtual environment after packaging when allowed;
- never delete unrelated user data;
- use pruning between scenarios when the protocol permits;
- package evidence without large tensor chunks when checksums and manifests are sufficient;
- gate large scenarios on available space;
- do not run independent scenarios concurrently.

## 15. The resident oracle can invalidate a future capacity claim

### Current behavior

Development steps often materialize complete state for comparison and restore verification.

### Lesson

- the bounded execution path and validation-only path must be measured separately;
- a larger-than-memory mode needs an explicit validation policy;
- integrity checks must remain even when complete resident comparison is disabled;
- no capacity claim should include a hidden full-state oracle in the same process.

## 16. Small verified increments are faster than large unverified rewrites

### Evidence from the project

The successful releases isolated:

- forward;
- backward;
- clipping;
- AdamW;
- publication;
- resume;
- real data;
- pruning;
- recomputation.

The failed M6C effort attempted source integration, package surface, persistent runtime, planner, tests, release changes, and transfer machinery together.

### Lesson

For M6C and later work:

1. merge schema and deterministic planner tests;
2. merge controlled CPU nearest-anchor runtime;
3. integrate persistent identity and resume;
4. add pruning and failure tests;
5. add CLI and examples;
6. run complete CI;
7. request target validation.

Each step should leave a working repository.
