# Open work and ordered roadmap

This roadmap is ordered by dependency. A successor should not skip directly to a large model because each later stage depends on correctness and observability established earlier.

## Phase 0. Stabilize the repository

Status: **not completed**.

### Required actions

1. restore the accepted CI matrix from commit `9f9365e597693e2cffa4f454180203e7219a7cde`;
2. remove `.github/workflows/apply-m6c-clean-once.yml`;
3. remove branch-specific diagnostic logic from `.github/workflows/ci.yml`;
4. close PR `#31` and PR `#33` as abandoned experiments;
5. delete or ignore temporary payload branches after preserving any useful patch notes;
6. confirm package version is `0.12.0` on the stabilized baseline;
7. run import, Ruff, mypy, pytest, compileall, and CPU smoke on Python 3.11 and 3.13;
8. verify no runtime source differs unintentionally from the accepted baseline;
9. create a clean M6C branch through normal Git commits.

### Acceptance

```text
package import: PASS
ruff:          PASS
mypy:          PASS
pytest:        PASS
compileall:    PASS
CPU smoke:     PASS
normal CI:     restored
transfer files: absent
privileged one-shot workflows: absent
```

Do not request another Mac execution before this phase passes.

## Phase 1. Reimplement M6C cleanly

Status: **specified, failed draft exists, accepted implementation absent**.

Issue `#27` is the requirements document.

### 1.1 Measurement profile

Implement a versioned profile with:

- schema version;
- planner version;
- backend and device identity;
- model signature;
- sequence length;
- microbatch;
- dtype;
- ordered group names;
- parameter bytes per group;
- boundary bytes per group;
- local workspace bytes;
- measured parameter-read time;
- measured materialization time;
- measured compute time;
- measured release time where useful;
- source result checksums;
- canonical profile checksum.

The profile must be deterministic when built from identical inputs.

### 1.2 Baseline schedules

The planner must produce comparable summaries for:

- `retain_all`;
- full-prefix `recompute`;
- fixed-interval anchors, diagnostic only;
- measured-budget dynamic anchors.

Each summary must include:

- anchor names;
- retained anchor bytes;
- maximum replay depth;
- total replayed groups;
- logical parameter rereads;
- estimated replay time;
- activation-budget status;
- workspace-budget status.

### 1.3 Dynamic planner

The selected planner must optimize a declared objective, not an undocumented heuristic.

Recommended first objective:

```text
minimize estimated replay cost
subject to:
    retained anchor bytes + local activation-gradient residency <= activation budget
    local workspace <= workspace budget
    maximum replay depth <= optional limit
```

For small group counts, enumerate feasible anchor sets exactly. For deeper models, use dynamic programming, shortest-path formulation, or a documented deterministic approximation.

A greedy fallback may exist, but it must be identified in the plan and tested separately.

### 1.4 Plan format

The plan must contain:

- schema and planner version;
- profile checksum;
- model signature;
- all planning inputs;
- selected anchor groups;
- ordered replay segments;
- maximum retained bytes;
- maximum workspace;
- maximum replay depth;
- total replayed groups;
- logical parameter rereads;
- estimated replay time;
- baseline summaries;
- feasibility flag;
- explicit rejection reason;
- canonical plan checksum.

Rejected plans should be serializable for diagnostics and must not start training.

### 1.5 Runtime

Implement nearest-anchor execution:

```text
forward:
    retain only selected boundary anchors

backward target group:
    load nearest preceding anchor
    replay only groups between anchor and target
    execute local backward
    release replay state
```

Required runtime details:

- anchors stored in a clearly declared tier;
- explicit activation accounting;
- explicit workspace accounting;
- parameter group release after replay;
- correct tied-gradient accumulation;
- unique tied-parameter AdamW update;
- replay chunk-read and byte telemetry;
- measured replay time;
- no final-head anchor;
- no hidden resident model inside the claimed hybrid path.

### 1.6 Persistent identity

The root training metadata must bind:

- `activation_policy: hybrid`;
- profile checksum;
- plan checksum;
- planner version;
- anchor schedule;
- activation budget;
- workspace budget;
- replay constraint;
- model and data identity.

A changed profile, plan, budget, or policy must reject resume before consuming a new batch.

### 1.7 CPU tests

Required tests:

- deterministic profile round-trip;
- deterministic plan round-trip;
- checksum mutation detection;
- model-signature mismatch rejection;
- activation-budget infeasibility;
- workspace-budget infeasibility;
- replay-depth infeasibility;
- anchor order validation;
- final-head anchor rejection;
- exact controlled state against `retain_all` and `recompute`;
- candidate restore exactness;
- process restart and resume;
- plan-identity rejection;
- pruning followed by hybrid resume;
- publication failure before manifest rename;
- publication failure before `CURRENT` rename;
- tied-gradient count 2;
- tied update count 1;
- Python 3.11 and 3.13 CI.

### M6C CPU acceptance

No M2 prompt should be issued until:

```text
clean source diff
package 0.13.0
public planner API
installed planner CLI
Ruff PASS
mypy PASS
pytest PASS
compileall PASS
CPU smoke PASS
normal CI PASS on 3.11 and 3.13
no temporary workflow or payload file
```

## Phase 2. M6C Apple M2 gate

Status: **blocked by Phase 1**.

Compare `retain_all`, `recompute`, and `hybrid` on:

- real-text micro, 18,624 parameters;
- tiny, 443,648 parameters;
- real-text small, 1,846,656 parameters.

### Required evidence

- profile and plan determinism;
- selected anchors;
- retained bytes by policy;
- maximum replay depth;
- total replayed groups;
- logical parameter rereads;
- estimated and measured replay time;
- complete parameter plus Adam state comparison;
- loss, validation loss, gradient norm, and clipping comparison;
- batch provenance;
- candidate restore;
- process resume;
- plan mismatch rejection;
- pruning and resume;
- publication failure preservation;
- RSS;
- MPS current allocation;
- Metal-driver allocation;
- swap and memory pressure;
- storage traffic;
- fallback and non-finite audit;
- clean Git state.

### Acceptance

For each workload:

```text
0 < hybrid retained bytes < retain_all retained bytes
0 < hybrid replay groups < recompute replay groups
hybrid parameter rereads < recompute parameter rereads
state comparison within declared numerical band
resume and pruning PASS
```

A lower RSS is preferred but not mandatory. Raw physical measurements must be reported even when noisy or unfavorable.

## Phase 3. Remove validation-only full-state dependence

Status: **not started**.

The current development path materializes complete state for oracle and verification operations. A larger-than-memory claim requires a mode where these operations are disabled, sampled, streamed, or moved to a separate machine or phase.

### Required work

- introduce an explicit validation level, for example `full`, `sampled`, `streamed`, `none-with-integrity`;
- preserve checksum, manifest, root, and restore checks even when the resident oracle is disabled;
- add tensor sampling or group-level numerical checks;
- stream validation loss where possible;
- avoid full model materialization for generation in capacity mode;
- report exactly which checks were omitted;
- keep full validation as the default for micro and small development tests.

### Acceptance

The capacity path must execute without allocating complete parameter plus optimizer state in one process, while still verifying authoritative storage and checkpoint integrity.

## Phase 4. Physical-memory controller

Status: **not started**.

Logical budgets alone do not bound physical unified-memory use.

### Required work

- sample process RSS;
- sample MPS current and driver allocations;
- sample memory pressure and swap;
- define a conservative safe-memory limit;
- pause, reduce prefetch, reduce anchors, or abort before uncontrolled pressure;
- distinguish transient spikes from steady-state residency;
- expose physical-pressure policy in configuration and checkpoint identity where it affects execution;
- test behavior near the safe limit.

### Acceptance

A run must either remain within the declared pressure policy or fail explicitly before corrupting or silently changing the training path.

## Phase 5. M7 asynchronous prefetch and writeback

Status: **planned**.

Correct synchronous behavior must remain the oracle.

### Candidate design

- bounded prefetch queue;
- double-buffered parameter-group staging;
- asynchronous candidate-store writes;
- explicit dependency barriers;
- queue memory included in budgets;
- cancellation and failure propagation;
- deterministic publication order;
- synchronous fallback mode.

### Required comparisons

- synchronous versus prefetch numerical state;
- throughput and first-step latency;
- storage read overlap;
- write overlap;
- RSS and memory pressure;
- crash during in-flight prefetch or writeback;
- SSD write amplification.

Do not add concurrency before state ownership and cancellation semantics are explicit.

## Phase 6. M8 intra-layer tiling

Status: **planned, likely required for 124M and 350M**.

Group-bounded execution fails when one group is itself larger than the budget.

### Tiling targets

- vocabulary embedding rows;
- output projection rows;
- attention QKV output channels;
- attention output channels;
- MLP expansion and contraction dimensions;
- optimizer updates over tensor slices.

### Required invariants

- tiled forward equals untiled forward;
- tiled backward accumulates exact final gradients;
- tiled AdamW equals untiled AdamW;
- tied embedding and output projection semantics remain correct;
- tile order is deterministic;
- partial tile state is never authoritative;
- failure during a tensor update preserves the previous tensor version;
- tile metadata is part of the execution plan.

### Acceptance

At least one model whose largest untiled group exceeds the configured budget must complete multiple correct optimizer steps using tiles.

## Phase 7. Storage-path optimization

Status: **partially implemented, optimization not started**.

Potential work:

- shared content-addressed chunk pool across checkpoint stores;
- deduplication across independent store directories;
- retained-store compaction;
- batched reads;
- larger sequential extents;
- reduced metadata and fsync overhead;
- storage-temperature or reuse-aware cache;
- optional APFS clone experiments;
- endurance modeling.

Do not claim physical NAND savings from logical byte counters alone.

## Phase 8. First larger-than-memory demonstration

Status: **not started**.

Start with the smallest model whose complete managed state exceeds the safe resident limit. Do not jump directly to 350M if a smaller configuration can prove the mechanism.

### Required protocol

1. calculate total managed state;
2. declare a safe resident limit below that total;
3. prove each active working set fits;
4. disable full resident validation materialization;
5. run several optimizer steps on real data;
6. interrupt and resume;
7. compare with a feasible reference at reduced scale or by group sampling;
8. report loss trajectory;
9. report storage reads, writes, reuse, and pruning;
10. report elapsed time, RSS, MPS, Metal, pressure, and swap;
11. verify no fallback or non-finite values;
12. preserve a clean repository and packaged artifacts.

### Scale ladder

Recommended order:

```text
5M to 20M controlled proof
    -> 50M class
    -> 124M
    -> 350M
```

The exact first size should be selected from measured group and tile limits, not marketing value.

## Phase 9. M9 bounded MLX execution

Status: **not started**.

After the PyTorch reference path is stable:

- map canonical stores to MLX tensors;
- implement bounded forward, backward, clipping, and AdamW;
- preserve checkpoint identity and recovery;
- compare numerical state with PyTorch;
- measure throughput and allocator behavior;
- decide whether MLX becomes the default Apple execution backend.

Do not duplicate storage, checkpoint, or data-provenance logic unnecessarily.

## Phase 10. Real training adapter and broader frontend

Status: **future**.

Potential scope:

- subword tokenizer identity;
- sharded datasets;
- epoch and shuffle state;
- streaming corpus adapter;
- external model definitions through a controlled interface;
- richer validation and sample generation;
- integration with the separate training project the project owner may provide.

Any adapter must keep dataset identity and cursor advancement bound to the authoritative checkpoint.

## Priority summary

The critical path is:

```text
stabilize repository
    -> clean M6C implementation
    -> M2 M6C gate
    -> remove full-state validation dependence
    -> physical-memory controller
    -> intra-layer tiling where required
    -> first larger-than-memory multi-step demonstration
```

Asynchronous I/O and bounded MLX are important for competitiveness, but they must not delay the first defensible capacity proof unless measured performance makes the proof impractical.
