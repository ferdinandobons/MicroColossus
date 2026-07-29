# Architecture and non-negotiable invariants

This file describes the architecture that a successor must preserve while adding new capabilities.

## 1. Controlled model boundary

MicroColossus currently targets a controlled decoder-only Transformer rather than arbitrary graphs.

Ordered execution groups are:

```text
embedding
block-0
block-1
...
final-head
```

This constraint makes tensor identity, dependency order, shared-weight behavior, replay, failure injection, state comparison, and recovery tractable.

A successor may generalize the model later, but must not break the controlled reference path while doing so.

## 2. Canonical tensor state

Every managed tensor must have explicit metadata:

```text
logical name
kind
shape
dtype
byte order
version
ordered chunk IDs
byte length
checksum
committed step
backend adapter metadata
```

Tensor bytes are stored in immutable content-addressed chunks. Tensor versions are represented by manifests. New state uses copy-on-write semantics.

### Invariants

1. A tensor checksum covers canonical bytes, not framework object identity.
2. A manifest references immutable chunks in a deterministic order.
3. A version is not authoritative until its owning transaction commits.
4. Corrupt or missing referenced chunks are detected before publication or resume.
5. Backend adapters must round-trip through the canonical representation.

## 3. Tensor-store transactions

The storage layer separates staging from authority.

Conceptual transaction flow:

```text
prepared
    -> writing
    -> validated
    -> committed
```

Failed or incomplete transactions are aborted or recovered conservatively.

### Invariants

- a failed transaction cannot replace `CURRENT`;
- recovery never guesses that incomplete state is valid;
- telemetry is derived evidence and must not alter tensor authority;
- read telemetry may grow after planning, but immutable state identity must remain stable;
- destructive operations must verify exact targets before deletion.

## 4. Root step bundles

A root bundle is the authoritative training checkpoint. It references:

- parameter-store manifest;
- optimizer-store manifest;
- optional gradient-store manifest;
- committed step;
- consumed batch checksum;
- parent bundle;
- root checksum.

Conceptual publication:

```text
build candidate child stores
    -> verify child stores
    -> write and fsync root manifest
    -> write candidate CURRENT
    -> fsync
    -> atomic CURRENT rename
```

### Invariants

1. `CURRENT` identifies the only authoritative root.
2. Candidate child stores are not authoritative by themselves.
3. Failure before final pointer replacement leaves the previous root authoritative.
4. A published root must reference verified child stores.
5. The committed root step is the authoritative next-batch cursor.
6. Parent lineage must be contiguous.
7. Model state, optimizer state, consumed batch, and data cursor advance together.

## 5. Bounded forward

For each execution group:

```text
read parameter records
    -> materialize active tensors
    -> execute local forward
    -> record boundary or output evidence
    -> release group tensors
```

The tied token embedding is read once for the embedding path and again for the output projection when needed.

### Invariants

- no execution group may exceed the declared parameter budget;
- group order is deterministic;
- outputs match the resident reference within the declared numerical band;
- releasing a group must not invalidate required activation or anchor state;
- hidden full-model materialization is not allowed inside the claimed bounded path.

## 6. Bounded backward

Backward executes in reverse group order:

```text
final-head
last block
...
block-0
embedding
```

Final gradients are published into a versioned gradient store.

### Shared-weight invariant

The tied token embedding receives two gradient contributions:

1. the output-head contribution;
2. the embedding-input contribution.

These contributions must accumulate into one final gradient tensor. The optimizer must update that parameter exactly once.

### Other invariants

- one backward group at a time must respect parameter, gradient, activation, and workspace budgets;
- final gradient names and structures match the reference;
- gradient values remain finite;
- every final gradient tensor has a deterministic version and checksum;
- an incomplete backward must not publish a new root.

## 7. Streamed global norm and clipping

The global gradient norm is computed from final gradients, one tensor at a time.

The canonical clipping coefficient is:

```text
min(1, max_norm / (global_norm + 1e-6))
```

### Invariants

- clipping uses the complete final gradient state;
- tied-gradient accumulation occurs before norm calculation;
- the same coefficient is used by all optimizer groups;
- no hidden resident concatenation is part of the claimed streamed path.

## 8. Group-bounded AdamW

For one unique optimizer group at a time, the runtime reads:

- parameters;
- gradients;
- first moments;
- second moments;
- step tensors;
- parameter-group metadata.

It writes new candidate parameter and optimizer versions.

### Invariants

1. each unique parameter is updated exactly once;
2. Adam step values advance monotonically and match the committed step;
3. candidate state is restorable exactly;
4. candidate state matches the resident oracle within the declared numerical band;
5. optimizer working-set accounting includes all local parameter, gradient, moment, and step bytes;
6. no candidate state becomes authoritative before root publication.

## 9. Data identity and cursor authority

The real-text frontend currently uses:

```text
tokenizer: utf8-bytes-v1
vocabulary: 256
sampler: deterministic random window
```

Data identity contains:

- source kind;
- tokenizer version;
- sampler or batch-stream version;
- split policy;
- train checksum and byte count;
- validation checksum and byte count;
- combined identity checksum.

### Invariants

- file path alone is not data identity;
- corpus bytes and declared semantics are both verified;
- batch cursor is derived from committed root step;
- batch seed, byte offsets, and checksum are recorded;
- a changed corpus or data policy rejects resume before consuming a new authoritative batch;
- validation and samples are derived evidence, not training authority.

## 10. Retention and pruning

Pruning is a two-stage explicit operation.

```text
dry-run plan
    -> verify CURRENT
    -> resolve retained checkpoints
    -> inventory exact deletion targets
    -> checksum the plan

apply
    -> verify immutable authority and target inventory
    -> publish operation journal
    -> delete selected unreachable paths
    -> verify retained state
    -> preserve CURRENT byte-for-byte
```

### Invariants

1. planning is non-mutating;
2. training never silently deletes history;
3. `CURRENT` is not rewritten during pruning;
4. retained checkpoints remain restorable and verified;
5. root manifests may remain as lightweight lineage after child state is pruned;
6. repeated apply is idempotent;
7. interruption can continue from the journal;
8. a changed deletion target rejects apply;
9. corruption in retained state blocks deletion;
10. pruning and training on one root are mutually exclusive.

## 11. Activation policies

### `retain_all`

Every non-final forward boundary is retained until its reverse group executes.

Advantages:

- minimal replay;
- simple backward reconstruction.

Costs:

- retained activation memory grows with depth, sequence length, microbatch, hidden width, and dtype.

### `recompute`

No forward boundary is retained for later backward use. Each reverse group reconstructs its input from token IDs and the authoritative parameter store.

Advantages:

- zero retained forward-boundary bytes.

Costs:

- full-prefix replay;
- logical parameter rereads;
- potentially quadratic replay with depth;
- physical RSS may increase even when logical retained bytes decrease.

### Future `hybrid`

A measured hybrid policy should retain selected anchors and reconstruct each backward input from its nearest preceding anchor.

Required identity:

- measurement-profile schema and checksum;
- planner version;
- backend and device identity;
- model and batch-shape signature;
- budgets and constraints;
- selected anchors;
- replay segments;
- plan checksum.

### Hybrid invariants

1. identical inputs produce the same plan;
2. each anchor precedes the target it serves;
3. final-head is never an activation anchor;
4. anchor plus local gradient residency respects the activation budget;
5. local replay and backward workspace respect the workspace budget;
6. a rejected plan publishes no root;
7. plan and profile checksums are checkpoint identity;
8. resume with a changed plan is rejected;
9. pruning does not remove the current plan identity;
10. hybrid state matches both accepted extremes within the numerical band.

## 12. Validation-only materialization

The current development runtime still materializes complete state for some checks:

- resident oracle;
- complete candidate comparison;
- full restore verification;
- validation;
- generation.

These paths are valuable during development but are excluded from larger-than-memory claims.

A future capacity mode must explicitly disable, sample, stream, or externalize these checks without weakening checkpoint integrity.

## 13. Backend separation

### PyTorch MPS

Role:

- numerical oracle;
- primary storage-backed reference;
- debugging and recovery semantics;
- target-hardware correctness.

### MLX

Role:

- optimized Apple Silicon execution candidate;
- currently validated mainly for resident benchmark and state round-trip.

Do not assume that a PyTorch bounded feature automatically exists in MLX.

## 14. Observability contract

Every meaningful run should report, where applicable:

- exact commit and package version;
- device and backend;
- parameter count;
- working-set budgets and observed maxima;
- loss, validation loss, gradient norm, and clipping coefficient;
- state-comparison distances;
- batch cursor, seed, offsets, and checksum;
- root lineage and optimizer steps;
- candidate restore exactness;
- bytes read and written;
- chunk reads, writes, and reuse;
- fsync and publication time;
- replay groups and parameter rereads;
- RSS;
- MPS or MLX allocator counters;
- Metal-driver allocation;
- memory pressure and swap;
- fallback and non-finite scan;
- final Git integrity.

Observability must not be confused with authority. Metrics may be recreated. `CURRENT` and its referenced stores define training state.
