# Safe Checkpoint Pruning and Compaction

This document records the MicroColossus 0.11 design for bounding persistent training-root growth without changing the authoritative `CURRENT` checkpoint or weakening resume, integrity, and recovery semantics.

## 1. Motivation

MicroColossus 0.10 validated real-text training and process resume on an 8 GB Apple M2. The approximately 1.85M-parameter ten-step run retained about 632 MB because every historical candidate store, optimizer store, gradient store, oracle artifact, and work directory remained on disk.

That retention policy is useful while debugging. It is not sustainable for hundreds or thousands of steps.

Version 0.11 therefore introduces an explicit pruning layer before activation offload, larger corpora, or model-capacity experiments.

## 2. Scope

The first pruning implementation supports:

- deterministic dry-run planning;
- a checksummed plan that identifies exact deletion targets;
- retention of `CURRENT` plus a declared number of prior checkpoints;
- optional milestone checkpoints at a fixed step interval;
- complete root-manifest lineage retention;
- deletion of child stores belonging only to unretained checkpoints;
- removal of unreferenced candidate, work, oracle, temporary, and unpublished state;
- verification of every retained root and child store before and after deletion;
- atomic operation journals;
- interruption recovery and idempotent repeated apply;
- explicit plan and apply CLI commands;
- application-level byte accounting.

The first implementation does not repack live chunks, deduplicate independent stores, use filesystem clone APIs, or prune automatically in the background.

## 3. Retention policy

Configuration:

```yaml
retention:
  keep_previous: 2
  milestone_interval: 5
```

The retained set contains:

1. the current checkpoint;
2. the configured number of immediately preceding checkpoints;
3. every checkpoint whose step is divisible by `milestone_interval`, when the interval is greater than zero.

For current step 20, `keep_previous: 2`, and `milestone_interval: 5`, the materialized retained checkpoints are:

```text
0, 5, 10, 15, 18, 19, 20
```

The retention configuration is operational policy, not training semantics. It may change between invocations without changing the model, optimizer, data, or schedule digest.

## 4. Manifest lineage versus restorable checkpoints

MicroColossus keeps every root bundle manifest in the authoritative lineage:

```text
step 0 -> step 1 -> ... -> step N
```

Root manifests are small and preserve:

- parent bundle IDs;
- committed steps;
- batch checksums;
- parameter, optimizer, and gradient references;
- bundle checksums;
- progress-record alignment.

After pruning, only checkpoints selected by the retention policy keep their referenced child stores. Older root manifests may still describe historical checkpoints whose tensor stores have been reclaimed.

Therefore:

- complete training lineage remains inspectable;
- `CURRENT` and selected retained checkpoints remain restorable;
- an already pruned checkpoint cannot be resurrected by changing the future retention policy;
- historical progress metrics remain available unless a future independent metrics policy removes them.

## 5. Plan contract

`microcolossus-prune plan` is non-mutating.

The plan records:

```text
schema version
CURRENT bundle ID and checksum
CURRENT pointer-file checksum
current committed step
retention policy
complete lineage IDs and steps
retained bundle IDs and steps
retained child-store paths
historical checkpoints losing child state
exact deletion paths
category, byte count, file count, and content digest per deletion path
managed bytes before pruning
selected bytes
plan checksum
```

Every deletion target is recursively inventoried. File content, path, and byte counts participate in the path digest. The plan is accepted only if its checksum validates.

A repeated dry run against unchanged state produces the same logical plan.

## 6. Apply preconditions

Before mutation, apply requires:

1. valid training metadata matching the supplied experiment configuration;
2. a valid checksummed pruning plan;
3. the same `CURRENT` bundle ID, bundle checksum, committed step, and pointer bytes recorded by the plan;
4. valid contiguous root lineage;
5. successful verification of every retained root and every retained child store;
6. a filesystem inventory matching the plan exactly;
7. exclusive ownership of the pruning lock.

A mismatch aborts before deletion.

## 7. Protected paths

The implementation never selects these root-level objects for deletion:

```text
CURRENT
TRAINING.json
metrics/
pruning/
root manifests in the committed lineage
```

The protected child-store set is the union of the parameter, optimizer, and optional gradient paths referenced by retained root bundles.

All ancestors needed to reach those paths are preserved.

## 8. Reclaimable paths

The first implementation may reclaim:

- unprotected subtrees under `candidates/`;
- unprotected subtrees under `work/`;
- candidate parameter and optimizer stores for unretained steps;
- gradient stores for unretained steps;
- validation-only oracle state;
- interrupted or unpublished candidate state;
- unreferenced corrupt orphan directories;
- unpublished root bundle manifests;
- safe temporary files outside protected state.

Only paths under an explicit allow-list may be deleted. Absolute paths, parent traversal, symbolic links, special files, and protected root metadata are rejected.

## 9. Atomic operation journal

Each apply uses:

```text
training-root/
  pruning/
    LOCK
    operations/
      <plan-checksum>.json
```

Journal states:

```text
prepared -> deleting -> completed
```

The journal is written through:

```text
temporary file
  -> file fsync
  -> atomic rename
  -> directory fsync
```

After each deletion, the completed path set and cumulative reclaimed bytes are published atomically.

## 10. Interruption recovery

If a process exits after deleting a path but before publishing the updated journal, the next apply observes that the planned path is already missing and records it as completed.

This is safe because:

- the plan already proved the path unreachable from every retained checkpoint;
- the plan stores the expected path inventory and byte count;
- `CURRENT` remains unchanged;
- retained state is verified again before completion.

A completed apply can be run again. It returns an idempotent report with zero newly reclaimed bytes.

## 11. Concurrency policy

Pruning creates an exclusive root lock. Persistent bounded training refuses to resume while that lock is active.

A stale lock from a terminated pruning process is reclaimed only by a subsequent pruning invocation after the recorded owner process is no longer alive.

The current implementation is intended for one training or pruning coordinator per root. Distributed writers are out of scope.

## 12. `CURRENT` invariant

Pruning never rewrites `CURRENT`.

The plan stores the checksum of the exact `CURRENT` file bytes. Apply checks this value before mutation, during deletion, and in the final report.

The core invariant is:

```text
CURRENT before pruning == CURRENT after pruning
```

A different current step requires a new plan.

## 13. Verification after apply

After all selected paths are removed, MicroColossus:

1. reopens `CURRENT`;
2. verifies the current bundle;
3. verifies every retained historical checkpoint;
4. verifies every retained parameter, optimizer, and gradient child store;
5. runs root recovery;
6. confirms `CURRENT` remains byte-identical;
7. records managed bytes remaining and cumulative reclaimed bytes;
8. marks the pruning operation completed.

## 14. Resume after pruning

Training resumes from the parameter and Adam stores referenced by `CURRENT`.

Because root manifests and progress records remain contiguous, the normal resume checks still validate:

- model and training configuration;
- data identity;
- committed cursor;
- root lineage;
- progress-record bundle IDs;
- optimizer step tensors.

CPU tests compare a pruned trajectory against an otherwise equivalent unpruned trajectory and require exact final canonical state.

## 15. CLI

Create a dry-run plan:

```bash
microcolossus-prune plan \
  --config examples/real-text-micro.yaml \
  --bundle-store runs/real-text-training \
  --output runs/pruning-plan.json
```

Override the configured policy:

```bash
microcolossus-prune plan \
  --config examples/real-text-micro.yaml \
  --bundle-store runs/real-text-training \
  --keep-previous 1 \
  --milestone-interval 10 \
  --output runs/pruning-plan.json
```

Inspect the plan before applying it. Apply is always explicit:

```bash
microcolossus-prune apply \
  --config examples/real-text-micro.yaml \
  --bundle-store runs/real-text-training \
  --plan runs/pruning-plan.json \
  --output runs/pruning-report.json
```

There is no automatic deletion during training in version 0.11.

## 16. Failure injection

The first pruning layer exposes these test points:

```text
before journal rename
before first deletion
after one path deletion
```

The validation suite requires:

- failure before journal publication deletes nothing;
- failure before the first deletion leaves all state unchanged;
- failure after deletion preserves every retained checkpoint;
- rerunning apply completes the same plan;
- `CURRENT` remains unchanged in every case.

## 17. Corruption policy

Corruption in retained state aborts pruning before deletion.

Corruption in an unreferenced orphan path does not prevent safe removal when the orphan is not needed to prove retained reachability.

The implementation does not claim to repair corrupted authoritative checkpoints.

## 18. Telemetry and endurance

Reports distinguish:

- managed bytes before pruning;
- bytes selected by the plan;
- cumulative bytes reclaimed for the operation;
- newly reclaimed bytes in the current invocation;
- managed bytes remaining;
- exact paths deleted or already absent after interruption.

These are application-level file-content measurements. They are not NAND-level writes or guaranteed filesystem block reclamation. APFS copy-on-write behavior, sparse allocation, metadata amplification, snapshots, compression, and SSD-controller behavior remain outside these counters.

## 19. Current validation status

Version 0.11 includes CPU unit, integration, CLI, interruption, corruption, idempotence, and resume-equivalence tests. Python 3.11 and 3.13 CI must pass before merge.

The milestone remains open until a clean Apple M2 filesystem validation demonstrates:

- deterministic plan output;
- material storage reclamation on a real training root;
- byte-identical `CURRENT`;
- retained-checkpoint verification;
- interrupted-apply recovery;
- process resume after pruning;
- clean source state.

## 20. Boundary

Version 0.11 does not yet provide:

- live chunk repacking across retained stores;
- content deduplication across store directories;
- automatic periodic pruning during training;
- background storage-pressure monitoring;
- remote or object storage;
- filesystem-specific reflinks;
- activation offload or recomputation;
- asynchronous prefetch or writeback;
- training state larger than unified memory.
