# MicroColossus Project Specification

> **Trade memory for time.**

This is the canonical description of why MicroColossus exists, what it aims to build, what is implemented, and how future claims must be validated.

## 1. Purpose

MicroColossus is an experimental open-source runtime project for training generative models whose complete training state does not fit in GPU memory or within a strict host-RAM budget.

The project investigates whether explicit scheduling across VRAM, RAM, and NVMe can exchange part of the resident-memory requirement for additional data movement, recomputation, and elapsed time.

The central question is:

> How can a runtime execute a valid full-parameter update when the complete model state fits only in storage, while active computation remains inside strict VRAM and RAM budgets?

MicroColossus does not claim that hardware limits disappear. Storage capacity, minimum tile size, total compute, PCIe bandwidth, NVMe throughput, CPU performance, and SSD endurance remain physical constraints.

## 2. Why the project exists

Training requires more than model weights. A full step may also require:

- gradients;
- optimizer states;
- optional master-weight copies;
- saved activations or recomputation metadata;
- logits and loss intermediates;
- temporary tensors and kernel workspaces;
- allocator, framework, driver, and runtime overhead.

A mixed-precision AdamW layout is often approximated at roughly 16 bytes or more of persistent state per trainable parameter before activations. The exact value depends on dtype, optimizer, implementation, and layout. These values are illustrative, not universal limits.

| Parameters | Illustrative persistent state at 16 bytes per parameter |
|---:|---:|
| 124 million | about 2.0 GB |
| 350 million | about 5.6 GB |
| 1 billion | about 16 GB |
| 7 billion | about 112 GB |

A small computer may have enough storage for this state but not enough VRAM or process RAM to keep it resident. MicroColossus treats that mismatch as a scheduling problem whose full cost must be measured.

## 3. Central hypothesis

For a useful class of Transformer models and training recipes, part of resident-memory demand may be exchanged for:

- tensor streaming;
- bounded caching;
- activation recomputation;
- layer-wise execution;
- intra-layer tiling;
- asynchronous prefetch and writeback;
- additional elapsed time.

The basic capacity constraints are:

```text
model state + checkpoints + metadata <= allocated NVMe capacity

active parameter tiles + activations + workspaces + transfer buffers
    <= configured VRAM budget

RAM cache + staging buffers + runtime overhead
    <= configured process RAM budget
```

These inequalities are necessary but not sufficient. A viable plan must also account for bandwidth, latency, CPU work, write amplification, endurance, numerical behavior, and failure recovery.

## 4. Goals and boundaries

MicroColossus aims to:

1. Train models whose complete state exceeds both VRAM and the configured process-RAM budget.
2. Apply explicit limits to VRAM, RAM, NVMe capacity, and cumulative SSD writes.
3. Keep only the current working set in GPU memory.
4. Stream tensor chunks through bounded buffers.
5. Recompute selected activations instead of retaining them.
6. Tile a single operation when it cannot fit in VRAM as a whole.
7. Preserve a clearly identified full-parameter reference path.
8. Track the identity, version, location, and integrity of managed tensor chunks.
9. Avoid publishing partially completed optimizer steps.
10. Report time, I/O, memory use, stalls, and SSD writes.

It does not promise:

- literally infinite models;
- datacenter-class throughput;
- initial support for arbitrary PyTorch graphs;
- feasibility for every architecture or context length;
- elimination of total training compute;
- unmanaged use of swap or unmeasured page-cache behavior;
- that offloading, checkpointing, or tensor streaming are new ideas;
- success at scale targets before measured demonstrations exist.

A larger parameter count alone is not sufficient evidence of success.

## 5. Initial target and modes

The first constrained-hardware target is:

- one CUDA GPU with 8 GB of VRAM;
- 8 GB of installed system RAM;
- a process budget below installed RAM;
- one consumer NVMe SSD;
- no distributed cluster;
- a controlled decoder-only Transformer.

Planned modes are kept separate:

- **`reference`**: all parameters are trainable. No silent adapters, low-rank gradient projections, or quantized optimizer states.
- **`compact`**: future mode for declared approximations such as quantized optimizer states or low-rank methods.
- **`adapter`**: future LoRA or QLoRA path, reported separately from full-parameter training.

Only `reference` is currently accepted by the configuration loader.

## 6. Proposed architecture

```text
Dataset stream
      |
Controlled model frontend
      |
Budget-aware planner
      |
Execution schedule
      |
+----------------------------------+
| Runtime                          |
| - executor                       |
| - transfer engine                |
| - activation manager             |
| - optimizer engine               |
| - checkpoint coordinator         |
| - telemetry                      |
+----------------------------------+
      |
VRAM cache <-> RAM cache <-> NVMe tensor store
```

The intended components are:

- a controlled frontend with a deliberately small operator set;
- a tensor manifest with shape, dtype, version, location, and checksum;
- a planner for tiling, prefetch, eviction, recomputation, and placement;
- bounded VRAM and RAM caches;
- an asynchronous transfer engine;
- a chunked, versioned, recoverable NVMe tensor store;
- layer-wise and tile-wise forward, backward, and optimizer execution;
- telemetry for memory, transfers, stalls, throughput, and writes.

## 7. Intended execution model

### Forward

1. Prefetch the next layer or tile from NVMe to a bounded RAM buffer.
2. Transfer the active weights to VRAM.
3. Execute the GPU operation.
4. Keep, offload, or discard the activation according to the plan.
5. Release the working set before advancing.

### Backward

1. Reload the required weights.
2. Recover or recompute the corresponding activations.
3. Compute input and parameter gradients.
4. Accumulate or write back gradients by chunk.
5. Apply the optimizer update by tile.
6. Publish the new tensor version only after its data is durable.

### Intra-layer tiling

Layer streaming is insufficient when one matrix, embedding, output head, or attention operation is larger than VRAM. Planned tiled paths include:

- linear projections and MLPs;
- partitioned embeddings;
- output projection and cross-entropy without all logits resident at once;
- normalization and streaming reductions;
- block-wise attention;
- shared embedding/output weights;
- reproducible dropout during recomputation.

The first tiled operator should be a linear layer because its forward and gradient equations can be decomposed and validated directly.

## 8. Current implementation

The repository now contains the executable foundation, not the out-of-core runtime.

Implemented:

- typed YAML configuration;
- controlled decoder-only Transformer;
- deterministic synthetic next-token data;
- resident full-parameter AdamW training;
- process-RAM and CUDA peak-allocation telemetry;
- gradient norms and deterministic model-state checksums;
- atomic JSON and JSONL experiment artifacts;
- a static memory estimator;
- CLI, tests, packaging, and CI.

Not implemented:

- real RAM-to-VRAM layer streaming;
- NVMe-backed tensors or optimizer state;
- activation offloading or managed recomputation;
- asynchronous transfers;
- hard runtime budget enforcement;
- intra-layer tiling;
- tensor manifests, journals, or crash recovery;
- resident-versus-streamed numerical comparison;
- training state larger than resident memory.

Current commands:

```bash
microcolossus plan --config examples/tiny-resident.yaml
microcolossus train --config examples/tiny-resident.yaml
python -m pytest
```

The training command writes:

```text
runs/tiny-resident/
  resolved-config.json
  memory-plan.json
  steps.jsonl
  summary.json
```

The workload is synthetic. It validates infrastructure and repeatability, not model quality.

### Static planner limitations

The planner calculates parameter, gradient, and Adam-state sizes from the instantiated model. Activation, workspace, transfer-buffer, and streamed-working-set values are heuristics.

It does not yet model:

- allocator fragmentation;
- page cache and pinned-memory overhead;
- exact operator workspaces;
- transfer/compute overlap;
- PCIe or NVMe latency;
- SSD write amplification;
- runtime-measured tensor lifetimes.

Its output is a planning hypothesis, not a guarantee that a run will fit.

## 9. Validation contract

Correctness precedes scale.

The resident implementation is the numerical oracle. A future streamed path must use the same:

- initial parameters;
- batches;
- random seeds;
- optimizer configuration;
- data types;
- gradient-accumulation schedule.

Comparisons must include:

- forward loss;
- selected activations;
- gradients;
- optimizer states;
- parameters after each update;
- the trajectory across multiple steps.

Floating-point differences must be measured with documented tolerances. Approximate equality must not be described as exact equality.

A streamed run must also report configured budgets, measured VRAM, process RSS, transfer buffers, and every budget violation.

## 10. Storage and recovery requirements

The future tensor store should use:

- immutable chunks or copy-on-write versions;
- logical tensor versions;
- per-chunk checksums;
- a write-ahead journal;
- atomic manifest publication;
- incremental checkpoints;
- failure-injection tests.

A step is committed only after all required chunks and the new manifest are valid. Until then, the previous committed version remains authoritative.

Recovery must distinguish committed steps, unpublished complete writes, partial writes, corrupt chunks, and stale cache entries.

## 11. Roadmap

### M0. Executable foundation. Implemented

- package, configuration, and CLI;
- controlled model;
- resident training baseline;
- static planner;
- telemetry and artifacts;
- reproducibility tests and CI.

### M1. RAM-to-VRAM layer streaming. Next

- define canonical CPU-owned model state;
- execute one block at a time on the active device;
- add deterministic recomputation where needed;
- record transfers and residency transitions;
- compare loss, gradients, AdamW state, and updated parameters;
- measure and enforce a VRAM budget.

### M2. Versioned tensor store

- tensor/chunk identifiers and manifest;
- checksums and copy-on-write versions;
- journal and recovery tests;
- read/write telemetry.

### M3. Synchronous NVMe execution

- model and optimizer state in the tensor store;
- bounded RAM staging buffers;
- layer-wise forward and backward;
- atomic step publication.

### M4. Asynchronous overlap

- double buffering;
- prefetch and writeback;
- stall measurement;
- memory and write-budget rejection.

### M5. Intra-layer tiling

- tiled linear and MLP paths;
- partitioned embeddings;
- tiled output projection and loss;
- normalization and attention paths.

### M6. Constrained-hardware demonstration

- validate around 124 million parameters;
- attempt full-parameter training around 350 million parameters on the target class of hardware;
- investigate around 1 billion parameters only after the smaller result is correct and measurable.

These scale targets are research goals, not current capabilities or guaranteed outcomes.

## 12. Benchmark and success criteria

Planned comparisons include resident PyTorch, activation checkpointing, applicable CPU offload, DeepSpeed ZeRO-Offload and ZeRO-Infinity, and separate MicroColossus modes.

Core metrics are:

- peak VRAM and process RAM;
- tokens per second and seconds per step;
- GPU utilization and stall time;
- PCIe and NVMe bytes;
- SSD writes and write amplification;
- checkpoint and recovery time;
- numerical distance from the resident baseline;
- energy per million tokens when measurable.

The first meaningful streamed milestone requires:

1. configured VRAM and RAM budgets are respected;
2. complete state exceeds at least one resident tier;
3. the update is compared with the resident baseline;
4. every tensor version is traceable;
5. storage reads and writes are reported;
6. interruption recovers the last committed state;
7. the result is reproducible from a documented configuration.

## 13. Main risks

- **I/O domination:** the GPU may remain idle while tensors move.
- **SSD endurance:** optimizer state and gradients may create impractical write volume.
- **Hidden RAM use:** page cache, pinned buffers, Python, drivers, and fragmentation may violate the budget.
- **Untileable operations:** some kernels may have an irreducible or inefficient working set.
- **Numerical divergence:** recomputation and changed reduction order may alter training.
- **Complexity without utility:** a larger trainable model may still be too slow to be useful.

Every benchmark must report these costs instead of presenting model size in isolation.

## 14. Related work and proposed focus

MicroColossus builds on established work, including:

- [PyTorch activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html)
- [ZeRO-Offload](https://arxiv.org/abs/2101.06840)
- [ZeRO-Infinity](https://arxiv.org/abs/2104.07857)
- [QLoRA](https://arxiv.org/abs/2305.14314)
- [GaLore](https://arxiv.org/abs/2403.03507)
- [LoHan](https://arxiv.org/abs/2403.06504)

The project does not present those underlying techniques as original.

Its proposed focus is the combination of:

- very limited host RAM as well as limited VRAM;
- explicit budgets across every memory tier;
- NVMe as a versioned canonical tensor store;
- tiling inside individual layers;
- planning that includes elapsed time and SSD writes;
- telemetry in bytes per step and bytes per token;
- recovery integrated into execution;
- strict separation between reference and approximate modes.

This differentiation remains a hypothesis until it is implemented and compared experimentally.

## 15. Project statement

```text
less resident memory
        in exchange for
more I/O + more recomputation + more elapsed time
```

The intended contribution is a runtime that makes memory limits explicit, schedules around them where practical, preserves a measurable reference path, and reports the complete cost of doing so.
