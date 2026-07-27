# MicroColossus Project Specification

> **Trade memory for time.**

## 1. Purpose and status

MicroColossus is an experimental open-source runtime for training generative models whose complete training state does not fit in the available GPU memory or the RAM budget of the host process.

The project explores whether explicit scheduling across VRAM, RAM, and NVMe storage can replace part of the resident-memory requirement with additional data movement, recomputation, and elapsed time.

This document defines the motivation, intended scope, system model, architecture, validation strategy, and development path.

> **Current status:** design specification only. The runtime has not been implemented, and no performance or model-scale result is claimed.

## 2. Why the project exists

The practical size of a trainable model is constrained by much more than the storage needed for its weights. A full training step may require:

- model weights used by forward and backward execution;
- gradients;
- optimizer states, such as Adam first and second moments;
- optional master-weight copies;
- saved activations or the information needed to recompute them;
- temporary tensors, logits, communication buffers, and kernel workspaces;
- runtime and allocator overhead.

A common mixed-precision AdamW configuration can require roughly 16 bytes or more of persistent state per trainable parameter before activations are counted. The exact amount depends on precision, optimizer, implementation, and memory layout. The following values are therefore illustrative, not universal limits.

| Parameter count | Illustrative persistent state at 16 bytes per parameter |
|---:|---:|
| 124 million | about 2.0 GB |
| 350 million | about 5.6 GB |
| 1 billion | about 16 GB |
| 7 billion | about 112 GB |

On a small machine, the full state may fit on an SSD while being far too large for VRAM or process RAM. Conventional execution often treats that mismatch as a hard capacity failure.

MicroColossus treats it as a scheduling problem.

The central research question is:

> How can the runtime execute a valid full-parameter update when the complete model state fits only in storage, while the active computation must remain within strict VRAM and RAM budgets?

The project is based on the view that memory capacity, data movement, recomputation, storage endurance, and elapsed time should be optimized together rather than considered separately.

## 3. Central hypothesis

For a useful class of Transformer models and training recipes, a meaningful part of resident-memory demand may be exchanged for:

- tensor streaming;
- bounded caching;
- activation recomputation;
- layer-wise execution;
- intra-layer tiling;
- asynchronous prefetch and writeback;
- additional training time.

The system should preserve a full-parameter reference path whose update semantics remain comparable with an ordinary resident implementation, within documented floating-point tolerances.

The basic capacity constraints are:

```text
model state + checkpoints + metadata <= allocated NVMe capacity

active parameter tiles + activations + workspaces + transfer buffers
    <= configured VRAM budget

RAM cache + staging buffers + runtime overhead
    <= configured process RAM budget
```

Meeting these inequalities is necessary but not sufficient. A viable runtime must also manage bandwidth, latency, CPU work, SSD writes, failure recovery, and numerical behavior.

## 4. What MicroColossus aims to achieve

The project aims to build a runtime that can:

1. Train models whose complete state exceeds both VRAM and the RAM budget of the process.
2. Apply hard limits to VRAM, RAM, NVMe capacity, and cumulative SSD writes.
3. Keep only the current working set in GPU memory.
4. Stream tensor chunks between NVMe, host RAM, and VRAM using bounded buffers.
5. Recompute selected activations instead of storing them.
6. Divide a single layer into smaller tiles when the complete layer cannot fit in VRAM.
7. Separate a reference full-parameter mode from approximate or parameter-efficient modes.
8. Record where every tensor version lives and when it is valid.
9. Recover from interrupted writes or process failure without silently publishing a partial training step.
10. Report the real cost of the strategy in time, I/O, memory, energy where measurable, and SSD endurance.

The initial hardware target is intentionally restrictive:

- one CUDA GPU with 8 GB of VRAM;
- 8 GB of installed system RAM;
- a process RAM budget lower than installed RAM, leaving space for the operating system and drivers;
- one consumer NVMe SSD;
- no requirement for a distributed cluster.

## 5. What the project does not promise

MicroColossus is not intended to provide:

- literally infinite model size;
- throughput comparable with a well-provisioned multi-GPU cluster;
- support for every PyTorch model and operator in the first implementation;
- automatic feasibility for every architecture or context length;
- a way to avoid the total compute cost of training;
- unmanaged reliance on swap or the operating system page cache;
- a claim that offloading, checkpointing, quantization, or tensor streaming are new ideas;
- guaranteed success at the proposed 350 million or 1 billion parameter milestones.

Storage capacity, minimum tile size, total floating-point work, PCIe bandwidth, NVMe throughput, CPU performance, and SSD endurance remain physical limits.

## 6. Design principles

### 6.1 Memory is an explicit hierarchy

VRAM, host RAM, and NVMe are separate tiers with different capacity, bandwidth, latency, and endurance characteristics. The runtime should model them explicitly.

### 6.2 Every tensor is accounted for

For every managed tensor or chunk, the runtime should know:

- logical identity;
- shape and data type;
- current version;
- authoritative location;
- cached locations;
- checksum or integrity metadata;
- next expected use;
- whether it may be discarded and recomputed.

### 6.3 Budgets are hard constraints

A plan that exceeds configured VRAM or RAM is invalid, even if it would probably succeed on a particular run. Memory safety should not depend on optimistic allocator behavior.

### 6.4 Reference and approximate execution remain distinct

A reference mode should prioritize complete parameter updates and documented numerical comparison. Quantized optimizer states, low-rank projections, gradient compression, LoRA, and similar methods belong to separately named modes.

### 6.5 Storage writes are part of the cost model

An approach that fits in memory but writes an impractical amount of data per step is not considered successful. Write amplification and estimated endurance consumption should be visible metrics.

### 6.6 Correctness precedes scale claims

The first important result is not the largest model. It is a runtime that can explain where each tensor lives, which version is valid, how it was produced, and how closely the update matches a resident baseline.

## 7. Planned training modes

### `reference`

All model parameters are trainable. The mode is intended to use full gradients and conventional optimizers such as AdamW or SGD, combined with activation checkpointing, offloading, and mathematically equivalent tiling where practical.

This mode must not silently introduce adapters, low-rank gradient projections, or quantized optimizer states.

### `compact`

This mode may use memory-reducing methods such as:

- quantized optimizer states;
- compressed gradients;
- reduced-rank optimizer methods;
- alternative optimizer representations;
- other explicitly documented approximations.

Every deviation from `reference` should be identified and benchmarked separately.

### `adapter`

A future mode for LoRA, QLoRA, and related parameter-efficient fine-tuning methods. Adapter training is useful, but it is not equivalent to full-parameter pretraining or full fine-tuning.

## 8. Proposed architecture

```text
Dataset stream
      |
Controlled model frontend
      |
Graph and tensor analysis
      |
Budget-aware planner
      |
Execution schedule
      |
+----------------------------------+
| Runtime                          |
|                                  |
|  Executor                        |
|  Transfer engine                 |
|  Activation manager              |
|  Optimizer engine                |
|  Checkpoint coordinator          |
|  Telemetry engine                |
+----------------------------------+
      |
VRAM cache <-> RAM cache <-> NVMe tensor store
```

### Controlled model frontend

The first implementation should support a deliberately limited decoder-only Transformer. A narrow frontend reduces ambiguity and makes numerical validation practical. Broader PyTorch integration can be added only after the core runtime is stable.

### Tensor manifest

The manifest is the control plane for tensor state. It records tensor identity, chunk layout, version, location, integrity information, and lifecycle metadata.

### Budget-aware planner

The planner selects:

- layer and tensor tile sizes;
- activation checkpoints;
- recomputation points;
- RAM and VRAM cache residency;
- prefetch distance;
- buffer count and size;
- optimizer placement;
- writeback timing.

A schedule is accepted only when its predicted peak usage remains within all configured budgets.

### Transfer engine

The transfer engine is responsible for bounded and asynchronous movement between storage, host memory, and device memory. It should use preallocated buffers and a strictly limited amount of pinned memory.

### Tensor store

The NVMe tensor store holds authoritative chunk versions, optimizer state, manifests, journals, and checkpoints. It should support integrity checks and recovery after interruption.

### Executor

The executor performs forward, backward, gradient accumulation, and optimizer work according to the schedule. Execution may be layer-wise or tile-wise.

### Telemetry engine

Telemetry should expose:

- peak allocated and reserved VRAM;
- process resident memory;
- page-cache effects where measurable;
- bytes transferred over PCIe;
- bytes read from and written to NVMe;
- transfer and compute overlap;
- GPU idle time;
- I/O stalls;
- write amplification;
- step time and token throughput;
- checkpoint and recovery time;
- numerical differences from the baseline.

## 9. Memory hierarchy

### VRAM

VRAM acts as the computation tier and fastest managed cache. It contains only the active parameter tiles, activations, gradients, and workspaces required by the current operation.

### Host RAM

Host RAM acts as a bounded cache and staging tier. It may hold prefetched chunks, evicted activations, transfer buffers, and limited optimizer data. The runtime must not assume that all model state can temporarily spill into RAM.

### NVMe

NVMe acts as the authoritative high-capacity tier for state that is not currently active. It may contain:

- model parameter chunks;
- gradient accumulation chunks;
- optimizer state chunks;
- activation spill data when selected by the planner;
- tensor manifests;
- write-ahead journals;
- incremental checkpoints.

The SSD is not treated as free memory. Its bandwidth, latency, and endurance are explicit constraints.

## 10. Execution model

### 10.1 Forward pass

For each layer or tile, the runtime is expected to:

1. Prefetch the required weights from NVMe into a bounded RAM buffer.
2. Transfer the active data from RAM into VRAM.
3. Execute the GPU operation.
4. Retain, offload, or discard the resulting activation according to the plan.
5. Release or evict data that is no longer part of the near-term working set.
6. Overlap the current computation with prefetch for the next operation when possible.

### 10.2 Backward pass

In reverse order, the runtime is expected to:

1. Reload the required weights or tiles.
2. Retrieve or recompute the associated activations.
3. Calculate input and parameter gradients.
4. Accumulate gradients in bounded chunks.
5. Run the optimizer update on the CPU or GPU according to the plan.
6. Write the next tensor version without invalidating the previous committed version.
7. Publish the completed step atomically after all required chunks are valid.

### 10.3 Gradient accumulation

Microbatches may be used to reduce activation memory. Gradient accumulation reduces the resident batch requirement, but it does not by itself solve the problem of model state that cannot fit in memory. The runtime therefore manages accumulated gradients using the same chunked hierarchy.

### 10.4 Activation policy

For each activation, the planner may select one of four actions:

- keep in VRAM;
- move to host RAM;
- write to NVMe;
- discard and recompute during backward execution.

The decision should account for size, next-use distance, recomputation cost, transfer cost, and available budgets.

## 11. Intra-layer tiling

Layer streaming alone handles models that are deep but still assumes that one layer fits in VRAM. A sufficiently wide layer, embedding table, output head, or attention operation may violate that assumption.

MicroColossus therefore plans to support tiling inside individual operations.

### Linear and MLP operations

Weight matrices can be divided by output features, input features, or two-dimensional blocks. Partial outputs and gradients must be accumulated without materializing the complete weight matrix in VRAM.

### Embeddings

Large embedding tables can be partitioned into chunks. The runtime must load only partitions referenced by the current token batch, while preserving correct gradient accumulation for repeated token IDs.

### Output head and loss

The runtime should avoid materializing all vocabulary logits at once when the output projection is too large. A tiled cross-entropy path may compute numerically stable reductions across vocabulary blocks.

### Attention

Attention should use blockwise execution and avoid storing unnecessary intermediate matrices. The implementation should distinguish standard efficient attention kernels from the separate problem of moving model state through the memory hierarchy.

### Normalization and reductions

Layer normalization and other reductions may require streaming accumulation of statistics across tiles. Numerical stability and operation ordering must be validated against the baseline.

### Shared weights and randomness

Weight tying between embeddings and the output head requires consistent version handling. Dropout and other random operations require reproducible random-number state during activation recomputation.

## 12. Planner and cost model

A representative configuration may look like this:

```yaml
hardware:
  vram_budget_gib: 7.0
  ram_budget_gib: 5.0
  nvme_budget_gib: 500
  ssd_write_budget_tb: 300

training:
  mode: reference
  micro_batch_size: 1
  sequence_length: 1024
  gradient_accumulation_steps: 8

planner:
  objective: min_step_time
  enforce_hard_budgets: true
  account_page_cache: true
```

The exact configuration format may change during implementation.

An idealized lower bound for step time is:

```text
step_time >= max(
    step_flops / sustained_gpu_throughput,
    pcie_bytes / sustained_pcie_bandwidth,
    nvme_bytes / sustained_nvme_bandwidth,
    optimizer_work / sustained_cpu_throughput
)
```

This expression assumes perfect overlap. Real execution will include synchronization, small-transfer overhead, dependency gaps, filesystem behavior, and other inefficiencies.

The planner should optimize more than step time. Candidate objectives include:

- minimum step time under hard memory limits;
- minimum SSD writes under a maximum step-time constraint;
- maximum model size under memory and endurance limits;
- minimum energy per token where reliable measurement is available.

The planner should reject schedules that fit capacity but exceed configured write or time limits.

## 13. Storage, consistency, and recovery

The tensor store should be chunked so that large tensors can be read, updated, and checksummed incrementally.

The planned consistency model includes:

- immutable or copy-on-write tensor chunks;
- logical tensor versions;
- a write-ahead journal;
- per-chunk integrity checks;
- atomic manifest publication;
- incremental checkpoints;
- failure-injection tests.

A training step is committed only after every required chunk and the manifest for the new logical version have been written successfully. Until that point, the previous committed version remains authoritative.

Recovery should be able to distinguish:

- fully committed steps;
- complete but unpublished writes;
- partial chunk writes;
- corrupt chunks;
- stale cache entries.

## 14. Validation strategy

The project should establish correctness before pursuing scale.

### Resident baseline

A small model must run in a conventional resident PyTorch implementation. The streamed implementation should use the same:

- initial parameters;
- input batches;
- random seeds;
- optimizer configuration;
- data types;
- gradient accumulation schedule.

### Numerical comparisons

Validation should compare:

- forward loss at each step;
- selected activations;
- parameter gradients;
- optimizer states;
- parameters after each update;
- final training trajectory.

Tolerance must be defined by operation and precision. Differences caused by changed floating-point reduction order should be measured rather than described as exact equivalence.

### Failure validation

Tests should terminate the process during:

- chunk writes;
- manifest publication;
- checkpoint creation;
- optimizer updates;
- cache eviction.

After restart, the runtime should recover the last valid committed state or report an integrity failure clearly.

## 15. Benchmark plan

Planned comparisons include:

- fully resident PyTorch;
- PyTorch with activation checkpointing;
- PyTorch with CPU offload where applicable;
- DeepSpeed ZeRO-Offload;
- DeepSpeed ZeRO-Infinity;
- MicroColossus `reference`;
- MicroColossus `compact`, reported separately.

Core metrics include:

- peak VRAM usage;
- peak process resident memory;
- tokens per second;
- seconds per training step;
- GPU utilization;
- I/O wait and GPU stall time;
- PCIe bytes per step;
- NVMe read and write bytes per step;
- write amplification;
- checkpoint and recovery time;
- numerical distance from the resident baseline;
- energy per million tokens, when measurement is available.

Benchmark reports must state hardware, software versions, model architecture, sequence length, batch configuration, precision, optimizer, storage configuration, and thermal or power limits.

## 16. Minimum viable implementation

### Phase 0. Simulator

Build a planner simulator without real training. It should model tensor sizes, dependencies, memory capacities, transfer bandwidth, latency, and SSD writes.

### Phase 1. Numerical baseline

Train a controlled decoder model around 124 million parameters in both resident and streamed forms. Compare losses, gradients, optimizer states, and parameters.

### Phase 2. Tensor store

Implement chunking, manifests, checksums, journaling, recovery, and I/O telemetry.

### Phase 3. Layer-wise streaming

Place weights and optimizer states on NVMe, enforce a bounded RAM cache, and overlap transfers through double buffering.

### Phase 4. Intra-layer tiling

Add tiled linear layers, MLPs, embeddings, output projection, loss, normalization, and attention paths.

### Phase 5. Constrained-hardware demonstration

Attempt full-parameter training around 350 million parameters on an 8 GB GPU, strongly limited process RAM, and one consumer NVMe SSD.

A model around 1 billion parameters is a stretch target to investigate after the smaller demonstration is correct and measurable.

## 17. Success criteria

The project reaches its first meaningful milestone when it can demonstrate all of the following:

1. A streamed training run completes without exceeding configured VRAM and RAM budgets.
2. The complete model state exceeds at least one resident-memory tier.
3. The update trajectory is numerically compared with a resident baseline.
4. Every managed tensor version is traceable through the manifest.
5. Storage reads, writes, and write amplification are reported.
6. A forced interruption can recover the last committed state.
7. The result is reproducible from a documented configuration.

Maximum parameter count alone is not a sufficient success metric.

## 18. Main technical risks

### I/O dominates execution

Streaming may leave the GPU idle for most of the step. Mitigation requires large sequential chunks, asynchronous transfers, prefetching, and realistic planner models.

### SSD endurance becomes unacceptable

Optimizer states and gradients can generate large write volumes. The runtime must reduce unnecessary writeback, use incremental updates where valid, and expose projected endurance use.

### Host RAM is consumed indirectly

Pinned buffers, filesystem cache, Python overhead, drivers, and allocator fragmentation can violate the intended budget. Measurement must include process RSS and relevant operating-system effects.

### A single operation cannot be tiled efficiently

Some operations may have a minimum useful working set or poor tile-level efficiency. The controlled frontend should initially support only operations with validated tiled paths.

### Numerical behavior diverges

Changed reduction order, recomputation, and precision conversions may alter training. The project must publish tolerances and avoid describing approximate behavior as exact.

### Complexity outweighs practical value

A runtime that technically supports a larger model but trains at unusable speed may have limited value. Benchmarking must present throughput and total cost alongside model size.

## 19. Proposed implementation stack

The initial implementation is expected to use:

- Python 3.11 or later;
- PyTorch 2.x;
- C++ and CUDA for critical paths;
- `io_uring`, `libaio`, or another measured asynchronous I/O backend;
- YAML for configuration;
- JSONL or a similar append-friendly telemetry format;
- pytest, Ruff, mypy, and pre-commit;
- GitHub Actions for automated checks.

These choices are provisional and should be validated against the actual runtime requirements.

## 20. Related work

MicroColossus builds on established research and engineering directions. Important references include:

- [PyTorch activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html)
- [ZeRO-Offload](https://arxiv.org/abs/2101.06840)
- [ZeRO-Infinity](https://arxiv.org/abs/2104.07857)
- [QLoRA](https://arxiv.org/abs/2305.14314)
- [GaLore](https://arxiv.org/abs/2403.03507)
- [LoHan](https://arxiv.org/abs/2403.06504)

The project does not present their underlying techniques as original. Its proposed focus is the combination of:

- an explicit target of very limited installed RAM as well as limited VRAM;
- hard budgets across every memory tier;
- NVMe as a canonical, versioned tensor store;
- tiling inside a layer, not only between layers;
- planning that accounts for elapsed time and SSD writes;
- telemetry expressed in bytes per step and bytes per token;
- crash recovery integrated into the execution model;
- strict separation of `reference`, `compact`, and `adapter` modes.

This differentiation remains a project hypothesis until it is implemented, compared with existing systems, and evaluated experimentally.

## 21. Final project statement

MicroColossus is based on a simple trade:

```text
less resident memory
        in exchange for
more I/O + more recomputation + more elapsed time
```

The intended contribution is not a claim that hardware limits disappear. It is a runtime that makes those limits explicit, schedules around them where possible, and reports the full cost of doing so.
