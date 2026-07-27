# MicroColossus Project Specification

> **Trade memory for time.**

This is the canonical description of why MicroColossus exists, what it is currently building, what has been implemented, and how future claims must be validated.

## 1. Purpose

MicroColossus is an experimental open-source runtime project for full-parameter training under strict memory constraints.

The project asks:

> How can a runtime execute a valid training update when the complete model state cannot remain resident, while the active computation stays inside explicit memory, storage, and endurance budgets?

The current primary platform is an Apple Silicon Mac, beginning with an M2 system with 8 GB of unified memory and local NVMe storage.

MicroColossus does not promise infinite models, datacenter throughput, or elimination of total training compute. Storage capacity, memory pressure, minimum useful tile size, total floating-point work, SSD bandwidth, SSD endurance, and elapsed time remain physical limits.

## 2. Why the project exists

Training requires more than model weights. A full step may also require:

- parameter gradients;
- Adam first and second moments;
- optional master-weight copies;
- saved activations or recomputation metadata;
- logits and loss intermediates;
- temporary tensors and operator workspaces;
- allocator, framework, driver, and runtime overhead.

A mixed-precision AdamW layout is often approximated at roughly 16 bytes or more of persistent state per trainable parameter before activations. The exact value depends on dtype, optimizer, implementation, and layout.

| Parameters | Illustrative persistent state at 16 bytes per parameter |
|---:|---:|
| 124 million | about 2.0 GB |
| 350 million | about 5.6 GB |
| 1 billion | about 16 GB |
| 7 billion | about 112 GB |

A small computer may have enough storage for this state while lacking enough resident memory to train it conventionally. MicroColossus treats that mismatch as a scheduling problem whose complete cost must be measured.

## 3. Why Apple M2 changes the design

Apple Silicon does not expose discrete CPU RAM and GPU VRAM in the same way as a conventional system with a dedicated GPU. CPU and Metal workloads share unified physical memory.

This creates four important rules.

### 3.1 Placement is not capacity offload

Moving a tensor from CPU execution to the MPS device changes how it is used and represented. It does not create a separate physical memory pool. Keeping a complete CPU copy while materializing an MPS copy can increase pressure on the same unified memory.

### 3.2 Memory counters overlap

Process RSS, MPS current tensor allocation, and Metal driver allocation describe different views of memory. They can overlap. They must not be added to estimate total physical use.

### 3.3 Storage is the capacity tier

NVMe, not CPU memory, is the first tier that can hold state without consuming unified memory. The useful long-term hierarchy is therefore:

```text
MPS execution and active tensor working set
                  |
bounded unified-memory staging and runtime state
                  |
versioned NVMe tensor store
```

### 3.4 The first bottleneck may be memory pressure, not transfer bandwidth

A schedule must manage the active MPS allocation, Metal driver allocation, process RSS, operating-system pressure, storage traffic, and step time together.

## 4. Central hypothesis

For a useful class of Transformer models and training recipes, part of resident-memory demand may be exchanged for:

- versioned tensor storage;
- bounded staging buffers;
- activation recomputation;
- layer-wise execution;
- intra-layer tiling;
- asynchronous prefetch and writeback;
- additional elapsed time.

The minimum capacity relationships are:

```text
model state + checkpoints + metadata <= allocated NVMe capacity

active parameter tiles + activations + workspaces + runtime buffers
    <= configured logical accelerator working-set budget

process state + staging + operating overhead
    remains below safe unified-memory pressure
```

These relationships are necessary but not sufficient. A viable schedule must also account for SSD bandwidth, write amplification, endurance, MPS operator support, numerical behavior, and recovery after interruption.

## 5. Goals and boundaries

MicroColossus aims to:

1. Train models whose complete state cannot remain resident in unified memory.
2. Keep only the active tensor working set available to MPS execution.
3. Store inactive parameter and optimizer chunks on NVMe.
4. Apply explicit logical budgets to MPS allocations, process memory, storage capacity, and cumulative SSD writes.
5. Recompute selected activations instead of retaining them.
6. Tile a single operation when it cannot fit as a whole.
7. Preserve a clearly identified full-parameter reference path.
8. Track the identity, version, location, and integrity of managed tensor chunks.
9. Avoid publishing a partially completed optimizer step.
10. Report time, memory, storage traffic, stalls, and numerical distance from the resident baseline.

It does not promise:

- literally infinite models;
- throughput comparable with multi-GPU infrastructure;
- support for arbitrary PyTorch graphs in the initial implementation;
- feasibility for every model architecture;
- that CPU-to-MPS movement alone increases capacity;
- silent CPU fallback for unsupported MPS operators;
- success at scale targets before measured demonstrations exist.

A larger parameter count alone is not sufficient evidence of success.

## 6. Primary target

The first target class is:

- Apple M2 or a closely related Apple Silicon system;
- 8 GB of unified memory as the constrained target;
- macOS and a PyTorch build with MPS support;
- one internal or external NVMe SSD;
- no distributed cluster;
- a controlled decoder-only Transformer;
- full-parameter reference training before approximate methods.

The implementation remains portable to CPU and CUDA for comparison, but architecture and validation decisions should prioritize MPS and unified memory.

## 7. Training modes

Planned modes remain separate.

- **`reference`**: all parameters are trainable. No silent adapters, low-rank gradient projections, or quantized optimizer states.
- **`compact`**: future mode for declared approximations such as quantized optimizer states or low-rank optimizer methods.
- **`adapter`**: future LoRA or QLoRA path, reported separately from full-parameter training.

Only `reference` is currently accepted.

## 8. Proposed architecture

```text
Dataset stream
      |
Controlled Transformer frontend
      |
Tensor and dependency analysis
      |
Unified-memory-aware planner
      |
Execution schedule
      |
+----------------------------------+
| Runtime                          |
| - MPS executor                   |
| - storage transfer engine        |
| - activation manager             |
| - optimizer engine               |
| - checkpoint coordinator         |
| - telemetry                      |
+----------------------------------+
      |
bounded unified-memory working set <-> NVMe tensor store
```

The intended components are:

- a controlled frontend with a deliberately small operator set;
- a tensor manifest with shape, dtype, version, location, and checksum;
- a planner for tiling, prefetch, eviction, recomputation, and placement;
- bounded staging and MPS working-set policies;
- a chunked, versioned, recoverable NVMe tensor store;
- layer-wise and tile-wise forward, backward, and optimizer execution;
- telemetry for process memory, MPS allocations, storage traffic, stalls, throughput, and writes.

## 9. Current implementation

The repository contains the executable foundation, not the out-of-core runtime.

Implemented:

- typed YAML configuration;
- controlled decoder-only Transformer;
- deterministic synthetic next-token data;
- resident full-parameter AdamW training;
- explicit `mps` device selection;
- `auto` device selection that prefers MPS when available;
- environment inspection through `microcolossus doctor`;
- process RSS telemetry;
- CUDA peak-allocation telemetry;
- MPS current tensor allocation telemetry;
- MPS driver allocation telemetry;
- MPS recommended maximum working-set telemetry;
- NumPy-independent model-state checksums;
- atomic JSON and JSONL experiment artifacts;
- a static memory estimator with unified-memory warnings;
- unit tests, packaging, and CI.

Not implemented:

- storage-backed model or optimizer state;
- real NVMe-to-MPS tensor streaming;
- activation offloading or managed recomputation;
- asynchronous transfers;
- strict runtime budget enforcement;
- intra-layer tiling;
- tensor manifests, journals, or crash recovery;
- resident-versus-streamed numerical comparison;
- training state larger than resident unified memory.

The CPU path has been exercised after the first diagnostic corrections. The MPS path has not yet been verified on a real Mac M2 by the maintainer environment.

## 10. Current commands

```bash
microcolossus doctor
microcolossus plan --config examples/tiny-mps.yaml
microcolossus train --config examples/tiny-mps.yaml
microcolossus train --config examples/tiny-resident.yaml --device cpu
```

A training run writes:

```text
runs/<experiment>/
  resolved-config.json
  memory-plan.json
  steps.jsonl
  summary.json
```

The workload is synthetic. It validates infrastructure, execution, and repeatability. It does not validate model quality.

## 11. MPS telemetry contract

Every synchronized MPS training step should report:

- process RSS;
- current memory occupied by MPS tensors;
- total memory allocated by the Metal driver for the process;
- the recommended maximum MPS working set when available;
- synchronized step duration;
- loss and gradient norm;
- model-state checksum.

The current MPS tensor allocation is not a peak measurement. The field `accelerator_memory_measurement` identifies the measurement as `mps-current-allocated`.

The runtime must not sum RSS, current MPS allocation, and driver allocation to claim total physical memory consumption.

## 12. Static planner limitations

The planner calculates parameter, gradient, and Adam-state sizes from the instantiated model. Activation, workspace, transfer-buffer, and streamed-working-set values are heuristics.

It does not yet model:

- unified-memory page residency;
- operating-system memory pressure and compression;
- allocator fragmentation;
- exact MPS operator workspaces;
- unsupported operator fallback;
- storage latency and throughput;
- transfer and compute overlap;
- SSD write amplification;
- runtime-measured tensor lifetimes.

Its output is a planning hypothesis, not a guarantee that a run will fit or perform acceptably.

## 13. Validation contract

Correctness precedes scale.

The resident implementation is the numerical oracle. A future streamed path must use the same:

- initial parameters;
- input batches;
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

An MPS result must also report:

- Mac model and chip;
- total unified memory;
- macOS version;
- PyTorch version;
- whether MPS is built and available;
- MPS device name and core count when exposed;
- all memory telemetry fields;
- every warning, fallback, and failed operator.

## 14. Storage and recovery requirements

The future tensor store should use:

- immutable chunks or copy-on-write versions;
- logical tensor versions;
- per-chunk checksums;
- a write-ahead journal;
- atomic manifest publication;
- incremental checkpoints;
- failure-injection tests.

A step is committed only after all required chunks and the new manifest are valid. Until then, the previous committed version remains authoritative.

Recovery must distinguish committed steps, unpublished complete writes, partial writes, corrupt chunks, and stale cached state.

## 15. Roadmap

### M0. Resident foundation and diagnostics

- package, configuration, and CLI;
- controlled model;
- resident training baseline;
- static planner;
- checksums, telemetry, and artifacts;
- CPU reproducibility tests;
- MPS device support and diagnostics.

Status: implemented in code. Independent Mac M2 validation is still required.

### M1. Mac M2 resident characterization

- execute the baseline on a real M2;
- verify all required operators on MPS;
- measure current and driver allocations across model sizes;
- measure process RSS and operating-system pressure;
- compare CPU and MPS numerical trajectories;
- define safe logical budgets for an 8 GB system;
- document reproducibility and unsupported operations.

### M2. Versioned NVMe tensor store

- tensor and chunk identifiers;
- checksums and copy-on-write versions;
- journal and recovery tests;
- read and write telemetry;
- bounded staging buffers.

### M3. Synchronous NVMe-to-MPS execution

- authoritative parameter and optimizer state in the tensor store;
- load one layer or tile into the active MPS working set;
- layer-wise forward and backward;
- atomic optimizer-step publication;
- comparison with the resident baseline.

### M4. Recomputation and strict budgets

- activation checkpoint selection;
- deterministic recomputation;
- MPS, process-memory, and storage-budget rejection;
- explicit handling of unified-memory pressure.

### M5. Asynchronous overlap

- double buffering;
- storage prefetch and writeback;
- stall measurement;
- storage and execution overlap.

### M6. Intra-layer tiling

- tiled linear and MLP paths;
- partitioned embeddings;
- tiled output projection and loss;
- normalization and attention paths.

### M7. Constrained-hardware demonstration

- validate around 124 million parameters;
- attempt full-parameter training around 350 million parameters on an 8 GB M2-class system;
- investigate larger targets only after the smaller result is correct and measurable.

These scale targets are research goals, not current capabilities or guaranteed outcomes.

## 16. Benchmark and success criteria

Core metrics are:

- process RSS;
- MPS tensor and driver allocations;
- operating-system memory pressure where measurable;
- tokens per second and seconds per step;
- storage read and write bytes;
- GPU or MPS stall time;
- SSD write amplification;
- checkpoint and recovery time;
- numerical distance from the resident baseline;
- energy per million tokens when measurement is available.

The first meaningful streamed milestone requires:

1. the active MPS working set remains inside configured logical limits;
2. the complete state exceeds what can remain resident safely;
3. the update is compared with the resident baseline;
4. every managed tensor version is traceable;
5. storage reads and writes are reported;
6. interruption recovers the last committed state;
7. the result is reproducible from a documented configuration.

## 17. Main risks

- **Unified-memory ambiguity:** counters overlap and may not reveal physical residency directly.
- **Memory pressure:** the operating system may compress or terminate the process before logical budgets are reached.
- **MPS operator gaps:** some operations or data types may be unsupported or behave differently.
- **Storage domination:** MPS may remain idle while tensor chunks are read or written.
- **SSD endurance:** optimizer state and gradients may create impractical write volume.
- **Numerical divergence:** recomputation and changed reduction order may alter training.
- **Complexity without utility:** a technically larger model may still train too slowly to be useful.

Every benchmark must report these costs instead of presenting parameter count in isolation.

## 18. Related work and references

MicroColossus builds on established work, including:

- [PyTorch MPS backend](https://docs.pytorch.org/docs/stable/notes/mps.html)
- [PyTorch activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html)
- [ZeRO-Offload](https://arxiv.org/abs/2101.06840)
- [ZeRO-Infinity](https://arxiv.org/abs/2104.07857)
- [QLoRA](https://arxiv.org/abs/2305.14314)
- [GaLore](https://arxiv.org/abs/2403.03507)
- [LoHan](https://arxiv.org/abs/2403.06504)

The project does not present those underlying techniques as original.

Its proposed focus is the combination of:

- an 8 GB Apple Silicon target;
- unified-memory-aware accounting;
- NVMe as a versioned canonical tensor store;
- full-parameter reference execution;
- tiling inside individual layers;
- planning that includes elapsed time and SSD writes;
- recovery integrated into execution;
- strict separation between reference and approximate modes.

This differentiation remains a hypothesis until it is implemented and compared experimentally.

## 19. Project statement

```text
less resident state
        in exchange for
more storage traffic + more recomputation + more elapsed time
```

The intended contribution is a runtime that makes memory limits explicit, schedules around them where practical, preserves a measurable reference path, and reports the complete cost of doing so.
