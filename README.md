# MicroColossus

> **Trade memory for time.**

MicroColossus is an experimental open-source training runtime for models whose complete training state does not fit in the available GPU memory or host RAM. It explores explicit tensor streaming, recomputation, NVMe offloading, and intra-layer tiling to exchange resident memory for I/O, computation, and elapsed time.

> **Status:** design specification. No implementation or benchmark results are available yet.

## Why this project exists

Training a generative model requires more than its weights. A full training step may also require gradients, optimizer states, master-weight copies, activations, temporary tensors, and kernel workspaces. On modest hardware, memory capacity can make an experiment impossible before compute time becomes the main constraint.

MicroColossus starts from one research question:

> Can a mathematically valid full-parameter update be executed when the complete model state fits only in storage, not in GPU memory or the RAM budget of the training process?

The project does not aim to create literally infinite models or match multi-GPU datacenter throughput. It aims to convert part of the memory-capacity problem into an explicit scheduling problem, while measuring the resulting cost in time, bandwidth, storage capacity, and SSD endurance.

## Core idea

MicroColossus treats local hardware as an explicit memory hierarchy:

```text
GPU compute
    |
VRAM cache and active working set
    |
Host RAM cache and staging buffers
    |
NVMe canonical tensor store
```

Only the tensors or tensor tiles required by the current operation remain in VRAM. The runtime plans prefetching, eviction, recomputation, and updates across the hierarchy. If a complete layer cannot fit in VRAM, the layer itself is divided into smaller computational tiles.

## Initial target

The first implementation is intended for a deliberately constrained machine:

- one CUDA GPU with 8 GB of VRAM;
- 8 GB of installed system RAM, with a lower hard budget for the training process;
- one consumer NVMe SSD;
- a controlled decoder-only Transformer architecture;
- full-parameter training in a reference mode before approximate optimizations are added.

The planned validation path begins with numerical comparison around 124 million parameters, then targets a full-parameter demonstration around 350 million parameters. A model around 1 billion parameters is a stretch target, not a promised result.

## Design principles

- **Hard budgets:** VRAM, RAM, NVMe capacity, and SSD writes are explicitly limited and measured.
- **Every tensor is accounted for:** the runtime should know where each tensor lives, which version is valid, and when it is needed again.
- **Reference and approximate methods remain separate:** exactness-oriented full-parameter execution must not be confused with quantized optimizer states, low-rank methods, or adapter training.
- **No invisible memory escape hatches:** unmanaged swap and unaccounted page-cache behavior are not part of the design.
- **Storage is a first-class resource:** bandwidth, latency, write amplification, and endurance are part of the optimization problem.
- **Claims require measurement:** feasibility, numerical behavior, and performance must be established by reproducible benchmarks.

## Planned system

MicroColossus is expected to include:

- a budget-aware planner;
- an asynchronous transfer engine;
- a bounded VRAM cache and RAM cache;
- a chunked, versioned, and recoverable NVMe tensor store;
- a layer-wise and tile-wise execution engine;
- activation checkpointing, offloading, and recomputation policies;
- telemetry for memory, I/O, stalls, throughput, and SSD writes;
- crash-safe step publication and checkpoint recovery.

## Documentation

The complete project definition is intentionally kept in one document:

- [Project specification](docs/project.md), including the motivation, scope, architecture, execution model, storage model, roadmap, validation plan, risks, and related work.

## Boundaries

MicroColossus does not claim that offloading, activation checkpointing, quantization, or tensor streaming are new ideas. Its proposed contribution is a strict-budget runtime designed around very limited local RAM and VRAM, with intra-layer tiling, explicit storage accounting, and a clear separation between reference and approximate training modes.

The proposed differentiation has not yet been demonstrated experimentally.

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
