# Vision, scope, and success criteria

## 1. Original idea

MicroColossus exists to explore a simple trade:

> **Trade memory for time.**

The project asks whether full-parameter generative-model training can continue when the complete managed training state cannot remain safely resident in the machine's available memory.

The intended mechanism is not a claim of unlimited computation. It is a runtime that divides training into bounded pieces, stores inactive state outside the active working set, reconstructs or reloads state when needed, and accepts additional elapsed time as the cost of lower resident-memory demand.

The primary target is an **8 GB Apple Silicon Mac**, beginning with the MacBook Air M2.

## 2. What must be moved or bounded

A meaningful training system must account for more than model weights:

- model parameters;
- gradients;
- Adam first moments;
- Adam second moments;
- optimizer step tensors and parameter-group metadata;
- forward activations;
- activation gradients;
- logits and local workspaces;
- temporary staging buffers;
- framework allocations;
- Metal-driver allocations;
- filesystem cache;
- compressed memory and swap;
- checkpoint history and recovery journals.

The runtime must control the active subset of this state rather than merely loading a large model and hoping macOS swap absorbs the excess.

## 3. Apple Silicon constraint

Apple Silicon uses unified physical memory. CPU tensors and MPS tensors compete for the same physical pool.

Therefore:

- moving a tensor from MPS execution to CPU execution is not a new capacity tier;
- RSS, MPS allocation, Metal-driver allocation, compressed memory, and swap must be reported separately;
- those counters must not be added as independent physical memories;
- storage is the first capacity tier outside unified memory;
- a logical working-set budget is useful but is not proof of a lower physical-memory peak.

## 4. Project objective

The long-term objective is a recoverable full-parameter training runtime with these properties:

1. complete managed state may exceed safe resident memory;
2. every active parameter, gradient, optimizer, activation, and workspace set stays within declared limits;
3. inactive state is stored in versioned, checksummed storage;
4. multiple optimizer steps consume prior committed state;
5. a process can stop and resume from the last authoritative checkpoint;
6. a failed publication does not advance authoritative state;
7. data cursor, batch identity, model state, and optimizer state advance together;
8. checkpoint history can be pruned without changing `CURRENT` or retained state;
9. numerical behavior is compared with a reference path;
10. storage traffic, write amplification, elapsed time, replay, memory pressure, and swap are measured;
11. real training loss and validation loss show a useful trajectory;
12. capacity and performance claims are made only after target-hardware validation.

## 5. Near-term product target

The first defensible completion target is not a production framework. It is a **larger-than-safe-memory proof** on the 8 GB M2 that demonstrates:

- a controlled decoder-only Transformer;
- real text and deterministic data provenance;
- full-parameter AdamW;
- complete managed state larger than the declared safe resident limit;
- bounded execution through every optimizer step;
- at least several consecutive steps;
- interruption and resume;
- exact candidate restore;
- numerical equivalence within a declared band;
- storage and endurance telemetry;
- no hidden CPU fallback;
- a clean repository and reproducible protocol.

## 6. Non-goals for the first completion target

The first completion target does not require:

- literally unlimited model size;
- datacenter throughput;
- support for arbitrary PyTorch programs;
- a production tokenizer or production corpus;
- distributed training;
- mixed-precision or quantized-state approximations;
- LoRA or QLoRA as a substitute for full-parameter training;
- direct NVMe I/O as a prerequisite;
- MLX parity before the PyTorch reference path is correct;
- state-of-the-art language-model quality.

Approximate or adapter modes can be added later, but they must be reported separately from the full-parameter reference mode.

## 7. What does not count as success

The following are insufficient by themselves:

- fitting a larger parameter count by using swap;
- moving tensors between CPU and MPS without lowering unified-memory pressure;
- running a single forward pass;
- running a single optimizer step from a freshly initialized model;
- restoring only model weights without optimizer state;
- using a resident oracle inside the claimed bounded path;
- disabling validation without preserving integrity checks;
- training a large model with most parameters frozen;
- reporting parameter count without total state and working-set measurements;
- presenting a draft implementation as an accepted release.

## 8. Definition of competitive

MicroColossus should remain competitive in engineering quality even when it intentionally sacrifices speed.

Competitive means:

- explicit state and failure semantics;
- deterministic plans;
- target-specific measurements;
- comparison with resident PyTorch MPS and MLX where relevant;
- no silent fallback or approximation;
- measurable improvements over a baseline on at least one declared dimension;
- transparent costs when an optimization reduces memory but increases replay or elapsed time.

The project is not required to beat every existing framework in throughput. It is required to justify each optimization with measurements and preserve correctness.
