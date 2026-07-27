# MicroColossus

> **Trade memory for time.**

MicroColossus is an experimental open-source project exploring full-parameter training when the complete training state cannot remain resident in available memory.

The primary target is **Apple Silicon, beginning with an 8 GB Mac M2**. PyTorch MPS is the portable numerical reference. Apple MLX is a mandatory native baseline and a candidate execution backend.

## Why Apple Silicon changes the design

Apple Silicon uses unified memory. CPU tensors, accelerator tensors, framework allocations, the Metal driver, and macOS compete for one physical pool.

Project rules:

- CPU-to-MPS placement is not a capacity offload;
- process RSS, MPS allocation, and driver allocation overlap;
- those counters must not be added as separate physical memories;
- NVMe is the first capacity tier outside unified memory;
- memory pressure, compression, swap, storage traffic, and elapsed time are first-class metrics;
- a larger model is not a successful result if correctness, throughput, or endurance are unacceptable.

The intended hierarchy is:

```text
MPS or MLX execution and active tensor working set
                         |
bounded unified-memory staging and runtime state
                         |
versioned NVMe tensor store
```

## Current status

The repository contains a validated resident foundation and the first competitive benchmark harness. It is not yet an out-of-core runtime.

Implemented:

- typed YAML experiment configuration;
- a controlled decoder-only Transformer;
- full-parameter resident AdamW training;
- CPU, MPS, CUDA, and automatic device selection;
- MPS diagnostics and synchronized memory telemetry;
- an MPS-safe gradient-norm calculation;
- NumPy-independent training checksums;
- a static planner with unified-memory warnings;
- deterministic portable benchmark weights and token batches;
- synchronized PyTorch resident benchmarking;
- a PyTorch activation-checkpointing benchmark;
- an optional equivalent MLX resident implementation;
- machine-readable benchmark and comparison JSON;
- tests, linting, typing, compilation, and CPU smoke checks in CI.

Not implemented:

- storage-backed model or optimizer state;
- bounded NVMe-to-accelerator execution;
- runtime-managed activation recomputation;
- asynchronous prefetch and writeback;
- intra-layer tiling;
- crash-safe optimizer-step publication and recovery;
- training of state larger than safe resident unified memory.

No out-of-core performance or model-scale claim is made yet.

## Validated Mac M2 baseline

A clean independent rerun tested commit `a56fc514f2f8e705654034f3c2f02e3a441c61f3` on a MacBook Air with Apple M2, 8 GB unified memory, native arm64 Python, PyTorch 2.13.0, and MPS enabled.

Strict protocol result: **PASS**.

Confirmed:

- clean working tree before and after the diagnostic;
- Ruff, mypy, compileall, and 21 pytest tests passed;
- MPS preflight passed;
- explicit MPS training passed;
- automatic device selection resolved to MPS;
- a larger resident MPS smoke run passed;
- CPU-versus-MPS comparison passed for the tested workload;
- fixed-batch loss moved from `5.566842079162598` to `0.4145740866661072`;
- no detected CPU fallback, unsupported operator, non-finite value, or MPS OOM.

MPS was not bitwise reproducible by final checksum. Reproducibility is therefore evaluated numerically with tensor-level tolerances as well as bitwise.

See [`docs/validation.md`](docs/validation.md) for the exact boundary of the evidence.

## Install

Python 3.11 or later is required.

Core development environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Competitive Apple Silicon environment with MLX:

```bash
python -m pip install -e ".[dev,benchmark]"
```

MLX is selected by the optional dependency marker only on native arm64 macOS. Core training and planning do not require MLX or NumPy.

## Inspect and run the resident reference

```bash
microcolossus doctor
microcolossus plan --config examples/tiny-mps.yaml
microcolossus train --config examples/tiny-mps.yaml
```

## Competitive benchmark harness

PyTorch MPS:

```bash
microcolossus benchmark \
  --config examples/tiny-mps.yaml \
  --backend pytorch \
  --warmup-steps 2 \
  --steps 10 \
  --output runs/competitive/pytorch-mps.json
```

PyTorch MPS with activation checkpointing:

```bash
microcolossus benchmark \
  --config examples/tiny-mps.yaml \
  --backend pytorch \
  --activation-checkpointing \
  --warmup-steps 2 \
  --steps 10 \
  --output runs/competitive/pytorch-mps-checkpointed.json
```

Native MLX:

```bash
microcolossus benchmark \
  --config examples/tiny-mps.yaml \
  --backend mlx \
  --warmup-steps 2 \
  --steps 10 \
  --output runs/competitive/mlx.json
```

Compare results:

```bash
microcolossus compare-benchmarks \
  --left runs/competitive/pytorch-mps.json \
  --right runs/competitive/mlx.json \
  --output runs/competitive/pytorch-vs-mlx.json
```

The harness records identical portable FP32 initial state and token batches, architecture and optimizer settings, warm-up and measured phases, synchronized latency, tokens per second, process RSS, allocator counters, available memory, and swap.

Framework memory counters are not treated as physically equivalent. macOS memory pressure, swap, and process footprint remain required external cross-checks.

The PyTorch benchmark path has CPU test coverage. The MLX path must still run on the target M2 before any PyTorch-versus-MLX performance conclusion is accepted.

## Competitive engineering policy

MicroColossus must be competitive by measurement.

The direct Apple Silicon baseline set begins with:

1. resident PyTorch MPS;
2. PyTorch MPS with activation checkpointing;
3. native MLX with the equivalent Transformer;
4. compiled MLX as a later optimized variant;
5. MLX-LM full-model fine-tuning when semantically comparable;
6. MicroColossus reference execution;
7. future compact and adapter modes, reported separately.

An optimization is accepted only when it improves a declared objective without violating correctness, memory, storage endurance, stability, recovery, or reproducibility.

If MLX materially outperforms PyTorch MPS for a required path, MicroColossus should add or select an MLX backend. PyTorch remains the portable numerical reference.

Full-parameter training must not be reported as equivalent to LoRA, QLoRA, low-rank optimizer methods, or quantized optimizer state.

## Checks

```bash
ruff check .
mypy microcolossus
python -m pytest
python -m compileall -q microcolossus
```

## Next gate

1. run PyTorch MPS, checkpointed PyTorch MPS, and MLX on the same clean M2;
2. validate equal state and batch checksums;
3. compare loss trajectories, latency, throughput, memory pressure, and swap;
4. add compiled MLX and other justified variants;
5. use measured evidence to select the first storage-backed backend;
6. implement the versioned NVMe tensor store;
7. implement bounded synchronous storage-to-accelerator execution.

## Documentation

- [`docs/project.md`](docs/project.md)
- [`docs/validation.md`](docs/validation.md)
- [PyTorch MPS documentation](https://docs.pytorch.org/docs/stable/notes/mps.html)
- [Apple MLX](https://github.com/ml-explore/mlx)
- [MLX-LM](https://github.com/ml-explore/mlx-lm)

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
