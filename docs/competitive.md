# Competitive Engineering Log

This document records the evidence used to select optimization techniques and execution backends for MicroColossus.

## Decision rule

MicroColossus must remain competitive by measurement. A technique is accepted only when it improves a declared objective without violating numerical correctness, memory limits, storage endurance, stability, recovery, or reproducibility.

Full-parameter training, compact optimizer methods, and adapter methods must be reported separately. PyTorch MPS is the portable numerical reference for Apple Silicon. MLX is the mandatory native comparison and remains a candidate execution backend.

## Validated resident foundation

A clean MacBook Air M2 run of commit `a56fc514f2f8e705654034f3c2f02e3a441c61f3` passed the resident validation protocol:

- native arm64 execution with 8 GB unified memory;
- MPS built and available;
- Ruff, mypy, compileall, and 21 tests passed;
- explicit MPS training and automatic MPS selection passed;
- CPU-versus-MPS numerical comparison passed for the tested workload;
- fixed-batch loss moved from `5.566842079162598` to `0.4145740866661072`;
- no detected fallback, unsupported operator, non-finite value, or MPS out-of-memory error.

This result validates only resident training. It does not validate storage-backed or out-of-core execution.

## Benchmark harness

Version 0.3 introduced a backend-neutral resident benchmark with:

- identical portable FP32 parameters and token batches;
- PyTorch MPS, checkpointed PyTorch MPS, and MLX variants;
- synchronized warm-up and measured phases;
- latency, throughput, process memory, allocator counters, available memory, and swap;
- atomic final-state artifacts and tensor-level numerical comparison.

## First competitive target run

The first target-hardware attempt tested commit `2f7963a5d0af3eabb5a31eab4013f422725c71c0` on a clean MacBook Air M2 checkout.

Result: **FAIL at the project-check gate**.

Successful checks:

- installation of `.[dev,benchmark]`;
- native arm64 execution without Rosetta;
- PyTorch 2.13.0 with MPS built and available;
- MLX 0.32.0 import;
- 26 pytest tests;
- compileall;
- doctor;
- tiny and 23,213,056-parameter static plans.

Blocking checks:

1. Ruff found import-member ordering errors in `benchmark_compare.py` and `benchmark_runner.py`.
2. mypy targeted Python 3.11 while the unconstrained dependency resolver installed NumPy 2.5.1. NumPy 2.5 dropped Python 3.11 support and its stubs use syntax unavailable to the configured Python 3.11 target.

The protocol correctly stopped before MPS and MLX preflights or performance benchmarks. No backend decision was made.

## Corrective release 0.3.2

The corrective release:

- applies Ruff's required import ordering;
- constrains benchmark and development NumPy to `>=2.4,<2.5`, preserving the project's Python 3.11 typing target while supporting Python 3.11 through 3.14;
- adds Python 3.11 and 3.13 CI jobs;
- adds packaging regression tests for version synchronization and the NumPy compatibility range.

The next target run must start from a clean checkout of version 0.3.2 or later and must repeat the full competitive protocol before choosing PyTorch MPS, MLX, a dual-backend design, or no decision.

## Current boundary

No competitive result yet establishes:

- a preferred backend;
- NVMe-backed parameters or optimizer state;
- storage-to-accelerator streaming;
- managed activation recomputation;
- asynchronous prefetch or writeback;
- intra-layer tiling;
- training state larger than safe resident unified memory.

Operational tracking remains in [GitHub issue #1](https://github.com/ferdinandobons/MicroColossus/issues/1).
