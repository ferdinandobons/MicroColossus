# MicroColossus

> **Trade memory for time.**

MicroColossus is an experimental open-source runtime project for full-parameter training when the complete training state cannot remain resident in available memory.

The primary target is **Apple Silicon, beginning with an 8 GB Mac M2**. The project follows a dual-backend direction:

- **MLX** is the preferred native execution candidate for Apple Silicon.
- **PyTorch MPS** is the portable numerical oracle and the reference for validation, debugging, state comparison, and recovery semantics.

This decision is based on measured resident-training results. It does not yet validate out-of-core execution.

## Why Apple Silicon changes the design

Apple Silicon uses unified memory. CPU tensors, accelerator tensors, framework allocations, the Metal driver, and macOS compete for one physical pool.

Project rules:

- CPU-to-MPS placement is not a capacity offload.
- Process RSS, MPS allocation, Metal driver allocation, and MLX allocator counters overlap or describe different scopes.
- Those counters must not be added as separate physical memories.
- NVMe is the first capacity tier outside unified memory.
- Memory pressure, compression, swap, storage traffic, elapsed time, and SSD writes are first-class metrics.
- A larger model is not a successful result when correctness, throughput, recovery, or storage endurance are unacceptable.

The intended hierarchy is:

```text
MLX or MPS execution and active tensor working set
                         |
bounded unified-memory staging and runtime state
                         |
versioned NVMe tensor store
```

## Current status

Completed:

- typed YAML experiment configuration;
- controlled decoder-only Transformer;
- full-parameter resident AdamW training;
- CPU, MPS, CUDA, and automatic device selection;
- MPS diagnostics and synchronized memory telemetry;
- static planning with unified-memory warnings;
- deterministic portable FP32 weights and token batches;
- resident PyTorch MPS benchmarking;
- PyTorch activation checkpointing;
- equivalent resident MLX backend;
- atomic final-state `.state.npz` artifacts;
- tensor-level numerical comparison;
- latency, throughput, RSS, allocator, available-memory, and swap reporting;
- clean Apple M2 validation of package quality and all resident smoke paths.

Not implemented:

- versioned NVMe-backed model or optimizer state;
- bounded storage-to-accelerator execution;
- runtime-managed activation recomputation;
- asynchronous prefetch and writeback;
- intra-layer tiling;
- transactional optimizer-step publication;
- crash recovery;
- training state larger than safe resident unified memory.

No out-of-core performance or model-scale claim is made yet.

## Validated resident M2 foundation

A clean independent run on a MacBook Air M2 with 8 GB unified memory validated the resident PyTorch MPS foundation:

- native arm64 execution without Rosetta;
- MPS built and available;
- explicit MPS training and automatic MPS selection;
- CPU-versus-MPS numerical comparison;
- fixed-batch learning from `5.566842079162598` to `0.4145740866661072`;
- no detected fallback, unsupported operator, non-finite value, or MPS out-of-memory failure in the tested path.

MPS was not bitwise reproducible by final checksum. Reproducibility is evaluated numerically and bitwise as separate properties.

## Competitive Apple Silicon result

A clean competitive run tested version `0.3.2` on a MacBook Air M2 with PyTorch `2.13.0`, MLX `0.32.0`, and NumPy `2.4.6`.

All resident variants completed:

- tiny PyTorch MPS;
- tiny checkpointed PyTorch MPS;
- tiny MLX;
- three counterbalanced rounds of each backend on a 23,213,056-parameter model.

Median competitive throughput:

| Variant | Tokens/s | Relative to PyTorch |
|---|---:|---:|
| PyTorch MPS | 1,379.23 | 1.000x |
| PyTorch MPS with checkpointing | 1,337.37 | 0.970x |
| MLX | 2,195.69 | 1.592x |

Numerical comparison:

- PyTorch versus checkpointed PyTorch was exactly equal in the competitive runs.
- PyTorch versus MLX had a maximum loss difference of about `1.91e-06`.
- PyTorch versus MLX had a maximum final-parameter absolute difference of about `3.80e-05`.
- The mean final-parameter absolute difference was about `7.38e-09`.
- All compared values were finite and stable across all three rounds.

No hidden CPU fallback, unsupported operator, thermal warning, or material order bias was detected in the tested path.

### Backend decision

The current decision is **DUAL BACKEND**:

- MLX is materially faster for resident execution on the tested M2 workload.
- PyTorch remains the numerical oracle and reference implementation.
- The storage layer must remain backend-neutral.
- Storage-backed execution must be benchmarked independently before assigning every runtime component to one backend.

Activation checkpointing was numerically equivalent but did not provide a decisive memory benefit for the tested 23.2M-parameter resident model. It remains a required baseline for larger activation footprints.

## Final release-quality verification

Version `0.3.3`, commit `b75d2f646da4ca4dce5acdee567a1f17adcc503c`, passed a clean Mac M2 verification:

- Ruff passed;
- mypy passed with no issues in 17 source files;
- pytest passed with 28 tests;
- compileall passed;
- `microcolossus doctor` detected MPS on Apple M2;
- PyTorch MPS, checkpointed PyTorch MPS, and MLX tiny smoke runs passed;
- portable-state and batch checksums were identical across all three variants;
- every JSON and `.state.npz` artifact was valid and finite;
- the source tree remained clean.

The tiny numerical behavior remained consistent with the earlier competitive baseline. The nine larger performance runs were not repeated because the `0.3.3` changes were typing-only and the smoke results did not materially change.

## Install

Core environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Apple Silicon competitive environment:

```bash
python -m pip install -e ".[dev,benchmark]"
```

## Inspect and run

```bash
microcolossus doctor
microcolossus plan --config examples/tiny-mps.yaml
microcolossus train --config examples/tiny-mps.yaml
```

Run the competitive preset with PyTorch MPS:

```bash
microcolossus benchmark \
  --config examples/m2-competitive.yaml \
  --backend pytorch \
  --warmup-steps 3 \
  --steps 10 \
  --output runs/competitive/pytorch.json
```

Run it with MLX:

```bash
microcolossus benchmark \
  --config examples/m2-competitive.yaml \
  --backend mlx \
  --warmup-steps 3 \
  --steps 10 \
  --output runs/competitive/mlx.json
```

## Competitive engineering policy

An optimization is accepted only when it improves a declared objective without violating:

- numerical correctness;
- memory limits;
- storage endurance;
- stability;
- recovery semantics;
- reproducibility.

Full-parameter training must not be reported as equivalent to LoRA, QLoRA, low-rank optimizer methods, quantized optimizer state, or adapter training.

## Next milestone

The resident and competitive phases are complete. The next milestone is the **backend-neutral versioned NVMe tensor store**:

1. define tensor, chunk, version, manifest, and transaction schemas;
2. implement immutable or copy-on-write chunk storage;
3. add per-chunk checksums and atomic manifest publication;
4. add write-ahead journaling and crash recovery;
5. add bounded staging buffers and storage telemetry;
6. validate exact export and restoration through both backend adapters;
7. only then implement synchronous storage-to-accelerator training execution.

## Documentation

- [`docs/project.md`](docs/project.md)
- [`docs/validation.md`](docs/validation.md)
- [`docs/competitive.md`](docs/competitive.md)

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
