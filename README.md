# MicroColossus

> **Trade memory for time.**

MicroColossus is an experimental open-source project exploring how to train generative models when the complete training state does not fit comfortably inside the available memory.

The current primary target is **Apple Silicon, starting with an 8 GB Mac M2 using PyTorch's Metal Performance Shaders backend**.

## Why focus on Apple Silicon

Apple Silicon uses unified memory. CPU tensors, MPS tensors, framework allocations, and the operating system compete for the same physical memory pool. This changes the project design:

- CPU-to-MPS movement is a placement and execution decision, not a free capacity offload;
- process RSS, MPS tensor allocation, and Metal driver allocation overlap and must not be added as if they were separate memories;
- storage offload, recomputation, bounded working sets, and intra-layer tiling become the main mechanisms for exceeding resident-memory capacity;
- every result must report the full memory and I/O cost.

MicroColossus does not claim that hardware limits disappear. It aims to make those limits explicit and schedule around them where practical.

## Current status

The repository contains an executable resident baseline, not an out-of-core runtime.

Implemented:

- typed YAML experiment configuration;
- a controlled decoder-only Transformer;
- full-parameter resident AdamW training;
- `cpu`, `mps`, `cuda`, and `auto` device selection;
- automatic preference for MPS when it is available;
- MPS environment diagnostics;
- MPS current tensor allocation, driver allocation, and recommended working-set telemetry;
- NumPy-independent model-state checksums;
- a static memory estimator with unified-memory warnings;
- JSON and JSONL run artifacts;
- tests, linting, type checking, compilation, and CPU smoke checks in CI.

Not implemented:

- storage-backed model or optimizer state;
- bounded NVMe-to-MPS streaming;
- activation offloading or planned recomputation;
- asynchronous prefetch and writeback;
- intra-layer tiling;
- crash-safe step publication and recovery;
- training of state larger than available unified memory.

No performance or model-scale claim is made yet.

## Install

Python 3.11 or later is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

## Inspect the machine

```bash
microcolossus doctor
```

On a compatible Mac, the report should distinguish whether PyTorch was built with MPS and whether MPS is available at runtime.

## Run the M2-oriented prototype

Build the MPS-oriented static plan:

```bash
microcolossus plan --config examples/tiny-mps.yaml
```

Run the resident baseline on MPS:

```bash
microcolossus train --config examples/tiny-mps.yaml
```

Run the portable resident example with automatic device selection:

```bash
microcolossus train --config examples/tiny-resident.yaml
```

Force a device or override the number of steps:

```bash
microcolossus train \
  --config examples/tiny-resident.yaml \
  --steps 10 \
  --device mps
```

An explicit `mps` request fails clearly when the installed PyTorch build or machine cannot provide MPS. The project does not silently enable CPU fallback for unsupported MPS operations.

## Run the checks

```bash
ruff check .
mypy microcolossus
python -m pytest
python -m compileall -q microcolossus
```

## Experiment artifacts

Each training run writes:

```text
runs/<experiment>/
  resolved-config.json
  memory-plan.json
  steps.jsonl
  summary.json
```

For MPS, each step records:

- process RSS;
- MPS current tensor allocation;
- total memory allocated by the Metal driver for the process;
- PyTorch's recommended maximum MPS working set;
- synchronized step duration;
- loss, gradient norm, and model checksum.

The current MPS allocation is not a measured peak. The telemetry labels the measurement kind explicitly.

## Development direction

The next implementation path is:

1. validate the resident baseline on a real Mac M2;
2. measure unified-memory behavior and operator compatibility;
3. create a versioned NVMe tensor store;
4. execute a bounded synchronous NVMe-to-MPS working set;
5. add activation recomputation and strict budget enforcement;
6. overlap storage, CPU preparation, and MPS execution;
7. tile operations that cannot fit as a whole.

A CPU-owned copy of the complete model is not considered a solution on Apple Silicon because it still occupies the same unified-memory pool.

## Documentation

- [Project specification](docs/project.md)
- [Validation ledger](docs/validation.md)
- [PyTorch MPS backend documentation](https://docs.pytorch.org/docs/stable/notes/mps.html)

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
