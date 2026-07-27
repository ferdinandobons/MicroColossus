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
- an MPS-safe gradient-norm calculation that does not create float64 tensors on Metal;
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

No out-of-core performance or model-scale claim is made yet.

## First real Mac M2 result

An independent diagnostic was run on a MacBook Air with an Apple M2, 8 GB of unified memory, native arm64 Python, PyTorch 2.13.0, and MPS enabled.

The original strict result was `FAIL` because the successful post-fix run intentionally left two tracked files modified. The functional result after those fixes was `PASS`:

- Ruff, pytest, compileall, and mypy passed;
- 20 tests passed;
- explicit MPS training completed;
- automatic device selection resolved to MPS;
- a larger resident MPS smoke run completed;
- a fixed-batch run reduced loss from `5.566842079162598` to `0.4145740866661072`;
- CPU-to-MPS maximum parameter difference was about `9.28e-06` absolute;
- no deliberate MPS-to-CPU fallback or unsupported-operation evidence was reported.

The two source fixes identified by that run are now incorporated in `main`. A clean rerun on the new commit is still required before the repository records a strict protocol `PASS`.

MPS runs were not bitwise reproducible by final checksum even when the reported loss and gradient norm matched at the first differing step. MicroColossus therefore treats MPS reproducibility as numerical, not automatically bitwise, until further evidence is collected.

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

## Competitive engineering policy

MicroColossus must be competitive by measurement, not by description.

Every important optimization must be evaluated against the strongest relevant baseline that can run on the target hardware. The initial comparison set is:

1. resident PyTorch MPS;
2. resident PyTorch MPS with activation checkpointing when applicable;
3. native Apple MLX using an equivalent controlled Transformer;
4. MLX-LM full-model fine-tuning when the workload is semantically comparable;
5. MicroColossus reference execution;
6. MicroColossus compact and adapter modes, reported separately;
7. storage-offload research systems such as ZeRO-Infinity and LoHan as architectural references where direct MPS execution is not possible.

An apples-to-apples comparison must hold constant, as far as the frameworks permit:

- model architecture and parameter count;
- initialization and input batches;
- precision and optimizer semantics;
- sequence length, microbatch, and update count;
- warm-up policy and synchronization points;
- machine, power state, and software versions.

Required metrics include numerical distance, step latency, tokens per second, process RSS, MPS current and driver allocations, swap growth, memory pressure, storage bytes, and SSD writes.

A technique is adopted only when it improves a declared objective without violating correctness, memory, endurance, or reproducibility constraints. If MLX materially outperforms PyTorch MPS for a required execution path, the project should implement an MLX backend or move that path to MLX rather than preserve a slower backend for historical reasons. PyTorch remains valuable as a portable numerical reference.

Full-parameter training must never be compared as if it were equivalent to LoRA, QLoRA, low-rank optimizer methods, or quantized optimizer states. Those methods belong to separately labeled modes.

## Development direction

The next implementation path is:

1. rerun the clean M2 diagnostic on the integrated fixes;
2. build an apples-to-apples PyTorch MPS versus MLX benchmark for the controlled Transformer;
3. profile resident execution, unified-memory pressure, and operator compatibility;
4. define a backend-neutral tensor manifest and execution contract;
5. create a versioned NVMe tensor store;
6. execute a bounded synchronous NVMe-to-MPS working set;
7. add activation recomputation and strict budget enforcement;
8. overlap storage, CPU preparation, and accelerator execution;
9. tile operations that cannot fit as a whole.

A CPU-owned copy of the complete model is not considered a solution on Apple Silicon because it still occupies the same unified-memory pool.

## Documentation

- [Project specification](docs/project.md)
- [Validation ledger](docs/validation.md)
- [PyTorch MPS backend documentation](https://docs.pytorch.org/docs/stable/notes/mps.html)
- [Apple MLX](https://github.com/ml-explore/mlx)
- [MLX-LM](https://github.com/ml-explore/mlx-lm)

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
