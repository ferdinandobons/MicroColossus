# MicroColossus

> **Trade memory for time.**

MicroColossus is an experimental open-source project exploring how to train generative models whose complete training state does not fit in local GPU memory or within a strict host RAM budget.

The long-term design treats VRAM, RAM, and NVMe storage as an explicit hierarchy. The intended runtime keeps only the active working set in VRAM, streams tensor chunks through bounded RAM buffers, recomputes selected activations, and tiles operations that are too large to execute as a whole.

## Current status

MicroColossus is in its first executable milestone.

Implemented now:

- a typed YAML experiment configuration;
- a controlled decoder-only Transformer reference model;
- a reproducible resident AdamW training baseline;
- a static memory estimator for model state and an illustrative streamed working set;
- JSON and JSONL experiment artifacts;
- process RAM and CUDA peak-allocation telemetry;
- model-state checksums for reproducibility checks;
- a command-line interface;
- automated tests and a GitHub Actions workflow.

Not implemented yet:

- RAM-to-VRAM layer streaming;
- NVMe-backed tensor state;
- activation offloading or recomputation policies;
- asynchronous prefetch and writeback;
- intra-layer tiling;
- crash-safe optimizer updates and checkpoint recovery;
- validated training of a model larger than resident memory.

No performance or model-scale result is claimed at this stage.

## Why the project exists

Training requires more than model weights. A full-parameter step may also require gradients, optimizer states, master-weight copies, activations, temporary tensors, and kernel workspaces. On modest hardware, memory capacity can make an experiment impossible before compute time becomes the main constraint.

MicroColossus asks:

> Can a mathematically valid full-parameter update be executed when the complete model state fits only in storage, while active computation remains inside strict VRAM and RAM budgets?

The project does not attempt to make models literally infinite or to match datacenter throughput. It investigates how much resident memory can be exchanged for additional I/O, recomputation, and elapsed time, while exposing the complete cost of that exchange.

## Intended memory hierarchy

```text
GPU compute
    |
VRAM cache and active working set
    |
Bounded host RAM cache and staging buffers
    |
Versioned NVMe tensor store
```

The first hardware target is deliberately constrained:

- one CUDA GPU with 8 GB of VRAM;
- 8 GB of installed system RAM, with a smaller hard budget for the process;
- one consumer NVMe SSD;
- one controlled decoder-only Transformer architecture;
- a full-parameter reference path before approximate methods are introduced.

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

## Run the current prototype

Build a static memory plan:

```bash
microcolossus plan --config examples/tiny-resident.yaml
```

Run the resident numerical baseline:

```bash
microcolossus train --config examples/tiny-resident.yaml
```

Override steps or execution device:

```bash
microcolossus train \
  --config examples/tiny-resident.yaml \
  --steps 10 \
  --device cpu
```

The package can also be invoked without the installed console script:

```bash
python -m microcolossus plan --config examples/tiny-resident.yaml
python -m microcolossus train --config examples/tiny-resident.yaml
```

Run the checks:

```bash
python -m pytest
python -m ruff check .
python -m compileall -q microcolossus
```

## Experiment artifacts

The example configuration writes into `runs/tiny-resident/`:

```text
runs/tiny-resident/
  resolved-config.json
  memory-plan.json
  steps.jsonl
  summary.json
```

`steps.jsonl` is replaced at the start of each new run in the same output directory. This prevents telemetry from separate executions being mixed silently.

## Current commands

### `microcolossus plan`

The planner currently reports:

- parameter count and parameter bytes;
- gradient, Adam-state, and optional master-weight bytes;
- an illustrative persistent resident-state estimate;
- a heuristic activation estimate;
- the largest execution-group parameter size;
- heuristic streamed VRAM and RAM working-set estimates;
- configured VRAM, process RAM, and NVMe budgets;
- basic capacity checks and explicit warnings.

The planner does not move tensors, enforce budgets, estimate real throughput, or guarantee that the streamed design is executable.

### `microcolossus train`

The training command runs the fully resident reference implementation on deterministic synthetic next-token batches. It performs forward, cross-entropy, backward, gradient-norm measurement, optional gradient clipping, and AdamW update.

The resident path exists to become the numerical oracle for future streamed execution.

## Development path

The next milestone is synchronous RAM-to-VRAM layer streaming with numerical comparison against the resident baseline. NVMe storage will be introduced only after the layer-wise execution model is correct and measurable.

Planned order:

1. resident reference baseline and static planner;
2. RAM-to-VRAM layer streaming;
3. versioned NVMe tensor store;
4. synchronous NVMe execution;
5. asynchronous overlap and bounded caches;
6. intra-layer tiling;
7. constrained-hardware demonstrations around 350 million parameters, followed by investigation of larger targets.

Targets beyond the implemented baseline are research goals, not promised results.

## Documentation

The complete motivation, architecture, implementation ledger, validation contract, roadmap, and technical risks are maintained in one document:

- [Project specification](docs/project.md)

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
