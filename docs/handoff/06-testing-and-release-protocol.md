# Testing, evidence, and release protocol

This protocol exists to prevent draft code, protocol mistakes, and target-specific assumptions from becoming accepted project claims.

## 1. Roles

### Implementation agent

The implementation agent:

- edits source code;
- writes unit and integration tests;
- updates documentation;
- runs available Linux CPU tests;
- opens a normal branch and pull request;
- diagnoses CI;
- prepares the exact target-hardware prompt.

### Apple M2 validation agent

The M2 agent:

- uses a fresh clone and fresh virtual environment;
- checks out one exact commit;
- executes the protocol without changing tracked source;
- records commands and telemetry;
- performs target-specific training and failure tests;
- packages artifacts;
- reports results to the implementation agent.

The M2 agent must not silently fix product source. External harness corrections are allowed only outside the repository, and the original harness failure must be preserved.

## 2. Branch and pull-request discipline

Every feature should use:

```text
branch: agent/<descriptive-name>
PR:     draft until quality and CPU gates pass
```

Required rules:

- no direct product commits to `main`;
- no self-modifying workflow that writes feature code into a branch;
- no base64 patch payload in a product PR;
- no temporary privileged workflow in an accepted merge;
- no unrelated changes in one PR;
- no merge while the PR is draft;
- no merge while required checks are failing or missing;
- use the exact PR head SHA in all target validation runs;
- after target PASS, update documentation in a separate small PR when practical.

## 3. Mandatory local and CI gate

Before target-hardware execution, all of these must pass from a fresh environment:

```bash
python -m pip install -e ".[dev]"
python -c "import microcolossus; print(microcolossus.__version__)"
ruff check .
mypy microcolossus
python -m pytest
python -m compileall -q microcolossus
microcolossus doctor
```

Feature-specific CLI `--help` and smoke commands must also pass.

CI must run at least Python 3.11 and 3.13.

### Stop rule

If package import, Ruff, mypy, pytest, compileall, or a required CLI fails, stop before M2 model experiments.

A target run after a failed quality gate adds noise and is not accepted evidence.

## 4. Test scale ladder

Run tests in this order.

### 4.1 Unit scale

Test:

- schemas;
- checksums;
- deterministic serialization;
- tensor math;
- budget calculations;
- corruption;
- failure injection;
- invalid configuration.

### 4.2 Synthetic micro

Use 11,456 parameters for fast numerical, storage, transaction, and failure tests.

### 4.3 Real-text micro

Use 18,624 parameters for:

- data identity;
- validation loss;
- sample generation;
- resume;
- pruning;
- activation-policy tests.

### 4.4 Tiny

Use 443,648 parameters for target-hardware numerical and telemetry gates.

### 4.5 Real-text small

Use 1,846,656 parameters for meaningful learning, storage reclamation, and activation-memory comparisons.

### 4.6 Capacity scale

Only after all smaller gates pass, select a larger model based on measured group, tile, storage, and memory limits.

## 5. Deterministic execution requirements

Every comparison must preserve:

- model configuration;
- initialization seed;
- training seed;
- batch cursor;
- batch offsets;
- batch checksum;
- data identity;
- optimizer hyperparameters;
- clipping semantics;
- dtype;
- activation policy;
- plan checksum where applicable;
- working-set budgets.

Do not silently retry with changed values. A changed configuration is a separate experiment.

## 6. Numerical classification

Compare complete canonical state where feasible:

- parameters;
- Adam first moments;
- Adam second moments;
- optimizer step tensors;
- parameter-group metadata.

### BITWISE_EXACT

```text
exact canonical bytes
identical names and structures
all finite
```

### GREEN

Recommended band:

```text
maximum absolute state difference <= 1e-4
mean absolute state difference    <= 1e-7
maximum loss difference           <= 1e-5
maximum gradient-norm difference  <= 1e-4
provenance equal
candidate restore exact
all values finite
```

### YELLOW

Diagnostic but potentially acceptable with explanation:

```text
maximum absolute state difference <= 1e-3
mean absolute state difference    <= 1e-6
provenance equal
candidate restore exact
all values finite
```

### RED

Any of:

- missing or mismatched state;
- non-finite value;
- provenance mismatch;
- candidate restore failure;
- difference above the declared band;
- unexplained optimizer-step mismatch.

Numerical bands must be declared before the run.

## 7. Candidate restore

Exact candidate restore is a separate requirement from resident-oracle proximity.

Required:

```text
candidate store -> restore -> canonical state
exact bytes: true
```

A numerically close candidate that cannot restore exactly is a storage or adapter failure.

## 8. Failure injection

Every new stateful feature must test failures at its publication boundaries.

At minimum:

- before child manifest publication where applicable;
- before root manifest rename;
- before `CURRENT` rename;
- during destructive maintenance after at least one deletion;
- stale or changed plan;
- corrupt retained state;
- active lock conflict.

Required outcome:

- previous `CURRENT` remains authoritative;
- previous root verifies;
- recovery identifies temporary or unpublished state;
- retry is safe or explicitly rejected;
- no partial state is silently promoted.

Expected simulated exceptions must be separated from unexpected failures in log scans.

## 9. Apple M2 environment protocol

Every accepted M2 run must record:

- Mac model and identifier;
- Apple chip and core counts;
- unified memory;
- macOS version;
- filesystem;
- native architecture;
- Rosetta status;
- Python version;
- PyTorch version;
- MLX version when used;
- MPS built and available;
- free storage;
- power source;
- thermal status;
- memory pressure;
- swap.

Required:

```text
uname -m: arm64
sysctl.proc_translated: 0
MPS built: true
MPS available: true
```

## 10. MPS fallback policy

Record relevant environment variables, then explicitly unset:

```bash
unset PYTORCH_ENABLE_MPS_FALLBACK
```

Do not alter MPS allocator variables unless the experiment explicitly studies them.

Scan logs using token-aware rules. Do not classify ordinary words containing `nan` as non-finite values. Do not classify an unset fallback variable as fallback evidence.

## 11. Memory reporting

Report separately:

- process RSS;
- PyTorch MPS current allocation;
- MPS driver allocation;
- MLX active, peak, and cache counters;
- system memory pressure;
- compressed-memory observations where available;
- swap before and after;
- logical activation bytes;
- logical workspace bytes;
- storage cache or file-size observations.

Never sum these counters as independent physical pools.

A logical memory reduction does not automatically imply a lower RSS.

## 12. Storage reporting

Report:

- logical bytes read and written;
- physical file bytes read and written when available;
- chunk reads and writes;
- chunk reuse;
- store size;
- `du` allocation;
- `df` free space;
- fsync time;
- manifest publication time;
- pruning selected, newly reclaimed, and cumulative reclaimed bytes;
- write amplification or cumulative writes.

Application byte counters are not proof of physical NAND writes or reclamation.

## 13. Required artifact package

Every target run should produce:

```text
REPORT.md
RESULTS.json
COMMANDS.jsonl
logs/
system snapshots/
comparison reports/
plans and profiles when relevant/
external harness scripts/
final Git status and diff outputs/
diagnostics ZIP
```

Exclude:

- virtual environments;
- full repository clones;
- credentials;
- serial numbers and hardware UUIDs;
- unrelated user files;
- large tensor chunks when paths, lengths, references, and checksums are sufficient.

## 14. Source integrity

At the end of every validation:

```bash
git status --short
git diff --check
git diff
```

Required:

```text
git status --short: empty
git diff --check: exit 0
git diff: empty
```

If tracked files changed, preserve the diff externally and classify protocol integrity as failed. Do not silently restore or commit those changes.

## 15. Evidence status

Use these labels in release notes and handoff updates:

### Implemented and accepted on target hardware

Requires:

- clean merged commit;
- CPU CI PASS;
- exact target commit tested;
- target protocol PASS;
- artifacts preserved;
- documentation updated.

### Implemented and accepted in CPU CI only

Target-specific claims are prohibited.

### Implemented but not accepted

A branch or PR exists, but one or more required gates are missing or failing.

### Designed but not implemented

Specification or issue exists without a working source path.

### Not started

No accepted design or implementation exists.

## 16. Release sequence

Recommended sequence:

1. open issue or update milestone specification;
2. create clean implementation branch;
3. add tests with the code;
4. run local CPU gates;
5. open draft PR;
6. obtain Python 3.11 and 3.13 CI PASS;
7. remove temporary diagnostics and make PR ready;
8. merge or test the final immutable PR head, according to project policy;
9. execute M2 protocol on exact commit;
10. analyze artifacts;
11. fix and repeat if necessary;
12. update documentation and close milestone issue only after acceptance.

## 17. Claim boundary template

Every report should end with explicit statements such as:

```text
Validated:
- ...

Not validated:
- ...

This result does not demonstrate:
- ...
```

Never use a broad label such as “out-of-core training” when only one bounded subsystem was tested.
