# Current repository state

Snapshot date: **2026-07-29**.

This file describes the repository before the handoff documentation is merged.

## 1. Repository

```text
repository:     ferdinandobons/MicroColossus
default branch: main
visibility:     private
```

The authenticated owner has administrative and push access.

## 2. Accepted source baseline

The latest accepted product version is:

```text
MicroColossus 0.12.0
```

The accepted M2 target commit for activation recomputation is:

```text
4742f8a7f57a46edb075159275fb66c83c78ced7
```

The accepted documentation and repository baseline before the M6C publication experiments is:

```text
9f9365e597693e2cffa4f454180203e7219a7cde
```

At handoff time, `pyproject.toml` on `main` still reports `0.12.0`, and the planner console script for 0.13.0 is not present.

## 3. Pre-handoff `main` condition

The pre-handoff `main` head was:

```text
4be4363ad10125f5a59e53a6fdb726f8a3351627
```

Relative to accepted baseline `9f9365e...`, `main` was ahead by 12 commits, but the source implementation remained 0.12.0. The material changes were limited to:

```text
.github/workflows/ci.yml
.github/workflows/apply-m6c-clean-once.yml
```

These commits were attempts to transfer and diagnose an M6C patch. They are not accepted product work.

### 3.1 CI is temporarily unsuitable

The normal Python 3.11 and 3.13 quality matrix was replaced by a branch-specific diagnostic job for `agent/m6c-hybrid-clean`.

Consequences:

- normal pull requests may not run the required project checks;
- the workflow is coupled to temporary payload files;
- the job writes diagnostic comments to PR `#33`;
- the workflow no longer represents the accepted release process.

### 3.2 Temporary privileged workflow

`main` also contains:

```text
.github/workflows/apply-m6c-clean-once.yml
```

It is a one-shot `pull_request_target` workflow with `contents: write`. It reconstructs a patch from `.m6c_payload/part*`, resets a branch to `9f9365e...`, applies the patch, and force-pushes the temporary branch.

This workflow must be removed. It is not part of MicroColossus runtime architecture and should not remain enabled.

## 4. Open M6C issue

Issue `#27` is the authoritative M6C design specification:

```text
M6C: Add a measured hybrid activation-anchor planner
```

The issue correctly requires:

- checksummed measurement profiles;
- deterministic plans;
- retain-all, full-recompute, fixed-interval, and measured-budget baselines;
- nearest-anchor replay;
- activation and workspace budget checks;
- checkpoint-bound plan identity;
- resume, pruning, and failure semantics;
- CPU CI on Python 3.11 and 3.13;
- Apple M2 comparison on micro, tiny, and small workloads.

The issue remains open and should stay open until a clean implementation and target gate pass.

## 5. Open pull requests that must not be merged

### PR #31

```text
title:     Add measured hybrid activation-anchor planning
branch:    agent/m6c-hybrid-activation-planner
head:      2077314ecf55a55d3d788731eaa057dd0ed354df
state:     open draft
```

This branch contains an incomplete and structurally broken candidate. Its M2 diagnostic stopped at the quality gate. It should be treated as a failed experiment and reference material only.

### PR #33

```text
title:     Apply clean M6C implementation
branch:    agent/m6c-hybrid-clean
head:      e83f4b6445a0da81178954f02dcd5c2b2a0d7071
state:     open draft
```

Its changed files are temporary transfer material:

```text
.github/workflows/apply-m6c-clean.yml
.m6c_payload/part01 ... part08
.m6c_payload/trigger-*.txt
```

It does not contain a normal reviewable implementation and must not be merged.

## 6. Required repository stabilization

Before implementing M6C, the successor must:

1. create a clean branch from the accepted baseline or from a cleaned current `main`;
2. restore the normal CI matrix from accepted commit `9f9365e...`;
3. remove `.github/workflows/apply-m6c-clean-once.yml`;
4. remove all patch-transfer or trigger files from any product branch;
5. confirm `main` source behavior still matches 0.12.0;
6. run package import, Ruff, mypy, pytest, compileall, and CPU smoke;
7. close PR `#31` and PR `#33` as abandoned after preserving useful notes;
8. keep issue `#27` open;
9. implement M6C through ordinary source commits, not self-modifying workflows;
10. require a clean PR diff before requesting M2 execution.

## 7. Safe starting points

Two valid approaches exist.

### Approach A. Restore the accepted baseline

Create a new branch from:

```text
9f9365e597693e2cffa4f454180203e7219a7cde
```

Then implement M6C cleanly and open a new PR to a restored `main`.

### Approach B. Clean current `main`

Revert or replace only the temporary workflow changes until the tree is equivalent to `9f9365e...` for CI and runtime source, then branch normally.

Approach A is easier to reason about. Approach B preserves later documentation commits if any are intentionally retained.

## 8. What not to do

Do not:

- merge PR `#31` or PR `#33`;
- treat the checksummed patch payload as accepted source;
- run another M2 test against the broken commit;
- keep a `pull_request_target` workflow that writes to a contributor branch;
- claim 0.13.0 because a draft file contains that version string;
- repair the branch through hidden or self-modifying CI steps;
- bypass Ruff, mypy, pytest, or package-import failures.
