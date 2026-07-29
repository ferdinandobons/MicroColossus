# Operating manual for the successor AI

This file describes how a new coding agent should take over the repository without repeating the failed M6C integration process.

## 1. Mission

The successor owns implementation and GitHub publication.

The project owner provides the Apple M2 for target execution. In the current
operating model, the same coding agent runs those checks directly when it has
access to the Mac:

```text
coding agent:
    design
    implementation
    tests
    documentation
    GitHub branches and PRs
    CI diagnosis
    fresh checkout
    exact-commit execution
    MPS and APFS tests
    telemetry
```

Target verification itself remains exact-commit scoped and must not modify
tracked product source.

## 2. First actions

Before writing feature code:

1. read every file in `docs/handoff/`;
2. inspect `README.md` and the existing `docs/` directory;
3. verify `pyproject.toml` reports 0.12.0 on the accepted baseline;
4. inspect issue `#27`;
5. inspect PR `#31` and PR `#33` only as failed historical material;
6. compare current `main` with `9f9365e597693e2cffa4f454180203e7219a7cde`;
7. restore normal CI;
8. remove temporary workflows;
9. run a fresh CPU quality gate;
10. record the stabilized commit in the handoff documents.

Do not begin M6C runtime work while CI is branch-specific or package import is failing.

## 3. Repository workflow

Use normal Git history.

Recommended sequence:

```text
main or accepted baseline
    -> agent/stabilize-repository
    -> PR and CI
    -> merge

clean main
    -> agent/m6c-profile-and-planner
    -> PR and CI
    -> merge

clean main
    -> agent/m6c-runtime
    -> PR and CI
    -> merge or continue in one well-scoped feature PR
```

A single M6C PR is acceptable if it remains reviewable and always passes import and compile checks. Multiple smaller PRs are safer.

### Forbidden workflow

Do not use:

- patch payload parts committed to the repository;
- CI that generates and force-pushes feature source;
- temporary replacement of normal CI;
- hidden source mutations in diagnostics;
- direct commits to `main` for feature implementation.

## 4. Status reporting

Every progress update should state:

```text
Current branch:
Current commit:
Package version:
Implemented:
Validated locally:
Validated in CI:
Validated on M2:
Known failures:
Next action:
Mac action required now: yes or no
```

Do not use “done” without specifying the validation level.

## 5. Decision process

For each proposed optimization:

1. define the bottleneck;
2. identify the baseline;
3. define the invariant that must not change;
4. define the metric expected to improve;
5. implement the smallest controlled version;
6. add a rejection or failure test;
7. compare numerical state;
8. measure on CPU where useful;
9. run target hardware only after CI;
10. document favorable and unfavorable results.

A result that is correct but slower or larger is still useful evidence. Do not tune the narrative to hide it.

## 6. M6C implementation order

Use this order.

### Step 1. Pure profile types

Implement only:

- profile dataclasses;
- canonical JSON;
- checksum;
- validation;
- deterministic model signature;
- round-trip and mutation tests.

No runtime dependency should be introduced here beyond static model metadata.

### Step 2. Pure planner

Implement:

- schedule representation;
- retain-all summary;
- recompute summary;
- fixed-interval summary;
- dynamic measured-budget selection;
- infeasible-plan serialization;
- deterministic checksum;
- exact small-search tests.

Keep the planner side-effect free.

### Step 3. Controlled nearest-anchor backward

Add a standalone one-step validation function using an existing authoritative parameter store.

Compare gradients and local inputs with the accepted policies before integrating persistent training.

### Step 4. Persistent step integration

Add:

- hybrid path selection;
- plan identity;
- result telemetry;
- candidate state;
- exact restore;
- root publication.

Preserve `retain_all` and `recompute` tests unchanged.

### Step 5. Resume and pruning

Add:

- root copy of plan or canonical plan identity;
- changed-plan rejection;
- process restart;
- pruning followed by resume;
- plan preservation after pruning.

### Step 6. Failure injection

Test both root publication boundaries and any new plan-publication boundary.

### Step 7. CLI, examples, release surface

Only after runtime tests pass:

- export APIs;
- add planner CLI;
- add official examples;
- update version;
- add release contract tests;
- update docs.

## 7. Test execution strategy

Run the fastest tests first.

Recommended loop:

```bash
python -m compileall -q microcolossus
ruff check <changed files>
mypy <changed modules>
python -m pytest <focused tests>
```

Before PR readiness:

```bash
ruff check .
mypy microcolossus
python -m pytest
python -m compileall -q microcolossus
```

Use separate processes for memory-heavy PyTorch test groups when local constraints require it, but the official CI must still run the normal suite.

## 8. Mac prompt policy

Do not ask the owner to execute a Mac prompt until:

- exact feature commit exists remotely;
- package import passes from a fresh environment;
- package version is correct;
- all required CLI commands are installed;
- Ruff, mypy, pytest, and compileall pass;
- normal CI is green on Python 3.11 and 3.13;
- repository contains no temporary transfer file or workflow;
- the prompt names one immutable commit;
- expected parameter counts and budgets are derived from actual plans.

The prompt should stop at the quality gate if any check fails.

## 9. Handling target results

When the owner returns a target report:

1. do not accept the report based only on its headline;
2. inspect exact commit and version;
3. distinguish protocol failure from product failure;
4. inspect quality gate;
5. inspect numerical state and provenance;
6. inspect restore and root authority;
7. inspect memory and storage measurements;
8. inspect fallback and non-finite scan;
9. inspect final Git state;
10. update docs only after classification.

If the harness was corrected externally, preserve both runs and use the clean rerun as the decision evidence.

## 10. Documentation obligation

Every milestone must update:

- `README.md` for user-facing capability and boundary;
- `docs/project.md` for architecture and roadmap;
- `docs/validation.md` for exact evidence;
- the feature-specific document;
- this handoff directory when status or next actions materially change.

Record:

- exact accepted commit;
- target machine;
- numerical bands;
- important raw measurements;
- failures and corrections;
- explicit non-validated features.

## 11. Progress optimization

To keep development fast:

- use synthetic micro for math and failures;
- use real-text micro for data and persistent semantics;
- use tiny for MPS numerical gates;
- use small for memory trade-offs;
- avoid large repeated training until the runtime path is stable;
- inspect training internally with per-step evidence;
- separate correctness runs from performance runs;
- prune diagnostic roots between independent scenarios when safe;
- package evidence without large chunks.

## 12. Escalation rules

Stop and explain the blocker when:

- GitHub writes cannot create normal source commits;
- CI cannot be restored safely;
- a required dependency cannot be installed;
- the only available path would weaken an invariant;
- a target protocol would require changing product source;
- free storage is insufficient for the required gate;
- a result is ambiguous and a smaller diagnostic can isolate it.

Do not work around a tooling limitation by adding permanent infrastructure unrelated to the product.

## 13. Completion behavior

Continue autonomously through implementation, tests, PR publication, and CI fixes.

Pause for the project owner only when one of these is required:

- execution on the Apple M2;
- access to a separate external training project or corpus;
- a product decision that changes the declared objective;
- credentials or permissions that cannot be resolved through available tools.

When Mac execution is required, provide one complete copy-paste prompt and state the exact commit being tested.
