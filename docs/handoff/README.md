# MicroColossus continuation handoff

This directory is the transfer package for an engineer or coding agent taking over MicroColossus.

Snapshot date: **2026-07-29**.

The handoff has four purposes:

1. preserve the original objective and the engineering constraints;
2. distinguish accepted, target-validated work from experiments and failed integration attempts;
3. document the current repository condition precisely;
4. provide an ordered path from the present state to a meaningful larger-than-memory training demonstration.

## Current summary

The latest accepted product baseline is **MicroColossus 0.12.0**. It includes persistent activation recomputation and was validated on an 8 GB Apple M2 at commit `4742f8a7f57a46edb075159275fb66c83c78ced7`.

The accepted documentation baseline before the M6C publication experiments is commit `9f9365e597693e2cffa4f454180203e7219a7cde`.

At the time of this handoff:

- the package on `main` still reports version `0.12.0`;
- M6C, the measured hybrid activation-anchor planner, is not accepted or merged;
- issue `#27` remains the authoritative M6C specification;
- PR `#31` is a broken draft candidate and must not be merged;
- PR `#33` is a temporary payload and workflow branch and must not be merged as product code;
- `main` contains temporary CI and workflow edits from failed M6C publication attempts;
- no final MicroColossus 0.13.0 release exists yet.

The first task for a successor is repository stabilization, not a new M2 benchmark.

## Reading order

Read these files in order:

1. [`01-vision-and-success-criteria.md`](01-vision-and-success-criteria.md)
2. [`02-history-and-validated-evidence.md`](02-history-and-validated-evidence.md)
3. [`03-current-repository-state.md`](03-current-repository-state.md)
4. [`04-architecture-and-invariants.md`](04-architecture-and-invariants.md)
5. [`05-open-work-and-roadmap.md`](05-open-work-and-roadmap.md)
6. [`06-testing-and-release-protocol.md`](06-testing-and-release-protocol.md)
7. [`07-known-failures-and-lessons.md`](07-known-failures-and-lessons.md)
8. [`08-next-ai-operating-manual.md`](08-next-ai-operating-manual.md)
9. [`09-definition-of-done.md`](09-definition-of-done.md)
10. [`AI-CONTINUATION-PROMPT.md`](AI-CONTINUATION-PROMPT.md)

## Source-of-truth priority

When documents disagree, use this order:

1. exact accepted commit and executable evidence;
2. current source code on the branch being tested;
3. this handoff directory;
4. the existing `docs/` specifications;
5. GitHub issue and pull-request descriptions;
6. conversational summaries.

Never infer that a feature exists because it appears in a draft PR, a temporary patch, a prompt, or a roadmap entry.

## Core transfer rule

Every future claim must identify one of these statuses:

- **Implemented and accepted on target hardware**;
- **Implemented and accepted in CPU CI only**;
- **Implemented but not accepted**;
- **Designed but not implemented**;
- **Not started**.

Do not collapse these categories into a single statement such as “completed.”
