# Validation Policy and Capacity-Mode Preparation

This document records the first Phase 3 work toward running capacity-oriented
persistent training without validation-only full-state materialization in the
claimed path.

## 1. Motivation

MicroColossus development runs historically used complete resident comparisons
after bounded execution:

- per-step resident optimizer oracle;
- resident-versus-candidate parameter and Adam state comparison;
- complete candidate restore into a model and optimizer;
- final resident replay from step zero;
- final bundle-versus-restored state comparison.

Those checks are valuable for micro, tiny, and small development gates. They
are not valid inside a future larger-than-memory capacity claim because they
materialize complete parameter plus optimizer state in one process.

## 2. Validation levels

Persistent bounded training now has an explicit validation level:

```yaml
training:
  validation_level: full
```

or:

```yaml
training:
  validation_level: integrity_only
```

The CLI can also override the YAML:

```bash
microcolossus-bounded-train \
  --config examples/micro-storage.yaml \
  --bundle-store runs/integrity-only \
  --target-step 2 \
  --output runs/integrity-only-step-2.json \
  --device cpu \
  --validation-level integrity_only
```

## 3. `full`

`full` is the default and preserves the established validation behavior.

It still performs:

- per-step resident optimizer oracle;
- resident-versus-candidate state comparison;
- candidate-versus-restored state comparison;
- final resident replay from step zero;
- final bundle-versus-restored state comparison.

Use `full` for micro and small correctness work, release-quality numerical
gates, and target validation where complete state fits safely.

## 4. `integrity_only`

`integrity_only` skips complete parameter plus optimizer state validation in
the persistent trainer.

It omits:

- per-step resident optimizer oracle;
- resident-versus-candidate state comparison;
- candidate-versus-restored state comparison;
- final resident replay from step zero;
- final bounded-versus-resident state comparison;
- final bundle-versus-restored state comparison.

It preserves:

- authoritative root publication;
- candidate parameter and optimizer store verification;
- final bundle verification;
- checkpoint lineage checks;
- progress-record and bundle-ID alignment;
- batch cursor derivation from committed step;
- data identity and resume checks;
- activation policy, profile, and plan identity checks;
- optimizer step tensor inspection without restoring the full model.

The result JSON records `validation_level`, `validation_omitted_checks`, and
`null` comparison fields for skipped full-state checks.

## 5. Current boundary

This is **implemented but not accepted** until the branch receives full local
gates and GitHub Actions.

It is not yet a complete larger-than-memory validation path. The lower-level
backward routines still use resident gradient oracle validation, and
evaluation or generation can still materialize a resident model. Future Phase 3
work must either disable, sample, stream, or externalize those paths before a
capacity demonstration can rely on `integrity_only`.

## 6. Acceptance target

The Phase 3 acceptance target remains:

```text
capacity path executes without complete parameter plus optimizer state
materialization in one process
authoritative storage and checkpoint integrity remain verified
all omitted checks are explicitly reported
full validation remains the default for development workloads
```
