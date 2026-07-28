# Deterministic Real-Text Training

This document records the MicroColossus 0.10 design and accepted Apple M2 evidence for training on local UTF-8 text while preserving the bounded execution, versioned state, atomic checkpoint, and process-resume contracts validated in earlier releases.

## 1. Scope

Version 0.10 adds the first real-data frontend. It is deliberately small and dependency-free so runtime correctness can be verified before integrating large corpora or external tokenizer libraries.

The first implementation supports:

- local UTF-8 text files;
- a fixed byte-level tokenizer with token IDs `0..255`;
- one source file with a deterministic tail validation split, or separate train and validation files;
- deterministic random-access training windows;
- fixed validation windows;
- validation loss at configurable committed steps;
- deterministic greedy sample generation;
- atomic progress records tied to committed root bundles;
- corpus and tokenizer identity in persistent training metadata;
- process restart and resume without changing the data trajectory.

It does not establish production model quality, large-corpus throughput, activation-bounded execution, or training state larger than unified memory.

## 2. Byte tokenizer

The tokenizer contract is:

```text
tokenizer version: utf8-bytes-v1
vocabulary size:   256
encode:             UTF-8 bytes -> integer token IDs
decode:             integer token IDs -> UTF-8 with replacement on invalid sequences
```

The tokenizer has no learned vocabulary and no external files. Its identity is still explicit because a future tokenizer must participate in the same checkpoint provenance rules.

A text configuration therefore requires:

```text
model.vocab_size = 256
```

## 3. Corpus identity

Every prepared source produces a checksummed `DataIdentity` containing:

```text
schema version
source kind
tokenizer version
batch-stream version
split policy
train SHA-256
validation SHA-256
train byte count
validation byte count
identity SHA-256
```

Paths are not treated as identity. File bytes and the declared policies are authoritative.

A resumed run reloads the configured files and recomputes the identity. Changed corpus bytes, tokenizer identity, sampler policy, or split policy cause resume rejection before another optimizer step is published.

## 4. Deterministic split

Two layouts are supported.

### 4.1 One input file

A deterministic tail split is used:

```text
train       = leading bytes
validation  = trailing validation_fraction bytes
```

Both partitions must contain at least `sequence_length + 1` bytes.

### 4.2 Separate input files

The train and validation files are read independently. Their exact bytes and checksums are persisted in the data identity.

## 5. Training windows

For committed cursor `N`, the training source uses:

```text
seed = training seed + 1 + N
```

That generator selects one start offset per microbatch row. Each row contains `sequence_length + 1` consecutive bytes and is split into shifted next-token inputs and targets.

The root committed step remains the authoritative next-batch cursor. There is no separately mutable dataset cursor file.

The progress record for each committed step includes:

- consumed cursor;
- derived seed;
- selected byte offsets;
- batch checksum;
- source kind;
- training loss;
- gradient norm;
- clipping coefficient.

## 6. Validation

Validation uses a separate deterministic seed range and fixed cursors beginning at zero. A configured number of validation batches is evaluated against the parameter store referenced by the committed root bundle.

Each evaluation records:

- committed step and bundle ID;
- data identity checksum;
- token-weighted validation loss;
- validation batch checksums;
- evaluated token count;
- elapsed time;
- RSS;
- accelerator and driver counters.

Validation materializes a resident model from the committed parameter store. This is an explicit evaluation path and is not included in bounded execution claims.

## 7. Sample generation

The first generator is deterministic greedy decoding:

```text
next token = argmax(final logits)
```

The configured prompt is encoded with the byte tokenizer. When no prompt is supplied, a prefix from the validation split is used.

Generated token IDs and decoded text are recorded together. Byte-level output may contain replacement characters, especially early in training. Sample readability is not a correctness criterion for the first gate.

## 8. Progress records

Progress files are stored under:

```text
training-root/
  metrics/
    step-00000000.json
    step-00000001.json
    ...
```

Each file is written through:

```text
temporary file
  -> file fsync
  -> atomic rename
  -> directory fsync
```

A progress record references one root bundle ID. The sequence must be contiguous and its bundle IDs must match root lineage.

The root bundle remains authoritative training state. Progress records are derived evidence. A process interrupted after root publication can recreate the missing current-step progress record on resume.

## 9. Configuration

Example data configuration:

```yaml
data:
  kind: utf8_text
  train_path: data/micro-corpus.txt
  validation_fraction: 0.15
  tokenizer: utf8-bytes-v1
  sampler: random-window-v1
```

Example evaluation configuration:

```yaml
evaluation:
  enabled: true
  interval_steps: 10
  validation_batches: 4
  sample_tokens: 64
  sample_prompt: "The workshop "
  generation: greedy-v1
```

Relative corpus paths are resolved from the experiment YAML directory.

## 10. Development workloads

### 10.1 `examples/real-text-micro.yaml`

- 18,624 parameters;
- byte vocabulary of 256 tokens;
- maximum positional table length 64;
- one Transformer block;
- training sequence length 32;
- microbatch 2;
- evaluation every committed step;
- intended for CI, CPU comparison, resume tests, and fast MPS diagnostics.

The older 11,456-parameter count belongs to the synthetic micro configuration with vocabulary size 64 and a shorter positional table. It does not apply to the real-text byte-tokenizer configuration.

### 10.2 `examples/real-text-small.yaml`

- 1,846,656 parameters;
- four Transformer blocks;
- hidden size 192;
- sequence length 128;
- evaluation every ten committed steps;
- intended for the first meaningful Apple M2 real-text learning trajectory.

The included corpus is original project text. It is a reproducible engineering fixture, not a representative language-model dataset.

## 11. Accepted Apple M2 validation

Accepted runtime commit:

```text
8bc277123267c3d3f15bf60cd640819fa823d2e3
```

Package version:

```text
0.10.0
```

The external report classified the release as formal `FAIL` only because its prompt expected the obsolete synthetic micro count of 11,456 parameters. The checked commit correctly planned 18,624 parameters for `real-text-micro.yaml`. The mismatch was in the validation protocol, not in the runtime or configuration. After correcting that protocol expectation, the target result is accepted as **PASS with a documented protocol correction**.

### 11.1 Environment and quality gate

- MacBook Air with Apple M2 and 8 GB unified memory;
- native arm64 without Rosetta;
- PyTorch 2.13.0 with MPS built and available;
- `PYTORCH_ENABLE_MPS_FALLBACK` unset;
- Ruff passed;
- mypy passed;
- pytest passed with 88 tests and 1 skip;
- compileall passed;
- doctor passed;
- final `git status --short` was empty;
- final `git diff --check` was clean.

### 11.2 Data identity and tokenizer

- independent-process data identity matched exactly;
- tokenizer version was `utf8-bytes-v1`;
- token range and UTF-8 round trip passed;
- training cursors, seeds, offsets, and batch checksums matched between uninterrupted and resumed execution;
- changed corpus bytes were rejected with `ResumeConfigurationError: data_identity` before step 3 could become authoritative.

### 11.3 Micro real-text trajectory

Uninterrupted training reached step 20 and was GREEN.

Validation loss decreased from:

```text
step 0:  5.548418998718262
step 20: 3.302267074584961
```

The learning-signal classification was `LEARNING_SIGNAL_GREEN`.

A separate root trained from step 0 to step 5, exited, and resumed in a new process from step 5 to step 20. The comparison was `NUMERICALLY_STABLE`:

```text
maximum absolute state difference: 1.1920928955078125e-07
mean absolute state difference:    8.844825718731097e-10
all values finite:                 true
names and structures equal:        true
```

Training-loss, validation-loss, gradient-norm, and clipping trajectories were close. Batch provenance, sample token IDs, and decoded sample completion were identical. Candidate-versus-restored state was exact at every inspected step.

### 11.4 Small real-text trajectory

The 1,846,656-parameter model reached step 10 and was GREEN.

Validation loss decreased from:

```text
step 0:  5.687370777130127
step 10: 4.083975553512573
```

The learning-signal classification was `LEARNING_SIGNAL_GREEN`.

Final bounded-versus-resident state remained numerically close:

```text
maximum absolute difference: 1.1920928955078125e-07
mean absolute difference:    1.2758380908789405e-10
candidate restore exact:     true
```

### 11.5 Lineage, storage, and resources

- micro lineage and progress records were contiguous from step 0 through step 20;
- small lineage and progress records were contiguous from step 0 through step 10;
- all root and referenced child stores verified and recovered;
- micro training root size was `25,805,954` bytes;
- small training root size was `631,729,632` bytes;
- maximum recorded RSS was `393,314,304` bytes;
- maximum recorded accelerator allocation was `62,586,880` bytes;
- no runtime fallback, unsupported operator, non-finite value, or unexpected command failure remained after auditing scanner false positives;
- free storage declined from about 7.20 GiB before installation to about 3.67 GiB after the small run because historical candidate and work stores are retained.

Application byte counters are not NAND-level SSD-write measurements. Historical-state pruning and compaction were not implemented or tested.

## 12. Current boundary

Version 0.10 does not provide or establish:

- a representative subword tokenizer;
- a representative production corpus or model-quality result;
- downloaded, large, or sharded datasets;
- persisted epoch and shuffle state beyond the deterministic random-access cursor;
- bounded validation or generation;
- activation offload or recomputation;
- strict total-memory-pressure enforcement;
- asynchronous data or storage prefetch;
- storage pruning or compaction;
- bounded MLX training;
- larger-than-memory training;
- 124M or 350M capacity demonstrations.

The accepted result establishes a real-text learning trajectory, deterministic data provenance, process resume, numerical stability, and storage integrity for the tested micro and 1.85M-parameter workloads. It is not yet a full out-of-core or production model-quality result.
