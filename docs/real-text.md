# Deterministic Real-Text Training

This document records the MicroColossus 0.10 design for training on local UTF-8 text while preserving the bounded execution, versioned state, atomic checkpoint, and process-resume contracts validated in 0.9.

## 1. Scope

Version 0.10 adds a first real-data frontend. It is deliberately small and dependency-free so runtime correctness can be verified before integrating large corpora or external tokenizer libraries.

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

It does not yet establish model quality, large-corpus throughput, activation-bounded execution, or training state larger than unified memory.

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

- 11,456 parameters;
- one Transformer block;
- sequence length 32;
- microbatch 2;
- evaluation every committed step;
- intended for CI, exact CPU comparison, resume tests, and fast MPS diagnostics.

### 10.2 `examples/real-text-small.yaml`

- approximately 1.85 million parameters;
- four Transformer blocks;
- hidden size 192;
- sequence length 128;
- evaluation every ten committed steps;
- intended for the first meaningful Apple M2 learning-trajectory experiment after the micro gate passes.

The included corpus is original project text. It is a reproducible engineering fixture, not a representative language-model dataset.

## 11. Validation plan

CPU CI requires:

- deterministic encoding, decoding, split, offsets, and checksums;
- relative path resolution;
- corpus identity changing when source bytes change;
- modified corpus rejected on resume;
- uninterrupted and resumed final state equality;
- contiguous progress and root lineage;
- validation loss and sample records tied to committed bundle IDs;
- evaluation interval behavior;
- resident loss changing on a repetitive real corpus;
- all existing synthetic tests remaining valid.

The Apple Silicon gate will then verify:

- native MPS execution with fallback disabled;
- micro uninterrupted and process-resumed real-text trajectories;
- exact or numerically stable final-state comparison;
- validation-loss and sample determinism;
- corpus mutation rejection;
- root and child-store verification and recovery;
- memory, swap, storage growth, and cumulative writes;
- a short approximately 1.85M-parameter run only after the micro gate succeeds.

## 12. Current boundary

Version 0.10 does not yet provide:

- subword tokenization;
- downloaded or sharded datasets;
- persisted epoch and shuffle state beyond the deterministic random-access cursor;
- bounded validation or generation;
- activation offload or recomputation;
- asynchronous data or storage prefetch;
- storage pruning or compaction;
- bounded MLX training;
- larger-than-memory training;
- a model-quality claim.

The next accepted result must report both training behavior and runtime costs. A decreasing training loss alone is not sufficient without checkpoint, resume, numerical, memory, storage, and recovery evidence.
