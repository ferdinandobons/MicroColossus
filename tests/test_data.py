from __future__ import annotations

from pathlib import Path

import pytest

from microcolossus.config import (
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    HardwareBudget,
    ModelConfig,
    TrainingConfig,
    load_experiment_config,
)
from microcolossus.data import prepare_data_source


def _config(path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        name="real-text-data-test",
        output_dir="runs/test",
        model=ModelConfig(
            vocab_size=256,
            max_sequence_length=16,
            layers=1,
            heads=2,
            hidden_size=16,
            mlp_ratio=2,
        ),
        training=TrainingConfig(
            steps=3,
            micro_batch_size=2,
            sequence_length=8,
            learning_rate=1e-3,
            seed=17,
            device="cpu",
        ),
        hardware=HardwareBudget(
            accelerator_memory_gib=1,
            process_ram_gib=1,
            nvme_gib=1,
            ssd_write_budget_tb=1,
        ),
        data=DataConfig(kind="utf8_text", train_path=str(path), validation_fraction=0.2),
        evaluation=EvaluationConfig(
            enabled=True,
            interval_steps=1,
            validation_batches=2,
            sample_tokens=4,
        ),
    )


def test_utf8_text_source_is_deterministic_and_roundtrips(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("A small river crosses the valley.\n" * 40, encoding="utf-8")
    source = prepare_data_source(_config(corpus))

    first = source.training_batch(0)
    repeated = source.training_batch(0)
    later = source.training_batch(1)

    assert first.offsets == repeated.offsets
    assert first.offsets != later.offsets
    assert first.input_ids.equal(repeated.input_ids)
    assert first.targets.equal(repeated.targets)
    assert source.decode_tokens(source.encode_text("città e fiume")) == "città e fiume"
    assert source.identity.source_kind == "utf8_text"
    assert source.identity.train_byte_count > source.identity.validation_byte_count
    source.identity.validate()


def test_relative_corpus_path_resolves_from_yaml_directory(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "corpus.txt").write_text("light and stone\n" * 20, encoding="utf-8")
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        """
name: text
output_dir: runs/text
model:
  vocab_size: 256
  max_sequence_length: 16
  layers: 1
  heads: 2
  hidden_size: 16
  mlp_ratio: 2
training:
  steps: 2
  micro_batch_size: 1
  sequence_length: 8
  learning_rate: 0.001
  seed: 5
  device: cpu
data:
  kind: utf8_text
  train_path: data/corpus.txt
  validation_fraction: 0.2
evaluation:
  enabled: true
hardware:
  accelerator_memory_gib: 1
  process_ram_gib: 1
  nvme_gib: 1
  ssd_write_budget_tb: 1
""",
        encoding="utf-8",
    )

    config = load_experiment_config(config_path)
    assert config.data.train_path == str((data_dir / "corpus.txt").resolve())
    assert prepare_data_source(config).identity.train_byte_count > 0


def test_corpus_identity_changes_when_source_bytes_change(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("one two three\n" * 30, encoding="utf-8")
    config = _config(corpus)
    before = prepare_data_source(config).identity.identity_checksum
    corpus.write_text("one two three four\n" * 30, encoding="utf-8")
    after = prepare_data_source(config).identity.identity_checksum
    assert before != after


def test_utf8_text_requires_byte_vocabulary(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("enough text for both deterministic splits\n" * 10, encoding="utf-8")
    with pytest.raises(ValueError, match="vocab_size"):
        ExperimentConfig(
            name="invalid",
            output_dir="runs/invalid",
            model=ModelConfig(vocab_size=128),
            training=TrainingConfig(),
            hardware=HardwareBudget(),
            data=DataConfig(kind="utf8_text", train_path=str(corpus)),
        )
