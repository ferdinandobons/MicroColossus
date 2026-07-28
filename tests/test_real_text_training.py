from __future__ import annotations

import math
from pathlib import Path

import pytest

from microcolossus.bounded_training import ResumeConfigurationError, run_bounded_training
from microcolossus.config import (
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    HardwareBudget,
    ModelConfig,
    TrainingConfig,
)
from microcolossus.step_bundle import StepBundleStore
from microcolossus.storage import VersionedTensorStore
from microcolossus.storage_training import compare_states
from microcolossus.training import run_resident_experiment


def _write_corpus(path: Path) -> None:
    path.write_text(
        (
            "the river turns beside the old observatory.\n"
            "the lantern waits beside the quiet door.\n"
            "the river carries light through the valley.\n"
            "the observatory opens when the bells are still.\n"
        )
        * 30,
        encoding="utf-8",
    )


def _config(corpus: Path, output: Path, *, interval: int = 1) -> ExperimentConfig:
    return ExperimentConfig(
        name="real-text-bounded-micro",
        output_dir=str(output),
        model=ModelConfig(
            vocab_size=256,
            max_sequence_length=16,
            layers=1,
            heads=2,
            hidden_size=16,
            mlp_ratio=2,
            dropout=0.0,
        ),
        training=TrainingConfig(
            steps=6,
            micro_batch_size=1,
            sequence_length=8,
            learning_rate=5e-3,
            weight_decay=0.01,
            gradient_clip_norm=1.0,
            seed=31,
            device="cpu",
        ),
        hardware=HardwareBudget(
            accelerator_memory_gib=1,
            process_ram_gib=1,
            nvme_gib=1,
            ssd_write_budget_tb=1,
            memory_architecture="unified",
            system_memory_gib=1,
        ),
        data=DataConfig(
            kind="utf8_text",
            train_path=str(corpus),
            validation_fraction=0.2,
        ),
        evaluation=EvaluationConfig(
            enabled=True,
            interval_steps=interval,
            validation_batches=2,
            sample_tokens=8,
            sample_prompt="the ",
        ),
    )


def _current_state(path: Path):
    bundle = StepBundleStore.open(path)
    current = bundle.current_manifest()
    stores = [
        VersionedTensorStore.open(path / current.parameter_store.path),
        VersionedTensorStore.open(path / current.optimizer_store.path),
    ]
    return tuple(
        sorted(
            (
                store.read_tensor(record.tensor_id)
                for store in stores
                for record in store.current_manifest().tensors
            ),
            key=lambda item: item.logical_name,
        )
    )


def test_real_text_training_resume_and_progress_are_deterministic(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    _write_corpus(corpus)
    config = _config(corpus, tmp_path / "resident")

    uninterrupted = run_bounded_training(
        config,
        bundle_store_path=tmp_path / "uninterrupted",
        target_step=3,
        device_override="cpu",
        optimizer_working_set_bytes=1024**2,
    )
    run_bounded_training(
        config,
        bundle_store_path=tmp_path / "resumed",
        target_step=1,
        device_override="cpu",
        optimizer_working_set_bytes=1024**2,
    )
    resumed = run_bounded_training(
        config,
        bundle_store_path=tmp_path / "resumed",
        target_step=3,
        device_override="cpu",
        optimizer_working_set_bytes=1024**2,
    )

    assert uninterrupted.data_identity.source_kind == "utf8_text"
    assert resumed.resumed
    assert compare_states(
        _current_state(tmp_path / "uninterrupted"),
        _current_state(tmp_path / "resumed"),
    ).exact_bytes
    assert [item.step for item in uninterrupted.progress_records] == [0, 1, 2, 3]
    assert [item.step for item in resumed.progress_records] == [0, 1, 2, 3]
    assert all(item.evaluation is not None for item in uninterrupted.progress_records)
    assert all(
        item.evaluation is not None and math.isfinite(item.evaluation.validation_loss)
        for item in uninterrupted.progress_records
    )
    assert all(item.batch_offsets for item in uninterrupted.progress_records[1:])
    assert (
        uninterrupted.progress_records[-1].evaluation.sample_token_ids
        == resumed.progress_records[-1].evaluation.sample_token_ids
    )
    assert uninterrupted.final_bundle_vs_restored_state.exact_bytes
    assert len(list((tmp_path / "uninterrupted" / "metrics").glob("step-*.json"))) == 4


def test_resume_rejects_modified_corpus(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    _write_corpus(corpus)
    config = _config(corpus, tmp_path / "resident")
    root = tmp_path / "bundle"
    run_bounded_training(
        config,
        bundle_store_path=root,
        target_step=1,
        device_override="cpu",
        optimizer_working_set_bytes=1024**2,
    )
    corpus.write_text(corpus.read_text(encoding="utf-8") + "changed\n", encoding="utf-8")

    with pytest.raises(ResumeConfigurationError, match="data_identity"):
        run_bounded_training(
            config,
            bundle_store_path=root,
            target_step=2,
            device_override="cpu",
            optimizer_working_set_bytes=1024**2,
        )
    assert StepBundleStore.open(root).current_manifest().committed_step == 1


def test_evaluation_interval_keeps_contiguous_progress(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    _write_corpus(corpus)
    result = run_bounded_training(
        _config(corpus, tmp_path / "resident", interval=2),
        bundle_store_path=tmp_path / "bundle",
        target_step=3,
        device_override="cpu",
        optimizer_working_set_bytes=1024**2,
    )
    assert [item.step for item in result.progress_records] == [0, 1, 2, 3]
    assert [item.evaluation is not None for item in result.progress_records] == [
        True,
        False,
        True,
        False,
    ]


def test_resident_real_text_training_changes_loss(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abc abc abc abc abc abc abc abc\n" * 50, encoding="utf-8")
    config = _config(corpus, tmp_path / "resident")
    metrics = run_resident_experiment(config, steps_override=12, device_override="cpu")
    assert all(math.isfinite(item.loss) for item in metrics)
    assert metrics[-1].loss < metrics[0].loss
