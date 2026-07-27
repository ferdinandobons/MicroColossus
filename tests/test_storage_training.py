from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from microcolossus.config import (
    ExperimentConfig,
    HardwareBudget,
    ModelConfig,
    TrainingConfig,
)
from microcolossus.storage import FailurePoint, SimulatedCrash, VersionedTensorStore
from microcolossus.storage.adapters import payload_from_torch
from microcolossus.storage.schema import TensorKind
from microcolossus.storage_training import (
    batch_checksum,
    compare_states,
    fail_at,
    run_observable_storage_step,
)


def _config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        name="storage-micro",
        output_dir=str(tmp_path / "unused"),
        model=ModelConfig(
            vocab_size=32,
            max_sequence_length=8,
            layers=1,
            heads=2,
            hidden_size=16,
            mlp_ratio=2,
            dropout=0.0,
        ),
        training=TrainingConfig(
            steps=1,
            micro_batch_size=1,
            sequence_length=4,
            learning_rate=1e-3,
            weight_decay=0.1,
            gradient_clip_norm=1.0,
            seed=11,
            device="cpu",
        ),
        hardware=HardwareBudget(
            accelerator_memory_gib=1.0,
            process_ram_gib=1.0,
            nvme_gib=0.02,
            ssd_write_budget_tb=1.0,
            memory_architecture="unified",
            system_memory_gib=1.0,
        ),
    )


def test_storage_backed_step_matches_resident_and_restores_exactly(
    tmp_path: Path,
) -> None:
    output = tmp_path / "result.json"
    result = run_observable_storage_step(
        _config(tmp_path),
        store_path=tmp_path / "store",
        output_path=output,
        device_override="cpu",
    )

    assert output.exists()
    assert json.loads(output.read_text())["schema_version"].endswith("v1")
    assert result.resident_compute.loss == result.storage_compute.loss
    assert result.resident_compute.gradient_norm == result.storage_compute.gradient_norm
    assert result.resident_vs_storage.exact_bytes
    assert result.resident_vs_storage.maximum_absolute_difference == 0.0
    assert result.storage_vs_restored.exact_bytes
    assert result.storage_final_state == result.restored_final_state
    assert result.initial_manifest_id != result.final_manifest_id
    assert result.initial_store_commit.logical_bytes_written > 0
    assert result.updated_store_commit.logical_bytes_written > 0
    assert result.state_read.logical_bytes == result.initial_state.logical_bytes
    assert any(version == 1 for _tensor_id, version in result.tensor_versions)
    assert any(version == 0 for _tensor_id, version in result.tensor_versions)


def test_repeated_micro_runs_have_identical_state_and_batch_digests(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    first = run_observable_storage_step(
        config,
        store_path=tmp_path / "first",
        device_override="cpu",
    )
    second = run_observable_storage_step(
        config,
        store_path=tmp_path / "second",
        device_override="cpu",
    )

    assert first.batch_checksum == second.batch_checksum
    assert first.initial_state == second.initial_state
    assert first.resident_final_state == second.resident_final_state
    assert first.storage_final_state == second.storage_final_state
    assert first.resident_compute.loss == second.resident_compute.loss
    assert first.storage_compute.loss == second.storage_compute.loss


def test_failed_update_keeps_initial_training_state_authoritative(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "crash-store"
    with pytest.raises(SimulatedCrash):
        run_observable_storage_step(
            _config(tmp_path),
            store_path=store_path,
            device_override="cpu",
            update_failure_injector=fail_at(FailurePoint.BEFORE_CURRENT_RENAME),
        )

    store = VersionedTensorStore.open(store_path)
    assert store.current_manifest().committed_step == 0
    recovery = store.recover()
    assert recovery.incomplete_transactions
    assert recovery.aborted_transactions
    assert store.current_manifest().committed_step == 0


def test_batch_checksum_does_not_require_numpy(monkeypatch) -> None:
    def fail_numpy(_tensor: torch.Tensor) -> None:
        raise AssertionError("batch checksum must not call Tensor.numpy")

    monkeypatch.setattr(torch.Tensor, "numpy", fail_numpy)
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    targets = torch.tensor([[2, 3, 4]], dtype=torch.long)
    assert len(batch_checksum(input_ids, targets)) == 64


def test_state_comparison_reports_tensor_difference() -> None:
    original = payload_from_torch(
        torch.tensor([1.0, 2.0]),
        logical_name="model.weight",
        kind=TensorKind.PARAMETER,
    )
    modified = replace(
        original,
        data=payload_from_torch(
            torch.tensor([1.0, 2.5]),
            logical_name="model.weight",
            kind=TensorKind.PARAMETER,
        ).data,
    )
    comparison = compare_states((original,), (modified,))

    assert not comparison.exact_bytes
    assert comparison.names_equal
    assert comparison.structures_equal
    assert comparison.maximum_absolute_difference == pytest.approx(0.5)
    assert comparison.worst_absolute_tensor == "parameter:model.weight"
