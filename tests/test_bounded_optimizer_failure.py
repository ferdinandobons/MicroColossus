from __future__ import annotations

from pathlib import Path

import pytest

from microcolossus.bounded_optimizer import run_bounded_optimizer_step
from microcolossus.config import (
    ExperimentConfig,
    HardwareBudget,
    ModelConfig,
    TrainingConfig,
)
from microcolossus.step_bundle import (
    BundleFailurePoint,
    BundleSimulatedCrash,
    StepBundleStore,
)
from microcolossus.storage import VersionedTensorStore


def _config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        name="bounded-optimizer-failure",
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
            seed=17,
            device="cpu",
        ),
        hardware=HardwareBudget(
            accelerator_memory_gib=1.0,
            process_ram_gib=1.0,
            nvme_gib=0.05,
            ssd_write_budget_tb=1.0,
            memory_architecture="unified",
            system_memory_gib=1.0,
        ),
    )


@pytest.mark.parametrize(
    "failure_point",
    [
        BundleFailurePoint.BEFORE_MANIFEST_RENAME,
        BundleFailurePoint.BEFORE_CURRENT_RENAME,
    ],
)
def test_failed_final_bundle_publish_preserves_step_zero(
    tmp_path: Path,
    failure_point: BundleFailurePoint,
) -> None:
    def fail(point: BundleFailurePoint, context: dict[str, object]) -> None:
        del context
        if point is failure_point:
            raise BundleSimulatedCrash(f"simulated bundle failure at {point.value}")

    bundle_path = tmp_path / "bundle"
    with pytest.raises(BundleSimulatedCrash):
        run_bounded_optimizer_step(
            _config(tmp_path),
            bundle_store_path=bundle_path,
            device_override="cpu",
            optimizer_working_set_bytes=1024**2,
            bundle_failure_injector=fail,
        )

    bundle = StepBundleStore.open(bundle_path)
    current = bundle.current_manifest()
    assert current.committed_step == 0
    assert bundle.verify().bundle_id == current.bundle_id
    recovery = bundle.recover()
    assert recovery.current_bundle_id == current.bundle_id

    candidate_parameters = VersionedTensorStore.open(
        bundle_path / "candidates" / "step-1-parameters"
    )
    candidate_optimizer = VersionedTensorStore.open(
        bundle_path / "candidates" / "step-1-optimizer"
    )
    assert candidate_parameters.verify().tensor_count == 12
    assert candidate_optimizer.verify().tensor_count == 37
