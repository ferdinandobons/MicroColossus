from __future__ import annotations

from pathlib import Path

import pytest

from microcolossus.activation_planner import (
    ActivationPlanInfeasibleError,
    build_activation_plan,
    build_logical_activation_profile,
    load_activation_plan,
    write_activation_plan,
)
from microcolossus.config import (
    ExperimentConfig,
    HardwareBudget,
    ModelConfig,
    TrainingConfig,
)


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        name="planner-test",
        output_dir="unused",
        model=ModelConfig(
            vocab_size=32,
            max_sequence_length=8,
            layers=2,
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
            seed=7,
            device="cpu",
        ),
        hardware=HardwareBudget(
            accelerator_memory_gib=1,
            process_ram_gib=1,
            nvme_gib=1,
            ssd_write_budget_tb=1,
        ),
    )


def test_dynamic_plan_is_deterministic_and_between_extremes(tmp_path: Path) -> None:
    profile = build_logical_activation_profile(_config())
    boundary = profile.groups[0].boundary_bytes
    budget = (2 * boundary) + boundary
    first = build_activation_plan(
        profile,
        activation_working_set_budget_bytes=budget,
        workspace_working_set_budget_bytes=4096,
        max_replay_groups=2,
    )
    second = build_activation_plan(
        profile,
        activation_working_set_budget_bytes=budget,
        workspace_working_set_budget_bytes=4096,
        max_replay_groups=2,
    )

    assert first.plan_checksum == second.plan_checksum
    assert first.to_dict() == second.to_dict()
    assert len(first.anchor_group_names) == 1
    assert first.total_replayed_groups < first.recompute_baseline.total_replayed_groups
    assert first.total_replayed_groups > first.retain_all_baseline.total_replayed_groups
    assert first.maximum_retained_anchor_bytes == boundary
    assert first.maximum_replayed_groups <= 2

    path = tmp_path / "plan.json"
    write_activation_plan(path, first)
    restored = load_activation_plan(path)
    assert restored == first


def test_infeasible_activation_budget_is_reported() -> None:
    profile = build_logical_activation_profile(_config())
    plan = build_activation_plan(
        profile,
        activation_working_set_budget_bytes=1,
        workspace_working_set_budget_bytes=4096,
    )

    assert not plan.feasible
    assert plan.rejection_reason is not None
    with pytest.raises(ActivationPlanInfeasibleError):
        plan.validate()


def test_fixed_interval_schedule_is_diagnostic_baseline() -> None:
    profile = build_logical_activation_profile(_config())
    plan = build_activation_plan(
        profile,
        policy_kind="fixed_interval_v1",
        activation_working_set_budget_bytes=4096,
        workspace_working_set_budget_bytes=4096,
        fixed_interval=2,
    )

    assert plan.anchor_group_names == ("block-0",)
    assert plan.fixed_interval_baseline.anchor_group_names == ("block-0",)
