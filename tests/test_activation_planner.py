from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

import microcolossus
from microcolossus.activation_planner import (
    ACTIVATION_PLAN_SCHEMA_VERSION,
    ACTIVATION_PLANNER_VERSION,
    ACTIVATION_PROFILE_SCHEMA_VERSION,
    ActivationMeasurementProfile,
    ActivationPlan,
    ActivationPlanIntegrityError,
    ActivationProfileIntegrityError,
    build_activation_measurement_profile,
    build_activation_plan,
    load_activation_plan,
    load_activation_profile,
    write_activation_plan,
    write_activation_profile,
)
from microcolossus.activation_planner_cli import build_parser, main
from microcolossus.bounded_training import run_bounded_training
from microcolossus.config import load_experiment_config


def _config():
    return load_experiment_config(Path("examples/micro-storage.yaml"))


def test_activation_planner_public_api() -> None:
    assert microcolossus.ACTIVATION_PROFILE_SCHEMA_VERSION == (
        ACTIVATION_PROFILE_SCHEMA_VERSION
    )
    assert microcolossus.ACTIVATION_PLAN_SCHEMA_VERSION == ACTIVATION_PLAN_SCHEMA_VERSION
    assert microcolossus.ACTIVATION_PLANNER_VERSION == ACTIVATION_PLANNER_VERSION
    assert microcolossus.ActivationMeasurementProfile is ActivationMeasurementProfile
    assert microcolossus.ActivationPlan is ActivationPlan
    assert microcolossus.build_activation_measurement_profile is (
        build_activation_measurement_profile
    )
    assert microcolossus.build_activation_plan is build_activation_plan


def test_profile_round_trip_is_deterministic(tmp_path: Path) -> None:
    first = build_activation_measurement_profile(_config())
    second = build_activation_measurement_profile(_config())

    assert first.schema_version == ACTIVATION_PROFILE_SCHEMA_VERSION
    assert first.profile_checksum == second.profile_checksum
    assert first.model_signature_checksum == second.model_signature_checksum
    assert tuple(item.name for item in first.groups) == (
        "embedding",
        "block-0",
        "final-head",
    )
    assert first.groups[-1].can_anchor is False
    assert first.groups[0].boundary_bytes == 1024

    path = tmp_path / "profile.json"
    write_activation_profile(path, first)
    loaded = load_activation_profile(path)
    assert loaded.profile_checksum == first.profile_checksum


def test_profile_mutation_is_rejected() -> None:
    profile = build_activation_measurement_profile(_config())
    payload = profile.to_dict()
    payload["groups"][0]["boundary_bytes"] += 4

    with pytest.raises(ActivationProfileIntegrityError):
        ActivationMeasurementProfile.from_dict(payload)


def test_measured_budget_plan_selects_intermediate_anchor(tmp_path: Path) -> None:
    profile = build_activation_measurement_profile(_config())
    plan = build_activation_plan(
        profile,
        activation_budget_bytes=2048,
        workspace_budget_bytes=1024**2,
    )

    assert plan.schema_version == ACTIVATION_PLAN_SCHEMA_VERSION
    assert plan.planner_version == ACTIVATION_PLANNER_VERSION
    assert plan.profile_checksum == profile.profile_checksum
    assert plan.feasible
    assert plan.selected_policy == "measured_budget_v1"
    assert len(plan.selected_anchor_group_names) == 1
    assert "final-head" not in plan.selected_anchor_group_names
    assert plan.maximum_retained_anchor_bytes == 1024
    assert plan.total_replayed_groups == 1
    assert plan.maximum_replay_depth_observed == 1
    summaries = {item.kind: item for item in plan.baseline_summaries}
    assert summaries["retain_all"].feasible is False
    assert summaries["recompute"].total_replayed_groups == 3
    assert plan.total_replayed_groups < summaries["recompute"].total_replayed_groups

    profile_path = tmp_path / "profile.json"
    plan_path = tmp_path / "plan.json"
    write_activation_profile(profile_path, profile)
    write_activation_plan(plan_path, plan)
    loaded_profile = load_activation_profile(profile_path)
    loaded_plan = load_activation_plan(plan_path)
    loaded_plan.validate(loaded_profile)
    assert loaded_plan.plan_checksum == plan.plan_checksum


def test_plan_mutation_is_rejected() -> None:
    profile = build_activation_measurement_profile(_config())
    plan = build_activation_plan(
        profile,
        activation_budget_bytes=2048,
        workspace_budget_bytes=1024**2,
    )
    payload = plan.to_dict()
    payload["total_replayed_groups"] += 1

    with pytest.raises(ActivationPlanIntegrityError):
        ActivationPlan.from_dict(payload)


def test_activation_plan_cli_writes_profile_and_plan(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = main(
        [
            "--config",
            "examples/micro-storage.yaml",
            "--profile-output",
            str(tmp_path / "profile.json"),
            "--plan-output",
            str(tmp_path / "plan.json"),
            "--activation-working-set-mib",
            "0.002",
            "--workspace-working-set-mib",
            "1",
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert result == 0
    assert output["schema_version"] == ACTIVATION_PLAN_SCHEMA_VERSION
    assert (tmp_path / "profile.json").is_file()
    assert (tmp_path / "plan.json").is_file()


def test_activation_plan_cli_parser_exposes_constraints() -> None:
    parser = build_parser()
    parsed = parser.parse_args(
        [
            "--config",
            "examples/micro-storage.yaml",
            "--profile-output",
            "profile.json",
            "--plan-output",
            "plan.json",
            "--fixed-interval",
            "3",
            "--max-replay-depth",
            "2",
        ]
    )

    assert parsed.fixed_interval == 3
    assert parsed.max_replay_depth == 2


def test_hybrid_config_does_not_fall_back_to_retain_all(tmp_path: Path) -> None:
    config = replace(
        _config(),
        training=replace(_config().training, activation_policy="hybrid"),
    )

    with pytest.raises(NotImplementedError):
        run_bounded_training(
            config,
            bundle_store_path=tmp_path / "hybrid",
            target_step=1,
            device_override="cpu",
        )
