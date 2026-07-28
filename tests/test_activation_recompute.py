from __future__ import annotations

from pathlib import Path

import pytest

from microcolossus.activation_recompute import (
    ACTIVATION_RECOMPUTE_SCHEMA_VERSION,
    ActivationWorkingSetExceededError,
    WorkspaceWorkingSetExceededError,
    run_activation_recompute_validation,
)
from microcolossus.activation_recompute_cli import build_parser
from microcolossus.config import load_experiment_config


def _config():
    return load_experiment_config(Path("examples/micro-storage.yaml"))


def test_activation_recompute_matches_resident_gradients(tmp_path: Path) -> None:
    result = run_activation_recompute_validation(
        _config(),
        parameter_store_path=tmp_path / "parameters",
        oracle_gradient_store_path=tmp_path / "oracle-gradients",
        gradient_store_path=tmp_path / "recomputed-gradients",
        output_path=tmp_path / "result.json",
        device_override="cpu",
        parameter_working_set_bytes=1024**2,
        gradient_working_set_bytes=1024**2,
        activation_working_set_bytes=1024**2,
        workspace_working_set_bytes=4 * 1024**2,
    )

    assert result.schema_version == ACTIVATION_RECOMPUTE_SCHEMA_VERSION
    assert result.activation_policy == "recompute"
    assert result.parameter_count == 11_456
    assert result.retained_forward_boundary_count == 0
    assert result.retained_forward_boundary_bytes == 0
    assert result.total_prefix_replayed_groups == 3
    assert result.backward_group_order == ("final-head", "block-0", "embedding")
    assert result.maximum_retained_activation_bytes > 0
    assert result.activation_budget_respected
    assert result.workspace_budget_respected
    assert result.loss_absolute_difference == 0.0
    assert result.resident_vs_recomputed_gradients.exact_bytes
    assert result.resident_vs_recomputed_gradients.maximum_absolute_difference == 0.0
    assert result.tied_gradient_accumulation_count == 2
    assert result.tied_gradient_version == 1
    assert result.parameter_manifest_unchanged
    assert result.gradient_tensor_count == 12
    assert (tmp_path / "result.json").is_file()


def test_activation_recompute_rejects_too_small_activation_budget(
    tmp_path: Path,
) -> None:
    with pytest.raises(ActivationWorkingSetExceededError):
        run_activation_recompute_validation(
            _config(),
            parameter_store_path=tmp_path / "parameters",
            oracle_gradient_store_path=tmp_path / "oracle-gradients",
            gradient_store_path=tmp_path / "recomputed-gradients",
            device_override="cpu",
            parameter_working_set_bytes=1024**2,
            gradient_working_set_bytes=1024**2,
            activation_working_set_bytes=1,
            workspace_working_set_bytes=4 * 1024**2,
        )


def test_activation_recompute_rejects_too_small_workspace_budget(
    tmp_path: Path,
) -> None:
    with pytest.raises(WorkspaceWorkingSetExceededError):
        run_activation_recompute_validation(
            _config(),
            parameter_store_path=tmp_path / "parameters",
            oracle_gradient_store_path=tmp_path / "oracle-gradients",
            gradient_store_path=tmp_path / "recomputed-gradients",
            device_override="cpu",
            parameter_working_set_bytes=1024**2,
            gradient_working_set_bytes=1024**2,
            activation_working_set_bytes=1024**2,
            workspace_working_set_bytes=1,
        )


def test_activation_recompute_cli_parser_exposes_all_budgets() -> None:
    parser = build_parser()
    parsed = parser.parse_args(
        [
            "--config",
            "examples/micro-storage.yaml",
            "--parameter-store",
            "parameters",
            "--oracle-gradient-store",
            "oracle",
            "--gradient-store",
            "gradients",
            "--output",
            "result.json",
            "--activation-working-set-mib",
            "2",
            "--workspace-working-set-mib",
            "8",
        ]
    )
    assert parsed.activation_working_set_mib == 2.0
    assert parsed.workspace_working_set_mib == 8.0
