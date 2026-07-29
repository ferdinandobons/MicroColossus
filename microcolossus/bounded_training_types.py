"""Result types for persistent bounded training."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .bounded_optimizer import OptimizerGroupMetrics
from .data import DataIdentity
from .evaluation import TrainingProgressRecord
from .step_bundle import (
    BundlePublicationTelemetry,
    BundleVerificationReport,
)
from .storage_training import StateComparison
from .training_checkpoint import BundleLineageEntry, TrainingMetadata


def _prefix_replay_traffic(
    result_path: str,
    activation_policy: str,
) -> tuple[int, int, int]:
    if activation_policy not in {"recompute", "hybrid"}:
        return 0, 0, 0
    value = json.loads(Path(result_path).read_text(encoding="utf-8"))
    return (
        int(value["total_prefix_parameter_tensor_reads"]),
        int(value["total_prefix_parameter_chunk_reads"]),
        int(value["total_prefix_parameter_logical_bytes_read"]),
    )


@dataclass(frozen=True)
class PersistentStepResult:
    step: int
    batch_cursor: int
    batch_seed: int
    batch_checksum: str
    source_bundle_id: str
    final_bundle_id: str
    source_parameter_store_path: str
    source_optimizer_store_path: str
    gradient_store_path: str
    candidate_parameter_store_path: str
    candidate_optimizer_store_path: str
    oracle_state_store_path: str | None
    bounded_backward_result_path: str
    activation_policy: str
    validation_level: str
    parameter_working_set_budget_bytes: int
    gradient_working_set_budget_bytes: int
    optimizer_working_set_budget_bytes: int
    activation_working_set_budget_bytes: int
    workspace_working_set_budget_bytes: int
    maximum_parameter_group_bytes: int
    maximum_gradient_group_bytes: int
    maximum_optimizer_group_bytes: int
    maximum_retained_activation_bytes: int
    maximum_workspace_bytes: int
    retained_forward_boundary_count: int
    retained_forward_boundary_bytes: int
    total_prefix_replayed_groups: int
    total_prefix_recomputation_seconds: float
    parameter_budget_respected: bool
    gradient_budget_respected: bool
    optimizer_budget_respected: bool
    activation_budget_respected: bool
    workspace_budget_respected: bool
    resident_loss: float
    bounded_loss: float
    loss_absolute_difference: float
    resident_gradient_norm: float
    bounded_gradient_norm: float
    gradient_norm_absolute_difference: float
    clipping_coefficient: float
    optimizer_group_order: tuple[str, ...]
    optimizer_groups: tuple[OptimizerGroupMetrics, ...]
    tied_parameter_update_count: int
    candidate_tensor_versions: tuple[tuple[str, int], ...]
    resident_vs_candidate_state: StateComparison | None
    candidate_vs_restored_state: StateComparison | None
    source_bundle_remained_authoritative_until_final_publish: bool
    final_bundle_is_authoritative: bool
    final_bundle_publication: BundlePublicationTelemetry
    final_bundle_verification: BundleVerificationReport
    total_parameter_logical_bytes_read: int
    total_gradient_logical_bytes_read: int
    total_optimizer_logical_bytes_read: int
    total_parameter_logical_bytes_written: int
    total_parameter_physical_bytes_written: int
    total_optimizer_logical_bytes_written: int
    total_optimizer_physical_bytes_written: int
    total_prefix_parameter_tensor_reads: int = field(init=False)
    total_prefix_parameter_chunk_reads: int = field(init=False)
    total_prefix_parameter_logical_bytes_read: int = field(init=False)
    full_candidate_state_materialized_for_validation: bool = True
    resident_oracle_materialized_for_validation: bool = True
    validation_omitted_checks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        tensor_reads, chunk_reads, logical_bytes = _prefix_replay_traffic(
            self.bounded_backward_result_path,
            self.activation_policy,
        )
        object.__setattr__(self, "total_prefix_parameter_tensor_reads", tensor_reads)
        object.__setattr__(self, "total_prefix_parameter_chunk_reads", chunk_reads)
        object.__setattr__(self, "total_prefix_parameter_logical_bytes_read", logical_bytes)


@dataclass(frozen=True)
class BoundedTrainingResult:
    schema_version: str
    experiment: str
    device: str
    bundle_store_path: str
    training_metadata: TrainingMetadata
    data_identity: DataIdentity
    requested_target_step: int
    started_step: int
    final_step: int
    resumed: bool
    initialized_bundle_id: str
    initialization_publication: BundlePublicationTelemetry | None
    activation_policy: str
    validation_level: str
    activation_working_set_budget_bytes: int
    workspace_working_set_budget_bytes: int
    steps: tuple[PersistentStepResult, ...]
    lineage: tuple[BundleLineageEntry, ...]
    progress_records: tuple[TrainingProgressRecord, ...]
    metrics_directory: str
    final_bundle_verification: BundleVerificationReport
    final_bounded_vs_resident_state: StateComparison | None
    final_bundle_vs_restored_state: StateComparison | None
    final_batch_cursor: int
    optimizer_step_values: tuple[tuple[str, float], ...]
    maximum_retained_activation_bytes: int
    maximum_workspace_bytes: int
    maximum_retained_forward_boundary_bytes: int
    total_prefix_replayed_groups: int
    total_prefix_recomputation_seconds: float
    total_parameter_logical_bytes_read: int
    total_gradient_logical_bytes_read: int
    total_optimizer_logical_bytes_read: int
    total_parameter_logical_bytes_written: int
    total_parameter_physical_bytes_written: int
    total_optimizer_logical_bytes_written: int
    total_optimizer_physical_bytes_written: int
    total_prefix_parameter_tensor_reads: int = field(init=False)
    total_prefix_parameter_chunk_reads: int = field(init=False)
    total_prefix_parameter_logical_bytes_read: int = field(init=False)
    batch_cursor_derived_from_committed_step: bool = True
    full_final_state_materialized_for_validation: bool = True
    resident_reference_replayed_from_step_zero: bool = True
    historical_bundles_retained: bool = True
    validation_omitted_checks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "total_prefix_parameter_tensor_reads",
            sum(item.total_prefix_parameter_tensor_reads for item in self.steps),
        )
        object.__setattr__(
            self,
            "total_prefix_parameter_chunk_reads",
            sum(item.total_prefix_parameter_chunk_reads for item in self.steps),
        )
        object.__setattr__(
            self,
            "total_prefix_parameter_logical_bytes_read",
            sum(item.total_prefix_parameter_logical_bytes_read for item in self.steps),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
