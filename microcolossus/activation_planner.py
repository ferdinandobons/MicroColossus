"""Measured activation-anchor profile and planning primitives.

This module implements the side-effect-free M6C planning layer. It does not
execute hybrid backward training yet; it creates deterministic, checksummed
profiles and plans that a later runtime integration can bind into checkpoint
identity.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from itertools import combinations
from pathlib import Path
from typing import Any, cast

from .bounded_forward import build_execution_groups
from .config import ExperimentConfig
from .model import DecoderOnlyTransformer
from .storage.adapters import export_pytorch_model
from .storage.schema import canonical_json_bytes, sha256_hex
from .telemetry import write_json_atomic
from .training import seed_everything

ACTIVATION_PROFILE_SCHEMA_VERSION = "microcolossus.activation-profile.v1"
ACTIVATION_PLAN_SCHEMA_VERSION = "microcolossus.activation-plan.v1"
ACTIVATION_PLANNER_VERSION = "m6c-measured-budget-v1"
STATIC_PROFILE_SOURCE = "static-model-estimate-v1"


class ActivationProfileIntegrityError(RuntimeError):
    """Raised when a profile checksum or structure is invalid."""


class ActivationPlanIntegrityError(RuntimeError):
    """Raised when a plan checksum or structure is invalid."""


def _positive_int(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")


def _non_negative_int(value: int, name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} cannot be negative")


def _non_negative_float(value: float, name: str) -> None:
    if value < 0.0:
        raise ValueError(f"{name} cannot be negative")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return cast(Mapping[str, Any], value)


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be a sequence")
    return cast(Sequence[Any], value)


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    return tuple(str(item) for item in _sequence(value, name))


def _checksum_payload(value: Mapping[str, Any]) -> str:
    return sha256_hex(canonical_json_bytes(dict(value)))


@dataclass(frozen=True)
class ActivationGroupMeasurement:
    """One execution group's activation, parameter, and timing measurements."""

    ordinal: int
    name: str
    tensor_names: tuple[str, ...]
    can_anchor: bool
    parameter_bytes: int
    boundary_bytes: int
    local_workspace_bytes: int
    parameter_read_seconds: float
    materialization_seconds: float
    compute_seconds: float
    release_seconds: float
    source_result_checksum: str
    source_checksum: str = ""

    def __post_init__(self) -> None:
        _non_negative_int(self.ordinal, "ordinal")
        if not self.name:
            raise ValueError("name cannot be empty")
        if not self.tensor_names:
            raise ValueError(f"{self.name} must contain at least one tensor")
        for value, name in (
            (self.parameter_bytes, "parameter_bytes"),
            (self.boundary_bytes, "boundary_bytes"),
            (self.local_workspace_bytes, "local_workspace_bytes"),
        ):
            _non_negative_int(value, name)
        for value, name in (
            (self.parameter_read_seconds, "parameter_read_seconds"),
            (self.materialization_seconds, "materialization_seconds"),
            (self.compute_seconds, "compute_seconds"),
            (self.release_seconds, "release_seconds"),
        ):
            _non_negative_float(value, name)
        if not self.source_result_checksum:
            raise ValueError("source_result_checksum cannot be empty")
        if self.name == "final-head" and self.can_anchor:
            raise ValueError("final-head cannot be an activation anchor")

    def payload_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("source_checksum")
        return value

    def compute_checksum(self) -> str:
        return _checksum_payload(self.payload_dict())

    def with_checksum(self) -> ActivationGroupMeasurement:
        return replace(self, source_checksum=self.compute_checksum())

    def validate(self) -> None:
        if self.source_checksum != self.compute_checksum():
            raise ActivationProfileIntegrityError(
                f"group measurement checksum mismatch: {self.name}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ActivationGroupMeasurement:
        result = cls(
            ordinal=int(value["ordinal"]),
            name=str(value["name"]),
            tensor_names=_string_tuple(value["tensor_names"], "tensor_names"),
            can_anchor=bool(value["can_anchor"]),
            parameter_bytes=int(value["parameter_bytes"]),
            boundary_bytes=int(value["boundary_bytes"]),
            local_workspace_bytes=int(value["local_workspace_bytes"]),
            parameter_read_seconds=float(value["parameter_read_seconds"]),
            materialization_seconds=float(value["materialization_seconds"]),
            compute_seconds=float(value["compute_seconds"]),
            release_seconds=float(value["release_seconds"]),
            source_result_checksum=str(value["source_result_checksum"]),
            source_checksum=str(value["source_checksum"]),
        )
        result.validate()
        return result


@dataclass(frozen=True)
class ActivationMeasurementProfile:
    """Versioned deterministic activation-measurement profile."""

    schema_version: str
    profile_source: str
    backend: str
    device_identity: str
    dtype: str
    model_signature: dict[str, Any]
    model_signature_checksum: str
    sequence_length: int
    microbatch: int
    groups: tuple[ActivationGroupMeasurement, ...]
    profile_checksum: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != ACTIVATION_PROFILE_SCHEMA_VERSION:
            raise ValueError(f"unsupported activation profile schema: {self.schema_version}")
        if not self.profile_source:
            raise ValueError("profile_source cannot be empty")
        if not self.backend:
            raise ValueError("backend cannot be empty")
        if not self.device_identity:
            raise ValueError("device_identity cannot be empty")
        if not self.dtype:
            raise ValueError("dtype cannot be empty")
        _positive_int(self.sequence_length, "sequence_length")
        _positive_int(self.microbatch, "microbatch")
        if not self.groups:
            raise ValueError("groups cannot be empty")
        ordinals = tuple(item.ordinal for item in self.groups)
        if ordinals != tuple(range(len(self.groups))):
            raise ValueError("group ordinals must be contiguous from zero")
        names = tuple(item.name for item in self.groups)
        if len(set(names)) != len(names):
            raise ValueError("group names must be unique")
        if names[-1] != "final-head":
            raise ValueError("final-head must be the last execution group")
        for item in self.groups:
            item.validate()
        if self.model_signature_checksum != _checksum_payload(self.model_signature):
            raise ValueError("model signature checksum mismatch")

    def payload_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("profile_checksum")
        return value

    def compute_checksum(self) -> str:
        return _checksum_payload(self.payload_dict())

    def with_checksum(self) -> ActivationMeasurementProfile:
        return replace(self, profile_checksum=self.compute_checksum())

    def validate(self) -> None:
        if self.profile_checksum != self.compute_checksum():
            raise ActivationProfileIntegrityError("activation profile checksum mismatch")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ActivationMeasurementProfile:
        groups = tuple(
            ActivationGroupMeasurement.from_dict(_mapping(item, "group"))
            for item in _sequence(value["groups"], "groups")
        )
        signature = dict(_mapping(value["model_signature"], "model_signature"))
        result = cls(
            schema_version=str(value["schema_version"]),
            profile_source=str(value["profile_source"]),
            backend=str(value["backend"]),
            device_identity=str(value["device_identity"]),
            dtype=str(value["dtype"]),
            model_signature=signature,
            model_signature_checksum=str(value["model_signature_checksum"]),
            sequence_length=int(value["sequence_length"]),
            microbatch=int(value["microbatch"]),
            groups=groups,
            profile_checksum=str(value["profile_checksum"]),
        )
        result.validate()
        return result


@dataclass(frozen=True)
class ActivationReplaySegment:
    """Replay needed to reconstruct one backward target input."""

    target_group: str
    anchor_group: str | None
    replayed_group_names: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ActivationReplaySegment:
        anchor = value.get("anchor_group")
        return cls(
            target_group=str(value["target_group"]),
            anchor_group=None if anchor is None else str(anchor),
            replayed_group_names=_string_tuple(
                value["replayed_group_names"], "replayed_group_names"
            ),
        )


@dataclass(frozen=True)
class ActivationScheduleSummary:
    """Comparable summary for one activation-anchor schedule."""

    kind: str
    anchor_group_names: tuple[str, ...]
    retained_anchor_bytes: int
    maximum_activation_residency_bytes: int
    maximum_workspace_bytes: int
    maximum_replay_depth: int
    total_replayed_groups: int
    logical_parameter_reread_bytes: int
    estimated_replay_seconds: float
    activation_budget_bytes: int
    workspace_budget_bytes: int
    activation_budget_respected: bool
    workspace_budget_respected: bool
    replay_depth_respected: bool
    feasible: bool
    rejection_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ActivationScheduleSummary:
        reason = value.get("rejection_reason")
        return cls(
            kind=str(value["kind"]),
            anchor_group_names=_string_tuple(
                value["anchor_group_names"], "anchor_group_names"
            ),
            retained_anchor_bytes=int(value["retained_anchor_bytes"]),
            maximum_activation_residency_bytes=int(
                value["maximum_activation_residency_bytes"]
            ),
            maximum_workspace_bytes=int(value["maximum_workspace_bytes"]),
            maximum_replay_depth=int(value["maximum_replay_depth"]),
            total_replayed_groups=int(value["total_replayed_groups"]),
            logical_parameter_reread_bytes=int(value["logical_parameter_reread_bytes"]),
            estimated_replay_seconds=float(value["estimated_replay_seconds"]),
            activation_budget_bytes=int(value["activation_budget_bytes"]),
            workspace_budget_bytes=int(value["workspace_budget_bytes"]),
            activation_budget_respected=bool(value["activation_budget_respected"]),
            workspace_budget_respected=bool(value["workspace_budget_respected"]),
            replay_depth_respected=bool(value["replay_depth_respected"]),
            feasible=bool(value["feasible"]),
            rejection_reason=None if reason is None else str(reason),
        )


@dataclass(frozen=True)
class ActivationPlan:
    """Checksummed deterministic activation-anchor plan."""

    schema_version: str
    planner_version: str
    profile_checksum: str
    model_signature_checksum: str
    objective: str
    activation_budget_bytes: int
    workspace_budget_bytes: int
    max_replay_depth: int | None
    selected_policy: str
    selected_anchor_group_names: tuple[str, ...]
    replay_segments: tuple[ActivationReplaySegment, ...]
    maximum_retained_anchor_bytes: int
    maximum_activation_residency_bytes: int
    maximum_workspace_bytes: int
    maximum_replay_depth_observed: int
    total_replayed_groups: int
    logical_parameter_reread_bytes: int
    estimated_replay_seconds: float
    baseline_summaries: tuple[ActivationScheduleSummary, ...]
    feasible: bool
    rejection_reason: str | None
    plan_checksum: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != ACTIVATION_PLAN_SCHEMA_VERSION:
            raise ValueError(f"unsupported activation plan schema: {self.schema_version}")
        if self.planner_version != ACTIVATION_PLANNER_VERSION:
            raise ValueError(f"unsupported activation planner: {self.planner_version}")
        if not self.profile_checksum:
            raise ValueError("profile_checksum cannot be empty")
        if not self.model_signature_checksum:
            raise ValueError("model_signature_checksum cannot be empty")
        _positive_int(self.activation_budget_bytes, "activation_budget_bytes")
        _positive_int(self.workspace_budget_bytes, "workspace_budget_bytes")
        if self.max_replay_depth is not None:
            _non_negative_int(self.max_replay_depth, "max_replay_depth")
        for value, name in (
            (self.maximum_retained_anchor_bytes, "maximum_retained_anchor_bytes"),
            (
                self.maximum_activation_residency_bytes,
                "maximum_activation_residency_bytes",
            ),
            (self.maximum_workspace_bytes, "maximum_workspace_bytes"),
            (
                self.maximum_replay_depth_observed,
                "maximum_replay_depth_observed",
            ),
            (self.total_replayed_groups, "total_replayed_groups"),
            (self.logical_parameter_reread_bytes, "logical_parameter_reread_bytes"),
        ):
            _non_negative_int(value, name)
        _non_negative_float(self.estimated_replay_seconds, "estimated_replay_seconds")
        if self.feasible and self.rejection_reason is not None:
            raise ValueError("a feasible plan cannot have a rejection reason")
        if not self.feasible and not self.rejection_reason:
            raise ValueError("an infeasible plan must include a rejection reason")
        if "final-head" in self.selected_anchor_group_names:
            raise ValueError("final-head cannot be an activation anchor")

    def payload_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("plan_checksum")
        return value

    def compute_checksum(self) -> str:
        return _checksum_payload(self.payload_dict())

    def with_checksum(self) -> ActivationPlan:
        return replace(self, plan_checksum=self.compute_checksum())

    def validate(self, profile: ActivationMeasurementProfile | None = None) -> None:
        if self.plan_checksum != self.compute_checksum():
            raise ActivationPlanIntegrityError("activation plan checksum mismatch")
        if profile is not None:
            profile.validate()
            if self.profile_checksum != profile.profile_checksum:
                raise ActivationPlanIntegrityError("plan/profile checksum mismatch")
            if self.model_signature_checksum != profile.model_signature_checksum:
                raise ActivationPlanIntegrityError(
                    "plan/profile model signature mismatch"
                )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ActivationPlan:
        max_depth = value.get("max_replay_depth")
        reason = value.get("rejection_reason")
        result = cls(
            schema_version=str(value["schema_version"]),
            planner_version=str(value["planner_version"]),
            profile_checksum=str(value["profile_checksum"]),
            model_signature_checksum=str(value["model_signature_checksum"]),
            objective=str(value["objective"]),
            activation_budget_bytes=int(value["activation_budget_bytes"]),
            workspace_budget_bytes=int(value["workspace_budget_bytes"]),
            max_replay_depth=None if max_depth is None else int(max_depth),
            selected_policy=str(value["selected_policy"]),
            selected_anchor_group_names=_string_tuple(
                value["selected_anchor_group_names"],
                "selected_anchor_group_names",
            ),
            replay_segments=tuple(
                ActivationReplaySegment.from_dict(_mapping(item, "replay_segment"))
                for item in _sequence(value["replay_segments"], "replay_segments")
            ),
            maximum_retained_anchor_bytes=int(
                value["maximum_retained_anchor_bytes"]
            ),
            maximum_activation_residency_bytes=int(
                value["maximum_activation_residency_bytes"]
            ),
            maximum_workspace_bytes=int(value["maximum_workspace_bytes"]),
            maximum_replay_depth_observed=int(
                value["maximum_replay_depth_observed"]
            ),
            total_replayed_groups=int(value["total_replayed_groups"]),
            logical_parameter_reread_bytes=int(value["logical_parameter_reread_bytes"]),
            estimated_replay_seconds=float(value["estimated_replay_seconds"]),
            baseline_summaries=tuple(
                ActivationScheduleSummary.from_dict(_mapping(item, "baseline_summary"))
                for item in _sequence(value["baseline_summaries"], "baseline_summaries")
            ),
            feasible=bool(value["feasible"]),
            rejection_reason=None if reason is None else str(reason),
            plan_checksum=str(value["plan_checksum"]),
        )
        result.validate()
        return result


def _model_signature(
    config: ExperimentConfig,
    *,
    parameter_count: int,
    group_names: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "model": asdict(config.model),
        "training_shape": {
            "micro_batch_size": config.training.micro_batch_size,
            "sequence_length": config.training.sequence_length,
            "dtype": "float32",
        },
        "parameter_count": parameter_count,
        "execution_groups": list(group_names),
    }


def _group_boundary_bytes(config: ExperimentConfig, group_name: str) -> int:
    batch = config.training.micro_batch_size
    sequence = config.training.sequence_length
    if group_name == "final-head":
        elements = batch * sequence * config.model.vocab_size
    else:
        elements = batch * sequence * config.model.hidden_size
    return elements * 4


def _group_workspace_bytes(config: ExperimentConfig, group_name: str) -> int:
    token_input_bytes = config.training.micro_batch_size * config.training.sequence_length * 8
    hidden_bytes = (
        config.training.micro_batch_size
        * config.training.sequence_length
        * config.model.hidden_size
        * 4
    )
    logit_bytes = (
        config.training.micro_batch_size
        * config.training.sequence_length
        * config.model.vocab_size
        * 4
    )
    if group_name == "embedding":
        input_bytes = token_input_bytes
        output_bytes = hidden_bytes
    elif group_name == "final-head":
        input_bytes = hidden_bytes
        output_bytes = logit_bytes
    else:
        input_bytes = hidden_bytes
        output_bytes = hidden_bytes
    return input_bytes + output_bytes + max(input_bytes, output_bytes)


def build_activation_measurement_profile(
    config: ExperimentConfig,
    *,
    backend: str = "pytorch",
    device_identity: str = "cpu",
    dtype: str = "float32",
    profile_source: str = STATIC_PROFILE_SOURCE,
) -> ActivationMeasurementProfile:
    """Build a deterministic static M6C profile from model metadata.

    The first profile source is intentionally static. Later target-hardware
    measurement can replace the timing fields while preserving the same schema.
    """

    seed_everything(config.training.seed)
    model = DecoderOnlyTransformer(config.model)
    parameter_count = model.parameter_count
    payloads = export_pytorch_model(model)
    payload_by_name = {item.logical_name: item for item in payloads}
    groups = build_execution_groups(config, set(payload_by_name))
    group_names = tuple(item.name for item in groups)
    signature = _model_signature(
        config,
        parameter_count=parameter_count,
        group_names=group_names,
    )
    signature_checksum = _checksum_payload(signature)
    measurements: list[ActivationGroupMeasurement] = []
    for group in groups:
        parameter_bytes = sum(payload_by_name[name].byte_length for name in group.tensor_names)
        source_result_checksum = _checksum_payload(
            {
                "group_name": group.name,
                "tensor_checksums": [
                    [name, payload_by_name[name].checksum] for name in group.tensor_names
                ],
                "boundary_bytes": _group_boundary_bytes(config, group.name),
                "workspace_bytes": _group_workspace_bytes(config, group.name),
            }
        )
        measurements.append(
            ActivationGroupMeasurement(
                ordinal=group.ordinal,
                name=group.name,
                tensor_names=group.tensor_names,
                can_anchor=group.name != "final-head",
                parameter_bytes=parameter_bytes,
                boundary_bytes=_group_boundary_bytes(config, group.name),
                local_workspace_bytes=_group_workspace_bytes(config, group.name),
                parameter_read_seconds=0.0,
                materialization_seconds=0.0,
                compute_seconds=0.0,
                release_seconds=0.0,
                source_result_checksum=source_result_checksum,
            ).with_checksum()
        )
    profile = ActivationMeasurementProfile(
        schema_version=ACTIVATION_PROFILE_SCHEMA_VERSION,
        profile_source=profile_source,
        backend=backend,
        device_identity=device_identity,
        dtype=dtype,
        model_signature=signature,
        model_signature_checksum=signature_checksum,
        sequence_length=config.training.sequence_length,
        microbatch=config.training.micro_batch_size,
        groups=tuple(measurements),
    )
    return profile.with_checksum()


def _anchor_ordinals(
    profile: ActivationMeasurementProfile,
    anchor_group_names: Sequence[str],
) -> tuple[int, ...]:
    by_name = {item.name: item for item in profile.groups}
    ordinals: list[int] = []
    for name in anchor_group_names:
        item = by_name.get(name)
        if item is None:
            raise ValueError(f"unknown anchor group: {name}")
        if not item.can_anchor:
            raise ValueError(f"group cannot be an activation anchor: {name}")
        ordinals.append(item.ordinal)
    if len(set(ordinals)) != len(ordinals):
        raise ValueError("anchor groups must be unique")
    return tuple(sorted(ordinals))


def _summary_replay_segments(
    profile: ActivationMeasurementProfile,
    anchor_ordinals: tuple[int, ...],
) -> tuple[ActivationReplaySegment, ...]:
    segments: list[ActivationReplaySegment] = []
    anchor_set = set(anchor_ordinals)
    for target in reversed(profile.groups):
        if target.ordinal == 0:
            segments.append(
                ActivationReplaySegment(
                    target_group=target.name,
                    anchor_group=None,
                    replayed_group_names=(),
                )
            )
            continue
        prior_anchors = tuple(item for item in anchor_ordinals if item < target.ordinal)
        if prior_anchors:
            anchor = prior_anchors[-1]
            start = anchor + 1
            anchor_name = profile.groups[anchor].name
        else:
            start = 0
            anchor_name = None
        replayed = tuple(
            profile.groups[index].name
            for index in range(start, target.ordinal)
            if index not in anchor_set or index >= target.ordinal
        )
        segments.append(
            ActivationReplaySegment(
                target_group=target.name,
                anchor_group=anchor_name,
                replayed_group_names=replayed,
            )
        )
    return tuple(segments)


def _schedule_summary(
    kind: str,
    profile: ActivationMeasurementProfile,
    *,
    anchor_group_names: Sequence[str],
    activation_budget_bytes: int,
    workspace_budget_bytes: int,
    max_replay_depth: int | None,
) -> tuple[ActivationScheduleSummary, tuple[ActivationReplaySegment, ...]]:
    anchor_ordinals = _anchor_ordinals(profile, anchor_group_names)
    anchor_names = tuple(profile.groups[index].name for index in anchor_ordinals)
    segments = _summary_replay_segments(profile, anchor_ordinals)
    group_by_name = {item.name: item for item in profile.groups}
    retained_anchor_bytes = sum(
        profile.groups[index].boundary_bytes for index in anchor_ordinals
    )
    adjacent_gradient_bytes = max(
        (item.boundary_bytes for item in profile.groups if item.can_anchor),
        default=0,
    )
    maximum_activation_residency_bytes = retained_anchor_bytes + adjacent_gradient_bytes
    maximum_workspace_bytes = max(
        (
            group_by_name[name].local_workspace_bytes
            for segment in segments
            for name in (segment.target_group, *segment.replayed_group_names)
        ),
        default=0,
    )
    maximum_replay_depth_observed = max(
        (len(segment.replayed_group_names) for segment in segments),
        default=0,
    )
    total_replayed_groups = sum(len(segment.replayed_group_names) for segment in segments)
    logical_parameter_reread_bytes = sum(
        group_by_name[name].parameter_bytes
        for segment in segments
        for name in segment.replayed_group_names
    )
    estimated_replay_seconds = float(
        sum(
            group_by_name[name].parameter_read_seconds
            + group_by_name[name].materialization_seconds
            + group_by_name[name].compute_seconds
            + group_by_name[name].release_seconds
            for segment in segments
            for name in segment.replayed_group_names
        )
    )
    activation_ok = maximum_activation_residency_bytes <= activation_budget_bytes
    workspace_ok = maximum_workspace_bytes <= workspace_budget_bytes
    replay_depth_ok = (
        max_replay_depth is None or maximum_replay_depth_observed <= max_replay_depth
    )
    failures: list[str] = []
    if not activation_ok:
        failures.append("activation_budget_exceeded")
    if not workspace_ok:
        failures.append("workspace_budget_exceeded")
    if not replay_depth_ok:
        failures.append("replay_depth_exceeded")
    feasible = not failures
    summary = ActivationScheduleSummary(
        kind=kind,
        anchor_group_names=anchor_names,
        retained_anchor_bytes=retained_anchor_bytes,
        maximum_activation_residency_bytes=maximum_activation_residency_bytes,
        maximum_workspace_bytes=maximum_workspace_bytes,
        maximum_replay_depth=maximum_replay_depth_observed,
        total_replayed_groups=total_replayed_groups,
        logical_parameter_reread_bytes=logical_parameter_reread_bytes,
        estimated_replay_seconds=estimated_replay_seconds,
        activation_budget_bytes=activation_budget_bytes,
        workspace_budget_bytes=workspace_budget_bytes,
        activation_budget_respected=activation_ok,
        workspace_budget_respected=workspace_ok,
        replay_depth_respected=replay_depth_ok,
        feasible=feasible,
        rejection_reason=None if feasible else ", ".join(failures),
    )
    return summary, segments


def _fixed_interval_anchors(
    profile: ActivationMeasurementProfile,
    interval: int,
) -> tuple[str, ...]:
    _positive_int(interval, "fixed_interval")
    return tuple(
        item.name
        for item in profile.groups
        if item.can_anchor and item.ordinal % interval == 0
    )


def _all_anchor_sets(profile: ActivationMeasurementProfile) -> tuple[tuple[str, ...], ...]:
    possible = tuple(item.name for item in profile.groups if item.can_anchor)
    if len(possible) > 20:
        raise ValueError("exact anchor search is limited to 20 candidate anchors")
    results: list[tuple[str, ...]] = []
    for size in range(len(possible) + 1):
        results.extend(tuple(item) for item in combinations(possible, size))
    return tuple(results)


def _summary_sort_key(summary: ActivationScheduleSummary) -> tuple[object, ...]:
    return (
        0 if summary.feasible else 1,
        summary.estimated_replay_seconds,
        summary.total_replayed_groups,
        summary.logical_parameter_reread_bytes,
        summary.retained_anchor_bytes,
        len(summary.anchor_group_names),
        summary.anchor_group_names,
    )


def build_activation_plan(
    profile: ActivationMeasurementProfile,
    *,
    activation_budget_bytes: int,
    workspace_budget_bytes: int,
    max_replay_depth: int | None = None,
    fixed_interval: int = 2,
    objective: str = "minimize_estimated_replay_seconds",
) -> ActivationPlan:
    """Build a deterministic measured-budget activation-anchor plan."""

    profile.validate()
    _positive_int(activation_budget_bytes, "activation_budget_bytes")
    _positive_int(workspace_budget_bytes, "workspace_budget_bytes")
    if max_replay_depth is not None:
        _non_negative_int(max_replay_depth, "max_replay_depth")
    if objective != "minimize_estimated_replay_seconds":
        raise ValueError("only minimize_estimated_replay_seconds is implemented")

    retain_all, _ = _schedule_summary(
        "retain_all",
        profile,
        anchor_group_names=tuple(item.name for item in profile.groups if item.can_anchor),
        activation_budget_bytes=activation_budget_bytes,
        workspace_budget_bytes=workspace_budget_bytes,
        max_replay_depth=max_replay_depth,
    )
    recompute, _ = _schedule_summary(
        "recompute",
        profile,
        anchor_group_names=(),
        activation_budget_bytes=activation_budget_bytes,
        workspace_budget_bytes=workspace_budget_bytes,
        max_replay_depth=max_replay_depth,
    )
    fixed_interval_summary, _ = _schedule_summary(
        "fixed_interval",
        profile,
        anchor_group_names=_fixed_interval_anchors(profile, fixed_interval),
        activation_budget_bytes=activation_budget_bytes,
        workspace_budget_bytes=workspace_budget_bytes,
        max_replay_depth=max_replay_depth,
    )

    dynamic_candidates: list[
        tuple[ActivationScheduleSummary, tuple[ActivationReplaySegment, ...]]
    ] = []
    for anchors in _all_anchor_sets(profile):
        dynamic_candidates.append(
            _schedule_summary(
                "measured_budget_v1",
                profile,
                anchor_group_names=anchors,
                activation_budget_bytes=activation_budget_bytes,
                workspace_budget_bytes=workspace_budget_bytes,
                max_replay_depth=max_replay_depth,
            )
        )
    dynamic_summary, dynamic_segments = min(
        dynamic_candidates,
        key=lambda item: _summary_sort_key(item[0]),
    )
    feasible_summaries = tuple(item[0] for item in dynamic_candidates if item[0].feasible)
    if feasible_summaries:
        selected_summary = dynamic_summary
        selected_segments = dynamic_segments
    else:
        selected_summary = replace(
            dynamic_summary,
            feasible=False,
            rejection_reason="no feasible anchor schedule",
        )
        selected_segments = dynamic_segments

    plan = ActivationPlan(
        schema_version=ACTIVATION_PLAN_SCHEMA_VERSION,
        planner_version=ACTIVATION_PLANNER_VERSION,
        profile_checksum=profile.profile_checksum,
        model_signature_checksum=profile.model_signature_checksum,
        objective=objective,
        activation_budget_bytes=activation_budget_bytes,
        workspace_budget_bytes=workspace_budget_bytes,
        max_replay_depth=max_replay_depth,
        selected_policy=selected_summary.kind,
        selected_anchor_group_names=selected_summary.anchor_group_names,
        replay_segments=selected_segments,
        maximum_retained_anchor_bytes=selected_summary.retained_anchor_bytes,
        maximum_activation_residency_bytes=(
            selected_summary.maximum_activation_residency_bytes
        ),
        maximum_workspace_bytes=selected_summary.maximum_workspace_bytes,
        maximum_replay_depth_observed=selected_summary.maximum_replay_depth,
        total_replayed_groups=selected_summary.total_replayed_groups,
        logical_parameter_reread_bytes=selected_summary.logical_parameter_reread_bytes,
        estimated_replay_seconds=selected_summary.estimated_replay_seconds,
        baseline_summaries=(
            retain_all,
            recompute,
            fixed_interval_summary,
            selected_summary,
        ),
        feasible=selected_summary.feasible,
        rejection_reason=selected_summary.rejection_reason,
    )
    return plan.with_checksum()


def write_activation_profile(
    path: str | Path,
    profile: ActivationMeasurementProfile,
) -> None:
    profile.validate()
    write_json_atomic(path, profile.to_dict())


def load_activation_profile(path: str | Path) -> ActivationMeasurementProfile:
    import json

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return ActivationMeasurementProfile.from_dict(_mapping(value, "activation profile"))


def write_activation_plan(path: str | Path, plan: ActivationPlan) -> None:
    plan.validate()
    write_json_atomic(path, plan.to_dict())


def load_activation_plan(path: str | Path) -> ActivationPlan:
    import json

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return ActivationPlan.from_dict(_mapping(value, "activation plan"))
