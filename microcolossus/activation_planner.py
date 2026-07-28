"""Deterministic activation measurement profiles and hybrid anchor plans."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from .bounded_forward import ExecutionGroupSpec, build_execution_groups
from .config import ExperimentConfig
from .model import DecoderOnlyTransformer
from .storage.schema import canonical_json_bytes, sha256_hex

ACTIVATION_PROFILE_SCHEMA_VERSION = "microcolossus.activation-profile.v1"
ACTIVATION_PLAN_SCHEMA_VERSION = "microcolossus.activation-plan.v1"
ACTIVATION_PLANNER_VERSION = "0.13.0"


class ActivationPlanError(RuntimeError):
    """Base class for activation-profile and activation-plan failures."""


class ActivationPlanInfeasibleError(ActivationPlanError):
    """Raised when no schedule can satisfy the declared logical budgets."""


@dataclass(frozen=True)
class ActivationGroupProfile:
    """Measured or estimated cost of one ordered execution group."""

    ordinal: int
    name: str
    parameter_bytes: int
    boundary_bytes: int
    workspace_bytes: int
    observed_read_seconds: float
    observed_compute_seconds: float

    @property
    def replay_seconds(self) -> float:
        return self.observed_read_seconds + self.observed_compute_seconds

    def validate(self) -> None:
        if self.ordinal < 0:
            raise ActivationPlanError("activation group ordinal cannot be negative")
        if not self.name:
            raise ActivationPlanError("activation group name cannot be empty")
        for value, name in (
            (self.parameter_bytes, "parameter_bytes"),
            (self.boundary_bytes, "boundary_bytes"),
            (self.workspace_bytes, "workspace_bytes"),
        ):
            if value < 0:
                raise ActivationPlanError(f"{name} cannot be negative")
        for value, name in (
            (self.observed_read_seconds, "observed_read_seconds"),
            (self.observed_compute_seconds, "observed_compute_seconds"),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ActivationPlanError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class ActivationMeasurementProfile:
    """Checksummed cost profile consumed by the hybrid planner."""

    schema_version: str
    planner_version: str
    source: str
    device_identity: str
    model_signature: str
    sequence_length: int
    micro_batch_size: int
    dtype: str
    groups: tuple[ActivationGroupProfile, ...]
    source_result_checksums: tuple[str, ...] = ()
    profile_checksum: str = ""

    def payload_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "planner_version": self.planner_version,
            "source": self.source,
            "device_identity": self.device_identity,
            "model_signature": self.model_signature,
            "sequence_length": self.sequence_length,
            "micro_batch_size": self.micro_batch_size,
            "dtype": self.dtype,
            "groups": [asdict(item) for item in self.groups],
            "source_result_checksums": list(self.source_result_checksums),
        }

    def compute_checksum(self) -> str:
        return sha256_hex(canonical_json_bytes(self.payload_dict()))

    def with_checksum(self) -> ActivationMeasurementProfile:
        return replace(self, profile_checksum=self.compute_checksum())

    def validate(self) -> None:
        if self.schema_version != ACTIVATION_PROFILE_SCHEMA_VERSION:
            raise ActivationPlanError(
                f"unsupported activation profile schema: {self.schema_version}"
            )
        if self.planner_version != ACTIVATION_PLANNER_VERSION:
            raise ActivationPlanError(
                f"unsupported activation planner version: {self.planner_version}"
            )
        if self.sequence_length <= 0 or self.micro_batch_size <= 0:
            raise ActivationPlanError("profile dimensions must be positive")
        if not self.groups:
            raise ActivationPlanError("activation profile must contain execution groups")
        for expected, group in enumerate(self.groups):
            group.validate()
            if group.ordinal != expected:
                raise ActivationPlanError("activation profile ordinals must be contiguous")
        if self.groups[-1].name != "final-head":
            raise ActivationPlanError("activation profile must terminate with final-head")
        if self.groups[-1].boundary_bytes != 0:
            raise ActivationPlanError("final-head cannot be retained as an activation anchor")
        if self.profile_checksum != self.compute_checksum():
            raise ActivationPlanError("activation profile checksum mismatch")

    def to_dict(self) -> dict[str, Any]:
        value = self.payload_dict()
        value["profile_checksum"] = self.profile_checksum
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ActivationMeasurementProfile:
        profile = cls(
            schema_version=str(value["schema_version"]),
            planner_version=str(value["planner_version"]),
            source=str(value["source"]),
            device_identity=str(value["device_identity"]),
            model_signature=str(value["model_signature"]),
            sequence_length=int(value["sequence_length"]),
            micro_batch_size=int(value["micro_batch_size"]),
            dtype=str(value["dtype"]),
            groups=tuple(
                ActivationGroupProfile(
                    ordinal=int(item["ordinal"]),
                    name=str(item["name"]),
                    parameter_bytes=int(item["parameter_bytes"]),
                    boundary_bytes=int(item["boundary_bytes"]),
                    workspace_bytes=int(item["workspace_bytes"]),
                    observed_read_seconds=float(item["observed_read_seconds"]),
                    observed_compute_seconds=float(item["observed_compute_seconds"]),
                )
                for item in value["groups"]
            ),
            source_result_checksums=tuple(
                str(item) for item in value.get("source_result_checksums", [])
            ),
            profile_checksum=str(value["profile_checksum"]),
        )
        profile.validate()
        return profile


@dataclass(frozen=True)
class ActivationReplaySegment:
    """Replay required to reconstruct one backward-group input."""

    target_group: str
    source_anchor_group: str | None
    replayed_group_names: tuple[str, ...]
    replayed_parameter_bytes: int
    estimated_replay_seconds: float


@dataclass(frozen=True)
class ActivationScheduleSummary:
    """Comparable cost summary for one anchor schedule."""

    kind: str
    anchor_group_names: tuple[str, ...]
    retained_anchor_bytes: int
    maximum_replayed_groups: int
    total_replayed_groups: int
    total_parameter_logical_bytes_reread: int
    estimated_replay_seconds: float
    activation_budget_respected: bool
    workspace_budget_respected: bool


@dataclass(frozen=True)
class ActivationPlan:
    """Checksummed deterministic anchor and replay schedule."""

    schema_version: str
    planner_version: str
    policy_kind: str
    profile_checksum: str
    model_signature: str
    activation_working_set_budget_bytes: int
    workspace_working_set_budget_bytes: int
    max_replay_groups_limit: int | None
    fixed_interval: int
    anchor_group_names: tuple[str, ...]
    segments: tuple[ActivationReplaySegment, ...]
    maximum_retained_anchor_bytes: int
    maximum_workspace_bytes: int
    maximum_replayed_groups: int
    total_replayed_groups: int
    total_parameter_logical_bytes_reread: int
    estimated_replay_seconds: float
    retain_all_baseline: ActivationScheduleSummary
    recompute_baseline: ActivationScheduleSummary
    fixed_interval_baseline: ActivationScheduleSummary
    feasible: bool
    rejection_reason: str | None
    plan_checksum: str = ""

    def payload_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "planner_version": self.planner_version,
            "policy_kind": self.policy_kind,
            "profile_checksum": self.profile_checksum,
            "model_signature": self.model_signature,
            "activation_working_set_budget_bytes": self.activation_working_set_budget_bytes,
            "workspace_working_set_budget_bytes": self.workspace_working_set_budget_bytes,
            "max_replay_groups_limit": self.max_replay_groups_limit,
            "fixed_interval": self.fixed_interval,
            "anchor_group_names": list(self.anchor_group_names),
            "segments": [asdict(item) for item in self.segments],
            "maximum_retained_anchor_bytes": self.maximum_retained_anchor_bytes,
            "maximum_workspace_bytes": self.maximum_workspace_bytes,
            "maximum_replayed_groups": self.maximum_replayed_groups,
            "total_replayed_groups": self.total_replayed_groups,
            "total_parameter_logical_bytes_reread": (
                self.total_parameter_logical_bytes_reread
            ),
            "estimated_replay_seconds": self.estimated_replay_seconds,
            "retain_all_baseline": asdict(self.retain_all_baseline),
            "recompute_baseline": asdict(self.recompute_baseline),
            "fixed_interval_baseline": asdict(self.fixed_interval_baseline),
            "feasible": self.feasible,
            "rejection_reason": self.rejection_reason,
        }

    def compute_checksum(self) -> str:
        return sha256_hex(canonical_json_bytes(self.payload_dict()))

    def with_checksum(self) -> ActivationPlan:
        return replace(self, plan_checksum=self.compute_checksum())

    def validate(self) -> None:
        if self.schema_version != ACTIVATION_PLAN_SCHEMA_VERSION:
            raise ActivationPlanError(
                f"unsupported activation plan schema: {self.schema_version}"
            )
        if self.planner_version != ACTIVATION_PLANNER_VERSION:
            raise ActivationPlanError(
                f"unsupported activation planner version: {self.planner_version}"
            )
        if self.activation_working_set_budget_bytes <= 0:
            raise ActivationPlanError("activation budget must be positive")
        if self.workspace_working_set_budget_bytes <= 0:
            raise ActivationPlanError("workspace budget must be positive")
        if self.plan_checksum != self.compute_checksum():
            raise ActivationPlanError("activation plan checksum mismatch")
        if not self.feasible:
            raise ActivationPlanInfeasibleError(
                self.rejection_reason or "activation plan is infeasible"
            )
        if self.maximum_retained_anchor_bytes > self.activation_working_set_budget_bytes:
            raise ActivationPlanError("activation plan exceeds its retained-anchor budget")
        if self.maximum_workspace_bytes > self.workspace_working_set_budget_bytes:
            raise ActivationPlanError("activation plan exceeds its workspace budget")

    def to_dict(self) -> dict[str, Any]:
        value = self.payload_dict()
        value["plan_checksum"] = self.plan_checksum
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ActivationPlan:
        def summary(name: str) -> ActivationScheduleSummary:
            item = value[name]
            return ActivationScheduleSummary(
                kind=str(item["kind"]),
                anchor_group_names=tuple(str(x) for x in item["anchor_group_names"]),
                retained_anchor_bytes=int(item["retained_anchor_bytes"]),
                maximum_replayed_groups=int(item["maximum_replayed_groups"]),
                total_replayed_groups=int(item["total_replayed_groups"]),
                total_parameter_logical_bytes_reread=int(
                    item["total_parameter_logical_bytes_reread"]
                ),
                estimated_replay_seconds=float(item["estimated_replay_seconds"]),
                activation_budget_respected=bool(item["activation_budget_respected"]),
                workspace_budget_respected=bool(item["workspace_budget_respected"]),
            )

        plan = cls(
            schema_version=str(value["schema_version"]),
            planner_version=str(value["planner_version"]),
            policy_kind=str(value["policy_kind"]),
            profile_checksum=str(value["profile_checksum"]),
            model_signature=str(value["model_signature"]),
            activation_working_set_budget_bytes=int(
                value["activation_working_set_budget_bytes"]
            ),
            workspace_working_set_budget_bytes=int(
                value["workspace_working_set_budget_bytes"]
            ),
            max_replay_groups_limit=(
                None
                if value.get("max_replay_groups_limit") is None
                else int(value["max_replay_groups_limit"])
            ),
            fixed_interval=int(value["fixed_interval"]),
            anchor_group_names=tuple(str(x) for x in value["anchor_group_names"]),
            segments=tuple(
                ActivationReplaySegment(
                    target_group=str(item["target_group"]),
                    source_anchor_group=(
                        None
                        if item.get("source_anchor_group") is None
                        else str(item["source_anchor_group"])
                    ),
                    replayed_group_names=tuple(
                        str(x) for x in item["replayed_group_names"]
                    ),
                    replayed_parameter_bytes=int(item["replayed_parameter_bytes"]),
                    estimated_replay_seconds=float(item["estimated_replay_seconds"]),
                )
                for item in value["segments"]
            ),
            maximum_retained_anchor_bytes=int(value["maximum_retained_anchor_bytes"]),
            maximum_workspace_bytes=int(value["maximum_workspace_bytes"]),
            maximum_replayed_groups=int(value["maximum_replayed_groups"]),
            total_replayed_groups=int(value["total_replayed_groups"]),
            total_parameter_logical_bytes_reread=int(
                value["total_parameter_logical_bytes_reread"]
            ),
            estimated_replay_seconds=float(value["estimated_replay_seconds"]),
            retain_all_baseline=summary("retain_all_baseline"),
            recompute_baseline=summary("recompute_baseline"),
            fixed_interval_baseline=summary("fixed_interval_baseline"),
            feasible=bool(value["feasible"]),
            rejection_reason=(
                None if value.get("rejection_reason") is None else str(value["rejection_reason"])
            ),
            plan_checksum=str(value["plan_checksum"]),
        )
        plan.validate()
        return plan


def activation_model_signature(config: ExperimentConfig) -> str:
    """Return the planner identity of model and batch-shape semantics."""

    payload = {
        "model": asdict(config.model),
        "sequence_length": config.training.sequence_length,
        "micro_batch_size": config.training.micro_batch_size,
        "dtype": "float32",
    }
    return sha256_hex(canonical_json_bytes(payload))


def _parameter_bytes_by_name(config: ExperimentConfig) -> tuple[dict[str, int], set[str]]:
    model = DecoderOnlyTransformer(config.model)
    values = {
        f"model.{name}": int(parameter.numel() * parameter.element_size())
        for name, parameter in model.named_parameters()
    }
    return values, set(values)


def _group_specs_and_parameter_bytes(
    config: ExperimentConfig,
) -> tuple[tuple[ExecutionGroupSpec, ...], tuple[int, ...]]:
    values, names = _parameter_bytes_by_name(config)
    groups = build_execution_groups(config, names)
    return groups, tuple(sum(values[name] for name in group.tensor_names) for group in groups)


def build_logical_activation_profile(
    config: ExperimentConfig,
    *,
    device_identity: str = "logical-estimate-v1",
) -> ActivationMeasurementProfile:
    """Build a deterministic fallback profile from tensor and activation sizes."""

    groups, parameter_bytes = _group_specs_and_parameter_bytes(config)
    batch = config.training.micro_batch_size
    sequence = config.training.sequence_length
    hidden = config.model.hidden_size
    vocabulary = config.model.vocab_size
    hidden_bytes = batch * sequence * hidden * 4
    token_bytes = batch * sequence * 8
    logits_bytes = batch * sequence * vocabulary * 4
    profiles: list[ActivationGroupProfile] = []
    for group, logical_parameters in zip(groups, parameter_bytes, strict=True):
        if group.name == "embedding":
            boundary = hidden_bytes
            workspace = token_bytes + hidden_bytes
        elif group.name == "final-head":
            boundary = 0
            workspace = 2 * hidden_bytes + logits_bytes
        else:
            boundary = hidden_bytes
            workspace = 4 * hidden_bytes
        profiles.append(
            ActivationGroupProfile(
                ordinal=group.ordinal,
                name=group.name,
                parameter_bytes=logical_parameters,
                boundary_bytes=boundary,
                workspace_bytes=workspace,
                observed_read_seconds=logical_parameters / 1_000_000_000.0,
                observed_compute_seconds=max(workspace, 1) / 1_000_000_000.0,
            )
        )
    return ActivationMeasurementProfile(
        schema_version=ACTIVATION_PROFILE_SCHEMA_VERSION,
        planner_version=ACTIVATION_PLANNER_VERSION,
        source="logical-estimate-v1",
        device_identity=device_identity,
        model_signature=activation_model_signature(config),
        sequence_length=sequence,
        micro_batch_size=batch,
        dtype="float32",
        groups=tuple(profiles),
    ).with_checksum()


def _file_checksum(path: Path) -> str:
    return sha256_hex(path.read_bytes())


def build_measured_activation_profile(
    config: ExperimentConfig,
    *,
    retain_result_path: str | Path,
    recompute_result_path: str | Path,
    device_identity: str,
) -> ActivationMeasurementProfile:
    """Calibrate a profile from completed retain-all and recompute result JSON."""

    retain_path = Path(retain_result_path)
    recompute_path = Path(recompute_result_path)
    retain = json.loads(retain_path.read_text(encoding="utf-8"))
    recompute = json.loads(recompute_path.read_text(encoding="utf-8"))
    if retain.get("activation_policy") != "retain_all":
        raise ActivationPlanError("retain calibration result is not retain_all")
    if recompute.get("activation_policy") != "recompute":
        raise ActivationPlanError("recompute calibration result is not recompute")
    detailed_paths = [
        Path(str(item["bounded_backward_result_path"])) for item in recompute["steps"]
    ]
    if not detailed_paths:
        raise ActivationPlanError("recompute calibration contains no optimizer steps")
    detailed = [json.loads(path.read_text(encoding="utf-8")) for path in detailed_paths]
    first_forward = detailed[0]["forward_groups"]
    names = [str(item["name"]) for item in first_forward]
    profiles: list[ActivationGroupProfile] = []
    for ordinal, name in enumerate(names):
        forward_items = [item["forward_groups"][ordinal] for item in detailed]
        backward_items = [
            next(group for group in item["backward_groups"] if group["name"] == name)
            for item in detailed
        ]
        parameter_bytes = int(forward_items[0]["logical_parameter_bytes"])
        boundary_bytes = (
            0 if name == "final-head" else int(forward_items[0]["output_activation_bytes"])
        )
        workspace_bytes = max(
            max(int(item["logical_workspace_bytes"]) for item in forward_items),
            max(int(item["logical_workspace_bytes"]) for item in backward_items),
        )
        read_seconds = math.fsum(
            float(item["parameter_read_seconds"]) + float(item["materialization_seconds"])
            for item in forward_items
        ) / len(forward_items)
        compute_seconds = math.fsum(
            float(item["compute_seconds"]) for item in forward_items
        ) / len(forward_items)
        profiles.append(
            ActivationGroupProfile(
                ordinal=ordinal,
                name=name,
                parameter_bytes=parameter_bytes,
                boundary_bytes=boundary_bytes,
                workspace_bytes=workspace_bytes,
                observed_read_seconds=read_seconds,
                observed_compute_seconds=compute_seconds,
            )
        )
    profile = ActivationMeasurementProfile(
        schema_version=ACTIVATION_PROFILE_SCHEMA_VERSION,
        planner_version=ACTIVATION_PLANNER_VERSION,
        source="measured-retain-recompute-v1",
        device_identity=device_identity,
        model_signature=activation_model_signature(config),
        sequence_length=config.training.sequence_length,
        micro_batch_size=config.training.micro_batch_size,
        dtype="float32",
        groups=tuple(profiles),
        source_result_checksums=(
            _file_checksum(retain_path),
            _file_checksum(recompute_path),
            *(_file_checksum(path) for path in detailed_paths),
        ),
    ).with_checksum()
    profile.validate()
    return profile


def load_activation_profile(path: str | Path) -> ActivationMeasurementProfile:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ActivationPlanError("activation profile must be a JSON object")
    return ActivationMeasurementProfile.from_dict(value)


def write_activation_profile(path: str | Path, profile: ActivationMeasurementProfile) -> None:
    profile.validate()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(canonical_json_bytes(profile.to_dict()) + b"\n")


def _segments_for(
    profile: ActivationMeasurementProfile,
    anchors: frozenset[int],
) -> tuple[ActivationReplaySegment, ...]:
    groups = profile.groups
    segments: list[ActivationReplaySegment] = []
    for target in groups[1:]:
        boundary_ordinal = target.ordinal - 1
        source = max((item for item in anchors if item <= boundary_ordinal), default=None)
        if source == boundary_ordinal:
            replayed: tuple[ActivationGroupProfile, ...] = ()
        else:
            start = 0 if source is None else source + 1
            replayed = groups[start : target.ordinal]
        segments.append(
            ActivationReplaySegment(
                target_group=target.name,
                source_anchor_group=None if source is None else groups[source].name,
                replayed_group_names=tuple(item.name for item in replayed),
                replayed_parameter_bytes=sum(item.parameter_bytes for item in replayed),
                estimated_replay_seconds=math.fsum(item.replay_seconds for item in replayed),
            )
        )
    return tuple(segments)


def _summary_for(
    profile: ActivationMeasurementProfile,
    anchors: frozenset[int],
    *,
    kind: str,
    activation_budget_bytes: int,
    workspace_budget_bytes: int,
) -> ActivationScheduleSummary:
    segments = _segments_for(profile, anchors)
    retained = sum(profile.groups[item].boundary_bytes for item in anchors)
    maximum_boundary = max(item.boundary_bytes for item in profile.groups)
    activation_required = retained + (2 * maximum_boundary)
    maximum_workspace = max(item.workspace_bytes for item in profile.groups)
    return ActivationScheduleSummary(
        kind=kind,
        anchor_group_names=tuple(profile.groups[item].name for item in sorted(anchors)),
        retained_anchor_bytes=retained,
        maximum_replayed_groups=max(
            (len(item.replayed_group_names) for item in segments), default=0
        ),
        total_replayed_groups=sum(len(item.replayed_group_names) for item in segments),
        total_parameter_logical_bytes_reread=sum(
            item.replayed_parameter_bytes for item in segments
        ),
        estimated_replay_seconds=math.fsum(
            item.estimated_replay_seconds for item in segments
        ),
        activation_budget_respected=activation_required <= activation_budget_bytes,
        workspace_budget_respected=maximum_workspace <= workspace_budget_bytes,
    )


def _dynamic_anchors(
    profile: ActivationMeasurementProfile,
    *,
    activation_budget_bytes: int,
    workspace_budget_bytes: int,
    max_replay_groups: int | None,
) -> frozenset[int]:
    maximum_boundary = max(item.boundary_bytes for item in profile.groups)
    capacity = activation_budget_bytes - (2 * maximum_boundary)
    if capacity < 0:
        raise ActivationPlanInfeasibleError(
            "activation budget cannot hold one local input and adjacent gradient"
        )
    if max(item.workspace_bytes for item in profile.groups) > workspace_budget_bytes:
        raise ActivationPlanInfeasibleError(
            "workspace budget is smaller than the largest measured local workspace"
        )
    candidates = tuple(
        item.ordinal for item in profile.groups[:-1] if item.boundary_bytes > 0
    )
    anchors: set[int] = set()

    def retained_bytes(values: set[int]) -> int:
        return sum(profile.groups[item].boundary_bytes for item in values)

    def summary(values: set[int]) -> ActivationScheduleSummary:
        return _summary_for(
            profile,
            frozenset(values),
            kind="candidate",
            activation_budget_bytes=activation_budget_bytes,
            workspace_budget_bytes=workspace_budget_bytes,
        )

    if max_replay_groups is not None:
        if max_replay_groups < 0:
            raise ActivationPlanError("max_replay_groups cannot be negative")
        while summary(anchors).maximum_replayed_groups > max_replay_groups:
            choices: list[tuple[int, float, int]] = []
            for candidate in candidates:
                if candidate in anchors:
                    continue
                proposed = anchors | {candidate}
                if retained_bytes(proposed) > capacity:
                    continue
                value = summary(proposed)
                choices.append(
                    (
                        value.maximum_replayed_groups,
                        value.estimated_replay_seconds,
                        candidate,
                    )
                )
            if not choices:
                raise ActivationPlanInfeasibleError(
                    "activation budget cannot satisfy max_replay_groups"
                )
            _, _, selected = min(choices)
            anchors.add(selected)

    while True:
        current = summary(anchors)
        choices: list[tuple[float, float, int]] = []
        for candidate in candidates:
            if candidate in anchors:
                continue
            proposed = anchors | {candidate}
            if retained_bytes(proposed) > capacity:
                continue
            value = summary(proposed)
            seconds_saved = current.estimated_replay_seconds - value.estimated_replay_seconds
            bytes_saved = (
                current.total_parameter_logical_bytes_reread
                - value.total_parameter_logical_bytes_reread
            )
            anchor_bytes = max(profile.groups[candidate].boundary_bytes, 1)
            density = seconds_saved / anchor_bytes
            choices.append((-density, -float(bytes_saved), candidate))
        if not choices:
            break
        _, _, selected = min(choices)
        proposed = anchors | {selected}
        if summary(proposed).estimated_replay_seconds >= current.estimated_replay_seconds:
            break
        anchors.add(selected)
    return frozenset(anchors)


def build_activation_plan(
    profile: ActivationMeasurementProfile,
    *,
    policy_kind: str = "measured_budget_v1",
    activation_working_set_budget_bytes: int,
    workspace_working_set_budget_bytes: int,
    max_replay_groups: int | None = None,
    fixed_interval: int = 2,
) -> ActivationPlan:
    """Build deterministic baselines and the selected hybrid schedule."""

    profile.validate()
    if activation_working_set_budget_bytes <= 0 or workspace_working_set_budget_bytes <= 0:
        raise ActivationPlanError("activation and workspace budgets must be positive")
    if fixed_interval <= 0:
        raise ActivationPlanError("fixed_interval must be positive")
    candidates = tuple(item.ordinal for item in profile.groups[:-1])
    retain_all = frozenset(candidates)
    recompute = frozenset()
    fixed = frozenset(
        item for item in candidates if (item + 1) % fixed_interval == 0
    )
    retain_summary = _summary_for(
        profile,
        retain_all,
        kind="retain_all",
        activation_budget_bytes=activation_working_set_budget_bytes,
        workspace_budget_bytes=workspace_working_set_budget_bytes,
    )
    recompute_summary = _summary_for(
        profile,
        recompute,
        kind="recompute",
        activation_budget_bytes=activation_working_set_budget_bytes,
        workspace_budget_bytes=workspace_working_set_budget_bytes,
    )
    fixed_summary = _summary_for(
        profile,
        fixed,
        kind="fixed_interval_v1",
        activation_budget_bytes=activation_working_set_budget_bytes,
        workspace_budget_bytes=workspace_working_set_budget_bytes,
    )
    try:
        if policy_kind == "fixed_interval_v1":
            selected = fixed
        elif policy_kind in {"measured_budget_v1", "logical_budget_v1"}:
            selected = _dynamic_anchors(
                profile,
                activation_budget_bytes=activation_working_set_budget_bytes,
                workspace_budget_bytes=workspace_working_set_budget_bytes,
                max_replay_groups=max_replay_groups,
            )
        else:
            raise ActivationPlanError(f"unsupported activation planner policy: {policy_kind}")
        selected_summary = _summary_for(
            profile,
            selected,
            kind=policy_kind,
            activation_budget_bytes=activation_working_set_budget_bytes,
            workspace_budget_bytes=workspace_working_set_budget_bytes,
        )
        if not selected_summary.activation_budget_respected:
            raise ActivationPlanInfeasibleError("selected anchors exceed activation budget")
        if not selected_summary.workspace_budget_respected:
            raise ActivationPlanInfeasibleError("selected schedule exceeds workspace budget")
        if (
            max_replay_groups is not None
            and selected_summary.maximum_replayed_groups > max_replay_groups
        ):
            raise ActivationPlanInfeasibleError("selected schedule exceeds replay limit")
        feasible = True
        rejection = None
    except ActivationPlanInfeasibleError as exc:
        selected = frozenset()
        selected_summary = recompute_summary
        feasible = False
        rejection = str(exc)
    segments = _segments_for(profile, selected)
    plan = ActivationPlan(
        schema_version=ACTIVATION_PLAN_SCHEMA_VERSION,
        planner_version=ACTIVATION_PLANNER_VERSION,
        policy_kind=policy_kind,
        profile_checksum=profile.profile_checksum,
        model_signature=profile.model_signature,
        activation_working_set_budget_bytes=activation_working_set_budget_bytes,
        workspace_working_set_budget_bytes=workspace_working_set_budget_bytes,
        max_replay_groups_limit=max_replay_groups,
        fixed_interval=fixed_interval,
        anchor_group_names=selected_summary.anchor_group_names,
        segments=segments,
        maximum_retained_anchor_bytes=selected_summary.retained_anchor_bytes,
        maximum_workspace_bytes=max(item.workspace_bytes for item in profile.groups),
        maximum_replayed_groups=selected_summary.maximum_replayed_groups,
        total_replayed_groups=selected_summary.total_replayed_groups,
        total_parameter_logical_bytes_reread=(
            selected_summary.total_parameter_logical_bytes_reread
        ),
        estimated_replay_seconds=selected_summary.estimated_replay_seconds,
        retain_all_baseline=retain_summary,
        recompute_baseline=recompute_summary,
        fixed_interval_baseline=fixed_summary,
        feasible=feasible,
        rejection_reason=rejection,
    ).with_checksum()
    if feasible:
        plan.validate()
    return plan


def load_activation_plan(path: str | Path) -> ActivationPlan:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ActivationPlanError("activation plan must be a JSON object")
    return ActivationPlan.from_dict(value)


def write_activation_plan(path: str | Path, plan: ActivationPlan) -> None:
    if plan.feasible:
        plan.validate()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(canonical_json_bytes(plan.to_dict()) + b"\n")


def validate_plan_for_config(
    config: ExperimentConfig,
    plan: ActivationPlan,
    *,
    activation_working_set_budget_bytes: int,
    workspace_working_set_budget_bytes: int,
) -> None:
    plan.validate()
    if plan.model_signature != activation_model_signature(config):
        raise ActivationPlanError("activation plan model signature mismatch")
    if plan.activation_working_set_budget_bytes != activation_working_set_budget_bytes:
        raise ActivationPlanError("activation plan budget differs from requested budget")
    if plan.workspace_working_set_budget_bytes != workspace_working_set_budget_bytes:
        raise ActivationPlanError("workspace plan budget differs from requested budget")
