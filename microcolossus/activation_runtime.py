"""Resolve and validate activation plans used by persistent training."""

from __future__ import annotations

from .activation_planner import (
    ActivationPlan,
    ActivationPlanError,
    build_activation_plan,
    build_logical_activation_profile,
    load_activation_plan,
    validate_plan_for_config,
)
from .config import ExperimentConfig


def resolve_activation_plan(
    config: ExperimentConfig,
    *,
    activation_working_set_bytes: int,
    workspace_working_set_bytes: int,
) -> ActivationPlan | None:
    """Return the deterministic hybrid plan, or ``None`` for older policies."""

    if config.training.activation_policy != "hybrid":
        return None
    policy = config.training.activation_anchor_policy
    if policy is None:
        raise ActivationPlanError("hybrid activation policy has no planner configuration")
    if policy.activation_working_set_bytes != activation_working_set_bytes:
        raise ActivationPlanError(
            "hybrid activation budget differs from activation_anchor_policy"
        )
    if policy.workspace_working_set_bytes != workspace_working_set_bytes:
        raise ActivationPlanError(
            "hybrid workspace budget differs from activation_anchor_policy"
        )
    if policy.kind == "measured_budget_v1":
        if policy.plan_path is None:
            raise ActivationPlanError("measured hybrid policy has no plan path")
        plan = load_activation_plan(policy.plan_path)
        if policy.plan_checksum != plan.plan_checksum:
            raise ActivationPlanError("configured activation plan checksum changed")
    else:
        profile = build_logical_activation_profile(config)
        plan = build_activation_plan(
            profile,
            policy_kind=policy.kind,
            activation_working_set_budget_bytes=activation_working_set_bytes,
            workspace_working_set_budget_bytes=workspace_working_set_bytes,
            max_replay_groups=policy.max_replay_groups,
            fixed_interval=policy.fixed_interval,
        )
    validate_plan_for_config(
        config,
        plan,
        activation_working_set_budget_bytes=activation_working_set_bytes,
        workspace_working_set_budget_bytes=workspace_working_set_bytes,
    )
    return plan
