from __future__ import annotations

import microcolossus

from microcolossus.activation_planner import (
    ACTIVATION_PLAN_SCHEMA_VERSION,
    ACTIVATION_PLANNER_VERSION,
    ACTIVATION_PROFILE_SCHEMA_VERSION,
)
from microcolossus.training_checkpoint import (
    BOUNDED_TRAINING_SCHEMA_VERSION,
    HYBRID_ACTIVATION_RUNTIME_VERSION,
)


def test_release_013_contract() -> None:
    assert microcolossus.__version__ == "0.13.0"
    assert ACTIVATION_PLANNER_VERSION == "0.13.0"
    assert HYBRID_ACTIVATION_RUNTIME_VERSION == "0.13.0"
    assert ACTIVATION_PROFILE_SCHEMA_VERSION == "microcolossus.activation-profile.v1"
    assert ACTIVATION_PLAN_SCHEMA_VERSION == "microcolossus.activation-plan.v1"
    assert BOUNDED_TRAINING_SCHEMA_VERSION == "microcolossus.bounded-training.v4"
    assert callable(microcolossus.build_activation_plan)
