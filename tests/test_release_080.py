from __future__ import annotations

import microcolossus
from microcolossus.bounded_optimizer import BOUNDED_OPTIMIZER_SCHEMA_VERSION
from microcolossus.step_bundle import STEP_BUNDLE_SCHEMA_VERSION


def test_bounded_step_release_version_and_public_api() -> None:
    assert microcolossus.__version__ == "0.8.0"
    assert callable(microcolossus.run_bounded_optimizer_step)
    assert microcolossus.StepBundleStore.__name__ == "StepBundleStore"
    assert BOUNDED_OPTIMIZER_SCHEMA_VERSION == "microcolossus.bounded-optimizer.v1"
    assert STEP_BUNDLE_SCHEMA_VERSION == "microcolossus.step-bundle.v1"
