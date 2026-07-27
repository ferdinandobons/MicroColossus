from __future__ import annotations

import microcolossus


def test_bounded_step_release_version_and_public_api() -> None:
    assert microcolossus.__version__ == "0.8.0"
    assert callable(microcolossus.run_bounded_optimizer_step)
    assert microcolossus.StepBundleStore.__name__ == "StepBundleStore"
