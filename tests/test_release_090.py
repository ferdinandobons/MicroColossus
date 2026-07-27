from __future__ import annotations

import microcolossus
from microcolossus.bounded_training import (
    BATCH_STREAM_VERSION,
    BOUNDED_TRAINING_SCHEMA_VERSION,
    MULTI_STEP_RUNTIME_VERSION,
)


def test_bounded_training_release_version_and_public_api() -> None:
    assert microcolossus.__version__ == "0.9.0"
    assert callable(microcolossus.run_bounded_training)
    assert microcolossus.BoundedTrainingResult.__name__ == "BoundedTrainingResult"
    assert BOUNDED_TRAINING_SCHEMA_VERSION == "microcolossus.bounded-training.v1"
    assert MULTI_STEP_RUNTIME_VERSION == "0.9.0"
    assert BATCH_STREAM_VERSION == "synthetic-seed-per-cursor-v1"
