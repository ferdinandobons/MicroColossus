from __future__ import annotations

from pathlib import Path

import microcolossus
from microcolossus.bounded_training import (
    BATCH_STREAM_VERSION,
    BOUNDED_TRAINING_SCHEMA_VERSION,
    MULTI_STEP_RUNTIME_VERSION,
)
from microcolossus.config import RetentionConfig, load_experiment_config
from microcolossus.data import (
    DATA_IDENTITY_SCHEMA_VERSION,
    TEXT_BATCH_STREAM_VERSION,
    UTF8_BYTE_TOKENIZER_VERSION,
)
from microcolossus.evaluation import EVALUATION_SCHEMA_VERSION, PROGRESS_SCHEMA_VERSION
from microcolossus.model import DecoderOnlyTransformer
from microcolossus.pruning import (
    PRUNING_OPERATION_SCHEMA_VERSION,
    PRUNING_PLAN_SCHEMA_VERSION,
    PRUNING_REPORT_SCHEMA_VERSION,
)


def test_real_text_training_release_version_and_public_api() -> None:
    assert microcolossus.__version__ == "0.11.0"
    assert callable(microcolossus.run_bounded_training)
    assert callable(microcolossus.prepare_data_source)
    assert callable(microcolossus.build_pruning_plan)
    assert callable(microcolossus.apply_pruning_plan)
    assert microcolossus.DataConfig.__name__ == "DataConfig"
    assert microcolossus.EvaluationResult.__name__ == "EvaluationResult"
    assert microcolossus.RetentionConfig.__name__ == "RetentionConfig"
    assert BOUNDED_TRAINING_SCHEMA_VERSION == "microcolossus.bounded-training.v2"
    assert MULTI_STEP_RUNTIME_VERSION == "0.10.0"
    assert BATCH_STREAM_VERSION == "configured-data-source-v1"
    assert DATA_IDENTITY_SCHEMA_VERSION == "microcolossus.data-identity.v1"
    assert UTF8_BYTE_TOKENIZER_VERSION == "utf8-bytes-v1"
    assert TEXT_BATCH_STREAM_VERSION == "utf8-byte-random-window-v1"
    assert EVALUATION_SCHEMA_VERSION == "microcolossus.evaluation.v1"
    assert PROGRESS_SCHEMA_VERSION == "microcolossus.training-progress.v1"
    assert PRUNING_PLAN_SCHEMA_VERSION == "microcolossus.pruning-plan.v1"
    assert PRUNING_OPERATION_SCHEMA_VERSION == "microcolossus.pruning-operation.v1"
    assert PRUNING_REPORT_SCHEMA_VERSION == "microcolossus.pruning-report.v1"


def test_real_text_micro_example_parameter_count_and_retention() -> None:
    config = load_experiment_config(Path("examples/real-text-micro.yaml"))
    assert config.model.vocab_size == 256
    assert config.model.max_sequence_length == 64
    assert DecoderOnlyTransformer(config.model).parameter_count == 18_624
    assert config.retention == RetentionConfig(keep_previous=2, milestone_interval=5)
