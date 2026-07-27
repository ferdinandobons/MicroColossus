from microcolossus.config import load_experiment_config
from microcolossus.model import DecoderOnlyTransformer
from microcolossus.planner import build_static_plan


def test_static_plan_reports_capacity_fields() -> None:
    config = load_experiment_config("examples/tiny-resident.yaml")
    model = DecoderOnlyTransformer(config.model)
    plan = build_static_plan(model, config)

    assert plan.parameter_count == model.parameter_count
    assert plan.parameter_bytes > 0
    assert plan.resident_persistent_bytes > plan.parameter_bytes
    assert plan.estimated_streamed_accelerator_peak_bytes > 0
    assert plan.model_state_fits_nvme_budget
    assert plan.estimate_kind.endswith("v1")


def test_mps_plan_explains_unified_memory() -> None:
    config = load_experiment_config("examples/tiny-mps.yaml")
    model = DecoderOnlyTransformer(config.model)
    plan = build_static_plan(model, config)

    assert plan.memory_architecture == "unified"
    assert plan.system_memory_budget_bytes == 8 * 1024**3
    assert any("unified physical memory" in warning for warning in plan.warnings)
    assert any("cannot be summed" in warning for warning in plan.warnings)
