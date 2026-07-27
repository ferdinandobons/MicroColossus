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
    assert plan.estimated_streamed_vram_peak_bytes > 0
    assert plan.model_state_fits_nvme_budget
    assert plan.estimate_kind.endswith("v0")
