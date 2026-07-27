"""Static memory estimator for the first MicroColossus milestones."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .config import ExperimentConfig


GIB = 1024**3


@dataclass(frozen=True)
class MemoryPlan:
    """Approximate memory report. It is not a runtime guarantee."""

    estimate_kind: str
    memory_architecture: str
    parameter_count: int
    parameter_bytes: int
    gradient_bytes: int
    optimizer_state_bytes: int
    master_weight_bytes: int
    resident_persistent_bytes: int
    estimated_saved_activation_bytes: int
    largest_layer_parameter_bytes: int
    estimated_streamed_accelerator_peak_bytes: int
    estimated_streamed_ram_peak_bytes: int
    accelerator_memory_budget_bytes: int
    process_ram_budget_bytes: int
    system_memory_budget_bytes: int | None
    nvme_budget_bytes: int
    resident_fits_accelerator_budget: bool
    streamed_working_set_fits_accelerator_budget: bool
    streamed_working_set_fits_ram_budget: bool
    model_state_fits_nvme_budget: bool
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _parameter_bytes(model: nn.Module) -> int:
    return sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def _largest_execution_group_bytes(model: nn.Module) -> int:
    groups: list[int] = []
    blocks = getattr(model, "blocks", None)
    if isinstance(blocks, nn.ModuleList):
        groups.extend(_parameter_bytes(block) for block in blocks)
    for module_name in ("token_embedding", "position_embedding", "final_norm", "lm_head"):
        module = getattr(model, module_name, None)
        if isinstance(module, nn.Module):
            groups.append(_parameter_bytes(module))
    return max(groups, default=_parameter_bytes(model))


def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def _resolve_memory_architecture(config: ExperimentConfig) -> str:
    configured = config.hardware.memory_architecture
    if configured != "auto":
        return configured
    if config.training.device == "mps":
        return "unified"
    if config.training.device == "auto" and _mps_available():
        return "unified"
    return "discrete"


def build_static_plan(model: nn.Module, config: ExperimentConfig) -> MemoryPlan:
    """Build a conservative first-pass plan from model and experiment metadata.

    The activation and streamed working-set estimates are heuristics. Runtime
    instrumentation will replace them as execution components are implemented.
    """

    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    parameter_bytes = _parameter_bytes(model)
    gradient_bytes = parameter_bytes
    optimizer_state_bytes = parameter_count * 8
    master_weight_bytes = sum(
        parameter.numel() * 4
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.element_size() < 4
    )
    resident_persistent_bytes = (
        parameter_bytes + gradient_bytes + optimizer_state_bytes + master_weight_bytes
    )

    model_config = config.model
    training_config = config.training
    activation_elements = (
        training_config.micro_batch_size
        * training_config.sequence_length
        * model_config.hidden_size
        * model_config.layers
    )
    estimated_saved_activation_bytes = activation_elements * 4 * 8

    largest_layer_parameter_bytes = _largest_execution_group_bytes(model)
    one_block_activation_bytes = (
        training_config.micro_batch_size
        * training_config.sequence_length
        * model_config.hidden_size
        * 4
        * 8
    )
    transfer_buffers = largest_layer_parameter_bytes * 2
    estimated_streamed_accelerator_peak_bytes = (
        largest_layer_parameter_bytes + one_block_activation_bytes + transfer_buffers
    )
    estimated_streamed_ram_peak_bytes = transfer_buffers + largest_layer_parameter_bytes

    hardware = config.hardware
    memory_architecture = _resolve_memory_architecture(config)
    accelerator_memory_budget_bytes = int(hardware.accelerator_memory_gib * GIB)
    process_ram_budget_bytes = int(hardware.process_ram_gib * GIB)
    system_memory_budget_bytes = (
        int(hardware.system_memory_gib * GIB)
        if hardware.system_memory_gib is not None
        else None
    )
    nvme_budget_bytes = int(hardware.nvme_gib * GIB)

    warnings = [
        "Activation and workspace estimates are heuristic and must be replaced by measured peaks.",
        "The planner does not yet model allocator fragmentation, page cache, "
        "or asynchronous overlap.",
        "A feasible capacity estimate does not imply acceptable throughput.",
    ]
    if memory_architecture == "unified":
        warnings.extend(
            [
                "Apple Silicon CPU and MPS allocations share unified physical memory; "
                "accelerator and process budgets are overlapping logical guardrails.",
                "Process RSS, MPS current allocation, and MPS driver allocation cannot be "
                "summed to infer total physical memory use.",
                "CPU-to-MPS placement alone is not a capacity offload mechanism on unified memory.",
            ]
        )
        if system_memory_budget_bytes is None:
            warnings.append(
                "No system_memory_gib value was configured for this unified-memory plan."
            )
    if config.training.device == "mps" and not _mps_available():
        warnings.append("MPS was requested but is not available in the current environment.")

    return MemoryPlan(
        estimate_kind="static-model-plus-heuristics-v1",
        memory_architecture=memory_architecture,
        parameter_count=parameter_count,
        parameter_bytes=parameter_bytes,
        gradient_bytes=gradient_bytes,
        optimizer_state_bytes=optimizer_state_bytes,
        master_weight_bytes=master_weight_bytes,
        resident_persistent_bytes=resident_persistent_bytes,
        estimated_saved_activation_bytes=estimated_saved_activation_bytes,
        largest_layer_parameter_bytes=largest_layer_parameter_bytes,
        estimated_streamed_accelerator_peak_bytes=estimated_streamed_accelerator_peak_bytes,
        estimated_streamed_ram_peak_bytes=estimated_streamed_ram_peak_bytes,
        accelerator_memory_budget_bytes=accelerator_memory_budget_bytes,
        process_ram_budget_bytes=process_ram_budget_bytes,
        system_memory_budget_bytes=system_memory_budget_bytes,
        nvme_budget_bytes=nvme_budget_bytes,
        resident_fits_accelerator_budget=(
            resident_persistent_bytes + estimated_saved_activation_bytes
            <= accelerator_memory_budget_bytes
        ),
        streamed_working_set_fits_accelerator_budget=(
            estimated_streamed_accelerator_peak_bytes <= accelerator_memory_budget_bytes
        ),
        streamed_working_set_fits_ram_budget=(
            estimated_streamed_ram_peak_bytes <= process_ram_budget_bytes
        ),
        model_state_fits_nvme_budget=(resident_persistent_bytes <= nvme_budget_bytes),
        warnings=tuple(warnings),
    )
