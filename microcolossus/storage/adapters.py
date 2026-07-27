"""PyTorch and MLX adapters for the canonical tensor representation."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any, cast

from .codec import TensorPayload, payload_from_numpy, payload_to_numpy
from .schema import TensorKind

_TORCH_DTYPE_NAMES = {
    "torch.bool": "bool",
    "torch.uint8": "uint8",
    "torch.int8": "int8",
    "torch.int16": "int16",
    "torch.int32": "int32",
    "torch.int64": "int64",
    "torch.float16": "float16",
    "torch.bfloat16": "bfloat16",
    "torch.float32": "float32",
    "torch.float64": "float64",
    "torch.complex64": "complex64",
    "torch.complex128": "complex128",
}


def _torch_dtype_name(dtype: Any) -> str:
    try:
        return _TORCH_DTYPE_NAMES[str(dtype)]
    except KeyError as exc:
        raise ValueError(f"unsupported PyTorch dtype: {dtype}") from exc


def _torch_bytes(tensor: Any) -> bytes:
    import torch

    value = tensor.detach().cpu().contiguous()
    if value.numel() == 0:
        return b""
    byte_view = value.reshape(-1).view(torch.uint8)
    storage = bytes(cast(Iterable[int], byte_view.untyped_storage()))
    start = byte_view.storage_offset()
    return storage[start : start + byte_view.numel()]


def payload_from_torch(
    tensor: Any,
    *,
    logical_name: str,
    kind: TensorKind,
    metadata: tuple[tuple[str, str], ...] = (),
) -> TensorPayload:
    dtype = _torch_dtype_name(tensor.dtype)
    byte_order = "not_applicable" if tensor.element_size() == 1 else "little"
    return TensorPayload(
        logical_name=logical_name,
        kind=kind,
        shape=tuple(int(item) for item in tensor.shape),
        dtype=dtype,
        byte_order=byte_order,
        data=_torch_bytes(tensor),
        metadata=metadata,
    )


def payload_to_torch(payload: TensorPayload, *, device: Any = "cpu") -> Any:
    import torch

    dtype_map = {
        "bool": torch.bool,
        "uint8": torch.uint8,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
        "complex64": torch.complex64,
        "complex128": torch.complex128,
    }
    try:
        dtype = dtype_map[payload.dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported PyTorch restore dtype: {payload.dtype}") from exc
    if payload.byte_length == 0:
        return torch.empty(payload.shape, dtype=dtype, device=device)
    value = torch.frombuffer(bytearray(payload.data), dtype=dtype).clone()
    return value.reshape(payload.shape).to(device)


def export_pytorch_model(model: Any, *, include_buffers: bool = True) -> tuple[TensorPayload, ...]:
    payloads: list[TensorPayload] = []
    for name, parameter in model.named_parameters(remove_duplicate=True):
        payloads.append(
            payload_from_torch(
                parameter,
                logical_name=f"model.{name}",
                kind=TensorKind.PARAMETER,
                metadata=(("backend", "pytorch"), ("model_name", name)),
            )
        )
    if include_buffers:
        for name, buffer in model.named_buffers(remove_duplicate=True):
            payloads.append(
                payload_from_torch(
                    buffer,
                    logical_name=f"model_buffer.{name}",
                    kind=TensorKind.METADATA,
                    metadata=(("backend", "pytorch"), ("buffer_name", name)),
                )
            )
    return tuple(sorted(payloads, key=lambda item: item.logical_name))


def restore_pytorch_model(model: Any, payloads: Iterable[TensorPayload]) -> None:
    import torch

    parameters = dict(model.named_parameters(remove_duplicate=True))
    buffers = dict(model.named_buffers(remove_duplicate=True))
    seen_parameters: set[str] = set()
    with torch.no_grad():
        for payload in payloads:
            metadata = dict(payload.metadata)
            if "model_name" in metadata:
                name = metadata["model_name"]
                if name not in parameters:
                    raise KeyError(f"PyTorch parameter not found: {name}")
                restored = payload_to_torch(payload, device=parameters[name].device)
                parameters[name].copy_(restored)
                seen_parameters.add(name)
            elif "buffer_name" in metadata:
                name = metadata["buffer_name"]
                if name not in buffers:
                    raise KeyError(f"PyTorch buffer not found: {name}")
                restored = payload_to_torch(payload, device=buffers[name].device)
                buffers[name].copy_(restored)
    missing = set(parameters) - seen_parameters
    if missing:
        raise KeyError(f"missing PyTorch parameters: {sorted(missing)}")


def _json_payload(
    value: Any,
    *,
    logical_name: str,
    metadata: tuple[tuple[str, str], ...] = (),
) -> TensorPayload:
    data = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return TensorPayload(
        logical_name=logical_name,
        kind=TensorKind.METADATA,
        shape=(len(data),),
        dtype="uint8",
        byte_order="not_applicable",
        data=data,
        metadata=metadata,
    )


def _json_from_payload(payload: TensorPayload) -> Any:
    return json.loads(payload.data.decode("utf-8"))


def export_pytorch_adamw(model: Any, optimizer: Any) -> tuple[TensorPayload, ...]:
    parameters = dict(model.named_parameters(remove_duplicate=True))
    name_by_identity = {id(parameter): name for name, parameter in parameters.items()}
    payloads: list[TensorPayload] = []
    for parameter, state in optimizer.state.items():
        name = name_by_identity.get(id(parameter))
        if name is None:
            raise KeyError("optimizer contains a parameter not present in the model")
        for state_key, value in sorted(state.items()):
            if not hasattr(value, "dtype"):
                payloads.append(
                    _json_payload(
                        value,
                        logical_name=f"optimizer.{name}.{state_key}",
                        metadata=(
                            ("backend", "pytorch"),
                            ("parameter_name", name),
                            ("state_key", str(state_key)),
                        ),
                    )
                )
                continue
            kind = {
                "exp_avg": TensorKind.ADAM_FIRST_MOMENT,
                "exp_avg_sq": TensorKind.ADAM_SECOND_MOMENT,
            }.get(str(state_key), TensorKind.METADATA)
            payloads.append(
                payload_from_torch(
                    value,
                    logical_name=f"optimizer.{name}.{state_key}",
                    kind=kind,
                    metadata=(
                        ("backend", "pytorch"),
                        ("parameter_name", name),
                        ("state_key", str(state_key)),
                    ),
                )
            )
    groups: list[dict[str, Any]] = []
    for group in optimizer.param_groups:
        encoded = {
            key: value
            for key, value in group.items()
            if key != "params" and isinstance(value, (str, int, float, bool, type(None)))
        }
        encoded["params"] = [name_by_identity[id(item)] for item in group["params"]]
        groups.append(encoded)
    payloads.append(
        _json_payload(
            groups,
            logical_name="optimizer.param_groups",
            metadata=(("backend", "pytorch"), ("record_type", "param_groups")),
        )
    )
    return tuple(sorted(payloads, key=lambda item: item.logical_name))


def restore_pytorch_adamw(
    model: Any,
    optimizer: Any,
    payloads: Iterable[TensorPayload],
) -> None:
    parameters = dict(model.named_parameters(remove_duplicate=True))
    group_payload: TensorPayload | None = None
    for payload in payloads:
        metadata = dict(payload.metadata)
        if metadata.get("record_type") == "param_groups":
            group_payload = payload
            continue
        parameter_name = metadata.get("parameter_name")
        state_key = metadata.get("state_key")
        if parameter_name is None or state_key is None:
            continue
        parameter = parameters[parameter_name]
        if payload.dtype == "uint8" and payload.kind is TensorKind.METADATA:
            value = _json_from_payload(payload)
        else:
            value = payload_to_torch(payload, device=parameter.device)
        optimizer.state[parameter][state_key] = value
    if group_payload is not None:
        encoded_groups = _json_from_payload(group_payload)
        if len(encoded_groups) != len(optimizer.param_groups):
            raise ValueError("optimizer parameter-group count does not match")
        for target, encoded in zip(optimizer.param_groups, encoded_groups, strict=True):
            expected_names = [
                name
                for name, parameter in parameters.items()
                if any(parameter is item for item in target["params"])
            ]
            if sorted(encoded["params"]) != sorted(expected_names):
                raise ValueError("optimizer parameter-group membership does not match")
            for key, value in encoded.items():
                if key != "params":
                    target[key] = value


def export_pytorch_state(model: Any, optimizer: Any | None = None) -> tuple[TensorPayload, ...]:
    payloads = list(export_pytorch_model(model))
    if optimizer is not None:
        payloads.extend(export_pytorch_adamw(model, optimizer))
    return tuple(sorted(payloads, key=lambda item: item.logical_name))


def restore_pytorch_state(
    model: Any,
    payloads: Iterable[TensorPayload],
    optimizer: Any | None = None,
) -> None:
    materialized = tuple(payloads)
    restore_pytorch_model(
        model,
        [item for item in materialized if dict(item.metadata).get("model_name")],
    )
    if optimizer is not None:
        restore_pytorch_adamw(
            model,
            optimizer,
            [item for item in materialized if item.logical_name.startswith("optimizer.")],
        )


def export_mlx_model(model: Any) -> tuple[TensorPayload, ...]:
    from mlx.utils import tree_flatten

    flattened = tree_flatten(model.parameters())
    if not isinstance(flattened, list):
        raise TypeError("MLX tree_flatten returned a mapping instead of a list")
    payloads = [
        payload_from_numpy(
            value,
            logical_name=f"model.{name}",
            kind=TensorKind.PARAMETER,
            metadata=(("backend", "mlx"), ("model_name", name)),
        )
        for name, value in flattened
    ]
    return tuple(sorted(payloads, key=lambda item: item.logical_name))


def restore_mlx_model(model: Any, payloads: Iterable[TensorPayload]) -> None:
    import mlx.core as mx

    weights = []
    for payload in payloads:
        metadata = dict(payload.metadata)
        name = metadata.get("model_name")
        if name is None:
            continue
        weights.append((name, mx.array(payload_to_numpy(payload))))
    model.load_weights(sorted(weights), strict=True)
    mx.eval(model.parameters())


def _mlx_optimizer_kind(path: str) -> TensorKind:
    suffix = path.rsplit(".", maxsplit=1)[-1]
    if suffix in {"m", "exp_avg"}:
        return TensorKind.ADAM_FIRST_MOMENT
    if suffix in {"v", "exp_avg_sq"}:
        return TensorKind.ADAM_SECOND_MOMENT
    return TensorKind.METADATA


def export_mlx_optimizer(optimizer: Any) -> tuple[TensorPayload, ...]:
    from mlx.utils import tree_flatten

    flattened = tree_flatten(optimizer.state)
    if not isinstance(flattened, list):
        raise TypeError("MLX tree_flatten returned a mapping instead of a list")
    return tuple(
        sorted(
            (
                payload_from_numpy(
                    value,
                    logical_name=f"optimizer.{path}",
                    kind=_mlx_optimizer_kind(path),
                    metadata=(
                        ("backend", "mlx"),
                        ("optimizer_path", path),
                    ),
                )
                for path, value in flattened
            ),
            key=lambda item: item.logical_name,
        )
    )


def restore_mlx_optimizer(optimizer: Any, payloads: Iterable[TensorPayload]) -> None:
    import mlx.core as mx
    from mlx.utils import tree_unflatten

    flattened = []
    for payload in payloads:
        path = dict(payload.metadata).get("optimizer_path")
        if path is not None:
            flattened.append((path, mx.array(payload_to_numpy(payload))))
    optimizer.state = tree_unflatten(sorted(flattened))
    mx.eval(optimizer.state)


def export_mlx_state(model: Any, optimizer: Any | None = None) -> tuple[TensorPayload, ...]:
    payloads = list(export_mlx_model(model))
    if optimizer is not None:
        payloads.extend(export_mlx_optimizer(optimizer))
    return tuple(sorted(payloads, key=lambda item: item.logical_name))


def restore_mlx_state(
    model: Any,
    payloads: Iterable[TensorPayload],
    optimizer: Any | None = None,
) -> None:
    materialized = tuple(payloads)
    restore_mlx_model(
        model,
        [item for item in materialized if dict(item.metadata).get("model_name")],
    )
    if optimizer is not None:
        restore_mlx_optimizer(
            optimizer,
            [item for item in materialized if item.logical_name.startswith("optimizer.")],
        )
