"""Backend-neutral, versioned tensor storage."""

from .adapters import (
    export_mlx_state,
    export_pytorch_state,
    restore_mlx_state,
    restore_pytorch_state,
)
from .codec import TensorPayload, payload_from_numpy, payload_to_numpy
from .schema import (
    ChunkRecord,
    FailurePoint,
    Manifest,
    StoreLimits,
    TensorKind,
    TensorRecord,
    TransactionState,
)
from .store import (
    BudgetExceededError,
    CommitResult,
    IntegrityError,
    RecoveryReport,
    SimulatedCrash,
    TensorStoreError,
    VerificationReport,
    VersionedTensorStore,
)

__all__ = [
    "BudgetExceededError",
    "ChunkRecord",
    "CommitResult",
    "FailurePoint",
    "IntegrityError",
    "Manifest",
    "RecoveryReport",
    "SimulatedCrash",
    "StoreLimits",
    "TensorKind",
    "TensorPayload",
    "TensorRecord",
    "TensorStoreError",
    "TransactionState",
    "VerificationReport",
    "VersionedTensorStore",
    "export_mlx_state",
    "export_pytorch_state",
    "payload_from_numpy",
    "payload_to_numpy",
    "restore_mlx_state",
    "restore_pytorch_state",
]
