"""Configuration for the UCCL-P2P transfer backend."""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

UCCL_MINIMUM_COMMIT = "51d91bceabf27be88fc9198cf79b6d4e702bed73"


@dataclass
class UcclConfig:
    """Runtime knobs for the low-level, CXI-compatible UCCL-P2P API."""

    local_gpu_idx: int = 0
    transport: str | None = None
    use_gpu_direct: bool = False
    timeout_seconds: float = 60.0
    poll_interval_seconds: float = 0.001
    retention_ttl_seconds: float = 300.0
    ack_bind_host: str = "0.0.0.0"
    ack_advertise_host: str | None = None

    def __post_init__(self) -> None:
        if self.local_gpu_idx < 0:
            raise ValueError("local_gpu_idx must be non-negative")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.poll_interval_seconds < 0:
            raise ValueError("poll_interval_seconds must be non-negative")
        if self.retention_ttl_seconds <= 0:
            raise ValueError("retention_ttl_seconds must be positive")

    @classmethod
    def from_flat_args(cls, args: Any) -> "UcclConfig":
        """Build from the ``uccl_*`` names emitted by config flattening."""

        def get(name: str, default: Any) -> Any:
            if isinstance(args, Mapping):
                value = args.get(f"uccl_{name}", default)
            else:
                value = getattr(args, f"uccl_{name}", default)
            return default if value is None else value

        defaults = cls()
        return cls(
            local_gpu_idx=get("local_gpu_idx", defaults.local_gpu_idx),
            transport=get("transport", defaults.transport),
            use_gpu_direct=get("use_gpu_direct", defaults.use_gpu_direct),
            timeout_seconds=get("timeout_seconds", defaults.timeout_seconds),
            poll_interval_seconds=get("poll_interval_seconds", defaults.poll_interval_seconds),
            retention_ttl_seconds=get("retention_ttl_seconds", defaults.retention_ttl_seconds),
            ack_bind_host=get("ack_bind_host", defaults.ack_bind_host),
            ack_advertise_host=get("ack_advertise_host", defaults.ack_advertise_host),
        )

    @classmethod
    def from_env(cls) -> "UcclConfig":
        """Build from environment inherited by an inference subprocess."""

        def optional(name: str) -> str | None:
            value = os.getenv(name)
            return value if value else None

        def boolean(name: str, default: bool) -> bool:
            value = optional(name)
            return default if value is None else value.lower() in {"1", "true", "yes", "on"}

        return cls(
            local_gpu_idx=int(os.getenv("UCCL_P2P_LOCAL_GPU_IDX", "0")),
            transport=optional("UCCL_P2P_TRANSPORT"),
            use_gpu_direct=boolean("UCCL_P2P_USE_GPU_DIRECT", False),
            timeout_seconds=float(os.getenv("UCCL_P2P_TIMEOUT_SECONDS", "60.0")),
            poll_interval_seconds=float(os.getenv("UCCL_P2P_POLL_INTERVAL_SECONDS", "0.001")),
            retention_ttl_seconds=float(os.getenv("UCCL_P2P_RETENTION_TTL_SECONDS", "300.0")),
            ack_bind_host=os.getenv("UCCL_P2P_ACK_BIND_HOST", "0.0.0.0"),
            ack_advertise_host=optional("UCCL_P2P_ACK_ADVERTISE_HOST"),
        )

    def export_env(self) -> None:
        """Export child-process settings without replacing explicit UCCL overrides."""

        os.environ["TORCHSPEC_TRANSFER_BACKEND"] = "uccl-p2p"
        values = {
            "UCCL_P2P_LOCAL_GPU_IDX": str(self.local_gpu_idx),
            "UCCL_P2P_USE_GPU_DIRECT": "1" if self.use_gpu_direct else "0",
            "UCCL_P2P_TIMEOUT_SECONDS": str(self.timeout_seconds),
            "UCCL_P2P_POLL_INTERVAL_SECONDS": str(self.poll_interval_seconds),
            "UCCL_P2P_RETENTION_TTL_SECONDS": str(self.retention_ttl_seconds),
            "UCCL_P2P_ACK_BIND_HOST": self.ack_bind_host,
        }
        if self.transport:
            values["UCCL_P2P_TRANSPORT"] = self.transport
        if self.ack_advertise_host:
            values["UCCL_P2P_ACK_ADVERTISE_HOST"] = self.ack_advertise_host
        for name, value in values.items():
            os.environ.setdefault(name, value)
