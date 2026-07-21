"""Lazy construction of configured transfer backends."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, replace
from typing import Any

from torchspec.transfer.base import TransferRole, TransferUnavailableError


def _select(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            return dict(OmegaConf.to_container(value, resolve=True))
    except ImportError:
        pass
    raise TypeError(f"transfer backend options must be a mapping, got {type(value).__name__}")


def create_transfer_backend(config: Any) -> Any:
    """Create the selected backend without importing optional backends eagerly.

    Optional backend modules are imported only after configuration selects them.
    """

    from torchspec.transfer.uccl import UcclConfig, UcclP2PBackend

    if isinstance(config, UcclConfig):
        return UcclP2PBackend(config)

    try:
        from torchspec.config.mooncake_config import MooncakeConfig

        if isinstance(MooncakeConfig, type) and isinstance(config, MooncakeConfig):
            from torchspec.transfer.mooncake import EagleMooncakeStore, MooncakeTransferBackend

            return MooncakeTransferBackend(EagleMooncakeStore(config))
    except ImportError:
        MooncakeConfig = None

    transfer = _select(config, "transfer", {})
    backend = str(
        _select(transfer, "backend", _select(config, "transfer_backend", "mooncake"))
    ).lower()
    options = _mapping(_select(config, "mooncake", {}))
    options.update(_mapping(_select(transfer, "mooncake", {})))
    options.update(_mapping(_select(transfer, "options", {})))
    options.update(_mapping(_select(config, "transfer_options", {})))

    if backend == "mooncake":
        try:
            from torchspec.config.mooncake_config import MooncakeConfig
            from torchspec.transfer.mooncake import EagleMooncakeStore, MooncakeTransferBackend
        except ImportError as exc:
            raise TransferUnavailableError(
                "Mooncake transfer backend is unavailable; install the Mooncake dependencies"
            ) from exc
        if _select(config, "transfer_backend", None) is not None:
            mooncake_config = MooncakeConfig.from_flat_args(config)
            if options:
                mooncake_config = replace(mooncake_config, **options)
        else:
            mooncake_config = MooncakeConfig(**options)
        return MooncakeTransferBackend(EagleMooncakeStore(mooncake_config))

    if backend in {"uccl", "uccl-p2p"}:
        uccl_options = _mapping(_select(config, "uccl", {}))
        if any(_select(config, f"uccl_{name}", None) is not None for name in asdict(UcclConfig())):
            uccl_options.update(asdict(UcclConfig.from_flat_args(config)))
        uccl_options.update(_mapping(_select(transfer, "options", {})))
        uccl_options.update(_mapping(_select(config, "transfer_options", {})))
        return UcclP2PBackend(UcclConfig(**uccl_options))

    raise TransferUnavailableError(f"unknown transfer backend: {backend!r}")


def create_transfer_backend_from_env(
    device: Any = None,
    role: TransferRole = TransferRole.CONSUMER,
) -> Any:
    """Construct and set up a backend inside an inherited child environment."""

    import os

    backend_name = os.getenv("TORCHSPEC_TRANSFER_BACKEND", "mooncake").lower()
    if backend_name == "mooncake":
        from torchspec.config.mooncake_config import MooncakeConfig

        config = MooncakeConfig.from_env()
    elif backend_name in {"uccl", "uccl-p2p"}:
        from torchspec.transfer.uccl import UcclConfig

        config = UcclConfig.from_env()
    else:
        raise TransferUnavailableError(f"unknown transfer backend: {backend_name!r}")
    backend = create_transfer_backend(config)
    backend.setup(role, device)
    return backend
