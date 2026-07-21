# Copyright (c) 2026 LightSeek Foundation

"""Backend-neutral adapter for the legacy Eagle Mooncake store."""

from __future__ import annotations

from typing import Any

import torch

from torchspec.transfer.base import (
    TensorBundle,
    TensorSpec,
    TransferBackend,
    TransferProtocolError,
    TransferRef,
    TransferRole,
)


def mooncake_ref_key(ref: TransferRef) -> str:
    return str(ref.locator.get("key", ref.object_id))


class MooncakeTransferBackend(TransferBackend):
    """Expose :class:`EagleMooncakeStore` through ``TransferBackend``."""

    def __init__(self, store: Any) -> None:
        super().__init__()
        self.store = store

    def _setup(self, role: TransferRole, device: torch.device | str | int | None) -> None:
        if hasattr(self.store, "setup"):
            self.store.setup(device=device)

    def _put(
        self,
        object_id: str,
        tensors: TensorBundle,
        expected_consumers: int,
    ) -> TransferRef:
        del expected_consumers  # Mooncake's replication policy lives in its config.
        result = self.store.put(
            key=object_id,
            hidden_states=tensors["hidden_states"],
            input_ids=tensors["input_ids"],
            target=tensors.get("target"),
            last_hidden_states=tensors.get("last_hidden_states"),
        )
        shapes = result.get("shapes", {}) if isinstance(result, dict) else {}
        dtypes = result.get("dtypes", {}) if isinstance(result, dict) else {}
        specs = {
            name: TensorSpec(
                tuple(shapes.get(name, tensor.shape)),
                str(dtypes.get(name, tensor.dtype)).removeprefix("torch."),
            )
            for name, tensor in tensors.items()
        }
        return TransferRef(
            backend="mooncake",
            object_id=object_id,
            tensors=specs,
            locator={"key": object_id},
        )

    def _flush(self) -> None:
        if hasattr(self.store, "flush"):
            self.store.flush()

    def _get(
        self,
        ref: TransferRef,
        device: torch.device | str | int | None,
    ) -> TensorBundle:
        self._validate_ref(ref)
        shapes = {name: spec.shape for name, spec in ref.tensors.items()}
        dtypes = {name: getattr(torch, spec.dtype) for name, spec in ref.tensors.items()}
        output = self.store.get(
            key=mooncake_ref_key(ref),
            shapes=shapes,
            dtypes=dtypes,
            device=device,
        )
        if hasattr(output, "to_tensor_dict"):
            return output.to_tensor_dict()
        return output

    def _release(self, ref: TransferRef) -> None:
        self._validate_ref(ref)
        names = ref.tensors
        self.store.remove_eagle3_tensors(
            mooncake_ref_key(ref),
            has_last_hidden_states="last_hidden_states" in names,
            has_target="target" in names,
        )

    def _health_check(self) -> None:
        if hasattr(self.store, "_ensure_initialized"):
            self.store._ensure_initialized()

    def _close(self) -> None:
        if hasattr(self.store, "close"):
            self.store.close()

    @staticmethod
    def _validate_ref(ref: TransferRef) -> None:
        if ref.backend != "mooncake":
            raise TransferProtocolError(
                f"Mooncake backend cannot use a {ref.backend!r} transfer reference"
            )


__all__ = ["MooncakeTransferBackend", "mooncake_ref_key"]
