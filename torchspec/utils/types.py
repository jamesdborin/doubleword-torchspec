# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from torchspec.transfer.base import TensorSpec, TransferProtocolError, TransferRef


def legacy_mooncake_transfer_ref(
    key: str,
    shapes: Mapping[str, tuple[int, ...]] | None,
    dtypes: Mapping[str, torch.dtype | str] | None = None,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> TransferRef:
    """Build a backend-neutral reference from the legacy Mooncake fields."""
    dtypes = dtypes or {}
    tensors = {
        name: TensorSpec(
            shape=tuple(shape),
            dtype=str(
                dtypes.get(name, torch.int64 if name == "input_ids" else torch.bfloat16)
            ).removeprefix("torch."),
        )
        for name, shape in (shapes or {}).items()
    }
    return TransferRef(
        backend="mooncake",
        object_id=key,
        tensors=tensors,
        locator={"key": key},
        metadata=metadata or {},
    )


def normalize_transfer_ref(value: TransferRef | Mapping[str, Any] | None) -> TransferRef | None:
    """Normalize Ray-serialized dictionaries while leaving TransferRef values intact."""
    if value is None or isinstance(value, TransferRef):
        return value
    if isinstance(value, Mapping):
        return TransferRef.from_dict(value)
    raise TransferProtocolError(f"invalid transfer reference: {value!r}")


@dataclass
class InferenceInput:
    """Input entry waiting to be sent to inference.

    For Eagle3 distillation training, input_ids and packed_loss_mask are provided
    from batch preprocessing. For other training modes, prompt is used.
    """

    data_id: str
    prompt: str | list[dict[str, str]] | None = None
    input_ids: torch.Tensor | None = None
    packed_loss_mask: str | None = None
    formatted_prompt: str | None = None
    metadata: dict = field(default_factory=dict)
    multimodal_inputs: dict = None


@dataclass
class InferenceOutput:
    """Output from inference with a backend-neutral tensor reference.

    ``mooncake_key`` and the shape/dtype mappings remain populated for callers
    that have not migrated yet.
    """

    data_id: str
    mooncake_key: str | None = None
    tensor_shapes: dict[str, tuple[int, ...]] | None = None
    tensor_dtypes: dict[str, torch.dtype | str] | None = None
    packed_loss_mask: str | None = None
    metadata: dict = field(default_factory=dict)
    transfer_ref: TransferRef | Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        self.transfer_ref = normalize_transfer_ref(self.transfer_ref)
        if self.transfer_ref is None:
            if not self.mooncake_key:
                raise ValueError("inference output requires transfer_ref or mooncake_key")
            self.transfer_ref = legacy_mooncake_transfer_ref(
                self.mooncake_key,
                self.tensor_shapes,
                self.tensor_dtypes,
                metadata=self.metadata,
            )

        if self.mooncake_key is None:
            self.mooncake_key = str(
                self.transfer_ref.locator.get("key", self.transfer_ref.object_id)
            )
        if self.tensor_shapes is None:
            self.tensor_shapes = {
                name: spec.shape for name, spec in self.transfer_ref.tensors.items()
            }
        if self.tensor_dtypes is None:
            self.tensor_dtypes = {
                name: getattr(torch, spec.dtype) for name, spec in self.transfer_ref.tensors.items()
            }

    @property
    def transfer_identity(self) -> tuple[str, str]:
        assert isinstance(self.transfer_ref, TransferRef)
        return self.transfer_ref.backend, self.transfer_ref.object_id
