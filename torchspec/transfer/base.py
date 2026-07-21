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

"""Backend-neutral transfer contracts and value types."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, Mapping, TypeAlias, TypeVar

import torch

TensorBundle: TypeAlias = Mapping[str, torch.Tensor]


class TransferError(RuntimeError):
    """Base exception for transfer backend failures."""


class TransferSetupError(TransferError):
    """A backend could not be initialized."""


class TransferUnavailableError(TransferError):
    """The requested transfer backend or service is unavailable."""


class TransferNotFoundError(TransferError):
    """A referenced transfer object no longer exists."""


class TransferTimeoutError(TransferError):
    """A transfer operation exceeded its deadline."""


class TransferCapacityError(TransferError):
    """A backend lacks capacity for the requested transfer."""


class TransferProtocolError(TransferError):
    """A backend returned malformed or incompatible transfer data."""


class TransferClosedError(TransferError):
    """An operation was attempted on a closed backend."""


class TransferStateError(TransferError):
    """An operation is invalid for the backend's current lifecycle state."""


class TransferRole(str, Enum):
    PRODUCER = "producer"
    CONSUMER = "consumer"


class TransferState(str, Enum):
    NEW = "new"
    READY = "ready"
    CLOSED = "closed"


_DTYPE_NBYTES = {
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
    "int16": 2,
    "uint16": 2,
    "float16": 2,
    "half": 2,
    "bfloat16": 2,
    "int32": 4,
    "uint32": 4,
    "float32": 4,
    "float": 4,
    "complex32": 4,
    "int64": 8,
    "uint64": 8,
    "float64": 8,
    "double": 8,
    "complex64": 8,
    "complex128": 16,
}


@dataclass(frozen=True)
class TensorSpec:
    """Serializable tensor shape and dtype metadata."""

    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        shape = tuple(self.shape)
        if any(not isinstance(dim, int) or isinstance(dim, bool) or dim < 0 for dim in shape):
            raise ValueError(f"tensor shape must contain non-negative integers, got {shape!r}")
        dtype = str(self.dtype).removeprefix("torch.")
        if not dtype:
            raise ValueError("tensor dtype must be a non-empty string")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "dtype", dtype)

    @property
    def nbytes(self) -> int:
        try:
            itemsize = _DTYPE_NBYTES[self.dtype]
        except KeyError as exc:
            raise TransferProtocolError(f"unsupported tensor dtype: {self.dtype!r}") from exc
        return math.prod(self.shape) * itemsize

    def to_dict(self) -> dict[str, Any]:
        return {"shape": list(self.shape), "dtype": self.dtype}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TensorSpec:
        try:
            return cls(shape=tuple(value["shape"]), dtype=value["dtype"])
        except (KeyError, TypeError) as exc:
            raise TransferProtocolError(f"invalid tensor specification: {value!r}") from exc


_ValueT = TypeVar("_ValueT")


class _FrozenMapping(Mapping[str, _ValueT], Generic[_ValueT]):
    """Small immutable mapping that remains pickle-friendly for Ray transport."""

    def __init__(self, value: Mapping[str, _ValueT]) -> None:
        self._data = dict(value)

    def __getitem__(self, key: str) -> _ValueT:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __reduce__(self) -> tuple[Any, tuple[dict[str, _ValueT]]]:
        return type(self), (self._data,)


def _immutable_mapping(value: Mapping[str, _ValueT], field_name: str) -> Mapping[str, _ValueT]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return _FrozenMapping(value)


@dataclass(frozen=True)
class TransferRef:
    """Portable reference to one logical bundle of transferred tensors."""

    backend: str
    object_id: str
    tensors: Mapping[str, TensorSpec]
    locator: Mapping[str, Any] = field(default_factory=dict)
    shards: tuple[TransferRef, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str) or not self.backend:
            raise ValueError("backend must be a non-empty string")
        if not isinstance(self.object_id, str) or not self.object_id:
            raise ValueError("object_id must be a non-empty string")

        tensors = _immutable_mapping(self.tensors, "tensors")
        if any(not isinstance(name, str) or not name for name in tensors):
            raise ValueError("tensor names must be non-empty strings")
        if any(not isinstance(spec, TensorSpec) for spec in tensors.values()):
            raise TypeError("tensor values must be TensorSpec instances")

        shards = tuple(self.shards)
        if any(not isinstance(shard, TransferRef) for shard in shards):
            raise TypeError("shards must contain TransferRef instances")

        object.__setattr__(self, "tensors", tensors)
        object.__setattr__(self, "locator", _immutable_mapping(self.locator, "locator"))
        object.__setattr__(self, "shards", shards)
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata, "metadata"))

    @property
    def nbytes(self) -> int:
        return sum(spec.nbytes for spec in self.tensors.values()) + sum(
            shard.nbytes for shard in self.shards
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "object_id": self.object_id,
            "tensors": {name: spec.to_dict() for name, spec in self.tensors.items()},
            "locator": dict(self.locator),
            "shards": [shard.to_dict() for shard in self.shards],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TransferRef:
        try:
            tensors = {name: TensorSpec.from_dict(spec) for name, spec in value["tensors"].items()}
            return cls(
                backend=value["backend"],
                object_id=value["object_id"],
                tensors=tensors,
                locator=value.get("locator", {}),
                shards=tuple(cls.from_dict(shard) for shard in value.get("shards", ())),
                metadata=value.get("metadata", {}),
            )
        except (AttributeError, KeyError, TypeError) as exc:
            raise TransferProtocolError(f"invalid transfer reference: {value!r}") from exc


class TransferBackend(ABC):
    """Lifecycle-enforcing interface implemented by transfer backends."""

    def __init__(self) -> None:
        self._transfer_state = TransferState.NEW
        self._transfer_role: TransferRole | None = None

    @property
    def state(self) -> TransferState:
        return self._transfer_state

    @property
    def role(self) -> TransferRole | None:
        return self._transfer_role

    def setup(self, role: TransferRole, device: torch.device | str | int | None = None) -> None:
        if not isinstance(role, TransferRole):
            raise TransferSetupError(f"invalid transfer role: {role!r}")
        if self.state is TransferState.CLOSED:
            raise TransferClosedError("a closed transfer backend cannot be set up again")
        if self.state is TransferState.READY:
            if role is self.role:
                return
            raise TransferStateError(
                f"backend is already set up as {self.role.value}; cannot change role to {role.value}"
            )
        try:
            self._setup(role, device)
        except TransferError:
            raise
        except Exception as exc:
            raise TransferSetupError("transfer backend setup failed") from exc
        self._transfer_role = role
        self._transfer_state = TransferState.READY

    def put(
        self,
        object_id: str,
        tensors: TensorBundle,
        expected_consumers: int = 1,
    ) -> TransferRef:
        self._require_ready(TransferRole.PRODUCER)
        if not object_id:
            raise ValueError("object_id must be non-empty")
        if expected_consumers < 1:
            raise ValueError("expected_consumers must be at least one")
        return self._put(object_id, tensors, expected_consumers)

    def flush(self) -> None:
        self._require_ready()
        self._flush()

    def get(self, ref: TransferRef, device: torch.device | str | int | None = None) -> TensorBundle:
        self._require_ready(TransferRole.CONSUMER)
        return self._get(ref, device)

    def release(self, ref: TransferRef) -> None:
        self._require_ready()
        self._release(ref)

    def health_check(self) -> None:
        self._require_ready()
        self._health_check()

    def close(self) -> None:
        if self.state is TransferState.CLOSED:
            return
        try:
            self._close()
        finally:
            self._transfer_state = TransferState.CLOSED

    def _require_ready(self, role: TransferRole | None = None) -> None:
        if self.state is TransferState.CLOSED:
            raise TransferClosedError("transfer backend is closed")
        if self.state is not TransferState.READY:
            raise TransferStateError("transfer backend must be set up before use")
        if role is not None and self.role is not role:
            raise TransferStateError(f"operation requires the {role.value} role")

    @abstractmethod
    def _setup(self, role: TransferRole, device: torch.device | str | int | None) -> None: ...

    @abstractmethod
    def _put(
        self, object_id: str, tensors: TensorBundle, expected_consumers: int
    ) -> TransferRef: ...

    @abstractmethod
    def _flush(self) -> None: ...

    @abstractmethod
    def _get(self, ref: TransferRef, device: torch.device | str | int | None) -> TensorBundle: ...

    @abstractmethod
    def _release(self, ref: TransferRef) -> None: ...

    @abstractmethod
    def _health_check(self) -> None: ...

    @abstractmethod
    def _close(self) -> None: ...
