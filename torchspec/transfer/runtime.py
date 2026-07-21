"""Lazy runtime holder for a configured transfer backend."""

from __future__ import annotations

from typing import Any, Callable

from torchspec.transfer.factory import create_transfer_backend


class TransferRuntime:
    """Construct a backend only when it is first needed."""

    def __init__(
        self,
        config: Any,
        factory: Callable[[Any], Any] = create_transfer_backend,
    ) -> None:
        self._config = config
        self._factory = factory
        self._backend: Any | None = None

    @property
    def initialized(self) -> bool:
        return self._backend is not None

    @property
    def backend(self) -> Any:
        if self._backend is None:
            self._backend = self._factory(self._config)
        return self._backend

    def close(self) -> None:
        if self._backend is not None:
            self._backend.close()
            self._backend = None
