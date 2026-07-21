"""UCCL-P2P transfer backend using the low-level CXI-compatible API."""

from __future__ import annotations

import base64
import importlib
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from typing import Any

import torch

from torchspec.transfer.base import (
    TensorBundle,
    TensorSpec,
    TransferBackend,
    TransferProtocolError,
    TransferRef,
    TransferRole,
    TransferSetupError,
    TransferTimeoutError,
    TransferUnavailableError,
)
from torchspec.transfer.uccl.config import UCCL_MINIMUM_COMMIT, UcclConfig
from torchspec.transfer.uccl.lifetime import LifetimeRegistry, ZmqAckServer, send_zmq_ack

_BACKEND_NAME = "uccl-p2p"


def _default_host_allocator(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)


def _encode_blob(value: bytes) -> str:
    return base64.b64encode(bytes(value)).decode("ascii")


def _decode_blob(value: Any, name: str) -> bytes:
    if not isinstance(value, str):
        raise TransferProtocolError(f"UCCL {name} must be a base64 string")
    try:
        return base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as exc:
        raise TransferProtocolError(f"UCCL {name} is not valid base64") from exc


def _require_success(result: Any, operation: str) -> Any:
    if not isinstance(result, tuple) or len(result) != 2 or not result[0]:
        raise TransferProtocolError(f"UCCL {operation} failed: {result!r}")
    return result[1]


def _resolve_cuda_device_index(
    device: torch.device | str | int | None, fallback_index: int
) -> int:
    if isinstance(device, int):
        return device
    if device is None:
        return fallback_index
    if isinstance(device, str):
        if device == "cuda":
            return torch.cuda.current_device()
        if device.startswith("cuda:"):
            return int(device.removeprefix("cuda:"))
        return fallback_index
    if device.type != "cuda":
        return fallback_index
    return torch.cuda.current_device() if device.index is None else device.index


class UcclP2PBackend(TransferBackend):
    """Tensor transfer over UCCL's explicit registration/read API.

    The high-level ``register_memory``/``transfer`` wrapper is intentionally not
    used because it does not support the CXI path. This adapter requires UCCL at
    or after commit ``51d91bceabf27be88fc9198cf79b6d4e702bed73``.
    """

    def __init__(
        self,
        config: UcclConfig | None = None,
        *,
        p2p_module: Any | None = None,
        ack_server_factory: Callable[..., Any] = ZmqAckServer,
        ack_sender: Callable[[str, str, float], bool] = send_zmq_ack,
        host_allocator: Callable[[tuple[int, ...], torch.dtype], torch.Tensor] = (
            _default_host_allocator
        ),
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        super().__init__()
        self.config = config or UcclConfig()
        self._p2p = p2p_module
        self._ack_server_factory = ack_server_factory
        self._ack_sender = ack_sender
        self._host_allocator = host_allocator
        self._clock = clock
        self._sleeper = sleeper
        self._endpoint: Any | None = None
        self._endpoint_lock = threading.Lock()
        self._registry: LifetimeRegistry | None = None
        self._ack_server: Any | None = None
        self._pending_acks: dict[str, tuple[str, str]] = {}
        self._device: torch.device | None = None

    def _setup(self, role: TransferRole, device: torch.device | str | int | None) -> None:
        endpoint_gpu_idx = _resolve_cuda_device_index(device, self.config.local_gpu_idx)
        if self.config.use_gpu_direct:
            if (
                isinstance(device, str)
                and not device.startswith("cuda")
                or device is not None
                and not isinstance(device, (str, int))
                and device.type != "cuda"
            ):
                raise TransferSetupError("UCCL GPU-direct mode requires a CUDA device")
            self._device = torch.device(f"cuda:{endpoint_gpu_idx}")
        self.config.export_env()
        if self._p2p is None:
            try:
                self._p2p = importlib.import_module("uccl.p2p")
            except ImportError as exc:
                raise TransferUnavailableError(
                    "UCCL-P2P is unavailable; install uccl built from commit "
                    f"{UCCL_MINIMUM_COMMIT} or newer"
                ) from exc

        try:
            previous_device = torch.cuda.current_device()
        except (AssertionError, RuntimeError):
            previous_device = None
        try:
            try:
                self._endpoint = self._p2p.Endpoint(endpoint_gpu_idx)
            except TypeError:
                self._endpoint = self._p2p.Endpoint()
        finally:
            # UCCL's Endpoint constructor calls cudaSetDevice().  Setup must not
            # silently change the trainer's current device after its model has
            # already been placed on a different GPU.
            if previous_device is not None:
                torch.cuda.set_device(previous_device)

        if role is TransferRole.PRODUCER:
            if not self._endpoint.start_passive_accept():
                raise TransferProtocolError("UCCL start_passive_accept failed")
            metadata = bytes(self._endpoint.get_metadata())
            advertise_host, _, _ = self._p2p.Endpoint.parse_metadata(metadata)
            advertise_host = self.config.ack_advertise_host or advertise_host
            self._registry = LifetimeRegistry(clock=self._clock)
            self._ack_server = self._ack_server_factory(
                self._registry,
                self.config.ack_bind_host,
                advertise_host,
            )

    def _put(self, object_id: str, tensors: TensorBundle, expected_consumers: int) -> TransferRef:
        if not tensors:
            raise ValueError("UCCL put requires at least one tensor")
        assert self._endpoint is not None
        assert self._registry is not None
        assert self._ack_server is not None

        names = tuple(tensors)
        if self.config.use_gpu_direct:
            assert self._device is not None
            staged = tuple(
                tensors[name].detach().to(device=self._device).contiguous() for name in names
            )
        else:
            host_tensors = []
            for name in names:
                source = tensors[name].detach()
                host = self._host_allocator(tuple(source.shape), source.dtype)
                host.copy_(source, non_blocking=False)
                host_tensors.append(host)
            staged = tuple(host_tensors)
        ptrs = [tensor.data_ptr() for tensor in staged]
        sizes = [tensor.numel() * tensor.element_size() for tensor in staged]

        with self._endpoint_lock:
            mr_ids = list(_require_success(self._endpoint.regv(ptrs, sizes), "regv"))
            try:
                advertisements = list(
                    _require_success(
                        self._endpoint.advertisev(mr_ids, ptrs, sizes, len(staged)),
                        "advertisev",
                    )
                )
            except Exception:
                for mr_id in mr_ids:
                    self._endpoint.dereg(mr_id)
                raise

        if len(mr_ids) != len(staged) or len(advertisements) != len(staged):
            with self._endpoint_lock:
                for mr_id in mr_ids:
                    self._endpoint.dereg(mr_id)
            raise TransferProtocolError("UCCL returned an unexpected registration vector length")

        token = uuid.uuid4().hex

        def release_registration() -> None:
            with self._endpoint_lock:
                for mr_id in mr_ids:
                    self._endpoint.dereg(mr_id)

        self._registry.register(
            token,
            staged,
            release_registration,
            expected_consumers,
            self.config.retention_ttl_seconds,
        )
        metadata = bytes(self._endpoint.get_metadata())
        return TransferRef(
            backend=_BACKEND_NAME,
            object_id=object_id,
            tensors={
                name: TensorSpec(tuple(tensor.shape), str(tensor.dtype))
                for name, tensor in zip(names, staged, strict=True)
            },
            locator={
                "endpoint_metadata": _encode_blob(metadata),
                "advertisements": [_encode_blob(blob) for blob in advertisements],
                "tensor_order": list(names),
                "ack_address": self._ack_server.address,
                "ack_token": token,
                "minimum_commit": UCCL_MINIMUM_COMMIT,
            },
            metadata={"expected_consumers": expected_consumers},
        )

    def _flush(self) -> None:
        if self._registry is not None:
            self._registry.cleanup_expired()

    def _get(self, ref: TransferRef, device: torch.device | str | int | None) -> TensorBundle:
        endpoint_metadata, advertisements, names, ack_address, ack_token = self._parse_ref(ref)
        assert self._endpoint is not None

        transport_buffers: dict[str, torch.Tensor] = {}
        for name in names:
            spec = ref.tensors[name]
            dtype = getattr(torch, spec.dtype, None)
            if not isinstance(dtype, torch.dtype):
                raise TransferProtocolError(f"unsupported torch dtype in UCCL ref: {spec.dtype!r}")
            if self.config.use_gpu_direct:
                assert self._device is not None
                transport_buffers[name] = torch.empty(spec.shape, dtype=dtype, device=self._device)
            else:
                transport_buffers[name] = self._host_allocator(spec.shape, dtype)

        ordered = [transport_buffers[name] for name in names]
        ptrs = [tensor.data_ptr() for tensor in ordered]
        sizes = [tensor.numel() * tensor.element_size() for tensor in ordered]
        mr_ids: list[Any] = []
        conn_id: Any | None = None
        try:
            try:
                remote_ip, remote_port, remote_gpu_bdf = self._p2p.Endpoint.parse_metadata(
                    endpoint_metadata
                )
            except Exception as exc:
                raise TransferProtocolError("UCCL endpoint metadata could not be parsed") from exc
            with self._endpoint_lock:
                conn_id = _require_success(
                    self._endpoint.connect(remote_ip, remote_gpu_bdf, remote_port),
                    "connect",
                )
                mr_ids = list(_require_success(self._endpoint.regv(ptrs, sizes), "regv"))
                if len(mr_ids) != len(ordered):
                    raise TransferProtocolError(
                        "UCCL returned an unexpected registration vector length"
                    )
                transfer_id = _require_success(
                    self._endpoint.readv_async(
                        conn_id,
                        mr_ids,
                        ptrs,
                        sizes,
                        advertisements,
                        len(ordered),
                    ),
                    "readv_async",
                )
            deadline = self._clock() + self.config.timeout_seconds
            while True:
                with self._endpoint_lock:
                    done = _require_success(self._endpoint.poll_async(transfer_id), "poll_async")
                if done:
                    break
                if self._clock() >= deadline:
                    raise TransferTimeoutError(
                        f"UCCL transfer {ref.object_id!r} timed out after "
                        f"{self.config.timeout_seconds}s"
                    )
                self._sleeper(self.config.poll_interval_seconds)
        finally:
            with self._endpoint_lock:
                for mr_id in mr_ids:
                    self._endpoint.dereg(mr_id)
                if conn_id is not None:
                    self._endpoint.remove_remote_endpoint(conn_id)

        self._pending_acks[ack_token] = (ack_address, ack_token)
        if self.config.use_gpu_direct:
            return transport_buffers
        target_device = torch.device(device) if device is not None else torch.device("cpu")
        if target_device.type == "cpu":
            return transport_buffers
        return {
            name: tensor.to(device=target_device, non_blocking=False)
            for name, tensor in transport_buffers.items()
        }

    def _release(self, ref: TransferRef) -> None:
        if self.role is TransferRole.PRODUCER:
            token = ref.locator.get("ack_token")
            if isinstance(token, str) and self._registry is not None:
                self._registry.release(token)
            return

        token = ref.locator.get("ack_token")
        pending = self._pending_acks.pop(token, None) if isinstance(token, str) else None
        if pending is not None:
            address, ack_token = pending
            if not self._ack_sender(address, ack_token, self.config.timeout_seconds):
                raise TransferProtocolError("producer rejected the UCCL transfer acknowledgement")

    def _health_check(self) -> None:
        if self._endpoint is None:
            raise TransferUnavailableError("UCCL endpoint is unavailable")
        if self._registry is not None:
            self._registry.cleanup_expired()

    def _close(self) -> None:
        if self.role is TransferRole.CONSUMER:
            for address, token in list(self._pending_acks.values()):
                try:
                    self._ack_sender(address, token, self.config.timeout_seconds)
                except Exception:
                    pass
            self._pending_acks.clear()
        if self._ack_server is not None:
            self._ack_server.close()
            self._ack_server = None
        if self._registry is not None:
            self._registry.close()
            self._registry = None
        endpoint_close = getattr(self._endpoint, "close", None)
        if callable(endpoint_close):
            endpoint_close()
        self._endpoint = None
        self._device = None

    @staticmethod
    def _parse_ref(ref: TransferRef) -> tuple[bytes, list[bytes], list[str], str, str]:
        if ref.backend != _BACKEND_NAME:
            raise TransferProtocolError(
                f"UCCL backend cannot read a {ref.backend!r} transfer reference"
            )
        locator: Mapping[str, Any] = ref.locator
        try:
            names = list(locator["tensor_order"])
            encoded_advertisements = list(locator["advertisements"])
            ack_address = locator["ack_address"]
            ack_token = locator["ack_token"]
        except (KeyError, TypeError) as exc:
            raise TransferProtocolError("UCCL transfer reference locator is incomplete") from exc
        if (
            not names
            or any(not isinstance(name, str) for name in names)
            or len(set(names)) != len(names)
            or set(names) != set(ref.tensors)
            or len(encoded_advertisements) != len(names)
            or not isinstance(ack_address, str)
            or not ack_address
            or not isinstance(ack_token, str)
            or not ack_token
        ):
            raise TransferProtocolError("UCCL transfer reference locator is malformed")
        endpoint_metadata = _decode_blob(locator.get("endpoint_metadata"), "endpoint metadata")
        advertisements = [
            _decode_blob(blob, f"advertisement {index}")
            for index, blob in enumerate(encoded_advertisements)
        ]
        return endpoint_metadata, advertisements, names, ack_address, ack_token
