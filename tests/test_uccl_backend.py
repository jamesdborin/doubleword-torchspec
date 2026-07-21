from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from torchspec.config.train_config import Config
from torchspec.transfer import (
    TensorSpec,
    TransferProtocolError,
    TransferRef,
    TransferRole,
    TransferSetupError,
    TransferTimeoutError,
    TransferUnavailableError,
    create_transfer_backend,
)
from torchspec.transfer.factory import create_transfer_backend_from_env
from torchspec.transfer.uccl import UCCL_MINIMUM_COMMIT, UcclConfig, UcclP2PBackend
from torchspec.transfer.uccl.backend import _default_host_allocator


class _FakeEndpoint:
    instances = []
    poll_done = True

    def __init__(self, local_gpu_idx=0):
        self.local_gpu_idx = local_gpu_idx
        self.deregistered = []
        self.removed = []
        self.closed = 0
        self.calls = []
        self._next_mr = 1
        type(self).instances.append(self)

    def start_passive_accept(self):
        self.calls.append("start_passive_accept")
        return True

    def get_metadata(self):
        return b"producer-endpoint"

    @staticmethod
    def parse_metadata(metadata):
        assert metadata == b"producer-endpoint"
        return "10.1.2.3", 17777, "0000:01:00.0"

    def connect(self, ip, gpu_bdf, port=-1):
        self.calls.append(("connect", ip, gpu_bdf, port))
        return True, 42

    def add_remote_endpoint(self, metadata):
        raise AssertionError("CXI path must use parse_metadata + connect")

    def regv(self, ptrs, sizes):
        self.calls.append(("regv", list(ptrs), list(sizes)))
        ids = list(range(self._next_mr, self._next_mr + len(ptrs)))
        self._next_mr += len(ptrs)
        return True, ids

    def advertisev(self, mr_ids, ptrs, sizes, num_iovs):
        self.calls.append(("advertisev", list(mr_ids), num_iovs))
        return True, [f"advertisement-{mr_id}".encode() for mr_id in mr_ids]

    def readv_async(self, conn_id, mr_ids, ptrs, sizes, blobs, num_iovs):
        self.calls.append(("readv_async", conn_id, list(blobs), num_iovs))
        return True, 99

    def poll_async(self, transfer_id):
        self.calls.append(("poll_async", transfer_id))
        return True, type(self).poll_done

    def dereg(self, mr_id):
        self.deregistered.append(mr_id)
        return True

    def remove_remote_endpoint(self, conn_id):
        self.removed.append(conn_id)
        return True

    def close(self):
        self.closed += 1


@pytest.fixture(autouse=True)
def _reset_endpoint(monkeypatch):
    _FakeEndpoint.instances = []
    _FakeEndpoint.poll_done = True
    for name in (
        "TORCHSPEC_TRANSFER_BACKEND",
        "UCCL_P2P_LOCAL_GPU_IDX",
        "UCCL_P2P_TRANSPORT",
        "UCCL_P2P_USE_GPU_DIRECT",
        "UCCL_P2P_TIMEOUT_SECONDS",
        "UCCL_P2P_POLL_INTERVAL_SECONDS",
        "UCCL_P2P_RETENTION_TTL_SECONDS",
        "UCCL_P2P_ACK_BIND_HOST",
        "UCCL_P2P_ACK_ADVERTISE_HOST",
    ):
        monkeypatch.delenv(name, raising=False)


_P2P = SimpleNamespace(Endpoint=_FakeEndpoint)
_ACK_SERVERS = {}
_HOST_ALLOCATIONS = []


class _FakeAckServer:
    def __init__(self, registry, bind_host, advertise_host):
        self.registry = registry
        self.address = f"inproc://{advertise_host}"
        self.closed = 0
        _ACK_SERVERS[self.address] = self

    def close(self):
        self.closed += 1


def _send_ack(address, token, timeout):
    del timeout
    return _ACK_SERVERS[address].registry.ack(token)


def _host_allocator(shape, dtype):
    tensor = torch.empty(shape, dtype=dtype, device="cpu")
    _HOST_ALLOCATIONS.append(tensor)
    return tensor


def _backend(**kwargs):
    return UcclP2PBackend(
        UcclConfig(timeout_seconds=0.1, poll_interval_seconds=0),
        p2p_module=_P2P,
        ack_server_factory=_FakeAckServer,
        ack_sender=_send_ack,
        host_allocator=_host_allocator,
        **kwargs,
    )


def test_successful_fetch_uses_low_level_cxi_path_and_ack_releases_producer():
    _HOST_ALLOCATIONS.clear()
    producer = _backend()
    producer.setup(TransferRole.PRODUCER, "cuda:0")
    ref = producer.put(
        "batch-7",
        {
            "hidden": torch.arange(8, dtype=torch.float32).reshape(2, 4),
            "tokens": torch.arange(2, dtype=torch.int64),
        },
    )
    producer_endpoint = _FakeEndpoint.instances[0]

    assert ref.backend == "uccl-p2p"
    assert ref.locator["tensor_order"] == ["hidden", "tokens"]
    assert (
        ref.locator["minimum_commit"]
        == UCCL_MINIMUM_COMMIT
        == "51d91bceabf27be88fc9198cf79b6d4e702bed73"
    )
    assert len(producer._registry) == 1
    assert len(_HOST_ALLOCATIONS) == 2
    assert all(tensor.device.type == "cpu" for tensor in _HOST_ALLOCATIONS)

    consumer = _backend()
    consumer.setup(TransferRole.CONSUMER, "cuda:0")
    result = consumer.get(ref, device="cpu")
    consumer_endpoint = _FakeEndpoint.instances[1]

    assert result["hidden"].shape == (2, 4)
    assert result["hidden"].dtype is torch.float32
    assert result["tokens"].shape == (2,)
    assert len(_HOST_ALLOCATIONS) == 4
    assert ("connect", "10.1.2.3", "0000:01:00.0", 17777) in consumer_endpoint.calls
    assert any(call[0] == "readv_async" for call in consumer_endpoint.calls)
    assert consumer_endpoint.removed == [42]
    assert consumer_endpoint.deregistered == [1, 2]
    assert producer_endpoint.deregistered == []

    consumer.release(ref)
    assert len(producer._registry) == 0
    assert producer_endpoint.deregistered == [1, 2]


def test_fetch_timeout_deregisters_and_does_not_queue_ack():
    producer = _backend()
    producer.setup(TransferRole.PRODUCER)
    ref = producer.put("slow", {"x": torch.ones(2)})
    _FakeEndpoint.poll_done = False

    now = [0.0]

    def clock():
        now[0] += 0.06
        return now[0]

    consumer = _backend(clock=clock)
    consumer.setup(TransferRole.CONSUMER)

    with pytest.raises(TransferTimeoutError, match="timed out"):
        consumer.get(ref, device="cpu")

    endpoint = _FakeEndpoint.instances[1]
    assert endpoint.deregistered == [1]
    assert endpoint.removed == [42]
    assert consumer._pending_acks == {}


def test_malformed_ref_is_rejected_before_connect():
    consumer = _backend()
    consumer.setup(TransferRole.CONSUMER)
    ref = TransferRef(
        backend="uccl-p2p",
        object_id="bad",
        tensors={"x": TensorSpec((2,), "float32")},
        locator={"endpoint_metadata": "not-base64"},
    )

    with pytest.raises(TransferProtocolError, match="locator"):
        consumer.get(ref, device="cpu")

    assert not any(call[0] == "connect" for call in _FakeEndpoint.instances[0].calls)


def test_producer_release_and_close_are_idempotent():
    producer = _backend()
    producer.setup(TransferRole.PRODUCER)
    ref = producer.put("unused", {"x": torch.ones(2)})
    endpoint = _FakeEndpoint.instances[0]
    ack_server = producer._ack_server

    producer.release(ref)
    producer.release(ref)
    assert endpoint.deregistered == [1]

    producer.close()
    producer.close()
    assert endpoint.closed == 1
    assert ack_server.closed == 1


def test_factory_selects_uccl_without_importing_optional_binding():
    config = Config(uccl={"local_gpu_idx": 2, "transport": "cxi"})
    config.transfer.backend = "uccl-p2p"
    config.transfer.options = {"timeout_seconds": 12.0}
    sentinel = object()

    with patch("torchspec.transfer.uccl.UcclP2PBackend", return_value=sentinel) as backend_cls:
        result = create_transfer_backend(config)

    assert result is sentinel
    uccl_config = backend_cls.call_args.args[0]
    assert uccl_config.local_gpu_idx == 2
    assert uccl_config.transport == "cxi"
    assert uccl_config.timeout_seconds == 12.0


def test_factory_accepts_flat_args_and_direct_uccl_config():
    args = Namespace(
        transfer_backend="uccl-p2p",
        transfer_options={},
        uccl_local_gpu_idx=3,
        uccl_transport="cxi",
        uccl_timeout_seconds=9.0,
    )
    with patch("torchspec.transfer.uccl.UcclP2PBackend") as backend_cls:
        create_transfer_backend(args)
    flat_config = backend_cls.call_args.args[0]
    assert flat_config.local_gpu_idx == 3
    assert flat_config.transport == "cxi"
    assert flat_config.timeout_seconds == 9.0

    direct_config = UcclConfig(local_gpu_idx=1)
    with patch("torchspec.transfer.uccl.UcclP2PBackend") as backend_cls:
        create_transfer_backend(direct_config)
    assert backend_cls.call_args.args[0] is direct_config


def test_uccl_config_flat_args_and_environment_round_trip(monkeypatch):
    args = Namespace(
        uccl_local_gpu_idx=2,
        uccl_transport="cxi",
        uccl_use_gpu_direct=False,
        uccl_timeout_seconds=8.0,
        uccl_poll_interval_seconds=0.02,
        uccl_retention_ttl_seconds=44.0,
        uccl_ack_bind_host="*",
        uccl_ack_advertise_host="host-a",
    )
    config = UcclConfig.from_flat_args(args)

    monkeypatch.setenv("UCCL_P2P_TIMEOUT_SECONDS", "99.0")
    config.export_env()

    assert config.local_gpu_idx == 2
    assert config.ack_advertise_host == "host-a"
    assert UcclConfig.from_env().timeout_seconds == 99.0
    assert UcclConfig.from_env().transport == "cxi"
    assert UcclConfig.from_env().local_gpu_idx == 2
    assert config.transport == "cxi"
    assert config.use_gpu_direct is False
    assert config.timeout_seconds == 8.0
    assert config.poll_interval_seconds == 0.02
    assert config.retention_ttl_seconds == 44.0
    assert config.ack_bind_host == "*"
    assert config.ack_advertise_host == "host-a"
    assert __import__("os").environ["TORCHSPEC_TRANSFER_BACKEND"] == "uccl-p2p"


def test_create_backend_from_env_sets_up_selected_backend(monkeypatch):
    monkeypatch.setenv("TORCHSPEC_TRANSFER_BACKEND", "uccl-p2p")
    setup_calls = []
    fake_backend = SimpleNamespace(setup=lambda role, device: setup_calls.append((role, device)))

    with patch("torchspec.transfer.factory.create_transfer_backend", return_value=fake_backend):
        result = create_transfer_backend_from_env("cpu", TransferRole.CONSUMER)

    assert result is fake_backend
    assert setup_calls == [(TransferRole.CONSUMER, "cpu")]


def test_optional_uccl_import_is_deferred_until_setup():
    backend = UcclP2PBackend()

    with patch("torchspec.transfer.uccl.backend.importlib.import_module", side_effect=ImportError):
        with pytest.raises(TransferUnavailableError, match=UCCL_MINIMUM_COMMIT):
            backend.setup(TransferRole.CONSUMER)


def test_default_host_allocator_requests_pinned_cpu_memory():
    sentinel = object()
    with patch("torchspec.transfer.uccl.backend.torch.empty", return_value=sentinel) as empty:
        result = _default_host_allocator((2, 3), torch.float32)

    assert result is sentinel
    empty.assert_called_once_with((2, 3), dtype=torch.float32, device="cpu", pin_memory=True)


def test_gpu_direct_rejects_cpu_registration_device():
    backend = UcclP2PBackend(UcclConfig(use_gpu_direct=True), p2p_module=_P2P)

    with pytest.raises(TransferSetupError, match="CUDA"):
        backend.setup(TransferRole.CONSUMER, "cpu")


def test_setup_uses_runtime_cuda_device_and_restores_current_device(monkeypatch):
    current_device = [1]

    def set_device(device):
        current_device[0] = int(device)

    class _DeviceChangingEndpoint(_FakeEndpoint):
        def __init__(self, local_gpu_idx=0):
            super().__init__(local_gpu_idx)
            set_device(0)

    config = UcclConfig(local_gpu_idx=0)
    backend = UcclP2PBackend(
        config,
        p2p_module=SimpleNamespace(Endpoint=_DeviceChangingEndpoint),
    )
    backend_globals = backend._setup.__func__.__globals__
    monkeypatch.setitem(
        backend_globals,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                current_device=lambda: current_device[0],
                set_device=set_device,
            )
        ),
    )

    backend.setup(TransferRole.CONSUMER, 1)

    assert backend._endpoint.local_gpu_idx == 1
    assert current_device[0] == 1
    assert config.local_gpu_idx == 0
