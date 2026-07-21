from types import SimpleNamespace
from unittest.mock import patch

import pytest
import ray.cloudpickle as cloudpickle
from omegaconf import OmegaConf

from torchspec.config.train_config import Config, config_to_flat_args, load_config
from torchspec.controller.setup import build_transfer_config
from torchspec.transfer import (
    TensorSpec,
    TransferBackend,
    TransferClosedError,
    TransferRef,
    TransferRole,
    TransferRuntime,
    TransferState,
    TransferStateError,
    create_transfer_backend,
)


def test_transfer_ref_round_trip_is_immutable():
    shard = TransferRef(
        backend="mooncake",
        object_id="batch-1/rank-0",
        tensors={"hidden": TensorSpec((2, 3), "torch.float16")},
        locator={"key": "rank-0"},
    )
    ref = TransferRef(
        backend="mooncake",
        object_id="batch-1",
        tensors={"tokens": TensorSpec((2,), "int64")},
        locator={"namespace": "training"},
        shards=(shard,),
        metadata={"step": 4},
    )

    restored = TransferRef.from_dict(ref.to_dict())
    cloudpickled = cloudpickle.loads(cloudpickle.dumps(ref))

    assert restored == ref
    assert restored.to_dict() == ref.to_dict()
    assert cloudpickled == ref
    assert cloudpickled.to_dict() == ref.to_dict()
    with pytest.raises(TypeError):
        restored.metadata["step"] = 5


def test_transfer_ref_nbytes_includes_shards():
    ref = TransferRef(
        backend="uccl-p2p",
        object_id="bundle",
        tensors={"tokens": TensorSpec((2, 4), "int64")},
        shards=(
            TransferRef(
                backend="uccl-p2p",
                object_id="bundle/0",
                tensors={"hidden": TensorSpec((2, 4, 8), "bfloat16")},
            ),
        ),
    )

    assert ref.nbytes == 8 * 8 + 2 * 4 * 8 * 2


class _MemoryBackend(TransferBackend):
    def _setup(self, role, device):
        self.items = {}

    def _put(self, object_id, tensors, expected_consumers):
        self.items[object_id] = tensors
        return TransferRef(
            backend="memory",
            object_id=object_id,
            tensors={
                name: TensorSpec(tuple(tensor.shape), str(tensor.dtype))
                for name, tensor in tensors.items()
            },
        )

    def _flush(self):
        pass

    def _get(self, ref, device):
        return self.items[ref.object_id]

    def _release(self, ref):
        self.items.pop(ref.object_id, None)

    def _health_check(self):
        pass

    def _close(self):
        pass


def test_backend_lifecycle_and_role_validation():
    backend = _MemoryBackend()
    assert backend.state is TransferState.NEW
    with pytest.raises(TransferStateError):
        backend.flush()

    backend.setup(TransferRole.PRODUCER, "cpu")
    assert backend.state is TransferState.READY
    tensor = SimpleNamespace(shape=(3,), dtype="float32")
    ref = backend.put("sample", {"x": tensor})
    assert ref.object_id == "sample"
    with pytest.raises(TransferStateError, match="consumer"):
        backend.get(ref)

    backend.close()
    assert backend.state is TransferState.CLOSED
    backend.close()
    with pytest.raises(TransferClosedError):
        backend.health_check()


def test_mooncake_factory_is_default_and_merges_legacy_options():
    config = Config(mooncake={"protocol": "tcp"})
    config.transfer.options = {"device_name": "cxi0", "global_segment_size": 1024}
    sentinel = object()

    with (
        patch("torchspec.config.mooncake_config.MooncakeConfig") as config_cls,
        patch("torchspec.transfer.mooncake.EagleMooncakeStore", return_value=sentinel) as store_cls,
    ):
        result = create_transfer_backend(config)

    assert result.store is sentinel
    config_cls.assert_called_once_with(protocol="tcp", device_name="cxi0", global_segment_size=1024)
    store_cls.assert_called_once_with(config_cls.return_value)


def test_transfer_and_uccl_config_fields_flatten_with_prefixes():
    config = OmegaConf.structured(Config)
    config.transfer.backend = "uccl-p2p"
    config.uccl.transport = "cxi"

    args = config_to_flat_args(config)

    assert args.transfer_backend == "uccl-p2p"
    assert args.transfer_options == {}
    assert args.uccl_transport == "cxi"

    selected = build_transfer_config(args)
    assert selected.transport == "cxi"


def test_config_rejects_unsupported_uccl_combinations():
    with pytest.raises(NotImplementedError, match="vLLM"):
        load_config(
            base_config=OmegaConf.create(
                {"model": {"target_model_backend": "vllm"}, "transfer": {"backend": "uccl-p2p"}}
            )
        )

    with pytest.raises(NotImplementedError, match="USP"):
        load_config(
            base_config=OmegaConf.create(
                {
                    "training": {"attention_backend": "usp"},
                    "transfer": {"backend": "uccl-p2p"},
                }
            )
        )


def test_transfer_runtime_constructs_lazily_and_closes():
    backend = _MemoryBackend()
    factory_calls = []
    runtime = TransferRuntime({}, lambda config: factory_calls.append(config) or backend)

    assert not runtime.initialized
    assert factory_calls == []
    assert runtime.backend is backend
    assert runtime.backend is backend
    assert factory_calls == [{}]
    runtime.close()
    assert backend.state is TransferState.CLOSED
    assert not runtime.initialized
