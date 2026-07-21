from __future__ import annotations

import queue
from unittest.mock import patch

import pytest
import torch

from tools.test_sglang_engine_patch import resolve_stored_tensor_metadata
from torchspec.training.data_fetcher import MooncakeDataFetcher, TrainSample, TransferDataset
from torchspec.transfer.base import TensorSpec, TransferBackend, TransferRef, TransferRole
from torchspec.transfer.mooncake.backend import MooncakeTransferBackend
from torchspec.utils.types import InferenceOutput


class _Queue:
    def __init__(self, *items):
        self.items = queue.Queue()
        for item in items:
            self.items.put(item)

    def get(self, block=True, timeout=None):
        return self.items.get(block=block, timeout=timeout)


class _Backend(TransferBackend):
    def __init__(self, *, fail=False):
        super().__init__()
        self.fail = fail
        self.released = []

    def _setup(self, role, device):
        pass

    def _put(self, object_id, tensors, expected_consumers):
        raise NotImplementedError

    def _flush(self):
        pass

    def _get(self, ref, device):
        if self.fail:
            raise RuntimeError("get failed")
        return {"input_ids": torch.arange(4, dtype=torch.int64)}

    def _release(self, ref):
        self.released.append(ref.object_id)

    def _health_check(self):
        pass

    def _close(self):
        pass


def _ref(object_id="sample"):
    return TransferRef(
        backend="test",
        object_id=object_id,
        tensors={"input_ids": TensorSpec((4,), "int64")},
    )


class _ExistenceStore:
    def __init__(self, *keys):
        self.keys = set(keys)

    def exists(self, key):
        return key in self.keys


def test_sglang_smoke_uses_serialized_transfer_ref_metadata():
    ref = TransferRef(
        backend="mooncake",
        object_id="stored",
        tensors={
            "hidden_states": TensorSpec((9, 32), "bfloat16"),
            "input_ids": TensorSpec((9,), "int64"),
        },
    )
    meta = {
        "prompt_tokens": 9,
        "spec_training_mooncake_store_keys": ["legacy"],
        "spec_training_transfer_refs": [ref.to_dict()],
    }

    key, shapes, dtypes = resolve_stored_tensor_metadata(
        meta, _ExistenceStore("legacy_lhs"), hidden_size=16, num_aux_layers=3
    )

    assert key == "stored"
    assert shapes == {"hidden_states": (9, 32), "input_ids": (9,)}
    assert dtypes == {"hidden_states": torch.bfloat16, "input_ids": torch.int64}


def test_sglang_smoke_falls_back_when_transfer_ref_type_is_unavailable():
    ref = _ref("new-protocol").to_dict()
    meta = {
        "prompt_tokens": 9,
        "spec_training_mooncake_store_keys": ["legacy"],
        "spec_training_transfer_refs": [ref],
    }
    real_import = __import__

    def baseline_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torchspec.transfer" and "TransferRef" in fromlist:
            raise ImportError("baseline has no TransferRef")
        return real_import(name, globals, locals, fromlist, level)

    with patch("builtins.__import__", side_effect=baseline_import):
        key, shapes, dtypes = resolve_stored_tensor_metadata(
            meta,
            _ExistenceStore(),
            hidden_size=16,
            num_aux_layers=3,
        )

    assert key == "legacy"
    assert shapes == {"hidden_states": (9, 48), "input_ids": (9,)}
    assert dtypes == {"hidden_states": torch.bfloat16, "input_ids": torch.int64}


@pytest.mark.parametrize("has_lhs", [False, True])
def test_sglang_smoke_legacy_metadata_probes_optional_lhs(has_lhs):
    store = _ExistenceStore(*(["legacy_lhs"] if has_lhs else []))
    meta = {"prompt_tokens": 9, "spec_training_mooncake_store_keys": ["legacy"]}

    key, shapes, dtypes = resolve_stored_tensor_metadata(
        meta, store, hidden_size=16, num_aux_layers=3
    )

    assert key == "legacy"
    assert shapes["hidden_states"] == (9, 48)
    assert shapes["input_ids"] == (9,)
    assert ("last_hidden_states" in shapes) is has_lhs
    assert ("last_hidden_states" in dtypes) is has_lhs


def test_inference_output_accepts_serialized_ref_and_populates_legacy_fields():
    output = InferenceOutput(data_id="d0", transfer_ref=_ref().to_dict())

    assert output.transfer_ref == _ref()
    assert output.mooncake_key == "sample"
    assert output.tensor_shapes == {"input_ids": (4,)}
    assert output.tensor_dtypes == {"input_ids": torch.int64}


def test_legacy_input_ids_defaults_to_int64():
    output = InferenceOutput(
        data_id="d0",
        mooncake_key="legacy",
        tensor_shapes={"input_ids": (4,), "hidden_states": (4, 2)},
    )

    assert output.transfer_ref.tensors["input_ids"].dtype == "int64"
    assert output.transfer_ref.tensors["hidden_states"].dtype == "bfloat16"
    assert output.transfer_ref.nbytes == 4 * 8 + 4 * 2 * 2


def test_transfer_dataset_releases_only_after_successful_materialization():
    sample = TrainSample(transfer_ref=_ref())
    backend = _Backend()
    dataset = TransferDataset(_Queue(sample, None), backend, torch.device("cpu"))

    result = list(dataset)

    assert result[0]["input_ids"].tolist() == [[0, 1, 2, 3]]
    assert backend.released == ["sample"]

    failing = _Backend(fail=True)
    failing_dataset = TransferDataset(_Queue(sample), failing, torch.device("cpu"))
    with pytest.raises(RuntimeError, match="get failed"):
        failing_dataset._load_from_transfer(sample)
    assert failing.released == []


def test_mooncake_alias_and_adapter_translate_legacy_store_calls():
    class Store:
        def __init__(self):
            self.removed = None

        def get(self, key, shapes, dtypes, device):
            assert key == "legacy"
            return {"input_ids": torch.ones(shapes["input_ids"], dtype=dtypes["input_ids"])}

        def remove_eagle3_tensors(self, key, **kwargs):
            self.removed = (key, kwargs)

    store = Store()
    backend = MooncakeTransferBackend(store)
    backend.setup(TransferRole.CONSUMER, "cpu")
    ref = TransferRef(
        backend="mooncake",
        object_id="legacy",
        locator={"key": "legacy"},
        tensors={"input_ids": TensorSpec((3,), "int64")},
    )

    assert backend.get(ref, "cpu")["input_ids"].tolist() == [1, 1, 1]
    backend.release(ref)
    assert store.removed == (
        "legacy",
        {"has_last_hidden_states": False, "has_target": False},
    )
    assert MooncakeDataFetcher.__name__ == "TransferDataFetcher"
