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

"""Queue-based data fetching with mooncake store.
Data flow:
  TrainActor -> MooncakeDataFetcher -> MooncakeDataset -> MooncakeStore -> Collator
                     |                      |                  |               |
                iter(fetcher)          queue.get()      store.get(key)     pad & batch
"""

import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from ray.util.queue import Queue as RayQueue
from torch.utils.data import DataLoader, IterableDataset

from torchspec.data.utils import deserialize_packed_loss_mask, resolve_loss_mask, unpack_loss_mask
from torchspec.transfer.base import (
    TensorSpec,
    TransferBackend,
    TransferProtocolError,
    TransferRef,
    TransferRole,
)
from torchspec.transfer.mooncake.backend import MooncakeTransferBackend
from torchspec.utils.distributed import (
    get_draft_sp_group,
    get_sp_ring_group,
    get_usp_rank_coords,
)
from torchspec.utils.logging import logger
from torchspec.utils.types import legacy_mooncake_transfer_ref, normalize_transfer_ref


@dataclass
class TrainSample:
    mooncake_key: str | None = None
    tensor_shapes: Dict[str, Tuple[int, ...]] | None = None
    tensor_dtypes: Optional[Dict[str, torch.dtype | str]] = None
    packed_loss_mask: Optional[str] = None
    last_turn_loss_only: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None
    transfer_ref: TransferRef | Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        self.transfer_ref = normalize_transfer_ref(self.transfer_ref)
        if self.transfer_ref is None:
            if not self.mooncake_key:
                raise ValueError("training sample requires transfer_ref or mooncake_key")
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


def _as_consumer_backend(value: Any, device: torch.device) -> TransferBackend:
    backend = value if isinstance(value, TransferBackend) else MooncakeTransferBackend(value)
    backend.setup(TransferRole.CONSUMER, device)
    return backend


class TransferDataset(IterableDataset):
    """IterableDataset that loads backend-neutral tensor references via queue.

    Each DP rank waits on its queue for TrainSample items sent by the
    centralized controller. Data is loaded from mooncake.
    """

    def __init__(
        self,
        ray_queue: RayQueue,
        transfer_backend=None,
        device: torch.device | None = None,
        prefetch_factor: int = 2,
        timeout: Optional[float] = None,
        assistant_header_ids: Optional[List[int]] = None,
        end_token_ids: Optional[List[int]] = None,
        dynamic_loss_mask: bool = False,
        last_turn_loss_only: bool = False,
        skip_after_header: int = 0,
        batch_size: int = 1,
        min_loss_tokens: int = 0,
        usp_enabled: bool = False,
        ttt_length: int = 1,
        max_seq_length: Optional[int] = None,
        *,
        mooncake_store=None,
    ):
        if transfer_backend is not None and mooncake_store is not None:
            raise ValueError("provide transfer_backend or mooncake_store, not both")
        transfer_backend = transfer_backend if transfer_backend is not None else mooncake_store
        if transfer_backend is None:
            raise ValueError("a transfer backend is required")
        if device is None:
            raise ValueError("device is required")
        self.ray_queue = ray_queue
        self.transfer_backend = _as_consumer_backend(transfer_backend, device)
        # Retained for callers/tests that introspect the legacy attribute.
        self.mooncake_store = getattr(self.transfer_backend, "store", transfer_backend)
        self.device = device
        self.prefetch_factor = prefetch_factor
        self.timeout = timeout
        self.assistant_header_ids = assistant_header_ids
        self.end_token_ids = end_token_ids
        self.dynamic_loss_mask = dynamic_loss_mask
        self.last_turn_loss_only = last_turn_loss_only
        self.skip_after_header = skip_after_header
        self._batch_size = batch_size
        self._min_loss_tokens = min_loss_tokens
        self.usp_enabled = usp_enabled
        self.ttt_length = ttt_length
        self.max_seq_length = max_seq_length
        self._init_sp_context()

    def _init_sp_context(self) -> None:
        self._sp_group = None
        self._sp_world_size = 1
        self._sp_rank = 0
        self._sp_ring_size = 1
        self._sp_ring_rank = 0
        if not self.usp_enabled:
            return

        sp_group = get_draft_sp_group()
        if sp_group is None:
            return

        self._sp_group = sp_group
        self._sp_world_size = dist.get_world_size(sp_group)
        self._sp_rank = dist.get_rank(sp_group)

        ring_group = get_sp_ring_group()
        if ring_group is not None:
            self._sp_ring_size = dist.get_world_size(ring_group)
            self._sp_ring_rank = dist.get_rank(ring_group)

    def _load_from_transfer(self, sample: TrainSample) -> Dict[str, Any]:
        """Load tensors through the selected transfer backend."""
        logger.debug(
            f"_load_from_transfer: key={sample.mooncake_key}, "
            f"requesting shapes={sample.tensor_shapes}"
        )
        assert isinstance(sample.transfer_ref, TransferRef)
        tensor_dict = dict(self.transfer_backend.get(sample.transfer_ref, self.device))
        if self._batch_size > 1:
            # Clone to prevent use-after-free: collator holds sample N while
            # fetching N+1, but cleanup frees the Mooncake buffer (Issue 31).
            # Note: clone() converts pinned → unpinned, breaking non_blocking
            # H2D transfers. Only do this when actually needed.
            result = {k: v.clone() for k, v in tensor_dict.items()}
        else:
            # batch_size=1: safe to use pinned views — consumed immediately.
            # Preserves pinned memory for async H2D via non_blocking=True.
            result = dict(tensor_dict)

        self.transfer_backend.release(sample.transfer_ref)
        if sample.packed_loss_mask is not None:
            result["packed_loss_mask"] = sample.packed_loss_mask
        if sample.last_turn_loss_only is not None:
            result["last_turn_loss_only"] = sample.last_turn_loss_only
        return result

    _load_from_mooncake = _load_from_transfer

    def _cleanup_mooncake_data(self, sample: TrainSample) -> None:
        """Remove data from mooncake store to release buffer space."""
        assert isinstance(sample.transfer_ref, TransferRef)
        self.transfer_backend.release(sample.transfer_ref)

    def _compute_loss_mask(self, data: Dict[str, Any]) -> torch.Tensor | None:
        return resolve_loss_mask(
            data,
            dynamic_loss_mask=self.dynamic_loss_mask,
            assistant_header_ids=self.assistant_header_ids,
            end_token_ids=self.end_token_ids,
            last_turn_loss_only=self.last_turn_loss_only,
            skip_after_header=self.skip_after_header,
        )

    def _should_skip_for_loss_mask(
        self, data: Dict[str, Any], mooncake_key: str, skip_count: int
    ) -> tuple[bool, int]:
        mask = self._compute_loss_mask(data)
        if mask is None:
            skip_count += 1
            logger.warning(
                f"Skipping sample with all-zero loss mask "
                f"(mooncake_key={mooncake_key}, total_skipped={skip_count})"
            )
            return True, skip_count

        if (
            self._min_loss_tokens > 0
            and isinstance(mask, torch.Tensor)
            and mask.sum() < self._min_loss_tokens
        ):
            skip_count += 1
            logger.warning(
                f"Skipping sample with too few loss-masked tokens "
                f"({int(mask.sum())} < {self._min_loss_tokens}, "
                f"mooncake_key={mooncake_key}, total_skipped={skip_count})"
            )
            return True, skip_count

        return False, skip_count

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over samples synchronously.

        Blocks waiting for each item from the queue and loads from mooncake.
        Skips samples whose loss mask is all zeros to avoid wasted compute.
        """
        yield_count = 0
        skip_count = 0
        while True:
            if self.usp_enabled:
                data, skipped = self._usp_get_sharded_item(skip_count=skip_count)
                skip_count += skipped
                if data is None:
                    break
                yield_count += 1
                yield data
                continue

            logger.debug(f"__iter__: waiting for item from ray_queue (yield_count={yield_count})")
            try:
                item = self.ray_queue.get(block=True, timeout=self.timeout)
            except Exception as e:
                logger.warning(f"__iter__: Exception waiting for data: {e}, timeout={self.timeout}")
                break

            if item is None:
                logger.debug("__iter__: received None sentinel, stopping iteration")
                break

            logger.debug(f"__iter__: got item, mooncake_key={item.mooncake_key}")
            data = self._load_from_transfer(item)

            should_skip, skip_count = self._should_skip_for_loss_mask(
                data, item.mooncake_key, skip_count
            )
            if should_skip:
                continue

            # Note: target is computed in the collator from last_hidden_states for sglang mode

            # Add batch dimension if missing (sglang stores without batch dim)
            for key, tensor in data.items():
                if tensor is not None and isinstance(tensor, torch.Tensor):
                    # Check if tensor is missing batch dimension
                    # 1D tensors (loss_mask, input_ids) should be 2D: (1, seq_len)
                    # 2D tensors (hidden_states, last_hidden_states) should be 3D: (1, seq_len, dim)
                    if tensor.dim() == 1:
                        data[key] = tensor.unsqueeze(0)  # (seq_len,) -> (1, seq_len)
                    elif tensor.dim() == 2 and key in [
                        "hidden_states",
                        "last_hidden_states",
                        "target",
                    ]:
                        data[key] = tensor.unsqueeze(0)  # (seq_len, dim) -> (1, seq_len, dim)

            # Debug: log all tensor shapes after adding batch dim
            if data:
                shapes_str = {
                    k: v.shape if hasattr(v, "shape") else type(v) for k, v in data.items()
                }
                logger.debug(f"final shapes (with batch dim): {shapes_str}")
            yield_count += 1
            logger.debug(f"__iter__: yielding batch {yield_count}, keys={list(data.keys())}")
            yield data

    def _usp_global_len(self, sample: TrainSample) -> int:
        global_len = sample.tensor_shapes["input_ids"][-1]
        if self.max_seq_length is not None:
            global_len = min(global_len, self.max_seq_length)
        return global_len

    def _usp_chunk_size(self, global_len: int) -> int:
        return (global_len + self._sp_world_size - 1) // self._sp_world_size

    def _usp_loss_mask(self, sample: TrainSample, global_len: int) -> torch.Tensor:
        if sample.packed_loss_mask is None:
            raise RuntimeError("USP sharded Mooncake reads require packed_loss_mask metadata")
        loss_mask = unpack_loss_mask(deserialize_packed_loss_mask(sample.packed_loss_mask))
        loss_mask = loss_mask[:global_len]
        if loss_mask.shape[0] < global_len:
            loss_mask = F.pad(loss_mask, (0, global_len - loss_mask.shape[0]))
        return loss_mask

    def _local_usp_shapes(self, sample: TrainSample) -> dict[str, tuple[int, ...]]:
        local_len = self._usp_chunk_size(self._usp_global_len(sample)) + self.ttt_length
        shapes: dict[str, tuple[int, ...]] = {
            "input_ids": (1, local_len),
            "hidden_states": (1, local_len, sample.tensor_shapes["hidden_states"][-1]),
        }
        if "last_hidden_states" in sample.tensor_shapes:
            shapes["last_hidden_states"] = (
                1,
                local_len,
                sample.tensor_shapes["last_hidden_states"][-1],
            )
        if "target" in sample.tensor_shapes:
            shapes["target"] = (1, local_len, sample.tensor_shapes["target"][-1])
        return shapes

    def _local_usp_loss_and_position(
        self,
        sample: TrainSample,
        local_len: int,
    ) -> dict[str, torch.Tensor]:
        sp_ulysses_size = max(1, self._sp_world_size // self._sp_ring_size)
        global_len = self._usp_global_len(sample)
        chunk_size = self._usp_chunk_size(global_len)
        start = self._sp_rank * chunk_size
        end = min(start + local_len, global_len)
        valid_len = max(0, end - start)

        loss_mask = self._usp_loss_mask(sample, global_len)[start:end].unsqueeze(0)
        if loss_mask.shape[-1] < local_len:
            loss_mask = F.pad(loss_mask, (0, local_len - loss_mask.shape[-1]))

        attention_mask = torch.zeros((1, local_len), dtype=torch.long)
        attention_mask[:, :valid_len] = 1

        usp_chunk_size = max(local_len - self.ttt_length, 0)
        ring_chunk = usp_chunk_size * sp_ulysses_size
        _, ring_rank = get_usp_rank_coords(
            sp_rank=self._sp_rank,
            sp_ulysses_size=sp_ulysses_size,
            sp_ring_size=self._sp_ring_size,
        )
        ring_start = ring_rank * ring_chunk
        position_ids = torch.arange(
            ring_start,
            ring_start + ring_chunk,
            dtype=torch.long,
        ).unsqueeze(0)

        return {
            "loss_mask": loss_mask.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "position_ids": position_ids.to(self.device),
        }

    def _should_skip_usp_sharded_sample(self, sample: TrainSample) -> bool:
        """Return the SP-consistent skip decision for a pre-sharded USP sample."""
        full_loss_mask = self._usp_loss_mask(sample, self._usp_global_len(sample))
        min_tokens = max(1, self._min_loss_tokens)
        return int(full_loss_mask.sum().item()) < min_tokens

    def _usp_get_sharded_item(self, skip_count: int) -> tuple[Dict[str, torch.Tensor] | None, int]:
        skipped = 0
        while True:
            try:
                item = self.ray_queue.get(block=True, timeout=self.timeout)
            except Exception as e:
                logger.warning(
                    f"_usp_get_sharded_item: Exception waiting for data: {e}, "
                    f"timeout={self.timeout}"
                )
                return None, skipped
            if item is None:
                return None, skipped

            metadata = item.metadata or {}
            if not metadata.get("usp_sharded", False):
                raise RuntimeError(
                    "USP sharded data fetcher received a non-sharded Mooncake sample. "
                    f"mooncake_key={item.mooncake_key}"
                )

            shapes = self._local_usp_shapes(item)
            should_skip = self._should_skip_usp_sharded_sample(item)
            shard_key = f"{item.mooncake_key}_usp{self._sp_rank}"
            assert isinstance(item.transfer_ref, TransferRef)
            if item.transfer_ref.shards:
                if self._sp_rank >= len(item.transfer_ref.shards):
                    raise TransferProtocolError(
                        f"transfer ref has {len(item.transfer_ref.shards)} USP shards, "
                        f"but rank {self._sp_rank} was requested"
                    )
                shard_ref = item.transfer_ref.shards[self._sp_rank]
            else:
                dtypes = item.tensor_dtypes or {}
                shard_ref = TransferRef(
                    backend=item.transfer_ref.backend,
                    object_id=shard_key,
                    tensors={
                        name: TensorSpec(
                            shape,
                            str(dtypes.get(name, torch.bfloat16)).removeprefix("torch."),
                        )
                        for name, shape in shapes.items()
                    },
                    locator={"key": shard_key},
                    metadata=item.transfer_ref.metadata,
                )
            tensors = dict(self.transfer_backend.get(shard_ref, self.device))
            tensors.update(self._local_usp_loss_and_position(item, shapes["input_ids"][-1]))
            self.transfer_backend.release(shard_ref)

            if should_skip:
                skipped += 1
                total_skipped = skip_count + skipped
                logger.warning(
                    f"Skipping USP sharded sample with global all-zero loss mask "
                    f"(mooncake_key={item.mooncake_key}, sp_rank={self._sp_rank}, "
                    f"total_skipped={total_skipped})"
                )
                continue

            return tensors, skipped


def create_transfer_dataloader(
    ray_queue: RayQueue,
    transfer_backend=None,
    collator: Callable[[List[Dict]], Dict[str, torch.Tensor]] | None = None,
    device: torch.device | None = None,
    batch_size: int = 1,
    prefetch_factor: int = 2,
    timeout: Optional[float] = None,
    assistant_header_ids: Optional[List[int]] = None,
    end_token_ids: Optional[List[int]] = None,
    dynamic_loss_mask: bool = False,
    last_turn_loss_only: bool = False,
    skip_after_header: int = 0,
    min_loss_tokens: int = 0,
    usp_enabled: bool = False,
    ttt_length: int = 1,
    max_seq_length: Optional[int] = None,
    *,
    mooncake_store=None,
) -> DataLoader:
    """Create a DataLoader that fetches transfer references via queue.

    Data flow:
      Controller (dispatches dispatch_batch_size samples) ->
      Ray Queue (per_dp_rank_batch_size samples per rank) ->
      DataLoader (batches per_dp_rank_batch_size samples together with padding) ->
      Training loop (one iteration per step)

    The collator pads sequences within the batch to the same length.

    Args:
        ray_queue: Ray Queue to receive TrainSample from controller.
        mooncake_store: Mooncake store client for loading tensors.
        collator: Collator for padding and batching samples.
        device: Target device for tensors.
        batch_size: Number of samples per batch (= per_dp_rank_batch_size).
        prefetch_factor: Unused, kept for API compatibility.
        timeout: Timeout in seconds for waiting on queue. None means wait forever.
        assistant_header_ids: Token IDs for assistant header (for loss mask skip check).
        end_token_ids: Token IDs for end of turn (for loss mask skip check).
        dynamic_loss_mask: Whether loss mask is computed dynamically from input_ids.
        last_turn_loss_only: Global fallback for last-turn-only loss masking.

    Returns:
        DataLoader instance.
    """
    if transfer_backend is not None and mooncake_store is not None:
        raise ValueError("provide transfer_backend or mooncake_store, not both")
    transfer_backend = transfer_backend if transfer_backend is not None else mooncake_store
    if transfer_backend is None or collator is None or device is None:
        raise ValueError("transfer backend, collator, and device are required")

    dataset = TransferDataset(
        ray_queue,
        transfer_backend,
        device,
        prefetch_factor,
        timeout,
        assistant_header_ids=assistant_header_ids,
        end_token_ids=end_token_ids,
        dynamic_loss_mask=dynamic_loss_mask,
        last_turn_loss_only=last_turn_loss_only,
        skip_after_header=skip_after_header,
        batch_size=batch_size,
        min_loss_tokens=min_loss_tokens,
        usp_enabled=usp_enabled,
        ttt_length=ttt_length,
        max_seq_length=max_seq_length,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=0,
    )


class TransferDataFetcher:
    """Queue-based data fetcher with a backend-neutral DataLoader backend.

    Provides iteration over training samples that are pushed to a Ray queue
    by the AsyncTrainingController and loaded from mooncake.

    Batch size design:
      - micro_batch_size: Samples per GPU per training step (user config)
      - per_dp_rank_batch_size = micro_batch_size * sp_size (derived)
      - dispatch_batch_size = per_dp_rank_batch_size * dp_size (derived)
      - DataLoader batch_size = per_dp_rank_batch_size (all samples batched together)
      - Training loop does ONE iteration per step

    The collator pads sequences within the batch to the max length.
    """

    def __init__(
        self,
        queue: RayQueue,
        transfer_backend=None,
        collator: Callable[[List[Dict]], Dict[str, torch.Tensor]] | None = None,
        device: torch.device | None = None,
        batch_size: int = 1,
        prefetch_factor: int = 2,
        timeout: Optional[float] = None,
        assistant_header_ids: Optional[List[int]] = None,
        end_token_ids: Optional[List[int]] = None,
        dynamic_loss_mask: bool = False,
        last_turn_loss_only: bool = False,
        skip_after_header: int = 0,
        min_loss_tokens: int = 0,
        usp_enabled: bool = False,
        ttt_length: int = 1,
        max_seq_length: Optional[int] = None,
        *,
        mooncake_store=None,
    ):
        if transfer_backend is not None and mooncake_store is not None:
            raise ValueError("provide transfer_backend or mooncake_store, not both")
        transfer_backend = transfer_backend if transfer_backend is not None else mooncake_store
        if transfer_backend is None or collator is None or device is None:
            raise ValueError("transfer backend, collator, and device are required")
        self.batch_size = batch_size
        self._dataloader = create_transfer_dataloader(
            ray_queue=queue,
            transfer_backend=transfer_backend,
            collator=collator,
            device=device,
            batch_size=batch_size,
            prefetch_factor=prefetch_factor,
            timeout=timeout,
            assistant_header_ids=assistant_header_ids,
            end_token_ids=end_token_ids,
            dynamic_loss_mask=dynamic_loss_mask,
            last_turn_loss_only=last_turn_loss_only,
            skip_after_header=skip_after_header,
            min_loss_tokens=min_loss_tokens,
            usp_enabled=usp_enabled,
            ttt_length=ttt_length,
            max_seq_length=max_seq_length,
        )

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        return iter(self._dataloader)


MooncakeDataset = TransferDataset
create_mooncake_dataloader = create_transfer_dataloader
MooncakeDataFetcher = TransferDataFetcher


class PrefetchedDataFetcher:
    """Wraps MooncakeDataFetcher with async pre-fetching.

    A background thread continuously fetches batches from the underlying
    MooncakeDataFetcher (which blocks on Mooncake TCP), staging them in a
    thread-safe queue.  The training loop reads from this queue, overlapping
    data transfer with GPU compute.

    Without prefetch: [data] → [compute] → [data] → [compute]  (sequential)
    With prefetch:    [compute] → [compute] → [compute]         (overlapped)
                      [data]      [data]      [data]

    The background thread starts lazily on the first ``__iter__`` call and
    keeps running across multiple ``itertools.islice`` invocations (one per
    training step).  The training loop simply reads from the shared queue.
    """

    _SENTINEL = object()

    def __init__(
        self,
        inner: MooncakeDataFetcher,
        prefetch_depth: int = 2,
        target_device: Optional[torch.device] = None,
    ):
        self.inner = inner
        self.prefetch_depth = prefetch_depth
        self.target_device = target_device
        self._queue: queue.Queue = queue.Queue(maxsize=prefetch_depth)
        self._thread: Optional[threading.Thread] = None
        self._started = False
        self._error: Optional[BaseException] = None

    def _prefetch_loop(self) -> None:
        try:
            for batch in self.inner:
                self._queue.put(batch)
        except Exception as e:
            # Preserve the original traceback so re-raise in __next__
            # points to the actual failure site, not to __next__ itself.
            import sys

            self._error = e.with_traceback(sys.exc_info()[2])
        finally:
            self._queue.put(self._SENTINEL)

    def _ensure_started(self) -> None:
        if not self._started:
            self._started = True
            self._thread = threading.Thread(target=self._prefetch_loop, daemon=True)
            self._thread.start()

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        self._ensure_started()
        return self

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move a batch of tensors to the target device (GPU)."""
        if self.target_device is None:
            return batch
        return {
            k: v.to(self.target_device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def __next__(self) -> Dict[str, torch.Tensor]:
        if self._error is not None:
            raise self._error
        item = self._queue.get()
        if item is self._SENTINEL:
            if self._error is not None:
                raise self._error
            raise StopIteration
        return self._to_device(item)
