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

"""
HF inference runner for generating training data from pre-tokenized inputs.

Uses HFTargetModel with HuggingFace backend. Accepts pre-tokenized input_ids
and packed_loss_mask, runs inference to extract hidden states and logits,
then stores tensors in mooncake and returns keys.
"""

import os
import uuid
from typing import Any, Optional

import torch
import torch.distributed as dist

from torchspec.config.inference_config import HFInferenceConfig
from torchspec.config.mooncake_config import MooncakeConfig
from torchspec.models.target import HFTargetModel
from torchspec.transfer.base import TransferBackend, TransferRole
from torchspec.transfer.factory import create_transfer_backend
from torchspec.transfer.mooncake.backend import MooncakeTransferBackend
from torchspec.transfer.mooncake.eagle_store import EagleMooncakeStore
from torchspec.utils.logging import logger


class _PreinitializedMooncakeTransferBackend(MooncakeTransferBackend):
    """Adapt a caller-owned store without invoking its setup hook again."""

    def _setup(self, role: TransferRole, device: torch.device | str | int | None) -> None:
        del role, device


class HFRunner:
    """Inference engine that generates training data from pre-tokenized inputs.

    This engine wraps HFTargetModel and produces training data
    (hidden_states, last_hidden_states, etc.) from pre-tokenized input_ids and loss_mask.
    Supports distributed inference with tensor parallelism.

    When mooncake_store is provided, tensors are stored there and keys are returned.
    This is the recommended mode for async training.
    """

    def __init__(
        self,
        config: HFInferenceConfig,
        mooncake_store: Optional[EagleMooncakeStore] = None,
        transfer_backend: Optional[TransferBackend] = None,
    ):
        """
        Args:
            config: HFInferenceConfig with model and backend settings.
            mooncake_store: Optional pre-initialized mooncake store for tensor storage.
                           If provided, uses this store directly.
                           If None and config.mooncake_config is set, initializes store during setup().
        """
        self.config = config
        self.mooncake_store = mooncake_store
        self.transfer_backend = transfer_backend
        self.target_model: Optional[HFTargetModel] = None
        self._initialized = False

    def init_mooncake_store(
        self,
        mooncake_config: Optional[MooncakeConfig] = None,
    ) -> EagleMooncakeStore:
        """Initialize and setup an EagleMooncakeStore.

        Args:
            mooncake_config: Optional config. If None, uses config.mooncake_config.

        Returns:
            Initialized EagleMooncakeStore instance.
        """
        if mooncake_config is None:
            mooncake_config = self.config.mooncake_config

        if mooncake_config is None:
            raise ValueError(
                "mooncake_config must be provided either in HFInferenceConfig "
                "or as argument to init_mooncake_store()"
            )

        store = EagleMooncakeStore(mooncake_config)
        backend = MooncakeTransferBackend(store)
        backend.setup(TransferRole.PRODUCER, self._producer_device())
        self.mooncake_store = store
        self.transfer_backend = backend

        tp_rank = self._get_tp_rank()
        logger.info(f"[Rank {tp_rank}] EagleMooncakeStore initialized")

        return store

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        max_seq_length: int = 8192,
        trust_remote_code: bool = False,
        aux_hidden_states_layers: Optional[list[int]] = None,
        mooncake_config: Optional[MooncakeConfig] = None,
        mooncake_store: Optional[EagleMooncakeStore] = None,
        transfer_config: Any = None,
        transfer_backend: Optional[TransferBackend] = None,
        **kwargs,
    ) -> "HFRunner":
        """Create HFRunner from a pretrained model path.

        Args:
            pretrained_model_name_or_path: Path to pretrained model.
            torch_dtype: Data type for model weights.
            max_seq_length: Maximum sequence length.
            trust_remote_code: Whether to trust remote code.
            aux_hidden_states_layers: Layers to capture auxiliary hidden states from.
            mooncake_config: Optional MooncakeConfig for initializing mooncake store.
            mooncake_store: Optional pre-initialized mooncake store for tensor storage.
            **kwargs: Additional arguments passed to HFTargetModel.from_pretrained.

        Returns:
            Initialized HFRunner instance.
        """
        dtype_str = "bfloat16"
        if torch_dtype == torch.float16:
            dtype_str = "float16"
        elif torch_dtype == torch.float32:
            dtype_str = "float32"

        config = HFInferenceConfig(
            model_path=pretrained_model_name_or_path,
            max_seq_length=max_seq_length,
            torch_dtype=dtype_str,
            trust_remote_code=trust_remote_code,
            aux_hidden_states_layers=aux_hidden_states_layers,
            mooncake_config=mooncake_config,
            transfer_config=transfer_config,
        )

        engine = cls(
            config=config,
            mooncake_store=mooncake_store,
            transfer_backend=transfer_backend,
        )
        engine.setup()
        return engine

    def set_mooncake_store(self, mooncake_store) -> None:
        """Set or update the mooncake store for tensor storage."""
        self.mooncake_store = mooncake_store
        backend = _PreinitializedMooncakeTransferBackend(mooncake_store)
        backend.setup(TransferRole.PRODUCER)
        self.transfer_backend = backend

    def init_transfer_backend(self, transfer_config: Any = None) -> TransferBackend:
        """Initialize the configured producer backend."""
        transfer_config = transfer_config or self.config.transfer_config
        if transfer_config is None:
            raise ValueError("transfer_config must be provided")
        backend = create_transfer_backend(transfer_config)
        backend.setup(TransferRole.PRODUCER, self._producer_device())
        self.transfer_backend = backend
        if isinstance(backend, MooncakeTransferBackend):
            self.mooncake_store = backend.store
        return backend

    def setup(self) -> None:
        """Initialize target model and mooncake store."""
        if self._initialized:
            return

        self._setup_target_model()
        self._init_mooncake_store_if_configured()
        if self.transfer_backend is not None:
            # TransferBackend.setup is idempotent for an already-ready backend
            # with the same role, while ensuring a caller-supplied NEW backend
            # is ready before generate() reaches put().
            self.transfer_backend.setup(TransferRole.PRODUCER, self._producer_device())
        elif self.config.transfer_config is not None:
            self.init_transfer_backend()
        elif self.mooncake_store is not None:
            backend = _PreinitializedMooncakeTransferBackend(self.mooncake_store)
            backend.setup(TransferRole.PRODUCER)
            self.transfer_backend = backend
        self._initialized = True

    def _init_mooncake_store_if_configured(self) -> None:
        """Initialize mooncake store if config is provided but store doesn't exist."""
        if (
            self.transfer_backend is None
            and self.mooncake_store is None
            and self.config.mooncake_config is not None
        ):
            self.init_mooncake_store()

    @staticmethod
    def _producer_device() -> int | None:
        """Return the active CUDA device without probing it on CPU-only hosts."""
        if not torch.cuda.is_available():
            return None
        return torch.cuda.current_device()

    def _get_tp_rank(self) -> int:
        """Get tensor parallel rank from distributed context or environment."""
        if dist.is_initialized():
            return dist.get_rank()
        local_rank = os.environ.get("LOCAL_RANK")
        if local_rank is not None:
            return int(local_rank)
        return 0

    def _is_main_rank(self) -> bool:
        """Check if this is the main rank (rank 0) for communication."""
        return self._get_tp_rank() == 0

    def _setup_target_model(self) -> None:
        """Initialize HFTargetModel."""
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.config.torch_dtype, torch.bfloat16)

        self.target_model = HFTargetModel.from_pretrained(
            pretrained_model_name_or_path=self.config.model_path,
            torch_dtype=torch_dtype,
        )
        self.target_model.set_aux_hidden_states_layers(self.config.aux_hidden_states_layers)

    def set_aux_hidden_states_layers(self, layers: Optional[list[int]]) -> None:
        """Update the auxiliary hidden states layers to capture."""
        self.config.aux_hidden_states_layers = layers
        if self.target_model is not None:
            self.target_model.set_aux_hidden_states_layers(layers)

    @torch.no_grad()
    def generate(
        self,
        data_id: str,
        input_ids_list: list[torch.Tensor],
        packed_loss_mask_list: list[str],
    ) -> list[dict[str, Any]]:
        """Generate training data from pre-tokenized inputs.

        Args:
            input_ids_list: List of input_ids tensors, one per sample.
            packed_loss_mask_list: List of packed loss_mask strings, one per sample (passed through).

        Returns:
            If mooncake_store is set:
                List of dicts with mooncake_key, tensor_shapes, tensor_dtypes, packed_loss_mask.
            Otherwise:
                List of dicts with "tensors" key containing tensor data.
        """
        assert input_ids_list is not None, "input_ids_list must not be None"
        assert packed_loss_mask_list is not None, "packed_loss_mask_list must not be None"

        if not self._initialized:
            self.setup()

        input_ids_list = [ids.cuda() if not ids.is_cuda else ids for ids in input_ids_list]

        inference_outputs = self._run_inference(
            input_ids=input_ids_list,
        )

        results = []
        for i, sample in enumerate(inference_outputs):
            if self.transfer_backend is not None:
                key = str(uuid.uuid4())
                ref = self.transfer_backend.put(key, sample)
                result = {
                    "transfer_ref": ref.to_dict(),
                    "packed_loss_mask": packed_loss_mask_list[i],
                }
                if ref.backend == "mooncake":
                    result.update(
                        mooncake_key=key,
                        tensor_shapes={name: spec.shape for name, spec in ref.tensors.items()},
                        tensor_dtypes={name: spec.dtype for name, spec in ref.tensors.items()},
                    )
                results.append(result)
            else:
                results.append({"tensors": sample, "packed_loss_mask": packed_loss_mask_list[i]})

        if self.transfer_backend is not None:
            self.transfer_backend.flush()

        return results

    def _run_inference(
        self,
        input_ids: list[torch.Tensor],
    ) -> list[dict[str, torch.Tensor]]:
        """Run inference using HFTargetModel.

        Args:
            input_ids: List of input token ID tensors, one per sample.

        Returns:
            List of dicts, one per sample, each containing tensors at original length.
        """
        input_ids = [ids.unsqueeze(0) if ids.dim() == 1 else ids for ids in input_ids]

        results = []
        for ids in input_ids:
            attention_mask = torch.ones_like(ids)
            loss_mask = torch.ones_like(ids)

            output = self.target_model.generate_eagle3_data(
                input_ids=ids,
                attention_mask=attention_mask,
                loss_mask=loss_mask,
            )

            results.append(
                {
                    "input_ids": output.input_ids,
                    "hidden_states": output.hidden_states,
                    "target": output.target,
                    "last_hidden_states": output.last_hidden_states,
                }
            )

        return results

    def shutdown(self) -> None:
        """Clean up resources including mooncake store."""
        if self.transfer_backend is not None:
            self.transfer_backend.close()
            tp_rank = self._get_tp_rank()
            logger.info(f"[Rank {tp_rank}] transfer backend closed")
            self.transfer_backend = None
            self.mooncake_store = None
        elif self.mooncake_store is not None and hasattr(self.mooncake_store, "close"):
            self.mooncake_store.close()
            self.mooncake_store = None

        if self.target_model is not None:
            self.target_model = None

        self._initialized = False

    def __enter__(self):
        """Context manager entry."""
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
        return False
