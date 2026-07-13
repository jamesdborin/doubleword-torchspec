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

"""DFlash training model: wraps the DFlash draft model with training-specific logic.

Handles anchor sampling, block-causal mask generation, noise input construction,
and cross-entropy loss with exponential decay weighting.

Matches SpecForge's OnlineDFlashModel (specforge/core/dflash.py).
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchspec.models.ops.flex_attention import compile_friendly_create_block_mask
from torchspec.models.ops.liger import is_liger_available, make_liger_fused_linear_ce
from torchspec.utils.logging import logger

_VALID_DFLASH_LOSS_OBJECTIVES = {"decay", "dpace"}


def _dpace_position_weights(confidences: torch.Tensor, alpha: float) -> torch.Tensor:
    """Compute detached D-PACE weights from per-position draft confidences."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"dflash_dpace_alpha must be in [0, 1], got {alpha}")

    with torch.no_grad():
        smoothed = (1.0 - alpha) * confidences.float() + alpha
        prefix_products = torch.cumprod(smoothed, dim=-1)
        weights = torch.flip(
            torch.cumsum(torch.flip(prefix_products, dims=[-1]), dim=-1),
            dims=[-1],
        )
        return weights.to(dtype=confidences.dtype)


def _create_dflash_mask_mod(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    ctx_len: int,
    block_size: int,
):
    """Create a mask_mod function for DFlash block-causal attention.

    KV: [Context (ctx_len tokens) | Block_0 | Block_1 | ... | Block_{n-1}]
    Q:  [Block_0 | Block_1 | ... | Block_{n-1}]

    Rules:
      1. Each block sees context strictly before its anchor (kv_idx < anchor_pos)
      2. Intra-block attention is bidirectional (per SpecForge PR #427)
      3. Different blocks are invisible to each other
      4. Invalid blocks (block_keep_mask=False) see nothing
    """
    num_anchors = anchor_positions.shape[1]

    def dflash_mask_mod(b, h, q_idx, kv_idx):
        q_block_id = q_idx // block_size
        anchor_pos = anchor_positions[b, q_block_id]

        is_context = kv_idx < ctx_len
        mask_context = is_context & (kv_idx < anchor_pos)

        is_draft = kv_idx >= ctx_len
        kv_block_id = (kv_idx - ctx_len) // block_size
        mask_draft = is_draft & (q_block_id == kv_block_id)

        is_valid_block = block_keep_mask[b, q_block_id]
        return (mask_context | mask_draft) & is_valid_block

    dflash_mask_mod.__name__ = f"dflash_mask_A{num_anchors}_B{block_size}_C{ctx_len}"
    return dflash_mask_mod


class DFlashModel(nn.Module):
    """DFlash training wrapper.

    Wraps the DFlash draft model with training-specific logic:
      - Random anchor sampling with block_keep_mask
      - Block-causal attention mask via FlexAttention
      - Noise input construction (anchor + MASK)
      - Cross-entropy loss with configurable position weighting
      - Per-position loss_mask application
    """

    def __init__(
        self,
        draft_model,
        block_size: int = 16,
        num_anchors: int = 512,
        loss_objective: str = "decay",
        dpace_alpha: float = 0.5,
        loss_decay_gamma: float = 7.0,
        use_liger_kernel: bool = True,
    ):
        super().__init__()
        loss_objective = loss_objective.lower()
        if loss_objective not in _VALID_DFLASH_LOSS_OBJECTIVES:
            valid = ", ".join(sorted(_VALID_DFLASH_LOSS_OBJECTIVES))
            raise ValueError(
                f"Unknown DFlash loss objective {loss_objective!r}; expected one of {valid}"
            )
        if not 0.0 <= dpace_alpha <= 1.0:
            raise ValueError(f"dflash_dpace_alpha must be in [0, 1], got {dpace_alpha}")

        self.draft_model = draft_model
        self.block_size = block_size
        self.num_anchors = num_anchors
        self.loss_objective = loss_objective
        self.dpace_alpha = dpace_alpha
        self.loss_decay_gamma = loss_decay_gamma
        self.use_liger_kernel = use_liger_kernel
        self._liger_fused_linear_ce = None
        self._liger_unavailable_logged = False

    def _get_lm_head_weight_bias(
        self,
        lm_head_weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        if hasattr(self.draft_model, "lm_head"):
            lm_head = self.draft_model.lm_head
            return lm_head.weight, getattr(lm_head, "bias", None)
        return lm_head_weight, None

    def _get_liger_fused_linear_ce(self):
        if self._liger_fused_linear_ce is None:
            self._liger_fused_linear_ce = make_liger_fused_linear_ce(
                reduction="sum",
                return_token_accuracy=True,
                accum_dtype=torch.float32,
            )
        return self._liger_fused_linear_ce

    def _can_use_liger_loss(self, device: torch.device) -> bool:
        if not self.use_liger_kernel:
            return False
        if self.loss_objective != "decay":
            return False
        if device.type != "cuda":
            return False
        if is_liger_available():
            return True
        if not self._liger_unavailable_logged:
            logger.warning("Liger-Kernel is not available; using PyTorch DFlash CE loss.")
            self._liger_unavailable_logged = True
        return False

    def _compute_torch_loss(
        self,
        draft_hidden: torch.Tensor,
        target_ids: torch.Tensor,
        weight_mask: torch.Tensor,
        binary_eval_mask: torch.Tensor,
        lm_head_weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, n_blocks, _ = target_ids.shape
        device = draft_hidden.device

        lm_weight, lm_bias = self._get_lm_head_weight_bias(lm_head_weight)
        logits = F.linear(draft_hidden, lm_weight, lm_bias)

        flat_logits = logits.view(-1, logits.size(-1))
        flat_targets = target_ids.view(-1)
        loss_per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")
        loss_per_token_by_position = loss_per_token.view(bsz, n_blocks, self.block_size)

        objective_weights = weight_mask
        if (
            self.loss_objective == "decay"
            and self.loss_decay_gamma is not None
            and self.loss_decay_gamma > 0
        ):
            # Loss decay: exp(-(k-1)/gamma) so k=1 gets weight 1.0.
            k = torch.arange(self.block_size, device=device).view(1, 1, -1)
            decay_weights = torch.exp(-(k - 1).clamp(min=0).float() / self.loss_decay_gamma)
            objective_weights = weight_mask * decay_weights
        elif self.loss_objective == "dpace":
            dpace_weights = torch.ones_like(weight_mask)
            if self.block_size > 1:
                with torch.no_grad():
                    target_confidences = torch.exp(-loss_per_token_by_position[..., 1:].float())
                    dpace_pred_weights = _dpace_position_weights(
                        target_confidences,
                        self.dpace_alpha,
                    ).to(dtype=weight_mask.dtype)
                dpace_weights[..., 1:] = dpace_pred_weights
            objective_weights = weight_mask * dpace_weights

        flat_weights = objective_weights.view(-1)
        valid_token_count = flat_weights.sum().clamp(min=1e-6)
        loss = (loss_per_token * flat_weights).sum() / valid_token_count

        with torch.no_grad():
            pred_ids = torch.argmax(flat_logits, dim=-1)
            correct = (pred_ids == flat_targets) & (binary_eval_mask > 0.5)
            actual_token_count = binary_eval_mask.sum().clamp(min=1e-6)
            accuracy = correct.sum().float() / actual_token_count

            binary_weights = binary_eval_mask.view(bsz, n_blocks, self.block_size)
            count_per_position = binary_weights.sum(dim=(0, 1))
            count_per_pos = count_per_position.clamp(min=1.0)

            loss_per_position = (
                loss_per_token.view(bsz, n_blocks, self.block_size) * binary_weights
            ).sum(dim=(0, 1)) / count_per_pos
            acc_per_position = (correct.view(bsz, n_blocks, self.block_size).float()).sum(
                dim=(0, 1)
            ) / count_per_pos

        return loss, accuracy, loss_per_position, acc_per_position, count_per_position

    def _compute_liger_decay_loss(
        self,
        draft_hidden: torch.Tensor,
        target_ids: torch.Tensor,
        weight_mask: torch.Tensor,
        binary_eval_mask: torch.Tensor,
        lm_head_weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the static-decay CE objective with Liger fused linear CE."""
        bsz, n_blocks, _ = target_ids.shape
        device = draft_hidden.device
        lm_weight, lm_bias = self._get_lm_head_weight_bias(lm_head_weight)
        liger_loss = self._get_liger_fused_linear_ce()

        flat_hidden = draft_hidden.reshape(-1, draft_hidden.size(-1))
        flat_targets = target_ids.reshape(-1)
        binary_by_position = binary_eval_mask.view(bsz, n_blocks, self.block_size)
        block_offsets = torch.arange(bsz * n_blocks, device=device) * self.block_size

        loss_sum = torch.zeros((), dtype=torch.float32, device=device)
        weighted_count = torch.zeros((), dtype=torch.float32, device=device)
        count_per_position = torch.zeros(self.block_size, dtype=torch.float32, device=device)
        loss_per_position = torch.zeros(self.block_size, dtype=torch.float32, device=device)
        acc_per_position = torch.zeros(self.block_size, dtype=torch.float32, device=device)

        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            k = torch.arange(self.block_size, device=device)
            decay_weights = torch.exp(-(k - 1).clamp(min=0).float() / self.loss_decay_gamma)
        else:
            decay_weights = torch.ones(self.block_size, dtype=torch.float32, device=device)

        for pos in range(self.block_size):
            valid_mask = binary_by_position[..., pos].reshape(-1) > 0.5
            count = valid_mask.sum()
            if count.item() == 0:
                continue

            block_idx = valid_mask.nonzero(as_tuple=False).squeeze(-1)
            valid_idx = block_offsets.index_select(0, block_idx) + pos
            hidden = flat_hidden.index_select(0, valid_idx).contiguous()
            targets = flat_targets.index_select(0, valid_idx).contiguous()
            result = liger_loss(lm_weight, hidden, targets, bias=lm_bias)

            ce_sum = result.loss.float() if hasattr(result, "loss") else result.float()
            token_accuracy = (
                result.token_accuracy.float()
                if hasattr(result, "token_accuracy") and result.token_accuracy is not None
                else torch.zeros((), dtype=torch.float32, device=device)
            )

            count_f = count.to(dtype=torch.float32)
            objective_weight = decay_weights[pos]
            loss_sum = loss_sum + ce_sum * objective_weight
            weighted_count = weighted_count + count_f * objective_weight
            count_per_position[pos] = count_f
            loss_per_position[pos] = ce_sum / count_f.clamp(min=1.0)
            acc_per_position[pos] = token_accuracy

        loss = loss_sum / weighted_count.clamp(min=1e-6)
        actual_token_count = count_per_position.sum().clamp(min=1e-6)
        accuracy = (acc_per_position * count_per_position).sum() / actual_token_count

        return loss, accuracy, loss_per_position, acc_per_position, count_per_position

    def _sample_anchor_positions(
        self,
        seq_len: int,
        loss_mask: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample anchor positions per sample; returns (anchors, keep_mask).

        Always returns exactly ``self.num_anchors`` anchor slots so that
        ``Q_LEN = num_anchors * block_size`` is constant across steps,
        preventing FlexAttention recompilation from shape changes.  Samples
        with fewer valid positions use ``block_keep_mask=False`` for the
        excess slots (those blocks are skipped by the block-sparse kernel).

        Args:
            seq_len: sequence length
            loss_mask: [B, seq_len] — 1 for valid positions, 0 for padding
            device: torch device

        Returns:
            anchors: [B, num_anchors] — sampled anchor positions (sorted)
            keep_mask: [B, num_anchors] — True for valid sampled anchors
        """
        bs = self.block_size
        bsz = loss_mask.shape[0]
        max_anchor = max(seq_len - bs, 0)
        max_n = self.num_anchors

        if max_anchor == 0:
            logger.warning(
                f"Sequence too short for anchor sampling (seq_len={seq_len}, "
                f"block_size={bs}). Returning dummy anchors so loss is zero."
            )
            anchors = torch.zeros(bsz, max_n, dtype=torch.long, device=device)
            keep_mask = torch.zeros(bsz, max_n, dtype=torch.bool, device=device)
            return anchors, keep_mask

        valid = loss_mask[:, : max_anchor + 1] > 0.5
        valid_counts = valid.sum(dim=1)

        indices = torch.arange(max_anchor + 1, device=device).unsqueeze(0).expand(bsz, -1)
        masked_indices = torch.where(valid, indices, seq_len + 1)

        random_vals = torch.rand(bsz, max_anchor + 1, device=device)
        random_vals = torch.where(valid, random_vals, 2.0)

        _, sorted_idx = random_vals.sort(dim=1)
        gathered = torch.gather(masked_indices, 1, sorted_idx)

        # Take up to num_anchors slots; pad with zeros if fewer valid positions
        take_n = min(max_n, gathered.shape[1])
        selected = gathered[:, :take_n].sort(dim=1).values
        if take_n < max_n:
            pad = torch.zeros(bsz, max_n - take_n, dtype=torch.long, device=device)
            selected = torch.cat([selected, pad], dim=1)
        anchors = selected

        keep_mask = torch.arange(max_n, device=device).unsqueeze(0) < valid_counts.unsqueeze(
            1
        ).clamp(max=max_n)
        anchors = torch.where(keep_mask, anchors, 0)

        return anchors, keep_mask

    def _create_position_ids(
        self, anchor_positions: torch.Tensor, seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create position IDs for context and draft tokens."""
        bsz, n_blocks = anchor_positions.shape
        device = anchor_positions.device

        context_position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        offsets = torch.arange(self.block_size, device=device).view(1, 1, -1)
        draft_position_ids = anchor_positions.unsqueeze(-1) + offsets
        draft_position_ids = draft_position_ids.view(bsz, -1)

        return context_position_ids, draft_position_ids

    def _create_noise_embed(
        self,
        input_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Create noise embeddings: anchor token at block starts, MASK elsewhere.

        Matches SpecForge's OnlineDFlashModel._create_noise_embed().
        """
        bsz, seq_len = input_ids.shape
        n = anchor_positions.shape[1]
        bs = self.block_size
        device = input_ids.device

        noise_ids = torch.full(
            (bsz, n * bs), self.draft_model.mask_token_id, dtype=torch.long, device=device
        )

        block_starts = torch.arange(n, device=device) * bs
        block_starts = block_starts.unsqueeze(0).expand(bsz, -1)

        valid_anchor_positions = anchor_positions.clamp(0, seq_len - 1)
        anchor_tokens = torch.gather(input_ids, 1, valid_anchor_positions)

        flat_batch_idx = torch.arange(bsz, device=device).unsqueeze(1).expand(bsz, n)
        noise_ids[flat_batch_idx, block_starts] = torch.where(
            block_keep_mask,
            anchor_tokens,
            torch.tensor(self.draft_model.mask_token_id, dtype=torch.long, device=device),
        )

        return self.draft_model.embed_tokens(noise_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states_list: List[torch.Tensor],
        loss_mask: torch.Tensor,
        lm_head_weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full DFlash training forward pass.

        Matches SpecForge's OnlineDFlashModel.forward().

        Returns:
            loss: scalar training loss (objective-weighted)
            accuracy: scalar accuracy (binary mask, no decay)
            loss_per_position: [block_size] mean loss at each within-block position
                (index 0 is the anchor slot and always 0; indices 1..B-1 are the
                predicted tokens at 1..B-1 steps past the anchor)
            acc_per_position: [block_size] mean accuracy at each within-block position
            count_per_position: [block_size] valid label count at each within-block
                position before loss decay is applied
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        # 1. Extract context features from target hidden states
        context_feature = self.draft_model.extract_context_feature(hidden_states_list)

        # 2. Sample anchor positions with validity mask
        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len, loss_mask, device
        )
        n_blocks = anchor_positions.shape[1]

        # 3. Create noise embeddings (anchor token + MASK tokens)
        noise_embedding = self._create_noise_embed(input_ids, anchor_positions, block_keep_mask)

        # 4. Create position IDs
        context_position_ids, draft_position_ids = self._create_position_ids(
            anchor_positions, seq_len
        )

        # 5. Create block-causal attention mask
        draft_len = n_blocks * self.block_size
        kv_len = seq_len + draft_len

        block_mask = None
        if device.type == "cuda":
            mask_mod = _create_dflash_mask_mod(
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                ctx_len=seq_len,
                block_size=self.block_size,
            )
            block_mask = compile_friendly_create_block_mask(
                mask_mod=mask_mod,
                B=bsz,
                H=None,
                Q_LEN=draft_len,
                KV_LEN=kv_len,
                device=device,
            )

        # 6. Draft model forward — pass embeddings directly
        draft_hidden = self.draft_model(
            draft_input_ids=None,
            context_feature=context_feature,
            draft_position_ids=draft_position_ids,
            context_position_ids=context_position_ids,
            block_mask=block_mask,
            noise_embedding=noise_embedding,
        )

        # 7. Compute labels and weight mask (SpecForge pattern)
        # Labels: same-position prediction (position k predicts token at anchor+k)
        label_offsets = torch.arange(0, self.block_size, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets  # [B, n_blocks, block_size]
        valid_label_mask = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)

        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, n_blocks, -1),
            2,
            safe_label_indices,
        )  # [B, n_blocks, block_size]

        # Weight mask: block validity × bounds × exclude anchor (pos 0) × loss_mask
        weight_mask = block_keep_mask.unsqueeze(-1).expand(-1, -1, self.block_size).float()
        weight_mask = weight_mask * valid_label_mask.float()

        pos_in_block = torch.arange(self.block_size, device=device).view(1, 1, -1)
        weight_mask = weight_mask * (pos_in_block > 0).float()

        # Gather original loss_mask at label positions
        original_loss_mask_gathered = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, n_blocks, -1),
            2,
            safe_label_indices,
        )
        weight_mask = weight_mask * original_loss_mask_gathered

        # Capture binary mask BEFORE applying objective weights. Accuracy measures
        # "did we predict correctly?" uniformly across positions, while weighting
        # only shapes gradient contribution. SpecForge uses no decay at all;
        # our objective weighting is an addition to the training signal, not the metric.
        binary_eval_mask = weight_mask.view(-1)

        if self._can_use_liger_loss(device):
            return self._compute_liger_decay_loss(
                draft_hidden=draft_hidden,
                target_ids=target_ids,
                weight_mask=weight_mask,
                binary_eval_mask=binary_eval_mask,
                lm_head_weight=lm_head_weight,
            )

        return self._compute_torch_loss(
            draft_hidden=draft_hidden,
            target_ids=target_ids,
            weight_mask=weight_mask,
            binary_eval_mask=binary_eval_mask,
            lm_head_weight=lm_head_weight,
        )
