#!/usr/bin/env python3
"""Verify DFlash PyTorch CE and Liger fused-linear CE produce close results."""

import argparse
import copy
import sys

import torch

from torchspec.models.dflash import DFlashModel
from torchspec.models.draft.dflash import DFlashConfig, DFlashDraftModel
from torchspec.models.ops.liger import is_liger_available


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--intermediate-size", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--num-target-layers", type=int, default=2)
    parser.add_argument("--num-hidden-layers", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--num-anchors", type=int, default=8)
    parser.add_argument("--loss-decay-gamma", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--rtol", type=float, default=3e-2)
    parser.add_argument("--atol", type=float, default=3e-2)
    parser.add_argument("--grad-rtol", type=float, default=8e-2)
    parser.add_argument("--grad-atol", type=float, default=8e-2)
    parser.add_argument("--skip-backward", action="store_true")
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def make_config(args: argparse.Namespace) -> DFlashConfig:
    return DFlashConfig(
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_kv_heads,
        vocab_size=args.vocab_size,
        rms_norm_eps=1e-6,
        max_position_embeddings=max(512, args.seq_len + args.num_anchors * args.block_size + 32),
        rope_theta=10000.0,
        num_target_layers=args.num_target_layers,
        target_hidden_size=args.hidden_size,
        target_num_hidden_layers=12,
        mask_token_id=args.vocab_size - 1,
    )


def make_model(
    config: DFlashConfig,
    args: argparse.Namespace,
    *,
    dtype: torch.dtype,
    use_liger_kernel: bool,
) -> DFlashModel:
    draft = DFlashDraftModel(config).to(device="cuda", dtype=dtype)
    draft.freeze_embedding()
    model = DFlashModel(
        draft_model=draft,
        block_size=args.block_size,
        num_anchors=args.num_anchors,
        loss_objective="decay",
        loss_decay_gamma=args.loss_decay_gamma,
        use_liger_kernel=use_liger_kernel,
    )
    model.train()
    return model


def make_batch(args: argparse.Namespace, dtype: torch.dtype) -> dict[str, object]:
    input_ids = torch.randint(
        low=0,
        high=args.vocab_size - 1,
        size=(args.batch_size, args.seq_len),
        device="cuda",
    )
    loss_mask = torch.ones(args.batch_size, args.seq_len, device="cuda", dtype=torch.float32)
    loss_mask[:, : args.block_size] = 0.0

    hidden_states_list = [
        torch.randn(
            args.batch_size,
            args.seq_len,
            args.hidden_size,
            device="cuda",
            dtype=dtype,
        )
        for _ in range(args.num_target_layers)
    ]
    lm_head_weight = torch.randn(
        args.vocab_size,
        args.hidden_size,
        device="cuda",
        dtype=dtype,
    )
    return {
        "input_ids": input_ids,
        "loss_mask": loss_mask,
        "hidden_states_list": hidden_states_list,
        "lm_head_weight": lm_head_weight,
    }


def clone_batch(batch: dict[str, object]) -> dict[str, object]:
    return {
        "input_ids": batch["input_ids"].clone(),
        "loss_mask": batch["loss_mask"].clone(),
        "hidden_states_list": [x.clone() for x in batch["hidden_states_list"]],
        "lm_head_weight": batch["lm_head_weight"].clone(),
    }


def run_model(
    model: DFlashModel,
    batch: dict[str, object],
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    return model(
        input_ids=batch["input_ids"],
        hidden_states_list=batch["hidden_states_list"],
        loss_mask=batch["loss_mask"],
        lm_head_weight=batch["lm_head_weight"],
    )


def compare_tensor(name: str, actual: torch.Tensor, expected: torch.Tensor, rtol: float, atol: float):
    torch.testing.assert_close(
        actual.detach().float(),
        expected.detach().float(),
        rtol=rtol,
        atol=atol,
        msg=lambda msg: f"{name} mismatch\n{msg}",
    )
    max_abs = (actual.detach().float() - expected.detach().float()).abs().max().item()
    print(f"{name}: close (max_abs={max_abs:.6g})")


def grad_norms(model: DFlashModel) -> dict[str, torch.Tensor]:
    norms = {}
    for name, param in model.draft_model.named_parameters():
        if param.requires_grad and param.grad is not None:
            norms[name] = param.grad.detach().float().norm()
    return norms


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        print("CUDA is required for Liger fused linear cross entropy.", file=sys.stderr)
        return 2
    if not is_liger_available():
        print("liger_kernel is not importable. Install liger-kernel first.", file=sys.stderr)
        return 2

    dtype = resolve_dtype(args.dtype)
    torch.manual_seed(args.seed)
    config = make_config(args)
    torch_model = make_model(config, args, dtype=dtype, use_liger_kernel=False)
    liger_model = make_model(copy.deepcopy(config), args, dtype=dtype, use_liger_kernel=True)
    liger_model.load_state_dict(torch_model.state_dict())

    batch = make_batch(args, dtype)
    torch_batch = clone_batch(batch)
    liger_batch = clone_batch(batch)

    torch_out = run_model(torch_model, torch_batch, args.seed + 1)
    liger_out = run_model(liger_model, liger_batch, args.seed + 1)
    if liger_model._liger_fused_linear_ce is None:
        raise AssertionError("Liger fused linear CE was not instantiated during the forward pass.")
    names = ("loss", "accuracy", "loss_per_position", "acc_per_position", "count_per_position")

    print("PyTorch outputs:")
    for name, value in zip(names, torch_out):
        print(f"  {name}: {value.detach().float().cpu()}")
    print("Liger outputs:")
    for name, value in zip(names, liger_out):
        print(f"  {name}: {value.detach().float().cpu()}")

    for name, torch_value, liger_value in zip(names, torch_out, liger_out):
        compare_tensor(name, liger_value, torch_value, args.rtol, args.atol)

    if not args.skip_backward:
        torch_model.zero_grad(set_to_none=True)
        liger_model.zero_grad(set_to_none=True)
        torch_out[0].backward()
        liger_out[0].backward()
        torch_grads = grad_norms(torch_model)
        liger_grads = grad_norms(liger_model)
        if not liger_grads:
            raise AssertionError("Liger backward produced no draft-model gradients.")
        if torch_grads.keys() != liger_grads.keys():
            missing = sorted(set(torch_grads) ^ set(liger_grads))
            raise AssertionError(f"Gradient key mismatch: {missing}")
        for name in sorted(torch_grads):
            compare_tensor(
                f"grad_norm:{name}",
                liger_grads[name],
                torch_grads[name],
                args.grad_rtol,
                args.grad_atol,
            )

    print("DFlash PyTorch and Liger loss paths are close.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
