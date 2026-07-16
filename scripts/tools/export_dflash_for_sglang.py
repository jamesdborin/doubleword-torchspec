#!/usr/bin/env python3
"""Export a TorchSpec DFlash FSDP checkpoint for SGLang serving."""

import argparse
import json
import os
import sys
from pathlib import Path

from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.convert_to_hf import _detect_model_dir, _extract_model_weights, _load_fsdp_state_dict


_KEY_RENAMES = {
    "context_proj.weight": "fc.weight",
    "context_norm.weight": "hidden_norm.weight",
    "final_norm.weight": "norm.weight",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument("--mask-token", default="<|endoftext|>")
    args = parser.parse_args()

    state = _extract_model_weights(
        _load_fsdp_state_dict(_detect_model_dir(args.checkpoint_dir))
    )
    state.pop("embed_tokens.weight", None)  # SGLang reuses the target embedding and LM head.
    state = {_KEY_RENAMES.get(key, key): value.contiguous() for key, value in state.items()}

    with open(args.training_config) as handle:
        training_config = json.load(handle)

    config = dict(training_config)
    config["architectures"] = ["DFlashDraftModel"]
    # Use a Transformers-recognized config class, as in the reference DFlash
    # checkpoints. SGLang dispatches the serving implementation by architecture.
    config["model_type"] = "qwen3"
    config["head_dim"] = training_config["hidden_size"] // training_config["num_attention_heads"]
    config["torch_dtype"] = str(next(iter(state.values())).dtype).removeprefix("torch.")
    config["dflash_config"] = {
        "block_size": args.block_size,
        "num_target_layers": training_config["target_num_hidden_layers"],
        "target_layer_ids": training_config["target_layer_ids"],
        "mask_token": args.mask_token,
        "mask_token_id": training_config["mask_token_id"],
    }

    os.makedirs(args.output_dir, exist_ok=True)
    save_file(state, os.path.join(args.output_dir, "model.safetensors"))
    with open(os.path.join(args.output_dir, "config.json"), "w") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")
    print(f"Exported {len(state)} tensors to {args.output_dir}")


if __name__ == "__main__":
    main()
