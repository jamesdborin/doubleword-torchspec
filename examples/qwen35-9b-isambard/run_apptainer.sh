#!/usr/bin/env bash

set -euo pipefail

SRC="${SRC:-$SCRATCH/torchspec/torchspec}"
WORK="${WORK:-$SCRATCH/torchspec-qwen35-9b}"
IMAGE="${IMAGE:-$WORK/images/torchspec-latest.sif}"
WANDB_ENV_FILE="${WANDB_ENV_FILE:-$WORK/secrets/wandb.env}"

[[ -s "$IMAGE" ]] || { echo "Missing image: $IMAGE" >&2; exit 1; }
[[ -s "$WANDB_ENV_FILE" ]] || { echo "Missing W&B environment file: $WANDB_ENV_FILE" >&2; exit 1; }

mkdir -p "$WORK"/{apptainer-cache,apptainer-tmp,cache,checkpoints,container-pydeps,hf-cache,logs,outputs,tmp,torchinductor}

set -a
# shellcheck disable=SC1090
source "$WANDB_ENV_FILE"
set +a

export APPTAINER_CACHEDIR="$WORK/apptainer-cache"
export APPTAINER_TMPDIR="$WORK/apptainer-tmp"
export HF_HOME="$WORK/hf-cache"
export HF_DATASETS_CACHE="$WORK/hf-cache/datasets"
export TRANSFORMERS_CACHE="$WORK/hf-cache/transformers"
export TORCH_HOME="$WORK/hf-cache/torch"
export TORCHINDUCTOR_CACHE_DIR="$WORK/torchinductor"
export TORCHSPEC_LOG_DIR="$WORK/logs"
export TORCHSPEC_LOG_LEVEL="${TORCHSPEC_LOG_LEVEL:-INFO}"
export MC_STORE_MEMCPY="${MC_STORE_MEMCPY:-0}"

# Ray Unix sockets have a 107-byte path limit, so keep these paths node-local and short.
export TMPDIR="/tmp/ts-${SLURM_JOB_ID:-manual}"
export RAY_TMPDIR="/tmp/ray-${SLURM_JOB_ID:-manual}"
mkdir -p "$TMPDIR" "$RAY_TMPDIR"

# mooncake-transfer-engine in the image requires the CUDA 12 runtime available on Isambard.
CUDA12_LIB="${CUDA12_LIB:-/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/targets/sbsa-linux/lib}"

export APPTAINERENV_HF_HOME="$HF_HOME"
export APPTAINERENV_HF_DATASETS_CACHE="$HF_DATASETS_CACHE"
export APPTAINERENV_TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE"
export APPTAINERENV_TORCH_HOME="$TORCH_HOME"
export APPTAINERENV_TORCHINDUCTOR_CACHE_DIR="$TORCHINDUCTOR_CACHE_DIR"
export APPTAINERENV_TORCHSPEC_LOG_DIR="$TORCHSPEC_LOG_DIR"
export APPTAINERENV_TORCHSPEC_LOG_LEVEL="$TORCHSPEC_LOG_LEVEL"
export APPTAINERENV_TMPDIR="$TMPDIR"
export APPTAINERENV_RAY_TMPDIR="$RAY_TMPDIR"
export APPTAINERENV_WANDB_API_KEY="$WANDB_API_KEY"
export APPTAINERENV_PYTHONPATH="/workspace/TorchSpec:$WORK/container-pydeps:${PYTHONPATH:-}"
export APPTAINERENV_LD_LIBRARY_PATH="$CUDA12_LIB:${LD_LIBRARY_PATH:-}"

apptainer exec --nv \
    --bind "$SRC:/workspace/TorchSpec" \
    --bind "$WORK:$WORK" \
    --bind "$CUDA12_LIB:$CUDA12_LIB" \
    "$IMAGE" \
    bash -lc '
        set -euo pipefail
        cd /workspace/TorchSpec
        python3 -m torchspec.train_entry \
          --config configs/sglang_qwen35_9b_dflash_ultrachat_magpie_2gpu.yaml \
          training.micro_batch_size=8 \
          training.draft_accumulation_steps=1 \
          training.dflash_num_anchors=256 \
          training.use_liger_kernel=true \
          inference.max_sample_pool_size=32 \
          inference.inference_buffer_threshold=16 \
          output_dir='"$WORK"'/outputs/qwen35-9b-dflash-ultrachat-liger-a256-b8-a1 \
          cache_dir='"$WORK"'/cache/qwen35-9b-dflash-ultrachat-liger-a256-b16 \
          model_download_dir='"$WORK"'/hf-cache
    '
