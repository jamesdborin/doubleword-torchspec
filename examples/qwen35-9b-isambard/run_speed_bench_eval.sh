#!/usr/bin/env bash

set -euo pipefail

SRC="${SRC:-$SCRATCH/torchspec/torchspec}"
WORK="${WORK:-$SCRATCH/torchspec-qwen35-9b}"
IMAGE="${IMAGE:-$WORK/images/torchspec-latest.sif}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-9B}"
CHECKPOINT="${CHECKPOINT:-$WORK/outputs/qwen35-9b-dflash-ultrachat-liger-a256-full-fsdp-mbs22-gbs66/checkpoints/iter_0004001}"
DRAFT_MODEL="${DRAFT_MODEL:-}"
NUM_DRAFT_TOKENS="${NUM_DRAFT_TOKENS:-8}"
RUN_SUFFIX="${RUN_SUFFIX:-iter4001-speed-bench-qualitative}"
OUTPUT_DIR="$WORK/outputs/eval-$RUN_SUFFIX"
DRAFT_DIR="${DRAFT_MODEL:-$OUTPUT_DIR/draft-model}"
RESULTS="$OUTPUT_DIR/results.json"
PORT="${PORT:-30000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
CONCURRENCY="${CONCURRENCY:-32}"
LIMIT="${LIMIT:-}"

SGLANG_DFLASH_INFO="/sgl-workspace/sglang/python/sglang/srt/speculative/dflash_info.py"
PATCH_FILE="$SRC/examples/qwen35-9b-isambard/patches/dflash-acceptance-histogram.patch"
PATCHED_DFLASH_INFO="$WORK/sglang-patches/dflash_info-${SLURM_JOB_ID:-manual}.py"
CUDA12_LIB="${CUDA12_LIB:-/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/targets/sbsa-linux/lib}"

[[ -s "$IMAGE" ]] || { echo "Missing image: $IMAGE" >&2; exit 1; }
if [[ -z "$DRAFT_MODEL" ]]; then
    [[ -d "$CHECKPOINT/model" ]] || { echo "Missing checkpoint: $CHECKPOINT" >&2; exit 1; }
fi
mkdir -p "$OUTPUT_DIR" "$WORK"/{apptainer-cache,hf-cache,logs,sglang-patches,torchinductor}

export APPTAINER_CACHEDIR="$WORK/apptainer-cache"
export APPTAINER_TMPDIR="/tmp/apptainer-${SLURM_JOB_ID:-manual}"
export TMPDIR="/tmp/ts-${SLURM_JOB_ID:-manual}"
mkdir -p "$APPTAINER_TMPDIR" "$TMPDIR"

apptainer exec "$IMAGE" cat "$SGLANG_DFLASH_INFO" > "$PATCHED_DFLASH_INFO"
patch --silent "$PATCHED_DFLASH_INFO" < "$PATCH_FILE"

if [[ -z "$DRAFT_MODEL" && ! -s "$DRAFT_DIR/model.safetensors" ]]; then
    apptainer exec \
        --bind "$SRC:/workspace/TorchSpec" \
        --bind "$WORK:$WORK" \
        "$IMAGE" bash -lc '
            cd /workspace/TorchSpec
            python3 scripts/tools/export_dflash_for_sglang.py \
              --checkpoint-dir '"$CHECKPOINT"' \
              --training-config torchspec/config/dflash_draft_config_qwen35_9b_ultrachat_magpie.json \
              --output-dir '"$DRAFT_DIR"' \
              --block-size 8
        '
fi

export HF_HOME="$WORK/hf-cache"
export HF_DATASETS_CACHE="$WORK/hf-cache/datasets"
export TORCHINDUCTOR_CACHE_DIR="$WORK/torchinductor"
export APPTAINERENV_HF_HOME="$HF_HOME"
export APPTAINERENV_HF_DATASETS_CACHE="$HF_DATASETS_CACHE"
export APPTAINERENV_TORCHINDUCTOR_CACHE_DIR="$TORCHINDUCTOR_CACHE_DIR"
export APPTAINERENV_TMPDIR="$TMPDIR"
export APPTAINERENV_PYTHONPATH="/workspace/TorchSpec:$WORK/container-pydeps:${PYTHONPATH:-}"
export APPTAINERENV_LD_LIBRARY_PATH="$CUDA12_LIB:${LD_LIBRARY_PATH:-}"
export APPTAINERENV_CUDA_VISIBLE_DEVICES="0,1,2,3"

LIMIT_ARG="${LIMIT:+--limit $LIMIT}"
apptainer exec --nv \
    --bind "$SRC:/workspace/TorchSpec" \
    --bind "$WORK:$WORK" \
    --bind "$CUDA12_LIB:$CUDA12_LIB" \
    --bind "$PATCHED_DFLASH_INFO:$SGLANG_DFLASH_INFO" \
    "$IMAGE" bash -lc '
        set -euo pipefail
        cd /workspace/TorchSpec
        python3 -m sglang.launch_server \
          --model-path '"$BASE_MODEL"' \
          --speculative-algorithm DFLASH \
          --speculative-draft-model-path '"$DRAFT_DIR"' \
          --speculative-num-draft-tokens '"$NUM_DRAFT_TOKENS"' \
          --tp-size 4 \
          --attention-backend flashinfer \
          --speculative-draft-attention-backend flashinfer \
          --disable-radix-cache \
          --mem-fraction-static 0.7 \
          --trust-remote-code \
          --host 127.0.0.1 --port '"$PORT"' \
          > '"$OUTPUT_DIR"'/server.log 2>&1 &
        server_pid=$!
        trap "kill $server_pid 2>/dev/null || true" EXIT
        python3 scripts/tools/evaluate_speed_bench_sglang.py \
          --base-url http://127.0.0.1:'"$PORT"' \
          --model '"$BASE_MODEL"' \
          --output '"$RESULTS"' \
          --hf-cache '"$HF_HOME"' \
          --max-new-tokens '"$MAX_NEW_TOKENS"' \
          --num-draft-tokens '"$NUM_DRAFT_TOKENS"' \
          --concurrency '"$CONCURRENCY"' \
          '"$LIMIT_ARG"'
    '
