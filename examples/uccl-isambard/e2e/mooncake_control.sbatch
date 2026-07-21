#!/usr/bin/env bash
#SBATCH --job-name=ts-mc-e2e10
#SBATCH --account=brics.s6p
#SBATCH --partition=workq
#SBATCH --reservation=brics_s6p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/s6p/%u/torchspec-uccl/e2e-logs/mooncake-%j.out
#SBATCH --error=/scratch/s6p/%u/torchspec-uccl/e2e-logs/mooncake-%j.err

set -euo pipefail

: "${SCRATCH:=/scratch/s6p/${USER:?USER is required}}"
: "${SRC:=$SCRATCH/torchspec-uccl/integration-20260720/source}"
: "${MODEL_WORK:=$SCRATCH/torchspec-qwen35-9b}"
: "${IMAGE:=$MODEL_WORK/images/torchspec-latest.sif}"
: "${RUN_ROOT:=$SCRATCH/torchspec-uccl/e2e-runs}"

RUN_DIR="$RUN_ROOT/mooncake-${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
SGLANG_QWEN3_VL=/sgl-workspace/sglang/python/sglang/srt/models/qwen3_vl.py
QWEN_PATCH="$SRC/examples/qwen35-9b-isambard/patches/qwen3_vl-qwen35-aux-capture.patch"
PATCHED_QWEN3_VL="$RUN_DIR/qwen3_vl.py"
CUDA12_LIB=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/targets/sbsa-linux/lib

[[ -s "$IMAGE" ]] || { echo "Missing image: $IMAGE" >&2; exit 1; }
[[ -s "$QWEN_PATCH" ]] || { echo "Missing Qwen3.5 patch: $QWEN_PATCH" >&2; exit 1; }
[[ -d "$CUDA12_LIB" ]] || { echo "Missing CUDA 12 runtime: $CUDA12_LIB" >&2; exit 1; }

mkdir -p \
    "$RUN_DIR" \
    "$RUN_DIR/output" \
    "$RUN_DIR/cache" \
    "$RUN_DIR/torchinductor" \
    "$MODEL_WORK/container-pydeps" \
    "$MODEL_WORK/hf-cache"

# The cached image predates the narrow Qwen3.5 auxiliary-state fix. Keep the
# image immutable and bind the corrected model wrapper over the installed file.
apptainer exec "$IMAGE" cat "$SGLANG_QWEN3_VL" > "$PATCHED_QWEN3_VL"
patch --silent "$PATCHED_QWEN3_VL" < "$QWEN_PATCH"

export CUDA_VISIBLE_DEVICES=0,1
export HF_HOME="$MODEL_WORK/hf-cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME="$HF_HOME/torch"
export TORCHINDUCTOR_CACHE_DIR="$RUN_DIR/torchinductor"
export TORCHSPEC_LOG_DIR="$RUN_DIR/logs"
export TORCHSPEC_LOG_LEVEL=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MC_STORE_MEMCPY=0
export TMPDIR="/tmp/ts-mc-${SLURM_JOB_ID}"
export RAY_TMPDIR="/tmp/ray-mc-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$RAY_TMPDIR" "$TORCHSPEC_LOG_DIR"

export APPTAINERENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
export APPTAINERENV_HF_HOME="$HF_HOME"
export APPTAINERENV_HF_DATASETS_CACHE="$HF_DATASETS_CACHE"
export APPTAINERENV_TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE"
export APPTAINERENV_TORCH_HOME="$TORCH_HOME"
export APPTAINERENV_TORCHINDUCTOR_CACHE_DIR="$TORCHINDUCTOR_CACHE_DIR"
export APPTAINERENV_TORCHSPEC_LOG_DIR="$TORCHSPEC_LOG_DIR"
export APPTAINERENV_TORCHSPEC_LOG_LEVEL="$TORCHSPEC_LOG_LEVEL"
export APPTAINERENV_PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"
export APPTAINERENV_MC_STORE_MEMCPY="$MC_STORE_MEMCPY"
export APPTAINERENV_TMPDIR="$TMPDIR"
export APPTAINERENV_RAY_TMPDIR="$RAY_TMPDIR"
export APPTAINERENV_PYTHONPATH="/workspace/TorchSpec:$MODEL_WORK/container-pydeps"
export APPTAINERENV_LD_LIBRARY_PATH="$CUDA12_LIB"

apptainer exec --cleanenv --nv \
    --bind "$SRC:/workspace/TorchSpec" \
    --bind "$MODEL_WORK:$MODEL_WORK" \
    --bind "$RUN_DIR:$RUN_DIR" \
    --bind "$CUDA12_LIB:$CUDA12_LIB" \
    --bind "$PATCHED_QWEN3_VL:$SGLANG_QWEN3_VL" \
    "$IMAGE" \
    bash -lc '
        set -euo pipefail
        cd /workspace/TorchSpec
        python3 -m torchspec.train_entry \
          --config examples/uccl-isambard/e2e/mooncake_control.yaml \
          output_dir='"$RUN_DIR"'/output \
          cache_dir='"$RUN_DIR"'/cache \
          model_download_dir='"$MODEL_WORK"'/hf-cache
    '

echo "MOONCAKE_E2E_10_STEPS_PASSED output=$RUN_DIR/output"
