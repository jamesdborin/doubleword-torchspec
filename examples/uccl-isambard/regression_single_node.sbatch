#!/usr/bin/env bash
#SBATCH --job-name=ts-xfer-regress
#SBATCH --account=brics.s6p
#SBATCH --partition=workq
#SBATCH --reservation=brics_s6p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=ts-xfer-regress-%j.out
#SBATCH --error=ts-xfer-regress-%j.err

set -euo pipefail
module load cuda/12.6
: "${SCRATCH:=/scratch/s6p/${USER:?USER is required}}"
: "${SRC:=$SCRATCH/torchspec/torchspec}"
: "${WORK:=$SCRATCH/torchspec-uccl}"
: "${IMAGE:=$SCRATCH/torchspec-qwen35-9b/images/torchspec-latest.sif}"
RUN_DIR="$WORK/regression-${SLURM_JOB_ID:?}"
HF_CACHE="$SCRATCH/huggingface"
CUDA12_LIB=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/targets/sbsa-linux/lib
mkdir -p "$RUN_DIR" "$HF_CACHE"

export APPTAINERENV_CUDA_VISIBLE_DEVICES=0
export APPTAINERENV_HF_HOME="$HF_CACHE"
export APPTAINERENV_HUGGINGFACE_HUB_CACHE="$HF_CACHE/hub"
export APPTAINERENV_TRANSFORMERS_CACHE="$HF_CACHE/hub"
export APPTAINERENV_HF_HUB_OFFLINE=1
export APPTAINERENV_TMPDIR="/tmp/ts-regression-$SLURM_JOB_ID"
export APPTAINERENV_LD_LIBRARY_PATH="$CUDA12_LIB:${LD_LIBRARY_PATH:-}"

apptainer exec --nv \
    --bind "$SRC:/workspace/TorchSpec" \
    --bind "$RUN_DIR:$RUN_DIR" \
    --bind "$HF_CACHE:$HF_CACHE" \
    --bind "$CUDA12_LIB:$CUDA12_LIB" \
    "$IMAGE" bash -lc '
        set -euo pipefail
        SGL_COPY='"$RUN_DIR"'/sglang
        cp -a /sgl-workspace/sglang "$SGL_COPY"
        cd "$SGL_COPY"
        git apply /workspace/TorchSpec/patches/sglang/v0.5.12/transfer_backend.patch
        export PYTHONPATH=/workspace/TorchSpec:$SGL_COPY/python:${PYTHONPATH:-}
        cd /workspace/TorchSpec
        CUDA_VISIBLE_DEVICES= python3 -m pytest -q \
            tests/test_transfer_base.py \
            tests/test_transfer_migration.py \
            tests/test_uccl_backend.py \
            tests/test_data_fetcher.py \
            tests/test_hf_runner.py \
            tests/test_hf_engine.py \
            tests/test_sgl_engine_decode.py \
            tests/test_save_config.py \
            tests/test_mooncake_force_delete.py
        python3 tools/test_sglang_engine_patch.py \
            --model-path Qwen/Qwen2.5-0.5B-Instruct \
            --aux-layer-ids 2 12 21
    '
