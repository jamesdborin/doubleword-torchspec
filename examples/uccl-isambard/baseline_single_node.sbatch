#!/usr/bin/env bash
#SBATCH --job-name=ts-mc-baseline
#SBATCH --account=brics.s6p
#SBATCH --partition=workq
#SBATCH --reservation=brics_s6p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=ts-mc-baseline-%j.out
#SBATCH --error=ts-mc-baseline-%j.err

set -euo pipefail

: "${SCRATCH:=/scratch/s6p/${USER:?USER is required}}"
: "${SRC:=$SCRATCH/torchspec-uccl/baseline-a19cdf7c357e/source}"
: "${WORK:=$SCRATCH/torchspec-uccl/baseline-rerun}"
: "${IMAGE:=$SCRATCH/torchspec-qwen35-9b/images/torchspec-latest.sif}"
: "${HARNESS_SRC:=$SRC}"
CUDA12_LIB=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/targets/sbsa-linux/lib
HF_CACHE="$SCRATCH/huggingface"
RUN_DIR="$WORK/run-$SLURM_JOB_ID"
mkdir -p "$RUN_DIR" "$WORK/logs" "$HF_CACHE"

export APPTAINERENV_CUDA_VISIBLE_DEVICES=0
export APPTAINERENV_HF_HOME="$HF_CACHE"
export APPTAINERENV_HUGGINGFACE_HUB_CACHE="$HF_CACHE/hub"
export APPTAINERENV_TRANSFORMERS_CACHE="$HF_CACHE/hub"
export APPTAINERENV_HF_HUB_OFFLINE=1
export APPTAINERENV_PYTHONPATH="$SRC"
export APPTAINERENV_TMPDIR="/tmp/ts-baseline-$SLURM_JOB_ID"
export APPTAINERENV_LD_LIBRARY_PATH="$CUDA12_LIB:${LD_LIBRARY_PATH:-}"

apptainer exec --nv \
    --bind "$SRC:$SRC" \
    --bind "$HARNESS_SRC:$HARNESS_SRC" \
    --bind "$WORK:$WORK" \
    --bind "$HF_CACHE:$HF_CACHE" \
    --bind "$CUDA12_LIB:$CUDA12_LIB" \
    "$IMAGE" bash -lc '
        set -euo pipefail
        cd '"$SRC"'
        python3 -m pytest -q \
            tests/test_data_fetcher.py \
            tests/test_hf_runner.py \
            tests/test_mooncake_force_delete.py \
            tests/test_mooncake_master.py \
            tests/test_sgl_engine_decode.py \
            tests/test_save_config.py
        python3 '"$HARNESS_SRC"'/tools/test_sglang_engine_patch.py \
            --model-path Qwen/Qwen2.5-0.5B-Instruct \
            --aux-layer-ids 2 12 21
    '
