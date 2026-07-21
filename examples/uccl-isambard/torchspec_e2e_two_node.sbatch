#!/usr/bin/env bash
#SBATCH --job-name=ts-uccl-e2e-2n
#SBATCH --account=brics.s6p
#SBATCH --partition=workq
#SBATCH --reservation=brics_s6p
#SBATCH --nodes=2
# nid010318 failed earlier container-storage diagnostics and nid010762 is in
# maintenance. The prefix-ordered PID-limit fix is used on both remaining
# reservation nodes.
#SBATCH --exclude=nid010318,nid010762
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/s6p/%u/torchspec-uccl/e2e-slurm-%j.out
#SBATCH --error=/scratch/s6p/%u/torchspec-uccl/e2e-slurm-%j.err

set -euo pipefail

: "${SCRATCH:=/scratch/s6p/${USER:?USER is required}}"
: "${SRC:=$SCRATCH/torchspec-uccl/integration-20260720/source}"
: "${WORK:=$SCRATCH/torchspec-uccl}"
: "${MODEL_WORK:=$SCRATCH/torchspec-qwen35-9b}"
: "${PODMANHPC_SQUASH_DIR:=/projects/s6p/${USER:?USER is required}/podman-store}"
: "${UCCL_ROOTFS:=/lus/lfs1aip2/projects/s6p/${USER:?USER is required}/torchspec-uccl/rootfs/torchspec-latest}"
: "${CONFIG_FILE:=$SRC/examples/uccl-isambard/e2e/uccl_tp4_fsdp4.yaml}"
: "${DATASET_FILE:=$SRC/examples/uccl-isambard/e2e/repeated_conversation_40.jsonl}"
: "${NUM_STEPS:=10}"
: "${OFI_PLUGIN_DIR:=/tools/brics/apps/linux-neoverse_v2/aws-ofi-nccl-1.18.0-sji722yevkw6isblkxnqjod7nseadbch/lib}"

RUN_DIR="$WORK/e2e-uccl-${SLURM_JOB_ID:?}"
mkdir -p "$RUN_DIR"
[[ -d "$OFI_PLUGIN_DIR" ]] || { echo "missing aws-ofi-nccl plugin: $OFI_PLUGIN_DIR" >&2; exit 1; }

# The reusable Podman wrapper supplies CXI, CUDA compatibility, UCCL, and the
# matched aws-ofi-nccl plugin required by TP/FSDP collectives.
export OFI_PLUGIN_DIR
export SRC WORK MODEL_WORK PODMANHPC_SQUASH_DIR UCCL_ROOTFS CONFIG_FILE DATASET_FILE NUM_STEPS

# Stage once from the SIF before launching concurrent per-node containers.
# shellcheck source=podman_common.sh
source "$SRC/examples/uccl-isambard/podman_common.sh"
if [[ ! -s "$PODMAN_CUDA_COMPAT/libcuda.so.1" ]]; then
    uccl_stage_podman_cuda_compat
fi
[[ -x "$UCCL_ROOTFS/bin/bash" ]] || { echo "missing extracted rootfs: $UCCL_ROOTFS" >&2; exit 1; }

echo "run_dir=$RUN_DIR nodes=$SLURM_JOB_NODELIST steps=$NUM_STEPS"
srun --mpi=none --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" \
    --ntasks-per-node=1 --kill-on-bad-exit=1 \
    "$SRC/examples/uccl-isambard/run_torchspec_e2e_node.sh" \
    "$RUN_DIR" "$CONFIG_FILE" "$DATASET_FILE" "$NUM_STEPS"
[[ "$(<"$RUN_DIR/driver.status")" == 0 ]] || { echo "TorchSpec driver failed" >&2; exit 1; }
[[ -s "$RUN_DIR/train.log" ]] || { echo "missing train log" >&2; exit 1; }
[[ -s "$RUN_DIR/metrics.json" ]] || { echo "missing metrics" >&2; exit 1; }
[[ -s "$RUN_DIR/complete.marker" ]] || { echo "missing completion marker" >&2; exit 1; }
