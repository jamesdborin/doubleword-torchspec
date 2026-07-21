#!/usr/bin/env bash
#SBATCH --job-name=ts-uccl-e2e-1n
#SBATCH --account=brics.s6p
#SBATCH --partition=workq
#SBATCH --reservation=brics_s6p
#SBATCH --nodes=1
# nid010318 has repeatedly returned SIGBUS in container storage diagnostics.
# The extracted-rootfs path avoids the immutable squash image, but retain the
# exclusion until that node is independently requalified.
#SBATCH --exclude=nid010318
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --network=single_node_vni
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/s6p/%u/torchspec-uccl/e2e-logs/uccl-1n-%j.out
#SBATCH --error=/scratch/s6p/%u/torchspec-uccl/e2e-logs/uccl-1n-%j.err

set -euo pipefail

: "${SCRATCH:=/scratch/s6p/${USER:?USER is required}}"
: "${SRC:=$SCRATCH/torchspec-uccl/integration-20260720/source}"
: "${WORK:=$SCRATCH/torchspec-uccl}"
: "${MODEL_WORK:=$SCRATCH/torchspec-qwen35-9b}"
: "${PODMANHPC_SQUASH_DIR:=/projects/s6p/${USER:?USER is required}/podman-store}"
# runc rejects rootfs paths containing symlink components; /projects resolves
# to this canonical Lustre path.
: "${UCCL_ROOTFS:=/lus/lfs1aip2/projects/s6p/${USER:?USER is required}/torchspec-uccl/rootfs/torchspec-latest}"

RUN_DIR="$WORK/e2e-runs/uccl-single-${SLURM_JOB_ID:?}"
CONFIG_FILE="$SRC/examples/uccl-isambard/e2e/uccl_tp4_fsdp4.yaml"
DATASET_FILE="$SRC/examples/uccl-isambard/e2e/repeated_conversation_40.jsonl"
mkdir -p "$RUN_DIR" "$RUN_DIR/ray-live" "$WORK/e2e-logs" \
    "$MODEL_WORK/hf-cache" "$MODEL_WORK/container-pydeps"

# Rootless Podman must not fall back to the site-owned /run/user directory.
export XDG_RUNTIME_DIR="/tmp/torchspec-podman-${UID}-${SLURM_JOB_ID}"
export UCCL_PODMAN_PID_MODE=private
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

# shellcheck source=../podman_common.sh
source "$SRC/examples/uccl-isambard/podman_common.sh"
uccl_heal_stale_podman_state
export E2E_LOCAL_IP="$(uccl_hsn_address)"
[[ -x "$UCCL_ROOTFS/bin/bash" ]] || { echo "missing extracted rootfs: $UCCL_ROOTFS" >&2; exit 1; }

export SRC WORK MODEL_WORK PODMANHPC_SQUASH_DIR
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_SOCKET_IFNAME=hsn
export GLOO_SOCKET_IFNAME=hsn0
export TP_SOCKET_IFNAME=hsn0
export UCCL_P2P_TRANSPORT=cxi
export UCCL_P2P_DISABLE_IPC=1
export UCCL_P2P_USE_GPU_DIRECT=0
export UCCL_SOCKET_IFNAME=hsn0
export UCCL_CXI_DEVICE_INDEX=0
export TORCHSPEC_TRANSFER_BACKEND=uccl-p2p
export HF_HOME="$MODEL_WORK/hf-cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TORCHSPEC_LOG_DIR="$RUN_DIR/logs"
export TORCHSPEC_LOG_LEVEL=INFO
export SGLANG_DISABLE_CUDNN_CHECK=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export E2E_RUN_DIR="$RUN_DIR"
export E2E_CONFIG_FILE="$CONFIG_FILE"
export E2E_DATASET_FILE="$DATASET_FILE"
export PYTHONPATH="/workspace/TorchSpec:$WORK/uccl-pydeps:$MODEL_WORK/container-pydeps"
export UCCL_PODMAN_FORWARD_ENV="PYTHONPATH HF_HOME HUGGINGFACE_HUB_CACHE TORCHSPEC_TRANSFER_BACKEND TORCHSPEC_LOG_DIR TORCHSPEC_LOG_LEVEL SGLANG_DISABLE_CUDNN_CHECK SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN GLOO_SOCKET_IFNAME TP_SOCKET_IFNAME E2E_RUN_DIR E2E_CONFIG_FILE E2E_DATASET_FILE E2E_LOCAL_IP"

# Rebuild into the canonical pydeps target so this run cannot accidentally use
# the pre-fix UCCL artifact that produced fi_getinfo(ENODATA).
if [[ "${SKIP_UCCL_INSTALL:-0}" != 1 ]]; then
    JOBS="$SLURM_CPUS_PER_TASK" "$SRC/examples/uccl-isambard/install_uccl.sh"
fi

UCCL_PODMAN_EXTRA_MOUNTS=(
    --tmpfs "/tmp:rw,size=64g,mode=1777"
    -v "$RUN_DIR/ray-live:/tmp/ray-live"
)
if [[ ! -s "$PODMAN_CUDA_COMPAT/libcuda.so.1" ]]; then
    uccl_stage_podman_cuda_compat
fi

export UCCL_PODMAN_USE_INIT=0
export UCCL_PODMAN_CGROUPS=
CONTAINER_NAME="$(uccl_podman_name single-e2e-rootfs-setsid)"
uccl_podman_rootfs_run "$CONTAINER_NAME" "$UCCL_ROOTFS" \
    bash -x /workspace/TorchSpec/examples/uccl-isambard/e2e/run_catatonit_container.sh \
    2>&1 | tee "$RUN_DIR/container.log"
[[ -s "$RUN_DIR/train.log" ]] || { echo "missing train log after container exec" >&2; exit 1; }
[[ -s "$RUN_DIR/metrics.json" ]] || { echo "missing metrics after container exec" >&2; exit 1; }
[[ -s "$RUN_DIR/complete.marker" ]] || { echo "missing completion marker" >&2; exit 1; }
