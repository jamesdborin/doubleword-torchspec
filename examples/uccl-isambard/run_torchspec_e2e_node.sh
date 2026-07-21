#!/usr/bin/env bash

# One task per physical node.  This script is launched by
# torchspec_e2e_two_node.sbatch and keeps one persistent podman-hpc container
# alive per node while Ray and TorchSpec run inside it.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=podman_common.sh
source "$SCRIPT_DIR/podman_common.sh"

RUN_DIR="${1:?run directory is required}"
CONFIG_FILE="${2:?config file is required}"
DATASET_FILE="${3:?dataset file is required}"
NUM_STEPS="${4:-10}"
: "${UCCL_ROOTFS:?extracted container rootfs is required}"

[[ "${SLURM_NNODES:-0}" == 2 ]] || { echo "expected exactly two nodes" >&2; exit 1; }
[[ "${SLURM_NTASKS:-0}" == 2 ]] || { echo "expected exactly two tasks" >&2; exit 1; }
[[ "${SLURM_LOCALID:-0}" == 0 ]] || { echo "expected one task per node" >&2; exit 1; }
[[ -s "$CONFIG_FILE" ]] || { echo "missing config: $CONFIG_FILE" >&2; exit 1; }
[[ -s "$DATASET_FILE" ]] || { echo "missing dataset: $DATASET_FILE" >&2; exit 1; }

mkdir -p "$RUN_DIR" "$RUN_DIR/logs" "$RUN_DIR/ray"
export XDG_RUNTIME_DIR="/tmp/torchspec-podman-${UID}-${SLURM_JOB_ID}"
export UCCL_PODMAN_PID_MODE=private
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
uccl_heal_stale_podman_state
UCCL_PODMAN_EXTRA_MOUNTS=(--tmpfs "/tmp:rw,size=64g,mode=1777")
LOCAL_IP="$(uccl_hsn_address)"
printf '%s\n' "$LOCAL_IP" >"$RUN_DIR/node-${SLURM_NODEID}.ip"

# Keep all distributed control/data-plane choices on Slingshot.  The aws-ofi
# plugin is matched to NCCL 2.28.9 in the TorchSpec CUDA 13 image.
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_NET="AWS Libfabric"
export NCCL_SOCKET_IFNAME=hsn
export GLOO_SOCKET_IFNAME=hsn0
export TP_SOCKET_IFNAME=hsn0
export UCCL_P2P_TRANSPORT=cxi
export UCCL_P2P_DISABLE_IPC=1
export UCCL_P2P_USE_GPU_DIRECT=0
export UCCL_SOCKET_IFNAME=hsn0
export UCCL_CXI_DEVICE_INDEX=0
export TORCHSPEC_TRANSFER_BACKEND=uccl-p2p
export MODEL_WORK="${MODEL_WORK:-$SCRATCH/torchspec-qwen35-9b}"
export HF_HOME="${HF_HOME:-$MODEL_WORK/hf-cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TORCHSPEC_LOG_DIR="$RUN_DIR/logs"
export TORCHSPEC_LOG_LEVEL="${TORCHSPEC_LOG_LEVEL:-INFO}"
export SGLANG_DISABLE_CUDNN_CHECK=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export RAY_GCS_PORT="${RAY_GCS_PORT:-$((20000 + SLURM_JOB_ID % 10000))}"
export RAY_CLIENT_PORT="$((RAY_GCS_PORT + 1))"
export RAY_DASHBOARD_PORT="$((RAY_GCS_PORT + 2))"
export RAY_NODE_MANAGER_PORT="$((RAY_GCS_PORT + 3))"
export RAY_OBJECT_MANAGER_PORT="$((RAY_GCS_PORT + 4))"
export RAY_MIN_WORKER_PORT="$((RAY_GCS_PORT + 100))"
export RAY_MAX_WORKER_PORT="$((RAY_GCS_PORT + 1099))"
export E2E_RUN_DIR="$RUN_DIR"
export E2E_CONFIG_FILE="$CONFIG_FILE"
export E2E_DATASET_FILE="$DATASET_FILE"
export E2E_NUM_STEPS="$NUM_STEPS"
export E2E_LOCAL_IP="$LOCAL_IP"
export E2E_NODE_ID="$SLURM_NODEID"
export PYTHONPATH="/workspace/TorchSpec:$UCCL_PYDEPS:$MODEL_WORK/container-pydeps"
export UCCL_PODMAN_FORWARD_ENV="${UCCL_PODMAN_FORWARD_ENV:-} PYTHONPATH HF_HOME HUGGINGFACE_HUB_CACHE TORCHSPEC_TRANSFER_BACKEND TORCHSPEC_LOG_DIR TORCHSPEC_LOG_LEVEL SGLANG_DISABLE_CUDNN_CHECK SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN GLOO_SOCKET_IFNAME TP_SOCKET_IFNAME E2E_RUN_DIR E2E_CONFIG_FILE E2E_DATASET_FILE E2E_NUM_STEPS E2E_LOCAL_IP E2E_NODE_ID"

CONTAINER_NAME="$(uccl_podman_name e2e-rootfs)"
uccl_podman_rootfs_run "$CONTAINER_NAME" "$UCCL_ROOTFS" bash \
    /workspace/TorchSpec/examples/uccl-isambard/run_torchspec_e2e_container.sh
