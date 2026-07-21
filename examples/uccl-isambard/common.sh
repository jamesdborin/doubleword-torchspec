#!/usr/bin/env bash

# Shared Isambard/Apptainer settings. Source this file; do not execute it.

set -euo pipefail

: "${SCRATCH:=/scratch/s6p/${USER:?USER is required}}"
: "${SRC:=$SCRATCH/torchspec/torchspec}"
: "${WORK:=$SCRATCH/torchspec-uccl}"
: "${IMAGE:=$SCRATCH/torchspec-qwen35-9b/images/torchspec-latest.sif}"
: "${UCCL_PYDEPS:=$WORK/uccl-pydeps}"
: "${CRAY_LIBFABRIC:=/opt/cray/libfabric/1.22.0}"
: "${CONTAINER_CUDA_HOME:=/usr/local/cuda}"
: "${CONTAINER_CUDA_LIB:=$CONTAINER_CUDA_HOME/targets/sbsa-linux/lib}"
: "${CONTAINER_CUDA_COMPAT:=$CONTAINER_CUDA_HOME/compat}"

uccl_hsn_address() {
    python3 - "${UCCL_SOCKET_IFNAME:-hsn0}" <<'PY'
import fcntl
import socket
import struct
import sys

name = sys.argv[1].encode("ascii")[:15]
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
    packed = fcntl.ioctl(sock.fileno(), 0x8915, struct.pack("256s", name))
print(socket.inet_ntoa(packed[20:24]))
PY
}

uccl_apptainer() {
    [[ -s "$IMAGE" ]] || { echo "Missing cached SIF: $IMAGE" >&2; return 1; }
    [[ -d "$SRC" ]] || { echo "Missing TorchSpec source: $SRC" >&2; return 1; }
    [[ -d "$UCCL_PYDEPS/uccl" ]] || {
        echo "Missing UCCL install: $UCCL_PYDEPS (run install_uccl.sh first)" >&2
        return 1
    }

    local binds=(
        --bind "$SRC:/workspace/TorchSpec"
        --bind "$WORK:$WORK"
        --bind "$UCCL_PYDEPS:$UCCL_PYDEPS"
        --bind "$CRAY_LIBFABRIC:$CRAY_LIBFABRIC"
    )
    if [[ -n "${MODEL_WORK:-}" && -d "$MODEL_WORK" ]]; then
        binds+=(--bind "$MODEL_WORK:$MODEL_WORK")
    fi
    local container_env=(
        --env "PYTHONPATH=/workspace/TorchSpec:$UCCL_PYDEPS"
        # --nv places the host 565 driver in /.singularity.d/libs. CUDA 13
        # needs the SIF's 580 forward-compatibility libcuda to win lookup,
        # while the actual kernel driver and GPU devices still come from --nv.
        --env "LD_LIBRARY_PATH=$CONTAINER_CUDA_COMPAT:$CONTAINER_CUDA_LIB:$UCCL_PYDEPS/uccl/lib:$CRAY_LIBFABRIC/lib64:/.singularity.d/libs"
        --env "UCCL_LIBFABRIC_SO=$CRAY_LIBFABRIC/lib64/libfabric.so.1"
        --env "UCCL_P2P_TRANSPORT=${UCCL_P2P_TRANSPORT:-cxi}"
        --env "UCCL_P2P_DISABLE_IPC=${UCCL_P2P_DISABLE_IPC:-1}"
        --env "UCCL_SOCKET_IFNAME=${UCCL_SOCKET_IFNAME:-hsn0}"
        --env "UCCL_CXI_DEVICE_INDEX=${UCCL_CXI_DEVICE_INDEX:-0}"
    )
    local host_lib container_lib cxi_device env_name
    for host_lib in \
        /usr/lib64/libcxi.so.1 \
        /usr/lib64/libatomic.so.1 \
        /usr/lib64/libnl-3.so.200 \
        /usr/lib64/libnl-route-3.so.200; do
        [[ -e "$host_lib" ]] || continue
        container_lib="/usr/lib/aarch64-linux-gnu/${host_lib##*/}"
        binds+=(--bind "$host_lib:$container_lib")
    done
    for cxi_device in /dev/cxi*; do
        [[ -e "$cxi_device" ]] && binds+=(--bind "$cxi_device")
    done
    for env_name in \
        CUDA_VISIBLE_DEVICES \
        FI_LOG_LEVEL \
        FI_LOG_PROV \
        UCCL_CXI_DOMAIN \
        UCCL_CXI_THREADING \
        UCCL_P2P_LOCAL_GPU_IDX \
        UCCL_P2P_LOG_LEVEL; do
        [[ -v "$env_name" ]] && container_env+=(--env "$env_name=${!env_name}")
    done
    # Cray's CXI provider derives authorization from the active Slurm job
    # context. Preserve that context despite Apptainer's clean environment.
    while IFS='=' read -r env_name env_value; do
        case "$env_name" in
            SLINGSHOT_*|FI_CXI_*|PMI_*)
                container_env+=(--env "$env_name=$env_value")
                ;;
            SLURM_JOB_ID|SLURM_JOB_UID|SLURM_JOB_GID|SLURM_PROCID|SLURM_LOCALID|SLURM_NODEID|SLURM_NTASKS)
                container_env+=(--env "$env_name=$env_value")
                ;;
        esac
    done < <(env)

    apptainer exec --cleanenv --nv "${binds[@]}" "${container_env[@]}" \
        "$IMAGE" "$@"
}

uccl_cuda_diagnostics() {
    uccl_apptainer python3 - <<'PY'
import ctypes
from pathlib import Path

import torch

cuda = ctypes.CDLL("libcuda.so.1")
driver_version = ctypes.c_int()
cu_init = cuda.cuInit(0)
cu_driver_version = cuda.cuDriverGetVersion(ctypes.byref(driver_version))
if cu_init != 0 or cu_driver_version != 0:
    raise RuntimeError(
        f"CUDA driver initialization failed: cuInit={cu_init} "
        f"cuDriverGetVersion={cu_driver_version}"
    )
torch.cuda.init()
mapped = sorted(
    {
        line.rsplit(maxsplit=1)[-1]
        for line in Path("/proc/self/maps").read_text().splitlines()
        if "libcuda.so" in line
    }
)
print(f"torch={torch.__version__} torch_cuda={torch.version.cuda}")
print(f"cuDriverGetVersion={driver_version.value}")
print(f"device={torch.cuda.get_device_name(torch.cuda.current_device())}")
print("mapped_libcuda=" + ",".join(mapped))
if not any("/compat/libcuda.so" in path for path in mapped):
    raise RuntimeError(f"CUDA forward-compatibility libcuda was not loaded: {mapped}")
PY
}
