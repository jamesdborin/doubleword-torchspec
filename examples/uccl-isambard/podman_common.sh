#!/usr/bin/env bash

# Reusable podman-hpc runtime for UCCL/CXI on Isambard. Source this file after
# common.sh (or source it directly; it loads common.sh itself).
#
# Mount and environment defaults are derived from doublewordai/isambard-skill
# commit 7bb5ca56e08368955f867c7d09665f450e2efa1a.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

: "${PROJECTDIR:=/projects/s6p}"
: "${UCCL_OCI_IMAGE:=ghcr.io/doublewordai/torchspec:latest}"
: "${PODMANHPC_SQUASH_DIR:=$PROJECTDIR/${USER:?USER is required}/podman-store}"
: "${PODMAN_CUDA_COMPAT:=$WORK/cuda-compat}"
: "${PODMAN_SHM_SIZE:=256g}"
: "${UCCL_PODMAN_PIDS_LIMIT:=-1}"
: "${UCCL_PODMAN_PID_MODE:=host}"
: "${UCCL_PODMAN_USE_INIT:=0}"
: "${UCCL_PODMAN_CGROUPS:=}"
: "${OFI_PLUGIN_DIR:=/tools/brics/apps/linux-neoverse_v2/aws-ofi-nccl-1.18.0-sji722yevkw6isblkxnqjod7nseadbch/lib}"

uccl_podman_cxi_env() {
    export FI_PROVIDER="${FI_PROVIDER:-cxi}"
    export FI_CXI_DEFAULT_CQ_SIZE="${FI_CXI_DEFAULT_CQ_SIZE:-131072}"
    export FI_CXI_DEFAULT_TX_SIZE="${FI_CXI_DEFAULT_TX_SIZE:-16384}"
    export FI_CXI_DEFAULT_RX_SIZE="${FI_CXI_DEFAULT_RX_SIZE:-512}"
    export FI_CXI_OFLOW_BUF_SIZE="${FI_CXI_OFLOW_BUF_SIZE:-16777216}"
    export FI_CXI_OFLOW_BUF_COUNT="${FI_CXI_OFLOW_BUF_COUNT:-32}"
    export FI_CXI_RDZV_PROTO="${FI_CXI_RDZV_PROTO:-alt_read}"
    export FI_CXI_RDZV_THRESHOLD="${FI_CXI_RDZV_THRESHOLD:-0}"
    export FI_CXI_RDZV_GET_MIN="${FI_CXI_RDZV_GET_MIN:-0}"
    export FI_CXI_RDZV_EAGER_SIZE="${FI_CXI_RDZV_EAGER_SIZE:-0}"
    export FI_CXI_RX_MATCH_MODE="${FI_CXI_RX_MATCH_MODE:-hybrid}"
    export FI_CXI_REQ_BUF_MIN_POSTED="${FI_CXI_REQ_BUF_MIN_POSTED:-64}"
    export FI_CXI_REQ_BUF_SIZE="${FI_CXI_REQ_BUF_SIZE:-16777216}"
    export FI_CXI_MSG_LOSSLESS="${FI_CXI_MSG_LOSSLESS:-1}"
    export FI_CXI_DISABLE_NON_INJECT_MSG_IDC="${FI_CXI_DISABLE_NON_INJECT_MSG_IDC:-1}"
    export FI_CXI_DISABLE_HOST_REGISTER="${FI_CXI_DISABLE_HOST_REGISTER:-1}"
    export FI_CXI_ENABLE_WRITEDATA="${FI_CXI_ENABLE_WRITEDATA:-1}"
    export FI_HMEM_CUDA_USE_GDRCOPY="${FI_HMEM_CUDA_USE_GDRCOPY:-1}"
    export FI_MR_CACHE_MONITOR="${FI_MR_CACHE_MONITOR:-userfaultfd}"
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-hsn}"
    export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-PHB}"
    export NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-1}"
    export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-4}"
    export NCCL_GDRCOPY_ENABLE="${NCCL_GDRCOPY_ENABLE:-1}"
    export NCCL_NET_FORCE_FLUSH="${NCCL_NET_FORCE_FLUSH:-0}"
    export NCCL_DEBUG="${NCCL_DEBUG:-VERSION}"
}

uccl_stage_podman_cuda_compat() {
    [[ -s "$IMAGE" ]] || { echo "Missing cached SIF: $IMAGE" >&2; return 1; }
    mkdir -p "$PODMAN_CUDA_COMPAT"
    apptainer exec --cleanenv --bind "$PODMAN_CUDA_COMPAT:/compat-output" \
        "$IMAGE" sh -c 'cp -a /usr/local/cuda/compat/. /compat-output/'
    [[ -s "$PODMAN_CUDA_COMPAT/libcuda.so.1" ]] || {
        echo "Failed to stage CUDA compatibility libraries in $PODMAN_CUDA_COMPAT" >&2
        return 1
    }
}

uccl_heal_stale_podman_state() {
    local uid mountpoint
    uid="$(id -u)"
    [[ "$uid" =~ ^[0-9]+$ ]] || { echo "invalid numeric uid: $uid" >&2; return 1; }
    if pgrep -u "$uid" -x conmon >/dev/null 2>&1; then
        echo "Refusing Podman reset: this user has an active conmon on $(hostname)" >&2
        return 1
    fi

    # This is the guarded reset from doublewordai/isambard-skill, narrowed to
    # this user's node-local runtime.  Shared squash/image stores are untouched.
    while read -r mountpoint; do
        stat "$mountpoint" >/dev/null 2>&1 || fusermount3 -uz "$mountpoint" 2>/dev/null || true
    done < <(awk -v prefix="/local/user/$uid/" '$2 ~ "^" prefix && $3 ~ /fuse/ {print $2}' /proc/mounts)
    podman unshare rm -rf "/local/user/$uid/storage" >/dev/null 2>&1 || true
    rm -rf "/local/user/$uid/libpod" "/local/user/$uid"/overlay* \
        "/local/user/$uid/runc" "/local/user/$uid/networks" \
        "/local/user/$uid/storage" 2>/dev/null || true
    if [[ "${UCCL_INIT_LEGACY_PODMAN_STORE:-1}" == 1 ]]; then
        # podman-hpc's legacy migrator assumes these JSON files exist.  Native
        # Podman local-overlay runs must instead initialize their own store.
        local local_store="/local/user/$uid/storage"
        install -d -m 700 "$local_store/overlay/l" \
            "$local_store/overlay-images" "$local_store/overlay-layers"
        printf '[]' >"$local_store/overlay-images/images.json"
        printf '[]' >"$local_store/overlay-layers/layers.json"
        : >"$local_store/overlay-images/images.lock"
        : >"$local_store/overlay-layers/layers.lock"
    fi
    echo "Reset node-local Podman state under /local/user/$uid on $(hostname)"
}

uccl_podman_name() {
    local suffix="${1:-runtime}"
    local host
    host="$(hostname -s)"
    printf 'torchspec-uccl-%s-%s-%s' "${SLURM_JOB_ID:-manual}" "$host" "$suffix" \
        | tr -c '[:alnum:]_.-' '-'
    printf '\n'
}

_uccl_podman_env_flags() {
    local name
    UCCL_PODMAN_ENV_FLAGS=(
        -e "LD_LIBRARY_PATH=/opt/cuda-compat:/opt/aws-ofi-nccl:/opt/libfabric/lib:/opt/cxi/lib:/opt/gdr/lib:$UCCL_PYDEPS/uccl/lib:/usr/local/cuda/lib64:/usr/local/cuda/targets/sbsa-linux/lib:/usr/lib/aarch64-linux-gnu:/hostusr/lib64"
        -e "PYTHONPATH=/workspace/TorchSpec:$UCCL_PYDEPS"
        -e "TRITON_LIBCUDA_PATH=/opt/cuda-compat"
        -e "UCCL_LIBFABRIC_SO=/opt/libfabric/lib/libfabric.so.1"
        -e "UCCL_P2P_TRANSPORT=${UCCL_P2P_TRANSPORT:-cxi}"
        -e "UCCL_P2P_DISABLE_IPC=${UCCL_P2P_DISABLE_IPC:-1}"
        -e "UCCL_SOCKET_IFNAME=${UCCL_SOCKET_IFNAME:-hsn0}"
        -e "UCCL_CXI_DEVICE_INDEX=${UCCL_CXI_DEVICE_INDEX:-0}"
    )
    while IFS='=' read -r name _; do
        case "$name" in
            FI_*|NCCL_*|SLINGSHOT_*|SLURM_*|PMI_*|RAY_*|MASTER_ADDR|MASTER_PORT|CUDA_VISIBLE_DEVICES|UCCL_P2P_LOCAL_GPU_IDX|UCCL_P2P_LOG_LEVEL)
                UCCL_PODMAN_ENV_FLAGS+=(-e "$name")
                ;;
        esac
    done < <(env)
    for name in ${UCCL_PODMAN_FORWARD_ENV:-}; do
        [[ "$name" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || {
            echo "Invalid variable in UCCL_PODMAN_FORWARD_ENV: $name" >&2
            return 1
        }
        [[ -v "$name" ]] && UCCL_PODMAN_ENV_FLAGS+=(-e "$name")
    done
}

_uccl_podman_prepare() {
    command -v podman-hpc >/dev/null || { echo "podman-hpc is not available" >&2; return 1; }
    [[ -d "$SRC" ]] || { echo "Missing TorchSpec source: $SRC" >&2; return 1; }
    [[ -d "$WORK" ]] || { echo "Missing work directory: $WORK" >&2; return 1; }
    [[ -d "$UCCL_PYDEPS/uccl" ]] || {
        echo "Missing UCCL install: $UCCL_PYDEPS (run install_uccl.sh first)" >&2
        return 1
    }
    [[ -s "$PODMAN_CUDA_COMPAT/libcuda.so.1" ]] || {
        echo "Missing CUDA forward-compatibility libraries: $PODMAN_CUDA_COMPAT" >&2
        echo "Run uccl_stage_podman_cuda_compat once on a compute node." >&2
        return 1
    }
    local required
    for required in \
        "$CRAY_LIBFABRIC/lib64/libfabric.so.1" \
        /usr/lib64/libcxi.so.1.5.0 \
        /usr/lib64/libgdrapi.so.2.4 \
        /usr/lib64/libnl-3.so.200.26.0 \
        "$OFI_PLUGIN_DIR"; do
        [[ -e "$required" ]] || { echo "Missing host CXI dependency: $required" >&2; return 1; }
    done

    uccl_podman_cxi_env
    _uccl_podman_env_flags
    UCCL_PODMAN_ARGS=(
        --squash-dir "$PODMANHPC_SQUASH_DIR" --log-level=warn
    )
    # Keep the process limit before the site-specific --gpu option.  The
    # podman-hpc option shim otherwise creates the runtime with its default
    # process ceiling even though the final inspect output reports unlimited.
    UCCL_PODMAN_RUN_PREFIX_ARGS=(--pids-limit="$UCCL_PODMAN_PIDS_LIMIT")
    UCCL_PODMAN_RUN_ARGS=(
        --gpu --ipc host --tmpfs "/dev/shm:rw,size=$PODMAN_SHM_SIZE,mode=1777"
        --network host --group-add keep-groups
        --cap-add=SYS_ADMIN --cap-add=SYS_PTRACE --security-opt seccomp=unconfined
        -v /lus/lfs1aip2:/lus/lfs1aip2
        -v "$SRC:/workspace/TorchSpec"
        -v "$WORK:$WORK"
        -v "$UCCL_PYDEPS:$UCCL_PYDEPS:ro"
        -v "$PODMAN_CUDA_COMPAT:/opt/cuda-compat:ro"
        -v "$CRAY_LIBFABRIC/include:/opt/libfabric/include:ro"
        -v "$CRAY_LIBFABRIC/lib64:/opt/libfabric/lib:ro"
        -v "$CRAY_LIBFABRIC/bin:/opt/libfabric/bin:ro"
        -v /usr/lib64/libcxi.so.1.5.0:/opt/cxi/lib/libcxi.so.1:ro
        -v /usr/lib64/libgdrapi.so.2.4:/opt/gdr/lib/libgdrapi.so.2:ro
        -v /usr/lib64/libnl-3.so.200.26.0:/opt/cxi/lib/libnl-3.so.200:ro
        -v /usr/lib64:/hostusr/lib64:ro
        -v "$OFI_PLUGIN_DIR:/opt/aws-ofi-nccl:ro"
    )
    if [[ -n "${MODEL_WORK:-}" && -d "$MODEL_WORK" ]]; then
        UCCL_PODMAN_RUN_ARGS+=(-v "$MODEL_WORK:$MODEL_WORK")
    fi
    if [[ "$UCCL_PODMAN_PID_MODE" == host ]]; then
        UCCL_PODMAN_RUN_ARGS+=(--pid host)
    elif [[ "$UCCL_PODMAN_PID_MODE" != private ]]; then
        echo "UCCL_PODMAN_PID_MODE must be host or private" >&2
        return 1
    fi
    if [[ "$UCCL_PODMAN_USE_INIT" == 1 ]]; then
        UCCL_PODMAN_RUN_ARGS+=(--init)
    elif [[ "$UCCL_PODMAN_USE_INIT" != 0 ]]; then
        echo "UCCL_PODMAN_USE_INIT must be 0 or 1" >&2
        return 1
    fi
    if [[ -n "$UCCL_PODMAN_CGROUPS" ]]; then
        UCCL_PODMAN_RUN_ARGS+=(--cgroups "$UCCL_PODMAN_CGROUPS")
    fi
    local index
    [[ -e /dev/gdrdrv ]] && UCCL_PODMAN_RUN_ARGS+=(--device /dev/gdrdrv)
    for index in 0 1 2 3; do
        [[ -e "/dev/cxi$index" ]] && UCCL_PODMAN_RUN_ARGS+=(--device "/dev/cxi$index")
    done
    if declare -p UCCL_PODMAN_EXTRA_MOUNTS >/dev/null 2>&1; then
        UCCL_PODMAN_RUN_ARGS+=("${UCCL_PODMAN_EXTRA_MOUNTS[@]}")
    fi
}

uccl_podman_start() {
    local name="${1:?container name is required}"
    [[ "$name" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "Invalid container name: $name" >&2; return 1; }
    _uccl_podman_prepare
    if podman-hpc "${UCCL_PODMAN_ARGS[@]}" container exists "$name" >/dev/null 2>&1; then
        echo "Container already exists: $name" >&2
        return 1
    fi
    podman-hpc "${UCCL_PODMAN_ARGS[@]}" run "${UCCL_PODMAN_RUN_PREFIX_ARGS[@]}" \
        -d --name "$name" \
        "${UCCL_PODMAN_RUN_ARGS[@]}" "${UCCL_PODMAN_ENV_FLAGS[@]}" \
        --workdir /workspace/TorchSpec --entrypoint '' "$UCCL_OCI_IMAGE" \
        /bin/bash -lc 'trap "exit 0" INT TERM; while :; do sleep 3600 & wait $!; done'
}

uccl_podman_run() {
    local name="${1:?container name is required}"
    shift
    [[ "$#" -gt 0 ]] || { echo "container command is required" >&2; return 1; }
    [[ "$name" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "Invalid container name: $name" >&2; return 1; }
    _uccl_podman_prepare
    podman-hpc "${UCCL_PODMAN_ARGS[@]}" run "${UCCL_PODMAN_RUN_PREFIX_ARGS[@]}" \
        --rm --name "$name" \
        "${UCCL_PODMAN_RUN_ARGS[@]}" "${UCCL_PODMAN_ENV_FLAGS[@]}" \
        --workdir /workspace/TorchSpec --entrypoint '' "$UCCL_OCI_IMAGE" "$@"
}

uccl_podman_shared_run() {
    [[ "$#" -gt 0 ]] || { echo "container command is required" >&2; return 1; }
    _uccl_podman_prepare
    podman-hpc "${UCCL_PODMAN_ARGS[@]}" shared-run \
        "${UCCL_PODMAN_RUN_PREFIX_ARGS[@]}" \
        "${UCCL_PODMAN_RUN_ARGS[@]}" "${UCCL_PODMAN_ENV_FLAGS[@]}" \
        --workdir /workspace/TorchSpec --entrypoint '' "$UCCL_OCI_IMAGE" "$@"
}

uccl_podman_rootfs_run() {
    local name="${1:?container name is required}"
    local rootfs="${2:?rootfs directory is required}"
    shift 2
    [[ "$#" -gt 0 ]] || { echo "container command is required" >&2; return 1; }
    [[ "$name" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "Invalid container name: $name" >&2; return 1; }
    [[ -d "$rootfs" && -x "$rootfs/bin/bash" ]] || {
        echo "Invalid extracted container rootfs: $rootfs" >&2
        return 1
    }
    _uccl_podman_prepare
    podman-hpc "${UCCL_PODMAN_ARGS[@]}" run "${UCCL_PODMAN_RUN_PREFIX_ARGS[@]}" \
        --rm --name "$name" \
        "${UCCL_PODMAN_RUN_ARGS[@]}" "${UCCL_PODMAN_ENV_FLAGS[@]}" \
        --read-only --workdir /workspace/TorchSpec --rootfs "$rootfs" "$@"
}

uccl_podman_exec() {
    local name="${1:?container name is required}"
    shift
    [[ "$#" -gt 0 ]] || { echo "container command is required" >&2; return 1; }
    _uccl_podman_prepare
    podman-hpc "${UCCL_PODMAN_ARGS[@]}" exec "${UCCL_PODMAN_ENV_FLAGS[@]}" \
        --workdir /workspace/TorchSpec "$name" "$@"
}

uccl_podman_stop() {
    local name="${1:?container name is required}"
    [[ "$name" =~ ^torchspec-uccl-[A-Za-z0-9_.-]+$ ]] || {
        echo "Refusing to stop an unexpected container name: $name" >&2
        return 1
    }
    UCCL_PODMAN_ARGS=(--squash-dir "$PODMANHPC_SQUASH_DIR" --log-level=warn)
    podman-hpc "${UCCL_PODMAN_ARGS[@]}" stop -t 30 "$name" >/dev/null
    podman-hpc "${UCCL_PODMAN_ARGS[@]}" rm "$name" >/dev/null
}

uccl_podman_smoke() {
    local name status=0
    name="$(uccl_podman_name smoke)"
    uccl_podman_start "$name"
    uccl_podman_exec "$name" "$@" || status=$?
    uccl_podman_stop "$name" || status=$?
    return "$status"
}
