#!/usr/bin/env bash

set -euo pipefail

: "${SRC:?}"
: "${WORK:?}"
export UCCL_INIT_LEGACY_PODMAN_STORE=0
export GOMAXPROCS="${SLURM_CPUS_PER_TASK:-4}"
# shellcheck source=../podman_common.sh
source "$SRC/examples/uccl-isambard/podman_common.sh"
uccl_heal_stale_podman_state
uid="$(id -u)"
podman --log-level=debug \
    --root "/local/user/$uid/storage" --runroot "/local/user/$uid" \
    --storage-opt mount_program=/usr/bin/fuse-overlayfs-wrap \
    --storage-opt ignore_chown_errors=true --cgroup-manager cgroupfs \
    pull ghcr.io/doublewordai/torchspec:latest
podman --root "/local/user/$uid/storage" --runroot "/local/user/$uid" images
