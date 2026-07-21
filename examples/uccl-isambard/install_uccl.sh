#!/usr/bin/env bash

# Run on an allocated Isambard compute node. Builds only UCCL-P2P and installs
# it in a persistent, user-writable Python target next to the cached SIF.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

UCCL_COMMIT="${UCCL_COMMIT:-4312141c93c4b2f3ff7b9f274ad58c355c14fab6}"
UCCL_SRC="${UCCL_SRC:-$WORK/uccl-src-$UCCL_COMMIT}"
JOBS="${JOBS:-${SLURM_CPUS_PER_TASK:-16}}"
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || { echo "JOBS must be a positive integer" >&2; exit 1; }

mkdir -p "$WORK" "$UCCL_PYDEPS"
if [[ ! -d "$UCCL_SRC/.git" ]]; then
    git clone https://github.com/uccl-project/uccl.git "$UCCL_SRC"
fi
if ! git -C "$UCCL_SRC" diff --quiet || ! git -C "$UCCL_SRC" diff --cached --quiet; then
    echo "Refusing to replace tracked edits in $UCCL_SRC" >&2
    exit 1
fi
git -C "$UCCL_SRC" fetch --quiet origin "$UCCL_COMMIT"
git -C "$UCCL_SRC" checkout --quiet --detach "$UCCL_COMMIT"

CXI_COMPAT_PATCH="$SRC/examples/uccl-isambard/uccl-cxi-libfabric-1.22.patch"
[[ -s "$CXI_COMPAT_PATCH" ]] || { echo "Missing CXI compatibility patch: $CXI_COMPAT_PATCH" >&2; exit 1; }
git -C "$UCCL_SRC" apply --check "$CXI_COMPAT_PATCH"
git -C "$UCCL_SRC" apply "$CXI_COMPAT_PATCH"
restore_uccl_source() {
    git -C "$UCCL_SRC" apply --reverse --check "$CXI_COMPAT_PATCH" >/dev/null 2>&1 &&
        git -C "$UCCL_SRC" apply --reverse "$CXI_COMPAT_PATCH"
}
trap restore_uccl_source EXIT

[[ -s "$IMAGE" ]] || { echo "Missing cached SIF: $IMAGE" >&2; exit 1; }
mkdir -p "$UCCL_SRC/uccl/lib"

apptainer exec --cleanenv \
    --bind "$UCCL_SRC:/uccl-src" \
    --bind "$UCCL_PYDEPS:$UCCL_PYDEPS" \
    --bind "$CRAY_LIBFABRIC:$CRAY_LIBFABRIC" \
    --env "PYTHONPATH=$UCCL_PYDEPS" \
    --env "UCCL_BUILD_JOBS=$JOBS" \
    --env "UCCL_BUILD_TARGET=$UCCL_PYDEPS" \
    --env "UCCL_BUILD_LIBFABRIC=$CRAY_LIBFABRIC" \
    --env "UCCL_BUILD_CUDA_HOME=$CONTAINER_CUDA_HOME" \
    "$IMAGE" bash --noprofile --norc -c '
        set -euo pipefail
        # Host CUDA/NVHPC modules commonly export compiler search paths. Use
        # GCC, Python, and the CUDA 13 toolkit provided by the SIF.
        unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH OBJC_INCLUDE_PATH
        unset COMPILER_PATH GCC_EXEC_PREFIX LIBRARY_PATH
        unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS
        unset CC CXX CPP LD CUDA_PATH CUDA_ROOT NVHPC NVHPC_ROOT
        unset CMAKE_PREFIX_PATH PKG_CONFIG_PATH
        export PATH="$UCCL_BUILD_CUDA_HOME/bin:/usr/local/bin:/usr/bin:/bin"
        export CUDA_HOME="$UCCL_BUILD_CUDA_HOME"
        [[ -s "$CUDA_HOME/include/cuda.h" ]] || {
            echo "Missing container CUDA toolkit: $CUDA_HOME" >&2
            exit 1
        }

        # The runtime SIF intentionally lacks development linker symlinks.
        # Link CUDA against the toolkit stub, and supply a private libelf.so
        # symlink to the versioned SIF library. Runtime selects the SIF
        # forward-compatible driver while --nv supplies devices and host glue.
        cuda_stub_dir="$CUDA_HOME/targets/sbsa-linux/lib/stubs"
        cuda_stub="$cuda_stub_dir/libcuda.so"
        # Resolve the real file: on this SIF libelf.so.1 is a relative symlink
        # whose target can be hidden by Apptainer batch-time library mounts.
        elf_runtime="$(find /usr/lib/aarch64-linux-gnu /lib/aarch64-linux-gnu \
            -maxdepth 1 -type f -name "libelf-*.so" -print -quit 2>/dev/null)"
        [[ -s "$cuda_stub" ]] || { echo "Missing CUDA linker stub: $cuda_stub" >&2; exit 1; }
        [[ -n "$elf_runtime" && -s "$elf_runtime" ]] || {
            echo "Missing versioned libelf runtime" >&2
            exit 1
        }
        link_dir="$UCCL_BUILD_TARGET/.torchspec-link-libs"
        mkdir -p "$link_dir"
        ln -sfn "$elf_runtime" "$link_dir/libelf.so"
        export LIBRARY_PATH="$cuda_stub_dir:$link_dir"

        echo "UCCL build compiler=$(command -v g++) cuda_home=$CUDA_HOME"
        echo "UCCL build GCC version=$(g++ -dumpfullversion -dumpversion)"
        echo "UCCL build CUDA version=$(nvcc --version | tail -1)"
        echo "UCCL build library_path=$LIBRARY_PATH"
        python3 -m pip install --no-cache-dir --upgrade \
            --target "$UCCL_BUILD_TARGET" nanobind==2.13.0
        cd /uccl-src/p2p
        make clean
        make -j"$UCCL_BUILD_JOBS" \
            PYTHON=python3 \
            CXX=/usr/bin/g++ \
            CUDA_HOME="$CUDA_HOME" \
            LIBFABRIC_HOME="$UCCL_BUILD_LIBFABRIC"
        mkdir -p "$UCCL_BUILD_TARGET/uccl/lib"
        cp ../uccl/__init__.py "$UCCL_BUILD_TARGET/uccl/"
        cp p2p.abi3.so utils.py "$UCCL_BUILD_TARGET/uccl/"
        cp libuccl_p2p.so "$UCCL_BUILD_TARGET/uccl/lib/"
    '

echo "Installed UCCL $UCCL_COMMIT in $UCCL_PYDEPS"
uccl_cuda_diagnostics
UCCL_P2P_TRANSPORT=cxi uccl_apptainer python3 -c \
    'import uccl.p2p; print("UCCL-P2P import OK")'
