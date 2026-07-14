#!/usr/bin/env bash

set -euo pipefail

WORK="${WORK:-$SCRATCH/torchspec-qwen35-9b}"
IMAGE_URI="${IMAGE_URI:-docker://ghcr.io/doublewordai/torchspec:latest}"
IMAGE="${IMAGE:-$WORK/images/torchspec-latest.sif}"
BUILD_TMP="${BUILD_TMP:-${SLURM_JOB_ID:+/tmp/apptainer-$SLURM_JOB_ID}}"
BUILD_TMP="${BUILD_TMP:-$WORK/apptainer-tmp}"

mkdir -p "$WORK"/{apptainer-cache,cache,checkpoints,container-pydeps,hf-cache,images,logs,outputs,tmp,torchinductor} "$BUILD_TMP"
export APPTAINER_CACHEDIR="$WORK/apptainer-cache"
export APPTAINER_TMPDIR="$BUILD_TMP"

if [[ ! -s "$IMAGE" ]]; then
    apptainer build "$IMAGE" "$IMAGE_URI"
fi

# Keep additions missing from the published image in a persistent scratch target.
# The image already owns PyTorch/CUDA dependencies, so never duplicate them here.
if ! APPTAINERENV_PYTHONPATH="$WORK/container-pydeps" \
    apptainer exec --bind "$WORK:$WORK" "$IMAGE" \
    python3 -c 'import liger_kernel'; then
    apptainer exec --bind "$WORK:$WORK" "$IMAGE" \
        python3 -m pip install --no-cache-dir --no-deps \
        --target "$WORK/container-pydeps" 'liger-kernel>=0.8.0'
fi

echo "Prepared persistent runtime at $WORK"
echo "Image: $IMAGE"
