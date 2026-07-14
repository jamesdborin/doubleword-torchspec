#!/usr/bin/env bash

set -euo pipefail

WORK="${WORK:-$SCRATCH/torchspec-qwen35-9b}"
IMAGE_URI="${IMAGE_URI:-docker://ghcr.io/doublewordai/torchspec:latest}"
IMAGE="${IMAGE:-$WORK/images/torchspec-latest.sif}"

mkdir -p "$WORK"/{apptainer-cache,apptainer-tmp,cache,checkpoints,container-pydeps,hf-cache,images,logs,outputs,tmp,torchinductor}
export APPTAINER_CACHEDIR="$WORK/apptainer-cache"
export APPTAINER_TMPDIR="$WORK/apptainer-tmp"

if [[ ! -s "$IMAGE" ]]; then
    apptainer build "$IMAGE" "$IMAGE_URI"
fi

# Keep additions missing from the published image in a persistent scratch target.
apptainer exec --bind "$WORK:$WORK" "$IMAGE" \
    python3 -m pip install --no-cache-dir --target "$WORK/container-pydeps" \
    'liger-kernel>=0.8.0'

echo "Prepared persistent runtime at $WORK"
echo "Image: $IMAGE"
