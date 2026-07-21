#!/usr/bin/env bash
#SBATCH --job-name=cuda13-compat
#SBATCH --account=brics.s6p
#SBATCH --partition=workq
#SBATCH --reservation=brics_s6p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00

set -euo pipefail

: "${SCRATCH:=/scratch/s6p/${USER:?USER is required}}"
: "${IMAGE:=$SCRATCH/torchspec-qwen35-9b/images/torchspec-latest.sif}"

apptainer exec --cleanenv --nv \
    --env 'LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/cuda/targets/sbsa-linux/lib:/.singularity.d/libs' \
    "$IMAGE" python3 - <<'PY'
import ctypes
from pathlib import Path

import torch

cuda = ctypes.CDLL("libcuda.so.1")
driver_version = ctypes.c_int()
assert cuda.cuInit(0) == 0
assert cuda.cuDriverGetVersion(ctypes.byref(driver_version)) == 0
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
print(f"device={torch.cuda.get_device_name(0)}")
print("mapped_libcuda=" + ",".join(mapped))
assert any("/compat/libcuda.so" in path for path in mapped), mapped
PY
