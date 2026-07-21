#!/usr/bin/env bash
#SBATCH --job-name=ts-uccl-1n
#SBATCH --account=brics.s6p
#SBATCH --partition=workq
#SBATCH --reservation=brics_s6p
#SBATCH --network=single_node_vni
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --output=ts-uccl-1n-%j.out
#SBATCH --error=ts-uccl-1n-%j.err

set -euo pipefail
: "${SCRATCH:=/scratch/s6p/${USER:?USER is required}}"
: "${SRC:=$SCRATCH/torchspec/torchspec}"
: "${WORK:=$SCRATCH/torchspec-uccl}"
RUN_DIR="$WORK/torchspec-single-${SLURM_JOB_ID:?}"
mkdir -p "$RUN_DIR"
srun --network=no_vni --kill-on-bad-exit=1 \
    "$SRC/examples/uccl-isambard/run_torchspec_task.sh" "$RUN_DIR"
