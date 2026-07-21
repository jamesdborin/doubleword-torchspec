#!/usr/bin/env bash
#SBATCH --job-name=uccl-cxi-2n
#SBATCH --account=brics.s6p
#SBATCH --partition=workq
#SBATCH --reservation=brics_s6p
#SBATCH --network=job_vni
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=uccl-cxi-2n-%j.out
#SBATCH --error=uccl-cxi-2n-%j.err

set -euo pipefail

: "${SCRATCH:=/scratch/s6p/${USER:?USER is required}}"
: "${SRC:=$SCRATCH/torchspec/torchspec}"
: "${WORK:=$SCRATCH/torchspec-uccl}"
RUN_DIR="$WORK/raw-two-node-${SLURM_JOB_ID:?}"
mkdir -p "$RUN_DIR"
echo "nodes=$SLURM_JOB_NODELIST payload_mib=${PAYLOAD_MIB:-1} rendezvous=$RUN_DIR"
srun --network=no_vni --kill-on-bad-exit=1 "$SRC/examples/uccl-isambard/run_raw_task.sh" \
    "$RUN_DIR" "${PAYLOAD_MIB:-1}"
