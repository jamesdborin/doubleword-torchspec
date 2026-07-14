#!/usr/bin/env bash
#SBATCH --job-name=qwen35-9b-dflash
#SBATCH --account=brics.s6p
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=72:00:00
#SBATCH --output=/scratch/s6p/%u/torchspec-qwen35-9b/logs/slurm-%j.out
#SBATCH --error=/scratch/s6p/%u/torchspec-qwen35-9b/logs/slurm-%j.err

set -euo pipefail

SRC="${SRC:-$SCRATCH/torchspec/torchspec}"
WORK="${WORK:-$SCRATCH/torchspec-qwen35-9b}"

cd "$SRC"
if [[ ! -s "$WORK/images/torchspec-latest.sif" ]]; then
    ./examples/qwen35-9b-isambard/prepare_isambard.sh
fi
exec ./examples/qwen35-9b-isambard/run_apptainer.sh
