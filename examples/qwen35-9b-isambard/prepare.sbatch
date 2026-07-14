#!/usr/bin/env bash
#SBATCH --job-name=qwen35-image-prep
#SBATCH --account=brics.s6p
#SBATCH --partition=workq
#SBATCH --reservation=brics_s6p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/s6p/%u/torchspec-qwen35-9b/logs/prepare-%j.out
#SBATCH --error=/scratch/s6p/%u/torchspec-qwen35-9b/logs/prepare-%j.err

set -euo pipefail

SRC="${SRC:-$SCRATCH/torchspec/torchspec}"
mkdir -p "$SCRATCH/torchspec-qwen35-9b/logs"
cd "$SRC"
exec ./examples/qwen35-9b-isambard/prepare_isambard.sh
