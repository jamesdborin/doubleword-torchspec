#!/usr/bin/env bash
#SBATCH --job-name=q35-dflash-speed
#SBATCH --account=brics.s6p
#SBATCH --partition=workq
#SBATCH --reservation=brics_s6p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/s6p/%u/torchspec-qwen35-9b/logs/speed-eval-%j.out
#SBATCH --error=/scratch/s6p/%u/torchspec-qwen35-9b/logs/speed-eval-%j.err

set -euo pipefail
SRC="${SRC:-$SCRATCH/torchspec/torchspec}"
cd "$SRC"
exec ./examples/qwen35-9b-isambard/run_speed_bench_eval.sh
