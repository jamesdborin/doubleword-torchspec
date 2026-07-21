#!/usr/bin/env bash
#SBATCH --job-name=uccl-cxi-1n
#SBATCH --account=brics.s6p
#SBATCH --partition=workq
#SBATCH --reservation=brics_s6p
#SBATCH --network=single_node_vni
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=uccl-cxi-1n-%j.out
#SBATCH --error=uccl-cxi-1n-%j.err

set -euo pipefail

: "${SCRATCH:=/scratch/s6p/${USER:?USER is required}}"
: "${SRC:=$SCRATCH/torchspec/torchspec}"
: "${WORK:=$SCRATCH/torchspec-uccl}"
: "${CRAY_LIBFABRIC:=/opt/cray/libfabric/1.22.0}"
RUN_DIR="$WORK/raw-single-${SLURM_JOB_ID:?}"
mkdir -p "$RUN_DIR"
echo "nodes=$SLURM_JOB_NODELIST payload_mib=${PAYLOAD_MIB:-1} rendezvous=$RUN_DIR"
if [[ "${SKIP_UCCL_INSTALL:-0}" != 1 ]]; then
    echo "Building pinned UCCL-P2P once before starting the two raw-smoke tasks"
    JOBS="${UCCL_BUILD_JOBS:-$SLURM_CPUS_PER_TASK}" \
        "$SRC/examples/uccl-isambard/install_uccl.sh"
fi
if [[ "${UCCL_CXI_PREFLIGHT:-0}" == 1 ]]; then
    echo "CXI devices:"
    ls -l /dev/cxi* || true
    echo "CXI/Slingshot environment:"
    env | grep -E '^(FI_CXI|SLINGSHOT|SLURM_(JOB|STEP)|PMI_)' | sort || true
    echo "Host fi_info cxi0:"
    "$CRAY_LIBFABRIC/bin/fi_info" -p cxi -d cxi0 || true
    # shellcheck source=common.sh
    source "$SRC/examples/uccl-isambard/common.sh"
    echo "Container fi_info cxi0:"
    uccl_apptainer "$CRAY_LIBFABRIC/bin/fi_info" -p cxi -d cxi0 || true
fi
if [[ "${UCCL_USE_SRUN:-0}" == 1 ]]; then
    # A Slurm step can supply CXI service/VNI authorization variables that
    # are absent from processes launched directly by the batch shell.
    srun --kill-on-bad-exit=1 "$SRC/examples/uccl-isambard/run_raw_task.sh" \
        "$RUN_DIR" "${PAYLOAD_MIB:-1}"
    exit $?
fi
pids=()
for rank in 0 1; do
    (
        export SLURM_PROCID="$rank" SLURM_LOCALID="$rank" CUDA_VISIBLE_DEVICES="$rank"
        "$SRC/examples/uccl-isambard/run_raw_task.sh" \
            "$RUN_DIR" "${PAYLOAD_MIB:-1}"
    ) &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=$?
done
exit "$status"
