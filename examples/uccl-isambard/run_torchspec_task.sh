#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

RENDEZVOUS="${1:?rendezvous directory is required}"
[[ "${SLURM_NTASKS:-0}" == 2 ]] || { echo "smoke requires exactly two tasks" >&2; exit 1; }
case "${SLURM_PROCID:?SLURM_PROCID is required}" in
    0) ROLE=producer ;;
    1) ROLE=consumer ;;
    *) echo "unexpected rank: $SLURM_PROCID" >&2; exit 1 ;;
esac

export TORCHSPEC_TRANSFER_BACKEND=uccl-p2p
export UCCL_CXI_DEVICE_INDEX="${UCCL_CXI_DEVICE_INDEX:-0}"
export UCCL_P2P_ACK_ADVERTISE_HOST="$(uccl_hsn_address)"
export UCCL_P2P_USE_GPU_DIRECT=0
echo "rank=$SLURM_PROCID role=$ROLE node=$(hostname) hsn=$UCCL_P2P_ACK_ADVERTISE_HOST"
uccl_apptainer python3 \
    /workspace/TorchSpec/examples/uccl-isambard/torchspec_smoke.py \
    --role "$ROLE" --rendezvous "$RENDEZVOUS"
