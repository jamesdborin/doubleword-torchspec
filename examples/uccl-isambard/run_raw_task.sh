#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

RENDEZVOUS="${1:?rendezvous directory is required}"
PAYLOAD_MIB="${2:-1}"
[[ "${SLURM_NTASKS:-0}" == 2 ]] || { echo "raw smoke requires exactly two tasks" >&2; exit 1; }
case "${SLURM_PROCID:?SLURM_PROCID is required}" in
    0) ROLE=producer ;;
    1) ROLE=consumer ;;
    *) echo "unexpected rank: $SLURM_PROCID" >&2; exit 1 ;;
esac

echo "rank=$SLURM_PROCID role=$ROLE node=$(hostname) hsn0=$(uccl_hsn_address)"
# CXI device numbering is independent of Slurm's task-local GPU numbering.
export UCCL_CXI_DEVICE_INDEX="${UCCL_CXI_DEVICE_INDEX:-0}"
uccl_apptainer python3 \
    /workspace/TorchSpec/examples/uccl-isambard/raw_smoke.py \
    --role "$ROLE" --rendezvous "$RENDEZVOUS" --payload-mib "$PAYLOAD_MIB"
