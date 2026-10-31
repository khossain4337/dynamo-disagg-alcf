#!/bin/bash
# Port of ../run_repro_nixl_2rank_transfer.sh for the nixl 1.5.2 env.
#   [LOG_ROOT=<dir>] [SHIM=1] [PPN=4] bash run_nixl_2rank_transfer_nixl_1.5.2.sh [python args]
# --mode aggregate sets PPN=4. Shim is off unless SHIM=1. --mem cuda unless overridden.
# Log: $LOG_ROOT/NIXL_LOG/ if set, else testing/nixl_logs/;
# name nixl1.5.2_N<nodes>_R<ppn>_<mode>_<mem>_<gib>GB_shim<0|1>_<timestamp>.txt
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_DIR="$(dirname "$HERE")"

if [ -z "${PMI_RANK:-}" ]; then
    MODE=pair MEM=cuda GIB=4 prev=""
    for a in "$@"; do
        case $prev in --mode) MODE=$a ;; --mem) MEM=$a ;; --gib) GIB=$a ;; esac
        prev=$a
    done
    [ "$MODE" = aggregate ] && PPN=${PPN:-4}
    PPN=${PPN:-1}
    NODES=$(sort -u "$PBS_NODEFILE" | grep -c .)
    # One mpiexec for all ranks: ranks from separate launches get different VNIs.
    VNI_FLAG=""
    [ "$NODES" -eq 1 ] && VNI_FLAG="--single-node-vni"
    LOG_DIR=${LOG_ROOT:+$LOG_ROOT/NIXL_LOG}
    LOG_DIR=${LOG_DIR:-/vast/draco/tara/projects/Tara_Deployment/software/testing/nixl_logs}
    mkdir -p "$LOG_DIR"
    LOG="$LOG_DIR/nixl1.5.2_N${NODES}_R${PPN}_${MODE}_${MEM}_${GIB}GB_shim${SHIM:-0}_$(date +%Y%m%d_%H%M%S).txt"
    echo "log: $LOG"
    mpiexec -n $((PPN * 2)) -ppn "$PPN" $VNI_FLAG bash "$0" "$@" 2>&1 | tee "$LOG"
    exit
fi

RANK_ARGS=("$@")
set --
set +eu
source "$HERE/env_vllm_0.32.1rc1_nixl_1.5.2.sh"
set -eu
set -- ${RANK_ARGS[@]+"${RANK_ARGS[@]}"}

export NIXL_LOG_LEVEL=${NIXL_LOG_LEVEL:-INFO}
[ "${SHIM:-0}" = 1 ] && export LD_PRELOAD="$REPRO_DIR/fi_getinfo_shim.so"
exec python3 "$REPRO_DIR/repro_nixl_2rank_transfer.py" --mem cuda "$@"
