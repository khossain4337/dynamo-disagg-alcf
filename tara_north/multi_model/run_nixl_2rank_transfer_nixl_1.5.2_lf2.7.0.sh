#!/bin/bash
# run_nixl_2rank_transfer_nixl_1.5.2.sh over upstream libfabric v2.7.0.
#   [LOG_ROOT=<dir>] [PPN=4] [FI_CXI_ENABLE_WRITEDATA=0] bash run_nixl_2rank_transfer_nixl_1.5.2_lf2.7.0.sh [python args]
# --op WRITE needs FI_CXI_ENABLE_WRITEDATA=1, the default here. --mem cuda unless overridden.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_DIR="$(dirname "$HERE")"
LF=/vast/draco/tara/projects/Tara_Deployment/software/repositories/libfabric_v2.7.0/v2.7.0_cu13.0

if [ -z "${PMI_RANK:-}" ]; then
    MODE=pair MEM=cuda GIB=4 OP=READ prev=""
    for a in "$@"; do
        case $prev in --mode) MODE=$a ;; --mem) MEM=$a ;; --gib) GIB=$a ;; --op) OP=$a ;; esac
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
    LOG="$LOG_DIR/nixl1.5.2_lf2.7.0_N${NODES}_R${PPN}_${MODE}_${OP}_${MEM}_${GIB}GB_wd${FI_CXI_ENABLE_WRITEDATA:-1}_$(date +%Y%m%d_%H%M%S).txt"
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

# After the env, so 2.7.0 wins over the libfabric/2.3.1 module.
export LD_LIBRARY_PATH=$LF/lib:$LD_LIBRARY_PATH
export FI_CXI_ENABLE_WRITEDATA=${FI_CXI_ENABLE_WRITEDATA:-1}
export NIXL_LOG_LEVEL=${NIXL_LOG_LEVEL:-INFO}
if [ "$PMI_RANK" = 0 ]; then
    ldd "$CONDA_PREFIX"/lib/python3.12/site-packages/.nixl_cu13.mesonpy.libs/plugins/libplugin_LIBFABRIC.so | grep libfabric
fi
exec python3 "$REPRO_DIR/repro_nixl_2rank_transfer.py" --mem cuda "$@"
