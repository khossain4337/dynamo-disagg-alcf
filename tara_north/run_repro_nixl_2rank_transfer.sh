#!/bin/bash
#
# Wrapper for repro_nixl_2rank_transfer.py.
#
# Self-launching: run it bare and it re-execs itself under mpiexec; run it
# already under mpiexec and it skips straight to the rank body. These are
# equivalent:
#
#     bash run_repro_nixl_2rank_transfer.sh
#     mpiexec -n 2 -ppn 1 bash run_repro_nixl_2rank_transfer.sh
#
# Extra args pass through to the Python:
#     bash run_repro_nixl_2rank_transfer.sh --gib 16 --mem cuda
#     bash run_repro_nixl_2rank_transfer.sh --op READ      # vLLM's direction
#
# WHY ONE mpiexec, NOT ONE PER RANK
# ---------------------------------
# Without a VNI, fi_domain() returns -FI_ENOSYS for the cxi provider. A
# bare `mpiexec -n 1` yields an EMPTY SLINGSHOT_VNIS; `--single-node-vni`
# does provision one for a single-rank launch, which makes "one mpiexec per
# engine" look viable. It is not.
#
# PALS allocates the VNI per APPLICATION, and a VNI is a fabric
# traffic-isolation domain -- endpoints on different VNIs cannot exchange
# traffic at all. Three separate launches were observed to get three
# different VNIs (1726, 1825, and 1873 -- the last from `-n 1
# --single-node-vni`). Two independently-launched ranks would therefore each
# create their LIBFABRIC agent happily and then never reach each other,
# trading today's loud, immediate ENOSYS for a silent hang. Both ranks must
# come from ONE launch.
#
# `--single-node-vni` is still useful in exactly one case, handled below:
# exercising this script's machinery inside a 1-node allocation. Note the
# transfer itself will then abort on the distinct-hosts guard in the Python,
# by design -- a same-node pass could be served by shm and proves nothing.
#
# LD_PRELOAD of the shim is set inside the rank body, never on mpiexec
# itself -- interposing fi_getinfo in PALS's own machinery is not wanted.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Launcher half: only runs when we are NOT yet a PALS rank ------------
if [ -z "${PMI_RANK:-}" ]; then
    NRANKS=${NRANKS:-2}
    PPN=${PPN:-1}

    # Add --single-node-vni only when the allocation really is one node.
    # On a multi-node allocation PALS provisions the VNI anyway, and we do
    # not want to depend on the flag being a no-op there.
    VNI_FLAG=""
    if [ -n "${PBS_NODEFILE:-}" ] && [ -r "${PBS_NODEFILE}" ]; then
        NNODES=$(sort -u "${PBS_NODEFILE}" | grep -c .)
        echo "Allocation has ${NNODES} node(s)."
        if [ "${NNODES}" -eq 1 ]; then
            VNI_FLAG="--single-node-vni"
            echo "Single-node allocation -> adding --single-node-vni."
            echo "NOTE: the transfer will abort on the distinct-hosts guard."
        fi
    else
        echo "WARNING: PBS_NODEFILE unset/unreadable -- assuming multi-node."
    fi

    echo "Launching: mpiexec -n ${NRANKS} -ppn ${PPN} ${VNI_FLAG} bash $0 $*"
    exec mpiexec -n "${NRANKS}" -ppn "${PPN}" ${VNI_FLAG} bash "$0" "$@"
fi

# ---- Rank half ------------------------------------------------------------
cd "${SCRIPT_DIR}"
source ./env_for_libfabric_topology_error.sh

export LD_PRELOAD="${SCRIPT_DIR}/fi_getinfo_shim.so"

exec python3 "${SCRIPT_DIR}/repro_nixl_2rank_transfer.py" "$@"
