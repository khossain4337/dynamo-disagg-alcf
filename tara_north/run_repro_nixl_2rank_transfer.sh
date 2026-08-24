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

# A sourced script INHERITS the caller's positional parameters, and conda's
# `bin/activate` reads $1 as the environment name. Sourcing the env script
# with our own args still set makes conda go looking for an environment
# literally named "--dump-api". Stash the args, clear them, source, restore.
#
# errexit/nounset are also lifted across the source: conda's activation
# machinery has unset-variable reads and non-zero intermediate returns that
# are harmless interactively but fatal under `set -eu`.
RANK_ARGS=( "$@" )
set --
set +eu
source ./env_for_libfabric_topology_error.sh
set -eu
set -- ${RANK_ARGS[@]+"${RANK_ARGS[@]}"}

export LD_PRELOAD="${SCRIPT_DIR}/fi_getinfo_shim.so"

# BLOCKER 3: fi_writedata failed on rail 0: Flags not supported (-FI_EBADFLAGS)
#
# NIXL's libfabric rail posts the MAIN DATA PATH with fi_writedata(), riding
# the transfer id in the RMA immediate data. That is free on EFA. On CXI the
# feature is OPT-IN and off by default, so domain_attr->cq_data_size
# negotiates to 0 and every write is rejected. Per fi_cxi(7):
#
#   "Controls provider support for the fi_writedata() and
#    fi_inject_writedata() RMA operations... This option is disabled by
#    default... Application support for FI_MR_PROV_KEY mr_mode is required
#    to use this feature."
#
# The FI_MR_PROV_KEY prerequisite is already met by the shim's mask:
#   0x674 = FI_MR_LOCAL(1<<2) | FI_MR_VIRT_ADDR(1<<4) | FI_MR_ALLOCATED(1<<5)
#         | FI_MR_PROV_KEY(1<<6) | FI_MR_ENDPOINT(1<<9) | FI_MR_HMEM(1<<10)
#
# MEASURED -- do not re-derive any of this:
#
#  a) `fi_info -p cxi -v | grep cq_data_size` reports 8 with AND without the
#     variable set. cq_data_size is the DOMAIN ATTRIBUTE ("the hardware can
#     carry 8 bytes of CQ data"), a PRECONDITION of the feature, not a
#     readout of whether it is switched on. Not a usable test.
#
#  b) THE KNOB DOES NOT EXIST IN THIS LIBFABRIC BUILD. All 72 knobs from
#     `fi_info -e 2>&1 | grep -a -oE 'FI_CXI_[A-Z0-9_]+' | sort -u` were
#     enumerated; FI_CXI_ENABLE_WRITEDATA is not among them. It postdates
#     this version. The export below is therefore a NO-OP here -- kept only
#     so it starts working if libfabric is ever upgraded.
#
#  c) NIXL 1.4.0 has NO fi_write fallback. `strings` on
#     libplugin_LIBFABRIC.so shows exactly four data-path ops:
#         fi_writedata   (RMA write -- carries FI_REMOTE_CQ_DATA, REJECTED by cxi)
#         fi_read        (RMA read  -- no immediate data, should be fine)
#         fi_senddata    (message send; CQ data on MSG is supported by cxi)
#         fi_recvmsg
#     So WRITE is unreachable on this stack, but READ should work.
#
# CONSEQUENCE: use --op READ. That is also what the real workload does --
# vLLM's NixlConnector is a PULL (NixlPullConnectorWorker): decode reads KV
# from prefill. vLLM never issues the WRITE path at all.
export FI_CXI_ENABLE_WRITEDATA="${FI_CXI_ENABLE_WRITEDATA:-1}"

exec python3 "${SCRIPT_DIR}/repro_nixl_2rank_transfer.py" "$@"
