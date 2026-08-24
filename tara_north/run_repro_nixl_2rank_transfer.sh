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
#     bash run_repro_nixl_2rank_transfer.sh --op WRITE     # reproduces BLOCKER 3
#
# MODES
# -----
# NIXL binds each GPU to ONE Slingshot rail (see MODES in the Python's module
# docstring for the derivation), so "how fast is VRAM over the fabric" has
# three different honest answers. --mode picks which one you are asking for,
# and THIS SCRIPT PICKS THE LAUNCH SHAPE TO MATCH -- do not set -n/-ppn by
# hand unless you also set NRANKS/PPN here.
#
#   --mode pair        (default)  2 ranks, 1 per node, GPU 0 <-> GPU 0.
#                      One rail. ~24 GB/s. What we have been running.
#
#   --mode aggregate              2N ranks, N per node, rank i on GPU i.
#                      N rails concurrently, reports the NODE total. This is
#                      what vLLM with TP=N actually does, so it is the number
#                      that matters for the real workload.
#                      PPN defaults to the GPU count; override with PPN=<n>.
#                      MEASURED (4 pairs, 4 GiB, READ, VRAM): 4 rails at
#                      16.26 GiB each, 1.02x of payload, NODE 87.67 GB/s --
#                      87.7% of the node's 100 GB/s line rate. The one-rail-
#                      per-GPU partition costs nothing once you run one rank
#                      per GPU, which is what TP=N is.
#
#   --mode solo-peak              2 ranks, but HWLOC_XMLFILE is pointed at a
#                      pruned topology with ONE GPU, on the theory that the
#                      survivor would inherit all 4 rails.
#                      MEASURED: IT DOES NOT. The prune works (3 of 4 GPUs
#                      removed, CUDA bound by UUID) and the transfer passes
#                      byte-exact, but the result is still 1 rail / cxi0 /
#                      23.24 GB/s -- identical to --mode pair. The NIC split
#                      in groupNicsWithAccel() is PER PCIe COMPLEX, and each
#                      complex here already holds 1 GPU + 1 NIC, so removing
#                      GPUs elsewhere changes nothing and merely orphans the
#                      other three NICs into accel-less groups that VRAM can
#                      never reach. See the MODES block in the Python for the
#                      full walk-through. Kept as a documented negative.
#                      Needs lstopo-no-graphics and nvidia-smi on the node.
#
# `--mem cuda` depends on the BLOCKER 4 workaround inside fi_getinfo_shim.c,
# which is on by default. NIXL_CXI_VRAM_SHIM=0 turns it off and restores the
# old DRAM-only behaviour.
#
# --op defaults to READ: it is vLLM's direction and the only one this stack
# supports. See BLOCKER 3 below.
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

# ---- --mode, read by BOTH halves -----------------------------------------
# The launcher half needs it to choose -n/-ppn; the rank half needs it to
# decide whether to build the pruned hwloc topology. Parsed here, before the
# positional parameters get stashed around the conda source below, so the
# value survives that.
MODE="pair"
_prev=""
for _a in "$@"; do
    case "${_prev}" in --mode) MODE="${_a}" ;; esac
    case "${_a}" in --mode=*) MODE="${_a#--mode=}" ;; esac
    _prev="${_a}"
done
case "${MODE}" in
    pair|aggregate|solo-peak) ;;
    *) echo "ERROR: unknown --mode '${MODE}' (pair|aggregate|solo-peak)" >&2; exit 2 ;;
esac

# ---- Launcher half: only runs when we are NOT yet a PALS rank ------------
if [ -z "${PMI_RANK:-}" ]; then
    if [ "${MODE}" = "aggregate" ] && [ -z "${PPN:-}" ]; then
        # One rank per GPU is the whole point of the mode. `|| true` because
        # grep -c exits 1 on no match and pipefail would kill the script
        # before the sanity check below gets to say something useful.
        PPN=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)
        if ! [ "${PPN:-0}" -gt 0 ] 2>/dev/null; then
            echo "ERROR: --mode aggregate could not count GPUs (nvidia-smi -L)." >&2
            echo "       Set PPN=<gpus per node> explicitly." >&2
            exit 1
        fi
        echo "aggregate: ${PPN} GPU(s) per node -> -ppn ${PPN}"
    fi
    PPN=${PPN:-1}
    # Two nodes, PPN ranks each. Pairing is by hostname in the Python, so the
    # rank ordering PALS happens to use does not matter.
    NRANKS=${NRANKS:-$((PPN * 2))}

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

# The shim now carries TWO workarounds; see the header of fi_getinfo_shim.c
# for the full derivation of each.
#
#   BLOCKER 1  cxi hints left mr_mode == 0 -> -FI_ENODATA -> backend aborts.
#   BLOCKER 4  cxi never advertises VRAM_SEG, so `--mem cuda` dies in
#              registerMem with "no available backends for mem type
#              'VRAM_SEG'". NIXL gates accelerator discovery behind
#              `provider_name == "efa"` and hardcodes num_nvidia_accel = 0
#              for everything else, so on Slingshot it concludes the node has
#              no GPUs. The shim relabels the DISCOVERED provider cxi -> efa
#              (device names untouched) so the GPU scan runs, then undoes the
#              spoof on the way back down so the data path stays 100% CXI.
#              Undoing it takes TWO patches, because provider_name is used as
#              a behavioural switch in two different ways: topology hints get
#              their prov_name rewritten efa -> cxi, while the per-rail hints
#              carry NO prov_name at all and instead pick their mr_mode by
#              string-matching the provider -- so those get the missing
#              FI_MR_ENDPOINT / FI_RMA_EVENT / mr_key_size=0 put back.
#
# NIXL_CXI_VRAM_SHIM=0 disables the BLOCKER 4 half only.
#
# MEASURED, and it is fine: riding the EFA branch makes hasPcieDevices() true,
# which could in principle switch DRAM rail selection from "all rails" to
# NUMA-aware and pick a SUBSET. It does not. `--mem dram` under the full shim
# still reports RAILS CARRYING PAYLOAD: 4 at 87.62 GB/s, because NUMA
# detection fails before the policy can narrow anything (rail_manager :620 ->
# :271, all-rails fallback). Details in the BLOCKER 3 block below.
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
# MEASURED with READ, 2 nodes x 1 rank, 4 GiB DRAM, 64 x 64 MiB descriptors,
# WITH the full shim loaded (PATCHES 1-4 all firing):
#   query_xfer_backend -> LIBFABRIC        (Q1: no silent fallback)
#   destination buffer byte-exact          (Q2: pass)
#   best 0.049s -> 87.62 GB/s, mean 75.98 GB/s vs a 100 GB/s 4-rail peak
#   RAILS CARRYING PAYLOAD: 4 (cxi0..cxi3), 4.06 GiB each, evenly spread
#   initiator rx 1.02x of payload          (Q3: pass)
#   target    tx 1.02x of payload          (Q3: pass)
#
# Both ranks now read 1.02x. An earlier run had the target at 0.77x, which was
# a SAMPLING artifact and not a transport one -- its four NICs disagreed by
# ~75 MB in read order while the initiator's agreed to 512 bytes. The Python
# settles the telemetry caches before sampling now, and that closed it.
#
# THE SHIM DOES NOT COST DRAM ANY RAILS -- this was the open worry, since
# riding the EFA branch makes hasPcieDevices() true and could in principle
# flip DRAM from "all rails" to a NUMA-aware subset. It does not, because
# NUMA detection fails first: libfabric_rail_manager.cpp:620 still logs
# "Could not deduce average bandwidth limit per NUMA node" and :271 falls back
# to the all-rails policy. So DRAM gets all 4 rails for the same reason it
# always did, and the shim is irrelevant to it. Caveat unchanged: if that NUMA
# detection is ever fixed, DRAM may drop to a subset and this number moves.
#
# For scale: DRAM 1 rank = 87.62 GB/s and VRAM 4 ranks = 87.67 GB/s land on
# the same figure, so ~87 GB/s is a shared fabric/host ceiling rather than
# anything path-specific.
#
# CONSEQUENCE: READ is the DEFAULT --op in the Python. That is also what the
# real workload does -- vLLM's NixlConnector is a PULL
# (NixlPullConnectorWorker): decode reads KV from prefill, and vLLM never
# issues the WRITE path at all. `--op WRITE` is still accepted and still
# fails here, so the blocker stays reproducible on demand.
export FI_CXI_ENABLE_WRITEDATA="${FI_CXI_ENABLE_WRITEDATA:-1}"

# ---- solo-peak: hand hwloc a topology with exactly one GPU ----------------
# NIXL divides NICs by accelerator count (groupNicsWithAccel Step 4:
# nics_per_group = nics.size() / num_groups). 4 GPUs / 4 NICs = 1 rail each.
# Show hwloc a single GPU and num_groups becomes 1, so that GPU gets all 4.
# There is no NIXL-side knob for this -- max_bw_per_dram_seg is the only
# LIBFABRIC tunable and it is DRAM-only. See prune_hwloc_xml.py.
#
# This runs in the RANK half because lstopo has to describe the node it runs
# on, and each node builds its own.
if [ "${MODE}" = "solo-peak" ]; then
    for _tool in lstopo-no-graphics nvidia-smi; do
        command -v "${_tool}" >/dev/null 2>&1 || {
            echo "ERROR: --mode solo-peak needs ${_tool} on PATH." >&2; exit 1; }
    done

    SOLO_DIR="${TMPDIR:-/tmp}/nixl_solo_${PALS_APID:-$$}"
    mkdir -p "${SOLO_DIR}"

    # --filter io:all keeps every PCI device. hwloc's default IO filter is
    # "important only", and a GPU whose driver did not register an OSDev can
    # be dropped by it -- which would prune the topology for us, silently and
    # by the wrong criterion.
    lstopo-no-graphics --filter io:all --of xml > "${SOLO_DIR}/full.xml" 2>/dev/null \
        || lstopo-no-graphics --whole-io --of xml > "${SOLO_DIR}/full.xml"

    # Pick the GPU from nvidia-smi rather than from the XML, so the device we
    # KEEP and the device we point CUDA at cannot disagree. nvidia-smi prints
    # an 8-digit PCI domain (00000000:0F:00.0); hwloc prints 4 (0000:0f:00.0).
    # Normalise, or --keep will never match and the pruner will bail.
    SOLO_LINE=$(nvidia-smi --query-gpu=pci.bus_id,uuid --format=csv,noheader | sort | head -1)
    SOLO_BDF=$(printf '%s' "${SOLO_LINE%%,*}" | tr 'A-Z' 'a-z' \
               | sed -E 's/^0+([0-9a-f]{4}:)/\1/')
    SOLO_UUID=$(printf '%s' "${SOLO_LINE#*,}" | tr -d ' ')

    python3 "${SCRIPT_DIR}/prune_hwloc_xml.py" \
        "${SOLO_DIR}/full.xml" "${SOLO_DIR}/solo.xml" --keep "${SOLO_BDF}" >/dev/null

    export HWLOC_XMLFILE="${SOLO_DIR}/solo.xml"
    # By UUID, not by index: with the topology pruned, "GPU 0" means different
    # things to torch and to NIXL unless the binding is unambiguous.
    export CUDA_VISIBLE_DEVICES="${SOLO_UUID}"
    echo "solo-peak: kept GPU ${SOLO_BDF} (${SOLO_UUID}), HWLOC_XMLFILE=${HWLOC_XMLFILE}"
fi

exec python3 "${SCRIPT_DIR}/repro_nixl_2rank_transfer.py" "$@"
