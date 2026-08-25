#!/bin/bash
set -uo pipefail

# =============================================================================
# 1P1D NIXL cross-node smoke test (PBS) -- LIBFABRIC/CXI edition
#
# Verifies three separate, chained questions:
#   Q1  Did a KV-cache transfer actually happen (vs D silently doing its own
#       local prefill)?              -> /metrics diff on P and D, plus timing
#   Q2  Did the bytes actually cross Slingshot?
#                                     -> CXI hardware octet counters, sampled
#   Q3  Did it use the LIBFABRIC backend with the shim, not a fallback?
#                                     -> shim PATCH lines + rail-manager lines
#
# Q2/Q3 only mean something once Q1 has passed.
#
# Usage:
#   bash run_pd_full_test_N2_R1.sh
#   NIXL_BACKEND=UCX bash run_pd_full_test_N2_R1.sh      # explicit A/B
#   PROMPT_REPEAT=2000 bash run_pd_full_test_N2_R1.sh    # bigger KV payload
#
# -----------------------------------------------------------------------------
# WHY ONE mpiexec AND NOT TWO ssh -- the thing that makes this file different
# from every earlier revision.
#
# CXI needs a VNI. Without one, fi_domain() returns -FI_ENOSYS and the
# LIBFABRIC backend never comes up. The VNI arrives only in SLINGSHOT_VNIS,
# provisioned by PALS. Two `ssh -n NODE "bash launch_x.sh"` calls -- what this
# script used to do -- carry NO PALS environment at all, so both servers die
# at backend creation.
#
# The obvious repair (wrap each side in its own `mpiexec -n 1
# --single-node-vni`) is WORSE, and this is measured, not theorised. PALS
# allocates a VNI per APPLICATION, and a VNI is a traffic-isolation domain:
# endpoints on different VNIs cannot reach each other at all. Three separate
# launches were observed getting three different VNIs (1726, 1825, 1873 --
# the last from `-n 1 --single-node-vni`). Two independent launches would each
# build their agent happily and then never connect, trading today's loud
# immediate ENOSYS for a silent hang. See run_repro_nixl_2rank_transfer.sh:62.
#
# So: ONE `mpiexec -n 2 -ppn 1` runs ONE role script on both nodes, and that
# script picks prefill-vs-decode by HOSTNAME. Pairing by hostname rather than
# by PMI_RANK means whatever rank order PALS happens to hand out is
# irrelevant -- the same trick repro_nixl_2rank_transfer.py already uses.
#
# The proxy and this driver script stay OUTSIDE the mpiexec. They are plain
# HTTP and touch no fabric, so they need no VNI.
#
# -----------------------------------------------------------------------------
# WHY Q1 IS NO LONGER A TIMING TEST
#
# Earlier revisions asserted "disagg time_starttransfer should be noticeably
# LOWER than baseline". That is wrong, and it would report a HEALTHY run as a
# failure. Baseline is prefill-on-D. Disagg is prefill-on-P plus a KV transfer
# plus a proxy hop -- the same prefill work, with strictly more overhead
# around it. For a single request, correct 1P1D is EXPECTED TO BE SLOWER.
# Disaggregation buys throughput and ITL under load, not single-shot TTFT.
#
# The discriminating evidence is whether D skipped prefill. That is read off a
# before/after diff of both servers' /metrics endpoints, which is
# self-describing -- we print every counter that moved rather than guessing
# metric names that shift between vLLM releases. Timing is kept, demoted to a
# sanity check.
# =============================================================================
# CONFIRMED via check_cxi_libfabric.sh probe (both nodes, 2026-08-19): fi_info -p cxi
# works, libfabric 2.3.1, and NIXL's plugin manager lists LIBFABRIC as loadable in
# this exact conda env. Defaulting to it -- NIXL's own default backend is UCX, and
# plain UCX has no native Slingshot/CXI transport (see handoff doc). Override to
# UCX for an explicit A/B comparison once LIBFABRIC is confirmed working end-to-end.
NIXL_BACKEND=${NIXL_BACKEND:-LIBFABRIC}
MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
STAMP=$(date +%Y%m%d_%H%M%S)
# All run records live under one root rather than scattering pd_smoke_<stamp>
# directories directly into .../testing/, which also holds source checkouts and
# conda envs. Must be on a filesystem both compute nodes mount -- the remote
# CXI pollers append to it, and the remote role scripts read common_env.sh and
# launch_role.sh out of it.
RUNS_ROOT=${RUNS_ROOT:-/vast/draco/tara/projects/Tara_Deployment/software/testing/RUNS}
SHARED=${RUNS_ROOT}/pd_smoke_${STAMP}
mkdir -p ${SHARED}/logs
P_PORT=8100; D_PORT=8200; PROXY_PORT=8000
# Override with PROXY_SCRIPT=/your/path if your checkout lives elsewhere or
# a newer vllm version moves this file.
PROXY_SCRIPT=${PROXY_SCRIPT:-/vast/draco/tara/projects/Tara_Deployment/software/testing/vllm_0.27.1_08_18_2026/vllm/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# KV payload size. Qwen2.5-0.5B-Instruct: 24 layers x 2 KV heads x 64 head_dim
# x 2 (K and V) x 2 bytes fp16 = 12,288 bytes PER TOKEN. The old 6-token
# prompt therefore moved ~74 KB, which is invisible against background traffic
# in the CXI octet counters -- Q2 could not have produced a real signal at that
# size regardless of anything else. 500 repeats of a 9-word sentence is roughly
# 5k tokens ~= 60 MB, which is unambiguous. Raise PROMPT_REPEAT for a bigger
# payload; Qwen2.5's 32k context is the ceiling.
PROMPT_REPEAT=${PROMPT_REPEAT:-500}

# Wall-clock budget for a server to come up. Deliberately generous: on aarch64
# several of vLLM's CUDA extensions have no prebuilt wheel and JIT-compile on
# first use, which is minutes, not seconds. Only slow on a cold cache.
HEALTH_TRIES=${HEALTH_TRIES:-120}     # x5s = 10 min

# --- Resolve the two allocated nodes -----------------------------------------
cat ${PBS_NODEFILE}
mapfile -t NODES < <(sort -u "${PBS_NODEFILE}")
if [ ${#NODES[@]} -ne 2 ]; then
    echo "Expected exactly 2 unique nodes from PBS_NODEFILE, got ${#NODES[@]}. Aborting."
    exit 1
fi

THIS_HOST=$(hostname -s)
NODE_P=""; NODE_D=""
for n in "${NODES[@]}"; do
    if [[ "$n" == "${THIS_HOST}"* ]]; then NODE_P="$n"; else NODE_D="$n"; fi
done
if [ -z "${NODE_P}" ] || [ -z "${NODE_D}" ]; then
    echo "WARNING: couldn't match current hostname (${THIS_HOST}) against PBS_NODEFILE."
    echo "Falling back to array order -- verify NODE_P/NODE_D below are correct."
    NODE_P="${NODES[0]}"; NODE_D="${NODES[1]}"
fi

# PBS_NODEFILE carries FQDNs (x4820c7s0b0n0.hostmgmt2820.north.tara.alcf.anl.gov)
# but `hostname -s` on the compute node returns only the leading label. Keep
# BOTH forms explicitly: ssh wants the FQDN, and launch_role.sh's hostname
# comparison has to be short-vs-short or it never matches. This is not
# hypothetical -- comparing a short hostname against the FQDN sent both ranks
# down the "unexpected host" branch and killed the whole application before
# either server started. %%.* is a no-op if PBS ever hands back short names.
NODE_P_SHORT="${NODE_P%%.*}"
NODE_D_SHORT="${NODE_D%%.*}"
echo "Prefill node: ${NODE_P}   Decode node: ${NODE_D}"
echo "  (short forms, used for the in-rank hostname match: ${NODE_P_SHORT} / ${NODE_D_SHORT})"

# --- Resolve hsn0 IPs (also validates SSH works before we launch anything) --
get_hsn_ip() {
    ssh -n "$1" "ip -4 -o addr show hsn0" | awk '{print $4}' | cut -d/ -f1
}
P_IP=$(get_hsn_ip "${NODE_P}")
D_IP=$(get_hsn_ip "${NODE_D}")
echo "Prefill hsn0: ${P_IP}   Decode hsn0: ${D_IP}"
if [ -z "${P_IP}" ] || [ -z "${D_IP}" ]; then
    echo "Failed to resolve hsn0 IP on one or both nodes. Stop here, don't proceed blind."
    exit 1
fi

# --- Confirm shared FS is actually visible on the remote node ---------------
ssh -n "${NODE_D}" "test -d ${SHARED}" || { echo "SHARED path not visible on ${NODE_D} -- is /vast/draco mounted there?"; exit 1; }

# --- Locate the shim and the env script, IN PLACE in the repo ----------------
# Used straight from SCRIPT_DIR, not copied into SHARED. An earlier revision
# copied them and justified it as "both nodes then see byte-identical files",
# which is vacuous here: the repo lives on /vast, so both nodes already see the
# same file. What the copy actually bought was a way to be running a DIFFERENT
# binary than the one you rebuild and hand-test in the repo -- rebuild the
# shim, LD_PRELOAD the repo copy by hand to check it, and the script would
# still be preloading a snapshot from staging. The working benchmark
# (run_repro_nixl_2rank_transfer.sh:189) references SCRIPT_DIR directly; match
# it, for the same reason common_env.sh now sources the shared env script
# rather than duplicating it.
#
# Provenance is kept by recording a checksum below instead of by hoarding a
# copy of the binary in every timestamped run directory.
for f in fi_getinfo_shim.so env_for_libfabric_topology_error.sh; do
    if [ ! -f "${SCRIPT_DIR}/${f}" ]; then
        echo "Missing ${SCRIPT_DIR}/${f} -- cannot continue."
        [ "${f}" = "fi_getinfo_shim.so" ] && echo "  Build it: gcc -shared -fPIC -o fi_getinfo_shim.so fi_getinfo_shim.c -ldl \$(pkg-config --cflags libfabric)"
        exit 1
    fi
done
SHIM="${SCRIPT_DIR}/fi_getinfo_shim.so"
ENV_SCRIPT="${SCRIPT_DIR}/env_for_libfabric_topology_error.sh"

# Referencing in place moves the burden onto SCRIPT_DIR being mounted on the
# compute nodes, which the copy used to paper over. So check it, rather than
# discover it as an LD_PRELOAD that silently does nothing. test -f on the shim
# itself, not test -d on the directory -- a stale automount can satisfy the
# latter. Checked on NODE_D because this script runs on NODE_P.
ssh -n "${NODE_D}" "test -f ${SHIM}" || {
    echo "${SHIM} is not visible from ${NODE_D}."
    echo "The repo has to live on a filesystem both compute nodes mount"
    echo "(e.g. /vast/draco/...), not in a node-local or unmounted home."
    exit 1
}

# The tracked .so has gone stale before -- committed carrying PATCH 1 only
# while the .c had already grown PATCHES 2-4. That build clears BLOCKER 1
# (mr_mode==0 -> -FI_ENODATA) and then dies later in registerMem with "no
# available backends for mem type 'VRAM_SEG'", which reads as "the shim is
# broken" rather than "the shim is old". Cheap to rule out up front.
N_PATCH=$(strings "${SHIM}" 2>/dev/null | grep -c 'PATCH [234]' || true)
if [ "${N_PATCH}" -eq 0 ]; then
    echo "*** STALE SHIM: ${SHIM} has no PATCH 2/3/4 markers.                       ***"
    echo "*** It carries BLOCKER 1 only; --mem cuda / VRAM registration will fail.  ***"
    echo "*** Rebuild on this node before running:                                  ***"
    echo "***   gcc -shared -fPIC -o fi_getinfo_shim.so fi_getinfo_shim.c -ldl \$(pkg-config --cflags libfabric)"
    exit 1
fi

# Provenance, in place of the copy: which binary ran, and was it the one you
# think you built. Written to the run directory as well as stdout so an old
# SHARED dir can still be matched against a rebuilt shim later.
{
    echo "shim:      ${SHIM}"
    echo "sha256:    $(sha256sum "${SHIM}" 2>/dev/null | awk '{print $1}')"
    echo "built:     $(date -r "${SHIM}" '+%Y-%m-%d %H:%M:%S' 2>/dev/null)"
    echo "env:       ${ENV_SCRIPT}"
    echo "env md5:   $(md5sum "${ENV_SCRIPT}" 2>/dev/null | awk '{print $1}')"
} | tee ${SHARED}/provenance.txt
echo "Shim OK (PATCH 2/3/4 markers present), visible from ${NODE_D}."

NO_PROXY_LIST="localhost,127.0.0.1,${P_IP},${D_IP},${NODE_P},${NODE_D},${NODE_P_SHORT},${NODE_D_SHORT}"

# --- Common env, sourced by both role scripts --------------------------------
# UCX-only tunables are gated on NIXL_BACKEND so the A/B still works, but they
# are dead weight on the LIBFABRIC path and were previously set unconditionally
# -- which made LIBFABRIC runs look UCX-configured in the logs.
UCX_LINES=""
if [ "${NIXL_BACKEND}" = "UCX" ]; then
    UCX_LINES='export UCX_TLS=cuda_copy,cuda_ipc,sm,tcp,self
export UCX_MODULE_DIR=$(python3 -c "import site,glob; print(glob.glob(site.getsitepackages()[0]+'"'"'/nixl_cu13.libs/ucx'"'"')[0])")'
fi

# GPU_ID intentionally NOT set by default -- CUDA_VISIBLE_DEVICES is left
# unset, so all GPUs on the node stay visible to the process. Was previously
# hardcoded to 0; removed per explicit ask, for two reasons:
#   1. Hardcoding to GPU 0 is exactly the kind of thing that silently breaks
#      once this script grows into the real sweep (Section 6) and needs a
#      specific GPU per role on a multi-GPU node -- easy to forget it's there.
#   2. Was: "logical inference, NOT confirmed" that restricting GPU visibility
#      could hand NIXL's rail selection a degenerate view of the topology.
#      NOW MEASURED, and the concern does not apply: the LIBFABRIC backend
#      reads its topology from hwloc (the whole node, always) and resolves
#      rails from the GPU's real PCI bus id, which CUDA_VISIBLE_DEVICES does
#      not change. Pinning one GPU gets you that GPU's one rail -- which is
#      the correct and expected 1-rail/25 GB/s behaviour for a TP=1 worker,
#      not a degraded topology. (The separate experiment that DID try to
#      widen a single GPU's rail count, by pruning the hwloc XML, is
#      --mode solo-peak in the benchmark, and it is a documented negative:
#      the NIC split is per PCIe complex, so pruning cannot help.)
# To pin a specific GPU for a future run, set GPU_ID rather than editing this
# file, e.g.: GPU_ID=2 bash run_pd_full_test_N2_R1.sh
GPU_PIN_LINE=""
if [ -n "${GPU_ID:-}" ]; then
    GPU_PIN_LINE="export CUDA_VISIBLE_DEVICES=${GPU_ID}"
    echo "GPU_ID=${GPU_ID} set -- pinning CUDA_VISIBLE_DEVICES=${GPU_ID} for both P and D."
fi

cat > ${SHARED}/common_env.sh <<EOF
export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
export NO_PROXY=${NO_PROXY_LIST}
export no_proxy=${NO_PROXY_LIST}

# Conda activation and the NVHPC CUDA_HOME fixup both come from
# env_for_libfabric_topology_error.sh in the repo -- the SAME file, at the same
# path, that the working NIXL benchmark sources. Previously this script carried
# its own inline duplicate of that logic, which is exactly how the two drift
# apart and how a run that "uses the same environment" quietly stops doing so.
# Sourced from SCRIPT_DIR rather than a staged copy for the same reason.
#
# set +u across the source for the same reason the benchmark does it: conda's
# activation machinery reads unset variables, which is fatal under nounset.
set +u
source ${ENV_SCRIPT}
set -u

export HF_TOKEN=\$(cat ~/.hf_token)
export PYTHONNOUSERSITE=1
export HF_HOME=/vast/draco/tara/projects/Tara_Deployment/software/model-weights
export HF_DATASETS_CACHE=\${HF_HOME}
export HF_MODULES_CACHE=\${HF_HOME}
export RAY_TMPDIR=/tmp
export TMPDIR=/tmp
export VLLM_LOGGING_LEVEL=DEBUG
${UCX_LINES}
${GPU_PIN_LINE}
EOF

# Appended with a QUOTED heredoc delimiter ('EOF') so the paths below are
# resolved when common_env.sh actually RUNS on the compute node, not now on
# whichever node this launcher script happens to be running on -- login and
# compute node module environments aren't guaranteed to match on Cray systems.
#
# Everything the old inline block did -- CC/CXX/CUDAHOSTCXX=nvc++, CUDA_HOME,
# LIBRARY_PATH, LD_LIBRARY_PATH, CPATH and the math_libs LIB dir -- now comes
# from env_for_libfabric_topology_error.sh, sourced above. This appendix keeps
# the ONE thing that file does not do.
cat >> ${SHARED}/common_env.sh <<'EOF'

# CONFIRMED (not guessed) from an actual "curandStatePhilox4_32_10_t
# undefined" failure: the CUDA math-library HEADERS -- curand, and by strong
# implication cublas/cusparse/cusolver/cufft -- live in a SEPARATE sibling
# tree, math_libs/<ver>/targets/<arch>/include, not inside cuda/<ver>/ at all.
# env_for_libfabric_topology_error.sh puts the math_libs LIB dir on
# LIBRARY_PATH/LD_LIBRARY_PATH but never adds its INCLUDE dir to CPATH, so a
# JIT compile that needs curand_kernel.h still fails without this block.
#
# Derived from CUDA_HOME (which that file exports) rather than from its
# internal shell variables, so this stays correct if it is ever refactored.
if [ -n "${CUDA_HOME:-}" ]; then
    _REAL_CUDA_LIB_DIR=$(find "${CUDA_HOME}" -name "libcudart.so" -printf '%h\n' 2>/dev/null | head -1)
    if [ -n "${_REAL_CUDA_LIB_DIR}" ]; then
        _TARGET_ROOT=$(dirname "${_REAL_CUDA_LIB_DIR}")           # .../targets/<arch>
        _NVHPC_VER_ROOT=$(dirname "$(dirname "${CUDA_HOME}")")    # .../<ver>
        _MATH_INC="${_NVHPC_VER_ROOT}/math_libs/$(basename "${CUDA_HOME}")/targets/$(basename "${_TARGET_ROOT}")/include"
        if [ -d "${_MATH_INC}" ]; then
            export CPATH="${_MATH_INC}:${CPATH:-}"
        else
            echo "WARNING: math_libs include dir not found at ${_MATH_INC}" >&2
            echo "         A JIT build needing curand_kernel.h will fail." >&2
        fi
    fi
fi
EOF

# --no-enable-prefix-caching matters here specifically: without it, if the
# baseline query (direct-to-D) and the disagg query (via proxy) use the same
# prompt, D could serve the second one from ITS OWN local prefix cache
# instead of actually pulling KV from the producer -- which would look fast
# for the wrong reason and silently invalidate the Q1 timing comparison.
# NIXL backend selection lives in kv_connector_extra_config.backends (per vLLM's
# NixlConnector docs). Leaving this unset defaults NIXL to UCX -- which is the
# whole reason last session's run couldn't have passed Q3 regardless of UCX_TLS
# tuning. NIXL_BACKEND=UCX still works if you want an explicit A/B later.
KV_XFER_CONFIG_P="{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_producer\",\"kv_connector_extra_config\":{\"backends\":[\"${NIXL_BACKEND}\"]}}"
KV_XFER_CONFIG_D="{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_consumer\",\"kv_connector_extra_config\":{\"backends\":[\"${NIXL_BACKEND}\"]}}"

# ONE role script for both nodes, selecting by hostname. See the VNI section
# in the header for why this cannot be two independent launches.
#
# LD_PRELOAD is set HERE and not in common_env.sh on purpose. common_env.sh is
# also sourced by launch_proxy.sh, and the shim has no business interposing
# fi_getinfo for a plain HTTP process. Setting it immediately before the exec
# also keeps it out of every helper command in this shell.
#
# It does need to survive into vLLM's children -- EngineCore is a separate
# process -- but LD_PRELOAD is a normal environment variable and is inherited
# across fork/exec, so no special handling is required. The shim announces
# itself on stderr, so `grep fi_getinfo_shim` on the log is free confirmation
# rather than an assumption; Q3 below does exactly that.
cat > ${SHARED}/launch_role.sh <<EOF
#!/bin/bash
source ${SHARED}/common_env.sh

# Short-vs-short. \$(hostname -s) is already unqualified on this system, but
# strip any domain anyway so the comparison holds either way -- getting this
# wrong is silent and total: both ranks fall through to the else branch, the
# first one to exit takes the whole PALS application down with it, and neither
# p.log nor d.log is ever written.
MY_HOST="\$(hostname -s)"
MY_HOST="\${MY_HOST%%.*}"
if [ "\${MY_HOST}" = "${NODE_P_SHORT}" ]; then
    ROLE=p
    LOG=${SHARED}/logs/p.log
    PORT=${P_PORT}
    SIDE_HOST=${P_IP}
    SIDE_PORT=5600
    KV_CFG='${KV_XFER_CONFIG_P}'
elif [ "\${MY_HOST}" = "${NODE_D_SHORT}" ]; then
    ROLE=d
    LOG=${SHARED}/logs/d.log
    PORT=${D_PORT}
    SIDE_HOST=${D_IP}
    SIDE_PORT=5601
    KV_CFG='${KV_XFER_CONFIG_D}'
else
    echo "Rank landed on unexpected host '\${MY_HOST}'." >&2
    echo "  expected one of: '${NODE_P_SHORT}' (prefill) or '${NODE_D_SHORT}' (decode)" >&2
    echo "  full node names from PBS_NODEFILE: ${NODE_P} / ${NODE_D}" >&2
    exit 1
fi

exec > "\${LOG}" 2>&1
echo "=== role=\${ROLE} host=\$(hostname -s) SLINGSHOT_VNIS=\${SLINGSHOT_VNIS:-<UNSET>} ==="
if [ -z "\${SLINGSHOT_VNIS:-}" ]; then
    echo "SLINGSHOT_VNIS is EMPTY. The cxi provider will fail fi_domain() with"
    echo "-FI_ENOSYS and the LIBFABRIC backend will not come up. This means the"
    echo "process was not launched under PALS -- check the mpiexec invocation."
fi

export VLLM_NIXL_SIDE_CHANNEL_HOST=\${SIDE_HOST}
export VLLM_NIXL_SIDE_CHANNEL_PORT=\${SIDE_PORT}
export NIXL_LOG_LEVEL=\${NIXL_LOG_LEVEL:-INFO}
export LD_PRELOAD=${SHIM}

exec vllm serve ${MODEL} --host 0.0.0.0 --port \${PORT} \\
    --gpu-memory-utilization 0.3 \\
    --no-enable-prefix-caching \\
    --kv-transfer-config "\${KV_CFG}"
EOF
echo "NIXL backend for this run: ${NIXL_BACKEND}"

# --- Launch both, from ONE application so they share a VNI -------------------
# -ppn 1 with 2 nodes puts exactly one server on each. Rank-to-node assignment
# is PALS's business; the role script sorts itself out by hostname.
#
# The logs are opened by the role script itself rather than redirected here,
# because one mpiexec has one stdout and both servers would otherwise
# interleave into a single unsplittable stream.
touch ${SHARED}/logs/p.log ${SHARED}/logs/d.log   # avoid a race with the tails below
mpiexec -n 2 -ppn 1 bash ${SHARED}/launch_role.sh > ${SHARED}/logs/mpiexec.log 2>&1 &
MPIEXEC_PID=$!

# Stream both server logs live, prefixed by role so interleaved output stays
# readable. `sed -u` (unbuffered) matters here -- without it, output piped
# through sed gets block-buffered and shows up in silent bursts instead of
# line-by-line, which looks exactly like "nothing is happening" even when it is.
#
# disown after each: cleanup() kills these, and without disown bash reports
# every one as a job-status line ("PID Killed exit 1") mixed into the teardown
# output, which reads like a fresh error at exactly the moment you are trying
# to work out what the real one was. Disowning only drops them from the job
# table; kill -TERM on the saved PID still works.
tail -n +1 -f ${SHARED}/logs/p.log | sed -u 's/^/[P] /' &
TAIL_P_PID=$!; disown ${TAIL_P_PID}
tail -n +1 -f ${SHARED}/logs/d.log | sed -u 's/^/[D] /' &
TAIL_D_PID=$!; disown ${TAIL_D_PID}
# mpiexec's own stream carries PALS errors that never reach either role log --
# a launch that dies before the role script runs is otherwise completely silent.
tail -n +1 -f ${SHARED}/logs/mpiexec.log | sed -u 's/^/[MPI] /' &
TAIL_M_PID=$!; disown ${TAIL_M_PID}

# pgrep/pkill patterns are bracketed ('[v]llm serve', not 'vllm serve') for a
# specific reason. `ssh host "pkill -f 'vllm serve'"` runs the command through
# `bash -c "pkill -f 'vllm serve'"` on the far side, and THAT shell's command
# line contains the literal string being searched for -- so pgrep/pkill match
# their own parent. Observed: cleanup reported "Stray vllm process ... force
# killing" on a run where no server ever started, then killed the wrapper shell
# and took the local ssh child down with it ("Killed exit 1" in the teardown
# output). The bracket makes the regex match the real process but not the
# command line containing the regex.
PAT_VLLM='[v]llm serve'
PAT_ENGINE='[E]ngineCore'
PAT_PROXY='[t]oy_proxy_server.py'

cleanup() {
    echo "=== Cleaning up ==="
    # CXI poll loops are setsid'd specifically so they survive an SSH session
    # closing -- which means they ALSO survive this script dying early, same
    # orphan-process problem as EngineCore below. Stop marker is a no-op if
    # the poll never started (POLL_STOP_MARKER unset under `set -u` is guarded
    # with :-).
    touch "${POLL_STOP_MARKER:-}" 2>/dev/null
    # SIGTERM the actual remote processes first, so each engine gets a real
    # chance to release its NIXL registrations and GPU memory cleanly before
    # anything gets SIGKILLed. Killing the launcher alone is not enough and
    # never was: with the old two-ssh launch there was no pty for a signal to
    # travel through, and with mpiexec, PALS's own teardown is not guaranteed
    # to reach a grandchild EngineCore before this script exits.
    for n in "${NODE_P}" "${NODE_D}"; do
        ssh -n "$n" "pkill -TERM -f \"${PAT_VLLM}\"" 2>/dev/null
    done
    ssh -n "${NODE_P}" "pkill -TERM -f \"${PAT_PROXY}\"" 2>/dev/null
    # :- guards matter here: cleanup can fire (via the EXIT trap) before the
    # proxy block ever runs -- e.g. if P or D fails its health check and the
    # script exits early -- in which case PROXY_SSH_PID/TAIL_PROXY_PID were
    # never assigned, and under `set -u` referencing them bare would abort
    # cleanup() partway through, skipping everything after that line.
    kill -TERM ${MPIEXEC_PID:-} ${TAIL_P_PID:-} ${TAIL_D_PID:-} ${TAIL_M_PID:-} ${PROXY_SSH_PID:-} ${TAIL_PROXY_PID:-} 2>/dev/null
    sleep 5
    # Belt-and-suspenders: an orphaned EngineCore child process (spawned
    # via multiprocessing) does NOT match a 'vllm serve' name pattern --
    # CONFIRMED via `top`, its process name shows as 'VLLM::EngineCor',
    # not anything containing 'vllm serve'. This is why it can survive
    # the pkill above and be left holding the NIXL side-channel port,
    # which is exactly what caused an "Address already in use" on a
    # later run. Kill both patterns, and free the known ports directly
    # too rather than relying on process-name matching alone.
    for n in "${NODE_P}" "${NODE_D}"; do
        ssh -n "$n" "pkill -KILL -f \"${PAT_ENGINE}\"" 2>/dev/null
    done
    ssh -n "${NODE_P}" "pkill -KILL -f \"${PAT_PROXY}\"" 2>/dev/null
    ssh -n "${NODE_P}" "fuser -k 5600/tcp" 2>/dev/null
    ssh -n "${NODE_D}" "fuser -k 5601/tcp" 2>/dev/null
    ssh -n "${NODE_P}" "fuser -k ${PROXY_PORT}/tcp" 2>/dev/null
    for n in "${NODE_P}" "${NODE_D}"; do
        if ssh -n "$n" "pgrep -f \"${PAT_VLLM}\"" > /dev/null 2>&1; then
            echo "Stray vllm process on $n -- force killing."
            ssh -n "$n" "pkill -KILL -f \"${PAT_VLLM}\""
        fi
    done
}
trap cleanup EXIT INT TERM

wait_healthy() {
    local ip=$1 port=$2 name=$3 path=${4:-/health}
    for i in $(seq 1 ${HEALTH_TRIES}); do
        if curl -s -o /dev/null -w "%{http_code}" http://${ip}:${port}${path} 2>/dev/null | grep -q 200; then
            echo "${name} healthy after $((i*5))s"; return 0
        fi
        # Fail fast instead of burning the full timeout: if mpiexec is already
        # gone, PALS tore the application down and no amount of waiting will
        # produce a healthy server. Skipped for the proxy, which is ssh-launched
        # and has no relationship to MPIEXEC_PID.
        if [ "${name#Proxy}" = "${name}" ] && ! kill -0 ${MPIEXEC_PID} 2>/dev/null; then
            echo "mpiexec exited while waiting for ${name}."
            echo "Look at ${SHARED}/logs/mpiexec.log first -- a PALS-level failure"
            echo "(bad VNI, node not in the allocation, launcher error) never"
            echo "reaches p.log or d.log at all."
            return 1
        fi
        sleep 5
    done
    echo "${name} never became healthy -- check ${SHARED}/logs/"; return 1
}
wait_healthy ${P_IP} ${P_PORT} "Prefill (${NODE_P})" || exit 1
wait_healthy ${D_IP} ${D_PORT} "Decode (${NODE_D})"  || exit 1

# =============================================================================
# Q3 (startup half) -- did the LIBFABRIC path actually come up, with the shim?
#
# Checked HERE, before any request, because all three of these are decided at
# engine init. If the shim never loaded or the backend fell back, there is no
# point sending traffic and then puzzling over the counters.
# =============================================================================
echo "=== Q3: transport evidence at startup ==="
for role in p d; do
    log=${SHARED}/logs/${role}.log
    echo "  --- ${role} ---"
    vni=$(grep -m1 -o 'SLINGSHOT_VNIS=[^ ]*' "${log}" 2>/dev/null)
    echo "      ${vni:-SLINGSHOT_VNIS=<line not found>}"
    n_shim=$(grep -c 'fi_getinfo_shim' "${log}" 2>/dev/null || true)
    echo "      shim announcements: ${n_shim}   (0 means LD_PRELOAD did not reach the engine)"
    grep -m3 -i 'System runtime:.*NVIDIA GPU\|rail\|LIBFABRIC' "${log}" 2>/dev/null \
        | sed 's/^/      /' || echo "      (no LIBFABRIC/rail lines -- is NIXL_LOG_LEVEL=INFO set?)"
done
echo "  A healthy LIBFABRIC run shows a non-empty VNI, a non-zero shim count, and"
echo "  the rail-manager's GPU/rail lines. Zero shim announcements with a working"
echo "  server means it came up on UCX, and Q2's counters will be misleading."
echo ""

# =============================================================================
# Q1 -- baseline: query D directly, bypassing the proxy entirely.
# This is D doing its own full local prefill, same as a standalone instance.
#
# NOTE the prompt size. The old 6-token prompt moved ~74 KB of KV, far below
# the noise floor of the CXI counters. See PROMPT_REPEAT at the top.
# =============================================================================
PROMPT=$(python3 -c "print(('The quick brown fox jumps over the lazy dog. ' * ${PROMPT_REPEAT}).strip())")
echo "Prompt: ${PROMPT_REPEAT} repeats, $(printf %s "${PROMPT}" | wc -c) chars"
echo "  ~$(( $(printf %s "${PROMPT}" | wc -c) / 4 )) tokens -> ~$(( $(printf %s "${PROMPT}" | wc -c) / 4 * 12288 / 1000000 )) MB of KV at 12,288 B/token"

# Both servers' /metrics, as a name->value map. Diffed around the disagg
# request to answer Q1 without hardcoding metric names: vLLM renames these
# between releases, and printing everything that MOVED is both more robust and
# more informative than asserting on one counter we guessed at.
snapshot_metrics() {
    local ip=$1 port=$2 out=$3
    curl -s "http://${ip}:${port}/metrics" 2>/dev/null \
        | grep -v '^#' | awk '{print $1" "$2}' | sort > "${out}"
}
# Grouped, and the vllm: counters are UNCAPPED. This used to be a flat
# `head -40` over everything that moved, which was actively harmful: /metrics
# is alphabetical, so http_* and process_* sort ahead of vllm:*, and 40 lines
# of http_request_duration histogram buckets consumed the entire budget. On the
# first passing run that hid vllm:prompt_tokens_total -- the single counter Q1
# is actually about -- while printing 20 near-identical bucket lines. Histogram
# buckets are the noise here and named counters are the signal, so split them.
diff_metrics() {
    local before=$1 after=$2 label=$3
    local moved="${after}.moved"
    join "${before}" "${after}" 2>/dev/null \
        | awk '$2 != $3 {printf "%-58s %s -> %s\n", $1, $2, $3}'  > "${moved}"
    join -v2 "${before}" "${after}" 2>/dev/null \
        | awk '{printf "%-58s (new) %s\n", $1, $2}'              >> "${moved}"

    echo "  -- ${label} --"
    echo "     vllm counters that moved (uncapped -- these are what Q1 turns on):"
    grep '^vllm:' "${moved}" | grep -v '_bucket' | sed 's/^/       /'
    echo "     (plus $(grep -c '^vllm:.*_bucket' "${moved}") vllm histogram buckets)"
    echo "     non-vllm counters that moved (first 12 of $(grep -cv '^vllm:' "${moved}")):"
    grep -v '^vllm:' "${moved}" | head -12 | sed 's/^/       /'
    echo "     full list: ${moved}"
}

# Body built by python3 -- json.dump, not shell interpolation. At 500 repeats
# the prompt is ~22 KB, well past the point where hand-escaping it into a
# -d "{...}" string is worth the risk, and python3 is guaranteed present here
# (the conda env is already active).
BODY_JSON=${SHARED}/req_body.json
python3 -c "
import json, sys
json.dump({'model': sys.argv[1], 'prompt': sys.argv[2], 'max_tokens': 10, 'stream': True},
          open(sys.argv[3], 'w'))
" "${MODEL}" "${PROMPT}" "${BODY_JSON}"

send_and_time() {
    local ip=$1 port=$2 label=$3
    local out=${SHARED}/logs/resp_${label}.txt
    local t
    t=$(curl -s -o ${out} -w "%{time_starttransfer}" http://${ip}:${port}/v1/completions \
        -H "Content-Type: application/json" --data-binary @${BODY_JSON})
    echo "${label}: time_starttransfer=${t}s  (raw response: ${out})" \
        | tee -a ${SHARED}/logs/timings.txt
}

M=${SHARED}/logs/metrics
echo "=== Q1a: baseline -- direct to decode instance, no proxy ==="
echo "  (P is NOT in this path at all. Whatever P's counters do here is the"
echo "   background floor that Q1b has to beat.)"
snapshot_metrics ${P_IP} ${P_PORT} ${M}_p_a_before
snapshot_metrics ${D_IP} ${D_PORT} ${M}_d_a_before
send_and_time ${D_IP} ${D_PORT} "baseline_decode_alone"
snapshot_metrics ${P_IP} ${P_PORT} ${M}_p_a_after
snapshot_metrics ${D_IP} ${D_PORT} ${M}_d_a_after
diff_metrics ${M}_p_a_before ${M}_p_a_after "Q1a PREFILL node -- expected: nothing moves"
diff_metrics ${M}_d_a_before ${M}_d_a_after "Q1a DECODE node -- expected: a full local prefill+decode"

# =============================================================================
# Direct P hop -- replicates the proxy's own first request to P by hand,
# BEFORE the proxy is even involved. Checks whether P's completion response
# actually contains real transfer coordinates (remote_engine_id/
# remote_block_ids/remote_host/remote_port) or leaves them null. Isolates
# whether a problem lives in P's own NixlConnector response versus
# downstream in the proxy's relay logic -- the proxy's own
# send_request_to_service() sends exactly this shape.
# =============================================================================
echo "=== Direct P hop: replicating the proxy's first request by hand ==="
# Same python-built-body reason as send_and_time: the prompt is now ~22 KB and
# has no business being pasted into a shell-quoted JSON literal.
PHOP_JSON=${SHARED}/req_body_phop.json
python3 -c "
import json, sys
json.dump({
    'model': sys.argv[1], 'prompt': sys.argv[2], 'max_tokens': 1, 'stream': False,
    'kv_transfer_params': {
        'do_remote_decode': True, 'do_remote_prefill': False,
        'remote_engine_id': None, 'remote_block_ids': None,
        'remote_host': None, 'remote_port': None,
    },
}, open(sys.argv[3], 'w'))
" "${MODEL}" "${PROMPT}" "${PHOP_JSON}"

curl -s http://${P_IP}:${P_PORT}/v1/completions \
    -H "Content-Type: application/json" \
    --data-binary @${PHOP_JSON} \
    | tee ${SHARED}/logs/resp_direct_p_hop.json | python3 -m json.tool 2>/dev/null \
    || echo "(response wasn't valid JSON -- see ${SHARED}/logs/resp_direct_p_hop.json raw)"
echo "=== Check kv_transfer_params above: real remote_engine_id/remote_block_ids/"
echo "remote_host/remote_port means P is populating it correctly; any of"
echo "those still null/None means the gap is upstream of the proxy entirely."
echo ""

# =============================================================================
# PROXY -- wired to the real reference implementation
# (tests/v1/kv_connector/nixl_integration/toy_proxy_server.py). Its health
# endpoint is /healthcheck, not /health like vLLM's own servers -- confirmed
# by reading the script, not assumed. It also sends
# "Authorization: Bearer $OPENAI_API_KEY" on every request to P and D; our
# vllm serve instances were never launched with --api-key so this shouldn't
# be validated either way, but set a real value rather than leave it to
# "Bearer None" on an unvalidated assumption. --host 0.0.0.0 is explicit
# rather than relying on the script's default 127.0.0.1 (loopback-only) --
# same "bound somewhere unreachable by default" pattern that bit the NIXL
# side-channel host earlier.
# =============================================================================
if [ ! -f "${PROXY_SCRIPT}" ]; then
    echo "PROXY_SCRIPT not found at ${PROXY_SCRIPT}."
    echo "Set PROXY_SCRIPT=/actual/path and rerun."
    exit 1
fi

cat > ${SHARED}/launch_proxy.sh <<EOF
#!/bin/bash
source ${SHARED}/common_env.sh
export OPENAI_API_KEY=smoke-test-dummy-key
python3 ${PROXY_SCRIPT} \
    --host 0.0.0.0 --port ${PROXY_PORT} \
    --prefiller-host ${P_IP} --prefiller-port ${P_PORT} \
    --decoder-host ${D_IP} --decoder-port ${D_PORT}
EOF

ssh -n "${NODE_P}" "bash ${SHARED}/launch_proxy.sh" > ${SHARED}/logs/proxy.log 2>&1 &
PROXY_SSH_PID=$!
touch ${SHARED}/logs/proxy.log
tail -n +1 -f ${SHARED}/logs/proxy.log | sed -u 's/^/[PROXY] /' &
TAIL_PROXY_PID=$!
wait_healthy ${P_IP} ${PROXY_PORT} "Proxy (${NODE_P})" "/healthcheck" || exit 1

echo ""
echo "=== Proxy wired in -- running Q1b/Q2 for real ==="
echo ""

# --- Q2 (legacy, kept as a cheap secondary signal) ---------------------------
# CAVEAT (confirmed architecture fact, not just a small-model noise problem):
# CXI RDMA is exposed via a kernel-bypass character device, separate from the
# hsn0 netdevice/classical-Ethernet path these /proc/net/dev counters read.
# A real CXI RDMA transfer may not register here at all, regardless of model
# size -- this is why the CXI sysfs polling below is now the primary check,
# not this one.
snapshot_counters() {
    local tag=$1
    for n in "${NODE_P}" "${NODE_D}"; do
        ssh -n "$n" "echo '--- $n ($tag) ---'; grep -E 'hsn0|bond0' /proc/net/dev"
    done
}

# --- Q2, real signal: CXI hardware octet counters, sampled DURING the request,
# not just before/after. Confirmed real path + counter names via
# check_cxi_libfabric.sh probe on both nodes (2026-08-19):
#   /sys/class/cxi/cxi<0-3>/device/telemetry/hni_sts_{tx,rx}_ok_octets
# Polls all 4 CXI devices per node (don't yet know which one is rail-aligned
# to CUDA_VISIBLE_DEVICES=0 on this system -- cheaper to poll all 4 and let
# the data show which moved than to assume cxi0<->GPU0).
cat > ${SHARED}/poll_cxi.sh <<'POLLEOF'
#!/bin/bash
STOP_MARKER="$1"
OUT="$2"
while [ ! -f "${STOP_MARKER}" ]; do
    ts=$(date +%s.%N)
    for i in 0 1 2 3; do
        tx=$(cat /sys/class/cxi/cxi${i}/device/telemetry/hni_sts_tx_ok_octets 2>/dev/null)
        rx=$(cat /sys/class/cxi/cxi${i}/device/telemetry/hni_sts_rx_ok_octets 2>/dev/null)
        echo "${ts},cxi${i},tx=${tx},rx=${rx}" >> "${OUT}"
    done
    sleep 0.05
done
POLLEOF

POLL_STOP_MARKER="${SHARED}/poll_stop_${STAMP}"
start_cxi_poll() {
    local node=$1 outfile=$2
    : > "${outfile}"
    # setsid+nohup+disown so the loop survives this ssh session closing --
    # same class of issue the handoff doc already hit with EngineCore cleanup
    # (signaling the local ssh client PID doesn't reach the remote process).
    ssh -n "$node" "setsid nohup bash ${SHARED}/poll_cxi.sh ${POLL_STOP_MARKER} ${outfile} > /dev/null 2>&1 < /dev/null &" 
}
stop_cxi_poll() {
    touch "${POLL_STOP_MARKER}"
    sleep 0.3   # >1 poll interval, let both remote loops notice and exit
}

# Wait for the counters to stop moving before stopping the pollers. A 200 ms
# fixed sleep (what this used to do) truncates the transfer: the KV pull is
# asynchronous with respect to the HTTP response, and the sysfs telemetry is
# itself sampled by firmware on its own cadence, so the last octets of a
# transfer land well after curl returns. The benchmark hit exactly this and
# solved it the same way (read_cxi_counters_settled in
# repro_nixl_2rank_transfer.py) -- sample until quiescent, don't guess a sleep.
#
# "Quiet" is a THRESHOLD, not equality. The first version required the counters
# to be byte-identical across three consecutive samples, which can never happen:
# these are node-wide NIC counters, and this very script is curling /metrics and
# ssh-ing across the same NIC while it waits. Observed on the first passing run
# -- it reported "STILL MOVING ... deltas are a lower bound" at the cap, when
# the transfer had in fact completed and the deltas were accurate to 2.2%.
# Crying wolf on a good run is worse than not checking at all.
#
# The threshold separates cleanly: background chatter is a few hundred KB per
# 250 ms window, while the transfer itself is ~60 MB in ~10 ms -- any window
# overlapping it is orders of magnitude above 1 MB.
#
# Summing in awk is safe despite these being 64-bit counters: eight values of
# ~1e12 sum to ~1e13, and a double is exact to 9e15.
#
# CAVEAT: the pollers run on the remote nodes and append to ${SHARED}; if that
# filesystem doesn't propagate appends promptly, "quiescent" here can mean
# "not yet visible". The fixed drain below runs first for exactly that reason
# -- it guarantees a real post-request sampling window regardless.
cxi_total_bytes() {
    { tail -n 4 "${SHARED}/logs/cxi_poll_p.csv" 2>/dev/null
      tail -n 4 "${SHARED}/logs/cxi_poll_d.csv" 2>/dev/null; } \
      | awk -F'[,=]' '{ s += $4 + $6 } END { printf "%.0f\n", s+0 }'
}
settle_cxi_poll() {
    local drain=${CXI_DRAIN_S:-2}                  # unconditional post-request window
    local quiet_bytes=${CXI_QUIET_BYTES:-1000000}  # per 250ms sample; ~4 MB/s of noise
    local quiet=0 tries=0 cur prev delta
    sleep "${drain}"
    prev=$(cxi_total_bytes)
    while [ ${quiet} -lt 3 ] && [ ${tries} -lt 40 ]; do   # cap ~10s past the drain
        sleep 0.25
        cur=$(cxi_total_bytes)
        delta=$(( ${cur:-0} - ${prev:-0} ))
        [ ${delta} -lt 0 ] && delta=$(( -delta ))
        if [ ${delta} -lt ${quiet_bytes} ]; then quiet=$((quiet + 1)); else quiet=0; fi
        prev=${cur}
        tries=$((tries + 1))
    done
    if [ ${quiet} -ge 3 ]; then
        echo "  CXI counters quiet (<${quiet_bytes} B/sample) after ${drain}s drain + $(awk "BEGIN{printf \"%.1f\", ${tries} * 0.25}")s"
    else
        echo "  CXI counters STILL MOVING at the settle cap (>${quiet_bytes} B/sample)."
        echo "  Deltas below are a lower bound. Either the transfer is genuinely still"
        echo "  running, or something else is saturating hsn0 -- check the time series"
        echo "  in cxi_poll_{p,d}.csv before trusting the totals."
    fi
}
summarize_cxi_poll() {
    local outfile=$1 label=$2
    echo "  -- ${label} --"
    if [ ! -s "${outfile}" ]; then
        echo "    (no samples captured -- poll loop may not have started; check permissions on the telemetry files)"
        return
    fi
    awk -F'[,=]' '
        { dev=$2
          if (!(dev in txfirst)) { txfirst[dev]=$4; rxfirst[dev]=$6 }
          txlast[dev]=$4; rxlast[dev]=$6; n[dev]++
        }
        END {
          for (d in txlast) printf "    %s: %d samples, tx_delta=%d bytes, rx_delta=%d bytes\n", d, n[d], txlast[d]-txfirst[d], rxlast[d]-rxfirst[d]
        }' "${outfile}" | sort
}

echo "=== Q2 (legacy/secondary): interface counters BEFORE disagg request ==="
snapshot_counters "before"

echo "=== Q2 (real signal): starting CXI octet-counter polling on both nodes ==="
start_cxi_poll "${NODE_P}" "${SHARED}/logs/cxi_poll_p.csv"
start_cxi_poll "${NODE_D}" "${SHARED}/logs/cxi_poll_d.csv"
sleep 0.2   # let both pollers get at least one sample before the request fires

echo "=== Q1b: disagg request, through the proxy ==="
snapshot_metrics ${P_IP} ${P_PORT} ${M}_p_b_before
snapshot_metrics ${D_IP} ${D_PORT} ${M}_d_b_before
send_and_time ${P_IP} ${PROXY_PORT} "disagg_via_proxy"

settle_cxi_poll
stop_cxi_poll

# Snapshot AFTER settling, not immediately after curl returns: the KV pull
# outlives the HTTP response, and so do the counters that record it.
snapshot_metrics ${P_IP} ${P_PORT} ${M}_p_b_after
snapshot_metrics ${D_IP} ${D_PORT} ${M}_d_b_after

echo "=== Q2 (legacy/secondary): interface counters AFTER disagg request ==="
snapshot_counters "after"

echo "=== Q2 (real signal): CXI octet deltas during the request window ==="
echo "  A non-zero tx/rx delta on ONE specific device, on both P and D, during"
echo "  this exact window is the actual evidence -- not the /proc/net/dev numbers"
echo "  above. Full time series in ${SHARED}/logs/cxi_poll_{p,d}.csv if you want"
echo "  to look at the shape rather than just the delta."
summarize_cxi_poll "${SHARED}/logs/cxi_poll_p.csv" "Prefill (${NODE_P})"
summarize_cxi_poll "${SHARED}/logs/cxi_poll_d.csv" "Decode (${NODE_D})"

# =============================================================================
# Q1 verdict -- read the metric diffs, not the clock.
#
# The old verdict here claimed disagg TTFT "should be noticeably LOWER than
# baseline". For a single request through 1P1D that is simply false, and
# treating it as the pass criterion would have failed a perfectly healthy run.
# Disagg does the SAME prefill work, then adds a KV transfer over the fabric
# and an extra proxy hop. One request in isolation is expected to be SLOWER.
# Disagg wins on aggregate throughput under load -- prefill and decode stop
# contending for the same GPU -- which is `vllm bench serve`'s job, not this
# script's. Timing is printed here only as a sanity check that neither path
# fell off a cliff.
#
# What actually decides Q1 is the pair of metric diffs.
# =============================================================================
echo ""
# NixlConnector's own telemetry, and the best single piece of evidence in the
# whole run -- better than anything this script computes. It reports transfer
# count, bytes, descriptor count and achieved throughput straight from the
# connector, so it distinguishes "a transfer happened" from "two servers each
# returned 200" with no inference at all. It is logged on the CONSUMER (D),
# since that is the side that issues the READ. Found by reading a passing run's
# log, not by design; surfaced here so it is never buried again.
#
# Cross-checks worth doing on the numbers it prints:
#   MB is MiB -- bytes / 12288 should be a whole multiple of the block size
#   descriptors should equal the model's layer count
#   the CXI tx delta below should exceed it by a few percent (wire framing)
echo "=== vLLM's own NIXL transfer telemetry ==="
if ! grep -h 'KV Transfer metrics' ${SHARED}/logs/p.log ${SHARED}/logs/d.log 2>/dev/null | tail -5 | sed 's/^/  /'; then
    echo "  NONE FOUND. NixlConnector logs this line only once a transfer actually"
    echo "  completes, so its absence means no KV moved -- regardless of what the"
    echo "  HTTP status codes and the metric diffs below suggest."
fi

echo ""
echo "=== Q1 verdict: read the metric diffs above ==="
diff_metrics ${M}_p_b_before ${M}_p_b_after "Q1b PREFILL node -- expected: a full prefill of the prompt"
diff_metrics ${M}_d_b_before ${M}_d_b_after "Q1b DECODE node  -- expected: request served, prefill work much smaller than Q1a"
echo ""
echo "  PASS looks like:"
echo "    * Q1a moved D's counters and left P's flat  (P is not in that path)."
echo "    * Q1b moved BOTH. P participating at all is the single most robust"
echo "      signal -- it is name-independent and it is the whole claim: the"
echo "      proxy really did route prefill to the other node."
echo "    * D's prefill-side work in Q1b is much smaller than in Q1a, because"
echo "      D received the KV instead of computing it."
echo "  FAIL looks like:"
echo "    * Q1b's D diff resembles Q1a's D diff and P stayed flat -- D quietly"
echo "      did its own prefill, i.e. the connector fell back and 200s mean"
echo "      nothing."
echo ""
echo "  Timing (sanity only -- disagg being slower here is EXPECTED):"
grep -h 'time_starttransfer' ${SHARED}/logs/timings.txt 2>/dev/null || true

# =============================================================================
# Q3 (transfer half) -- the startup half ran before the requests. This is the
# part that can only be checked afterwards: did LIBFABRIC actually carry a
# transfer, and did the shim stay in the loop?
# =============================================================================
echo ""
echo "=== Q3: transport evidence after the transfer ==="
for role in p d; do
    log=${SHARED}/logs/${role}.log
    echo "  -- ${role}.log --"
    echo "     LIBFABRIC/CXI mentions:  $(grep -ci 'libfabric\|cxi' "${log}" 2>/dev/null || echo 0)"
    echo "     shim announcements:      $(grep -c 'fi_getinfo_shim' "${log}" 2>/dev/null || echo 0)"
    # A raw `grep -ci ucx` is useless as a fallback detector: a healthy
    # LIBFABRIC run has exactly two benign UCX lines on each side, confirmed by
    # reading them on the first passing run --
    #   nixl_utils.py:32  vLLM sets UCX_RCACHE_MAX_UNRELEASED unconditionally,
    #                     whatever backend was requested. Env var, not usage.
    #   nixl_plugin_manager.cpp:621  "Discovered backend plugin: UCX" is the
    #                     plugin manager enumerating what is loadable.
    # Counting those as failures would fire on every good run, which just
    # teaches you to ignore the check. Subtract the two known-benign patterns
    # and report what is left -- that residue is what would need explaining.
    ucx_other=$(grep -i 'ucx' "${log}" 2>/dev/null \
        | grep -v 'UCX_RCACHE_MAX_UNRELEASED' \
        | grep -vc 'Discovered backend plugin: UCX' || true)
    echo "     unexplained UCX lines (want 0 on a LIBFABRIC run): ${ucx_other}"
    if [ "${ucx_other}" -ne 0 ] 2>/dev/null; then
        grep -i 'ucx' "${log}" 2>/dev/null \
            | grep -v 'UCX_RCACHE_MAX_UNRELEASED' \
            | grep -v 'Discovered backend plugin: UCX' | head -5 | sed 's/^/       ! /'
    fi
    grep -i 'nixl\|libfabric\|fi_getinfo_shim' "${log}" 2>/dev/null | tail -20 | sed 's/^/       /'
done
echo ""
echo "  A LIBFABRIC run with UCX mentions > 0 is the failure mode to watch for:"
echo "  a missing backend is a WARNING in NIXL's python API, not an error, so"
echo "  the agent comes up on UCX and everything downstream still returns 200s."

echo ""
echo "Full logs at: ${SHARED}/logs/"
