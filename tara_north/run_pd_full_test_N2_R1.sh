#!/bin/bash
set -uo pipefail

# =============================================================================
# 1P1D NIXL cross-node smoke test (PBS)
#
# Verifies three separate, chained questions:
#   Q1  Did a KV-cache transfer actually happen (vs D silently doing its own
#       local prefill)?              -> timing comparison, baseline vs disagg
#   Q2  Did it go over hsn0, not bond0 (mgmt net)?
#                                     -> /proc/net/dev byte counters, before/after
#   Q3  Did it use Slingshot properly (UCX/CXI), not a TCP fallback?
#                                     -> optional UCX_DEBUG=1 diagnostic pass
#
# Q2/Q3 only mean something once Q1 has passed. A TCP fallback over hsn0
# would still pass Q2 and still complete Q1's timing test with SOME
# improvement -- Q3 is the only thing that catches that specific failure mode.
#
# Usage:
#   Normal run:        bash pd_nixl_multinode_smoke.sh
#   Transport-debug:   UCX_DEBUG=1 bash pd_nixl_multinode_smoke.sh
#     (UCX_DEBUG adds UCX_LOG_LEVEL=debug and skips the timing/counter checks
#      -- it's noisy and meant as a one-off pass to answer Q3, not a
#      combined run. Do the normal run first.)
# =============================================================================

UCX_DEBUG=${UCX_DEBUG:-0}
# CONFIRMED via check_cxi_libfabric.sh probe (both nodes, 2026-08-19): fi_info -p cxi
# works, libfabric 2.3.1, and NIXL's plugin manager lists LIBFABRIC as loadable in
# this exact conda env. Defaulting to it -- NIXL's own default backend is UCX, and
# plain UCX has no native Slingshot/CXI transport (see handoff doc). Override to
# UCX for an explicit A/B comparison once LIBFABRIC is confirmed working end-to-end.
NIXL_BACKEND=${NIXL_BACKEND:-LIBFABRIC}
MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
STAMP=$(date +%Y%m%d_%H%M%S)
SHARED=/vast/draco/tara/projects/Tara_Deployment/software/testing/pd_smoke_${STAMP}
mkdir -p ${SHARED}/logs
P_PORT=8100; D_PORT=8200; PROXY_PORT=8000
# Override with PROXY_SCRIPT=/your/path if your checkout lives elsewhere or
# a newer vllm version moves this file.
PROXY_SCRIPT=${PROXY_SCRIPT:-/vast/draco/tara/projects/Tara_Deployment/software/testing/vllm_0.27.1_08_18_2026/vllm/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py}

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
echo "Prefill node: ${NODE_P}   Decode node: ${NODE_D}"

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

NO_PROXY_LIST="localhost,127.0.0.1,${P_IP},${D_IP},${NODE_P},${NODE_D}"

# --- Common env, sourced by both role scripts --------------------------------
UCX_LOG_LINE=""
if [ "${UCX_DEBUG}" = "1" ]; then
    UCX_LOG_LINE="export UCX_LOG_LEVEL=debug"
    echo "*** UCX_DEBUG=1: transport-negotiation diagnostic mode. Logs will be large. ***"
fi

# GPU_ID intentionally NOT set by default -- CUDA_VISIBLE_DEVICES is left
# unset, so all GPUs on the node stay visible to the process. Was previously
# hardcoded to 0; removed per explicit ask, for two reasons:
#   1. Hardcoding to GPU 0 is exactly the kind of thing that silently breaks
#      once this script grows into the real sweep (Section 6) and needs a
#      specific GPU per role on a multi-GPU node -- easy to forget it's there.
#   2. Logical inference, NOT confirmed against NIXL's source: the LIBFABRIC
#      backend does topology-aware GPU-to-NIC rail selection (NUMA-aware,
#      PCI-bus-ID-based, per NIXL's own release notes). Restricting GPU
#      visibility to one device could hand that logic a degenerate view of
#      the node's topology. Leaving all GPUs visible avoids the risk for free
#      -- vLLM still defaults to tensor-parallel-size=1 and lands on the
#      first visible device, so this smoke test's behavior is unchanged.
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

source /vast/draco/tara/projects/Tara_Deployment/software/miniforge3/bin/activate
conda activate /vast/draco/tara/projects/Tara_Deployment/software/envs/conda_envs/vllm_0.27.1_nixl_1.4.0_python_3.12.12

export HF_TOKEN=\$(cat ~/.hf_token)
export PYTHONNOUSERSITE=1
export HF_HOME=/vast/draco/tara/projects/Tara_Deployment/software/model-weights
export HF_DATASETS_CACHE=\${HF_HOME}
export HF_MODULES_CACHE=\${HF_HOME}
export RAY_TMPDIR=/tmp
export TMPDIR=/tmp
export VLLM_LOGGING_LEVEL=DEBUG
export UCX_TLS=cuda_copy,cuda_ipc,sm,tcp,self
export UCX_MODULE_DIR=\$(python3 -c "import site,glob; print(glob.glob(site.getsitepackages()[0]+'/nixl_cu13.libs/ucx')[0])")
${GPU_PIN_LINE}
${UCX_LOG_LINE}
EOF

# Appended with a QUOTED heredoc delimiter ('EOF') so `which nvc++` is
# resolved when common_env.sh actually RUNS on the compute node, not now on
# whichever node this launcher script happens to be running on -- login and
# compute node module environments aren't guaranteed to match on Cray systems.
cat >> ${SHARED}/common_env.sh <<'EOF'

# GH200 is aarch64; several CUDA/C++ extensions vLLM ships prebuilt for
# x86_64 don't have aarch64 wheels, so they JIT-compile via nvcc on first
# run. CONFIRMED from an actual crash log (not guessed): the JIT build path
# feeds nvcc's -ccbin from $CC, not $CXX. On this system CC=nvc (the NVHPC C
# compiler), and nvcc rejects any NVHPC-family compiler except nvc++:
#   "nvcc fatal: Unsupported NVHPC compiler found. nvc++ is the only
#    NVHPC compiler that is supported."
# Fix: point CC at nvc++ too (yes, both CC and CXX -- verified against an
# actual failing -ccbin invocation, not assumed from convention).
# If this clears THIS error but produces a *different* compile failure
# inside FlashInfer's sources, nvc++ itself may not be a reliable nvcc host
# compiler on this NVHPC version -- check `which gcc g++` and switch both
# CC and CXX to those instead, matching the GNU ABI the rest of this env's
# wheels were almost certainly built against.
NVCXX=$(which nvc++ 2>/dev/null || true)
if [ -n "${NVCXX}" ]; then
    export CC="${NVCXX}"
    export CXX="${NVCXX}"
    export CUDAHOSTCXX="${NVCXX}"
else
    echo "WARNING: nvc++ not found on PATH on $(hostname -s)." >&2
    echo "If a JIT compile step fails with an NVHPC host-compiler error," >&2
    echo "run 'module avail nvhpc' on this node and load the right module" >&2
    echo "before this script's launch_{p,d}.sh runs." >&2
fi

# CONFIRMED from an actual failing link command: torch's CUDA_HOME
# auto-detection derives from nvcc's location (dirname(dirname($(which
# nvcc)))), which lands in NVHPC's *compiler-only* tree
# (.../<ver>/compilers). NVHPC SDK layouts keep the real CUDA toolkit
# (libcudart.so etc.) in a SIBLING directory, .../<ver>/cuda/<cuda-ver>/ --
# so the auto-detected path is structurally wrong, not just misconfigured.
# This block finds the real one and points CUDA_HOME/LIBRARY_PATH/
# LD_LIBRARY_PATH at it explicitly, rather than trusting auto-detection.
if [ -n "${NVCXX}" ]; then
    NVHPC_COMPILERS_DIR=$(dirname "$(dirname "${NVCXX}")")   # .../<ver>/compilers
    NVHPC_VER_ROOT=$(dirname "${NVHPC_COMPILERS_DIR}")        # .../<ver>
    # Deliberately searches .../<ver>/cuda, NOT .../<ver>/REDIST/cuda --
    # REDIST is NVHPC's copy for bundling into containers/distributions,
    # not the one meant to be built against.
    CUDA_VER_DIR=$(find "${NVHPC_VER_ROOT}/cuda" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort -V | tail -1)
    # NVHPC/CUDA's multi-target layout nests the actual lib dir under
    # <cuda_ver_dir>/targets/<arch>-linux/lib/, not a flat lib64/ --
    # confirmed via `find` on this system (aarch64 -> sbsa-linux) rather
    # than assumed, since the exact "targets/<arch>" segment varies by
    # platform and shouldn't be hardcoded.
    REAL_CUDA_LIB_DIR=""
    if [ -n "${CUDA_VER_DIR}" ]; then
        REAL_CUDA_LIB_DIR=$(find "${CUDA_VER_DIR}" -name "libcudart.so" -printf '%h\n' 2>/dev/null | head -1)
    fi
    if [ -n "${REAL_CUDA_LIB_DIR}" ]; then
        export CUDA_HOME="${CUDA_VER_DIR}"
        export LIBRARY_PATH="${REAL_CUDA_LIB_DIR}:${LIBRARY_PATH:-}"
        export LD_LIBRARY_PATH="${REAL_CUDA_LIB_DIR}:${LD_LIBRARY_PATH:-}"
        # Same nesting problem as the lib dir, one level up: NVHPC's
        # multi-target layout keeps the COMPLETE header set as a sibling
        # of the lib dir (targets/<arch>/include), not under the
        # version-root's own include/. Confirmed via an actual
        # "curandStatePhilox4_32_10_t undefined" failure -- curand_kernel.h
        # wasn't reachable from the flat include dir. Fixed generally via
        # CPATH rather than one missing header at a time, since other CUDA
        # math-library headers (cublas, cusparse, etc.) likely have the
        # identical nesting problem waiting on whatever kernel needs them
        # next.
        TARGET_ROOT=$(dirname "${REAL_CUDA_LIB_DIR}")   # .../cuda/<ver>/targets/<arch>
        if [ -d "${TARGET_ROOT}/include" ]; then
            export CPATH="${TARGET_ROOT}/include:${CPATH:-}"
        fi
        # CONFIRMED (not guessed): curand_kernel.h -- and by strong
        # implication the rest of the CUDA math libraries (cublas,
        # cusparse, cusolver, cufft) -- live in a SEPARATE sibling tree,
        # math_libs/<ver>/, not inside cuda/<ver>/ at all. Reuses the
        # version string and arch name already confirmed for the cuda/
        # tree rather than a second blind search, since NVHPC names both
        # trees the same way.
        CUDA_VER=$(basename "${CUDA_VER_DIR}")            # e.g. 13.0
        ARCH_NAME=$(basename "${TARGET_ROOT}")             # e.g. sbsa-linux
        MATH_LIBS_TARGET="${NVHPC_VER_ROOT}/math_libs/${CUDA_VER}/targets/${ARCH_NAME}"
        if [ -d "${MATH_LIBS_TARGET}/include" ]; then
            export CPATH="${MATH_LIBS_TARGET}/include:${CPATH:-}"
        else
            echo "WARNING: expected math_libs include dir not found at ${MATH_LIBS_TARGET}/include" >&2
        fi
        if [ -d "${MATH_LIBS_TARGET}/lib" ]; then
            export LIBRARY_PATH="${MATH_LIBS_TARGET}/lib:${LIBRARY_PATH}"
            export LD_LIBRARY_PATH="${MATH_LIBS_TARGET}/lib:${LD_LIBRARY_PATH}"
        else
            echo "WARNING: expected math_libs lib dir not found at ${MATH_LIBS_TARGET}/lib" >&2
        fi
    else
        echo "WARNING: couldn't find libcudart.so under ${NVHPC_VER_ROOT}/cuda" >&2
        echo "-lcudart link failures are likely. Check manually:" >&2
        echo "  find ${NVHPC_VER_ROOT}/cuda -name 'libcudart.so*'" >&2
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

cat > ${SHARED}/launch_p.sh <<EOF
#!/bin/bash
source ${SHARED}/common_env.sh
export VLLM_NIXL_SIDE_CHANNEL_HOST=${P_IP}
export VLLM_NIXL_SIDE_CHANNEL_PORT=5600
vllm serve ${MODEL} --host 0.0.0.0 --port ${P_PORT} \\
    --gpu-memory-utilization 0.3 \\
    --no-enable-prefix-caching \\
    --kv-transfer-config '${KV_XFER_CONFIG_P}'
EOF

cat > ${SHARED}/launch_d.sh <<EOF
#!/bin/bash
source ${SHARED}/common_env.sh
export VLLM_NIXL_SIDE_CHANNEL_HOST=${D_IP}
export VLLM_NIXL_SIDE_CHANNEL_PORT=5601
vllm serve ${MODEL} --host 0.0.0.0 --port ${D_PORT} \\
    --gpu-memory-utilization 0.3 \\
    --no-enable-prefix-caching \\
    --kv-transfer-config '${KV_XFER_CONFIG_D}'
EOF
echo "NIXL backend for this run: ${NIXL_BACKEND}"

# --- Launch both -------------------------------------------------------------
ssh -n "${NODE_P}" "bash ${SHARED}/launch_p.sh" > ${SHARED}/logs/p.log 2>&1 &
P_SSH_PID=$!
ssh -n "${NODE_D}" "bash ${SHARED}/launch_d.sh" > ${SHARED}/logs/d.log 2>&1 &
D_SSH_PID=$!

# Stream both server logs live, prefixed by role so interleaved output stays
# readable. `sed -u` (unbuffered) matters here -- without it, output piped
# through sed gets block-buffered and shows up in silent bursts instead of
# line-by-line, which looks exactly like "nothing is happening" even when it is.
touch ${SHARED}/logs/p.log ${SHARED}/logs/d.log   # avoid a race if tail starts before ssh creates the file
tail -n +1 -f ${SHARED}/logs/p.log | sed -u 's/^/[P] /' &
TAIL_P_PID=$!
tail -n +1 -f ${SHARED}/logs/d.log | sed -u 's/^/[D] /' &
TAIL_D_PID=$!

cleanup() {
    echo "=== Cleaning up ==="
    # CXI poll loops are setsid'd specifically so they survive an SSH session
    # closing -- which means they ALSO survive this script dying early, same
    # orphan-process problem as EngineCore below. Stop marker is a no-op if
    # the poll never started (POLL_STOP_MARKER unset under `set -u` is guarded
    # with :-).
    touch "${POLL_STOP_MARKER:-}" 2>/dev/null
    # Signaling the local ssh client PIDs does NOT reach the remote vllm
    # serve processes -- no pty was allocated, so the remote shell has
    # nothing to forward a signal through. SIGTERM the actual remote
    # processes directly first, so the engine gets a real chance to
    # release NIXL registrations and GPU memory cleanly, before falling
    # back to SIGKILL.
    for n in "${NODE_P}" "${NODE_D}"; do
        ssh -n "$n" "pkill -TERM -f 'vllm serve'" 2>/dev/null
    done
    ssh -n "${NODE_P}" "pkill -TERM -f 'toy_proxy_server.py'" 2>/dev/null
    # :- guards matter here: cleanup can fire (via the EXIT trap) before the
    # proxy block ever runs -- e.g. if P or D fails its health check and the
    # script exits early -- in which case PROXY_SSH_PID/TAIL_PROXY_PID were
    # never assigned, and under `set -u` referencing them bare would abort
    # cleanup() partway through, skipping everything after that line.
    kill -TERM ${P_SSH_PID} ${D_SSH_PID} ${TAIL_P_PID} ${TAIL_D_PID} ${PROXY_SSH_PID:-} ${TAIL_PROXY_PID:-} 2>/dev/null
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
        ssh -n "$n" "pkill -KILL -f 'EngineCore'" 2>/dev/null
    done
    ssh -n "${NODE_P}" "pkill -KILL -f 'toy_proxy_server.py'" 2>/dev/null
    ssh -n "${NODE_P}" "fuser -k 5600/tcp" 2>/dev/null
    ssh -n "${NODE_D}" "fuser -k 5601/tcp" 2>/dev/null
    ssh -n "${NODE_P}" "fuser -k ${PROXY_PORT}/tcp" 2>/dev/null
    for n in "${NODE_P}" "${NODE_D}"; do
        if ssh -n "$n" "pgrep -f 'vllm serve'" > /dev/null 2>&1; then
            echo "Stray vllm process on $n -- force killing."
            ssh -n "$n" "pkill -KILL -f 'vllm serve'"
        fi
    done
}
trap cleanup EXIT INT TERM

wait_healthy() {
    local ip=$1 port=$2 name=$3 path=${4:-/health}
    for i in $(seq 1 60); do
        if curl -s -o /dev/null -w "%{http_code}" http://${ip}:${port}${path} 2>/dev/null | grep -q 200; then
            echo "${name} healthy after $((i*5))s"; return 0
        fi
        sleep 5
    done
    echo "${name} never became healthy -- check ${SHARED}/logs/"; return 1
}
wait_healthy ${P_IP} ${P_PORT} "Prefill (${NODE_P})" || exit 1
wait_healthy ${D_IP} ${D_PORT} "Decode (${NODE_D})"  || exit 1

if [ "${UCX_DEBUG}" = "1" ]; then
    echo "=== UCX_DEBUG mode: instances are up with UCX_LOG_LEVEL=debug. ==="
    echo "Send your normal disagg request now (or through the proxy once wired"
    echo "in below), then inspect ${SHARED}/logs/{p,d}.log for the negotiated"
    echo "transport at connection setup -- look for the device/transport name"
    echo "near the UCX endpoint-creation lines. A CXI-related name is the"
    echo "success case; a plain 'tcp' endpoint means it fell back."
    echo "Skipping the timing/counter checks below in this mode -- rerun"
    echo "without UCX_DEBUG for those."
    exit 0
fi

# =============================================================================
# Q1 -- baseline timing: query D directly, bypassing the proxy entirely.
# This is D doing its own full local prefill, same as a standalone instance.
# =============================================================================
PROMPT="The capital of France is"

send_and_time() {
    local ip=$1 port=$2 label=$3
    local out=${SHARED}/logs/resp_${label}.txt
    local t
    t=$(curl -s -o ${out} -w "%{time_starttransfer}" http://${ip}:${port}/v1/completions \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"${MODEL}\", \"prompt\": \"${PROMPT}\", \"max_tokens\": 10, \"stream\": true}")
    echo "${label}: time_starttransfer=${t}s  (raw response: ${out})"
}

echo "=== Q1a: baseline -- direct to decode instance, no proxy ==="
send_and_time ${D_IP} ${D_PORT} "baseline_decode_alone"

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
curl -s http://${P_IP}:${P_PORT}/v1/completions \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"${MODEL}\",
      \"prompt\": \"${PROMPT}\",
      \"max_tokens\": 1,
      \"stream\": false,
      \"kv_transfer_params\": {
        \"do_remote_decode\": true,
        \"do_remote_prefill\": false,
        \"remote_engine_id\": null,
        \"remote_block_ids\": null,
        \"remote_host\": null,
        \"remote_port\": null
      }
    }" | tee ${SHARED}/logs/resp_direct_p_hop.json | python3 -m json.tool 2>/dev/null \
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
send_and_time ${P_IP} ${PROXY_PORT} "disagg_via_proxy"

sleep 0.2   # capture at least one post-request sample before stopping
stop_cxi_poll

echo "=== Q2 (legacy/secondary): interface counters AFTER disagg request ==="
snapshot_counters "after"

echo "=== Q2 (real signal): CXI octet deltas during the request window ==="
echo "  A non-zero tx/rx delta on ONE specific device, on both P and D, during"
echo "  this exact window is the actual evidence -- not the /proc/net/dev numbers"
echo "  above. Full time series in ${SHARED}/logs/cxi_poll_{p,d}.csv if you want"
echo "  to look at the shape rather than just the delta."
summarize_cxi_poll "${SHARED}/logs/cxi_poll_p.csv" "Prefill (${NODE_P})"
summarize_cxi_poll "${SHARED}/logs/cxi_poll_d.csv" "Decode (${NODE_D})"

echo "=== Q1 verdict: compare baseline_decode_alone vs disagg_via_proxy above ==="
echo "disagg time_starttransfer should be noticeably LOWER than baseline if a"
echo "real remote KV pull happened. Roughly equal times = Q1 likely failed"
echo "even though both requests returned 200s."

echo "=== Log-level evidence (necessary but not sufficient on its own) ==="
grep -i nixl ${SHARED}/logs/p.log | tail -20
grep -i nixl ${SHARED}/logs/d.log | tail -20

echo "Full logs at: ${SHARED}/logs/"
