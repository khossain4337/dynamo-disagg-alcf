#!/bin/bash
set -uo pipefail

# Disagg arm (1P:1D), unattended: bring up two roles of DP2 x TP4 plus a proxy,
# bench one (ISL,OSL) across a concurrency ladder, tear down. Flag rationale and
# THE ONE RULE are in run_inkling_1p1d_N4.sh -- this file must issue the same
# `vllm serve`.
#
# Runs ON THE CLIENT NODE. Needs 4 engine nodes + 1 proxy node + this one.
#
# Exit  0 ladder ran to completion -- per-cell outcomes in bench_status.tsv
#       1 config | 10 dirty node (one DIRTY_NODE line per node) | 11 pool gate
#      30 a server died mid-ladder; the cells after it did not run

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Also in common_env.sh; set here too so they hold even if that file is wrong,
# and so the launch line needs no env prefix. mpiexec forwards them to the
# engines. Standing constraints -- HANDOFF.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1

ISL=${ISL:?set ISL}
OSL=${OSL:?set OSL}
CONCURRENCIES=${CONCURRENCIES:-16}
PROMPTS_PER_STREAM=${PROMPTS_PER_STREAM:-4}          # num_prompts = 4c, CLOSED.md
# 8192 of slack: --dataset-name random lands near ISL, not on it, and an exact
# budget rejects whatever rounds up (HTTP 400, which reads as a fast request).
MAX_MODEL_LEN=${MAX_MODEL_LEN:-$(( ISL + OSL + 8192 ))}

DP=${DP:-2}
TP=${TP:-4}
DP_LOCAL=${DP_LOCAL:-1}
# The node wiring below is exactly head + headless per role, and TP>4 would put
# the attention all-reduce on Slingshot. A different shape gets a different
# script.
if [ "${DP}" -ne 2 ] || [ "${TP}" -gt 4 ]; then
    echo "FATAL: this script is DP=2 x TP<=4 per role; got DP=${DP} TP=${TP}." >&2
    exit 1
fi

GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-auto}
EXPERT_PARALLEL=${EXPERT_PARALLEL:-1}
HMA_FLAG=${HMA_FLAG:---no-disable-hybrid-kv-cache-manager}
PREFIX_CACHING_FLAG=${PREFIX_CACHING_FLAG:---no-enable-prefix-caching}

# FROZEN. The colocated baseline's 16384 / 256 is max(P, D) per knob; change any
# of these and the published colocated sweep has to be re-run.
P_MAX_NUM_BATCHED_TOKENS=${P_MAX_NUM_BATCHED_TOKENS:-16384}
P_MAX_NUM_SEQS=${P_MAX_NUM_SEQS:-32}
D_MAX_NUM_BATCHED_TOKENS=${D_MAX_NUM_BATCHED_TOKENS:-2048}
D_MAX_NUM_SEQS=${D_MAX_NUM_SEQS:-256}

BLOCK_SIZE=${BLOCK_SIZE:-}
ENFORCE_EAGER=${ENFORCE_EAGER:-}
NUMA_BIND=${NUMA_BIND:-1}
# Space-separated and indexed by GPU index. The comma form dies at argparse and
# takes the whole PALS application with it.
NUMA_BIND_NODES="${NUMA_BIND_NODES:-0 1 2 3}"
# Shared default with the colocated arm (THE ONE RULE). Passed explicitly
# because omitting it is not 1 -- serve.py:121 substitutes data_parallel_size.
API_SERVER_COUNT=${API_SERVER_COUNT:-16}

NIXL_BACKEND=${NIXL_BACKEND:-LIBFABRIC}
KV_LEASE_DURATION=${KV_LEASE_DURATION:-30}
KV_EXTRA="\"backends\":[\"${NIXL_BACKEND}\"],\"kv_lease_duration\":${KV_LEASE_DURATION}"
KV_XFER_CONFIG_P="{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_producer\",\"kv_connector_extra_config\":{${KV_EXTRA}}}"
KV_XFER_CONFIG_D="{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_consumer\",\"kv_connector_extra_config\":{${KV_EXTRA}}}"

# base + data_parallel_index (base_scheduler.py:65), so at DP=2 each role spans
# two ports. 100 apart keeps the port saying which role it belongs to.
SIDE_PORT_P=${SIDE_PORT_P:-5600}
SIDE_PORT_D=${SIDE_PORT_D:-5700}

P_PORT=${P_PORT:-8100}
D_PORT=${D_PORT:-8200}
PROXY_PORT=${PROXY_PORT:-8000}
DP_RPC_PORT_P=${DP_RPC_PORT_P:-29550}
DP_RPC_PORT_D=${DP_RPC_PORT_D:-29560}
PROXY_SCRIPT=${PROXY_SCRIPT:-/vast/draco/tara/projects/Tara_Deployment/software/testing/vllm_0.27.1_08_18_2026/vllm/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py}

# 480 x 5s = 40 min, one 532 GB load. Per role, not for both: the two waits run
# in sequence and each gets its own budget.
HEALTH_TRIES=${HEALTH_TRIES:-480}
SAMPLE_INTERVAL_S=${SAMPLE_INTERVAL_S:-10}
GPU_DIRTY_MIB=${GPU_DIRTY_MIB:-4096}
# 0 for unattended: holding the nodes past the ladder blocks the next PBS job.
# Set 1 when you are at the terminal and want to retry a failed cell by hand
# without paying another startup.
KEEP_ALIVE=${KEEP_ALIVE:-0}
KEEP_ALIVE_POLL_S=${KEEP_ALIVE_POLL_S:-300}

RUNS_ROOT=${RUNS_ROOT:-/vast/draco/tara/projects/Tara_Deployment/software/testing/RUNS}
STAMP=$(date +%Y%m%d_%H%M%S)
SHARED=${SHARED:-${RUNS_ROOT}/inkling_1p1d_dp${DP}tp${TP}_isl${ISL}_osl${OSL}_${STAMP}}

# --- Model: resolved from the cache, because that also proves it is there -----
HF_CACHE_ROOT=${HF_CACHE_ROOT:-/vast/draco/tara/projects/Tara_Deployment/software/model-weights/hub}
MODEL=${MODEL:-}
if [ -z "${MODEL}" ]; then
    _small=()
    while IFS= read -r _c; do
        case "${_c}" in *[Ss]mall*) _small+=("${_c}") ;; esac
    done < <(find "${HF_CACHE_ROOT}" -maxdepth 1 -type d -name 'models--*Inkling*' 2>/dev/null | sort)
    if [ "${#_small[@]}" -ne 1 ]; then
        echo "FATAL: ${#_small[@]} Inkling-Small dirs under ${HF_CACHE_ROOT}; pass MODEL=<repo id>." >&2
        exit 1
    fi
    MODEL=$(basename "${_small[0]}"); MODEL=${MODEL#models--}; MODEL=${MODEL//--/\/}
fi

# --- Nodes. This node is the client and never serves. -------------------------
# Structural, not a convention: a client on a serving node cost -37.7%
# throughput and +14.4% engine work per token (TECHNICAL_REPORT_1.tex S7).
if [ -z "${PBS_NODEFILE:-}" ] || [ ! -r "${PBS_NODEFILE}" ]; then
    echo "FATAL: no readable PBS_NODEFILE." >&2
    exit 1
fi
mapfile -t ALL_NODES < <(sort -u "${PBS_NODEFILE}")
CLIENT_HOST=$(hostname -s); CLIENT_HOST="${CLIENT_HOST%%.*}"
ROLE_NODES=()
for n in "${ALL_NODES[@]}"; do
    [[ "${n%%.*}" == "${CLIENT_HOST}" ]] && continue
    ROLE_NODES+=("$n")
done
NEED_ENGINE_NODES=$(( DP * 2 ))
NEED_ROLE_NODES=$(( NEED_ENGINE_NODES + 1 ))
if [ "${#ROLE_NODES[@]}" -lt "${NEED_ROLE_NODES}" ]; then
    echo "FATAL: need ${NEED_ENGINE_NODES} engine + 1 proxy node plus this client;" >&2
    echo "  have ${#ROLE_NODES[@]}. Ask for $(( NEED_ROLE_NODES + 1 ))." >&2
    echo "  PROXY_ON_P_HEAD=1 drops to $(( NEED_ENGINE_NODES + 1 )) at the cost of a" >&2
    echo "  frontend-tax difference against the colocated arm. Record it." >&2
    exit 1
fi

NODE_P_HEAD="${ROLE_NODES[0]}"; NODE_P_HEAD_SHORT="${NODE_P_HEAD%%.*}"
NODE_P_TAIL="${ROLE_NODES[1]}"; NODE_P_TAIL_SHORT="${NODE_P_TAIL%%.*}"
NODE_D_HEAD="${ROLE_NODES[2]}"; NODE_D_HEAD_SHORT="${NODE_D_HEAD%%.*}"
NODE_D_TAIL="${ROLE_NODES[3]}"; NODE_D_TAIL_SHORT="${NODE_D_TAIL%%.*}"
PROXY_ON_P_HEAD=${PROXY_ON_P_HEAD:-0}
if [ "${PROXY_ON_P_HEAD}" = "1" ]; then
    NODE_PROXY="${NODE_P_HEAD}"
else
    NODE_PROXY="${ROLE_NODES[4]}"
fi
NODE_PROXY_SHORT="${NODE_PROXY%%.*}"
ENGINE_NODES=("${NODE_P_HEAD}" "${NODE_P_TAIL}" "${NODE_D_HEAD}" "${NODE_D_TAIL}")

# ssh wants the FQDN; launch_role.sh's hostname test must be short-vs-short.
# Getting that wrong is silent and total: every rank falls to the else branch,
# the first to exit takes the application down, and no role log is written.
get_hsn_ip() { ssh -n "$1" "ip -4 -o addr show hsn0" 2>/dev/null | awk '{print $4}' | cut -d/ -f1; }
P_HEAD_IP=$(get_hsn_ip "${NODE_P_HEAD}")
P_TAIL_IP=$(get_hsn_ip "${NODE_P_TAIL}")
D_HEAD_IP=$(get_hsn_ip "${NODE_D_HEAD}")
D_TAIL_IP=$(get_hsn_ip "${NODE_D_TAIL}")
PROXY_IP=$(get_hsn_ip "${NODE_PROXY}")
CLIENT_IP=$(ip -4 -o addr show hsn0 2>/dev/null | awk '{print $4}' | cut -d/ -f1)
for _pair in "P_HEAD:${P_HEAD_IP}" "P_TAIL:${P_TAIL_IP}" "D_HEAD:${D_HEAD_IP}" \
             "D_TAIL:${D_TAIL_IP}" "PROXY:${PROXY_IP}" "CLIENT:${CLIENT_IP}"; do
    if [ -z "${_pair#*:}" ]; then
        echo "FATAL: no hsn0 address for ${_pair%%:*}." >&2
        exit 1
    fi
done
echo "client ${CLIENT_HOST} ${CLIENT_IP}"
echo "P ${NODE_P_HEAD_SHORT} ${P_HEAD_IP} / ${NODE_P_TAIL_SHORT} ${P_TAIL_IP}"
echo "D ${NODE_D_HEAD_SHORT} ${D_HEAD_IP} / ${NODE_D_TAIL_SHORT} ${D_TAIL_IP}"
echo "proxy ${NODE_PROXY_SHORT} ${PROXY_IP}$([ "${PROXY_ON_P_HEAD}" = "1" ] && echo '  (ON P HEAD -- degraded, record it)')"

mkdir -p "${SHARED}/logs" || { echo "FATAL: cannot create ${SHARED}" >&2; exit 1; }

for f in fi_getinfo_shim.so env_for_libfabric_topology_error.sh gpu_cleanup.sh \
         emit_common_env.sh emit_conn_sampler.sh workload_profile.sh bench_arm.sh; do
    [ -f "${SCRIPT_DIR}/${f}" ] || { echo "FATAL: missing ${SCRIPT_DIR}/${f}" >&2; exit 1; }
done
SHIM="${SCRIPT_DIR}/fi_getinfo_shim.so"
if [ "$(strings "${SHIM}" 2>/dev/null | grep -c 'PATCH [234]' || true)" -lt 1 ]; then
    echo "WARNING: ${SHIM} looks stale against its .c -- rebuild before trusting a cross-arm diff." >&2
fi
for n in "${ENGINE_NODES[@]}" "${NODE_PROXY}"; do
    ssh -n "$n" "test -d ${SHARED} && test -f ${SHIM}" || {
        echo "FATAL: ${SHARED} or the shim is not visible on ${n%%.*}." >&2
        exit 1
    }
done

# --- Preflight. gpu_cleanup.sh owns the pattern list; sourced, never copied. ---
# shellcheck source=./gpu_cleanup.sh
source "${SCRIPT_DIR}/gpu_cleanup.sh"
PAT_VLLM="${VLLM_PROC_PATTERNS[0]}"
PAT_PROXY='[t]oy_proxy_server.py'
_dirty=()
for n in "${ENGINE_NODES[@]}"; do
    ssh -n "$n" "GPU_DIRTY_MIB=${GPU_DIRTY_MIB} bash ${SCRIPT_DIR}/gpu_cleanup.sh report" \
        2>&1 | sed "s/^/  [${n%%.*}] /"
    # PIPESTATUS, not $? -- $? is sed's and the gate would pass unconditionally.
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then _dirty+=("${n%%.*}"); fi
done
ssh -n "${NODE_P_HEAD}" "fuser ${P_PORT}/tcp"     >/dev/null 2>&1 && _dirty+=("${NODE_P_HEAD_SHORT}")
ssh -n "${NODE_D_HEAD}" "fuser ${D_PORT}/tcp"     >/dev/null 2>&1 && _dirty+=("${NODE_D_HEAD_SHORT}")
ssh -n "${NODE_PROXY}"  "fuser ${PROXY_PORT}/tcp" >/dev/null 2>&1 && _dirty+=("${NODE_PROXY_SHORT}")
if [ "${#_dirty[@]}" -gt 0 ]; then
    printf 'DIRTY_NODE:%s\n' "${_dirty[@]}" | sort -u >&2
    echo "Clear with: ssh <n> bash ${SCRIPT_DIR}/gpu_cleanup.sh kill ${P_PORT} ${D_PORT} ${DP_RPC_PORT_P} ${DP_RPC_PORT_D}" >&2
    exit 10
fi

# --- Shared env and sampler, from the shared emitters -------------------------
# The client node is in no_proxy too: the bench runs there, and without it curl
# to an hsn0 address returns a Squid page that reads as a dead server.
NO_PROXY_LIST="localhost,127.0.0.1,${P_HEAD_IP},${P_TAIL_IP},${D_HEAD_IP},${D_TAIL_IP},${PROXY_IP},${CLIENT_IP},${NODE_P_HEAD},${NODE_P_TAIL},${NODE_D_HEAD},${NODE_D_TAIL},${NODE_PROXY},${NODE_P_HEAD_SHORT},${NODE_P_TAIL_SHORT},${NODE_D_HEAD_SHORT},${NODE_D_TAIL_SHORT},${NODE_PROXY_SHORT},${CLIENT_HOST}"
ENV_SCRIPT="${SCRIPT_DIR}/env_for_libfabric_topology_error.sh"
UCX_LINES=""
GPU_PIN_LINE=""
# shellcheck source=./emit_common_env.sh
source "${SCRIPT_DIR}/emit_common_env.sh"
emit_common_env "${SHARED}/common_env.sh" || exit 1
# shellcheck source=./emit_conn_sampler.sh
source "${SCRIPT_DIR}/emit_conn_sampler.sh"
emit_conn_sampler "${SHARED}/sample_conns.sh"

# --- Role script: one file, four nodes, selecting by hostname -----------------
# One mpiexec, because PALS allocates a VNI per application and two launches get
# two VNIs that cannot reach each other. The three NIXL variables and
# --kv-transfer-config, plus the batching split, are the COMPLETE difference
# from the colocated arm's launch_role.sh.
cat > "${SHARED}/launch_role.sh" <<EOF
#!/bin/bash
MY_HOST="\$(hostname -s)"
MY_HOST="\${MY_HOST%%.*}"
if [ "\${MY_HOST}" = "${NODE_P_HEAD_SHORT}" ]; then
    LOG=${SHARED}/logs/p-head.log
    WAIT_FOR_D=1
    DP_ARGS=(--data-parallel-size ${DP}
             --data-parallel-size-local ${DP_LOCAL}
             --data-parallel-address ${P_HEAD_IP}
             --data-parallel-rpc-port ${DP_RPC_PORT_P}
             --api-server-count ${API_SERVER_COUNT})
    SERVE_ARGS=(--host 0.0.0.0 --port ${P_PORT})
    SIDE_HOST=${P_HEAD_IP}; SIDE_PORT=${SIDE_PORT_P}
    KV_CFG='${KV_XFER_CONFIG_P}'
    MAX_BATCHED_TOKENS=${P_MAX_NUM_BATCHED_TOKENS}; MAX_SEQS=${P_MAX_NUM_SEQS}
elif [ "\${MY_HOST}" = "${NODE_P_TAIL_SHORT}" ]; then
    LOG=${SHARED}/logs/p-headless.log
    WAIT_FOR_D=1
    # --api-server-count is a hard error with --headless (serve.py:66-71).
    DP_ARGS=(--headless
             --data-parallel-size ${DP}
             --data-parallel-size-local ${DP_LOCAL}
             --data-parallel-start-rank 1
             --data-parallel-address ${P_HEAD_IP}
             --data-parallel-rpc-port ${DP_RPC_PORT_P})
    SERVE_ARGS=()
    # Its OWN address: this engine is its own NIXL agent on its own node.
    # Advertising the head's address here produces a transfer that hangs.
    SIDE_HOST=${P_TAIL_IP}; SIDE_PORT=${SIDE_PORT_P}
    KV_CFG='${KV_XFER_CONFIG_P}'
    MAX_BATCHED_TOKENS=${P_MAX_NUM_BATCHED_TOKENS}; MAX_SEQS=${P_MAX_NUM_SEQS}
elif [ "\${MY_HOST}" = "${NODE_D_HEAD_SHORT}" ]; then
    LOG=${SHARED}/logs/d-head.log
    WAIT_FOR_D=0
    DP_ARGS=(--data-parallel-size ${DP}
             --data-parallel-size-local ${DP_LOCAL}
             --data-parallel-address ${D_HEAD_IP}
             --data-parallel-rpc-port ${DP_RPC_PORT_D}
             --api-server-count ${API_SERVER_COUNT})
    SERVE_ARGS=(--host 0.0.0.0 --port ${D_PORT})
    SIDE_HOST=${D_HEAD_IP}; SIDE_PORT=${SIDE_PORT_D}
    KV_CFG='${KV_XFER_CONFIG_D}'
    MAX_BATCHED_TOKENS=${D_MAX_NUM_BATCHED_TOKENS}; MAX_SEQS=${D_MAX_NUM_SEQS}
elif [ "\${MY_HOST}" = "${NODE_D_TAIL_SHORT}" ]; then
    LOG=${SHARED}/logs/d-headless.log
    WAIT_FOR_D=0
    DP_ARGS=(--headless
             --data-parallel-size ${DP}
             --data-parallel-size-local ${DP_LOCAL}
             --data-parallel-start-rank 1
             --data-parallel-address ${D_HEAD_IP}
             --data-parallel-rpc-port ${DP_RPC_PORT_D})
    SERVE_ARGS=()
    SIDE_HOST=${D_TAIL_IP}; SIDE_PORT=${SIDE_PORT_D}
    KV_CFG='${KV_XFER_CONFIG_D}'
    MAX_BATCHED_TOKENS=${D_MAX_NUM_BATCHED_TOKENS}; MAX_SEQS=${D_MAX_NUM_SEQS}
else
    echo "Rank landed on unexpected host '\${MY_HOST}'." >&2
    exit 1
fi

# Redirect BEFORE sourcing: a common_env.sh that fails to generate correctly
# reports it here rather than into mpiexec.log, which is where today's
# ENV_SCRIPT bug hid.
exec > "\${LOG}" 2>&1
source ${SHARED}/common_env.sh
echo "=== host=\$(hostname -s) SLINGSHOT_VNIS=\${SLINGSHOT_VNIS:-<UNSET>} ==="
if [ -z "\${SLINGSHOT_VNIS:-}" ]; then
    echo "SLINGSHOT_VNIS is EMPTY -- not launched under PALS. Anything reaching"
    echo "the cxi provider fails fi_domain() with -FI_ENOSYS: no KV transfer."
fi

# P waits for D to finish loading. A fixed sleep only OFFSETS two 532 GB reads
# off /vast; this serialises them. Still one mpiexec -- splitting the launch
# would give the roles different VNIs and they could never connect. D's head
# cannot answer /health until both its DP ranks have registered, so this one
# check covers the whole role.
if [ "\${WAIT_FOR_D}" = "1" ]; then
    echo "=== waiting for D head http://${D_HEAD_IP}:${D_PORT}/health ==="
    for i in \$(seq 1 ${HEALTH_TRIES}); do
        if curl -s -o /dev/null -w '%{http_code}' \\
            "http://${D_HEAD_IP}:${D_PORT}/health" 2>/dev/null | grep -q 200; then
            echo "=== D up after \$(( i * 5 ))s; starting \$(date -Is) ==="
            break
        fi
        sleep 5
    done
fi

export LD_PRELOAD=${SHIM}
export VLLM_NIXL_SIDE_CHANNEL_HOST=\${SIDE_HOST}
export VLLM_NIXL_SIDE_CHANNEL_PORT=\${SIDE_PORT}
export NIXL_LOG_LEVEL=\${NIXL_LOG_LEVEL:-INFO}

EXTRA=()
[ -n "${BLOCK_SIZE}" ]         && EXTRA+=(--block-size ${BLOCK_SIZE})
[ "${EXPERT_PARALLEL}" = "1" ] && EXTRA+=(--enable-expert-parallel)
[ -n "${ENFORCE_EAGER}" ]      && EXTRA+=(--enforce-eager)
[ "${NUMA_BIND}" = "1" ]       && EXTRA+=(--numa-bind --numa-bind-nodes ${NUMA_BIND_NODES})

exec vllm serve ${MODEL} \\
    \${SERVE_ARGS[@]+"\${SERVE_ARGS[@]}"} \\
    \${DP_ARGS[@]+"\${DP_ARGS[@]}"} \\
    --distributed-executor-backend mp \\
    --tensor-parallel-size ${TP} \\
    --max-model-len ${MAX_MODEL_LEN} \\
    --gpu-memory-utilization ${GPU_MEM_UTIL} \\
    --dtype bfloat16 \\
    --kv-cache-dtype ${KV_CACHE_DTYPE} \\
    ${HMA_FLAG} \\
    ${PREFIX_CACHING_FLAG} \\
    --max-num-batched-tokens \${MAX_BATCHED_TOKENS} \\
    --max-num-seqs \${MAX_SEQS} \\
    \${EXTRA[@]+"\${EXTRA[@]}"} \\
    --kv-transfer-config "\${KV_CFG}"
EOF
chmod +x "${SHARED}/launch_role.sh"

# toy_proxy_server.py health endpoint is /healthcheck, not /health. --host is
# explicit because its default is 127.0.0.1. No LD_PRELOAD: the shim has no
# business interposing fi_getinfo for a plain HTTP process.
cat > "${SHARED}/launch_proxy.sh" <<EOF
#!/bin/bash
source ${SHARED}/common_env.sh
export OPENAI_API_KEY=inkling-1p1d-dummy-key
python3 ${PROXY_SCRIPT} \\
    --host 0.0.0.0 --port ${PROXY_PORT} \\
    --prefiller-host ${P_HEAD_IP} --prefiller-port ${P_PORT} \\
    --decoder-host ${D_HEAD_IP} --decoder-port ${D_PORT}
EOF
chmod +x "${SHARED}/launch_proxy.sh"

{
    echo "stamp            ${STAMP}"
    echo "arm              disagg 1P:1D   isl ${ISL}  osl ${OSL}  concurrencies ${CONCURRENCIES}"
    echo "model            ${MODEL}"
    echo "client           ${CLIENT_HOST} ${CLIENT_IP}"
    echo "P head           ${NODE_P_HEAD_SHORT} ${P_HEAD_IP} port ${P_PORT}  api ${API_SERVER_COUNT}  side ${SIDE_PORT_P}"
    echo "P headless       ${NODE_P_TAIL_SHORT} ${P_TAIL_IP}  side ${SIDE_PORT_P}+1"
    echo "D head           ${NODE_D_HEAD_SHORT} ${D_HEAD_IP} port ${D_PORT}  api ${API_SERVER_COUNT}  side ${SIDE_PORT_D}"
    echo "D headless       ${NODE_D_TAIL_SHORT} ${D_TAIL_IP}  side ${SIDE_PORT_D}+1"
    echo "proxy            ${NODE_PROXY_SHORT} ${PROXY_IP} port ${PROXY_PORT}  on-p-head ${PROXY_ON_P_HEAD}"
    echo "parallelism      DP ${DP} x TP ${TP} per role  expert-parallel ${EXPERT_PARALLEL}  GPUs $(( DP * TP * 2 ))"
    echo "nixl             backend ${NIXL_BACKEND}  kv_lease_duration ${KV_LEASE_DURATION}"
    echo "startup          serialised: P waits on D /health (budget ${HEALTH_TRIES} x 5s per role)"
    echo "max-model-len    ${MAX_MODEL_LEN}"
    echo "batching         P ${P_MAX_NUM_BATCHED_TOKENS}/${P_MAX_NUM_SEQS}  D ${D_MAX_NUM_BATCHED_TOKENS}/${D_MAX_NUM_SEQS}"
    echo "numa-bind        ${NUMA_BIND} nodes ${NUMA_BIND_NODES}"
    echo "shim             sha256 $(sha256sum "${SHIM}" 2>/dev/null | awk '{print $1}')"
    echo "git              $(git -C "${SCRIPT_DIR}" rev-parse --short HEAD 2>/dev/null || echo '(none)')"
} > "${SHARED}/run_config.txt"
cat "${SHARED}/run_config.txt"

SAMPLE_STOP_MARKER="${SHARED}/sample_stop_${STAMP}"
cleanup() {
    touch "${SAMPLE_STOP_MARKER}" 2>/dev/null
    ssh -n "${NODE_PROXY}" "pkill -TERM -f \"${PAT_PROXY}\"" 2>/dev/null
    for n in "${ENGINE_NODES[@]}"; do
        ssh -n "$n" "pkill -TERM -f \"${PAT_VLLM}\"" 2>/dev/null
    done
    kill -TERM ${MPIEXEC_PID:-} ${TAIL_PH_PID:-} ${TAIL_PT_PID:-} ${TAIL_DH_PID:-} \
               ${TAIL_DT_PID:-} ${TAIL_M_PID:-} ${PROXY_SSH_PID:-} ${TAIL_PROXY_PID:-} 2>/dev/null
    sleep 5
    ssh -n "${NODE_PROXY}" "pkill -KILL -f \"${PAT_PROXY}\"; fuser -k ${PROXY_PORT}/tcp" 2>/dev/null
    # Same file preflight screened with. An orphaned EngineCore matches no
    # 'vllm serve' pattern, holds a CUDA context and its share of the KV pool,
    # and the symptom lands on the NEXT run. Read the VERDICT lines.
    for n in "${ENGINE_NODES[@]}"; do
        ssh -n "$n" "bash ${SCRIPT_DIR}/gpu_cleanup.sh kill ${P_PORT} ${D_PORT} ${DP_RPC_PORT_P} ${DP_RPC_PORT_D}" \
            2>&1 | sed "s/^/  [${n%%.*}] /"
    done
    echo "RUN_DIR:${SHARED}"
}
# EXIT is the single teardown path. A handler on INT/TERM that ran cleanup would
# RETURN, resuming the health loop or the ladder against killed servers, and
# then fire cleanup again at EXIT. These only exit; EXIT does the work.
trap cleanup EXIT
trap 'exit 130' INT TERM

# --cpu-bind none: PALS otherwise confines all four TP workers to rank 0's core
# slice, which looks like a fabric problem rather than an error.
MPI_CPU_BIND=${MPI_CPU_BIND-none}
CPU_BIND_ARGS=()
[ -n "${MPI_CPU_BIND}" ] && CPU_BIND_ARGS=(--cpu-bind "${MPI_CPU_BIND}")

# Pre-create all six: `cmd > file &` creates the file in the child, so tail -f
# can lose the race and silently follow nothing. mpiexec.log is the only stream
# that carries PALS-level failures.
touch "${SHARED}/logs/p-head.log" "${SHARED}/logs/p-headless.log" \
      "${SHARED}/logs/d-head.log" "${SHARED}/logs/d-headless.log" \
      "${SHARED}/logs/mpiexec.log" "${SHARED}/logs/proxy.log"
# --hosts, not a bare -n: the allocation also holds the client and proxy nodes,
# and -ppn 1 over all of them would start an engine on both.
mpiexec -n "${NEED_ENGINE_NODES}" -ppn 1 \
    --hosts "${NODE_P_HEAD},${NODE_P_TAIL},${NODE_D_HEAD},${NODE_D_TAIL}" \
    ${CPU_BIND_ARGS[@]+"${CPU_BIND_ARGS[@]}"} \
    bash "${SHARED}/launch_role.sh" > "${SHARED}/logs/mpiexec.log" 2>&1 &
MPIEXEC_PID=$!

# $! on a pipeline names its LAST member, so `| sed` would capture sed and the
# tails outlive the kill, dying on EPIPE. Process substitution keeps $! as tail.
tail -n +1 -f "${SHARED}/logs/p-head.log"     > >(sed -u 's/^/[P-HEAD] /') & TAIL_PH_PID=$!; disown ${TAIL_PH_PID}
tail -n +1 -f "${SHARED}/logs/p-headless.log" > >(sed -u 's/^/[P-TAIL] /') & TAIL_PT_PID=$!; disown ${TAIL_PT_PID}
tail -n +1 -f "${SHARED}/logs/d-head.log"     > >(sed -u 's/^/[D-HEAD] /') & TAIL_DH_PID=$!; disown ${TAIL_DH_PID}
tail -n +1 -f "${SHARED}/logs/d-headless.log" > >(sed -u 's/^/[D-TAIL] /') & TAIL_DT_PID=$!; disown ${TAIL_DT_PID}
tail -n +1 -f "${SHARED}/logs/mpiexec.log"    > >(sed -u 's/^/[MPI]    /') & TAIL_M_PID=$!; disown ${TAIL_M_PID}

# This one string only. 'init_device' appears 12-16 times in a HEALTHY log, so
# matching it would rotate a good node on any slow startup.
dirty_signature() {
    grep -lF 'No available memory for the cache blocks' "${SHARED}"/logs/[pd]-head*.log 2>/dev/null
}

wait_healthy() {
    local ip=$1 port=$2 name=$3 path=${4:-/health} watch=${5:-d-head}
    local t0; t0=$(date +%s)
    for i in $(seq 1 "${HEALTH_TRIES}"); do
        if curl -s -o /dev/null -w '%{http_code}' "http://${ip}:${port}${path}" 2>/dev/null | grep -q 200; then
            echo "${name} healthy after $(( $(date +%s) - t0 ))s"; return 0
        fi
        # mpiexec is the liveness source of truth: a dead startup and a slow one
        # are identical from outside, and PALS failures never reach a role log.
        if ! kill -0 "${MPIEXEC_PID}" 2>/dev/null; then
            echo "mpiexec exited while waiting for ${name}." >&2
            # Dump here rather than naming the files: cleanup kills the tail
            # pipelines within a second, and an argparse death writes its reason
            # and exits faster than the pipeline flushes.
            for _l in mpiexec d-head d-headless p-head p-headless; do
                echo "--- tail logs/${_l}.log ---" >&2
                tail -n 40 "${SHARED}/logs/${_l}.log" 2>/dev/null | sed 's/^/    /' >&2
            done
            return 1
        fi
        [ $(( i % 12 )) -eq 0 ] && printf '  [%4ds] %s\n' "$(( $(date +%s) - t0 ))" \
            "$(tail -n 1 "${SHARED}/logs/${watch}.log" 2>/dev/null | cut -c1-100)"
        sleep 5
    done
    echo "${name} never became healthy -- check ${SHARED}/logs/" >&2
    return 1
}

# D first, because D loads first and P is blocked on it. Only the heads answer
# HTTP; each headless node's liveness is implied, since a head cannot become
# healthy until every DP rank in its role has registered.
_boot_fail=0
wait_healthy "${D_HEAD_IP}" "${D_PORT}" "D head (${NODE_D_HEAD_SHORT})" /health d-head || _boot_fail=1
[ "${_boot_fail}" -eq 0 ] && { wait_healthy "${P_HEAD_IP}" "${P_PORT}" "P head (${NODE_P_HEAD_SHORT})" /health p-head || _boot_fail=1; }
kill -TERM "${TAIL_PH_PID}" "${TAIL_PT_PID}" "${TAIL_DH_PID}" "${TAIL_DT_PID}" "${TAIL_M_PID}" 2>/dev/null
if [ "${_boot_fail}" -ne 0 ]; then
    _sig=$(dirty_signature)
    if [ -n "${_sig}" ]; then
        # All four, not just the heads: 'p-headless.log' does not match the
        # 'p-head.log' pattern, and exiting 10 without naming a node leaves the
        # wrapper nothing to rotate.
        case "${_sig}" in *p-head.log*)     echo "DIRTY_NODE:${NODE_P_HEAD_SHORT}" >&2 ;; esac
        case "${_sig}" in *p-headless.log*) echo "DIRTY_NODE:${NODE_P_TAIL_SHORT}" >&2 ;; esac
        case "${_sig}" in *d-head.log*)     echo "DIRTY_NODE:${NODE_D_HEAD_SHORT}" >&2 ;; esac
        case "${_sig}" in *d-headless.log*) echo "DIRTY_NODE:${NODE_D_TAIL_SHORT}" >&2 ;; esac
        exit 10
    fi
    exit 1
fi

# --- KV pool. Per role, never across roles: P and D have different
# --max-num-batched-tokens, so different activation profiles and different
# pools. The colocated arm's number is meaningless on either.
read_pool() {
    grep -h -o 'GPU KV cache size: [0-9,]* tokens' "$@" 2>/dev/null \
        | grep -o '[0-9,]*' | tr -d ',' | grep -v '^$' | sort -n | head -1
}
_pool_p=$(read_pool "${SHARED}/logs/p-head.log" "${SHARED}/logs/p-headless.log")
_pool_d=$(read_pool "${SHARED}/logs/d-head.log" "${SHARED}/logs/d-headless.log")
echo "${_pool_p:-unknown}" > "${SHARED}/kv_pool_tokens_p.txt"
echo "${_pool_d:-unknown}" > "${SHARED}/kv_pool_tokens_d.txt"
_pool_fail=0
gate_pool() {
    local role=$1 pool=$2 expect=$3
    if [ -z "${pool}" ]; then
        echo "WARNING: could not read the ${role} pool from either rank's log." >&2
        return
    fi
    if [ -z "${expect}" ]; then
        echo "pool ${role} ${pool} -- no EXPECT_POOL_${role} set."
        return
    fi
    local p=$(( (pool - expect) * 100 / expect ))
    if [ "${p}" -lt -2 ] || [ "${p}" -gt 2 ]; then
        # The pool lottery: one rank allocates more during init than its peers.
        # Relaunch, do not tune -- a run benched against a short pool compares
        # to nothing.
        echo "POOL GATE FAILED ${role}: expected ${expect}, got ${pool} (${p}%)." >&2
        _pool_fail=1
    else
        echo "pool ${role} ${pool} (gate passed, ${p}%)"
    fi
}
gate_pool P "${_pool_p}" "${EXPECT_POOL_P:-}"
gate_pool D "${_pool_d}" "${EXPECT_POOL_D:-}"
[ "${_pool_fail}" -ne 0 ] && exit 11

# A disagg arm whose connector failed to attach is two independent servers
# behind a proxy, and it produces plausible-looking numbers.
for _r in p-head d-head; do
    grep -qiE 'NixlConnector|kv_transfer_config|KVConnector' "${SHARED}/logs/${_r}.log" \
        || echo "WARNING: ${_r}.log never mentions a KV connector. Do not bench this." >&2
done

echo ""
echo "=== KV cache groups ==="
grep -hiE 'kv_cache_group|KVCacheGroupSpec|SlidingWindowSpec|FullAttentionSpec|num_kv_cache_groups' \
    "${SHARED}/logs/d-head.log" 2>/dev/null | head -12 | sed 's/^/  /'

# --- Q3 and Q4. These lines print ONCE at agent construction: not in /metrics,
# not on the wire, not recoverable from a running server. A launch that skips
# them has to be relaunched to get them. WARN, never exit -- a relaunch costs
# the allocation the checks exist to protect.
_q_warn=0
echo ""
echo "=== Q3: is the transport LIBFABRIC/cxi, or a fallback? ==="
# fabric_attr->name is libfabric reporting the provider it actually OPENED.
# 'Created 4 rails using provider=efa' is NIXL's belief after the shim's PATCH 2
# relabels cxi as efa, and is not a contradiction -- trust fabric_attr.
for _r in p-head p-headless d-head d-headless; do
    _log="${SHARED}/logs/${_r}.log"
    _cxi=$(grep -c 'fabric_attr->name cxi' "${_log}" 2>/dev/null || true); _cxi=${_cxi:-0}
    _lfb=$(grep -c 'Initializing Libfabric Backend' "${_log}" 2>/dev/null || true); _lfb=${_lfb:-0}
    _vni=$(grep -hoE 'SLINGSHOT_VNIS[=: ]+[0-9]+' "${_log}" 2>/dev/null | head -1 || true)
    printf '    %-12s cxi=%-4s libfabric=%-4s %s\n' "${_r}" "${_cxi}" "${_lfb}" "${_vni:-VNI:unread}"
    if [ "${_cxi}" -eq 0 ] || [ "${_lfb}" -lt 4 ]; then
        echo "    ^^ WARNING: ${_r} shows no cxi provider or fewer than 4 backends." >&2
        _q_warn=1
    fi
done
echo "  VNIs must be IDENTICAL on all four nodes -- one mpiexec, one PALS VNI."

echo ""
echo "=== Q4: did every TP worker build an agent, and did the rails fan out? ==="
# 4 rail managers x 4 rails per node, and one 'Registered memory on 1 rails' per
# device 0-3 -- each worker on its OWN rail. The Nemotron launcher's "four lines
# saying 1 rail" comment is miscalibrated; this is the correct shape.
for _r in p-head p-headless d-head d-headless; do
    _log="${SHARED}/logs/${_r}.log"
    _rails=$(grep -c 'Rail Manager created with 4 rails' "${_log}" 2>/dev/null || true); _rails=${_rails:-0}
    _reg=$(grep -c 'Registered memory on 1 rails, mem_type=1' "${_log}" 2>/dev/null || true); _reg=${_reg:-0}
    _devs=$(grep -hoE 'mem_type=1 device=[0-9]+' "${_log}" 2>/dev/null | grep -oE '[0-9]+$' | sort -u | paste -sd, - || true)
    printf '    %-12s rail_managers=%-4s vram_registrations=%-4s devices=%s\n' \
        "${_r}" "${_rails}" "${_reg}" "${_devs:-none}"
    if [ "${_rails}" -lt 4 ] || [ "${_reg}" -lt 4 ]; then
        echo "    ^^ WARNING: ${_r} did not fan out to 4 rails on 4 devices." >&2
        _q_warn=1
    fi
done
if [ "${_q_warn}" -ne 0 ]; then
    echo "  Q3/Q4 RAISED A WARNING -- read ${SHARED}/logs/ before trusting this run." >&2
else
    echo "  Q3 and Q4 PASS on all four nodes."
fi

# --- Proxy. After both roles are healthy, so a proxy failure means the proxy.
ssh -n "${NODE_PROXY}" "bash ${SHARED}/launch_proxy.sh" > "${SHARED}/logs/proxy.log" 2>&1 &
PROXY_SSH_PID=$!
tail -n +1 -f "${SHARED}/logs/proxy.log" > >(sed -u 's/^/[PROXY]  /') & TAIL_PROXY_PID=$!; disown ${TAIL_PROXY_PID}
wait_healthy "${PROXY_IP}" "${PROXY_PORT}" "proxy (${NODE_PROXY_SHORT})" /healthcheck proxy || exit 1
kill -TERM "${TAIL_PROXY_PID}" 2>/dev/null

# --- Samplers. The connection split is decided in the first seconds of a bench
# and cannot be recovered afterwards.
start_sampler() {
    : > "$3"
    ssh -n "$1" "setsid nohup bash ${SHARED}/sample_conns.sh ${SAMPLE_STOP_MARKER} \
        $3 $2 ${SAMPLE_INTERVAL_S} $4 serving > /dev/null 2>&1 < /dev/null &"
}
start_sampler "${NODE_P_HEAD}" "${P_PORT}"     "${SHARED}/logs/conns_p-head.log"     p-head
start_sampler "${NODE_P_TAIL}" "-"             "${SHARED}/logs/conns_p-headless.log" p-headless
start_sampler "${NODE_D_HEAD}" "${D_PORT}"     "${SHARED}/logs/conns_d-head.log"     d-head
start_sampler "${NODE_D_TAIL}" "-"             "${SHARED}/logs/conns_d-headless.log" d-headless
start_sampler "${NODE_PROXY}"  "${PROXY_PORT}" "${SHARED}/logs/conns_proxy.log"      proxy
: > "${SHARED}/logs/conns_client.log"
setsid nohup bash "${SHARED}/sample_conns.sh" "${SAMPLE_STOP_MARKER}" \
    "${SHARED}/logs/conns_client.log" "-" "${SAMPLE_INTERVAL_S}" client client \
    > /dev/null 2>&1 < /dev/null &
sleep 2

# Unattended: nothing else gives this shell $no_proxy (bench_arm.sh preflight 0
# hard-fails without it) or HF_HOME for the tokenizer load. After the servers
# are up, so the launcher's CC/CXX never reach them.
source "${SHARED}/common_env.sh"

# --- The ladder ---------------------------------------------------------------
# One launch per (ISL,OSL): --max-model-len depends only on ISL+OSL, and a
# relaunch costs two serialised weight loads per concurrency.
# A failed cell does not fail the job: the wrapper retries the cells this file
# marks failed and skips the rest. bench_arm.sh creates OUT_DIR before it can
# fail, so a half-written bench dir is indistinguishable from a finished one --
# this file is the only record of which dirs are trustworthy.
STATUS_TSV="${SHARED}/bench_status.tsv"
printf 'isl\tosl\tconcurrency\tstatus\texit_code\tbench_dir\n' > "${STATUS_TSV}"
RC=0
for c in ${CONCURRENCIES}; do
    echo ""
    echo "=== cell isl=${ISL} osl=${OSL} c=${c} ==="
    _out="${SHARED}/bench/$(date +%Y%m%d_%H%M%S)_disagg_isl${ISL}_osl${OSL}_c${c}"
    # iter32k supplies the warmup (16 prompts, OSL 128); everything else below
    # overrides it. Its label and SLOs still land in config.json and are NOT
    # this cell's -- post-processing reads the explicit isl/osl fields and the
    # directory tag, never the label.
    ARM=disagg \
    MODEL="${MODEL}" \
    GPUS=$(( DP * TP * 2 )) \
    BASE_URL="http://${PROXY_IP}:${PROXY_PORT}" \
    METRICS_URLS="http://${P_HEAD_IP}:${P_PORT},http://${D_HEAD_IP}:${D_PORT}" \
    CXI_NODES="${NODE_P_HEAD},${NODE_P_TAIL},${NODE_D_HEAD},${NODE_D_TAIL}" \
    WORKLOAD=iter32k \
    ISL="${ISL}" OSL="${OSL}" \
    NUM_PROMPTS=$(( PROMPTS_PER_STREAM * c )) \
    MAX_CONCURRENCY="${c}" \
    RUN_DIR="${SHARED}" \
    OUT_DIR="${_out}" \
        bash "${SCRIPT_DIR}/bench_arm.sh"
    _rc=$?
    if [ "${_rc}" -eq 0 ]; then _st=ok; else _st=failed; fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${ISL}" "${OSL}" "${c}" "${_st}" "${_rc}" "${_out}" >> "${STATUS_TSV}"
    # Carry on after a failed cell -- the launch cost two weight loads -- but not
    # into a dead role. Both heads, not the proxy: the proxy answers
    # /healthcheck from its own process and says nothing about P or D.
    for _h in "${P_HEAD_IP}:${P_PORT}:P" "${D_HEAD_IP}:${D_PORT}:D"; do
        if ! curl -s -o /dev/null -w '%{http_code}' "http://${_h%:*}/health" 2>/dev/null | grep -q 200; then
            echo "${_h##*:} unhealthy after c=${c}; the cells after it did not run." >&2
            RC=30; break 2
        fi
    done
done

if [ "${KEEP_ALIVE}" = "1" ]; then
    echo ""
    echo "KEEP_ALIVE=1 -- servers held. Ctrl-C tears down."
    echo "  bench endpoint  http://${PROXY_IP}:${PROXY_PORT}   (source ${SHARED}/common_env.sh first)"
    echo "  P direct        http://${P_HEAD_IP}:${P_PORT}"
    echo "  D direct        http://${D_HEAD_IP}:${D_PORT}"
    while true; do
        sleep "${KEEP_ALIVE_POLL_S}"
        printf '  [keep-alive %s] up\n' "$(date +%H:%M:%S)"
    done
fi

echo ""
summarize_conn_sampler "${SHARED}/logs/conns_p-head.log" "p-head"
summarize_conn_sampler "${SHARED}/logs/conns_d-head.log" "d-head"
summarize_conn_sampler "${SHARED}/logs/conns_proxy.log"  "proxy"
summarize_conn_sampler "${SHARED}/logs/conns_client.log" "client"
exit "${RC}"
