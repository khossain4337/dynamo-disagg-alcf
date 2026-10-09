#!/bin/bash
set -uo pipefail

# Colocated arm, unattended: bring up DP2 x TP4, bench one (ISL,OSL) across a
# concurrency ladder, tear down. Flag rationale and THE ONE RULE are in
# run_inkling_colocated_N2.sh -- this file must issue the same `vllm serve`.
#
# Runs ON THE CLIENT NODE. Needs DP serving nodes plus this one.
#
# Exit  0 ladder ran to completion -- per-cell outcomes in bench_status.tsv
#       1 config | 10 dirty node (one DIRTY_NODE line per node) | 11 pool gate
#      30 server died mid-ladder; the cells after it did not run

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Also in common_env.sh; set here too so they hold even if that file is wrong,
# and so the launch line needs no env prefix. mpiexec forwards them to the
# server. Standing constraints -- HANDOFF.
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
# The node wiring below is exactly head + headless, and TP>4 would put the
# attention all-reduce on Slingshot. A different shape gets a different script.
if [ "${DP}" -ne 2 ] || [ "${TP}" -gt 4 ]; then
    echo "FATAL: this script is DP=2 x TP<=4; got DP=${DP} TP=${TP}." >&2
    exit 1
fi

GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-auto}
EXPERT_PARALLEL=${EXPERT_PARALLEL:-1}
HMA_FLAG=${HMA_FLAG:---no-disable-hybrid-kv-cache-manager}
PREFIX_CACHING_FLAG=${PREFIX_CACHING_FLAG:---no-enable-prefix-caching}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-16384}   # per-knob max of P and D
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}
BLOCK_SIZE=${BLOCK_SIZE:-}
ENFORCE_EAGER=${ENFORCE_EAGER:-}
NUMA_BIND=${NUMA_BIND:-1}
NUMA_BIND_NODES="${NUMA_BIND_NODES:-0 1 2 3}"            # indexed by GPU, space-separated
# Shared default with the disagg arm so the two agree by construction (THE ONE
# RULE). Passed explicitly because omitting it is not 1 -- serve.py:121
# silently substitutes data_parallel_size.
API_SERVER_COUNT=${API_SERVER_COUNT:-4}                  # 2026-10-08b

PORT=${PORT:-8100}
DP_RPC_PORT=${DP_RPC_PORT:-29550}
HEALTH_TRIES=${HEALTH_TRIES:-480}                        # x5s = 40 min
SAMPLE_INTERVAL_S=${SAMPLE_INTERVAL_S:-10}
GPU_DIRTY_MIB=${GPU_DIRTY_MIB:-4096}
# Default 1: the startup-memory failure cannot be scheduled, so the value is in
# being armed when it happens. A PBS script exports 0 to silence it.
GPU_WATCH=${GPU_WATCH:-1}
GPU_WATCH_INTERVAL_S=${GPU_WATCH_INTERVAL_S:-1}
RUNS_ROOT=${RUNS_ROOT:-/vast/draco/tara/projects/Tara_Deployment/software/testing/RUNS}
STAMP=$(date +%Y%m%d_%H%M%S)
RUN_TAG=${RUN_TAG:-}
SHARED=${SHARED:-${RUNS_ROOT}/inkling_colocated_dp${DP}tp${TP}_isl${ISL}_osl${OSL}_mml${MAX_MODEL_LEN}${RUN_TAG:+_${RUN_TAG}}_${STAMP}}

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
if [ "${#ROLE_NODES[@]}" -lt "${DP}" ]; then
    echo "FATAL: need ${DP} serving nodes plus this client; have ${#ROLE_NODES[@]}." >&2
    exit 1
fi
NODE_HEAD="${ROLE_NODES[0]}"; NODE_HEAD_SHORT="${NODE_HEAD%%.*}"
NODE_TAIL="${ROLE_NODES[1]}"; NODE_TAIL_SHORT="${NODE_TAIL%%.*}"

# ssh wants the FQDN; launch_role.sh's hostname test must be short-vs-short.
get_hsn_ip() { ssh -n "$1" "ip -4 -o addr show hsn0" 2>/dev/null | awk '{print $4}' | cut -d/ -f1; }
HEAD_IP=$(get_hsn_ip "${NODE_HEAD}")
TAIL_IP=$(get_hsn_ip "${NODE_TAIL}")
CLIENT_IP=$(ip -4 -o addr show hsn0 2>/dev/null | awk '{print $4}' | cut -d/ -f1)
if [ -z "${HEAD_IP}" ] || [ -z "${TAIL_IP}" ] || [ -z "${CLIENT_IP}" ]; then
    echo "FATAL: no hsn0 address on head=${HEAD_IP:-?} tail=${TAIL_IP:-?} client=${CLIENT_IP:-?}" >&2
    exit 1
fi
echo "client ${CLIENT_HOST}  head ${NODE_HEAD_SHORT} ${HEAD_IP}  headless ${NODE_TAIL_SHORT} ${TAIL_IP}"

# logs/nccl, not logs: NCCL will not create its own directory, and if it cannot
# open NCCL_DEBUG_FILE it falls back to stdout SILENTLY.
mkdir -p "${SHARED}/logs/nccl" || { echo "FATAL: cannot create ${SHARED}" >&2; exit 1; }

# Launcher defaults, forwarded by mpiexec. NCCL writes unprefixed and collides
# with vLLM's own prefixes, which is what mangled a KV line in the disagg arm
# (-10-07d). %h and %p keep the ranks off one file on a filesystem with no
# locking.
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,GRAPH,TUNING}
export NCCL_DEBUG_FILE=${NCCL_DEBUG_FILE:-${SHARED}/logs/nccl/nccl.%h.%p.log}

for f in fi_getinfo_shim.so env_for_libfabric_topology_error.sh gpu_cleanup.sh \
         emit_common_env.sh emit_conn_sampler.sh workload_profile.sh bench_arm.sh; do
    [ -f "${SCRIPT_DIR}/${f}" ] || { echo "FATAL: missing ${SCRIPT_DIR}/${f}" >&2; exit 1; }
done
SHIM="${SCRIPT_DIR}/fi_getinfo_shim.so"
if [ "$(strings "${SHIM}" 2>/dev/null | grep -c 'PATCH [234]' || true)" -lt 1 ]; then
    echo "WARNING: ${SHIM} looks stale against its .c -- rebuild before trusting a cross-arm diff." >&2
fi
ssh -n "${NODE_TAIL}" "test -d ${SHARED} && test -f ${SHIM}" || {
    echo "FATAL: ${SHARED} or the shim is not visible on ${NODE_TAIL_SHORT}." >&2
    exit 1
}

# --- Preflight. gpu_cleanup.sh owns the pattern list; sourced, never copied. ---
# shellcheck source=./gpu_cleanup.sh
source "${SCRIPT_DIR}/gpu_cleanup.sh"
PAT_VLLM="${VLLM_PROC_PATTERNS[0]}"
_dirty=()
for n in "${NODE_HEAD}" "${NODE_TAIL}"; do
    ssh -n "$n" "GPU_DIRTY_MIB=${GPU_DIRTY_MIB} bash ${SCRIPT_DIR}/gpu_cleanup.sh report" \
        2>&1 | sed "s/^/  [${n%%.*}] /"
    # PIPESTATUS, not $? -- $? is sed's and the gate would pass unconditionally.
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then _dirty+=("${n%%.*}"); fi
done
if ssh -n "${NODE_HEAD}" "fuser ${PORT}/tcp" >/dev/null 2>&1; then
    _dirty+=("${NODE_HEAD_SHORT}")
fi
if [ "${#_dirty[@]}" -gt 0 ]; then
    printf 'DIRTY_NODE:%s\n' "${_dirty[@]}" | sort -u >&2
    echo "Clear with: ssh <n> bash ${SCRIPT_DIR}/gpu_cleanup.sh kill ${PORT} ${DP_RPC_PORT}" >&2
    exit 10
fi

# --- Shared env and sampler, from the shared emitters -------------------------
# The client node is in no_proxy too: the bench runs there, and without it curl
# to an hsn0 address returns a Squid page that reads as a dead server.
NO_PROXY_LIST="localhost,127.0.0.1,${HEAD_IP},${TAIL_IP},${CLIENT_IP},${NODE_HEAD},${NODE_TAIL},${NODE_HEAD_SHORT},${NODE_TAIL_SHORT},${CLIENT_HOST}"
ENV_SCRIPT="${SCRIPT_DIR}/env_for_libfabric_topology_error.sh"
UCX_LINES=""
GPU_PIN_LINE=""
# shellcheck source=./emit_common_env.sh
source "${SCRIPT_DIR}/emit_common_env.sh"
emit_common_env "${SHARED}/common_env.sh" || exit 1
# shellcheck source=./emit_conn_sampler.sh
source "${SCRIPT_DIR}/emit_conn_sampler.sh"
emit_conn_sampler "${SHARED}/sample_conns.sh"

# --- Role script: one file, both nodes, selecting by hostname -----------------
# One mpiexec, because PALS allocates a VNI per application and two launches get
# two VNIs that cannot reach each other. No NIXL env here -- that, the connector
# flag and the batching split are the complete difference from the disagg arm.
cat > "${SHARED}/launch_role.sh" <<EOF
#!/bin/bash
MY_HOST="\$(hostname -s)"
MY_HOST="\${MY_HOST%%.*}"
if [ "\${MY_HOST}" = "${NODE_HEAD_SHORT}" ]; then
    LOG=${SHARED}/logs/head.log
    DP_ARGS=(--data-parallel-size ${DP}
             --data-parallel-size-local ${DP_LOCAL}
             --data-parallel-address ${HEAD_IP}
             --data-parallel-rpc-port ${DP_RPC_PORT}
             --api-server-count ${API_SERVER_COUNT})
    SERVE_ARGS=(--host 0.0.0.0 --port ${PORT})
elif [ "\${MY_HOST}" = "${NODE_TAIL_SHORT}" ]; then
    LOG=${SHARED}/logs/headless.log
    # --api-server-count is a hard error with --headless (serve.py:66-71).
    DP_ARGS=(--headless
             --data-parallel-size ${DP}
             --data-parallel-size-local ${DP_LOCAL}
             --data-parallel-start-rank 1
             --data-parallel-address ${HEAD_IP}
             --data-parallel-rpc-port ${DP_RPC_PORT})
    SERVE_ARGS=()
else
    echo "Rank landed on unexpected host '\${MY_HOST}'." >&2
    exit 1
fi

# Redirect BEFORE sourcing: a common_env.sh that fails to generate correctly
# reports it here rather than into mpiexec.log.
exec > "\${LOG}" 2>&1
source ${SHARED}/common_env.sh
echo "=== host=\$(hostname -s) SLINGSHOT_VNIS=\${SLINGSHOT_VNIS:-<UNSET>} ==="
if [ -z "\${SLINGSHOT_VNIS:-}" ]; then
    echo "SLINGSHOT_VNIS is EMPTY -- not launched under PALS. Anything reaching"
    echo "the cxi provider fails fi_domain() with -FI_ENOSYS."
fi
export LD_PRELOAD=${SHIM}

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
    --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} \\
    --max-num-seqs ${MAX_NUM_SEQS} \\
    \${EXTRA[@]+"\${EXTRA[@]}"}
EOF
chmod +x "${SHARED}/launch_role.sh"

{
    echo "stamp            ${STAMP}"
    echo "arm              colocated   isl ${ISL}  osl ${OSL}  concurrencies ${CONCURRENCIES}"
    echo "model            ${MODEL}"
    echo "client           ${CLIENT_HOST} ${CLIENT_IP}"
    echo "head             ${NODE_HEAD_SHORT} ${HEAD_IP} port ${PORT}  api-servers ${API_SERVER_COUNT}"
    echo "headless         ${NODE_TAIL_SHORT} ${TAIL_IP}"
    echo "parallelism      DP ${DP} x TP ${TP}  expert-parallel ${EXPERT_PARALLEL}"
    echo "max-model-len    ${MAX_MODEL_LEN}"
    echo "batching         ${MAX_NUM_BATCHED_TOKENS} tok / ${MAX_NUM_SEQS} seq"
    echo "numa-bind        ${NUMA_BIND} nodes ${NUMA_BIND_NODES}"
    # No moe_tuned line: what the server actually loaded is its own log's
    # "Using configuration from ... for MoE layer" / "Using default MoE
    # config", which the post-step greps out of RUNS/ (-10-02c).
    echo "shim             sha256 $(sha256sum "${SHIM}" 2>/dev/null | awk '{print $1}')"
    echo "git              $(git -C "${SCRIPT_DIR}" rev-parse --short HEAD 2>/dev/null || echo '(none)')"
} > "${SHARED}/run_config.txt"
cat "${SHARED}/run_config.txt"

SAMPLE_STOP_MARKER="${SHARED}/sample_stop_${STAMP}"
GPU_WATCH_STOP_MARKER="${SHARED}/gpu_watch_stop_${STAMP}"
cleanup() {
    # The GPU marker is also touched at healthy. This one only covers an abort
    # before that point, which would otherwise leave two 1 Hz loops writing to
    # /vast forever.
    touch "${SAMPLE_STOP_MARKER}" "${GPU_WATCH_STOP_MARKER}" 2>/dev/null
    for n in "${NODE_HEAD}" "${NODE_TAIL}"; do
        ssh -n "$n" "pkill -TERM -f \"${PAT_VLLM}\"" 2>/dev/null
    done
    kill -TERM ${MPIEXEC_PID:-} ${TAIL_H_PID:-} ${TAIL_T_PID:-} ${TAIL_M_PID:-} 2>/dev/null
    sleep 5
    # Same file preflight screened with, so a launch cannot check one pattern
    # set and kill another. Read the VERDICT lines: a node left DIRTY is the
    # next run's failure, and this is the last moment it can be handed back.
    for n in "${NODE_HEAD}" "${NODE_TAIL}"; do
        ssh -n "$n" "bash ${SCRIPT_DIR}/gpu_cleanup.sh kill ${PORT} ${DP_RPC_PORT}" \
            2>&1 | sed "s/^/  [${n%%.*}] /"
    done
    echo "RUN_DIR:${SHARED}"
}
# EXIT is the single teardown path. A handler on INT/TERM that ran cleanup
# would RETURN, resuming the health loop or the ladder against killed servers,
# and then fire cleanup again at EXIT. These only exit; EXIT does the work.
trap cleanup EXIT
trap 'exit 130' INT TERM

# --cpu-bind none: PALS otherwise confines all four TP workers to rank 0's core
# slice, which looks like a fabric problem rather than an error.
MPI_CPU_BIND=${MPI_CPU_BIND-none}
CPU_BIND_ARGS=()
[ -n "${MPI_CPU_BIND}" ] && CPU_BIND_ARGS=(--cpu-bind "${MPI_CPU_BIND}")

# Pre-create all three: `cmd > file &` creates the file in the child, so tail -f
# can lose the race and silently follow nothing. mpiexec.log is the only stream
# that carries PALS-level failures.
touch "${SHARED}/logs/head.log" "${SHARED}/logs/headless.log" "${SHARED}/logs/mpiexec.log"

# --- GPU watcher: a 1 Hz memory trace per engine node, launch to healthy. It
# must not reach the ladder -- 1 Hz on a serving node adds jitter against a
# ~42 ms clean step (-10-07c). The remote loop exits on its own when it sees the
# marker, so there is no pid to kill.
if [ "${GPU_WATCH}" = "1" ]; then
    cat > "${SHARED}/gpu_watch.sh" <<EOF
#!/bin/bash
# Two line types, tagged in column 1: 'dev' is per device, 'app' is per holding
# process. Device memory that no 'app' line accounts for is held by a process
# this node cannot enumerate -- on 165759 that gap was ~27 GiB while the node's
# own worker was attributed correctly in the same second.
while [ ! -f "${GPU_WATCH_STOP_MARKER}" ]; do
    nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total \\
        --format=csv,noheader,nounits | sed 's/^/dev, /'
    nvidia-smi --query-compute-apps=timestamp,pid,used_gpu_memory \\
        --format=csv,noheader,nounits | sed 's/^/app, /'
    sleep ${GPU_WATCH_INTERVAL_S}
done
EOF
    for n in "${NODE_HEAD}" "${NODE_TAIL}"; do
        ssh -n "$n" "setsid nohup bash ${SHARED}/gpu_watch.sh \
            > ${SHARED}/logs/gpu_${n%%.*}.log 2>&1 < /dev/null &"
    done
fi

mpiexec -n "${DP}" -ppn 1 --hosts "${NODE_HEAD},${NODE_TAIL}" \
    ${CPU_BIND_ARGS[@]+"${CPU_BIND_ARGS[@]}"} \
    bash "${SHARED}/launch_role.sh" > "${SHARED}/logs/mpiexec.log" 2>&1 &
MPIEXEC_PID=$!

# $! on a pipeline names its LAST member, so `| sed` captured sed and the tails
# outlived the kill, dying on EPIPE. Process substitution keeps $! as tail.
tail -n +1 -f "${SHARED}/logs/head.log"     > >(sed -u 's/^/[HEAD] /') & TAIL_H_PID=$!; disown ${TAIL_H_PID}
tail -n +1 -f "${SHARED}/logs/headless.log" > >(sed -u 's/^/[TAIL] /') & TAIL_T_PID=$!; disown ${TAIL_T_PID}
tail -n +1 -f "${SHARED}/logs/mpiexec.log"  > >(sed -u 's/^/[MPI]  /') & TAIL_M_PID=$!; disown ${TAIL_M_PID}

# A node can read 0 MiB and still fail init_device (-16d); memory is not a
# reliable detector, so the log is. Remedy all three times was to give that node
# the client role -- hence the DIRTY_NODE line and exit 10.
#
# This string alone, no alternation. It exists only in the raising path,
# v1/worker/utils.py:418-427. 'Free memory on device cuda' matches the same runs
# but would also match every healthy run if a vLLM bump ever added the device id
# to the healthy INFO line (-10-07d).
dirty_signature() {
    grep -lF 'is less than desired GPU memory utilization' \
        "${SHARED}"/logs/head*.log 2>/dev/null
}

t0=$(date +%s)
healthy=0
for i in $(seq 1 "${HEALTH_TRIES}"); do
    if curl -s -o /dev/null -w '%{http_code}' "http://${HEAD_IP}:${PORT}/health" 2>/dev/null | grep -q 200; then
        healthy=1; echo "healthy after $(( $(date +%s) - t0 ))s"; break
    fi
    # mpiexec is the liveness source of truth: a dead startup and a slow one are
    # identical from outside, and PALS failures never reach the role logs.
    if ! kill -0 "${MPIEXEC_PID}" 2>/dev/null; then
        echo "mpiexec exited during startup." >&2
        for _l in mpiexec head headless; do
            echo "--- tail logs/${_l}.log ---" >&2
            tail -n 40 "${SHARED}/logs/${_l}.log" 2>/dev/null | sed 's/^/    /' >&2
        done
        break
    fi
    [ $(( i % 12 )) -eq 0 ] && printf '  [%4ds] %s\n' "$(( $(date +%s) - t0 ))" \
        "$(tail -n 1 "${SHARED}/logs/head.log" 2>/dev/null | cut -c1-100)"
    sleep 5
done
# Here, not in cleanup: on a boot failure cleanup runs too, but on a healthy
# boot the trap would not fire until after the ladder. The marker is only the
# backstop -- it took ~50 s to propagate on /vast (2026-10-08) and the jitter
# rule needs the watchers gone now.
touch "${GPU_WATCH_STOP_MARKER}" 2>/dev/null
if [ "${GPU_WATCH}" = "1" ]; then
    for n in "${NODE_HEAD}" "${NODE_TAIL}"; do
        ssh -n "$n" "pkill -f ${SHARED}/gpu_watch.sh" 2>/dev/null
    done
fi
kill -TERM "${TAIL_H_PID}" "${TAIL_T_PID}" "${TAIL_M_PID}" 2>/dev/null
if [ "${healthy}" -ne 1 ]; then
    _sig=$(dirty_signature)
    if [ -n "${_sig}" ]; then
        case "${_sig}" in *head.log*) echo "DIRTY_NODE:${NODE_HEAD_SHORT}" >&2 ;; esac
        case "${_sig}" in *headless.log*) echo "DIRTY_NODE:${NODE_TAIL_SHORT}" >&2 ;; esac
        exit 10
    fi
    exit 1
fi

# --- KV pool. Sized by the worst worker, so the minimum across ranks is it. ----
_pool=$(grep -h -o 'GPU KV cache size: [0-9,]* tokens' "${SHARED}/logs/head.log" \
        "${SHARED}/logs/headless.log" 2>/dev/null | grep -o '[0-9,]*' | tr -d ',' \
        | grep -v '^$' | sort -n | head -1)
echo "${_pool:-unknown}" > "${SHARED}/kv_pool_tokens.txt"
if [ -z "${_pool}" ]; then
    echo "WARNING: could not read the KV pool from either log." >&2
elif [ -n "${EXPECT_POOL:-}" ]; then
    _pct=$(( (_pool - EXPECT_POOL) * 100 / EXPECT_POOL ))
    if [ "${_pct}" -lt -2 ] || [ "${_pct}" -gt 2 ]; then
        # The pool lottery: one rank allocates more during init than its peers.
        # Relaunch, do not tune -- a run benched against a short pool compares
        # to nothing.
        echo "POOL GATE FAILED: expected ${EXPECT_POOL}, got ${_pool} (${_pct}%)." >&2
        exit 11
    fi
    echo "pool ${_pool} (gate passed, ${_pct}%)"
else
    echo "pool ${_pool} -- no EXPECT_POOL set; pass this on subsequent colocated launches."
fi

# Unattended: nothing else gives this shell $no_proxy (bench_arm.sh preflight 0
# hard-fails without it) or HF_HOME for the tokenizer load. After the servers
# are up, so the launcher's CC/CXX never reach them.
source "${SHARED}/common_env.sh"

# This arm is not a baseline if a connector is attached.
if grep -qiE 'NixlConnector|kv_transfer_config|KVConnector' "${SHARED}/logs/head.log"; then
    echo "WARNING: the head log mentions a KV connector. This arm must have none." >&2
fi

# Non-empty is the only proof the redirect took. NOT a count: %p is per process
# and every process that touches NCCL gets a file.
if [ -z "$(find "${SHARED}/logs/nccl" -name 'nccl.*.log' -size +0c -print -quit 2>/dev/null)" ]; then
    echo "WARNING: no non-empty file under logs/nccl -- NCCL_DEBUG_FILE did not" >&2
    echo "         take and NCCL went to stdout, back into the role logs." >&2
fi

echo ""
echo "=== KV cache groups ==="
grep -hiE 'kv_cache_group|KVCacheGroupSpec|SlidingWindowSpec|FullAttentionSpec|num_kv_cache_groups' \
    "${SHARED}/logs/head.log" 2>/dev/null | head -12 | sed 's/^/  /'

# --- Samplers. The connection split is decided in the first seconds of a bench
# and cannot be recovered afterwards.
start_sampler() {
    : > "$3"
    ssh -n "$1" "setsid nohup bash ${SHARED}/sample_conns.sh ${SAMPLE_STOP_MARKER} \
        $3 $2 ${SAMPLE_INTERVAL_S} $4 serving > /dev/null 2>&1 < /dev/null &"
}
start_sampler "${NODE_HEAD}" "${PORT}" "${SHARED}/logs/conns_head.log" head
start_sampler "${NODE_TAIL}" "-" "${SHARED}/logs/conns_headless.log" headless
: > "${SHARED}/logs/conns_client.log"
setsid nohup bash "${SHARED}/sample_conns.sh" "${SAMPLE_STOP_MARKER}" \
    "${SHARED}/logs/conns_client.log" "-" "${SAMPLE_INTERVAL_S}" client client \
    > /dev/null 2>&1 < /dev/null &
sleep 2

# --- The ladder ---------------------------------------------------------------
# One launch per (ISL,OSL): --max-model-len depends only on ISL+OSL, and a
# relaunch costs ~40 min of weight loading per concurrency.
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
    _out="${SHARED}/bench/$(date +%Y%m%d_%H%M%S)_colocated_isl${ISL}_osl${OSL}_c${c}"
    # iter32k supplies the warmup (16 prompts, OSL 128); everything else below
    # overrides it. Its label and SLOs still land in config.json and are NOT
    # this cell's -- post-processing reads the explicit isl/osl fields and the
    # directory tag, never the label.
    ARM=colocated \
    MODEL="${MODEL}" \
    GPUS=$(( DP * TP )) \
    BASE_URL="http://${HEAD_IP}:${PORT}" \
    METRICS_URLS="http://${HEAD_IP}:${PORT}" \
    CXI_NODES="${NODE_HEAD},${NODE_TAIL}" \
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
    # Carry on after a failed cell -- the launch cost 40 minutes -- but not into
    # a dead server, which would burn the rest of the ladder producing nothing.
    if ! curl -s -o /dev/null -w '%{http_code}' "http://${HEAD_IP}:${PORT}/health" 2>/dev/null | grep -q 200; then
        echo "server unhealthy after c=${c}; the cells after it did not run." >&2
        RC=30; break
    fi
done

echo ""
summarize_conn_sampler "${SHARED}/logs/conns_head.log"   "head"
summarize_conn_sampler "${SHARED}/logs/conns_client.log" "client"
exit "${RC}"
