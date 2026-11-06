#!/bin/bash
set -uo pipefail
# Muse-Spark router arm: 2 colocated TP4 x PP8 instances, vllm-router round_robin on its own node,
# gates, one (ISL,OSL) ladder through the router, teardown. Runs on the client node.
# Ported from run_muse_spark_parity_pd_tp4pp8_N16.sh and run_muse_spark_pilot_tp4pp8_N8.sh.
# Exit 0 ladder ran (per cell: bench_status.tsv) | 1 config or boot | 10 dirty node | 30 server died
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TN="$(cd "${HERE}/../.." && pwd)"

ISL=${ISL:?set ISL}
OSL=${OSL:?set OSL}
CONCURRENCIES=${CONCURRENCIES:-32}
EP=${EP:?set EP}
EXCLUDE_NODES=${EXCLUDE_NODES:-}
PROMPTS_PER_STREAM=4
MODEL=meta-models/Muse-Spark-Open-816B-A42B
NNODES=8 TP=4 PP=8
MAX_MODEL_LEN=1048576
API_SERVER_COUNT=4
A_PORT=8100 B_PORT=8200 ROUTER_PORT=8000 ROUTER_METRICS_PORT=29000
A_MASTER_PORT=29550 B_MASTER_PORT=29560
HEALTH_S=7500
GPU_DIRTY_MIB=4096
RUNS_ROOT=/vast/draco/tara/projects/Tara_Deployment/software/testing/RUNS
STAMP=$(date +%Y%m%d_%H%M%S)
SHARED=${RUNS_ROOT}/muse_spark_router_tp${TP}pp${PP}x2_ep${EP}_isl${ISL}_osl${OSL}_mml${MAX_MODEL_LEN}${RUN_TAG:+_${RUN_TAG}}_${STAMP}
EP_FLAG=; [ "${EP}" = 1 ] && EP_FLAG=--enable-expert-parallel

ls -d /vast/draco/tara/projects/Tara_Deployment/software/model-weights/hub/models--meta-models--Muse-Spark-Open-816B-A42B/snapshots/0fb02478* >/dev/null 2>&1 \
    || { echo "FATAL: Muse-Spark snapshot 0fb02478 not in the cache." >&2; exit 1; }

# --- Nodes: A = 0..7 (node k = stage k), B = 8..15, router = 16. This client never serves.
[ -r "${PBS_NODEFILE:-}" ] || { echo "FATAL: no readable PBS_NODEFILE." >&2; exit 1; }
CLIENT_HOST=$(hostname -s); CLIENT_HOST=${CLIENT_HOST%%.*}
NODES=(); SHORT=()
while read -r n; do
    s=${n%%.*}
    [ "${s}" = "${CLIENT_HOST}" ] && continue
    [[ " ${EXCLUDE_NODES} " == *" ${s} "* ]] && continue
    NODES+=("${n}"); SHORT+=("${s}")
done < <(sort -u "${PBS_NODEFILE}")
NEED=$(( 2 * NNODES + 1 ))
[ "${#NODES[@]}" -ge "${NEED}" ] || { echo "FATAL: need ${NEED} nodes besides the client; have ${#NODES[@]}." >&2; exit 1; }
NODES=("${NODES[@]:0:NEED}"); SHORT=("${SHORT[@]:0:NEED}")
ENGINE=("${NODES[@]:0:2*NNODES}")
B0=${NNODES} RT=$(( 2 * NNODES ))

hsn() { ssh -n "$1" "ip -4 -o addr show hsn0" 2>/dev/null | awk '{print $4}' | cut -d/ -f1; }
A_IP=$(hsn "${NODES[0]}"); B_IP=$(hsn "${NODES[B0]}"); ROUTER_IP=$(hsn "${NODES[RT]}")
CLIENT_IP=$(ip -4 -o addr show hsn0 2>/dev/null | awk '{print $4}' | cut -d/ -f1)
[ -n "${A_IP}" ] && [ -n "${B_IP}" ] && [ -n "${ROUTER_IP}" ] && [ -n "${CLIENT_IP}" ] \
    || { echo "FATAL: missing an hsn0 address." >&2; exit 1; }
ROUTER_URL=http://${ROUTER_IP}:${ROUTER_PORT}
echo "client ${CLIENT_HOST} ${CLIENT_IP}  router ${SHORT[RT]} ${ROUTER_IP}"
echo "A ${A_IP} stages ${SHORT[*]:0:NNODES}"
echo "B ${B_IP} stages ${SHORT[*]:B0:NNODES}"

# NCCL will not create its directory, and falls back to stdout silently.
mkdir -p "${SHARED}/logs/nccl" "${SHARED}/router" || { echo "FATAL: cannot create ${SHARED}" >&2; exit 1; }

# --- Preflight ---------------------------------------------------------------
_dirty=()
for i in "${!ENGINE[@]}"; do
    ssh -n "${NODES[$i]}" "GPU_DIRTY_MIB=${GPU_DIRTY_MIB} bash ${TN}/gpu_cleanup.sh report" 2>&1 | sed "s/^/  [${SHORT[$i]}] /"
    [ "${PIPESTATUS[0]}" -eq 0 ] || _dirty+=("${SHORT[$i]}")
done
ssh -n "${NODES[0]}"  "fuser ${A_PORT}/tcp" >/dev/null 2>&1 && _dirty+=("${SHORT[0]}")
ssh -n "${NODES[B0]}" "fuser ${B_PORT}/tcp" >/dev/null 2>&1 && _dirty+=("${SHORT[B0]}")
ssh -n "${NODES[RT]}" "fuser ${ROUTER_PORT}/tcp ${ROUTER_METRICS_PORT}/tcp" >/dev/null 2>&1 && _dirty+=("${SHORT[RT]}")
if [ "${#_dirty[@]}" -gt 0 ]; then
    printf 'DIRTY_NODE:%s\n' "${_dirty[@]}" | sort -u >&2
    exit 10
fi

NO_PROXY_LIST="localhost,127.0.0.1,${A_IP},${B_IP},${ROUTER_IP},${CLIENT_IP},${NODES[0]},${SHORT[0]},${NODES[B0]},${SHORT[B0]},${NODES[RT]},${SHORT[RT]},${CLIENT_HOST}"
bash "${HERE}/../emit_common_env_disaggpp_nixl_1.5.2.sh" "${SHARED}/common_env.sh" "${NO_PROXY_LIST}" \
    "${SHARED}/logs/nccl/nccl.%h.%p.log" || exit 1
for n in "${NODES[2*NNODES-1]}" "${NODES[RT]}"; do
    ssh -n "${n}" "test -f ${SHARED}/common_env.sh" || { echo "FATAL: ${SHARED} not visible on ${n%%.*}." >&2; exit 1; }
done

# --- Role script: one file, 16 nodes; instance and node rank from this host's place in the list.
cat > "${SHARED}/launch_role.sh" <<EOF
#!/bin/bash
NODES=(${SHORT[*]:0:2*NNODES})
ME=\$(hostname -s); ME=\${ME%%.*}
RANK=
for i in "\${!NODES[@]}"; do [ "\${NODES[\$i]}" = "\${ME}" ] && RANK=\$i; done
[ -n "\${RANK}" ] || { echo "Rank landed on unexpected host \${ME}." >&2; exit 1; }
if [ "\${RANK}" -lt ${NNODES} ]; then
    ROLE=a NR=\${RANK} HEAD=${A_IP} PORT=${A_PORT} MPORT=${A_MASTER_PORT}
else
    ROLE=b NR=\$(( RANK - ${NNODES} )) HEAD=${B_IP} PORT=${B_PORT} MPORT=${B_MASTER_PORT}
fi
exec > ${SHARED}/logs/\${ROLE}\${NR}.log 2>&1
source ${SHARED}/common_env.sh
# The ZMQ control queues advertise this address; unset, vLLM picks the management NIC.
export VLLM_HOST_IP=\$(ip -4 -o addr show hsn0 | awk '{print \$4}' | cut -d/ -f1)
[ -n "\${VLLM_HOST_IP}" ] || { echo "FATAL: no hsn0 address on \${ME}."; exit 1; }
env | sort > ${SHARED}/logs/env_\${ROLE}\${NR}.txt
echo "=== host=\${ME} role=\${ROLE} node_rank=\${NR} VLLM_HOST_IP=\${VLLM_HOST_IP} SLINGSHOT_VNIS=\${SLINGSHOT_VNIS:-<UNSET>} ==="
# B loads after A: two 1.5 TiB reads off /vast in sequence, not at once.
if [ "\${ROLE}" = b ]; then
    t0=\$(date +%s)
    until curl --noproxy '*' -s -o /dev/null -w '%{http_code}' http://${A_IP}:${A_PORT}/health 2>/dev/null | grep -q 200; do
        [ \$(( \$(date +%s) - t0 )) -lt ${HEALTH_S} ] || { echo "A never healthy."; exit 1; }
        sleep 5
    done
    echo "=== A up after \$(( \$(date +%s) - t0 ))s ==="
fi
if [ "\${NR}" -eq 0 ]; then
    SERVE=(--host 0.0.0.0 --port \${PORT} --api-server-count ${API_SERVER_COUNT})
else
    SERVE=(--headless)
fi
exec vllm serve ${MODEL} "\${SERVE[@]}" \\
    --nnodes ${NNODES} --node-rank \${NR} --master-addr \${HEAD} --master-port \${MPORT} \\
    --distributed-executor-backend mp \\
    --tensor-parallel-size ${TP} --pipeline-parallel-size ${PP} ${EP_FLAG} \\
    --max-model-len ${MAX_MODEL_LEN} \\
    --gpu-memory-utilization 0.85 --dtype bfloat16 --kv-cache-dtype auto --moe-backend triton \\
    --no-disable-hybrid-kv-cache-manager --no-enable-prefix-caching \\
    --max-num-batched-tokens 16384 --max-num-seqs 256 --long-prefill-token-threshold 8192 \\
    --language-model-only \\
    --numa-bind --numa-bind-nodes 0 1 2 3 --numa-bind-cpus 4-71 76-143 148-215 220-287 \\
    --distributed-timeout-seconds 3600 --cpu-distributed-timeout-seconds 3600 \\
    --enable-logging-iteration-details --cudagraph-metrics
EOF
chmod +x "${SHARED}/launch_role.sh"

# Unset prometheus port disables router metrics. Retries and the breaker would hide failed requests.
cat > "${SHARED}/launch_router.sh" <<EOF
#!/bin/bash
source ${SHARED}/common_env.sh
exec vllm-router --host 0.0.0.0 --port ${ROUTER_PORT} \\
    --worker-urls http://${A_IP}:${A_PORT} http://${B_IP}:${B_PORT} --policy round_robin \\
    --request-timeout-secs 172800 --disable-retries --disable-circuit-breaker \\
    --health-check-timeout-secs 30 \\
    --prometheus-host 0.0.0.0 --prometheus-port ${ROUTER_METRICS_PORT}
EOF
chmod +x "${SHARED}/launch_router.sh"

{
    echo "stamp            ${STAMP}"
    echo "model            ${MODEL}"
    echo "workload         isl ${ISL}  osl ${OSL}  concurrencies ${CONCURRENCIES}  through the router"
    echo "client           ${CLIENT_HOST} ${CLIENT_IP}"
    echo "A                ${A_IP} port ${A_PORT}  stages ${SHORT[*]:0:NNODES}"
    echo "B                ${B_IP} port ${B_PORT}  stages ${SHORT[*]:B0:NNODES}"
    echo "router           ${SHORT[RT]} ${ROUTER_IP} port ${ROUTER_PORT} metrics ${ROUTER_METRICS_PORT}  round_robin"
    echo "excluded         ${EXCLUDE_NODES:-none}"
    echo "parallelism      2 x (TP ${TP} x PP ${PP}), EP ${EP}"
    echo "git              $(git -C "${TN}" rev-parse --short HEAD 2>/dev/null || echo '(none)')"
} > "${SHARED}/run_config.txt"
cat "${SHARED}/run_config.txt"

GPU_WATCH_STOP="${SHARED}/gpu_watch_stop"
cleanup() {
    touch "${GPU_WATCH_STOP}" 2>/dev/null
    kill -TERM ${MPIEXEC_PID:-} ${TAIL_A_PID:-} ${TAIL_B_PID:-} ${TAIL_M_PID:-} ${ROUTER_SSH_PID:-} 2>/dev/null
    ssh -n "${NODES[RT]}" "pkill -f '[v]llm-router'; fuser -k ${ROUTER_PORT}/tcp ${ROUTER_METRICS_PORT}/tcp" 2>/dev/null
    for i in "${!ENGINE[@]}"; do
        ssh -n "${NODES[$i]}" "bash ${TN}/gpu_cleanup.sh kill ${A_PORT} ${B_PORT} ${A_MASTER_PORT} ${B_MASTER_PORT}" 2>&1 | sed "s/^/  [${SHORT[$i]}] /" &
    done
    wait
    echo "RUN_DIR:${SHARED}"
}
# EXIT is the only teardown path; INT/TERM just exit into it.
trap cleanup EXIT
trap 'exit 130' INT TERM

for k in $(seq 0 $((NNODES - 1))); do touch "${SHARED}/logs/a${k}.log" "${SHARED}/logs/b${k}.log"; done
touch "${SHARED}/logs/mpiexec.log" "${SHARED}/logs/router.log"

# 1 Hz GPU memory trace per node, launch to healthy only (jitter on a serving node).
cat > "${SHARED}/gpu_watch.sh" <<EOF
#!/bin/bash
while [ ! -f "${GPU_WATCH_STOP}" ]; do
    nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total --format=csv,noheader,nounits | sed 's/^/dev, /'
    nvidia-smi --query-compute-apps=timestamp,pid,used_gpu_memory --format=csv,noheader,nounits | sed 's/^/app, /'
    sleep 1
done
EOF
for i in "${!ENGINE[@]}"; do
    ssh -n "${NODES[$i]}" "setsid nohup bash ${SHARED}/gpu_watch.sh > ${SHARED}/logs/gpu_${SHORT[$i]}.log 2>&1 < /dev/null &"
done

# One mpiexec for both instances: they never talk, so this only keeps the P:D launch shape.
mpiexec -n "$(( 2 * NNODES ))" -ppn 1 --hosts "$(IFS=,; echo "${ENGINE[*]}")" --cpu-bind none \
    bash "${SHARED}/launch_role.sh" > "${SHARED}/logs/mpiexec.log" 2>&1 &
MPIEXEC_PID=$!
tail -n +1 -f "${SHARED}/logs/a0.log"      > >(sed -u 's/^/[A0]  /') & TAIL_A_PID=$!; disown "${TAIL_A_PID}"
tail -n +1 -f "${SHARED}/logs/b0.log"      > >(sed -u 's/^/[B0]  /') & TAIL_B_PID=$!; disown "${TAIL_B_PID}"
tail -n +1 -f "${SHARED}/logs/mpiexec.log" > >(sed -u 's/^/[MPI] /') & TAIL_M_PID=$!; disown "${TAIL_M_PID}"

up() { curl --noproxy '*' -s -o /dev/null -w '%{http_code}' "http://$1$2" 2>/dev/null | grep -q 200; }
wait_up() {
    local t0; t0=$(date +%s)
    while [ $(( $(date +%s) - t0 )) -lt "${HEALTH_S}" ]; do
        up "$1" "$2" && { echo "$3 healthy after $(( $(date +%s) - t0 ))s" | tee -a "${SHARED}/boot_time.txt"; return 0; }
        if ! kill -0 "${MPIEXEC_PID}" 2>/dev/null; then
            echo "mpiexec exited while waiting for $3." >&2
            for f in mpiexec a0 a7 b0 b7; do
                echo "--- tail logs/${f}.log ---" >&2
                tail -n 30 "${SHARED}/logs/${f}.log" 2>/dev/null | sed 's/^/    /' >&2
            done
            return 1
        fi
        sleep 5
    done
    echo "$3 never healthy." >&2
    return 1
}
healthy=0
wait_up "${A_IP}:${A_PORT}" /health A && wait_up "${B_IP}:${B_PORT}" /health B && healthy=1
touch "${GPU_WATCH_STOP}"
for i in "${!ENGINE[@]}"; do ssh -n "${NODES[$i]}" "pkill -f ${SHARED}/gpu_watch.sh" 2>/dev/null; done
kill -TERM "${TAIL_A_PID}" "${TAIL_B_PID}" "${TAIL_M_PID}" 2>/dev/null

if [ "${healthy}" -ne 1 ]; then
    # Only the raising path prints this string (Inkling -10-07d).
    for f in $(grep -lF 'is less than desired GPU memory utilization' "${SHARED}"/logs/[ab][0-9].log 2>/dev/null); do
        b=$(basename "${f}" .log); k=${b:1}; [ "${b:0:1}" = b ] && k=$(( k + NNODES ))
        echo "DIRTY_NODE:${SHORT[$k]}" >&2; _bad=1
    done
    [ -n "${_bad:-}" ] && exit 10
    exit 1
fi

# After boot, so the launcher env never reaches the servers.
source "${SHARED}/common_env.sh"

ssh -n "${NODES[RT]}" "bash ${SHARED}/launch_router.sh" > "${SHARED}/logs/router.log" 2>&1 &
ROUTER_SSH_PID=$!
wait_up "${ROUTER_IP}:${ROUTER_PORT}" /readiness router || { tail -n 30 "${SHARED}/logs/router.log" >&2; exit 1; }

# /readiness passes on one healthy worker; the arm needs both.
two_workers() { curl --noproxy '*' -s "${ROUTER_URL}/readiness" | grep -q '"healthy_workers":2'; }
rm_scrape() { curl --noproxy '*' -s "http://${ROUTER_IP}:${ROUTER_METRICS_PORT}/metrics" \
    | grep -E '^vllm_router_(processed_requests_total|request_errors_total|retries_total|worker_load)' > "${SHARED}/router/metrics_$1.txt"; }

# --- Gates -------------------------------------------------------------------
{
    echo "=== gates ==="
    for r in a b; do
        echo "kv_pool_tokens ${r}  $(grep -ho 'GPU KV cache size: [0-9,]* tokens' "${SHARED}"/logs/${r}[0-9].log | tr -dc '0-9\n' | sort -n | head -1)"
        for k in $(seq 0 $((NNODES - 1))); do
            echo "  ${r} stage ${k}  $(grep -o 'Available KV cache memory: [0-9.]* GiB' "${SHARED}/logs/${r}${k}.log" | awk '{print $5}' | sort -n | tr '\n' ' ')"
        done
    done
    echo "nccl logs non-empty $(find "${SHARED}/logs/nccl" -name 'nccl.*.log' -size +0c | wc -l)  with plugin $(grep -l 'Loaded net plugin' "${SHARED}"/logs/nccl/nccl.*.log 2>/dev/null | wc -l)  (want equal)"
    echo "scheduler lines (want no 'disabled' / 'does not yet support'):"
    grep -hoE 'Async scheduling is disabled.*|Model Runner V2 does not yet support.*|Hidden layers were unevenly partitioned.*' \
        "${SHARED}/logs/a0.log" "${SHARED}/logs/b0.log" | sort -u | sed 's/^/  /'
    echo "router readiness: $(curl --noproxy '*' -s "${ROUTER_URL}/readiness")  (want healthy_workers 2)"
    # round_robin: two requests land one on each instance.
    echo "coherence through the router, x2:"
    for i in 1 2; do
        curl --noproxy '*' -s "${ROUTER_URL}/v1/completions" -H 'Content-Type: application/json' \
            -d "{\"model\":\"${MODEL}\",\"prompt\":\"The capital of France is\",\"max_tokens\":32,\"temperature\":0}" \
            > "${SHARED}/coherence_${i}.json"
        python -c 'import json,sys; print("  " + repr(json.load(open(sys.argv[1]))["choices"][0]["text"]))' \
            "${SHARED}/coherence_${i}.json" 2>/dev/null || sed 's/^/  /' "${SHARED}/coherence_${i}.json"
    done
} 2>&1 | tee "${SHARED}/gates.txt"
two_workers || { echo "router does not see both instances healthy." >&2; exit 1; }

# --- Ladder: a failed cell does not stop it; a dead instance does. -----------
rm_scrape start
STATUS_TSV="${SHARED}/bench_status.tsv"
printf 'isl\tosl\tconcurrency\tstatus\texit_code\tbench_dir\n' > "${STATUS_TSV}"
RC=0
for c in ${CONCURRENCIES}; do
    echo ""
    echo "=== cell isl=${ISL} osl=${OSL} c=${c} ==="
    _out="${SHARED}/bench/$(date +%Y%m%d_%H%M%S)_muse_spark_router_isl${ISL}_osl${OSL}_c${c}"
    # iter32k only supplies the warmup; its label in config.json is not this cell's.
    ARM=router MODEL="${MODEL}" GPUS=$(( 2 * NNODES * TP )) \
    BASE_URL="${ROUTER_URL}" METRICS_URLS="http://${A_IP}:${A_PORT},http://${B_IP}:${B_PORT}" \
    CXI_NODES="$(IFS=,; echo "${ENGINE[*]}")" WORKLOAD=iter32k \
    ISL="${ISL}" OSL="${OSL}" NUM_PROMPTS=$(( PROMPTS_PER_STREAM * c )) MAX_CONCURRENCY="${c}" \
    RUN_DIR="${SHARED}" OUT_DIR="${_out}" \
        bash "${TN}/bench_arm.sh"
    _rc=$?
    # vllm bench exits 0 even when requests fail.
    [ "${_rc}" -eq 0 ] && ! python -c 'import json,sys; r=json.load(open(sys.argv[1])); sys.exit(r["failed"] > 0 or r["completed"] != int(sys.argv[2]))' \
        "${_out}/result.json" "$(( PROMPTS_PER_STREAM * c ))" && _rc=20
    if [ "${_rc}" -eq 0 ]; then _st=ok; else _st=failed; fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${ISL}" "${OSL}" "${c}" "${_st}" "${_rc}" "${_out}" >> "${STATUS_TSV}"
    rm_scrape "c${c}"
    if ! up "${A_IP}:${A_PORT}" /health || ! up "${B_IP}:${B_PORT}" /health || ! two_workers; then
        echo "an instance is unhealthy after c=${c}; the cells after it did not run." >&2
        RC=30; break
    fi
done
exit "${RC}"
