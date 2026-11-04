#!/bin/bash
set -uo pipefail
# Muse-Spark greedy parity, colocated reference: one TP4 x PP8 instance on 8 nodes, gates,
# parity_client.py twice per concurrency, teardown. Ported from run_muse_spark_pilot_tp4pp8_N8.sh.
# Exit 0 parity ran | 1 config or boot | 10 dirty node | 20 failed requests | 30 server died
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TN="$(cd "${HERE}/../.." && pwd)"

PARITY_CONCURRENCIES=${PARITY_CONCURRENCIES:-1 32}
EP=${EP:-1}
EXCLUDE_NODES=${EXCLUDE_NODES:-}
MODEL=meta-models/Muse-Spark-Open-816B-A42B
NNODES=8 TP=4 PP=8
MAX_MODEL_LEN=1048576
API_SERVER_COUNT=4
PORT=8100
MASTER_PORT=29550
HEALTH_S=7500
GPU_DIRTY_MIB=4096
RUNS_ROOT=/vast/draco/tara/projects/Tara_Deployment/software/testing/RUNS
STAMP=$(date +%Y%m%d_%H%M%S)
SHARED=${RUNS_ROOT}/muse_spark_parity_colocated_tp${TP}pp${PP}_ep${EP}_mml${MAX_MODEL_LEN}${RUN_TAG:+_${RUN_TAG}}_${STAMP}
EP_FLAG=; [ "${EP}" = 1 ] && EP_FLAG=--enable-expert-parallel

ls -d /vast/draco/tara/projects/Tara_Deployment/software/model-weights/hub/models--meta-models--Muse-Spark-Open-816B-A42B/snapshots/0fb02478* >/dev/null 2>&1 \
    || { echo "FATAL: Muse-Spark snapshot 0fb02478 not in the cache." >&2; exit 1; }

# --- Nodes: everything but this client and the parked ones; first 8 serve, node k = PP stage k.
[ -r "${PBS_NODEFILE:-}" ] || { echo "FATAL: no readable PBS_NODEFILE." >&2; exit 1; }
CLIENT_HOST=$(hostname -s); CLIENT_HOST=${CLIENT_HOST%%.*}
NODES=(); SHORT=()
while read -r n; do
    s=${n%%.*}
    [ "${s}" = "${CLIENT_HOST}" ] && continue
    [[ " ${EXCLUDE_NODES} " == *" ${s} "* ]] && continue
    NODES+=("${n}"); SHORT+=("${s}")
done < <(sort -u "${PBS_NODEFILE}")
[ "${#NODES[@]}" -ge "${NNODES}" ] || { echo "FATAL: need ${NNODES} serving nodes; have ${#NODES[@]}." >&2; exit 1; }
NODES=("${NODES[@]:0:NNODES}"); SHORT=("${SHORT[@]:0:NNODES}")

HEAD_IP=$(ssh -n "${NODES[0]}" "ip -4 -o addr show hsn0" 2>/dev/null | awk '{print $4}' | cut -d/ -f1)
CLIENT_IP=$(ip -4 -o addr show hsn0 2>/dev/null | awk '{print $4}' | cut -d/ -f1)
[ -n "${HEAD_IP}" ] && [ -n "${CLIENT_IP}" ] || { echo "FATAL: no hsn0 address on head or client." >&2; exit 1; }
echo "client ${CLIENT_HOST} ${CLIENT_IP}  head ${SHORT[0]} ${HEAD_IP}  stages ${SHORT[*]}"

# NCCL will not create its directory, and falls back to stdout silently.
mkdir -p "${SHARED}/logs/nccl" || { echo "FATAL: cannot create ${SHARED}" >&2; exit 1; }

# --- Preflight ---------------------------------------------------------------
_dirty=()
for i in "${!NODES[@]}"; do
    ssh -n "${NODES[$i]}" "GPU_DIRTY_MIB=${GPU_DIRTY_MIB} bash ${TN}/gpu_cleanup.sh report" 2>&1 | sed "s/^/  [${SHORT[$i]}] /"
    [ "${PIPESTATUS[0]}" -eq 0 ] || _dirty+=("${SHORT[$i]}")
done
ssh -n "${NODES[0]}" "fuser ${PORT}/tcp" >/dev/null 2>&1 && _dirty+=("${SHORT[0]}")
if [ "${#_dirty[@]}" -gt 0 ]; then
    printf 'DIRTY_NODE:%s\n' "${_dirty[@]}" | sort -u >&2
    exit 10
fi

NO_PROXY_LIST="localhost,127.0.0.1,${HEAD_IP},${CLIENT_IP},${NODES[0]},${SHORT[0]},${CLIENT_HOST}"
bash "${HERE}/../emit_common_env_disaggpp_nixl_1.5.2.sh" "${SHARED}/common_env.sh" "${NO_PROXY_LIST}" \
    "${SHARED}/logs/nccl/nccl.%h.%p.log" || exit 1
ssh -n "${NODES[NNODES-1]}" "test -f ${SHARED}/common_env.sh" \
    || { echo "FATAL: ${SHARED} not visible on ${SHORT[NNODES-1]}." >&2; exit 1; }

# --- Role script: one file, all nodes; node rank from this host's place in the list.
cat > "${SHARED}/launch_role.sh" <<EOF
#!/bin/bash
NODES=(${SHORT[*]})
ME=\$(hostname -s); ME=\${ME%%.*}
RANK=
for i in "\${!NODES[@]}"; do [ "\${NODES[\$i]}" = "\${ME}" ] && RANK=\$i; done
[ -n "\${RANK}" ] || { echo "Rank landed on unexpected host \${ME}." >&2; exit 1; }
exec > ${SHARED}/logs/node\${RANK}.log 2>&1
source ${SHARED}/common_env.sh
# The ZMQ control queues advertise this address; unset, vLLM picks the management NIC.
export VLLM_HOST_IP=\$(ip -4 -o addr show hsn0 | awk '{print \$4}' | cut -d/ -f1)
[ -n "\${VLLM_HOST_IP}" ] || { echo "FATAL: no hsn0 address on \${ME}."; exit 1; }
env | sort > ${SHARED}/logs/env_node\${RANK}.txt
echo "=== host=\${ME} node_rank=\${RANK} VLLM_HOST_IP=\${VLLM_HOST_IP} gcc=\$(command -v gcc) SLINGSHOT_VNIS=\${SLINGSHOT_VNIS:-<UNSET>} ==="
if [ "\${RANK}" -eq 0 ]; then
    ROLE=(--host 0.0.0.0 --port ${PORT} --api-server-count ${API_SERVER_COUNT})
else
    ROLE=(--headless)
fi
exec vllm serve ${MODEL} "\${ROLE[@]}" \\
    --nnodes ${NNODES} --node-rank \${RANK} --master-addr ${HEAD_IP} --master-port ${MASTER_PORT} \\
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

{
    echo "stamp            ${STAMP}"
    echo "model            ${MODEL}"
    echo "workload         greedy parity, colocated  concurrencies ${PARITY_CONCURRENCIES}"
    echo "client           ${CLIENT_HOST} ${CLIENT_IP}"
    echo "head             ${SHORT[0]} ${HEAD_IP} port ${PORT}  api-servers ${API_SERVER_COUNT}"
    echo "stages           ${SHORT[*]}"
    echo "excluded         ${EXCLUDE_NODES:-none}"
    echo "parallelism      TP ${TP} x PP ${PP}, EP ${EP}"
    echo "git              $(git -C "${TN}" rev-parse --short HEAD 2>/dev/null || echo '(none)')"
} > "${SHARED}/run_config.txt"
cat "${SHARED}/run_config.txt"

GPU_WATCH_STOP="${SHARED}/gpu_watch_stop"
cleanup() {
    touch "${GPU_WATCH_STOP}" 2>/dev/null
    kill -TERM ${MPIEXEC_PID:-} ${TAIL_H_PID:-} ${TAIL_M_PID:-} 2>/dev/null
    for i in "${!NODES[@]}"; do
        ssh -n "${NODES[$i]}" "bash ${TN}/gpu_cleanup.sh kill ${PORT} ${MASTER_PORT}" 2>&1 | sed "s/^/  [${SHORT[$i]}] /" &
    done
    wait
    echo "RUN_DIR:${SHARED}"
}
# EXIT is the only teardown path; INT/TERM just exit into it.
trap cleanup EXIT
trap 'exit 130' INT TERM

for k in $(seq 0 $((NNODES - 1))); do touch "${SHARED}/logs/node${k}.log"; done
touch "${SHARED}/logs/mpiexec.log"

# 1 Hz GPU memory trace per node, launch to healthy only (jitter on a serving node).
cat > "${SHARED}/gpu_watch.sh" <<EOF
#!/bin/bash
while [ ! -f "${GPU_WATCH_STOP}" ]; do
    nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total --format=csv,noheader,nounits | sed 's/^/dev, /'
    nvidia-smi --query-compute-apps=timestamp,pid,used_gpu_memory --format=csv,noheader,nounits | sed 's/^/app, /'
    sleep 1
done
EOF
for i in "${!NODES[@]}"; do
    ssh -n "${NODES[$i]}" "setsid nohup bash ${SHARED}/gpu_watch.sh > ${SHARED}/logs/gpu_${SHORT[$i]}.log 2>&1 < /dev/null &"
done

# One mpiexec: PALS gives each launch its own VNI. --cpu-bind none: vLLM's --numa-bind does the pinning.
mpiexec -n "${NNODES}" -ppn 1 --hosts "$(IFS=,; echo "${NODES[*]}")" --cpu-bind none \
    bash "${SHARED}/launch_role.sh" > "${SHARED}/logs/mpiexec.log" 2>&1 &
MPIEXEC_PID=$!
tail -n +1 -f "${SHARED}/logs/node0.log"   > >(sed -u 's/^/[HEAD] /') & TAIL_H_PID=$!; disown "${TAIL_H_PID}"
tail -n +1 -f "${SHARED}/logs/mpiexec.log" > >(sed -u 's/^/[MPI]  /') & TAIL_M_PID=$!; disown "${TAIL_M_PID}"

t0=$(date +%s)
healthy=0
while [ $(( $(date +%s) - t0 )) -lt "${HEALTH_S}" ]; do
    if curl --noproxy '*' -s -o /dev/null -w '%{http_code}' "http://${HEAD_IP}:${PORT}/health" 2>/dev/null | grep -q 200; then
        healthy=1; break
    fi
    if ! kill -0 "${MPIEXEC_PID}" 2>/dev/null; then
        echo "mpiexec exited during startup." >&2
        for f in mpiexec node0 node1 node7; do
            echo "--- tail logs/${f}.log ---" >&2
            tail -n 30 "${SHARED}/logs/${f}.log" 2>/dev/null | sed 's/^/    /' >&2
        done
        break
    fi
    sleep 5
done
echo "healthy=${healthy} after $(( $(date +%s) - t0 ))s" | tee "${SHARED}/boot_time.txt"
touch "${GPU_WATCH_STOP}"
for i in "${!NODES[@]}"; do ssh -n "${NODES[$i]}" "pkill -f ${SHARED}/gpu_watch.sh" 2>/dev/null; done
kill -TERM "${TAIL_H_PID}" "${TAIL_M_PID}" 2>/dev/null

if [ "${healthy}" -ne 1 ]; then
    # Only the raising path prints this string (Inkling -10-07d).
    _bad=$(grep -lF 'is less than desired GPU memory utilization' "${SHARED}"/logs/node*.log 2>/dev/null)
    if [ -n "${_bad}" ]; then
        for f in ${_bad}; do k=${f##*node}; echo "DIRTY_NODE:${SHORT[${k%.log}]}" >&2; done
        exit 10
    fi
    exit 1
fi

# After boot, so the launcher env never reaches the servers.
source "${SHARED}/common_env.sh"

# --- Gates -------------------------------------------------------------------
{
    echo "=== gates ==="
    echo "kv_pool_tokens  $(grep -ho 'GPU KV cache size: [0-9,]* tokens' "${SHARED}"/logs/node*.log | tr -dc '0-9\n' | sort -n | head -1)"
    echo "kv_gib per stage (one value per rank):"
    for k in $(seq 0 $((NNODES - 1))); do
        echo "  stage ${k} ${SHORT[$k]}  $(grep -o 'Available KV cache memory: [0-9.]* GiB' "${SHARED}/logs/node${k}.log" | awk '{print $5}' | sort -n | tr '\n' ' ')"
    done
    echo "nccl logs non-empty $(find "${SHARED}/logs/nccl" -name 'nccl.*.log' -size +0c | wc -l)  with plugin $(grep -l 'Loaded net plugin' "${SHARED}"/logs/nccl/nccl.*.log 2>/dev/null | wc -l)  (want equal)"
    grep -ho 'Using network [A-Za-z]*' "${SHARED}"/logs/nccl/nccl.*.log 2>/dev/null | sort | uniq -c | sed 's/^/  /'
    echo "scheduler lines (want no 'disabled' / 'does not yet support'):"
    grep -hoE 'Async scheduling is disabled.*|Model Runner V2 does not yet support.*|Hidden layers were unevenly partitioned.*' \
        "${SHARED}/logs/node0.log" | sort -u | sed 's/^/  /'
    echo "coherence:"
    curl --noproxy '*' -s "http://${HEAD_IP}:${PORT}/v1/completions" -H 'Content-Type: application/json' \
        -d "{\"model\":\"${MODEL}\",\"prompt\":\"The capital of France is\",\"max_tokens\":32,\"temperature\":0}" \
        > "${SHARED}/coherence.json"
    python -c 'import json,sys; print("  " + repr(json.load(open(sys.argv[1]))["choices"][0]["text"]))' \
        "${SHARED}/coherence.json" 2>/dev/null || sed 's/^/  /' "${SHARED}/coherence.json"
} 2>&1 | tee "${SHARED}/gates.txt"

# --- Parity: each concurrency twice; pass 2 vs pass 1 is the same-vs-same floor.
mkdir -p "${SHARED}/parity"
RC=0
for c in ${PARITY_CONCURRENCIES}; do
    for p in 1 2; do
        python "${HERE}/parity_client.py" "http://${HEAD_IP}:${PORT}" "${MODEL}" \
            "${SHARED}/parity/c${c}_p${p}.jsonl" "${c}" "${p}" || RC=20
    done
    python "${HERE}/parity_compare.py" "${SHARED}/parity/c${c}_p1.jsonl" "${SHARED}/parity/c${c}_p2.jsonl" \
        | tee "${SHARED}/parity/same_c${c}.txt"
    if ! curl --noproxy '*' -s -o /dev/null -w '%{http_code}' "http://${HEAD_IP}:${PORT}/health" 2>/dev/null | grep -q 200; then
        echo "server unhealthy after c=${c}." >&2
        RC=30; break
    fi
done
exit "${RC}"
