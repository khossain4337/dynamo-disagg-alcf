#!/bin/bash
set -uo pipefail
# Muse-Spark greedy parity, P:D: TP4 x PP8 per role, NixlPushConnector, proxy on its own node,
# parity_client.py twice per concurrency, teardown. Needs the disaggpp env (decode-side PP patch).
# Ported from run_muse_spark_parity_tp4pp8_N8.sh and ../../run_inkling_sweep_1p1d_N4.sh.
# Exit 0 parity ran | 1 config or boot | 10 dirty node | 20 failed requests | 30 server died
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TN="$(cd "${HERE}/../.." && pwd)"

PARITY_CONCURRENCIES=${PARITY_CONCURRENCIES:-1 32}
EP=${EP:-1}
EXCLUDE_NODES=${EXCLUDE_NODES:-}
# Verbatim copy of tests/v1/kv_connector/nixl_integration/toy_proxy_server.py at the pin.
PROXY_SCRIPT=${PROXY_SCRIPT:-${HERE}/toy_proxy_server_9e3db48873.py}
MODEL=meta-models/Muse-Spark-Open-816B-A42B
NNODES=8 TP=4 PP=8
MAX_MODEL_LEN=1048576
API_SERVER_COUNT=4
P_PORT=8100 D_PORT=8200 PROXY_PORT=8000
P_MASTER_PORT=29550 D_MASTER_PORT=29560
SIDE_PORT_P=5600 SIDE_PORT_D=5700
HEALTH_S=7500
GPU_DIRTY_MIB=4096
RUNS_ROOT=/vast/draco/tara/projects/Tara_Deployment/software/testing/RUNS
STAMP=$(date +%Y%m%d_%H%M%S)
SHARED=${RUNS_ROOT}/muse_spark_parity_pd_tp${TP}pp${PP}_ep${EP}_mml${MAX_MODEL_LEN}${RUN_TAG:+_${RUN_TAG}}_${STAMP}
EP_FLAG=; [ "${EP}" = 1 ] && EP_FLAG=--enable-expert-parallel
KV_EXTRA='"backends":["LIBFABRIC"],"kv_lease_duration":30'
KV_P="{\"kv_connector\":\"NixlPushConnector\",\"kv_role\":\"kv_producer\",\"kv_connector_extra_config\":{${KV_EXTRA}}}"
KV_D="{\"kv_connector\":\"NixlPushConnector\",\"kv_role\":\"kv_consumer\",\"kv_connector_extra_config\":{${KV_EXTRA}}}"

ls -d /vast/draco/tara/projects/Tara_Deployment/software/model-weights/hub/models--meta-models--Muse-Spark-Open-816B-A42B/snapshots/0fb02478* >/dev/null 2>&1 \
    || { echo "FATAL: Muse-Spark snapshot 0fb02478 not in the cache." >&2; exit 1; }
[ -f "${PROXY_SCRIPT}" ] || { echo "FATAL: no ${PROXY_SCRIPT}." >&2; exit 1; }

# --- Nodes: P = 0..7 (node k = stage k), D = 8..15, proxy = 16. This client never serves.
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
D0=${NNODES} PX=$(( 2 * NNODES ))

hsn() { ssh -n "$1" "ip -4 -o addr show hsn0" 2>/dev/null | awk '{print $4}' | cut -d/ -f1; }
P_IP=$(hsn "${NODES[0]}"); D_IP=$(hsn "${NODES[D0]}"); PROXY_IP=$(hsn "${NODES[PX]}")
CLIENT_IP=$(ip -4 -o addr show hsn0 2>/dev/null | awk '{print $4}' | cut -d/ -f1)
[ -n "${P_IP}" ] && [ -n "${D_IP}" ] && [ -n "${PROXY_IP}" ] && [ -n "${CLIENT_IP}" ] \
    || { echo "FATAL: missing an hsn0 address." >&2; exit 1; }
echo "client ${CLIENT_HOST} ${CLIENT_IP}  proxy ${SHORT[PX]} ${PROXY_IP}"
echo "P ${P_IP} stages ${SHORT[*]:0:NNODES}"
echo "D ${D_IP} stages ${SHORT[*]:D0:NNODES}"

# NCCL will not create its directory, and falls back to stdout silently.
mkdir -p "${SHARED}/logs/nccl" || { echo "FATAL: cannot create ${SHARED}" >&2; exit 1; }

# --- Preflight ---------------------------------------------------------------
_dirty=()
for i in "${!ENGINE[@]}"; do
    ssh -n "${NODES[$i]}" "GPU_DIRTY_MIB=${GPU_DIRTY_MIB} bash ${TN}/gpu_cleanup.sh report" 2>&1 | sed "s/^/  [${SHORT[$i]}] /"
    [ "${PIPESTATUS[0]}" -eq 0 ] || _dirty+=("${SHORT[$i]}")
done
ssh -n "${NODES[0]}"  "fuser ${P_PORT}/tcp"     >/dev/null 2>&1 && _dirty+=("${SHORT[0]}")
ssh -n "${NODES[D0]}" "fuser ${D_PORT}/tcp"     >/dev/null 2>&1 && _dirty+=("${SHORT[D0]}")
ssh -n "${NODES[PX]}" "fuser ${PROXY_PORT}/tcp" >/dev/null 2>&1 && _dirty+=("${SHORT[PX]}")
if [ "${#_dirty[@]}" -gt 0 ]; then
    printf 'DIRTY_NODE:%s\n' "${_dirty[@]}" | sort -u >&2
    exit 10
fi

NO_PROXY_LIST="localhost,127.0.0.1,${P_IP},${D_IP},${PROXY_IP},${CLIENT_IP},${NODES[0]},${SHORT[0]},${NODES[D0]},${SHORT[D0]},${NODES[PX]},${SHORT[PX]},${CLIENT_HOST}"
bash "${HERE}/../emit_common_env_disaggpp_nixl_1.5.2.sh" "${SHARED}/common_env.sh" "${NO_PROXY_LIST}" \
    "${SHARED}/logs/nccl/nccl.%h.%p.log" || exit 1
for n in "${NODES[2*NNODES-1]}" "${NODES[PX]}"; do
    ssh -n "${n}" "test -f ${SHARED}/common_env.sh" || { echo "FATAL: ${SHARED} not visible on ${n%%.*}." >&2; exit 1; }
done

# --- Role script: one file, 16 nodes; role and node rank from this host's place in the list.
cat > "${SHARED}/launch_role.sh" <<EOF
#!/bin/bash
NODES=(${SHORT[*]:0:2*NNODES})
ME=\$(hostname -s); ME=\${ME%%.*}
RANK=
for i in "\${!NODES[@]}"; do [ "\${NODES[\$i]}" = "\${ME}" ] && RANK=\$i; done
[ -n "\${RANK}" ] || { echo "Rank landed on unexpected host \${ME}." >&2; exit 1; }
if [ "\${RANK}" -lt ${NNODES} ]; then
    ROLE=p NR=\${RANK} HEAD=${P_IP} PORT=${P_PORT} MPORT=${P_MASTER_PORT} SIDE=${SIDE_PORT_P} KV='${KV_P}'
else
    ROLE=d NR=\$(( RANK - ${NNODES} )) HEAD=${D_IP} PORT=${D_PORT} MPORT=${D_MASTER_PORT} SIDE=${SIDE_PORT_D} KV='${KV_D}'
fi
exec > ${SHARED}/logs/\${ROLE}\${NR}.log 2>&1
source ${SHARED}/common_env.sh
# The ZMQ control queues advertise this address; unset, vLLM picks the management NIC.
export VLLM_HOST_IP=\$(ip -4 -o addr show hsn0 | awk '{print \$4}' | cut -d/ -f1)
[ -n "\${VLLM_HOST_IP}" ] || { echo "FATAL: no hsn0 address on \${ME}."; exit 1; }
env | sort > ${SHARED}/logs/env_\${ROLE}\${NR}.txt
echo "=== host=\${ME} role=\${ROLE} node_rank=\${NR} VLLM_HOST_IP=\${VLLM_HOST_IP} SLINGSHOT_VNIS=\${SLINGSHOT_VNIS:-<UNSET>} ==="
# P loads after D: two 1.5 TiB reads off /vast in sequence, not at once. Still one mpiexec (one VNI).
if [ "\${ROLE}" = p ]; then
    t0=\$(date +%s)
    until curl --noproxy '*' -s -o /dev/null -w '%{http_code}' http://${D_IP}:${D_PORT}/health 2>/dev/null | grep -q 200; do
        [ \$(( \$(date +%s) - t0 )) -lt ${HEALTH_S} ] || { echo "D never healthy."; exit 1; }
        sleep 5
    done
    echo "=== D up after \$(( \$(date +%s) - t0 ))s ==="
fi
# The handshake listener runs in the scheduler, on the role head only, and serves every (PP, TP) rank.
export VLLM_NIXL_SIDE_CHANNEL_HOST=\${HEAD} VLLM_NIXL_SIDE_CHANNEL_PORT=\${SIDE} NIXL_LOG_LEVEL=INFO
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
    --enable-logging-iteration-details --cudagraph-metrics \\
    --kv-transfer-config "\${KV}"
EOF
chmod +x "${SHARED}/launch_role.sh"

# toy_proxy_server.py's health path is /healthcheck; its --host default is 127.0.0.1.
cat > "${SHARED}/launch_proxy.sh" <<EOF
#!/bin/bash
source ${SHARED}/common_env.sh
export OPENAI_API_KEY=muse-pd-dummy-key
exec python3 ${PROXY_SCRIPT} --host 0.0.0.0 --port ${PROXY_PORT} \\
    --prefiller-host ${P_IP} --prefiller-port ${P_PORT} --decoder-host ${D_IP} --decoder-port ${D_PORT}
EOF
chmod +x "${SHARED}/launch_proxy.sh"

{
    echo "stamp            ${STAMP}"
    echo "model            ${MODEL}"
    echo "workload         greedy parity, P:D  concurrencies ${PARITY_CONCURRENCIES}"
    echo "client           ${CLIENT_HOST} ${CLIENT_IP}"
    echo "P                ${P_IP} port ${P_PORT} side ${SIDE_PORT_P}  stages ${SHORT[*]:0:NNODES}"
    echo "D                ${D_IP} port ${D_PORT} side ${SIDE_PORT_D}  stages ${SHORT[*]:D0:NNODES}"
    echo "proxy            ${SHORT[PX]} ${PROXY_IP} port ${PROXY_PORT}  ${PROXY_SCRIPT}"
    echo "excluded         ${EXCLUDE_NODES:-none}"
    echo "parallelism      TP ${TP} x PP ${PP} per role, EP ${EP}"
    echo "kv               ${KV_P}"
    echo "git              $(git -C "${TN}" rev-parse --short HEAD 2>/dev/null || echo '(none)')"
} > "${SHARED}/run_config.txt"
cat "${SHARED}/run_config.txt"

GPU_WATCH_STOP="${SHARED}/gpu_watch_stop"
cleanup() {
    touch "${GPU_WATCH_STOP}" 2>/dev/null
    kill -TERM ${MPIEXEC_PID:-} ${TAIL_P_PID:-} ${TAIL_D_PID:-} ${TAIL_M_PID:-} ${PROXY_SSH_PID:-} 2>/dev/null
    ssh -n "${NODES[PX]}" "pkill -f '[t]oy_proxy_server.py'; fuser -k ${PROXY_PORT}/tcp" 2>/dev/null
    for i in "${!ENGINE[@]}"; do
        ssh -n "${NODES[$i]}" "bash ${TN}/gpu_cleanup.sh kill ${P_PORT} ${D_PORT} ${P_MASTER_PORT} ${D_MASTER_PORT}" 2>&1 | sed "s/^/  [${SHORT[$i]}] /" &
    done
    wait
    echo "RUN_DIR:${SHARED}"
}
# EXIT is the only teardown path; INT/TERM just exit into it.
trap cleanup EXIT
trap 'exit 130' INT TERM

for k in $(seq 0 $((NNODES - 1))); do touch "${SHARED}/logs/p${k}.log" "${SHARED}/logs/d${k}.log"; done
touch "${SHARED}/logs/mpiexec.log" "${SHARED}/logs/proxy.log"

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

# One mpiexec for both roles: PALS gives each launch its own VNI, and two VNIs cannot reach each other.
mpiexec -n "$(( 2 * NNODES ))" -ppn 1 --hosts "$(IFS=,; echo "${ENGINE[*]}")" --cpu-bind none \
    bash "${SHARED}/launch_role.sh" > "${SHARED}/logs/mpiexec.log" 2>&1 &
MPIEXEC_PID=$!
tail -n +1 -f "${SHARED}/logs/d0.log"      > >(sed -u 's/^/[D0]  /') & TAIL_D_PID=$!; disown "${TAIL_D_PID}"
tail -n +1 -f "${SHARED}/logs/p0.log"      > >(sed -u 's/^/[P0]  /') & TAIL_P_PID=$!; disown "${TAIL_P_PID}"
tail -n +1 -f "${SHARED}/logs/mpiexec.log" > >(sed -u 's/^/[MPI] /') & TAIL_M_PID=$!; disown "${TAIL_M_PID}"

up() { curl --noproxy '*' -s -o /dev/null -w '%{http_code}' "http://$1$2" 2>/dev/null | grep -q 200; }
wait_up() {
    local t0; t0=$(date +%s)
    while [ $(( $(date +%s) - t0 )) -lt "${HEALTH_S}" ]; do
        up "$1" "$2" && { echo "$3 healthy after $(( $(date +%s) - t0 ))s" | tee -a "${SHARED}/boot_time.txt"; return 0; }
        if ! kill -0 "${MPIEXEC_PID}" 2>/dev/null; then
            echo "mpiexec exited while waiting for $3." >&2
            for f in mpiexec d0 d7 p0 p7; do
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
wait_up "${D_IP}:${D_PORT}" /health D && wait_up "${P_IP}:${P_PORT}" /health P && healthy=1
touch "${GPU_WATCH_STOP}"
for i in "${!ENGINE[@]}"; do ssh -n "${NODES[$i]}" "pkill -f ${SHARED}/gpu_watch.sh" 2>/dev/null; done
kill -TERM "${TAIL_P_PID}" "${TAIL_D_PID}" "${TAIL_M_PID}" 2>/dev/null

if [ "${healthy}" -ne 1 ]; then
    # Only the raising path prints this string (Inkling -10-07d).
    for f in $(grep -lF 'is less than desired GPU memory utilization' "${SHARED}"/logs/[pd][0-9].log 2>/dev/null); do
        b=$(basename "${f}" .log); k=${b:1}; [ "${b:0:1}" = d ] && k=$(( k + NNODES ))
        echo "DIRTY_NODE:${SHORT[$k]}" >&2; _bad=1
    done
    [ -n "${_bad:-}" ] && exit 10
    exit 1
fi

# After boot, so the launcher env never reaches the servers.
source "${SHARED}/common_env.sh"

ssh -n "${NODES[PX]}" "bash ${SHARED}/launch_proxy.sh" > "${SHARED}/logs/proxy.log" 2>&1 &
PROXY_SSH_PID=$!
wait_up "${PROXY_IP}:${PROXY_PORT}" /healthcheck proxy || { tail -n 30 "${SHARED}/logs/proxy.log" >&2; exit 1; }

# --- Gates -------------------------------------------------------------------
{
    echo "=== gates ==="
    for r in p d; do
        echo "kv_pool_tokens ${r}  $(grep -ho 'GPU KV cache size: [0-9,]* tokens' "${SHARED}"/logs/${r}[0-9].log | tr -dc '0-9\n' | sort -n | head -1)"
        for k in $(seq 0 $((NNODES - 1))); do
            echo "  ${r} stage ${k}  $(grep -o 'Available KV cache memory: [0-9.]* GiB' "${SHARED}/logs/${r}${k}.log" | awk '{print $5}' | sort -n | tr '\n' ' ')"
        done
    done
    echo "nccl logs non-empty $(find "${SHARED}/logs/nccl" -name 'nccl.*.log' -size +0c | wc -l)  with plugin $(grep -l 'Loaded net plugin' "${SHARED}"/logs/nccl/nccl.*.log 2>/dev/null | wc -l)  (want equal)"
    echo "scheduler lines (want no 'disabled' / 'does not yet support'):"
    grep -hoE 'Async scheduling is disabled.*|Model Runner V2 does not yet support.*|Hidden layers were unevenly partitioned.*' \
        "${SHARED}/logs/p0.log" "${SHARED}/logs/d0.log" | sort -u | sed 's/^/  /'
    echo "coherence through the proxy:"
    curl --noproxy '*' -s "http://${PROXY_IP}:${PROXY_PORT}/v1/completions" -H 'Content-Type: application/json' \
        -d "{\"model\":\"${MODEL}\",\"prompt\":\"The capital of France is\",\"max_tokens\":32,\"temperature\":0}" \
        > "${SHARED}/coherence.json"
    python -c 'import json,sys; print("  " + repr(json.load(open(sys.argv[1]))["choices"][0]["text"]))' \
        "${SHARED}/coherence.json" 2>/dev/null || sed 's/^/  /' "${SHARED}/coherence.json"
    # Per-node counts from DECISIONS -11-04b; head adds the scheduler. After the first request.
    echo "nixl gate (mismatches as got/want):"
    ok=1
    for f in "${SHARED}"/logs/[pd][0-9].log; do
        b=$(basename "${f}" .log)
        while IFS='|' read -r wh wo s; do
            w=${wo}; [ "${b:1}" = 0 ] && w=${wh}
            n=$(grep -cF -- "${s}" "${f}")
            [ "${n}" = "${w}" ] || { echo "  ${b}  ${n}/${w}  ${s}"; ok=0; }
        done <<'T'
5|4|Creating v1 connector with name: NixlPushConnector
1|0|Initializing NIXL Scheduler
4|4|Created 4 rails using provider=cxi
16|16|Using provider with FI_HMEM support
4|4|Registered memory on 1 rails, mem_type=1 device=1
4|4|NIXL compatibility check passed
0|0|drainPostQueue: post failed
0|0|NIXL_ERR_BACKEND
0|0|ret=-22
T
    done
    [ "${ok}" = 1 ] && echo "  all 16 nodes match"
} 2>&1 | tee "${SHARED}/gates.txt"

# --- Parity: each concurrency twice, through the proxy.
mkdir -p "${SHARED}/parity"
RC=0
for c in ${PARITY_CONCURRENCIES}; do
    for p in 1 2; do
        python "${HERE}/parity_client.py" "http://${PROXY_IP}:${PROXY_PORT}" "${MODEL}" \
            "${SHARED}/parity/c${c}_p${p}.jsonl" "${c}" "${p}" || RC=20
    done
    python "${HERE}/parity_compare.py" "${SHARED}/parity/c${c}_p1.jsonl" "${SHARED}/parity/c${c}_p2.jsonl" \
        | tee "${SHARED}/parity/same_c${c}.txt"
    if ! up "${P_IP}:${P_PORT}" /health || ! up "${D_IP}:${D_PORT}" /health; then
        echo "a role is unhealthy after c=${c}." >&2
        RC=30; break
    fi
done
exit "${RC}"
