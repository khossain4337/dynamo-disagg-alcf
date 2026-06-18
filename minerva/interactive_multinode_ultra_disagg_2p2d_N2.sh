#!/bin/bash
set -euo pipefail

# ==========================================
# Nemotron Ultra NVFP4 — Dynamo DISAGG 2P:2D Multi-Node
# Run this SAME script on BOTH nodes, in separate SSH sessions.
# The script detects which node it is on and does the right thing.
#
# USAGE:
#   Terminal 1 (minerva-dgx-01): bash interactive_multinode_ultra_disagg_2p2d_N2.sh
#   Terminal 2 (minerva-dgx-02): bash interactive_multinode_ultra_disagg_2p2d_N2.sh
#   Start node1 first. Start node2 once node1 prints "NODE1 WORKERS READY".
#
# Topology (2P:2D):
#   minerva-dgx-01 (NODE1):
#     etcd        → 0.0.0.0:2379, advertised at 10.125.13.11:2379
#     NATS        → 0.0.0.0:4222
#     Frontend    → HTTP port 8000
#     Decode-1    → GPUs 4,5,6,7  DYN_SYSTEM_PORT=8081  NUMA node 1
#     Prefill-1   → GPUs 0,1,2,3  DYN_SYSTEM_PORT=8082  NUMA node 0  NIXL_SC=20097
#
#   minerva-dgx-02 (NODE2):
#     Decode-2    → GPUs 4,5,6,7  DYN_SYSTEM_PORT=8084  NUMA node 1
#     Prefill-2   → GPUs 0,1,2,3  DYN_SYSTEM_PORT=8083  NUMA node 0  NIXL_SC=20098
#
# Difference from 2P:1D:
#   Node2 now runs TWO workers instead of one:
#     - Decode-2  on GPUs 4-7 (starts first on node2, before prefill-2)
#     - Prefill-2 on GPUs 0-3 (starts after decode-2 is ready)
#   Node1 waits for BOTH node2 workers before proceeding to benchmark.
#
# KV transfer paths:
#   Prefill-1 (node1 GPUs 0-3) → Decode-1 (node1 GPUs 4-7): NVLink ~600 GB/s
#   Prefill-1 (node1 GPUs 0-3) → Decode-2 (node2 GPUs 4-7): IB ~25 GB/s
#   Prefill-2 (node2 GPUs 0-3) → Decode-1 (node1 GPUs 4-7): IB ~25 GB/s
#   Prefill-2 (node2 GPUs 0-3) → Decode-2 (node2 GPUs 4-7): NVLink ~600 GB/s
# [GUESS] Dynamo router may not be KV-topology-aware — may send P1→D2 and P2→D1
#         (both IB paths) as often as the fast NVLink paths. Watch routing logs.
#
# Motivation: 2P:1D showed decode saturation at OSL=2048 (peak concurrent=54,
# effective 0.89 req/s at rate=5). 2P:2D doubles decode capacity — predicted
# to recover throughput and TTFT at decode-heavy workloads.
#
# [FACT] All 2P:1D facts apply here. Additional 2P:2D specifics:
# [FACT] Decode-2 on node2 starts FIRST (before prefill-2) — same rule as node1.
# [FACT] DYN_SYSTEM_PORT=8084 for decode-2 — unique, not used by any other worker.
# [FACT] No VLLM_NIXL_SIDE_CHANNEL_PORT on decode workers (prefill only).
# [FACT] No --kv-events-config on decode workers (prefill only).
# [FACT] Node1 polls BOTH node2:8084 (decode-2) AND node2:8083 (prefill-2)
#        before proceeding. Both must be healthy before benchmark starts.
# [FACT] WORKER_READY_TIMEOUT=900 per worker. Node2 has 2 sequential workers:
#        decode-2 (~326s) + prefill-2 (~315s) = ~640s total on node2.
#        Start node2 as early as possible during node1 prefill-1 load.
# ==========================================

tstamp() { date +"%Y-%m-%d-%H%M%S"; }

# ==========================================
# Cluster topology — confirmed values
# ==========================================
NODE1_HOST="minerva-dgx-01"
NODE2_HOST="minerva-dgx-02"
NODE1_IP="10.125.13.11"
NODE2_IP="10.125.13.12"

# ==========================================
# Detect which node we are on
# ==========================================
THIS_HOST=$(hostname -s)

if [ "$THIS_HOST" = "$NODE1_HOST" ]; then
    THIS_ROLE="node1"
elif [ "$THIS_HOST" = "$NODE2_HOST" ]; then
    THIS_ROLE="node2"
else
    echo "ERROR: Unknown hostname '$THIS_HOST'." >&2
    echo "Expected '$NODE1_HOST' or '$NODE2_HOST'." >&2
    exit 1
fi

echo "=========================================="
echo "Detected role: $THIS_ROLE ($THIS_HOST)"
echo "=========================================="

# ==========================================
# Environment — same on both nodes
# ==========================================
export HF_TOKEN=$(cat ~/.hf_token)
export HF_HOME="/home/hossainm/software/model-weights"
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128
export no_proxy="127.0.0.1,localhost,${NODE1_IP},${NODE2_IP},${NODE1_HOST},${NODE2_HOST}"
export NO_PROXY="127.0.0.1,localhost,${NODE1_IP},${NODE2_IP},${NODE1_HOST},${NODE2_HOST}"

export PYTHONHASHSEED=0
export VLLM_SSM_CONV_STATE_LAYOUT=DS
export ETCD_ENDPOINTS="http://${NODE1_IP}:2379"
export NATS_SERVER="nats://${NODE1_IP}:4222"

source /home/hossainm/miniforge3/bin/activate
eval "$(conda shell.bash hook)"
conda activate /home/hossainm/software/envs/conda_envs/dynamo_vllm_1.3.0_dev1

# ==========================================
# Configuration — same on both nodes
# ==========================================
MODEL="nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4"
MAX_MODEL_LEN=32768
MAX_NUM_SEQS=32
WORKER_READY_TIMEOUT=900

# Ports
HTTP_PORT=8000
DECODE_SYSTEM_PORT_1=8081    # node1 decode
PREFILL_SYSTEM_PORT_1=8082   # node1 prefill
PREFILL_SYSTEM_PORT_2=8083   # node2 prefill
DECODE_SYSTEM_PORT_2=8084    # node2 decode — NEW vs 2P:1D

# NIXL side channel — prefill workers only, unique across all nodes
NIXL_SIDE_CHANNEL_PORT_1=20097
NIXL_SIDE_CHANNEL_PORT_2=20098

# KV events ports — prefill workers only, unique across all nodes
KV_EVENTS_PORT_1=20081
KV_EVENTS_PORT_2=20082

# GPU assignments
# [FACT] NUMA node 0: GPUs 0-3 | NUMA node 1: GPUs 4-7 (same on both nodes)
GPUS_DECODE_1="4,5,6,7"    # node1 — NUMA node 1
GPUS_PREFILL_1="0,1,2,3"   # node1 — NUMA node 0
GPUS_PREFILL_2="0,1,2,3"   # node2 — NUMA node 0
GPUS_DECODE_2="4,5,6,7"    # node2 — NUMA node 1 — NEW vs 2P:1D

RESULTS_DIR="$HOME/software/vllm_efforts/benchmark_results"

# Benchmark scenarios
# [FACT] 2P:1D decode-heavy saturated at rate=5 (0.89 effective req/s, peak=54).
# Start at rate=5 to compare directly. Then rate=3 for prefill-heavy comparison.
SCENARIOS=(
    "16384,2048,200,5,32"    # decode-heavy — primary comparison vs 2P:1D
    "16384,256,500,10,32"    # prefill-heavy — compare vs 2P:1D rate=10
    #"16384,256,500,3,32"    # prefill-heavy unsaturated — uncomment if needed
)

# ==========================================
# Shared helpers
# ==========================================

wait_for_worker() {
    local url="$1"
    local name="$2"
    local timeout="${3:-$WORKER_READY_TIMEOUT}"
    local start=$SECONDS
    echo "Waiting for $name at $url (timeout: ${timeout}s)..."
    while (( SECONDS - start < timeout )); do
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
            --max-time 2 "$url") || HTTP_CODE="000"
        if [ "$HTTP_CODE" = "200" ]; then
            echo "$name ready after $(( SECONDS - start ))s ✓"
            return 0
        fi
        sleep 5
        echo "  ...$name still starting ($(( SECONDS - start ))s elapsed) [HTTP: $HTTP_CODE]"
    done
    echo "ERROR: $name not ready after ${timeout}s" >&2
    return 1
}

check_ssm_patch() {
    local patch_file="$HOME/software/envs/conda_envs/dynamo_vllm_1.3.0_dev1/lib/python3.12/site-packages/vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py"
    if grep -q "num_blocks = min" "$patch_file" 2>/dev/null; then
        echo "  SSM patch: CONFIRMED ✓"
    else
        echo "ERROR: SSM patch NOT found in nixl/worker.py" >&2
        echo "Run vllm_nixl_assert_patch.sh on node1 (NFS propagates to node2)." >&2
        exit 1
    fi
}

check_gpu_memory() {
    nvidia-smi --query-gpu=index,memory.free,memory.total \
        --format=csv,noheader,nounits | while IFS=, read -r idx free total; do
        free=$(echo $free | tr -d ' ')
        total=$(echo $total | tr -d ' ')
        pct=$(( free * 100 / total ))
        if [ "$pct" -lt 90 ]; then
            echo "ERROR: GPU $idx only ${free}/${total} MiB free (${pct}%) — not safe" >&2
            exit 1
        fi
        echo "  GPU $idx: ${free}/${total} MiB free (${pct}%) ✓"
    done
}

kill_leftovers() {
    local leftover
    leftover=$(pgrep -f "vllm|dynamo.vllm|dynamo.frontend" 2>/dev/null || true)
    if [ -n "$leftover" ]; then
        echo "WARNING: Killing leftover processes: $leftover"
        kill -9 $leftover 2>/dev/null || true
        sleep 5
        echo "Leftover processes cleared."
    else
        echo "  No leftover processes found."
    fi
}

# ==========================================
# ==========================================
# NODE 1 BRANCH
# ==========================================
# ==========================================
run_node1() {
    local RUN_ID
    RUN_ID="$(tstamp)"
    local LOG_PREFIX="ULTRA_DISAGG_2P2D_2NODE"
    mkdir -p "$RESULTS_DIR"

    local FRONTEND_LOG="$RESULTS_DIR/${LOG_PREFIX}__frontend_${RUN_ID}.log"
    local DECODE1_LOG="$RESULTS_DIR/${LOG_PREFIX}__decode1_${RUN_ID}.log"
    local PREFILL1_LOG="$RESULTS_DIR/${LOG_PREFIX}__prefill1_${RUN_ID}.log"
    local ETCD_LOG="$RESULTS_DIR/${LOG_PREFIX}__etcd_${RUN_ID}.log"
    local NATS_LOG="$RESULTS_DIR/${LOG_PREFIX}__nats_${RUN_ID}.log"
    touch "$FRONTEND_LOG" "$DECODE1_LOG" "$PREFILL1_LOG" "$ETCD_LOG" "$NATS_LOG"

    echo "=========================================="
    echo "Dynamo DISAGG 2P:2D Multi-Node — NODE1"
    echo "Dynamo version: 1.3.0-dev.1"
    echo "=========================================="
    echo "Run ID:        $RUN_ID"
    echo "Model:         $MODEL"
    echo "etcd:          $ETCD_ENDPOINTS  (this node)"
    echo "NATS:          $NATS_SERVER  (this node)"
    echo "Frontend:      HTTP port $HTTP_PORT"
    echo "Decode-1:      GPUs $GPUS_DECODE_1  (DYN_SYSTEM_PORT $DECODE_SYSTEM_PORT_1,  NUMA 1)"
    echo "Prefill-1:     GPUs $GPUS_PREFILL_1 (DYN_SYSTEM_PORT $PREFILL_SYSTEM_PORT_1, NUMA 0)"
    echo "Decode-2:      GPUs 4-7 on $NODE2_HOST (DYN_SYSTEM_PORT $DECODE_SYSTEM_PORT_2, NUMA 1)"
    echo "Prefill-2:     GPUs 0-3 on $NODE2_HOST (DYN_SYSTEM_PORT $PREFILL_SYSTEM_PORT_2, NUMA 0)"
    echo "Results:       $RESULTS_DIR"
    echo "Time:          $(tstamp)"
    echo "=========================================="

    echo "Sanity checks..."
    kill_leftovers
    echo "GPU memory:"
    check_gpu_memory
    check_ssm_patch

    cleanup_node1() {
        echo ""
        echo "Cleaning up node1 processes..."
        kill 0 2>/dev/null || true
        pkill -9 -f "dynamo.vllm|dynamo.frontend" 2>/dev/null || true
        sleep 2
        echo "node1 cleanup done."
        echo "NOTE: Kill node2 workers manually (Ctrl+C in its terminal)."
    }
    trap cleanup_node1 EXIT

    # --- etcd ---
    echo "Starting etcd (0.0.0.0:2379, advertising ${NODE1_IP})..."
    etcd \
        --data-dir /tmp/etcd_dynamo_${RUN_ID} \
        --listen-client-urls "http://0.0.0.0:2379" \
        --advertise-client-urls "http://${NODE1_IP}:2379" \
        --listen-peer-urls "http://0.0.0.0:2380" \
        --initial-advertise-peer-urls "http://${NODE1_IP}:2380" \
        --initial-cluster "default=http://${NODE1_IP}:2380" \
        > "$ETCD_LOG" 2>&1 &
    ETCD_PID=$!
    echo "etcd PID: $ETCD_PID"
    sleep 3

    local etcd_ok=0
    for i in {1..10}; do
        if curl -sf "http://${NODE1_IP}:2379/health" > /dev/null 2>&1; then
            echo "etcd healthy on ${NODE1_IP}:2379 ✓"
            etcd_ok=1; break
        fi
        sleep 2
        echo "  ...etcd not yet ready (attempt $i/10)"
    done
    [ "$etcd_ok" -eq 1 ] || { echo "ERROR: etcd failed on ${NODE1_IP}:2379" >&2; exit 1; }

    # --- NATS ---
    echo "Starting NATS (0.0.0.0:4222)..."
    nats-server -p 4222 -a 0.0.0.0 > "$NATS_LOG" 2>&1 &
    NATS_PID=$!
    echo "NATS PID: $NATS_PID"
    sleep 2

    # --- Frontend ---
    echo "Starting Dynamo frontend on port $HTTP_PORT..."
    DYN_HTTP_PORT=$HTTP_PORT \
    python -m dynamo.frontend \
        > "$FRONTEND_LOG" 2>&1 &
    FRONTEND_PID=$!
    echo "Frontend PID: $FRONTEND_PID"
    sleep 5

    # --- Decode-1 (GPUs 4-7, NUMA 1) ---
    echo "Starting decode worker 1 on GPUs $GPUS_DECODE_1 (DYN_SYSTEM_PORT $DECODE_SYSTEM_PORT_1)..."
    VLLM_ENGINE_STARTUP_TIMEOUT=600 \
    CUDA_VISIBLE_DEVICES=$GPUS_DECODE_1 \
    DYN_SYSTEM_PORT=$DECODE_SYSTEM_PORT_1 \
    ETCD_ENDPOINTS="http://${NODE1_IP}:2379" \
    NATS_SERVER="nats://${NODE1_IP}:4222" \
    numactl --cpunodebind=1 --membind=1 \
    python3 -m dynamo.vllm \
        --model "$MODEL" \
        --served-model-name "$MODEL" \
        --disaggregation-mode decode \
        --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
        --tensor-parallel-size 4 \
        --max-model-len $MAX_MODEL_LEN \
        --max-num-seqs $MAX_NUM_SEQS \
        --no-disable-hybrid-kv-cache-manager \
        --trust-remote-code \
        --dyn-reasoning-parser nemotron_v3 \
        > "$DECODE1_LOG" 2>&1 &
    DECODE1_PID=$!
    echo "Decode-1 PID: $DECODE1_PID"
    wait_for_worker "http://127.0.0.1:${DECODE_SYSTEM_PORT_1}/health" "decode worker 1"

    # --- Prefill-1 (GPUs 0-3, NUMA 0) ---
    echo "Starting prefill worker 1 on GPUs $GPUS_PREFILL_1 (DYN_SYSTEM_PORT $PREFILL_SYSTEM_PORT_1)..."
    VLLM_ENGINE_STARTUP_TIMEOUT=600 \
    CUDA_VISIBLE_DEVICES=$GPUS_PREFILL_1 \
    DYN_SYSTEM_PORT=$PREFILL_SYSTEM_PORT_1 \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$NIXL_SIDE_CHANNEL_PORT_1 \
    ETCD_ENDPOINTS="http://${NODE1_IP}:2379" \
    NATS_SERVER="nats://${NODE1_IP}:4222" \
    numactl --cpunodebind=0 --membind=0 \
    python3 -m dynamo.vllm \
        --model "$MODEL" \
        --served-model-name "$MODEL" \
        --disaggregation-mode prefill \
        --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
        --kv-events-config "{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:${KV_EVENTS_PORT_1}\",\"enable_kv_cache_events\":true}" \
        --tensor-parallel-size 4 \
        --max-model-len $MAX_MODEL_LEN \
        --max-num-seqs $MAX_NUM_SEQS \
        --no-disable-hybrid-kv-cache-manager \
        --trust-remote-code \
        --dyn-reasoning-parser nemotron_v3 \
        > "$PREFILL1_LOG" 2>&1 &
    PREFILL1_PID=$!
    echo "Prefill-1 PID: $PREFILL1_PID"
    wait_for_worker "http://127.0.0.1:${PREFILL_SYSTEM_PORT_1}/health" "prefill worker 1"

    # --- Signal node2 ---
    echo ""
    echo "=========================================="
    echo "NODE1 WORKERS READY."
    echo "NOW open a new terminal and run:"
    echo "  ssh $NODE2_HOST"
    echo "  bash $(realpath $0)"
    echo "Waiting for node2 decode-2 AND prefill-2..."
    echo "(Node2 runs decode-2 first, then prefill-2 — ~640s total)"
    echo "=========================================="
    echo ""

    # --- Wait for decode-2 on node2 ---
    local d2_start=$SECONDS
    local d2_timeout=1200
    local d2_ready=0
    while (( SECONDS - d2_start < d2_timeout )); do
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
            --max-time 3 "http://${NODE2_IP}:${DECODE_SYSTEM_PORT_2}/health") || HTTP_CODE="000"
        if [ "$HTTP_CODE" = "200" ]; then
            echo "Decode-2 (node2) ready after $(( SECONDS - d2_start ))s ✓"
            d2_ready=1; break
        fi
        sleep 10
        echo "  ...decode-2 still starting ($(( SECONDS - d2_start ))s elapsed) [HTTP: $HTTP_CODE]"
    done
    [ "$d2_ready" -eq 1 ] || {
        echo "ERROR: decode-2 on $NODE2_HOST not ready after ${d2_timeout}s" >&2; exit 1
    }

    # --- Wait for prefill-2 on node2 ---
    local p2_start=$SECONDS
    local p2_timeout=1200
    local p2_ready=0
    while (( SECONDS - p2_start < p2_timeout )); do
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
            --max-time 3 "http://${NODE2_IP}:${PREFILL_SYSTEM_PORT_2}/health") || HTTP_CODE="000"
        if [ "$HTTP_CODE" = "200" ]; then
            echo "Prefill-2 (node2) ready after $(( SECONDS - p2_start ))s ✓"
            p2_ready=1; break
        fi
        sleep 10
        echo "  ...prefill-2 still starting ($(( SECONDS - p2_start ))s elapsed) [HTTP: $HTTP_CODE]"
    done
    [ "$p2_ready" -eq 1 ] || {
        echo "ERROR: prefill-2 on $NODE2_HOST not ready after ${p2_timeout}s" >&2; exit 1
    }

    # --- Wait for frontend ---
    wait_for_worker "http://127.0.0.1:${HTTP_PORT}/health" "Dynamo frontend"

    echo "Allowing all 4 workers to fully register in etcd (30s)..."
    sleep 30
    echo "All workers registered. Ready to benchmark!"
    echo ""

    echo "--- Frontend log (last 5 lines) ---"
    tail -5 "$FRONTEND_LOG"
    echo "--- Decode-1 log (last 3 lines) ---"
    tail -3 "$DECODE1_LOG"
    echo "--- Prefill-1 log (last 3 lines) ---"
    tail -3 "$PREFILL1_LOG"
    echo ""

    # --- Benchmarks ---
    for SCENARIO in "${SCENARIOS[@]}"; do
        local INPUT_LEN OUTPUT_LEN NUM_PROMPTS REQUEST_RATE MAX_CONCURRENCY
        INPUT_LEN=$(echo $SCENARIO    | cut -d',' -f1)
        OUTPUT_LEN=$(echo $SCENARIO   | cut -d',' -f2)
        NUM_PROMPTS=$(echo $SCENARIO  | cut -d',' -f3)
        REQUEST_RATE=$(echo $SCENARIO | cut -d',' -f4)
        MAX_CONCURRENCY=$(echo $SCENARIO | cut -d',' -f5)

        echo "=========================================="
        echo "Benchmarking (ULTRA DISAGG 2P:2D Multi-Node):"
        echo "  Input len:       $INPUT_LEN tokens"
        echo "  Output len:      $OUTPUT_LEN tokens"
        echo "  Num prompts:     $NUM_PROMPTS"
        echo "  Request rate:    $REQUEST_RATE req/s"
        echo "  Max concurrency: $MAX_CONCURRENCY"
        echo "  Topology:        2P:2D, TP=4, inter-node NIXL/IB"
        echo "  Time:            $(tstamp)"
        echo "=========================================="

        local RESULT_FILE
        RESULT_FILE="$RESULTS_DIR/ULTRA_DISAGG_2P2D_2NODE__in${INPUT_LEN}_out${OUTPUT_LEN}__prm${NUM_PROMPTS}__rate${REQUEST_RATE}__seq${MAX_CONCURRENCY}__${RUN_ID}.json"

        vllm bench serve \
            --backend openai-chat \
            --model "$MODEL" \
            --dataset-name random \
            --random-input-len $INPUT_LEN \
            --random-output-len $OUTPUT_LEN \
            --endpoint /v1/chat/completions \
            --host 127.0.0.1 \
            --port $HTTP_PORT \
            --num-prompts $NUM_PROMPTS \
            --request-rate $REQUEST_RATE \
            --max-concurrency $MAX_CONCURRENCY \
            --save-result \
            --result-filename "$RESULT_FILE"

        echo "Done: input=$INPUT_LEN output=$OUTPUT_LEN"
        echo "Results: $RESULT_FILE"
        echo ""
    done

    echo "=========================================="
    echo "All ULTRA DISAGG 2P:2D multi-node benchmarks complete!"
    echo "Results:"
    ls -lh "$RESULTS_DIR"/ULTRA_DISAGG_2P2D_2NODE__* 2>/dev/null || \
        echo "No result files found"
    echo "NOTE: Kill node2 workers with Ctrl+C in its terminal."
    echo "=========================================="
}

# ==========================================
# ==========================================
# NODE 2 BRANCH
# ==========================================
# ==========================================
run_node2() {
    local RUN_ID
    RUN_ID="$(tstamp)"
    mkdir -p "$RESULTS_DIR"

    local DECODE2_LOG="$RESULTS_DIR/ULTRA_DISAGG_2P2D_2NODE__decode2_${RUN_ID}.log"
    local PREFILL2_LOG="$RESULTS_DIR/ULTRA_DISAGG_2P2D_2NODE__prefill2_${RUN_ID}.log"
    touch "$DECODE2_LOG" "$PREFILL2_LOG"

    echo "=========================================="
    echo "Dynamo DISAGG 2P:2D Multi-Node — NODE2"
    echo "Dynamo version: 1.3.0-dev.1"
    echo "=========================================="
    echo "Run ID:         $RUN_ID"
    echo "Model:          $MODEL"
    echo "etcd:           $ETCD_ENDPOINTS  (on node1)"
    echo "NATS:           $NATS_SERVER  (on node1)"
    echo "Decode-2:       GPUs $GPUS_DECODE_2  (DYN_SYSTEM_PORT $DECODE_SYSTEM_PORT_2, NUMA 1)"
    echo "Prefill-2:      GPUs $GPUS_PREFILL_2 (DYN_SYSTEM_PORT $PREFILL_SYSTEM_PORT_2, NUMA 0)"
    echo "NIXL SC port:   $NIXL_SIDE_CHANNEL_PORT_2 (prefill-2 only)"
    echo "KV events port: $KV_EVENTS_PORT_2 (prefill-2 only)"
    echo "Time:           $(tstamp)"
    echo "=========================================="

    echo "Sanity checks..."
    kill_leftovers
    echo "GPU memory:"
    check_gpu_memory
    check_ssm_patch

    echo "Verifying etcd reachable at ${NODE1_IP}:2379..."
    local etcd_ok=0
    for i in {1..12}; do
        if curl -sf "http://${NODE1_IP}:2379/health" > /dev/null 2>&1; then
            echo "  etcd reachable ✓"; etcd_ok=1; break
        fi
        sleep 5
        echo "  ...etcd not reachable yet (attempt $i/12) — is node1 script running?"
    done
    [ "$etcd_ok" -eq 1 ] || {
        echo "ERROR: Cannot reach etcd at ${NODE1_IP}:2379" >&2
        echo "Make sure node1 script is running and past the etcd startup step." >&2
        exit 1
    }

    cleanup_node2() {
        echo ""
        echo "Cleaning up node2 processes..."
        kill 0 2>/dev/null || true
        pkill -9 -f "dynamo.vllm" 2>/dev/null || true
        sleep 2
        echo "node2 cleanup done."
    }
    trap cleanup_node2 EXIT

    # --- Decode-2 (GPUs 4-7, NUMA 1) ---
    # [FACT] Decode starts first on node2 too — same rule as node1.
    # [FACT] No VLLM_NIXL_SIDE_CHANNEL_PORT on decode workers.
    # [FACT] No --kv-events-config on decode workers.
    echo "Starting decode worker 2 on GPUs $GPUS_DECODE_2 (DYN_SYSTEM_PORT $DECODE_SYSTEM_PORT_2)..."
    VLLM_ENGINE_STARTUP_TIMEOUT=600 \
    CUDA_VISIBLE_DEVICES=$GPUS_DECODE_2 \
    DYN_SYSTEM_PORT=$DECODE_SYSTEM_PORT_2 \
    ETCD_ENDPOINTS="http://${NODE1_IP}:2379" \
    NATS_SERVER="nats://${NODE1_IP}:4222" \
    numactl --cpunodebind=1 --membind=1 \
    python3 -m dynamo.vllm \
        --model "$MODEL" \
        --served-model-name "$MODEL" \
        --disaggregation-mode decode \
        --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
        --tensor-parallel-size 4 \
        --max-model-len $MAX_MODEL_LEN \
        --max-num-seqs $MAX_NUM_SEQS \
        --no-disable-hybrid-kv-cache-manager \
        --trust-remote-code \
        --dyn-reasoning-parser nemotron_v3 \
        > "$DECODE2_LOG" 2>&1 &
    DECODE2_PID=$!
    echo "Decode-2 PID: $DECODE2_PID"
    wait_for_worker "http://127.0.0.1:${DECODE_SYSTEM_PORT_2}/health" "decode worker 2"

    # --- Prefill-2 (GPUs 0-3, NUMA 0) ---
    echo "Starting prefill worker 2 on GPUs $GPUS_PREFILL_2 (DYN_SYSTEM_PORT $PREFILL_SYSTEM_PORT_2)..."
    VLLM_ENGINE_STARTUP_TIMEOUT=600 \
    CUDA_VISIBLE_DEVICES=$GPUS_PREFILL_2 \
    DYN_SYSTEM_PORT=$PREFILL_SYSTEM_PORT_2 \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$NIXL_SIDE_CHANNEL_PORT_2 \
    ETCD_ENDPOINTS="http://${NODE1_IP}:2379" \
    NATS_SERVER="nats://${NODE1_IP}:4222" \
    numactl --cpunodebind=0 --membind=0 \
    python3 -m dynamo.vllm \
        --model "$MODEL" \
        --served-model-name "$MODEL" \
        --disaggregation-mode prefill \
        --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
        --kv-events-config "{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:${KV_EVENTS_PORT_2}\",\"enable_kv_cache_events\":true}" \
        --tensor-parallel-size 4 \
        --max-model-len $MAX_MODEL_LEN \
        --max-num-seqs $MAX_NUM_SEQS \
        --no-disable-hybrid-kv-cache-manager \
        --trust-remote-code \
        --dyn-reasoning-parser nemotron_v3 \
        > "$PREFILL2_LOG" 2>&1 &
    PREFILL2_PID=$!
    echo "Prefill-2 PID: $PREFILL2_PID"
    wait_for_worker "http://127.0.0.1:${PREFILL_SYSTEM_PORT_2}/health" "prefill worker 2"

    echo ""
    echo "=========================================="
    echo "NODE2 WORKERS READY (decode-2 + prefill-2)."
    echo "Node1 benchmark will start automatically."
    echo "Ctrl+C here when node1 reports benchmarks complete."
    echo "Decode-2 log: $DECODE2_LOG"
    echo "Prefill-2 log: $PREFILL2_LOG"
    echo "=========================================="
    echo ""

    # Tail prefill-2 log for visibility — decode-2 log available separately
    tail -f "$PREFILL2_LOG"
}

# ==========================================
# Dispatch
# ==========================================
if [ "$THIS_ROLE" = "node1" ]; then
    run_node1
else
    run_node2
fi
