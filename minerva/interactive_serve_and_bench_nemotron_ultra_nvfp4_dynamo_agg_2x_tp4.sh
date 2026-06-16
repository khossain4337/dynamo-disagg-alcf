#!/bin/bash
set -euo pipefail

# ==========================================
# Nemotron Ultra NVFP4 — Dynamo AGG Serve + Benchmark
# 2× TP4 workers behind a single Dynamo frontend
# Dynamo's built-in router handles dispatch (least-loaded)
#
# PURPOSE: Compare Dynamo smart routing vs pure vLLM round-robin.
# Uses same 8 GPUs as pure vLLM AGG and DISAGG 1P:1D scripts.
#
# GPU assignment:
#   GPUs 0,1,2,3 → worker 0 (DYN_SYSTEM_PORT=8081)
#   GPUs 4,5,6,7 → worker 1 (DYN_SYSTEM_PORT=8082)
#   Dynamo frontend → HTTP port 8000
#
# Naming: DYNAMO_AGG_2xTP4__ prefix for easy result comparison
#
# [FACT] In Dynamo AGG mode, workers register with the frontend via etcd.
# The frontend's router dispatches requests — no manual proxy needed.
# Router mode: least-loaded (routes to worker with fewer in-flight requests)
#
# [FACT] --disaggregation-mode is NOT set — workers run in standard AGG mode.
# No KV transfer config needed — no NIXL transfers in AGG mode.
#
# [FACT] --dyn-reasoning-parser is the Dynamo-specific flag (not --reasoning-parser).
# No --reasoning-parser-plugin needed — dynamo.vllm has built-in parser support.
#
# [FACT] Do NOT set --kv-cache-dtype — vLLM auto-selects fp8_e4m3 for NVFP4.
# [FACT] Do NOT set VLLM_USE_FLASHINFER_MOE_FP8 — Ultra uses FP4 MoE kernel.
# [FACT] WORKER_READY_TIMEOUT=900 — startup ~306s on B200 with warm cache.
# [FACT] Sequential worker startup required — worker 0 first, then worker 1,
# then frontend. Frontend discovers workers via etcd registration.
# ==========================================

tstamp() { date +"%Y-%m-%d-%H%M%S"; }

# ==========================================
# Environment
# ==========================================
export HF_TOKEN=$(cat ~/.hf_token)
export HF_HOME="/home/hossainm/software/model-weights"
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128
export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"

# [FACT] Required for deterministic Dynamo routing
export PYTHONHASHSEED=0

# [FACT] Required for Mamba-2 KV cache layout consistency
export VLLM_SSM_CONV_STATE_LAYOUT=DS

source /home/hossainm/miniforge3/bin/activate
eval "$(conda shell.bash hook)"
conda activate /home/hossainm/software/envs/conda_envs/dynamo_vllm_1.3.0_dev1

# ==========================================
# Configuration — edit here
# ==========================================
MODEL="nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4"

# GPU assignment — 2 independent TP=4 AGG workers
GPUS_WORKER_0="0,1,2,3"
GPUS_WORKER_1="4,5,6,7"

# Ports
HTTP_PORT=8000
WORKER_SYSTEM_PORT_0=8081
WORKER_SYSTEM_PORT_1=8082

# Router mode — controls how Dynamo frontend dispatches requests
# Options: round-robin, random, power-of-two, kv, direct, least-loaded, device-aware-weighted
# [FACT] Default is round-robin — set explicitly for clarity and reproducibility
# Change to least-loaded for second run
ROUTER_MODE="least-loaded"

# Serving config — must match other Ultra scripts for fair comparison
MAX_MODEL_LEN=32768
MAX_NUM_SEQS=32

# Worker ready timeout (seconds)
# [FACT] Warm startup ~306s on B200. 900s needed for sequential
# startup — worker 0 then worker 1, each up to ~306s.
WORKER_READY_TIMEOUT=900

# Results
RESULTS_DIR="$HOME/software/vllm_efforts/benchmark_results"

# Benchmark scenarios
# Format: "input_len,output_len,num_prompts,request_rate,max_concurrency"
SCENARIOS=(
    "16384,256,500,10,32"    # prefill-heavy — matches pure vLLM AGG scenario
    #"16384,2048,200,5,32"   # decode-heavy  — uncomment when ready
    #"32768,256,500,10,32"   # prefill-heavy, longer context
    #"32768,2048,200,5,32"   # decode-heavy,  longer context
)

# ==========================================
# Setup
# ==========================================
mkdir -p "$RESULTS_DIR"
RUN_ID="$(tstamp)"
FRONTEND_LOG="$RESULTS_DIR/DYNAMO_AGG_2xTP4__frontend_${RUN_ID}.log"
WORKER_LOG_0="$RESULTS_DIR/DYNAMO_AGG_2xTP4__worker0_${RUN_ID}.log"
WORKER_LOG_1="$RESULTS_DIR/DYNAMO_AGG_2xTP4__worker1_${RUN_ID}.log"
ETCD_LOG="$RESULTS_DIR/DYNAMO_AGG_2xTP4__etcd_${RUN_ID}.log"
NATS_LOG="$RESULTS_DIR/DYNAMO_AGG_2xTP4__nats_${RUN_ID}.log"
touch "$FRONTEND_LOG" "$WORKER_LOG_0" "$WORKER_LOG_1" "$ETCD_LOG" "$NATS_LOG"

echo "=========================================="
echo "Dynamo AGG 2×TP4 — Nemotron Ultra NVFP4"
echo "Dynamo version: 1.3.0-dev.1"
echo "=========================================="
echo "Run ID:      $RUN_ID"
echo "Model:       $MODEL"
echo "GPUs 0-3:    worker 0 → DYN_SYSTEM_PORT $WORKER_SYSTEM_PORT_0"
echo "GPUs 4-7:    worker 1 → DYN_SYSTEM_PORT $WORKER_SYSTEM_PORT_1"
echo "HTTP port:   $HTTP_PORT (Dynamo frontend)"
echo "Router:      $ROUTER_MODE"
echo "Topology:    2× AGG workers, TP=4 each, Dynamo frontend"
echo "Purpose:     Compare smart routing vs pure vLLM round-robin"
echo "Results:     $RESULTS_DIR"
echo "Time:        $(tstamp)"
echo "=========================================="

# ==========================================
# Sanity checks — kill leftovers, check GPUs
# ==========================================
echo "Checking for leftover GPU processes..."
LEFTOVER=$(pgrep -f "VLLM\|vllm serve\|dynamo.vllm\|dynamo.frontend" 2>/dev/null || true)
if [ -n "$LEFTOVER" ]; then
    echo "WARNING: Killing leftover processes: $LEFTOVER"
    kill -9 $LEFTOVER 2>/dev/null || true
    sleep 5
    echo "Leftover processes cleared."
else
    echo "No leftover processes found."
fi

echo "Checking GPU memory..."
nvidia-smi --query-gpu=index,memory.free,memory.total \
    --format=csv,noheader,nounits | while IFS=, read -r idx free total; do
    free=$(echo $free | tr -d ' ')
    total=$(echo $total | tr -d ' ')
    pct=$(( free * 100 / total ))
    if [ "$pct" -lt 90 ]; then
        echo "ERROR: GPU $idx only ${free}/${total} MiB free (${pct}%) — not safe to launch" >&2
        exit 1
    fi
    echo "  GPU $idx: ${free}/${total} MiB free (${pct}%) ✓"
done
echo "GPU memory check: OK"

# ==========================================
# Cleanup on exit
# ==========================================
cleanup() {
    echo ""
    echo "Cleaning up all Dynamo processes..."
    kill 0 2>/dev/null || true
    pkill -9 -f "VLLM\|dynamo.vllm\|dynamo.frontend" 2>/dev/null || true
    sleep 2
    echo "Done!"
}
trap cleanup EXIT

# ==========================================
# wait_for_worker <url> <name> <timeout>
# ==========================================
wait_for_worker() {
    local url="$1"
    local name="$2"
    local timeout="${3:-$WORKER_READY_TIMEOUT}"
    local start=$SECONDS
    echo "Waiting for $name at $url (timeout: ${timeout}s)..."
    while (( SECONDS - start < timeout )); do
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 "$url") || HTTP_CODE="000"
        if [ "$HTTP_CODE" = "200" ]; then
            echo "$name ready after $(( SECONDS - start ))s"
            return 0
        fi
        sleep 5
        echo "  ...$name still starting ($(( SECONDS - start ))s elapsed) [HTTP: $HTTP_CODE]"
    done
    echo "ERROR: $name not ready after ${timeout}s" >&2
    return 1
}

# ==========================================
# Start etcd
# ==========================================
echo "Starting etcd..."
etcd --data-dir /tmp/etcd_dynamo_${RUN_ID} \
    --listen-client-urls http://127.0.0.1:2379 \
    --advertise-client-urls http://127.0.0.1:2379 \
    --listen-peer-urls http://127.0.0.1:2380 \
    > "$ETCD_LOG" 2>&1 &
ETCD_PID=$!
echo "etcd PID: $ETCD_PID"
sleep 2

# ==========================================
# Start NATS
# ==========================================
echo "Starting NATS..."
nats-server -p 4222 \
    > "$NATS_LOG" 2>&1 &
NATS_PID=$!
echo "NATS PID: $NATS_PID"
sleep 2

# ==========================================
# Start Dynamo frontend
# [FACT] Frontend starts before workers — it waits for workers
# to register via etcd before serving requests.
# ==========================================
echo "Starting Dynamo frontend on port $HTTP_PORT..."
DYN_HTTP_PORT=$HTTP_PORT \
python -m dynamo.frontend \
    --router-mode "$ROUTER_MODE" \
    > "$FRONTEND_LOG" 2>&1 &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"
sleep 5

# ==========================================
# Start worker 0 (GPUs 0-3)
# [FACT] numactl pins to NUMA node 0 — GPUs 0-3 affinity
# [FACT] No --disaggregation-mode — pure AGG worker
# [FACT] No --kv-transfer-config — no NIXL transfers in AGG mode
# ==========================================
echo "Starting worker 0 on GPUs $GPUS_WORKER_0 (DYN_SYSTEM_PORT $WORKER_SYSTEM_PORT_0)..."
VLLM_ENGINE_STARTUP_TIMEOUT=600 \
CUDA_VISIBLE_DEVICES=$GPUS_WORKER_0 \
DYN_SYSTEM_PORT=$WORKER_SYSTEM_PORT_0 \
numactl --cpunodebind=0 --membind=0 \
python3 -m dynamo.vllm \
    --model "$MODEL" \
    --served-model-name "$MODEL" \
    --tensor-parallel-size 4 \
    --max-model-len $MAX_MODEL_LEN \
    --max-num-seqs $MAX_NUM_SEQS \
    --no-disable-hybrid-kv-cache-manager \
    --trust-remote-code \
    --dyn-reasoning-parser nemotron_v3 \
    > "$WORKER_LOG_0" 2>&1 &
WORKER_PID_0=$!
echo "Worker 0 PID: $WORKER_PID_0"

wait_for_worker "http://127.0.0.1:${WORKER_SYSTEM_PORT_0}/health" "worker 0"

# ==========================================
# Start worker 1 (GPUs 4-7)
# [FACT] numactl pins to NUMA node 1 — GPUs 4-7 affinity
# [FACT] Sequential after worker 0 — Dynamo AGG workers must
# register sequentially to avoid etcd race conditions
# [GUESS] Race condition risk is low but not zero — monitor
# frontend logs if worker 1 fails to register
# ==========================================
echo "Starting worker 1 on GPUs $GPUS_WORKER_1 (DYN_SYSTEM_PORT $WORKER_SYSTEM_PORT_1)..."
VLLM_ENGINE_STARTUP_TIMEOUT=600 \
CUDA_VISIBLE_DEVICES=$GPUS_WORKER_1 \
DYN_SYSTEM_PORT=$WORKER_SYSTEM_PORT_1 \
numactl --cpunodebind=1 --membind=1 \
python3 -m dynamo.vllm \
    --model "$MODEL" \
    --served-model-name "$MODEL" \
    --tensor-parallel-size 4 \
    --max-model-len $MAX_MODEL_LEN \
    --max-num-seqs $MAX_NUM_SEQS \
    --no-disable-hybrid-kv-cache-manager \
    --trust-remote-code \
    --dyn-reasoning-parser nemotron_v3 \
    > "$WORKER_LOG_1" 2>&1 &
WORKER_PID_1=$!
echo "Worker 1 PID: $WORKER_PID_1"

wait_for_worker "http://127.0.0.1:${WORKER_SYSTEM_PORT_1}/health" "worker 1"

# ==========================================
# Wait for Dynamo frontend to be ready
# Both workers must register before frontend serves requests
# ==========================================
wait_for_worker "http://127.0.0.1:${HTTP_PORT}/health" "Dynamo frontend"

# [FACT] Extra sleep — frontend may return 200 before both
# workers are fully registered in etcd
echo "Allowing workers to fully register in etcd (10s)..."
sleep 10
echo "All workers registered. Ready to benchmark!"
echo ""

# ==========================================
# Tail logs briefly
# ==========================================
echo "--- Frontend log (last 3 lines) ---"
tail -3 "$FRONTEND_LOG"
echo "--- Worker 0 log (last 3 lines) ---"
tail -3 "$WORKER_LOG_0"
echo "--- Worker 1 log (last 3 lines) ---"
tail -3 "$WORKER_LOG_1"
echo ""

# ==========================================
# Run benchmarks
# ==========================================
for SCENARIO in "${SCENARIOS[@]}"; do
    INPUT_LEN=$(echo $SCENARIO    | cut -d',' -f1)
    OUTPUT_LEN=$(echo $SCENARIO   | cut -d',' -f2)
    NUM_PROMPTS=$(echo $SCENARIO  | cut -d',' -f3)
    REQUEST_RATE=$(echo $SCENARIO | cut -d',' -f4)
    MAX_CONCURRENCY=$(echo $SCENARIO | cut -d',' -f5)

    echo "=========================================="
    echo "Benchmarking scenario (Dynamo AGG 2×TP4):"
    echo "  Input len:       $INPUT_LEN tokens"
    echo "  Output len:      $OUTPUT_LEN tokens"
    echo "  Num prompts:     $NUM_PROMPTS"
    echo "  Request rate:    $REQUEST_RATE req/s"
    echo "  Max concurrency: $MAX_CONCURRENCY"
    echo "  Topology:        2× AGG workers, TP=4 each, Dynamo $ROUTER_MODE"
    echo "  Time:            $(tstamp)"
    echo "=========================================="

    RESULT_FILE="$RESULTS_DIR/DYNAMO_AGG_2xTP4__${ROUTER_MODE}__nemotron_ultra_nvfp4__in${INPUT_LEN}_out${OUTPUT_LEN}__prm${NUM_PROMPTS}__rate${REQUEST_RATE}__seq${MAX_CONCURRENCY}__${RUN_ID}.json"

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

# ==========================================
# Summary
# ==========================================
echo "=========================================="
echo "All Dynamo AGG 2×TP4 benchmarks complete!"
echo "Results:"
ls -lh "$RESULTS_DIR"/DYNAMO_AGG_2xTP4__${ROUTER_MODE}__* 2>/dev/null || echo "No DYNAMO_AGG_2xTP4 result files found"
echo "=========================================="

# cleanup() handles all process teardown via EXIT trap
