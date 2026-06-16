#!/bin/bash
set -euo pipefail

# ==========================================
# Nemotron Ultra NVFP4 — Dynamo DISAGG 1P:1D Serve + Benchmark
# Single node, 1 prefill worker + 1 decode worker, TP=4 each
#
# PURPOSE: Main DISAGG comparison against AGG 2×TP4 baseline.
# Separates prefill and decode onto dedicated GPU sets.
# KV cache transferred via NIXL over NVLink (intra-node).
#
# GPU assignment:
#   GPUs 0,1,2,3 → decode worker  (DYN_SYSTEM_PORT=8081)
#   GPUs 4,5,6,7 → prefill worker (DYN_SYSTEM_PORT=8082)
#   Dynamo frontend → HTTP port 8000
#
# Startup order: decode → prefill → frontend → sleep 10
# [FACT] Decode starts first — prefill workers load decode
# metadata from etcd on first request.
#
# Naming: DYNAMO_DISAGG_1P1D__ prefix for easy result comparison
#
# [FACT] --dyn-reasoning-parser nemotron_v3 — NOT ultra_v3.
#   ultra_v3 is not a built-in parser in dynamo.vllm v1.3.0-dev.1.
# [FACT] --enable-auto-tool-choice and --tool-call-parser are NOT
#   supported in dynamo.vllm — omit from all Dynamo scripts.
# [FACT] Do NOT set --kv-cache-dtype — vLLM auto-selects fp8_e4m3.
# [FACT] Do NOT set VLLM_USE_FLASHINFER_MOE_FP8 — Ultra uses FP4 MoE kernel.
# [FACT] VLLM_SSM_CONV_STATE_LAYOUT=DS required for Mamba-2 KV transfer via NIXL.
# [FACT] nixl/worker.py SSM assert patch must be applied before running.
#   Patch replaces assert num_local_blocks == num_remote_blocks with
#   trim-to-minimum for SSM groups. Apply via vllm_nixl_assert_patch.sh.
#   WITHOUT the patch, the first request will crash with AssertionError.
# [FACT] WORKER_READY_TIMEOUT=900 — sequential startup, each worker ~306s.
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

# [FACT] Required for Mamba-2 KV transfer layout via NIXL
export VLLM_SSM_CONV_STATE_LAYOUT=DS

source /home/hossainm/miniforge3/bin/activate
eval "$(conda shell.bash hook)"
conda activate /home/hossainm/software/envs/conda_envs/dynamo_vllm_1.3.0_dev1

# ==========================================
# Configuration — edit here
# ==========================================
MODEL="nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4"

# GPU assignment — decode first, then prefill
# [FACT] Decode on NUMA node 0 (GPUs 0-3), prefill on NUMA node 1 (GPUs 4-7)
# This matches GPU→NUMA affinity confirmed by nvidia-smi topo -m
GPUS_DECODE="0,1,2,3"
GPUS_PREFILL="4,5,6,7"

# Ports
HTTP_PORT=8000
DECODE_SYSTEM_PORT=8081
PREFILL_SYSTEM_PORT=8082

# NIXL side channel port — unique per prefill worker
# [FACT] Required for prefill→decode KV transfer coordination
NIXL_SIDE_CHANNEL_PORT=20097

# KV events port — unique per prefill worker
KV_EVENTS_PORT=20081

# Serving config — must match AGG scripts for fair comparison
MAX_MODEL_LEN=32768
MAX_NUM_SEQS=32

# Worker ready timeout (seconds)
# [FACT] Sequential startup: decode (~306s) then prefill (~306s) = ~612s total.
# 900s per worker gives safe headroom for each.
WORKER_READY_TIMEOUT=900

# Results
RESULTS_DIR="$HOME/software/vllm_efforts/benchmark_results"

# Benchmark scenarios
# Format: "input_len,output_len,num_prompts,request_rate,max_concurrency"
SCENARIOS=(
    "16384,256,500,10,32"    # prefill-heavy — matches AGG scenario exactly
    #"16384,2048,200,5,32"   # decode-heavy  — uncomment when ready
    #"32768,256,500,10,32"   # prefill-heavy, longer context
    #"32768,2048,200,5,32"   # decode-heavy,  longer context
)

# ==========================================
# Setup
# ==========================================
mkdir -p "$RESULTS_DIR"
RUN_ID="$(tstamp)"
FRONTEND_LOG="$RESULTS_DIR/DYNAMO_DISAGG_1P1D__frontend_${RUN_ID}.log"
DECODE_LOG="$RESULTS_DIR/DYNAMO_DISAGG_1P1D__decode_${RUN_ID}.log"
PREFILL_LOG="$RESULTS_DIR/DYNAMO_DISAGG_1P1D__prefill_${RUN_ID}.log"
ETCD_LOG="$RESULTS_DIR/DYNAMO_DISAGG_1P1D__etcd_${RUN_ID}.log"
NATS_LOG="$RESULTS_DIR/DYNAMO_DISAGG_1P1D__nats_${RUN_ID}.log"
touch "$FRONTEND_LOG" "$DECODE_LOG" "$PREFILL_LOG" "$ETCD_LOG" "$NATS_LOG"

echo "=========================================="
echo "Dynamo DISAGG 1P:1D — Nemotron Ultra NVFP4"
echo "Dynamo version: 1.3.0-dev.1"
echo "=========================================="
echo "Run ID:        $RUN_ID"
echo "Model:         $MODEL"
echo "Decode GPUs:   $GPUS_DECODE (DYN_SYSTEM_PORT $DECODE_SYSTEM_PORT, NUMA node 0)"
echo "Prefill GPUs:  $GPUS_PREFILL (DYN_SYSTEM_PORT $PREFILL_SYSTEM_PORT, NUMA node 1)"
echo "HTTP port:     $HTTP_PORT (Dynamo frontend)"
echo "KV transfer:   NIXL over NVLink (intra-node)"
echo "Topology:      1 prefill worker + 1 decode worker, TP=4 each"
echo "Purpose:       Main DISAGG comparison vs AGG 2×TP4"
echo "Results:       $RESULTS_DIR"
echo "Time:          $(tstamp)"
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
    echo "Cleaning up all Dynamo DISAGG processes..."
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
# ==========================================
echo "Starting Dynamo frontend on port $HTTP_PORT..."
DYN_HTTP_PORT=$HTTP_PORT \
python -m dynamo.frontend \
    > "$FRONTEND_LOG" 2>&1 &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"
sleep 5

# ==========================================
# Start decode worker (GPUs 0-3)
# [FACT] Decode starts first — prefill loads decode metadata
# from etcd on first request.
# [FACT] numactl pins to NUMA node 0 — GPUs 0-3 affinity confirmed
# by nvidia-smi topo -m (CPU affinity: 0-55,112-167)
# [FACT] No VLLM_NIXL_SIDE_CHANNEL_PORT on decode worker
# [FACT] No --kv-events-config on decode worker
# ==========================================
echo "Starting decode worker on GPUs $GPUS_DECODE (DYN_SYSTEM_PORT $DECODE_SYSTEM_PORT)..."
VLLM_ENGINE_STARTUP_TIMEOUT=600 \
CUDA_VISIBLE_DEVICES=$GPUS_DECODE \
DYN_SYSTEM_PORT=$DECODE_SYSTEM_PORT \
numactl --cpunodebind=0 --membind=0 \
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
    > "$DECODE_LOG" 2>&1 &
DECODE_PID=$!
echo "Decode worker PID: $DECODE_PID"

wait_for_worker "http://127.0.0.1:${DECODE_SYSTEM_PORT}/health" "decode worker"

# ==========================================
# Start prefill worker (GPUs 4-7)
# [FACT] numactl pins to NUMA node 1 — GPUs 4-7 affinity confirmed
# by nvidia-smi topo -m (CPU affinity: 56-111,168-223)
# [FACT] VLLM_NIXL_SIDE_CHANNEL_PORT required on prefill worker
# [FACT] --kv-events-config required on prefill worker only
# [FACT] SSM assert patch must be applied — without it, first
# request will crash with AssertionError in nixl/worker.py
# ==========================================
echo "Starting prefill worker on GPUs $GPUS_PREFILL (DYN_SYSTEM_PORT $PREFILL_SYSTEM_PORT)..."
VLLM_ENGINE_STARTUP_TIMEOUT=600 \
CUDA_VISIBLE_DEVICES=$GPUS_PREFILL \
DYN_SYSTEM_PORT=$PREFILL_SYSTEM_PORT \
VLLM_NIXL_SIDE_CHANNEL_PORT=$NIXL_SIDE_CHANNEL_PORT \
numactl --cpunodebind=1 --membind=1 \
python3 -m dynamo.vllm \
    --model "$MODEL" \
    --served-model-name "$MODEL" \
    --disaggregation-mode prefill \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
    --kv-events-config "{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:${KV_EVENTS_PORT}\",\"enable_kv_cache_events\":true}" \
    --tensor-parallel-size 4 \
    --max-model-len $MAX_MODEL_LEN \
    --max-num-seqs $MAX_NUM_SEQS \
    --no-disable-hybrid-kv-cache-manager \
    --trust-remote-code \
    --dyn-reasoning-parser nemotron_v3 \
    > "$PREFILL_LOG" 2>&1 &
PREFILL_PID=$!
echo "Prefill worker PID: $PREFILL_PID"

wait_for_worker "http://127.0.0.1:${PREFILL_SYSTEM_PORT}/health" "prefill worker"

# ==========================================
# Wait for Dynamo frontend
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
echo "--- Decode log (last 3 lines) ---"
tail -3 "$DECODE_LOG"
echo "--- Prefill log (last 3 lines) ---"
tail -3 "$PREFILL_LOG"
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
    echo "Benchmarking scenario (Dynamo DISAGG 1P:1D):"
    echo "  Input len:       $INPUT_LEN tokens"
    echo "  Output len:      $OUTPUT_LEN tokens"
    echo "  Num prompts:     $NUM_PROMPTS"
    echo "  Request rate:    $REQUEST_RATE req/s"
    echo "  Max concurrency: $MAX_CONCURRENCY"
    echo "  Topology:        1P:1D, TP=4 each, NIXL KV transfer"
    echo "  Time:            $(tstamp)"
    echo "=========================================="

    RESULT_FILE="$RESULTS_DIR/DYNAMO_DISAGG_1P1D__nemotron_ultra_nvfp4__in${INPUT_LEN}_out${OUTPUT_LEN}__prm${NUM_PROMPTS}__rate${REQUEST_RATE}__seq${MAX_CONCURRENCY}__${RUN_ID}.json"

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
echo "All Dynamo DISAGG 1P:1D benchmarks complete!"
echo "Results:"
ls -lh "$RESULTS_DIR"/DYNAMO_DISAGG_1P1D__* 2>/dev/null || echo "No DYNAMO_DISAGG_1P1D result files found"
echo "=========================================="

# cleanup() handles all process teardown via EXIT trap
