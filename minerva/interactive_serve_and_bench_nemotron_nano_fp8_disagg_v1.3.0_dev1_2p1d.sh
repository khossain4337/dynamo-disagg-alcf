#!/bin/bash
set -euo pipefail

# ==========================================
# Nemotron Nano FP8 — Dynamo Disaggregated Serve + Benchmark
# Single node, 2P:1D topology
# GPU 0 = decode, GPU 1 = prefill 1, GPU 2 = prefill 2
# Designed for ISL >> OSL (prefill-heavy) workloads
# Compare directly against AGG and DISAGG 1P:1D scripts
#
# v1.3.0-dev.1 changes from v1.2.0:
#   - --no-enable-prefix-caching on all workers — workaround for
#     SSM block count mismatch in nixl/worker.py _apply_prefix_caching
#     assert num_local_blocks == num_remote_blocks fires with 2+ prefill
#     workers on hybrid Mamba/Attention models. Root cause: vLLM 0.22.0
#     NIXL connector does not handle SSM prefix cache blocks correctly
#     in multi-prefill topology. Disabling prefix caching sidesteps it.
#   - conda env: dynamo_vllm_1.3.0_dev1
#   - --reasoning-parser → --dyn-reasoning-parser (built-in, no plugin needed)
#   - --reasoning-parser-plugin removed (built-in parsers only in v1.3.0)
#   - nemotron_v3 parser is built-in — no custom .py file required
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

# [FACT] PYTHONHASHSEED=0 required for deterministic disaggregated routing
export PYTHONHASHSEED=0

# [FACT] Required for Nemotron hybrid (Mamba) KV transfer via NIXL
export VLLM_SSM_CONV_STATE_LAYOUT=DS

source /home/hossainm/miniforge3/bin/activate
eval "$(conda shell.bash hook)"
conda activate /home/hossainm/software/envs/conda_envs/dynamo_vllm_1.3.0_dev1

# ==========================================
# Configuration — edit here
# ==========================================
MODEL="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"

# GPU assignment — 1 decode + 2 prefill
DECODE_GPU=0
PREFILL_GPU_1=1
PREFILL_GPU_2=2

# Ports — Dynamo internal system ports
HTTP_PORT=8000
DECODE_SYSTEM_PORT=8081
PREFILL_SYSTEM_PORT_1=8082
PREFILL_SYSTEM_PORT_2=8083

# NIXL side channel ports (must be unique per prefill worker)
NIXL_SIDE_CHANNEL_PORT_1=20097
NIXL_SIDE_CHANNEL_PORT_2=20098

# KV events ports (must be unique per prefill worker)
KV_EVENTS_PORT_1=20081
KV_EVENTS_PORT_2=20082

# Serving config
MAX_MODEL_LEN=32768
MAX_NUM_SEQS=32

# Worker ready timeout (seconds)
# [FACT] First startup in fresh env requires FlashInfer JIT compilation of
# B200-specific (sm_100a) MoE kernels — can take 15-20 minutes cold.
# Cached in ~/.cache/flashinfer/ — subsequent startups are fast (~12s).
WORKER_READY_TIMEOUT=1800

# Results
RESULTS_DIR="$HOME/software/vllm_efforts/benchmark_results"

# Benchmark scenarios — ISL >> OSL (prefill-heavy)
# Format: "input_len,output_len,num_prompts,request_rate,max_concurrency"
SCENARIOS=(
    #"16384,256,500,10,16"
    "16384,256,500,10,32"
    #"8192,256,500,10,32"
    #"32768,256,200,5,32"
)

# ==========================================
# Setup
# ==========================================
mkdir -p "$RESULTS_DIR"
RUN_ID="$(tstamp)"
FRONTEND_LOG="$RESULTS_DIR/disagg_2p1d_frontend_${RUN_ID}.log"
DECODE_LOG="$RESULTS_DIR/disagg_2p1d_decode_${RUN_ID}.log"
PREFILL_LOG_1="$RESULTS_DIR/disagg_2p1d_prefill1_${RUN_ID}.log"
PREFILL_LOG_2="$RESULTS_DIR/disagg_2p1d_prefill2_${RUN_ID}.log"
touch "$FRONTEND_LOG" "$DECODE_LOG" "$PREFILL_LOG_1" "$PREFILL_LOG_2"

echo "=========================================="
echo "Dynamo Disaggregated Serving 2P:1D — Nemotron Nano FP8"
echo "Dynamo version: 1.3.0-dev.1"
echo "=========================================="
echo "Run ID:        $RUN_ID"
echo "Model:         $MODEL"
echo "Decode GPU:    $DECODE_GPU"
echo "Prefill GPU 1: $PREFILL_GPU_1"
echo "Prefill GPU 2: $PREFILL_GPU_2"
echo "HTTP Port:     $HTTP_PORT"
echo "Topology:      2 prefill + 1 decode"
echo "Workload:      ISL >> OSL (prefill-heavy)"
echo "Results:       $RESULTS_DIR"
echo "Time:          $(tstamp)"
echo "=========================================="

# ==========================================
# Cleanup on exit
# ==========================================
cleanup() {
    echo ""
    echo "Cleaning up all Dynamo processes..."
    kill 0 2>/dev/null || true
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
    > "$RESULTS_DIR/etcd_${RUN_ID}.log" 2>&1 &
ETCD_PID=$!
echo "etcd PID: $ETCD_PID"
sleep 2

# ==========================================
# Start NATS
# ==========================================
echo "Starting NATS..."
nats-server -p 4222 \
    > "$RESULTS_DIR/nats_${RUN_ID}.log" 2>&1 &
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
# Start decode worker (GPU 0)
# [FACT] Decode starts first — prefill workers load decode
# metadata from etcd on first request
# ==========================================
echo "Starting decode worker on GPU $DECODE_GPU (port $DECODE_SYSTEM_PORT)..."
VLLM_ENGINE_STARTUP_TIMEOUT=600 \
CUDA_VISIBLE_DEVICES=$DECODE_GPU \
DYN_SYSTEM_PORT=$DECODE_SYSTEM_PORT \
VLLM_USE_FLASHINFER_MOE_FP8=1 \
python3 -m dynamo.vllm \
    --model "$MODEL" \
    --served-model-name "$MODEL" \
    --disaggregation-mode decode \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
    --tensor-parallel-size 1 \
    --max-model-len $MAX_MODEL_LEN \
    --max-num-seqs $MAX_NUM_SEQS \
    --kv-cache-dtype fp8 \
    --no-disable-hybrid-kv-cache-manager \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --dyn-reasoning-parser nemotron_v3 \
    > "$DECODE_LOG" 2>&1 &
DECODE_PID=$!
echo "Decode worker PID: $DECODE_PID"

wait_for_worker "http://127.0.0.1:${DECODE_SYSTEM_PORT}/health" "decode worker"

# ==========================================
# Start prefill worker 1 (GPU 1)
# ==========================================
echo "Starting prefill worker 1 on GPU $PREFILL_GPU_1 (port $PREFILL_SYSTEM_PORT_1)..."
VLLM_ENGINE_STARTUP_TIMEOUT=600 \
CUDA_VISIBLE_DEVICES=$PREFILL_GPU_1 \
DYN_SYSTEM_PORT=$PREFILL_SYSTEM_PORT_1 \
VLLM_NIXL_SIDE_CHANNEL_PORT=$NIXL_SIDE_CHANNEL_PORT_1 \
VLLM_USE_FLASHINFER_MOE_FP8=1 \
python3 -m dynamo.vllm \
    --model "$MODEL" \
    --served-model-name "$MODEL" \
    --disaggregation-mode prefill \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
    --kv-events-config "{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:${KV_EVENTS_PORT_1}\",\"enable_kv_cache_events\":true}" \
    --tensor-parallel-size 1 \
    --max-model-len $MAX_MODEL_LEN \
    --max-num-seqs $MAX_NUM_SEQS \
    --kv-cache-dtype fp8 \
    --no-disable-hybrid-kv-cache-manager \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --dyn-reasoning-parser nemotron_v3 \
    > "$PREFILL_LOG_1" 2>&1 &
PREFILL_PID_1=$!
echo "Prefill worker 1 PID: $PREFILL_PID_1"

wait_for_worker "http://127.0.0.1:${PREFILL_SYSTEM_PORT_1}/health" "prefill worker 1"

# ==========================================
# Start prefill worker 2 (GPU 2)
# ==========================================
echo "Starting prefill worker 2 on GPU $PREFILL_GPU_2 (port $PREFILL_SYSTEM_PORT_2)..."
VLLM_ENGINE_STARTUP_TIMEOUT=600 \
CUDA_VISIBLE_DEVICES=$PREFILL_GPU_2 \
DYN_SYSTEM_PORT=$PREFILL_SYSTEM_PORT_2 \
VLLM_NIXL_SIDE_CHANNEL_PORT=$NIXL_SIDE_CHANNEL_PORT_2 \
VLLM_USE_FLASHINFER_MOE_FP8=1 \
python3 -m dynamo.vllm \
    --model "$MODEL" \
    --served-model-name "$MODEL" \
    --disaggregation-mode prefill \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
    --kv-events-config "{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:${KV_EVENTS_PORT_2}\",\"enable_kv_cache_events\":true}" \
    --tensor-parallel-size 1 \
    --max-model-len $MAX_MODEL_LEN \
    --max-num-seqs $MAX_NUM_SEQS \
    --kv-cache-dtype fp8 \
    --no-disable-hybrid-kv-cache-manager \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --dyn-reasoning-parser nemotron_v3 \
    > "$PREFILL_LOG_2" 2>&1 &
PREFILL_PID_2=$!
echo "Prefill worker 2 PID: $PREFILL_PID_2"

wait_for_worker "http://127.0.0.1:${PREFILL_SYSTEM_PORT_2}/health" "prefill worker 2"

# ==========================================
# Wait for Dynamo frontend to be ready
# All 3 workers must register before frontend serves requests
# ==========================================
wait_for_worker "http://127.0.0.1:${HTTP_PORT}/health" "Dynamo frontend"

# [FACT] Extra wait for all workers to fully register in etcd
# Frontend may return 200 before all workers are registered
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
echo "--- Prefill 1 log (last 3 lines) ---"
tail -3 "$PREFILL_LOG_1"
echo "--- Prefill 2 log (last 3 lines) ---"
tail -3 "$PREFILL_LOG_2"
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
    echo "Benchmarking scenario (DISAGG 2P:1D):"
    echo "  Input len:       $INPUT_LEN tokens"
    echo "  Output len:      $OUTPUT_LEN tokens"
    echo "  Num prompts:     $NUM_PROMPTS"
    echo "  Request rate:    $REQUEST_RATE req/s"
    echo "  Max concurrency: $MAX_CONCURRENCY"
    echo "  Topology:        2P:1D"
    echo "  Time:            $(tstamp)"
    echo "=========================================="

    RESULT_FILE="$RESULTS_DIR/DISAGG_2P1D__nemotron_nano_fp8__in${INPUT_LEN}_out${OUTPUT_LEN}__prm${NUM_PROMPTS}__rate${REQUEST_RATE}__seq${MAX_CONCURRENCY}__${RUN_ID}.json"

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
done

# ==========================================
# Summary
# ==========================================
echo "=========================================="
echo "All 2P:1D disaggregated benchmarks complete!"
echo "Results:"
ls -lh "$RESULTS_DIR"/DISAGG_2P1D__* 2>/dev/null || echo "No DISAGG_2P1D result files found"
echo "=========================================="

# cleanup() handles all process teardown via EXIT trap
