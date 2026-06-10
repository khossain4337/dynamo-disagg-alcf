#!/bin/bash
set -euo pipefail

# ==========================================
# Nemotron Nano FP8 — Aggregated Serve + Benchmark
# 3-GPU baseline: 3 independent vLLM replicas (1 GPU each)
# Round-robin proxy distributes requests across replicas
#
# PURPOSE: Fair comparison against DISAGG 2P:1D which also uses 3 GPUs.
# Each replica is identical to the single-GPU AGG baseline.
#
# GPU assignment:
#   GPU 0 → replica 0 (port 8001)
#   GPU 1 → replica 1 (port 8002)
#   GPU 2 → replica 2 (port 8003)
#   Proxy → port 8000 (benchmark target)
#
# Requires: round_robin_proxy.py in the same directory as this script
#
# Naming: AGG_3GPU__ prefix for easy result comparison
#
# [FACT] vLLM 0.22.0 reasoning parser flags live under StructuredOutputsConfig:
#   --reasoning-parser-plugin  path to plugin .py file
#   --reasoning-parser         parser name (nano_v3 for Nemotron Nano)
# These are the same flags as v1.2.0 — NOT renamed to --dyn-reasoning-parser.
# --dyn-reasoning-parser is Dynamo-specific (dynamo.vllm only).
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

source /home/hossainm/miniforge3/bin/activate
eval "$(conda shell.bash hook)"
conda activate /home/hossainm/software/envs/conda_envs/dynamo_vllm_1.3.0_dev1

# ==========================================
# Configuration — edit here
# ==========================================
MODEL="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"
PARSER_PATH="$HF_HOME/reasoning_parsers/nano_v3_reasoning_parser.py"

# GPU assignment — 3 independent replicas
GPU_0=0
GPU_1=1
GPU_2=2

# Ports — proxy listens on 8000, replicas on 8001-8003
PROXY_PORT=8000
REPLICA_PORT_0=8001
REPLICA_PORT_1=8002
REPLICA_PORT_2=8003

# Serving config — must match DISAGG scripts for fair comparison
MAX_MODEL_LEN=32768
MAX_NUM_SEQS=32

# Worker ready timeout (seconds)
# [FACT] First startup requires FlashInfer JIT compilation — can take
# 15-20 minutes cold. Cached in ~/.cache/flashinfer/ after first run.
WORKER_READY_TIMEOUT=1800

# Proxy script location — same directory as this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROXY_SCRIPT="$SCRIPT_DIR/round_robin_proxy.py"

# Results
RESULTS_DIR="$HOME/software/vllm_efforts/benchmark_results"

# Benchmark scenarios — must match DISAGG scripts for fair comparison
# Format: "input_len,output_len,num_prompts,request_rate,max_concurrency"
SCENARIOS=(
    #"16384,256,500,10,32"
    "16384,2048,200,5,32"
    #"8192,256,500,10,32"
)

# ==========================================
# Setup
# ==========================================
mkdir -p "$RESULTS_DIR"
RUN_ID="$(tstamp)"
PROXY_LOG="$RESULTS_DIR/AGG_3GPU__proxy_${RUN_ID}.log"
REPLICA_LOG_0="$RESULTS_DIR/AGG_3GPU__replica0_${RUN_ID}.log"
REPLICA_LOG_1="$RESULTS_DIR/AGG_3GPU__replica1_${RUN_ID}.log"
REPLICA_LOG_2="$RESULTS_DIR/AGG_3GPU__replica2_${RUN_ID}.log"
touch "$PROXY_LOG" "$REPLICA_LOG_0" "$REPLICA_LOG_1" "$REPLICA_LOG_2"

echo "=========================================="
echo "AGG 3-GPU Baseline — Nemotron Nano FP8"
echo "vLLM version: 0.22.0 (dynamo_vllm_1.3.0_dev1 env)"
echo "=========================================="
echo "Run ID:      $RUN_ID"
echo "Model:       $MODEL"
echo "GPU 0:       replica 0 → port $REPLICA_PORT_0"
echo "GPU 1:       replica 1 → port $REPLICA_PORT_1"
echo "GPU 2:       replica 2 → port $REPLICA_PORT_2"
echo "Proxy port:  $PROXY_PORT (benchmark target)"
echo "Topology:    3x independent AGG replicas (round-robin)"
echo "Purpose:     Fair 3-GPU baseline vs DISAGG 2P:1D"
echo "Results:     $RESULTS_DIR"
echo "Time:        $(tstamp)"
echo "=========================================="

# ==========================================
# Sanity checks
# ==========================================
if [ ! -f "$PROXY_SCRIPT" ]; then
    echo "ERROR: round_robin_proxy.py not found at $PROXY_SCRIPT" >&2
    echo "Place round_robin_proxy.py in the same directory as this script." >&2
    exit 1
fi

# ==========================================
# Download reasoning parser if missing
# ==========================================
if [ ! -f "$PARSER_PATH" ]; then
    echo "Downloading custom reasoning parser..."
    mkdir -p "$(dirname $PARSER_PATH)"
    wget -q \
         -e use_proxy=yes \
         -e https_proxy=$HTTPS_PROXY \
         -O "$PARSER_PATH" \
         https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8/resolve/main/nano_v3_reasoning_parser.py
    echo "Parser downloaded: $PARSER_PATH"
else
    echo "Reasoning parser found: $PARSER_PATH"
fi

# ==========================================
# Cleanup on exit
# ==========================================
cleanup() {
    echo ""
    echo "Cleaning up all processes..."
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
# Start replica 0 (GPU 0, port 8001)
# ==========================================
echo "Starting replica 0 on GPU $GPU_0 (port $REPLICA_PORT_0)..."
CUDA_VISIBLE_DEVICES=$GPU_0 \
VLLM_USE_FLASHINFER_MOE_FP8=1 \
vllm serve "$MODEL" \
    --served-model-name "$MODEL" \
    --tensor-parallel-size 1 \
    --max-model-len $MAX_MODEL_LEN \
    --max-num-seqs $MAX_NUM_SEQS \
    --kv-cache-dtype fp8 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser-plugin "$PARSER_PATH" \
    --reasoning-parser nano_v3 \
    --port $REPLICA_PORT_0 \
    > "$REPLICA_LOG_0" 2>&1 &
REPLICA_PID_0=$!
echo "Replica 0 PID: $REPLICA_PID_0"

wait_for_worker "http://127.0.0.1:${REPLICA_PORT_0}/health" "replica 0"

# ==========================================
# Start replica 1 (GPU 1, port 8002)
# ==========================================
echo "Starting replica 1 on GPU $GPU_1 (port $REPLICA_PORT_1)..."
CUDA_VISIBLE_DEVICES=$GPU_1 \
VLLM_USE_FLASHINFER_MOE_FP8=1 \
vllm serve "$MODEL" \
    --served-model-name "$MODEL" \
    --tensor-parallel-size 1 \
    --max-model-len $MAX_MODEL_LEN \
    --max-num-seqs $MAX_NUM_SEQS \
    --kv-cache-dtype fp8 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser-plugin "$PARSER_PATH" \
    --reasoning-parser nano_v3 \
    --port $REPLICA_PORT_1 \
    > "$REPLICA_LOG_1" 2>&1 &
REPLICA_PID_1=$!
echo "Replica 1 PID: $REPLICA_PID_1"

wait_for_worker "http://127.0.0.1:${REPLICA_PORT_1}/health" "replica 1"

# ==========================================
# Start replica 2 (GPU 2, port 8003)
# ==========================================
echo "Starting replica 2 on GPU $GPU_2 (port $REPLICA_PORT_2)..."
CUDA_VISIBLE_DEVICES=$GPU_2 \
VLLM_USE_FLASHINFER_MOE_FP8=1 \
vllm serve "$MODEL" \
    --served-model-name "$MODEL" \
    --tensor-parallel-size 1 \
    --max-model-len $MAX_MODEL_LEN \
    --max-num-seqs $MAX_NUM_SEQS \
    --kv-cache-dtype fp8 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser-plugin "$PARSER_PATH" \
    --reasoning-parser nano_v3 \
    --port $REPLICA_PORT_2 \
    > "$REPLICA_LOG_2" 2>&1 &
REPLICA_PID_2=$!
echo "Replica 2 PID: $REPLICA_PID_2"

wait_for_worker "http://127.0.0.1:${REPLICA_PORT_2}/health" "replica 2"

# ==========================================
# Start round-robin proxy
# ==========================================
echo "Starting round-robin proxy on port $PROXY_PORT..."
python3 "$PROXY_SCRIPT" \
    --port $PROXY_PORT \
    --backends $REPLICA_PORT_0 $REPLICA_PORT_1 $REPLICA_PORT_2 \
    > "$PROXY_LOG" 2>&1 &
PROXY_PID=$!
echo "Proxy PID: $PROXY_PID"
sleep 3

wait_for_worker "http://127.0.0.1:${PROXY_PORT}/health" "proxy (all replicas)"

echo "All 3 replicas ready. Proxy distributing requests round-robin."
echo ""

# ==========================================
# Tail logs briefly
# ==========================================
echo "--- Proxy log (last 3 lines) ---"
tail -3 "$PROXY_LOG"
echo "--- Replica 0 log (last 3 lines) ---"
tail -3 "$REPLICA_LOG_0"
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
    echo "Benchmarking scenario (AGG 3-GPU):"
    echo "  Input len:       $INPUT_LEN tokens"
    echo "  Output len:      $OUTPUT_LEN tokens"
    echo "  Num prompts:     $NUM_PROMPTS"
    echo "  Request rate:    $REQUEST_RATE req/s"
    echo "  Max concurrency: $MAX_CONCURRENCY"
    echo "  Topology:        3x AGG replicas (round-robin)"
    echo "  Time:            $(tstamp)"
    echo "=========================================="

    RESULT_FILE="$RESULTS_DIR/AGG_3GPU__nemotron_nano_fp8__in${INPUT_LEN}_out${OUTPUT_LEN}__prm${NUM_PROMPTS}__rate${REQUEST_RATE}__seq${MAX_CONCURRENCY}__${RUN_ID}.json"

    vllm bench serve \
        --backend openai-chat \
        --model "$MODEL" \
        --dataset-name random \
        --random-input-len $INPUT_LEN \
        --random-output-len $OUTPUT_LEN \
        --endpoint /v1/chat/completions \
        --host 127.0.0.1 \
        --port $PROXY_PORT \
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
echo "All AGG 3-GPU benchmarks complete!"
echo "Results:"
ls -lh "$RESULTS_DIR"/AGG_3GPU__* 2>/dev/null || echo "No AGG_3GPU result files found"
echo "=========================================="

# cleanup() handles all process teardown via EXIT trap
