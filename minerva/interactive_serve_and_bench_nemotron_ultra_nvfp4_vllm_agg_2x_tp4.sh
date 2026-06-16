#!/bin/bash
set -euo pipefail

# ==========================================
# Nemotron Ultra NVFP4 — Aggregated Serve + Benchmark
# 2× TP4 baseline: 2 independent vLLM replicas (4 GPUs each)
# Round-robin proxy distributes requests across replicas
#
# PURPOSE: Fair comparison against DISAGG 1P:1D which also uses 8 GPUs.
# Each replica is identical — pure vLLM, no Dynamo orchestration.
#
# GPU assignment:
#   GPUs 0,1,2,3 → replica 0 (port 8001)
#   GPUs 4,5,6,7 → replica 1 (port 8002)
#   Proxy        → port 8000 (benchmark target)
#
# Requires: round_robin_proxy.py in the same directory as this script
#
# Naming: AGG_2xTP4__ prefix for easy result comparison
#
# [FACT] vLLM 0.22.0 reasoning parser flags live under StructuredOutputsConfig:
#   --reasoning-parser-plugin  path to plugin .py file
#   --reasoning-parser         parser name (ultra_v3 for Nemotron Ultra)
# These are plain vLLM flags — NOT --dyn-reasoning-parser (Dynamo-specific).
#
# [FACT] Do NOT set --kv-cache-dtype — vLLM auto-selects fp8_e4m3 for NVFP4.
# [FACT] Do NOT set VLLM_USE_FLASHINFER_MOE_FP8 — Ultra uses FP4 MoE kernel.
# [FACT] WORKER_READY_TIMEOUT=600 — dry run showed 218s init time on B200.
# [FACT] Block size is 8304 tokens (Mamba page alignment) — not the usual 16/32.
# [FACT] 81.18 GiB KV cache per GPU at max_model_len=32768 → ~421 max seqs.
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

# [FACT] Required for Mamba-2 KV transfer layout — even in AGG mode,
# ensures consistent KV cache layout with future DISAGG runs.
export VLLM_SSM_CONV_STATE_LAYOUT=DS

source /home/hossainm/miniforge3/bin/activate
eval "$(conda shell.bash hook)"
conda activate /home/hossainm/software/envs/conda_envs/dynamo_vllm_1.3.0_dev1

# ==========================================
# Configuration — edit here
# ==========================================
MODEL="nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4"
PARSER_PATH="$HF_HOME/reasoning_parsers/ultra_v3_reasoning_parser.py"

# GPU assignment — 2 independent TP=4 replicas
GPUS_REPLICA_0="0,1,2,3"
GPUS_REPLICA_1="4,5,6,7"

# Ports — proxy listens on 8000, replicas on 8001-8002
PROXY_PORT=8000
REPLICA_PORT_0=8001
REPLICA_PORT_1=8002

# Serving config — must match DISAGG scripts for fair comparison
MAX_MODEL_LEN=32768
MAX_NUM_SEQS=32

# Worker ready timeout (seconds)
# [FACT] Dry run confirmed 218s init time on B200 (FlashInfer JIT + kernel
# autotuning). 360s gives comfortable headroom. Cached after first run.
WORKER_READY_TIMEOUT=600

# Proxy script location — same directory as this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROXY_SCRIPT="$SCRIPT_DIR/round_robin_proxy.py"

# Results
RESULTS_DIR="$HOME/software/vllm_efforts/benchmark_results"

# Benchmark scenarios — matches Nano AGG 3-GPU scenarios for cross-model comparison
# Format: "input_len,output_len,num_prompts,request_rate,max_concurrency"
SCENARIOS=(
    #"16384,256,500,10,32"     # prefill-heavy — matches Nano scenario exactly
    "16384,2048,200,5,32"   # decode-heavy  — matches Nano scenario exactly
    #"32768,256,500,10,32"   # prefill-heavy, longer context — Ultra-specific
    #"32768,2048,200,5,32"   # decode-heavy,  longer context — Ultra-specific
)

# ==========================================
# Setup
# ==========================================
mkdir -p "$RESULTS_DIR"
RUN_ID="$(tstamp)"
PROXY_LOG="$RESULTS_DIR/AGG_2xTP4__proxy_${RUN_ID}.log"
REPLICA_LOG_0="$RESULTS_DIR/AGG_2xTP4__replica0_${RUN_ID}.log"
REPLICA_LOG_1="$RESULTS_DIR/AGG_2xTP4__replica1_${RUN_ID}.log"
touch "$PROXY_LOG" "$REPLICA_LOG_0" "$REPLICA_LOG_1"

echo "=========================================="
echo "AGG 2×TP4 Baseline — Nemotron Ultra NVFP4"
echo "vLLM version: 0.22.0 (dynamo_vllm_1.3.0_dev1 env)"
echo "=========================================="
echo "Run ID:      $RUN_ID"
echo "Model:       $MODEL"
echo "GPUs 0-3:    replica 0 → port $REPLICA_PORT_0"
echo "GPUs 4-7:    replica 1 → port $REPLICA_PORT_1"
echo "Proxy port:  $PROXY_PORT (benchmark target)"
echo "Topology:    2× independent AGG replicas, TP=4 each (round-robin)"
echo "Purpose:     Fair 8-GPU baseline vs DISAGG 1P:1D"
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

# Kill any leftover vLLM worker processes from previous runs
echo "Checking for leftover GPU processes..."
LEFTOVER=$(pgrep -f "VLLM\|vllm serve\|dynamo.vllm" 2>/dev/null || true)
if [ -n "$LEFTOVER" ]; then
    echo "WARNING: Killing leftover processes: $LEFTOVER"
    kill -9 $LEFTOVER 2>/dev/null || true
    sleep 5
    echo "Leftover processes cleared."
else
    echo "No leftover processes found."
fi

# Verify all GPUs are free before launching
echo "Checking GPU memory..."
nvidia-smi --query-gpu=index,memory.free,memory.total     --format=csv,noheader,nounits | while IFS=, read -r idx free total; do
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
# Download reasoning parser if missing
# ==========================================
if [ ! -f "$PARSER_PATH" ]; then
    echo "Downloading Ultra reasoning parser..."
    mkdir -p "$(dirname $PARSER_PATH)"
    wget -q \
         -e use_proxy=yes \
         -e https_proxy=$HTTPS_PROXY \
         -O "$PARSER_PATH" \
         "https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4/resolve/main/ultra_v3_reasoning_parser.py"
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
    # Also kill worker processes that may have escaped the process group
    pkill -9 -f "VLLM::Worker" 2>/dev/null || true
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
# Start both replicas in parallel
# [FACT] GPUs 0-3 and GPUs 4-7 are fully independent — disjoint HBM,
# separate FlashInfer JIT caches, no coordination during startup.
# Parallel launch cuts startup time from ~436s to ~218s.
# ==========================================
echo "Starting replica 0 on GPUs $GPUS_REPLICA_0 (port $REPLICA_PORT_0)..."
# [FACT] numactl pins CPU and memory to NUMA node 0 (GPUs 0-3 affinity)
# eliminates CPU contention with replica 1 during parallel autotuning
CUDA_VISIBLE_DEVICES=$GPUS_REPLICA_0 numactl --cpunodebind=0 --membind=0 \
vllm serve "$MODEL" \
    --served-model-name "$MODEL" \
    --tensor-parallel-size 4 \
    --max-model-len $MAX_MODEL_LEN \
    --max-num-seqs $MAX_NUM_SEQS \
    --trust-remote-code \
    --no-disable-hybrid-kv-cache-manager \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser-plugin "$PARSER_PATH" \
    --reasoning-parser ultra_v3 \
    --port $REPLICA_PORT_0 \
    > "$REPLICA_LOG_0" 2>&1 &
REPLICA_PID_0=$!
echo "Replica 0 PID: $REPLICA_PID_0"

echo "Starting replica 1 on GPUs $GPUS_REPLICA_1 (port $REPLICA_PORT_1)..."
# [FACT] numactl pins CPU and memory to NUMA node 1 (GPUs 4-7 affinity)
# eliminates CPU contention with replica 0 during parallel autotuning
# NOTE: node 1 has ~106GB free RAM vs 296GB on node 0 — monitor for CPU OOM
CUDA_VISIBLE_DEVICES=$GPUS_REPLICA_1 numactl --cpunodebind=1 --membind=1 \
vllm serve "$MODEL" \
    --served-model-name "$MODEL" \
    --tensor-parallel-size 4 \
    --max-model-len $MAX_MODEL_LEN \
    --max-num-seqs $MAX_NUM_SEQS \
    --trust-remote-code \
    --no-disable-hybrid-kv-cache-manager \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser-plugin "$PARSER_PATH" \
    --reasoning-parser ultra_v3 \
    --port $REPLICA_PORT_1 \
    > "$REPLICA_LOG_1" 2>&1 &
REPLICA_PID_1=$!
echo "Replica 1 PID: $REPLICA_PID_1"

echo "Both replicas launched in parallel — waiting for each to become ready..."
wait_for_worker "http://127.0.0.1:${REPLICA_PORT_0}/health" "replica 0"
wait_for_worker "http://127.0.0.1:${REPLICA_PORT_1}/health" "replica 1"

# ==========================================
# Start round-robin proxy
# ==========================================
echo "Starting round-robin proxy on port $PROXY_PORT..."
python3 "$PROXY_SCRIPT" \
    --port $PROXY_PORT \
    --backends $REPLICA_PORT_0 $REPLICA_PORT_1 \
    > "$PROXY_LOG" 2>&1 &
PROXY_PID=$!
echo "Proxy PID: $PROXY_PID"
sleep 3

wait_for_worker "http://127.0.0.1:${PROXY_PORT}/health" "proxy (all replicas)"

echo "Both replicas ready. Proxy distributing requests round-robin."
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
    echo "Benchmarking scenario (AGG 2×TP4):"
    echo "  Input len:       $INPUT_LEN tokens"
    echo "  Output len:      $OUTPUT_LEN tokens"
    echo "  Num prompts:     $NUM_PROMPTS"
    echo "  Request rate:    $REQUEST_RATE req/s"
    echo "  Max concurrency: $MAX_CONCURRENCY"
    echo "  Topology:        2× AGG replicas, TP=4 each (round-robin)"
    echo "  Time:            $(tstamp)"
    echo "=========================================="

    RESULT_FILE="$RESULTS_DIR/AGG_2xTP4__nemotron_ultra_nvfp4__in${INPUT_LEN}_out${OUTPUT_LEN}__prm${NUM_PROMPTS}__rate${REQUEST_RATE}__seq${MAX_CONCURRENCY}__${RUN_ID}.json"

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
    echo ""
done

# ==========================================
# Summary
# ==========================================
echo "=========================================="
echo "All AGG 2×TP4 benchmarks complete!"
echo "Results:"
ls -lh "$RESULTS_DIR"/AGG_2xTP4__* 2>/dev/null || echo "No AGG_2xTP4 result files found"
echo "=========================================="

# cleanup() handles all process teardown via EXIT trap
