#!/bin/bash
set -euo pipefail

# =============================================================================
# Nemotron Ultra NVFP4 — Dynamo AGG 4×TP4 — Rate Sweep — Single Entry Point
# Run this script ONLY on minerva-dgx-01. It SSHes to node2 automatically.
#
# USAGE:
#   bash interactive_single_entry_multinode_ultra_agg_4xtp4_rr_rate_sweep_N2.sh
#
# WHAT IT DOES:
#   1. Starts etcd, NATS, Dynamo frontend (round-robin) on node1
#   2. Starts workers SEQUENTIALLY to avoid FlashInfer JIT NFS race:
#        worker-0 (node1, GPU 0-3, NUMA 0) → wait healthy
#        worker-1 (node1, GPU 4-7, NUMA 1) → wait healthy
#        SSH → worker-2 (node2, GPU 0-3)   → wait healthy
#              worker-3 (node2, GPU 4-7)   → wait healthy
#   3. Sweeps RATE: 0.05 → 0.10 → 0.20 → 0.50 → 1.00 req/s
#      ISL=131,072 and OSL=256 fixed; max_model_len=131,328 fixed for all steps.
#      Workers stay alive across all rate steps — only benchmark reruns.
#   4. On exit (Ctrl+C or completion): kills node1 processes AND node2 via SSH
#
# Topology (4× AGG, TP=4 per worker — apples-to-apples with DISAGG 2P:2D):
#   minerva-dgx-01 (node1) — run this script here:
#     etcd        → 0.0.0.0:2379, advertised at 10.125.13.11:2379
#     NATS        → 0.0.0.0:4222
#     Frontend    → HTTP 8000 (Dynamo round-robin)
#     Worker-0    → GPUs 0,1,2,3  PORT=8081  NUMA 0
#     Worker-1    → GPUs 4,5,6,7  PORT=8082  NUMA 1
#
#   minerva-dgx-02 (node2) — launched automatically via SSH from this script:
#     Worker-2    → GPUs 0,1,2,3  PORT=8083  NUMA 0
#     Worker-3    → GPUs 4,5,6,7  PORT=8084  NUMA 1
#
# Comparison with DISAGG 2P:2D:
#   Same total GPUs: 16 (4 workers × TP=4)
#   Same nodes: 2
#   Same model: Nemotron Ultra NVFP4
#   Same ISL/OSL/rates
#   Only difference: AGG (no prefill/decode separation, no KV transfer)
#
# [FACT] No --disaggregation-mode flag — workers run in standard AGG mode.
# [FACT] No --kv-transfer-config — no NIXL transfers in AGG mode.
# [FACT] No --kv-events-config — no KV event publishing needed in AGG mode.
# [FACT] SSM patch (nixl/worker.py) checked at startup — belt-and-suspenders.
# [FACT] Workers are NOT restarted between rate steps — ISL/OSL/max_model_len
#        are identical across all steps. Only the benchmark client rate changes.
# [FACT] Sequential startup: worker-0 → worker-1 → SSH(worker-2 → worker-3).
#        Each worker waits for healthy before the next fires.
#        Prevents FlashInfer JIT NFS race (all ranks writing same .so path).
#        Cost: ~+5-10min vs parallel. Worth it for reliability on NFS.
# =============================================================================

tstamp() { date +"%Y-%m-%d-%H%M%S"; }
log()    { echo "[$(tstamp)] $*"; }

# =============================================================================
# SECTION 1 — Cluster topology
# =============================================================================
NODE1_HOST="minerva-dgx-01"
NODE2_HOST="minerva-dgx-02"
NODE1_IP="10.125.13.11"
NODE2_IP="10.125.13.12"

# Guard: this script must run on node1 only
THIS_HOST=$(hostname -s)
if [ "$THIS_HOST" != "$NODE1_HOST" ]; then
    echo "ERROR: Run this script on $NODE1_HOST, not $THIS_HOST." >&2
    exit 1
fi

# =============================================================================
# SECTION 2 — Environment (node1)
# =============================================================================
export HF_TOKEN=$(cat ~/.hf_token)
export HF_HOME="/home/hossainm/software/model-weights"

export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128
# CRITICAL: exclude both node IPs — ALCF proxy intercepts etcd/NATS TCP silently
export no_proxy="127.0.0.1,localhost,${NODE1_IP},${NODE2_IP},${NODE1_HOST},${NODE2_HOST}"
export NO_PROXY="127.0.0.1,localhost,${NODE1_IP},${NODE2_IP},${NODE1_HOST},${NODE2_HOST}"

export PYTHONHASHSEED=0
export VLLM_SSM_CONV_STATE_LAYOUT=DS
export ETCD_ENDPOINTS="http://${NODE1_IP}:2379"
export NATS_SERVER="nats://${NODE1_IP}:4222"

source /home/hossainm/miniforge3/bin/activate
eval "$(conda shell.bash hook)"
conda activate /home/hossainm/software/envs/conda_envs/dynamo_vllm_1.3.0_dev1

# =============================================================================
# SECTION 3 — Fixed configuration (identical across all rate steps)
# =============================================================================
# [FACT] Split MODEL_PATH and MODEL_NAME to bypass HF resolver entirely.
# Workers load from local snapshot path — no download attempt, no refs/main
# dependency. Benchmark client and --served-model-name use canonical HF ID.
# See PROJECT_MEMORY §28 for root cause of HF resolver bug.
MODEL_PATH="/home/hossainm/software/model-weights/hub/models--nvidia--NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4/snapshots/504c145dce0744d9a61fd458d11febb88aa8890c"
MODEL_NAME="nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4"
TP=4

HTTP_PORT=8000

# Worker system ports — unique per worker across both nodes
WORKER_PORT_0=8081    # node1 worker-0
WORKER_PORT_1=8082    # node1 worker-1
WORKER_PORT_2=8083    # node2 worker-2
WORKER_PORT_3=8084    # node2 worker-3

# GPU assignments
GPUS_W0="0,1,2,3"    # node1 NUMA 0
GPUS_W1="4,5,6,7"    # node1 NUMA 1
GPUS_W2="0,1,2,3"    # node2 NUMA 0
GPUS_W3="4,5,6,7"    # node2 NUMA 1

# Fixed ISL/OSL for all rate steps
ISL=131072
OSL=256
MAX_MODEL_LEN=$(( ISL + OSL ))   # 131328
MAX_NUM_SEQS=32

RESULTS_DIR="$HOME/software/vllm_efforts/benchmark_results"
WORKER_READY_TIMEOUT=2700   # 15 min per worker (covers cold FlashInfer JIT)
TTFT_LIMIT_MS=999999999    # no early stop — collect full curve

# =============================================================================
# SECTION 4 — Rate sweep definition
#
# Format: "NUM_PROMPTS,REQUEST_RATE,MAX_CONCURRENCY"
# ISL, OSL, MAX_NUM_SEQS, MAX_MODEL_LEN are fixed above — not per-step.
#
# [LOGICAL DEDUCTION] AGG λ* expected LOWER than DISAGG (prefill/decode share
# GPUs). DISAGG flat region held to rate=0.20. AGG may saturate earlier.
# We use identical rate points for direct comparison.
#
# MAX_CONCURRENCY scales with rate — same logic as DISAGG rate sweep.
# =============================================================================
RATE_SWEEP=(
    "20,0.05,4"
    "20,0.10,4"
    "20,0.20,8"
    "20,0.50,16"
    "20,1.00,32"
)

# =============================================================================
# SECTION 5 — Helper functions
# =============================================================================
wait_for_worker() {
    local url="$1" name="$2" timeout="${3:-$WORKER_READY_TIMEOUT}"
    local start=$SECONDS
    log "Waiting for $name at $url (timeout: ${timeout}s)..."
    while (( SECONDS - start < timeout )); do
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
            --max-time 3 "$url") || HTTP_CODE="000"
        if [ "$HTTP_CODE" = "200" ]; then
            log "$name ready after $(( SECONDS - start ))s ✓"
            return 0
        fi
        sleep 10
        log "  ...$name still starting ($(( SECONDS - start ))s elapsed) [HTTP: $HTTP_CODE]"
    done
    log "ERROR: $name not ready after ${timeout}s" >&2
    return 1
}

wait_for_worker_remote() {
    local ip="$1" port="$2" name="$3" timeout="${4:-$WORKER_READY_TIMEOUT}"
    local start=$SECONDS
    log "Waiting for $name at http://${ip}:${port}/health (timeout: ${timeout}s)..."
    while (( SECONDS - start < timeout )); do
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
            --max-time 3 "http://${ip}:${port}/health") || HTTP_CODE="000"
        if [ "$HTTP_CODE" = "200" ]; then
            log "$name ready after $(( SECONDS - start ))s ✓"
            return 0
        fi
        sleep 10
        log "  ...$name still starting ($(( SECONDS - start ))s elapsed) [HTTP: $HTTP_CODE]"
    done
    log "ERROR: $name at ${ip}:${port} not ready after ${timeout}s" >&2
    return 1
}

check_ssm_patch() {
    local patch_file="$HOME/software/envs/conda_envs/dynamo_vllm_1.3.0_dev1/lib/python3.12/site-packages/vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py"
    if grep -q "num_blocks = min" "$patch_file" 2>/dev/null; then
        log "  SSM patch: CONFIRMED ✓"
    else
        log "ERROR: SSM patch NOT found in nixl/worker.py" >&2
        log "Run vllm_nixl_assert_patch.sh first. Patch on NFS propagates to node2." >&2
        exit 1
    fi
}

# =============================================================================
# SECTION 6 — Cleanup trap
# =============================================================================
NODE2_SSH_PID=""

cleanup() {
    log ""
    log "=========================================="
    log "CLEANUP: Shutting down all workers..."
    log "=========================================="
    log "Killing node1 processes (kill 0)..."
    kill 0 2>/dev/null || true
    log "Killing node2 workers via SSH..."
    ssh "$NODE2_HOST" \
        "pkill -9 -if 'dynamo.vllm|vllm' 2>/dev/null || true; echo 'node2 kill done'" \
        2>/dev/null || log "  (SSH to node2 during cleanup failed — may already be dead)"
    if [ -n "$NODE2_SSH_PID" ]; then
        kill "$NODE2_SSH_PID" 2>/dev/null || true
        log "  Background SSH (PID $NODE2_SSH_PID) killed."
    fi
    sleep 3
    log "Cleanup done. Goodbye."
}
trap cleanup EXIT

# =============================================================================
# SECTION 7 — Pre-flight checks
# =============================================================================
mkdir -p "$RESULTS_DIR"
RUN_ID="$(tstamp)"
SWEEP_SUMMARY="$RESULTS_DIR/ULTRA_AGG_4xTP4_RR_2NODE__rate_sweep_summary_${RUN_ID}.txt"

log "=========================================="
log "Nemotron Ultra NVFP4 — AGG 4×TP4 Round-Robin Rate Sweep"
log "ISL=${ISL}, OSL=${OSL} (fixed), max_model_len=${MAX_MODEL_LEN}"
log "Single entry point — node1 launches node2 via SSH"
log "Dynamo v1.3.0-dev.1"
log "=========================================="
log "Run ID:    $RUN_ID"
log "Model:     $MODEL_NAME"
log "Results:   $RESULTS_DIR"
log "Summary:   $SWEEP_SUMMARY"
log "Node1:     $NODE1_HOST ($NODE1_IP) — etcd, NATS, frontend, worker-0, worker-1"
log "Node2:     $NODE2_HOST ($NODE2_IP) — worker-2, worker-3 (auto-launched)"
log "Rate steps: ${#RATE_SWEEP[@]}"
log "Router:    Dynamo round-robin"
log "TTFT limit: ${TTFT_LIMIT_MS}ms"
log "Startup:   SEQUENTIAL (w0→w1→w2→w3) — avoids FlashInfer JIT NFS race"
log "=========================================="

check_ssm_patch

log "Checking SSH to node2..."
if ! ssh -o ConnectTimeout=10 "$NODE2_HOST" "hostname" > /dev/null 2>&1; then
    log "ERROR: Cannot SSH to $NODE2_HOST. Is the PBS co-allocation active?" >&2
    exit 1
fi
log "  SSH to $NODE2_HOST: OK ✓"

log "GPU memory (node1):"
nvidia-smi --query-gpu=index,memory.free,memory.total \
    --format=csv,noheader,nounits | while IFS=, read -r idx free total; do
    free=$(echo "$free" | tr -d ' ')
    total=$(echo "$total" | tr -d ' ')
    pct=$(( free * 100 / total ))
    if [ "$pct" -lt 90 ]; then
        log "  ERROR: GPU $idx only ${free}/${total} MiB free (${pct}%) — not safe" >&2
        exit 1
    fi
    log "  GPU $idx: ${free}/${total} MiB free (${pct}%) ✓"
done

log "GPU memory (node2):"
ssh "$NODE2_HOST" "nvidia-smi --query-gpu=index,memory.free,memory.total \
    --format=csv,noheader,nounits" | while IFS=, read -r idx free total; do
    free=$(echo "$free" | tr -d ' ')
    total=$(echo "$total" | tr -d ' ')
    pct=$(( free * 100 / total ))
    log "  GPU $idx: ${free}/${total} MiB free (${pct}%)"
done

# =============================================================================
# SECTION 8 — Start infrastructure (etcd, NATS, frontend)
#             These stay up for the ENTIRE sweep.
# =============================================================================
ETCD_LOG="$RESULTS_DIR/ULTRA_AGG_4xTP4_RR_2NODE__etcd_${RUN_ID}.log"
NATS_LOG="$RESULTS_DIR/ULTRA_AGG_4xTP4_RR_2NODE__nats_${RUN_ID}.log"
FRONTEND_LOG="$RESULTS_DIR/ULTRA_AGG_4xTP4_RR_2NODE__frontend_${RUN_ID}.log"

log "Starting etcd (0.0.0.0:2379, advertising ${NODE1_IP})..."
etcd \
    --data-dir "/tmp/etcd_dynamo_${RUN_ID}" \
    --listen-client-urls "http://0.0.0.0:2379" \
    --advertise-client-urls "http://${NODE1_IP}:2379" \
    --listen-peer-urls "http://0.0.0.0:2380" \
    --initial-advertise-peer-urls "http://${NODE1_IP}:2380" \
    --initial-cluster "default=http://${NODE1_IP}:2380" \
    > "$ETCD_LOG" 2>&1 &
ETCD_PID=$!
log "etcd PID: $ETCD_PID"
sleep 3

ETCD_OK=0
for i in {1..10}; do
    if curl -sf "http://${NODE1_IP}:2379/health" > /dev/null 2>&1; then
        log "etcd healthy ✓"; ETCD_OK=1; break
    fi
    sleep 2
    log "  ...etcd not ready yet (attempt $i/10)"
done
[ "$ETCD_OK" -eq 1 ] || { log "ERROR: etcd failed to start" >&2; exit 1; }

log "Starting NATS (0.0.0.0:4222)..."
nats-server -p 4222 -a 0.0.0.0 > "$NATS_LOG" 2>&1 &
NATS_PID=$!
log "NATS PID: $NATS_PID"
sleep 2

log "Starting Dynamo frontend (round-robin router, port $HTTP_PORT)..."
DYN_ROUTER_MODE=round-robin \
DYN_HTTP_PORT=$HTTP_PORT \
ETCD_ENDPOINTS="http://${NODE1_IP}:2379" \
NATS_SERVER="nats://${NODE1_IP}:4222" \
python -m dynamo.frontend \
    > "$FRONTEND_LOG" 2>&1 &
FRONTEND_PID=$!
log "Frontend PID: $FRONTEND_PID"
sleep 5

# =============================================================================
# SECTION 9 — Start all 4 workers SEQUENTIALLY
#
# Startup sequence (sequential — avoids FlashInfer JIT NFS race):
#   t=0s     worker-0 fires (node1, GPU 0-3, NUMA 0)
#   t=~320s  worker-0 healthy → worker-1 fires (node1, GPU 4-7, NUMA 1)
#   t=~640s  worker-1 healthy → SSH fires → worker-2 fires (node2, GPU 0-3)
#   t=~960s  worker-2 healthy (polled from node1) → worker-3 fires (node2, GPU 4-7)
#   t=~1280s worker-3 healthy → 30s etcd settle → benchmark loop starts
#
# [FACT] Sequential startup adds ~+15-20min vs parallel.
# [LOGICAL DEDUCTION] Worth it: FlashInfer JIT writes a .so to NFS under
#   ~/.cache/flashinfer/. With need_lock=False, concurrent ranks race to
#   write the same file — partial writes corrupt the cache, causing the
#   Ninja build failure seen on worker-3. Sequential startup ensures each
#   worker's JIT compile completes before the next rank touches the same path.
# [FACT] In AGG mode, worker ordering does NOT affect correctness — all 4
#   workers are symmetric. Sequential order is arbitrary.
# =============================================================================
W0_LOG="$RESULTS_DIR/ULTRA_AGG_4xTP4_RR_2NODE__worker0_${RUN_ID}.log"
W1_LOG="$RESULTS_DIR/ULTRA_AGG_4xTP4_RR_2NODE__worker1_${RUN_ID}.log"
W2_LOG="$RESULTS_DIR/ULTRA_AGG_4xTP4_RR_2NODE__worker2_${RUN_ID}.log"
W3_LOG="$RESULTS_DIR/ULTRA_AGG_4xTP4_RR_2NODE__worker3_${RUN_ID}.log"
touch "$W0_LOG" "$W1_LOG" "$W2_LOG" "$W3_LOG"

STARTUP_START=$SECONDS

# ---------------------------------------------------------------------------
# Worker-0 (node1, GPUs 0-3, NUMA 0) — fires first
# ---------------------------------------------------------------------------
log "Starting worker-0 (GPUs $GPUS_W0, NUMA 0, PORT $WORKER_PORT_0)..."
VLLM_ENGINE_STARTUP_TIMEOUT=600 \
CUDA_VISIBLE_DEVICES=$GPUS_W0 \
DYN_SYSTEM_PORT=$WORKER_PORT_0 \
ETCD_ENDPOINTS="http://${NODE1_IP}:2379" \
NATS_SERVER="nats://${NODE1_IP}:4222" \
numactl --cpunodebind=0 --membind=0 \
python3 -m dynamo.vllm \
    --model "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --tensor-parallel-size $TP \
    --max-model-len $MAX_MODEL_LEN \
    --max-num-seqs $MAX_NUM_SEQS \
    --no-disable-hybrid-kv-cache-manager \
    --trust-remote-code \
    --dyn-reasoning-parser nemotron_v3 \
    > "$W0_LOG" 2>&1 &
WORKER0_PID=$!
log "  worker-0 PID: $WORKER0_PID"

# Wait for worker-0 before firing worker-1
wait_for_worker "http://127.0.0.1:${WORKER_PORT_0}/health" "worker-0 (node1)"

# ---------------------------------------------------------------------------
# Worker-1 (node1, GPUs 4-7, NUMA 1) — fires after worker-0 is healthy
# ---------------------------------------------------------------------------
log "Starting worker-1 (GPUs $GPUS_W1, NUMA 1, PORT $WORKER_PORT_1)..."
VLLM_ENGINE_STARTUP_TIMEOUT=600 \
CUDA_VISIBLE_DEVICES=$GPUS_W1 \
DYN_SYSTEM_PORT=$WORKER_PORT_1 \
ETCD_ENDPOINTS="http://${NODE1_IP}:2379" \
NATS_SERVER="nats://${NODE1_IP}:4222" \
numactl --cpunodebind=1 --membind=1 \
python3 -m dynamo.vllm \
    --model "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --tensor-parallel-size $TP \
    --max-model-len $MAX_MODEL_LEN \
    --max-num-seqs $MAX_NUM_SEQS \
    --no-disable-hybrid-kv-cache-manager \
    --trust-remote-code \
    --dyn-reasoning-parser nemotron_v3 \
    > "$W1_LOG" 2>&1 &
WORKER1_PID=$!
log "  worker-1 PID: $WORKER1_PID"

# Wait for worker-1 before SSHing to node2
wait_for_worker "http://127.0.0.1:${WORKER_PORT_1}/health" "worker-1 (node1)"

# ---------------------------------------------------------------------------
# SSH to node2 — fires after worker-1 is healthy.
# Node2 starts worker-2, waits for it, then starts worker-3.
# Node1 polls node2 health from outside using NODE2_IP (after SSH fires).
# ---------------------------------------------------------------------------
log "node1 workers healthy. SSHing to node2 to start worker-2 then worker-3..."
ssh "$NODE2_HOST" \
    NODE1_IP="$NODE1_IP" \
    NODE2_IP="$NODE2_IP" \
    NODE1_HOST="$NODE1_HOST" \
    NODE2_HOST="$NODE2_HOST" \
    MODEL_PATH="$MODEL_PATH" \
    MODEL_NAME="$MODEL_NAME" \
    MAX_MODEL_LEN="$MAX_MODEL_LEN" \
    MAX_NUM_SEQS="$MAX_NUM_SEQS" \
    TP="$TP" \
    WORKER_PORT_2="$WORKER_PORT_2" \
    WORKER_PORT_3="$WORKER_PORT_3" \
    GPUS_W2="$GPUS_W2" \
    GPUS_W3="$GPUS_W3" \
    W2_LOG="$W2_LOG" \
    W3_LOG="$W3_LOG" \
    bash <<'ENDSSH' &

# ---- Everything below runs on node2 ----
set -euo pipefail

export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128
export no_proxy="127.0.0.1,localhost,${NODE1_IP},${NODE2_IP},${NODE1_HOST},${NODE2_HOST}"
export NO_PROXY="127.0.0.1,localhost,${NODE1_IP},${NODE2_IP},${NODE1_HOST},${NODE2_HOST}"

export HF_TOKEN=$(cat ~/.hf_token)
export HF_HOME="/home/hossainm/software/model-weights"
export PYTHONHASHSEED=0
export VLLM_SSM_CONV_STATE_LAYOUT=DS
export ETCD_ENDPOINTS="http://${NODE1_IP}:2379"
export NATS_SERVER="nats://${NODE1_IP}:4222"

source /home/hossainm/miniforge3/bin/activate
eval "$(conda shell.bash hook)"
conda activate /home/hossainm/software/envs/conda_envs/dynamo_vllm_1.3.0_dev1

echo "[node2] Environment ready."
echo "[node2] MAX_MODEL_LEN=$MAX_MODEL_LEN  MAX_NUM_SEQS=$MAX_NUM_SEQS  MODEL=$MODEL_NAME"
echo "[node2] W2 PORT=$WORKER_PORT_2  W3 PORT=$WORKER_PORT_3"

# Confirm etcd on node1 is reachable
for i in {1..12}; do
    if curl -sf "http://${NODE1_IP}:2379/health" > /dev/null 2>&1; then
        echo "[node2] etcd reachable ✓"; break
    fi
    if [ "$i" -eq 12 ]; then
        echo "[node2] ERROR: Cannot reach etcd at ${NODE1_IP}:2379 after 60s" >&2; exit 1
    fi
    sleep 5
    echo "[node2] ...etcd not reachable yet (attempt $i/12)"
done

# Kill any leftovers on node2
pkill -9 -if "dynamo.vllm|vllm" 2>/dev/null || true
sleep 2

# Worker-2 (GPUs 0-3, NUMA 0)
echo "[node2] Starting worker-2 (GPUs $GPUS_W2, NUMA 0, PORT $WORKER_PORT_2)..."
VLLM_ENGINE_STARTUP_TIMEOUT=600 \
CUDA_VISIBLE_DEVICES=$GPUS_W2 \
DYN_SYSTEM_PORT=$WORKER_PORT_2 \
ETCD_ENDPOINTS="http://${NODE1_IP}:2379" \
NATS_SERVER="nats://${NODE1_IP}:4222" \
numactl --cpunodebind=0 --membind=0 \
python3 -m dynamo.vllm \
    --model "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --tensor-parallel-size $TP \
    --max-model-len $MAX_MODEL_LEN \
    --max-num-seqs $MAX_NUM_SEQS \
    --no-disable-hybrid-kv-cache-manager \
    --trust-remote-code \
    --dyn-reasoning-parser nemotron_v3 \
    >> "$W2_LOG" 2>&1 &
WORKER2_PID=$!
echo "[node2] worker-2 PID: $WORKER2_PID"

# Wait for worker-2 before starting worker-3
W2_START=$SECONDS
while (( SECONDS - W2_START < 900 )); do
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
        --max-time 3 "http://127.0.0.1:${WORKER_PORT_2}/health") || HTTP_CODE="000"
    if [ "$HTTP_CODE" = "200" ]; then
        echo "[node2] worker-2 ready after $(( SECONDS - W2_START ))s ✓"; break
    fi
    sleep 10
    echo "[node2] ...worker-2 still starting ($(( SECONDS - W2_START ))s) [HTTP: $HTTP_CODE]"
done

# Worker-3 (GPUs 4-7, NUMA 1) — starts after worker-2 is healthy
echo "[node2] Starting worker-3 (GPUs $GPUS_W3, NUMA 1, PORT $WORKER_PORT_3)..."
VLLM_ENGINE_STARTUP_TIMEOUT=600 \
CUDA_VISIBLE_DEVICES=$GPUS_W3 \
DYN_SYSTEM_PORT=$WORKER_PORT_3 \
ETCD_ENDPOINTS="http://${NODE1_IP}:2379" \
NATS_SERVER="nats://${NODE1_IP}:4222" \
numactl --cpunodebind=1 --membind=1 \
python3 -m dynamo.vllm \
    --model "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --tensor-parallel-size $TP \
    --max-model-len $MAX_MODEL_LEN \
    --max-num-seqs $MAX_NUM_SEQS \
    --no-disable-hybrid-kv-cache-manager \
    --trust-remote-code \
    --dyn-reasoning-parser nemotron_v3 \
    >> "$W3_LOG" 2>&1 &
WORKER3_PID=$!
echo "[node2] worker-3 PID: $WORKER3_PID"

# Wait for worker-3, then stay alive
W3_START=$SECONDS
while (( SECONDS - W3_START < 900 )); do
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
        --max-time 3 "http://127.0.0.1:${WORKER_PORT_3}/health") || HTTP_CODE="000"
    if [ "$HTTP_CODE" = "200" ]; then
        echo "[node2] worker-3 ready after $(( SECONDS - W3_START ))s ✓"; break
    fi
    sleep 10
    echo "[node2] ...worker-3 still starting ($(( SECONDS - W3_START ))s) [HTTP: $HTTP_CODE]"
done

echo "[node2] ALL NODE2 WORKERS READY. SSH session staying alive..."
# Tail suppressed — logs on NFS, monitor separately:
#   tail -f $RESULTS_DIR/ULTRA_AGG_4xTP4_RR_2NODE__worker{2,3}_<RUN_ID>.log
tail -f "$W3_LOG" > /dev/null 2>&1

ENDSSH
NODE2_SSH_PID=$!
log "  Node2 SSH background PID: $NODE2_SSH_PID"

# Poll node2 workers from node1 (SSH is already running in background)
wait_for_worker_remote "$NODE2_IP" "$WORKER_PORT_2" "worker-2 (node2)" "$WORKER_READY_TIMEOUT"
wait_for_worker_remote "$NODE2_IP" "$WORKER_PORT_3" "worker-3 (node2)" "$WORKER_READY_TIMEOUT"

STARTUP_SECS=$(( SECONDS - STARTUP_START ))
log "All 4 workers healthy. Total startup time: ${STARTUP_SECS}s"

# Wait for frontend and etcd registration
wait_for_worker "http://127.0.0.1:${HTTP_PORT}/health" "frontend"
log "Waiting 30s for all workers to register in etcd..."
sleep 30
log "All workers registered. Ready to sweep!"

# =============================================================================
# SECTION 10 — Rate sweep loop (workers stay alive across all steps)
# =============================================================================
echo "" >> "$SWEEP_SUMMARY"
echo "Rate Sweep — Nemotron Ultra NVFP4 AGG 4×TP4 Round-Robin — Run $RUN_ID" >> "$SWEEP_SUMMARY"
echo "ISL=${ISL}, OSL=${OSL} (fixed), max_model_len=${MAX_MODEL_LEN}, TTFT_LIMIT=${TTFT_LIMIT_MS}ms" >> "$SWEEP_SUMMARY"
echo "RATE | NUM_PROMPTS | TTFT_median(ms) | TTFT_p99(ms) | TPOT_p99(ms) | tok/s | verdict" \
    >> "$SWEEP_SUMMARY"
echo "-----+-------------+-----------------+--------------+--------------+-------+---------" \
    >> "$SWEEP_SUMMARY"

for STEP in "${RATE_SWEEP[@]}"; do
    NUM_PROMPTS=$(echo "$STEP" | cut -d',' -f1)
    RATE=$(       echo "$STEP" | cut -d',' -f2)
    MAX_CONC=$(   echo "$STEP" | cut -d',' -f3)

    log ""
    log "=========================================="
    log "--- Benchmark probe: RATE=$RATE  ISL=$ISL  OSL=$OSL  NUM_PROMPTS=$NUM_PROMPTS ---"
    log "=========================================="

    RESULT_FILE="$RESULTS_DIR/ULTRA_AGG_4xTP4_RR_2NODE__in${ISL}_out${OSL}__prm${NUM_PROMPTS}__rate${RATE}__${RUN_ID}.json"

    set +e
    vllm bench serve \
        --backend openai-chat \
        --model "$MODEL_NAME" \
        --dataset-name random \
        --random-input-len  "$ISL" \
        --random-output-len "$OSL" \
        --endpoint /v1/chat/completions \
        --host 127.0.0.1 \
        --port "$HTTP_PORT" \
        --num-prompts "$NUM_PROMPTS" \
        --request-rate "$RATE" \
        --max-concurrency "$MAX_CONC" \
        --save-result \
        --result-filename "$RESULT_FILE"
    BENCH_RC=$?
    set -e

    if [ "$BENCH_RC" -ne 0 ] || [ ! -f "$RESULT_FILE" ]; then
        log "WARNING: Benchmark failed for RATE=$RATE (RC=$BENCH_RC)"
        printf "%-8s | %-11s | %-15s | %-12s | %-12s | %-5s | %s\n" \
            "$RATE" "$NUM_PROMPTS" "N/A" "N/A" "N/A" "N/A" "BENCH_FAIL" \
            >> "$SWEEP_SUMMARY"
        log "Stopping sweep early due to benchmark failure."
        break
    fi

    # Parse key metrics from JSON
    median_ttft_ms=$(python3 -c "
import json, sys
d = json.load(open('$RESULT_FILE'))
v = d.get('median_ttft_ms') or d.get('mean_ttft_ms') or 0
print(f'{v:.1f}')
" 2>/dev/null || echo "0")

    p99_ttft_ms=$(python3 -c "
import json, sys
d = json.load(open('$RESULT_FILE'))
pct = d.get('percentiles_ttft_ms', {})
v = pct.get('99') or pct.get('p99') or 0
print(f'{v:.1f}')
" 2>/dev/null || echo "0")

    p99_tpot_ms=$(python3 -c "
import json, sys
d = json.load(open('$RESULT_FILE'))
pct = d.get('percentiles_tpot_ms', {})
v = pct.get('99') or pct.get('p99') or 0
print(f'{v:.1f}')
" 2>/dev/null || echo "0")

    output_toks=$(python3 -c "
import json
d = json.load(open('$RESULT_FILE'))
v = d.get('output_throughput', 0)
print(f'{v:.1f}')
" 2>/dev/null || echo "0")

    log "  TTFT median: ${median_ttft_ms}ms"
    log "  TTFT P99:    ${p99_ttft_ms}ms"
    log "  TPOT P99:    ${p99_tpot_ms}ms"
    log "  Output tok/s: ${output_toks}"

    # Check TTFT limit
    median_ttft_int=$(python3 -c "print(int(float('$median_ttft_ms')))" 2>/dev/null || echo "0")
    if [ "$median_ttft_int" -gt "$TTFT_LIMIT_MS" ]; then
        VERDICT="TTFT_LIMIT"
        log "  Median TTFT ${median_ttft_ms}ms exceeds limit ${TTFT_LIMIT_MS}ms — stopping sweep."
    else
        VERDICT="OK"
        log "  Verdict for RATE=$RATE: OK"
        log "  RATE=$RATE passed. Proceeding to next step..."
    fi

    printf "%-8s | %-11s | %-15s | %-12s | %-12s | %-5s | %s\n" \
        "$RATE" "$NUM_PROMPTS" "${median_ttft_ms}ms" "${p99_ttft_ms}ms" \
        "${p99_tpot_ms}ms" "$output_toks" "$VERDICT" \
        >> "$SWEEP_SUMMARY"

    if [ "$VERDICT" = "TTFT_LIMIT" ]; then
        break
    fi

    log ""
    log "=========================================="
done

# =============================================================================
# SECTION 11 — Summary
# =============================================================================
log ""
log "=========================================="
log "RATE SWEEP COMPLETE"
log ""
log "Results summary ($SWEEP_SUMMARY):"
cat "$SWEEP_SUMMARY"
log ""
log "Full result JSONs in: $RESULTS_DIR"
log "Worker logs in:       $RESULTS_DIR (ULTRA_AGG_4xTP4_RR_2NODE__*.log)"
log "=========================================="
log ""
log "=========================================="
log "CLEANUP: Shutting down all workers..."
log "=========================================="

# cleanup() handles teardown via EXIT trap
