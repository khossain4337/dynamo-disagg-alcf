#!/bin/bash
set -euo pipefail

# =============================================================================
# Nemotron Ultra NVFP4 — Dynamo AGG 4×TP4 — OSL Sweep — Single Entry Point
# Run this script ONLY on minerva-dgx-01. It SSHes to node2 automatically.
#
# USAGE:
#   bash interactive_single_entry_multinode_ultra_agg_4xtp4_rr_osl_sweep_N2.sh
#
# WHAT IT DOES:
#   1. Starts etcd, NATS, Dynamo frontend (round-robin) on node1
#   2. Starts workers SEQUENTIALLY to avoid FlashInfer JIT NFS race:
#        worker-0 (node1, GPU 0-3, NUMA 0) → wait healthy
#        worker-1 (node1, GPU 4-7, NUMA 1) → wait healthy
#        SSH → worker-2 (node2, GPU 0-3)   → wait healthy
#              worker-3 (node2, GPU 4-7)   → wait healthy
#   3. Sweeps OSL: 256 → 512 → 1024 → 2048 → 4096 → 8192
#      ISL=131,072 and RATE=0.05 fixed; max_model_len=139,264 (ISL + max OSL)
#      fixed for ALL steps — workers start ONCE, only benchmark reruns.
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
# Why OSL sweep?
#   The rate sweep (§32) showed AGG dominates DISAGG at ISL=131K, OSL=256.
#   DISAGG's decode-isolation advantage only matters when decode is expensive.
#   This sweep finds the OSL crossover where DISAGG pulls ahead on TPOT.
#   Mirrors the DISAGG OSL sweep config for direct comparison.
#
# [FACT] Workers start ONCE at max_model_len=147456 (ISL + max OSL=16384).
#        No restarts between OSL steps — saves ~5×640s = ~53min of startup.
# [FACT] Sequential startup: worker-0 → worker-1 → SSH(worker-2 → worker-3).
#        Each worker waits for healthy before the next fires.
#        Prevents FlashInfer JIT NFS race (all ranks writing same .so path).
# [FACT] RATE=0.05 fixed — rate-controlled (near-zero queue delay).
#        Matches DISAGG OSL sweep config for apples-to-apples comparison.
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
# SECTION 3 — Fixed configuration
# =============================================================================
MODEL_PATH="/home/hossainm/software/model-weights/hub/models--nvidia--NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4/snapshots/504c145dce0744d9a61fd458d11febb88aa8890c"
MODEL_NAME="nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4"
TP=4

HTTP_PORT=8000

WORKER_PORT_0=8081
WORKER_PORT_1=8082
WORKER_PORT_2=8083
WORKER_PORT_3=8084

GPUS_W0="0,1,2,3"
GPUS_W1="4,5,6,7"
GPUS_W2="0,1,2,3"
GPUS_W3="4,5,6,7"

# Fixed ISL and rate for all OSL steps
ISL=131072
RATE="0.05"
NUM_PROMPTS=5
MAX_CONCURRENCY=4
MAX_NUM_SEQS=32

# [FACT] MAX_MODEL_LEN = ISL + max(OSL) = 131072 + 16384 = 147456.
# Workers start ONCE at this value — covers all OSL steps without restart.
MAX_OSL=16384
MAX_MODEL_LEN=$(( ISL + MAX_OSL ))   # 147456

RESULTS_DIR="$HOME/software/vllm_efforts/benchmark_results"
WORKER_READY_TIMEOUT=2700
TTFT_LIMIT_MS=999999999

# =============================================================================
# SECTION 4 — OSL sweep definition
#
# Format: "OSL"
# ISL=131072, RATE=0.05, NUM_PROMPTS=5, MAX_CONCURRENCY=4 fixed above.
# MAX_MODEL_LEN=147456 fixed — workers do NOT restart between steps.
#
# [FACT] These 4 OSL values match the DISAGG OSL sweep (§24) exactly —
# apples-to-apples comparison. Intermediate values 512/1024/2048 were
# skipped in the DISAGG run and are skipped here for consistency.
# max_model_len per step = ISL + OSL (all under 262,144 ceiling):
#   OSL=256   → 131,328
#   OSL=4096  → 135,168
#   OSL=8192  → 139,264
#   OSL=16384 → 147,456  ← max, sets MAX_MODEL_LEN above
# =============================================================================
OSL_SWEEP=(
    "256"
    "4096"
    "8192"
    "16384"
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
SWEEP_SUMMARY="$RESULTS_DIR/ULTRA_AGG_4xTP4_RR_2NODE__osl_sweep_summary_${RUN_ID}.txt"

log "=========================================="
log "Nemotron Ultra NVFP4 — AGG 4×TP4 Round-Robin OSL Sweep"
log "ISL=${ISL} (fixed), RATE=${RATE} req/s (fixed), max_model_len=${MAX_MODEL_LEN}"
log "OSL steps: ${OSL_SWEEP[*]}"
log "Single entry point — node1 launches node2 via SSH"
log "Dynamo v1.3.0-dev.1"
log "=========================================="
log "Run ID:    $RUN_ID"
log "Model:     $MODEL_NAME"
log "Results:   $RESULTS_DIR"
log "Summary:   $SWEEP_SUMMARY"
log "Node1:     $NODE1_HOST ($NODE1_IP) — etcd, NATS, frontend, worker-0, worker-1"
log "Node2:     $NODE2_HOST ($NODE2_IP) — worker-2, worker-3 (auto-launched)"
log "OSL steps: ${#OSL_SWEEP[@]}"
log "Router:    Dynamo round-robin"
log "Startup:   SEQUENTIAL (w0→w1→w2→w3) — avoids FlashInfer JIT NFS race"
log "Workers start ONCE at max_model_len=${MAX_MODEL_LEN} — no restarts between OSL steps"
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
#             max_model_len=MAX_MODEL_LEN covers all OSL steps — no restarts.
# =============================================================================
W0_LOG="$RESULTS_DIR/ULTRA_AGG_4xTP4_RR_2NODE__worker0_${RUN_ID}.log"
W1_LOG="$RESULTS_DIR/ULTRA_AGG_4xTP4_RR_2NODE__worker1_${RUN_ID}.log"
W2_LOG="$RESULTS_DIR/ULTRA_AGG_4xTP4_RR_2NODE__worker2_${RUN_ID}.log"
W3_LOG="$RESULTS_DIR/ULTRA_AGG_4xTP4_RR_2NODE__worker3_${RUN_ID}.log"
touch "$W0_LOG" "$W1_LOG" "$W2_LOG" "$W3_LOG"

STARTUP_START=$SECONDS

# ---------------------------------------------------------------------------
# Worker-0 (node1, GPUs 0-3, NUMA 0)
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
wait_for_worker "http://127.0.0.1:${WORKER_PORT_0}/health" "worker-0 (node1)"

# ---------------------------------------------------------------------------
# Worker-1 (node1, GPUs 4-7, NUMA 1)
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
wait_for_worker "http://127.0.0.1:${WORKER_PORT_1}/health" "worker-1 (node1)"

# ---------------------------------------------------------------------------
# SSH to node2 — fires after worker-1 is healthy.
# Node2 starts worker-2, waits for it, then starts worker-3.
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

pkill -9 -if "dynamo.vllm|vllm" 2>/dev/null || true
sleep 2

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

W2_START=$SECONDS
while (( SECONDS - W2_START < 2700 )); do
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
        --max-time 3 "http://127.0.0.1:${WORKER_PORT_2}/health") || HTTP_CODE="000"
    if [ "$HTTP_CODE" = "200" ]; then
        echo "[node2] worker-2 ready after $(( SECONDS - W2_START ))s ✓"; break
    fi
    sleep 10
    echo "[node2] ...worker-2 still starting ($(( SECONDS - W2_START ))s) [HTTP: $HTTP_CODE]"
done

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

W3_START=$SECONDS
while (( SECONDS - W3_START < 2700 )); do
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
        --max-time 3 "http://127.0.0.1:${WORKER_PORT_3}/health") || HTTP_CODE="000"
    if [ "$HTTP_CODE" = "200" ]; then
        echo "[node2] worker-3 ready after $(( SECONDS - W3_START ))s ✓"; break
    fi
    sleep 10
    echo "[node2] ...worker-3 still starting ($(( SECONDS - W3_START ))s) [HTTP: $HTTP_CODE]"
done

echo "[node2] ALL NODE2 WORKERS READY. SSH session staying alive..."
tail -f "$W3_LOG" > /dev/null 2>&1

ENDSSH
NODE2_SSH_PID=$!
log "  Node2 SSH background PID: $NODE2_SSH_PID"

wait_for_worker_remote "$NODE2_IP" "$WORKER_PORT_2" "worker-2 (node2)" "$WORKER_READY_TIMEOUT"
wait_for_worker_remote "$NODE2_IP" "$WORKER_PORT_3" "worker-3 (node2)" "$WORKER_READY_TIMEOUT"

STARTUP_SECS=$(( SECONDS - STARTUP_START ))
log "All 4 workers healthy. Total startup time: ${STARTUP_SECS}s"

wait_for_worker "http://127.0.0.1:${HTTP_PORT}/health" "frontend"
log "Waiting 30s for all workers to register in etcd..."
sleep 30
log "All workers registered. Ready to sweep!"

# =============================================================================
# SECTION 10 — OSL sweep loop (workers stay alive across all steps)
# =============================================================================
echo ""                                                                          >> "$SWEEP_SUMMARY"
echo "OSL Sweep — Nemotron Ultra NVFP4 AGG 4×TP4 Round-Robin — Run $RUN_ID"   >> "$SWEEP_SUMMARY"
echo "ISL=${ISL} (fixed), RATE=${RATE} req/s (fixed), max_model_len=${MAX_MODEL_LEN}" >> "$SWEEP_SUMMARY"
echo "NUM_PROMPTS=${NUM_PROMPTS}, MAX_CONCURRENCY=${MAX_CONCURRENCY}"           >> "$SWEEP_SUMMARY"
echo "OSL | TTFT_median(ms) | TTFT_p99(ms) | TPOT_p99(ms) | tok/s | verdict"  >> "$SWEEP_SUMMARY"
echo "----+-----------------+--------------+--------------+-------+---------"   >> "$SWEEP_SUMMARY"

for OSL in "${OSL_SWEEP[@]}"; do

    log ""
    log "=========================================="
    log "--- Benchmark probe: OSL=$OSL  ISL=$ISL  RATE=$RATE  NUM_PROMPTS=$NUM_PROMPTS ---"
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
        --max-concurrency "$MAX_CONCURRENCY" \
        --save-result \
        --result-filename "$RESULT_FILE"
    BENCH_RC=$?
    set -e

    if [ "$BENCH_RC" -ne 0 ] || [ ! -f "$RESULT_FILE" ]; then
        log "WARNING: Benchmark failed for OSL=$OSL (RC=$BENCH_RC)"
        printf "%-6s | %-15s | %-12s | %-12s | %-5s | %s\n" \
            "$OSL" "N/A" "N/A" "N/A" "N/A" "BENCH_FAIL" \
            >> "$SWEEP_SUMMARY"
        log "Stopping sweep early due to benchmark failure."
        break
    fi

    # ------------------------------------------------------------------
    # Parse key metrics from JSON.
    # [FACT — BUG FIX] Summary file previously showed P99=0.0ms because
    # the parser looked only for nested dict keys ('percentiles_ttft_ms'.99)
    # while vllm bench serve writes flat keys (p99_ttft_ms, p99_tpot_ms).
    # Fix: try flat keys first, fall back to nested dict, then mean.
    # ------------------------------------------------------------------
    median_ttft_ms=$(python3 -c "
import json
d = json.load(open('$RESULT_FILE'))
v = (d.get('median_ttft_ms')
     or d.get('mean_ttft_ms')
     or 0)
print(f'{float(v):.1f}')
" 2>/dev/null || echo "0")

    p99_ttft_ms=$(python3 -c "
import json
d = json.load(open('$RESULT_FILE'))
# Try flat key first (vllm bench serve standard output)
v = d.get('p99_ttft_ms')
# Fall back to nested percentiles dict
if v is None:
    pct = d.get('percentiles_ttft_ms', {})
    v = pct.get('99') or pct.get('p99') or pct.get(99)
print(f'{float(v or 0):.1f}')
" 2>/dev/null || echo "0")

    p99_tpot_ms=$(python3 -c "
import json
d = json.load(open('$RESULT_FILE'))
v = d.get('p99_tpot_ms')
if v is None:
    pct = d.get('percentiles_tpot_ms', {})
    v = pct.get('99') or pct.get('p99') or pct.get(99)
print(f'{float(v or 0):.1f}')
" 2>/dev/null || echo "0")

    output_toks=$(python3 -c "
import json
d = json.load(open('$RESULT_FILE'))
v = d.get('output_throughput', 0)
print(f'{float(v):.1f}')
" 2>/dev/null || echo "0")

    log "  TTFT median: ${median_ttft_ms}ms"
    log "  TTFT P99:    ${p99_ttft_ms}ms"
    log "  TPOT P99:    ${p99_tpot_ms}ms"
    log "  Output tok/s: ${output_toks}"

    median_ttft_int=$(python3 -c "print(int(float('$median_ttft_ms')))" 2>/dev/null || echo "0")
    if [ "$median_ttft_int" -gt "$TTFT_LIMIT_MS" ]; then
        VERDICT="TTFT_LIMIT"
        log "  Median TTFT ${median_ttft_ms}ms exceeds limit ${TTFT_LIMIT_MS}ms — stopping sweep."
    else
        VERDICT="OK"
        log "  Verdict for OSL=$OSL: OK"
        log "  OSL=$OSL passed. Proceeding to next step..."
    fi

    printf "%-6s | %-15s | %-12s | %-12s | %-5s | %s\n" \
        "$OSL" "${median_ttft_ms}ms" "${p99_ttft_ms}ms" \
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
log "OSL SWEEP COMPLETE"
log ""
log "Results summary ($SWEEP_SUMMARY):"
cat "$SWEEP_SUMMARY"
log ""
log "Full result JSONs in: $RESULTS_DIR"
log "Worker logs in:       $RESULTS_DIR (ULTRA_AGG_4xTP4_RR_2NODE__*.log)"
log "=========================================="

# cleanup() handles teardown via EXIT trap
