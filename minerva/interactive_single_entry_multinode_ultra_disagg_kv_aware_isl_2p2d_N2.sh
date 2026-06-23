#!/bin/bash
set -euo pipefail

# =============================================================================
# Nemotron Ultra NVFP4 — Dynamo DISAGG 2P:2D — Single Entry Point
# Run this script ONLY on minerva-dgx-01. It SSHes to node2 automatically.
#
# USAGE:
#   bash single_entry_ultra_disagg_2p2d_isl_sweep.sh
#
# WHAT IT DOES:
#   1. Starts etcd, NATS, frontend, decode-1, prefill-1 on node1
#   2. SSHes to node2 and starts decode-2, prefill-2 there (background SSH)
#   3. Waits for all 4 workers to be healthy
#   4. Sweeps ISL: 32K → 64K → 128K → 262K → 524K → 1M
#      At each ISL step: restarts all workers with new MAX_MODEL_LEN + correct
#      MAX_NUM_SEQS, then runs a small benchmark probe
#   5. Stops at the first ISL where median TTFT > TTFT_LIMIT_MS or startup OOM
#   6. On exit (Ctrl+C or completion): kills node1 processes AND kills node2
#      processes via SSH
#
# Topology (2P:2D, TP=4 per worker):
#   minerva-dgx-01 (node1) — run this script here:
#     etcd        → 0.0.0.0:2379, advertised at 10.125.13.11:2379
#     NATS        → 0.0.0.0:4222
#     Frontend    → HTTP 8000
#     Decode-1    → GPUs 4,5,6,7  PORT=8081  NUMA 1
#     Prefill-1   → GPUs 0,1,2,3  PORT=8082  NUMA 0  NIXL_SC=20097
#
#   minerva-dgx-02 (node2) — launched automatically via SSH from this script:
#     Decode-2    → GPUs 4,5,6,7  PORT=8084  NUMA 1
#     Prefill-2   → GPUs 0,1,2,3  PORT=8083  NUMA 0  NIXL_SC=20098
#
# KV transfer paths:
#   Prefill-1 → Decode-1: NVLink  ~600 GB/s  (intra-node, fast path)
#   Prefill-2 → Decode-2: NVLink  ~600 GB/s  (intra-node, fast path)
#   Prefill-1 → Decode-2: IB NDR  ~25  GB/s  (inter-node, slow path)
#   Prefill-2 → Decode-1: IB NDR  ~25  GB/s  (inter-node, slow path)
#
# [FACT] All environment variables are set explicitly inside the SSH heredoc.
#        No reliance on .bashrc, .bash_profile, or login shell on node2.
#        This makes node2 startup fully deterministic and debuggable.
#
# [FACT] SSM patch (nixl/worker.py) is on NFS — applies to both nodes.
# [FACT] FlashInfer JIT cache (~/.cache/flashinfer/) is on NFS — warm on
#        repeat runs. Cold first run per ISL can take up to 15 min per worker.
# =============================================================================

tstamp() { date +"%Y-%m-%d-%H%M%S"; }
log()    { echo "[$(tstamp)] $*"; }

# =============================================================================
# SECTION 1 — Cluster topology (edit if node IPs change)
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
# SECTION 2 — Environment (node1, set here explicitly)
# =============================================================================
export HF_TOKEN=$(cat ~/.hf_token)
export HF_HOME="/home/hossainm/software/model-weights"

export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128
# CRITICAL: exclude both node IPs and hostnames, or ALCF proxy intercepts
# etcd/NATS TCP silently and breaks cross-node worker registration
export no_proxy="127.0.0.1,localhost,${NODE1_IP},${NODE2_IP},${NODE1_HOST},${NODE2_HOST}"
export NO_PROXY="127.0.0.1,localhost,${NODE1_IP},${NODE2_IP},${NODE1_HOST},${NODE2_HOST}"

export PYTHONHASHSEED=0
export VLLM_SSM_CONV_STATE_LAYOUT=DS
export ETCD_ENDPOINTS="http://${NODE1_IP}:2379"
export NATS_SERVER="nats://${NODE1_IP}:4222"

# Conda activation
source /home/hossainm/miniforge3/bin/activate
eval "$(conda shell.bash hook)"
conda activate /home/hossainm/software/envs/conda_envs/dynamo_vllm_1.3.0_dev1

# =============================================================================
# SECTION 3 — Fixed configuration (does NOT change between ISL steps)
# =============================================================================
MODEL="nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4"
TP=4

HTTP_PORT=8000

# Worker system ports — unique per worker across all nodes
DECODE_PORT_1=8081    # node1 decode
PREFILL_PORT_1=8082   # node1 prefill
PREFILL_PORT_2=8083   # node2 prefill
DECODE_PORT_2=8084    # node2 decode

# NIXL side channel ports — prefill workers only, unique per worker
NIXL_SC_1=20097
NIXL_SC_2=20098

# KV events ZMQ ports — prefill workers only, unique per worker
KV_EVT_1=20081
KV_EVT_2=20082

# GPU assignments (same physical layout on both nodes)
GPUS_DECODE_1="4,5,6,7"   # node1 NUMA 1
GPUS_PREFILL_1="0,1,2,3"  # node1 NUMA 0
GPUS_DECODE_2="4,5,6,7"   # node2 NUMA 1
GPUS_PREFILL_2="0,1,2,3"  # node2 NUMA 0

RESULTS_DIR="$HOME/software/vllm_efforts/benchmark_results"
WORKER_READY_TIMEOUT=900   # 15 min per worker (covers cold FlashInfer JIT)

# ISL sweep stop condition (ms). If median TTFT exceeds this, stop the sweep.
# [MY RECOMMENDATION] 30,000ms = 30s is a reasonable "practically useless" ceiling.
# Adjust based on your tolerance. Comment out to run all ISL steps regardless.
TTFT_LIMIT_MS=999999999

# =============================================================================
# SECTION 4 — Ceiling experiment definition
#
# Format per entry: "ISL,OSL,MAX_NUM_SEQS,NUM_PROMPTS,REQUEST_RATE,MAX_CONCURRENCY"
#
# NOTE: OSL is now per-experiment (not a fixed global) because these ceiling
# experiments use different OSL values. The script reads OSL from each entry.
#
# -------------------------------------------------------------------------
# ARCHITECTURAL CEILING — HARD LIMIT: ISL + OSL ≤ 262,144 tokens
# -------------------------------------------------------------------------
# [FACT — confirmed from config.json, June 18, 2026]
# Ultra NVFP4 config.json shows:
#   max_position_embeddings: 262144
#   rope_scaling: None          ← vanilla RoPE, NO long-context extension
#   rope_theta: 10000           ← standard pretraining value
#
# This means attention layers use vanilla RoPE with no YaRN/LongRoPE/ALiBi.
# Positions beyond 262,144 will produce NaN in attention layers.
# VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 bypasses the vLLM guard but does NOT fix
# the underlying RoPE math — results beyond 262,144 are numerically unsafe.
#
# The "1M context" claim in the Ultra model card likely refers to the Mamba-2
# SSM layers which have no positional encoding and handle arbitrary length.
# The hybrid architecture caps at 262,144 due to the attention layers.
#
# SAFE RULE: set --max-model-len = ISL + OSL ≤ 262,144 in all worker launches.
# -------------------------------------------------------------------------
#
# MAX_NUM_SEQS: at ISL+OSL=262,144, each seq needs ceil(262144/8304)=32 blocks.
# KV pool has ~1,685 blocks total → max_concurrent_seqs = 1685/32 = 52.
# We cap at 8 (conservative) since these are latency probes not throughput runs.
#
# REQUEST RATE DERIVATION (from linear fit TTFT = 160.5ms × ISL/1000 − 1944ms):
#
#   Experiment 1: ISL=261,888, OSL=256
#     pred_TTFT   = 160.5 × 261.888 − 1944 = ~40,100ms = ~40s
#     T_decode    = 256 × 8ms = 2,048ms = ~2s
#     T_total     = ~42s per request
#     mu_prefill  = 2 workers / 40s = 0.050 req/s
#     safe rate   = 50% × mu = 0.025 req/s → floor to 0.05 req/s
#     num_prompts = 5 (at 0.05 req/s → ~100s benchmark, manageable)
#     est. benchmark duration: ~15 min
#
#   Experiment 2: ISL=260,096, OSL=2,048
#     pred_TTFT   = 160.5 × 260.096 − 1944 = ~39,800ms = ~40s
#     T_decode    = 2048 × 8ms = 16,384ms = ~16s
#     T_total     = ~56s per request
#     mu_prefill  = 2 workers / 40s = 0.050 req/s
#     safe rate   = 50% × mu = 0.025 req/s → floor to 0.05 req/s
#     num_prompts = 5 (at 0.05 req/s → ~112s benchmark, manageable)
#     est. benchmark duration: ~17 min
#
# LONGER EXPERIMENTS (NOT run tonight — wall-clock too long):
#
#   Experiment 3: ISL=196,608, OSL=65,536
#     pred_TTFT   = 160.5 × 196.608 − 1944 = ~29,600ms = ~30s
#     T_decode    = 65536 × 8ms = 524,288ms = ~524s = ~8.7 min per request
#     T_total     = ~554s per request
#     num_prompts = 5 → est. benchmark duration: ~46 min
#     mu_prefill  = 2 / 30s = 0.067 req/s → safe rate = 0.05 req/s
#
#   Experiment 4: ISL=131,072, OSL=131,072
#     pred_TTFT   = 160.5 × 131.072 − 1944 = ~19,100ms = ~19s
#     T_decode    = 131072 × 8ms = 1,048,576ms = ~1049s = ~17.5 min per request
#     T_total     = ~1068s per request
#     num_prompts = 5 → est. benchmark duration: ~89 min
#     mu_prefill  = 2 / 19s = 0.105 req/s → safe rate = 0.05 req/s
#
# =============================================================================

# ISL, OSL, MAX_NUM_SEQS, NUM_PROMPTS, RATE, MAX_CONCURRENCY
ISL_SWEEP=(
    #"261888,256,52,5,0.05,8"
    #"260096,2048,52,5,0.05,8"
    #"131072,256,32,5,0.05,4"
    "32768,256,32,5,0.05,4"
    "65536,256,32,5,0.05,4"
    # --- LONGER RUNS — see duration estimates above before uncommenting ---
    # "196608,65536,52,5,0.05,8"    # ~46 min benchmark
    # "131072,131072,52,5,0.05,8"   # ~89 min benchmark
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
    # Poll a remote worker health endpoint FROM node1 using the node2 IP.
    # Node1 cannot poll 127.0.0.1:PORT for a node2 process — it must use NODE2_IP.
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

check_gpu_memory_local() {
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
}

kill_leftovers_local() {
    log "Killing any leftover dynamo/vllm processes on node1..."
    pkill -9 -if "dynamo.vllm|vllm" 2>/dev/null || true  # frontend excluded — managed separately
    sleep 3
    log "  Node1 leftovers cleared."
}

kill_leftovers_remote() {
    log "Killing any leftover dynamo/vllm processes on node2 via SSH..."
    ssh "$NODE2_HOST" "pkill -9 -if 'dynamo.vllm|vllm' 2>/dev/null || true" || true
    sleep 3
    log "  Node2 leftovers cleared."
}

# =============================================================================
# SECTION 6 — Cleanup trap (runs on EXIT, Ctrl+C, or error)
#
# [IMPORTANT] kill 0 sends SIGKILL to every process in the current process group
# — that includes all background workers started by this script on node1.
# For node2, we SSH in and kill explicitly by process name.
# We store the SSH background PID so we can also kill the SSH connection itself.
# =============================================================================
NODE2_SSH_PID=""

cleanup() {
    log ""
    log "=========================================="
    log "CLEANUP: Shutting down all workers..."
    log "=========================================="

    # Kill node1 workers (decode-1, prefill-1, frontend, etcd, NATS)
    log "Killing node1 processes (kill 0)..."
    kill 0 2>/dev/null || true

    # Kill node2 workers via SSH
    log "Killing node2 workers via SSH..."
    ssh "$NODE2_HOST" \
        "pkill -9 -if 'dynamo.vllm|vllm' 2>/dev/null || true; echo 'node2 kill done'" \
        2>/dev/null || log "  (SSH to node2 during cleanup failed — may already be dead)"

    # Kill the background SSH process that is tailing node2 logs
    if [ -n "$NODE2_SSH_PID" ]; then
        kill "$NODE2_SSH_PID" 2>/dev/null || true
        log "  Background SSH (PID $NODE2_SSH_PID) killed."
    fi

    # Wait a moment for port release before exiting
    sleep 3
    log "Cleanup done. Goodbye."
}
trap cleanup EXIT

# =============================================================================
# SECTION 7 — Pre-flight checks
# =============================================================================
mkdir -p "$RESULTS_DIR"
RUN_ID="$(tstamp)"
SWEEP_SUMMARY="$RESULTS_DIR/ULTRA_DISAGG_2P2D_KV_2NODE__isl_sweep_summary_${RUN_ID}.txt"

log "=========================================="
log "Nemotron Ultra NVFP4 — DISAGG 2P:2D ISL Sweep"
log "Single entry point — node1 launches node2 via SSH"
log "Dynamo v1.3.0-dev.1"
log "=========================================="
log "Run ID:    $RUN_ID"
log "Model:     $MODEL"
log "Results:   $RESULTS_DIR"
log "Summary:   $SWEEP_SUMMARY"
log "Node1:     $NODE1_HOST ($NODE1_IP) — etcd, NATS, frontend, decode-1, prefill-1"
log "Node2:     $NODE2_HOST ($NODE2_IP) — decode-2, prefill-2 (auto-launched)"
log "ISL steps: ${#ISL_SWEEP[@]}"
log "OSL: per-experiment (see ISL_SWEEP definition)"
log "TTFT limit: ${TTFT_LIMIT_MS}ms (stop sweep if exceeded)"
log "=========================================="

# SSM patch must exist
check_ssm_patch

# Confirm node2 is reachable via SSH before we start anything
log "Checking SSH to node2..."
if ! ssh -o ConnectTimeout=10 "$NODE2_HOST" "hostname" > /dev/null 2>&1; then
    log "ERROR: Cannot SSH to $NODE2_HOST. Is the PBS co-allocation active?" >&2
    exit 1
fi
log "  SSH to $NODE2_HOST: OK ✓"

# Check node2 GPU memory before starting
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
#             These stay up for the ENTIRE sweep — no restart between ISL steps
# =============================================================================

ETCD_LOG="$RESULTS_DIR/ULTRA_DISAGG_2P2D_KV_2NODE__etcd_${RUN_ID}.log"
NATS_LOG="$RESULTS_DIR/ULTRA_DISAGG_2P2D_KV_2NODE__nats_${RUN_ID}.log"
FRONTEND_LOG="$RESULTS_DIR/ULTRA_DISAGG_2P2D_KV_2NODE__frontend_${RUN_ID}.log"

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

# [NOTE] Frontend is NOT started here. It is started fresh at the top of
# each ISL step (section 9b) so it gets a clean state per step and is never
# accidentally killed by the between-step worker cleanup.

# =============================================================================
# SECTION 9 — ISL sweep loop
#
# For each ISL step:
#   a) Kill and restart all 4 workers with new MAX_MODEL_LEN + MAX_NUM_SEQS
#   b) Launch node2 workers via SSH heredoc (fully explicit env, no .bashrc)
#   c) Wait for all 4 workers healthy
#   d) Run benchmark probe
#   e) Check TTFT against limit — stop sweep if exceeded
# =============================================================================

echo "" >> "$SWEEP_SUMMARY"
echo "ISL Sweep — Nemotron Ultra NVFP4 DISAGG 2P:2D — Run $RUN_ID" >> "$SWEEP_SUMMARY"
echo "OSL=per-experiment, TTFT_LIMIT=${TTFT_LIMIT_MS}ms" >> "$SWEEP_SUMMARY"
echo "ISL | MAX_NUM_SEQS | Startup | TTFT_median(ms) | TTFT_p99(ms) | TPOT_p99(ms) | tok/s | verdict" \
    >> "$SWEEP_SUMMARY"
echo "----+-------------+---------+-----------------+--------------+--------------+-------+---------" \
    >> "$SWEEP_SUMMARY"

SWEEP_STOPPED_EARLY=0

for STEP in "${ISL_SWEEP[@]}"; do
    ISL=$(         echo "$STEP" | cut -d',' -f1)
    OSL=$(         echo "$STEP" | cut -d',' -f2)
    MAX_NUM_SEQS=$(echo "$STEP" | cut -d',' -f3)
    NUM_PROMPTS=$( echo "$STEP" | cut -d',' -f4)
    RATE=$(        echo "$STEP" | cut -d',' -f5)
    MAX_CONC=$(    echo "$STEP" | cut -d',' -f6)

    log ""
    log "=========================================="
    log "ISL STEP: ISL=$ISL  MAX_NUM_SEQS=$MAX_NUM_SEQS  OSL=$OSL"
    log "          NUM_PROMPTS=$NUM_PROMPTS  RATE=$RATE  MAX_CONC=$MAX_CONC"
    log "=========================================="

    # ------------------------------------------------------------------
    # 9a — Kill previous workers (if any) before restarting with new ISL
    # ------------------------------------------------------------------
    log "Stopping any running workers from previous ISL step..."
    kill_leftovers_local
    kill_leftovers_remote

    # Wait for HTTP port 8000 to be fully released before relaunching frontend
    log "Waiting for port 8000 to be free..."
    for i in {1..30}; do
        if ! ss -tlnp 2>/dev/null | grep -q ":${HTTP_PORT}"; then
            log "  Port $HTTP_PORT free ✓"; break
        fi
        sleep 2
        log "  ...port $HTTP_PORT still in use (attempt $i/30)"
    done

    # ------------------------------------------------------------------
    # 9b — Restart frontend for this ISL step
    #
    # [FACT] Frontend is killed by kill_leftovers_local (workers only now,
    # but we restart frontend here anyway to get a clean state per ISL step).
    # [FACT] Frontend does not depend on max-model-len — it is stateless and
    # just routes requests. But restarting it ensures clean etcd registration
    # state between ISL steps.
    # ------------------------------------------------------------------
    STEP_FRONTEND_LOG="$RESULTS_DIR/ULTRA_DISAGG_2P2D_KV_2NODE__frontend__ISL${ISL}_OSL${OSL}__${RUN_ID}.log"
    log "Starting Dynamo frontend (kv router) for ISL=${ISL}..."
    DYN_ROUTER_MODE=kv \
    DYN_ROUTER_ASSUME_KV_REUSE=false \
    DYN_HTTP_PORT=$HTTP_PORT \
    python -m dynamo.frontend \
        > "$STEP_FRONTEND_LOG" 2>&1 &
    FRONTEND_PID=$!
    log "  Frontend PID: $FRONTEND_PID"
    sleep 5

    # ------------------------------------------------------------------
    # 9b2 — Log files for this ISL step
    # ------------------------------------------------------------------
    STEP_TAG="ISL${ISL}_OSL${OSL}"
    D1_LOG="$RESULTS_DIR/ULTRA_DISAGG_2P2D_KV_2NODE__decode1__${STEP_TAG}__${RUN_ID}.log"
    P1_LOG="$RESULTS_DIR/ULTRA_DISAGG_2P2D_KV_2NODE__prefill1__${STEP_TAG}__${RUN_ID}.log"
    D2_LOG="$RESULTS_DIR/ULTRA_DISAGG_2P2D_KV_2NODE__decode2__${STEP_TAG}__${RUN_ID}.log"
    P2_LOG="$RESULTS_DIR/ULTRA_DISAGG_2P2D_KV_2NODE__prefill2__${STEP_TAG}__${RUN_ID}.log"
    touch "$D1_LOG" "$P1_LOG" "$D2_LOG" "$P2_LOG"

    STEP_START=$SECONDS

    # ------------------------------------------------------------------
    # 9c+9d — Start node1 AND node2 workers IN PARALLEL
    #
    # Within each node: decode starts first, prefill starts after decode is
    # healthy. [FACT] This ordering is mandatory — prefill registers its NIXL
    # KV endpoint and the decode worker must be ready to receive it.
    # Across nodes: no ordering constraint. Node2 SSH fires immediately after
    # node1 decode launches. Both nodes load weights in parallel, saving ~300s
    # per ISL step. All health checks happen together at section 9e below.
    # ------------------------------------------------------------------

    # Decode-1 (node1, GPUs 4-7, NUMA 1) — fire, don't wait yet
    log "Starting decode-1 (GPUs $GPUS_DECODE_1, NUMA 1, PORT $DECODE_PORT_1)..."
    VLLM_ENGINE_STARTUP_TIMEOUT=600 \
    CUDA_VISIBLE_DEVICES=$GPUS_DECODE_1 \
    DYN_SYSTEM_PORT=$DECODE_PORT_1 \
    ETCD_ENDPOINTS="http://${NODE1_IP}:2379" \
    NATS_SERVER="nats://${NODE1_IP}:4222" \
    numactl --cpunodebind=1 --membind=1 \
    python3 -m dynamo.vllm \
        --model "$MODEL" \
        --served-model-name "$MODEL" \
        --disaggregation-mode decode \
        --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
        --tensor-parallel-size $TP \
        --max-model-len $(( ISL + OSL )) \
        --max-num-seqs $MAX_NUM_SEQS \
        --no-disable-hybrid-kv-cache-manager \
        --trust-remote-code \
        --dyn-reasoning-parser nemotron_v3 \
        > "$D1_LOG" 2>&1 &
    DECODE1_PID=$!
    log "  decode-1 PID: $DECODE1_PID"

    # ------------------------------------------------------------------
    # SSH to node2 fires HERE — in parallel with node1 decode loading.
    # Node2 internally waits for decode-2 before starting prefill-2.
    # Node1 prefill-1 starts below after we confirm decode-1 is healthy.
    #
    # DESIGN NOTES (read before editing):
    # - The heredoc is quoted (<<'ENDSSH') so that $VARIABLES inside are
    #   evaluated on node2's shell, not expanded here on node1.
    #   Exception: we pass node1-side values we WANT expanded as arguments.
    # - We set every env variable explicitly. No sourcing of .bashrc.
    #   conda is activated by sourcing the miniforge init script directly.
    # - We redirect node2 logs to NFS paths so node1 can tail them.
    # - The SSH runs in background (&). NODE2_SSH_PID captures it for cleanup.
    # - decode-2 starts before prefill-2 on node2 (same rule as node1).
    # - Node2 workers bind to 127.0.0.1 for their health endpoints;
    #   node1 polls them at NODE2_IP:PORT (see wait_for_worker_remote).
    # ------------------------------------------------------------------
    ssh "$NODE2_HOST" \
        NODE1_IP="$NODE1_IP" \
        NODE2_IP="$NODE2_IP" \
        NODE1_HOST="$NODE1_HOST" \
        NODE2_HOST="$NODE2_HOST" \
        MODEL="$MODEL" \
        ISL="$ISL" \
        OSL="$OSL" \
        MAX_NUM_SEQS="$MAX_NUM_SEQS" \
        TP="$TP" \
        DECODE_PORT_2="$DECODE_PORT_2" \
        PREFILL_PORT_2="$PREFILL_PORT_2" \
        NIXL_SC_2="$NIXL_SC_2" \
        KV_EVT_2="$KV_EVT_2" \
        GPUS_DECODE_2="$GPUS_DECODE_2" \
        GPUS_PREFILL_2="$GPUS_PREFILL_2" \
        D2_LOG="$D2_LOG" \
        P2_LOG="$P2_LOG" \
        bash <<'ENDSSH' &

# ---- Everything below runs on node2 ----
# No .bashrc sourced. All env is explicit.

set -euo pipefail

# Proxy — identical to node1
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128
export no_proxy="127.0.0.1,localhost,${NODE1_IP},${NODE2_IP},${NODE1_HOST},${NODE2_HOST}"
export NO_PROXY="127.0.0.1,localhost,${NODE1_IP},${NODE2_IP},${NODE1_HOST},${NODE2_HOST}"

# Model env
export HF_TOKEN=$(cat ~/.hf_token)
export HF_HOME="/home/hossainm/software/model-weights"
export PYTHONHASHSEED=0
export VLLM_SSM_CONV_STATE_LAYOUT=DS

# Dynamo service discovery — points at node1
export ETCD_ENDPOINTS="http://${NODE1_IP}:2379"
export NATS_SERVER="nats://${NODE1_IP}:4222"

# Conda — source miniforge init directly, do NOT rely on .bashrc
source /home/hossainm/miniforge3/bin/activate
eval "$(conda shell.bash hook)"
conda activate /home/hossainm/software/envs/conda_envs/dynamo_vllm_1.3.0_dev1

echo "[node2] Environment ready. Conda env: $(conda info --envs | grep '*' | awk '{print $1}')"
echo "[node2] Python: $(which python3)"
echo "[node2] ISL=$ISL  MAX_NUM_SEQS=$MAX_NUM_SEQS  DECODE_PORT=$DECODE_PORT_2  PREFILL_PORT=$PREFILL_PORT_2"

# Confirm etcd on node1 is reachable before starting workers
for i in {1..12}; do
    if curl -sf "http://${NODE1_IP}:2379/health" > /dev/null 2>&1; then
        echo "[node2] etcd reachable ✓"; break
    fi
    if [ "$i" -eq 12 ]; then
        echo "[node2] ERROR: Cannot reach etcd at ${NODE1_IP}:2379 after 60s" >&2
        exit 1
    fi
    sleep 5
    echo "[node2] ...etcd not reachable yet (attempt $i/12)"
done

# Kill any leftovers on node2 from a previous run
pkill -9 -if "dynamo.vllm|vllm" 2>/dev/null || true
sleep 2

# Decode-2 (GPUs 4-7, NUMA 1) — starts first on node2
echo "[node2] Starting decode-2 (GPUs $GPUS_DECODE_2, NUMA 1, PORT $DECODE_PORT_2)..."
VLLM_ENGINE_STARTUP_TIMEOUT=600 \
CUDA_VISIBLE_DEVICES=$GPUS_DECODE_2 \
DYN_SYSTEM_PORT=$DECODE_PORT_2 \
ETCD_ENDPOINTS="http://${NODE1_IP}:2379" \
NATS_SERVER="nats://${NODE1_IP}:4222" \
numactl --cpunodebind=1 --membind=1 \
python3 -m dynamo.vllm \
    --model "$MODEL" \
    --served-model-name "$MODEL" \
    --disaggregation-mode decode \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
    --tensor-parallel-size $TP \
    --max-model-len $(( ISL + OSL )) \
    --max-num-seqs $MAX_NUM_SEQS \
    --no-disable-hybrid-kv-cache-manager \
    --trust-remote-code \
    --dyn-reasoning-parser nemotron_v3 \
    >> "$D2_LOG" 2>&1 &
DECODE2_PID=$!
echo "[node2] decode-2 PID: $DECODE2_PID"

# Wait for decode-2 locally (127.0.0.1 is fine here — we are ON node2)
D2_START=$SECONDS
while (( SECONDS - D2_START < 900 )); do
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
        --max-time 3 "http://127.0.0.1:${DECODE_PORT_2}/health") || HTTP_CODE="000"
    if [ "$HTTP_CODE" = "200" ]; then
        echo "[node2] decode-2 ready after $(( SECONDS - D2_START ))s ✓"; break
    fi
    sleep 10
    echo "[node2] ...decode-2 still starting ($(( SECONDS - D2_START ))s) [HTTP: $HTTP_CODE]"
done

# Prefill-2 (GPUs 0-3, NUMA 0) — starts after decode-2 is healthy
echo "[node2] Starting prefill-2 (GPUs $GPUS_PREFILL_2, NUMA 0, PORT $PREFILL_PORT_2)..."
VLLM_ENGINE_STARTUP_TIMEOUT=600 \
CUDA_VISIBLE_DEVICES=$GPUS_PREFILL_2 \
DYN_SYSTEM_PORT=$PREFILL_PORT_2 \
VLLM_NIXL_SIDE_CHANNEL_PORT=$NIXL_SC_2 \
ETCD_ENDPOINTS="http://${NODE1_IP}:2379" \
NATS_SERVER="nats://${NODE1_IP}:4222" \
numactl --cpunodebind=0 --membind=0 \
python3 -m dynamo.vllm \
    --model "$MODEL" \
    --served-model-name "$MODEL" \
    --disaggregation-mode prefill \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
    --kv-events-config "{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:${KV_EVT_2}\",\"enable_kv_cache_events\":true}" \
    --tensor-parallel-size $TP \
    --max-model-len $(( ISL + OSL )) \
    --max-num-seqs $MAX_NUM_SEQS \
    --no-disable-hybrid-kv-cache-manager \
    --trust-remote-code \
    --dyn-reasoning-parser nemotron_v3 \
    >> "$P2_LOG" 2>&1 &
PREFILL2_PID=$!
echo "[node2] prefill-2 PID: $PREFILL2_PID"

# Wait for prefill-2 to be healthy, then stay alive (tail log)
# Node1 polls our health from outside; we just need to not exit
P2_START=$SECONDS
while (( SECONDS - P2_START < 900 )); do
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
        --max-time 3 "http://127.0.0.1:${PREFILL_PORT_2}/health") || HTTP_CODE="000"
    if [ "$HTTP_CODE" = "200" ]; then
        echo "[node2] prefill-2 ready after $(( SECONDS - P2_START ))s ✓"; break
    fi
    sleep 10
    echo "[node2] ...prefill-2 still starting ($(( SECONDS - P2_START ))s) [HTTP: $HTTP_CODE]"
done

echo "[node2] ALL NODE2 WORKERS READY. SSH session staying alive (log suppressed on node1 terminal)..."
# Redirect tail to /dev/null so prefill-2 log lines do not spill into
# node1 terminal. Logs are on NFS — tail them separately if needed:
#   tail -f $RESULTS_DIR/ULTRA_DISAGG_2P2D_KV_2NODE__prefill2__ISL*__<RUN_ID>.log
tail -f "$P2_LOG" > /dev/null 2>&1

ENDSSH
    NODE2_SSH_PID=$!
    log "  Node2 SSH background PID: $NODE2_SSH_PID"

    # ------------------------------------------------------------------
    # 9e — Wait for decode-1, then start prefill-1, then wait for all 4
    #
    # Sequence:
    #   1. Wait decode-1 healthy  (node1 — mandatory before prefill-1)
    #   2. Start prefill-1        (node1)
    #   3. Wait decode-2 healthy  (node2 — in parallel with prefill-1 loading)
    #   4. Wait prefill-1 healthy (node1)
    #   5. Wait prefill-2 healthy (node2)
    # Steps 3+4 run concurrently in wall-clock time since prefill-1 takes
    # ~315s and node2 has been loading since the SSH fired ~5s ago.
    # ------------------------------------------------------------------

    # 1. Wait for decode-1 (mandatory before we can start prefill-1)
    wait_for_worker "http://127.0.0.1:${DECODE_PORT_1}/health" "decode-1"

    # 2. Start prefill-1 now that decode-1 is ready
    log "Starting prefill-1 (GPUs $GPUS_PREFILL_1, NUMA 0, PORT $PREFILL_PORT_1)..."
    VLLM_ENGINE_STARTUP_TIMEOUT=600 \
    CUDA_VISIBLE_DEVICES=$GPUS_PREFILL_1 \
    DYN_SYSTEM_PORT=$PREFILL_PORT_1 \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$NIXL_SC_1 \
    ETCD_ENDPOINTS="http://${NODE1_IP}:2379" \
    NATS_SERVER="nats://${NODE1_IP}:4222" \
    numactl --cpunodebind=0 --membind=0 \
    python3 -m dynamo.vllm \
        --model "$MODEL" \
        --served-model-name "$MODEL" \
        --disaggregation-mode prefill \
        --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
        --kv-events-config "{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:${KV_EVT_1}\",\"enable_kv_cache_events\":true}" \
        --tensor-parallel-size $TP \
        --max-model-len $(( ISL + OSL )) \
        --max-num-seqs $MAX_NUM_SEQS \
        --no-disable-hybrid-kv-cache-manager \
        --trust-remote-code \
        --dyn-reasoning-parser nemotron_v3 \
        > "$P1_LOG" 2>&1 &
    PREFILL1_PID=$!
    log "  prefill-1 PID: $PREFILL1_PID"

    # 3+4+5. Wait for remaining 3 workers — node2 has been loading this whole time
    wait_for_worker_remote "$NODE2_IP" "$DECODE_PORT_2"  "decode-2  (node2)" "$WORKER_READY_TIMEOUT"
    wait_for_worker "http://127.0.0.1:${PREFILL_PORT_1}/health" "prefill-1"
    wait_for_worker_remote "$NODE2_IP" "$PREFILL_PORT_2" "prefill-2 (node2)" "$WORKER_READY_TIMEOUT"

    STARTUP_SECS=$(( SECONDS - STEP_START ))
    log "All 4 workers healthy. Startup time: ${STARTUP_SECS}s"

    # Wait for frontend to be healthy and workers to register in etcd
    wait_for_worker "http://127.0.0.1:${HTTP_PORT}/health" "frontend"
    log "Waiting 30s for all workers to register in etcd..."
    sleep 30

    # ------------------------------------------------------------------
    # 9f — Run benchmark probe for this ISL
    # ------------------------------------------------------------------
    RESULT_FILE="$RESULTS_DIR/ULTRA_DISAGG_2P2D_KV_2NODE__in${ISL}_out${OSL}__prm${NUM_PROMPTS}__rate${RATE}__${RUN_ID}.json"

    log ""
    log "--- Benchmark probe: ISL=$ISL OSL=$OSL NUM_PROMPTS=$NUM_PROMPTS RATE=$RATE ---"
    set +e   # don't abort if benchmark fails
    vllm bench serve \
        --backend openai-chat \
        --model "$MODEL" \
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

    # ------------------------------------------------------------------
    # 9g — Parse result and check TTFT limit
    # ------------------------------------------------------------------
    if [ "$BENCH_RC" -ne 0 ] || [ ! -f "$RESULT_FILE" ]; then
        log "WARNING: Benchmark failed or no result file for ISL=$ISL (RC=$BENCH_RC)"
        VERDICT="BENCH_FAIL"
        printf "%-8s | %-11s | %-7s | %-15s | %-12s | %-12s | %-5s | %s\n" \
            "$ISL" "$MAX_NUM_SEQS" "${STARTUP_SECS}s" "N/A" "N/A" "N/A" "N/A" "$VERDICT" \
            >> "$SWEEP_SUMMARY"
        log "Stopping sweep early due to benchmark failure."
        SWEEP_STOPPED_EARLY=1
        break
    fi

    # Extract key metrics from JSON result
    # vllm bench serve saves a JSON with these fields
    TTFT_MED=$(python3 -c "
import json, sys
d = json.load(open('$RESULT_FILE'))
print(round(d.get('median_ttft_ms', d.get('mean_ttft_ms', -1)), 1))
" 2>/dev/null || echo "-1")
    TTFT_P99=$(python3 -c "
import json, sys
d = json.load(open('$RESULT_FILE'))
p = d.get('percentiles_ttft_ms', {})
print(round(p.get('P99', p.get('p99', d.get('p99_ttft_ms', -1))), 1))
" 2>/dev/null || echo "-1")
    TPOT_P99=$(python3 -c "
import json, sys
d = json.load(open('$RESULT_FILE'))
p = d.get('percentiles_tpot_ms', {})
print(round(p.get('P99', p.get('p99', d.get('p99_tpot_ms', -1))), 1))
" 2>/dev/null || echo "-1")
    TOKS=$(python3 -c "
import json, sys
d = json.load(open('$RESULT_FILE'))
print(round(d.get('output_throughput', -1), 1))
" 2>/dev/null || echo "-1")

    log "  TTFT median: ${TTFT_MED}ms"
    log "  TTFT P99:    ${TTFT_P99}ms"
    log "  TPOT P99:    ${TPOT_P99}ms"
    log "  Output tok/s: ${TOKS}"

    # Determine verdict
    VERDICT="OK"
    if python3 -c "import sys; sys.exit(0 if float('$TTFT_MED') > $TTFT_LIMIT_MS else 1)" 2>/dev/null; then
        VERDICT="TTFT_LIMIT_EXCEEDED"
    fi

    printf "%-8s | %-11s | %-7s | %-15s | %-12s | %-12s | %-5s | %s\n" \
        "$ISL" "$MAX_NUM_SEQS" "${STARTUP_SECS}s" "${TTFT_MED}ms" \
        "${TTFT_P99}ms" "${TPOT_P99}ms" "$TOKS" "$VERDICT" \
        >> "$SWEEP_SUMMARY"

    log "Verdict for ISL=$ISL: $VERDICT"

    if [ "$VERDICT" = "TTFT_LIMIT_EXCEEDED" ]; then
        log ""
        log "TTFT ${TTFT_MED}ms > limit ${TTFT_LIMIT_MS}ms — stopping sweep."
        SWEEP_STOPPED_EARLY=1
        break
    fi

    log "ISL=$ISL passed. Proceeding to next step..."

done  # end ISL_SWEEP loop

# =============================================================================
# SECTION 10 — Final summary
# =============================================================================
log ""
log "=========================================="
log "ISL SWEEP COMPLETE"
if [ "$SWEEP_STOPPED_EARLY" -eq 1 ]; then
    log "Stopped early — see reason in table below."
fi
log ""
log "Results summary ($SWEEP_SUMMARY):"
cat "$SWEEP_SUMMARY"
log ""
log "Full result JSONs in: $RESULTS_DIR"
log "Worker logs in:       $RESULTS_DIR (ULTRA_DISAGG_2P2D_KV_2NODE__*.log)"
log "=========================================="

# EXIT trap fires here — kills all node1 and node2 processes
