#!/bin/bash
set -uo pipefail

# =============================================================================
# 1P1D NIXL cross-node serve -- NEMOTRON-3-SUPER-120B-A12B-BF16, TP=4 per node
#
# Third in the chain. run_pd_full_test_N2_R1.sh proved 1P1D works at TP=1 on a
# 0.5B dense model; run_pd_tp4_N2.sh proved it works at TP=4 and that the four
# workers each take their own CXI rail. This one keeps that machinery bit for
# bit and changes the model to a 120B hybrid SSM/MoE, which is the first case
# where the KV transfer is not "KV" at all.
#
# WHAT IS ACTUALLY DIFFERENT ABOUT THIS MODEL
#
# NemotronHForCausalLM, 88 layers on the pattern
#   MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*
#   EMEMEMEM*EMEMEMEME
# which is 40 Mamba-2 layers, 40 latent-MoE layers and 8 full-attention layers.
# Only those 8 layers have a KV cache. The other 40 carry recurrent state, and
# that state is what dominates the transfer:
#
#   attention KV   8 layers x 1 KV head/rank x 128 head_dim x 2 (K,V) x 2 B
#                  = 4,096 B per token per rank   (2 total KV heads, and
#                    max(1, 2//4) = 1 per rank at TP=4 -- REPLICATED, not
#                    sharded, see model.py:1430)
#   Mamba SSM      40 layers x (128/4 heads) x 64 head_dim x 128 d_state x 4 B
#                  = 41.9 MB per SEQUENCE per rank, fp32, INDEPENDENT OF LENGTH
#   Mamba conv     40 layers x (10240/4) x 4 x 2 B = 0.8 MB per sequence per rank
#
# So each request moves ~42.7 MB per rank (~171 MB node-wide) of constant state
# plus 4 KB per token of KV. The crossover where context length starts to
# matter is around 10k tokens. Below that, disaggregation is paying a fixed
# ~171 MB toll per request for a prefill it barely saved -- so EXPECT 1P1D to
# look worse than co-located at short context, and do not read that as a
# broken setup. The interesting measurements are long-context and
# high-concurrency, which is `vllm bench serve`'s job, not this script's.
#
# THIS SCRIPT IS A BRING-UP, NOT A BENCHMARK. Its job is to get both servers
# healthy, prove one request's state crosses the fabric, and print a
# configuration worth tuning from. Q1-Q4 are inherited unchanged.
#
# Verifies four separate, chained questions:
#   Q1  Did a KV-cache transfer actually happen (vs D silently doing its own
#       local prefill)?              -> /metrics diff on P and D, plus timing
#   Q2  Did the bytes actually cross Slingshot?
#                                     -> CXI hardware octet counters, sampled
#   Q3  Did it use the LIBFABRIC backend with the shim, not a fallback?
#                                     -> shim PATCH lines + rail-manager lines
#   Q4  NEW AT TP>1. Did the four TP workers spread across the four CXI NICs,
#       or did they all pile onto one?
#                                     -> per-worker "Rail Manager created with"
#                                        lines, cross-checked against how many
#                                        cxi devices actually moved octets
#
# Q2/Q3 only mean something once Q1 has passed; Q4 only means something once
# Q2 has passed.
#
# Usage:
#   bash run_pd_nemotron_1p1d_N2.sh
#   ENFORCE_EAGER=1 bash run_pd_nemotron_1p1d_N2.sh    # skip CUDA graph capture
#   EXPERT_PARALLEL=1 bash run_pd_nemotron_1p1d_N2.sh  # EP instead of TP for MoE
#   MAX_MODEL_LEN=131072 bash run_pd_nemotron_1p1d_N2.sh
#   PROMPT_REPEAT=4000 bash run_pd_nemotron_1p1d_N2.sh # past the ~10k crossover
#
# PREREQUISITE: the weights must already be in the shared HF cache. 247 GB does
# not download inside a health-check window; the run will time out, cleanup()
# will fire, and the partial download is abandoned mid-shard. Run
#   bash download_model.sh
# to completion first -- rerunning it is the verification step.
#
# -----------------------------------------------------------------------------
# WHY Q4 IS STILL CHECKED HERE, HAVING ALREADY PASSED AT TP=4 ON QWEN
#
# Because the descriptors are different. Qwen registered one flat FA region per
# layer. This model registers FA descriptors over physical blocks AND Mamba
# descriptors over logical blocks, in one HMA-pooled tensor, with the conv
# state split three ways. Rail assignment is decided per registration, so
# "four rails on Qwen" does not carry over for free.
#
# The Phase 0 microbenchmark established that NIXL's LIBFABRIC backend gives
# ONE rail per GPU -- groupNicsWithAccel() partitions NICs by PCIe complex, so
# a process bound to one GPU gets that GPU's one NIC and 23.87 GB/s, while four
# processes on four GPUs aggregate to 87.67 GB/s. That was measured by a
# purpose-built MPI benchmark that explicitly bound one rank per GPU.
#
# vLLM at TP=4 is a different animal: four worker processes spawned by one
# engine, each setting its own CUDA device, each constructing its own NIXL
# agent. Whether NIXL's topology code sees four distinct GPU affinities (four
# rails, ~88 GB/s available) or four identical ones (one rail, 24 GB/s, and a
# 4x bottleneck nobody would notice until the Nemotron runs look slow) has
# never been checked. This is the first time the one-rail-per-GPU partition is
# exercised by vLLM rather than by the microbenchmark.
#
# Two independent readings, because either alone can mislead:
#   (a) the log:  NIXL prints "Rail Manager created with N rails" once per
#                 agent. Four workers => four such lines per node, each saying
#                 1 rail. Four lines saying 1 rail is the healthy shape; one
#                 line, or four lines that all name the same NIC, is not.
#   (b) the wire: the CXI octet poll already samples all four devices. Four
#                 rails in use means four devices with non-trivial deltas.
#                 One device carrying everything is the failure, whatever the
#                 log claims.
#
# -----------------------------------------------------------------------------
# WHY ONE mpiexec AND NOT TWO ssh -- the thing that makes this file different
# from every earlier revision.
#
# CXI needs a VNI. Without one, fi_domain() returns -FI_ENOSYS and the
# LIBFABRIC backend never comes up. The VNI arrives only in SLINGSHOT_VNIS,
# provisioned by PALS. Two `ssh -n NODE "bash launch_x.sh"` calls -- what this
# script used to do -- carry NO PALS environment at all, so both servers die
# at backend creation.
#
# The obvious repair (wrap each side in its own `mpiexec -n 1
# --single-node-vni`) is WORSE, and this is measured, not theorised. PALS
# allocates a VNI per APPLICATION, and a VNI is a traffic-isolation domain:
# endpoints on different VNIs cannot reach each other at all. Three separate
# launches were observed getting three different VNIs (1726, 1825, 1873 --
# the last from `-n 1 --single-node-vni`). Two independent launches would each
# build their agent happily and then never connect, trading today's loud
# immediate ENOSYS for a silent hang. See run_repro_nixl_2rank_transfer.sh:62.
#
# So: ONE `mpiexec -n 2 -ppn 1` runs ONE role script on both nodes, and that
# script picks prefill-vs-decode by HOSTNAME. Pairing by hostname rather than
# by PMI_RANK means whatever rank order PALS happens to hand out is
# irrelevant -- the same trick repro_nixl_2rank_transfer.py already uses.
#
# The proxy and this driver script stay OUTSIDE the mpiexec. They are plain
# HTTP and touch no fabric, so they need no VNI.
#
# -----------------------------------------------------------------------------
# WHY Q1 IS NO LONGER A TIMING TEST
#
# Earlier revisions asserted "disagg time_starttransfer should be noticeably
# LOWER than baseline". That is wrong, and it would report a HEALTHY run as a
# failure. Baseline is prefill-on-D. Disagg is prefill-on-P plus a KV transfer
# plus a proxy hop -- the same prefill work, with strictly more overhead
# around it. For a single request, correct 1P1D is EXPECTED TO BE SLOWER.
# Disaggregation buys throughput and ITL under load, not single-shot TTFT.
#
# The discriminating evidence is whether D skipped prefill. That is read off a
# before/after diff of both servers' /metrics endpoints, which is
# self-describing -- we print every counter that moved rather than guessing
# metric names that shift between vLLM releases. Timing is kept, demoted to a
# sanity check.
# =============================================================================
# CONFIRMED via check_cxi_libfabric.sh probe (both nodes, 2026-08-19): fi_info -p cxi
# works, libfabric 2.3.1, and NIXL's plugin manager lists LIBFABRIC as loadable in
# this exact conda env. Defaulting to it -- NIXL's own default backend is UCX, and
# plain UCX has no native Slingshot/CXI transport (see handoff doc). Override to
# UCX for an explicit A/B comparison once LIBFABRIC is confirmed working end-to-end.
NIXL_BACKEND=${NIXL_BACKEND:-LIBFABRIC}
MODEL=${MODEL:-nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16}

# =============================================================================
# SERVING CONFIGURATION
#
# Three tiers, and the tier a flag is in matters more than its value:
#
#   LOCKED       changing it breaks the run outright, or breaks the P/D
#                handshake, or silently invalidates the comparison. Each one
#                below cites the line of vLLM that enforces it.
#   SYMMETRIC    free to tune, but must be the SAME on P and D.
#   ASYMMETRIC   the actual point of disaggregation. P and D are different
#                workloads and want different numbers. This is the surface to
#                sweep once the run is healthy.
#
# What keeps this honest: NixlConnector hashes the load-bearing config into a
# compatibility hash exchanged at handshake, and enforces it by default
# (base_worker.py:554, enforce_handshake_compat defaults True). The hash covers
# vllm version, model, dtype, total KV heads, head size, layer count,
# attention backend name, kv cache dtype and whether HMA is on
# (metadata.py:113-127). Block size, TP size and kv_cache_layout are checked
# separately at handshake (metadata.py:96-98, base_worker.py:1699+). A mismatch
# is a loud error, not corruption -- so an illegal asymmetry cannot quietly
# produce plausible-looking numbers.
# =============================================================================

# --- LOCKED ------------------------------------------------------------------

# Homogeneous, and not negotiable. base_worker.py's handshake validation ends
# with "Mamba doesn't support heterogeneous TP"; the tp_mapping machinery that
# makes P_TP != D_TP work for dense models has no Mamba path. 4 = one worker
# per GH200. 240 GB of BF16 weights over 4 GPUs is 60 GB each, on 120 GB cards.
TP=${TP:-4}

# Hard assert at engine init:
#   base_worker.py:316   assert is_conv_state_dim_first(),
#                        "3-read Mamba conv transfer requires DS conv state
#                         layout. Set VLLM_SSM_CONV_STATE_LAYOUT=DS"
# It fires for ANY Mamba model under NixlConnector -- the blog that documents
# this presents it as a heterogeneous-TP requirement, which is wrong and would
# have cost a startup failure to discover. The connector decomposes each conv
# state into contiguous x/B/C sub-projections and needs (dim, state_len)
# ordering to do it. Valid values are exactly "SD" and "DS" (envs.py:1723).
SSM_CONV_STATE_LAYOUT=${SSM_CONV_STATE_LAYOUT:-DS}

# HMA pools the FA and SSM layers into shared tensors and pads the SSM page
# size to match the FA page size in bytes; without it the hybrid model has no
# valid cache layout and base_worker.py:311 asserts. It is ON by default here
# because NixlBaseConnector declares SupportsHMA (connector.py:79) and
# scheduler_config.disable_hybrid_kv_cache_manager is None => auto.
#
# Passed EXPLICITLY anyway, and that is the whole point of passing it. Read
# vllm.py:1605-1676 -- there are three states, not two:
#   unset (None)  -> vLLM decides. If ANY of {non-GPU platform, chunked local
#                    attention, a connector that does not subclass SupportsHMA}
#                    applies, HMA is silently turned off with a warning, and a
#                    hybrid SSM model then fails at startup for a reason that
#                    does not mention HMA.
#   explicit ON   -> vllm.py:1664-1671 RAISES if the runtime wanted it off.
#   explicit OFF  -> always respected.
# So the flag converts a silent downgrade into a loud config-time error. A
# tripwire, not a setting. (Note the asymmetry: the connector-support check
# lives inside the `is None` branch, so explicit-ON skips it -- but an
# unsupporting connector then fails loudly in factory.py instead. Either path
# is loud, which is all that is being bought here.)
HMA_FLAG=${HMA_FLAG:---no-disable-hybrid-kv-cache-manager}

# Two independent reasons, either one sufficient:
#   1. Measurement. If P and D share a prompt, D can serve the second request
#      from its own prefix cache instead of pulling state, which looks fast for
#      the wrong reason. Same reason the TP=1 baseline sets it.
#   2. Legality. base_worker.py:1728-1738 raises RuntimeError for
#      "Prefix caching with heterogeneous physical_blocks_per_logical ... not
#      supported for Mamba hybrid models", and tells you to pass exactly this.
PREFIX_CACHING_FLAG=${PREFIX_CACHING_FLAG:---no-enable-prefix-caching}

# The SSM state is the payload here, so its precision is not a detail: at
# fp32 it is 41.9 MB/sequence/rank, at bf16 half that and numerically worse
# through 40 recurrent layers. fp32 is vLLM's recommendation for Mamba-2 and
# the conservative choice for a first run. MUST MATCH ON P AND D -- P writes
# the state, D reads it as raw bytes; a width mismatch is not caught by the
# compat hash (which covers cache_dtype, not mamba_ssm_cache_dtype) and would
# produce garbage rather than an error. Both roles read this one variable.
MAMBA_SSM_CACHE_DTYPE=${MAMBA_SSM_CACHE_DTYPE:-float32}

# In the compat hash (metadata.py:125). "auto" = same as model dtype = bf16.
# Do not set fp8 on one side only; do not set it at all until BF16 works.
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-auto}

# MTP IS DELIBERATELY OFF, which means passing nothing: config.json declares
# num_nextn_predict_layers=1 and vLLM registers NemotronHMTPModel
# (registry.py:651), but speculative decoding is only active with an explicit
# --speculative-config. Leave it that way for now -- the interaction between
# SSM state transfer and a drafter that mutates that state across a P/D
# boundary is not validated, and enabling it on a bring-up run means debugging
# two new things at once. Revisit after the baseline is trustworthy.
#
# --trust-remote-code IS ALSO DELIBERATELY ABSENT. NemotronHForCausalLM is a
# first-class vLLM architecture (registry.py:182), so remote code would be
# executed for no benefit.

# --- SYMMETRIC (same on both roles, free to tune) ----------------------------

# 32k, not the model's declared 262144. Two reasons. Memory: max_model_len
# feeds the per-sequence attention KV reservation, and at 262k tokens x 4,096
# B/token/rank a single sequence would want ~1.07 GB of KV per rank before any
# concurrency. Signal: the SSM-state toll means anything under ~10k tokens is
# transfer-dominated, so 32k straddles the crossover and is where the shape of
# the disagg curve is actually visible. Raise it once the run is healthy.
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}

# 0.90 of 120 GB = 108 GB, minus ~60 GB of weights = ~48 GB/GPU for cache and
# state. Higher than the 0.3 the small-model runs used because that headroom
# is the entire concurrency budget: each concurrent sequence costs 41.9 MB of
# SSM state before a single KV byte. Backed off from 0.95 on purpose -- NIXL
# registers dma-buf regions over this memory and CUDA graph capture needs
# working space, and OOM at step 3 of engine init on a 240 GB model is an
# expensive way to save 6 GB.
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}

# UNSET BY DEFAULT, and that is the considered choice. Under HMA the attention
# block size is not a free parameter: vLLM derives an FA block size that makes
# the FA page size match the padded SSM page size in bytes (the reference
# deployment of this model reports it landing near 4224 at TP=4). Passing
# --block-size overrides the FA half of that pairing while the SSM half is
# still derived from the model, so a hand-picked value is as likely to fight
# the allocator as help it. Let vLLM derive it, read what it chose out of the
# startup log, THEN tune with a number instead of a guess.
#
# Heterogeneous block sizes between P and D are in fact supported
# (base_worker.py:1740, transfers happen at remote-block granularity and mamba
# pages go 1:1), so this is a performance knob and not a correctness one. Set
# BLOCK_SIZE=<n> to experiment; it applies to both roles.
#
# Its sibling --mamba-block-size is deliberately NOT offered: cache.py:128-130
# says it "can be set only when prefix caching is enabled", and prefix caching
# is off above for reasons that outrank block tuning.
BLOCK_SIZE=${BLOCK_SIZE:-}

# OFF by default. With 512 routed experts and top-k 22, expert parallelism is
# very likely the right answer eventually -- EP gives each GPU 128 whole
# experts instead of a quarter of all 512, and replaces a full all-reduce with
# a dispatch/combine all-to-all. But it changes the intra-node communication
# pattern completely, and on a node whose GPU-to-GPU topology has not been
# characterised yet (see the nvidia-smi topo table this script records), that
# is a second unknown layered on the one being measured. Bring up on TP-only,
# then A/B with EXPERT_PARALLEL=1 and compare.
EXPERT_PARALLEL=${EXPERT_PARALLEL:-}

# Escape hatch, off by default. CUDA graph capture over 88 layers x 4 workers
# x 2 nodes is minutes of startup and is the most likely place for an aarch64
# JIT problem to surface. ENFORCE_EAGER=1 removes graphs entirely: much slower
# decode, much simpler failure modes. Use it to answer "is this a kernel
# problem or a transfer problem", not for any run whose numbers get quoted.
ENFORCE_EAGER=${ENFORCE_EAGER:-}

# --- ASYMMETRIC (P and D are different workloads) ----------------------------
#
# This is the part worth sweeping, and the part that makes disaggregation pay.
# Co-located serving has to pick ONE batching policy for two workloads with
# opposite shapes. Splitting them is the entire argument for 1P1D, so leaving
# both sides on defaults would be measuring disaggregation with its main
# advantage switched off.
#
# Prefill is compute-bound and wants few, large batches: many tokens per step,
# few concurrent sequences. Each in-flight sequence on P costs a full 41.9 MB
# SSM allocation per rank whether or not it is being worked on, which is a
# second reason to keep P's sequence count low.
#
# Decode is memory-bandwidth-bound and wants the opposite: as many concurrent
# sequences as the state budget allows, each contributing one token per step.
# D's ceiling is arithmetic -- 48 GB/rank of budget divided by 41.9 MB of SSM
# state per sequence is ~1,100 sequences before attention KV is counted at all,
# so 256 is comfortable and 512 is worth trying.
#
# Starting points, not conclusions:
P_MAX_NUM_BATCHED_TOKENS=${P_MAX_NUM_BATCHED_TOKENS:-16384}
P_MAX_NUM_SEQS=${P_MAX_NUM_SEQS:-32}
D_MAX_NUM_BATCHED_TOKENS=${D_MAX_NUM_BATCHED_TOKENS:-2048}
D_MAX_NUM_SEQS=${D_MAX_NUM_SEQS:-256}

# NOT SET, deliberately, but worth knowing about while tuning:
#   --async-scheduling      already ON. In 0.27.1 it defaults to enabled
#                           whenever nothing incompatible is configured
#                           (vllm.py:1095-1143), and KV connectors are NOT on
#                           the incompatibility list. It overlaps scheduling
#                           with the forward pass and mostly helps D. Disable
#                           with --no-async-scheduling if a timing anomaly
#                           needs ruling out.
#   --long-prefill-token-threshold
#                           splits long prefills across steps. Irrelevant on P
#                           in 1P1D (P does nothing but prefill), potentially
#                           useful on D if it ever has to prefill locally.
#   --cuda-graph-sizes      the capture list. Worth narrowing on P, which runs
#                           few distinct decode batch shapes and pays full
#                           capture cost for all of them.

STAMP=$(date +%Y%m%d_%H%M%S)
# All run records live under one root rather than scattering pd_smoke_<stamp>
# directories directly into .../testing/, which also holds source checkouts and
# conda envs. Must be on a filesystem both compute nodes mount -- the remote
# CXI pollers append to it, and the remote role scripts read common_env.sh and
# launch_role.sh out of it.
RUNS_ROOT=${RUNS_ROOT:-/vast/draco/tara/projects/Tara_Deployment/software/testing/RUNS}
SHARED=${RUNS_ROOT}/pd_nemotron_tp${TP}_${STAMP}
mkdir -p ${SHARED}/logs
P_PORT=8100; D_PORT=8200; PROXY_PORT=8000
# Override with PROXY_SCRIPT=/your/path if your checkout lives elsewhere or
# a newer vllm version moves this file.
PROXY_SCRIPT=${PROXY_SCRIPT:-/vast/draco/tara/projects/Tara_Deployment/software/testing/vllm_0.27.1_08_18_2026/vllm/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Prompt length. 500 repeats of a 9-word sentence is ~5k tokens.
#
# Kept at the Qwen default even though this model has a 262k context, because
# 5k is the interesting place to start: it is BELOW the ~10k-token crossover
# where the constant Mamba-state toll stops dominating. Starting under the
# crossover means the first number this script produces is the WORST case for
# disaggregation, which is the honest way round -- if the plumbing works there,
# every longer prompt only improves. PROMPT_REPEAT=4000 (~40k tokens) crosses
# over and is the second run to make.
PROMPT_REPEAT=${PROMPT_REPEAT:-500}

# WHAT ACTUALLY CROSSES THE WIRE, which for this model is not mostly KV.
#
# Attention, per token per rank:
#   8 attention layers x 1 KV head x 128 head_dim x 2 (K,V) x 2 B bf16 = 4,096 B
# The KV head count is 1 and not 2 because of the same floor as Qwen:
# get_num_kv_heads() is max(1, total // tp) (config/model.py:1430), so at TP=4
# the two logical heads are REPLICATED across four ranks rather than sharded.
#
# SSM state, per SEQUENCE per rank, INDEPENDENT OF LENGTH:
#   ssm:  40 layers x (128/4) heads x 64 head_dim x 128 d_state x 4 B fp32
#         = 41.9 MB          <- MAMBA_SSM_CACHE_DTYPE, set to float32 below
#   conv: 40 layers x ((128/4) x 64 + 2 x (8/4) groups x 128) x 4 width x 2 B
#         = 0.8 MB           <- mamba_cache_dtype, left "auto" => bf16
#   total ~42.7 MB/rank, ~171 MB node-wide, for EVERY request regardless of
#   whether the prompt is 10 tokens or 100k.
#
# Note the two dtypes are SEPARATE knobs (cache.py:131-138): mamba_cache_dtype
# governs the conv state, mamba_ssm_cache_dtype governs the SSM state and falls
# back to the former when "auto". Only the SSM term is worth spending fp32 on,
# which is why only it is set.
#
# That constant is the defining property of disaggregating a hybrid model, and
# it sets the crossover: attention KV only exceeds the state toll at roughly
#   42.7e6 / 4096 ~= 10,400 tokens.
# Below that, a 1P1D request pays more for state than for cache and will look
# WORSE than co-located. Above it, the usual disagg argument applies.
#
# The variable feeds only the informational line printed with the prompt.
# Ground truth is vLLM's own measured "Avg MB per transfer".
KV_BYTES_PER_TOKEN_PER_RANK=${KV_BYTES_PER_TOKEN_PER_RANK:-4096}

# Constant per-request SSM+conv cost per rank, in bytes -- printed alongside the
# token-proportional estimate so the two are visible as separate terms rather
# than one blended number. Scale by 0.5 if MAMBA_SSM_CACHE_DTYPE is bfloat16.
MAMBA_STATE_BYTES_PER_SEQ_PER_RANK=${MAMBA_STATE_BYTES_PER_SEQ_PER_RANK:-42700000}

# 30 minutes, up from the small-model script's 10. Not padding:
#   - 247 GB of safetensors read from shared /vast by EIGHT worker processes
#     (4 per node x 2 nodes) at once, all wanting the same shards.
#   - CUDA graph capture over 88 layers per worker.
#   - On aarch64 several vLLM CUDA extensions have no prebuilt wheel and
#     JIT-compile on first use.
# The cost of guessing low is that HEALTH_TRIES expires, cleanup() kills both
# servers, and a loading run is destroyed minutes before it would have
# succeeded -- an expensive way to save nothing. If it does time out, check the
# log for progress before raising this: a server stuck at the same shard for
# ten minutes is a hang, not slowness.
HEALTH_TRIES=${HEALTH_TRIES:-360}     # x5s = 30 min

# --- KEEP_ALIVE: hold the servers up for an external bench client -------------
# Default 0 preserves the historical behaviour exactly: run the Q-checks, fall
# off the end, EXIT trap tears both servers down. Set to 1 and the script blocks
# after the checks instead of exiting, so `vllm bench serve` has something to
# talk to.
#
# It has to BLOCK rather than background-and-exit. mpiexec is a child of this
# shell (MPIEXEC_PID), and PALS ties the application's lifetime to the launcher:
# if this script exits, the EXIT trap fires cleanup() and both engines die --
# and even suppressing the trap would leave mpiexec orphaned with no parent to
# signal it. Blocking in the foreground keeps the process tree intact and makes
# Ctrl-C (INT -> the same trap) the intended teardown path.
KEEP_ALIVE=${KEEP_ALIVE:-0}
# Heartbeat interval for the hold loop. A silent terminal for an hour is
# indistinguishable from a dead one, which is the same complaint the deferred
# wait_healthy heartbeat exists to fix; do not set this to 0.
KEEP_ALIVE_POLL_S=${KEEP_ALIVE_POLL_S:-60}

# --- Resolve the two allocated nodes -----------------------------------------
cat ${PBS_NODEFILE}
mapfile -t NODES < <(sort -u "${PBS_NODEFILE}")
if [ ${#NODES[@]} -ne 2 ]; then
    echo "Expected exactly 2 unique nodes from PBS_NODEFILE, got ${#NODES[@]}. Aborting."
    exit 1
fi

THIS_HOST=$(hostname -s)
NODE_P=""; NODE_D=""
for n in "${NODES[@]}"; do
    if [[ "$n" == "${THIS_HOST}"* ]]; then NODE_P="$n"; else NODE_D="$n"; fi
done
if [ -z "${NODE_P}" ] || [ -z "${NODE_D}" ]; then
    echo "WARNING: couldn't match current hostname (${THIS_HOST}) against PBS_NODEFILE."
    echo "Falling back to array order -- verify NODE_P/NODE_D below are correct."
    NODE_P="${NODES[0]}"; NODE_D="${NODES[1]}"
fi

# PBS_NODEFILE carries FQDNs (x4820c7s0b0n0.hostmgmt2820.north.tara.alcf.anl.gov)
# but `hostname -s` on the compute node returns only the leading label. Keep
# BOTH forms explicitly: ssh wants the FQDN, and launch_role.sh's hostname
# comparison has to be short-vs-short or it never matches. This is not
# hypothetical -- comparing a short hostname against the FQDN sent both ranks
# down the "unexpected host" branch and killed the whole application before
# either server started. %%.* is a no-op if PBS ever hands back short names.
NODE_P_SHORT="${NODE_P%%.*}"
NODE_D_SHORT="${NODE_D%%.*}"
echo "Prefill node: ${NODE_P}   Decode node: ${NODE_D}"
echo "  (short forms, used for the in-rank hostname match: ${NODE_P_SHORT} / ${NODE_D_SHORT})"

# --- Resolve hsn0 IPs (also validates SSH works before we launch anything) --
get_hsn_ip() {
    ssh -n "$1" "ip -4 -o addr show hsn0" | awk '{print $4}' | cut -d/ -f1
}
P_IP=$(get_hsn_ip "${NODE_P}")
D_IP=$(get_hsn_ip "${NODE_D}")
echo "Prefill hsn0: ${P_IP}   Decode hsn0: ${D_IP}"
if [ -z "${P_IP}" ] || [ -z "${D_IP}" ]; then
    echo "Failed to resolve hsn0 IP on one or both nodes. Stop here, don't proceed blind."
    exit 1
fi

# --- Confirm shared FS is actually visible on the remote node ---------------
ssh -n "${NODE_D}" "test -d ${SHARED}" || { echo "SHARED path not visible on ${NODE_D} -- is /vast/draco mounted there?"; exit 1; }

# --- Locate the shim and the env script, IN PLACE in the repo ----------------
# Used straight from SCRIPT_DIR, not copied into SHARED. An earlier revision
# copied them and justified it as "both nodes then see byte-identical files",
# which is vacuous here: the repo lives on /vast, so both nodes already see the
# same file. What the copy actually bought was a way to be running a DIFFERENT
# binary than the one you rebuild and hand-test in the repo -- rebuild the
# shim, LD_PRELOAD the repo copy by hand to check it, and the script would
# still be preloading a snapshot from staging. The working benchmark
# (run_repro_nixl_2rank_transfer.sh:189) references SCRIPT_DIR directly; match
# it, for the same reason common_env.sh now sources the shared env script
# rather than duplicating it.
#
# Provenance is kept by recording a checksum below instead of by hoarding a
# copy of the binary in every timestamped run directory.
for f in fi_getinfo_shim.so env_for_libfabric_topology_error.sh; do
    if [ ! -f "${SCRIPT_DIR}/${f}" ]; then
        echo "Missing ${SCRIPT_DIR}/${f} -- cannot continue."
        [ "${f}" = "fi_getinfo_shim.so" ] && echo "  Build it: gcc -shared -fPIC -o fi_getinfo_shim.so fi_getinfo_shim.c -ldl \$(pkg-config --cflags libfabric)"
        exit 1
    fi
done
SHIM="${SCRIPT_DIR}/fi_getinfo_shim.so"
ENV_SCRIPT="${SCRIPT_DIR}/env_for_libfabric_topology_error.sh"

# Referencing in place moves the burden onto SCRIPT_DIR being mounted on the
# compute nodes, which the copy used to paper over. So check it, rather than
# discover it as an LD_PRELOAD that silently does nothing. test -f on the shim
# itself, not test -d on the directory -- a stale automount can satisfy the
# latter. Checked on NODE_D because this script runs on NODE_P.
ssh -n "${NODE_D}" "test -f ${SHIM}" || {
    echo "${SHIM} is not visible from ${NODE_D}."
    echo "The repo has to live on a filesystem both compute nodes mount"
    echo "(e.g. /vast/draco/...), not in a node-local or unmounted home."
    exit 1
}

# The tracked .so has gone stale before -- committed carrying PATCH 1 only
# while the .c had already grown PATCHES 2-4. That build clears BLOCKER 1
# (mr_mode==0 -> -FI_ENODATA) and then dies later in registerMem with "no
# available backends for mem type 'VRAM_SEG'", which reads as "the shim is
# broken" rather than "the shim is old". Cheap to rule out up front.
N_PATCH=$(strings "${SHIM}" 2>/dev/null | grep -c 'PATCH [234]' || true)
if [ "${N_PATCH}" -eq 0 ]; then
    echo "*** STALE SHIM: ${SHIM} has no PATCH 2/3/4 markers.                       ***"
    echo "*** It carries BLOCKER 1 only; --mem cuda / VRAM registration will fail.  ***"
    echo "*** Rebuild on this node before running:                                  ***"
    echo "***   gcc -shared -fPIC -o fi_getinfo_shim.so fi_getinfo_shim.c -ldl \$(pkg-config --cflags libfabric)"
    exit 1
fi

# Provenance, in place of the copy: which binary ran, and was it the one you
# think you built. Written to the run directory as well as stdout so an old
# SHARED dir can still be matched against a rebuilt shim later.
{
    echo "shim:      ${SHIM}"
    echo "sha256:    $(sha256sum "${SHIM}" 2>/dev/null | awk '{print $1}')"
    echo "built:     $(date -r "${SHIM}" '+%Y-%m-%d %H:%M:%S' 2>/dev/null)"
    echo "env:       ${ENV_SCRIPT}"
    echo "env md5:   $(md5sum "${ENV_SCRIPT}" 2>/dev/null | awk '{print $1}')"
} | tee ${SHARED}/provenance.txt
echo "Shim OK (PATCH 2/3/4 markers present), visible from ${NODE_D}."

# --- Intra-node GPU topology, recorded once ----------------------------------
# TP=${TP} makes every decode step an all-reduce across the ${TP} GPUs on the
# node, so how those GPUs are connected sets a ceiling on TP that has nothing
# to do with Slingshot. On a GH200 node the answer is not obvious: some are
# NVL4 (NVLink between the four superchips, ~hundreds of GB/s), others connect
# the Hopper dies only through their Grace CPUs, in which case the "NV#" cells
# below read "SYS" or "PHB" and every all-reduce crosses PCIe at a small
# fraction of that. If this run's TP=4 throughput disappoints, this table is
# the first thing to check -- and if it says SYS, the honest conclusion is that
# TP=4 is the wrong shape for this node, not that NIXL is slow.
#
# Recorded rather than asserted: this is a fact about the machine, and the
# right place for it is provenance.txt next to the run it explains.
{
    echo ""
    echo "=== nvidia-smi topo -m on ${NODE_P_SHORT} ==="
    ssh -n "${NODE_P}" "nvidia-smi topo -m" 2>&1
} >> "${SHARED}/provenance.txt"
echo "Intra-node GPU topology (also in provenance.txt):"
ssh -n "${NODE_P}" "nvidia-smi topo -m" 2>/dev/null | head -8 | sed 's/^/  /' \
    || echo "  (nvidia-smi topo unavailable)"

NO_PROXY_LIST="localhost,127.0.0.1,${P_IP},${D_IP},${NODE_P},${NODE_D},${NODE_P_SHORT},${NODE_D_SHORT}"

# --- Common env, sourced by both role scripts --------------------------------
# UCX-only tunables are gated on NIXL_BACKEND so the A/B still works, but they
# are dead weight on the LIBFABRIC path and were previously set unconditionally
# -- which made LIBFABRIC runs look UCX-configured in the logs.
UCX_LINES=""
if [ "${NIXL_BACKEND}" = "UCX" ]; then
    UCX_LINES='export UCX_TLS=cuda_copy,cuda_ipc,sm,tcp,self
export UCX_MODULE_DIR=$(python3 -c "import site,glob; print(glob.glob(site.getsitepackages()[0]+'"'"'/nixl_cu13.libs/ucx'"'"')[0])")'
fi

# GPU_ID intentionally NOT set by default -- CUDA_VISIBLE_DEVICES is left
# unset, so all GPUs on the node stay visible to the process. Was previously
# hardcoded to 0; removed per explicit ask, for two reasons:
#   1. Hardcoding to GPU 0 is exactly the kind of thing that silently breaks
#      once this script grows into the real sweep (Section 6) and needs a
#      specific GPU per role on a multi-GPU node -- easy to forget it's there.
#   2. Was: "logical inference, NOT confirmed" that restricting GPU visibility
#      could hand NIXL's rail selection a degenerate view of the topology.
#      NOW MEASURED, and the concern does not apply: the LIBFABRIC backend
#      reads its topology from hwloc (the whole node, always) and resolves
#      rails from the GPU's real PCI bus id, which CUDA_VISIBLE_DEVICES does
#      not change. Pinning one GPU gets you that GPU's one rail -- which is
#      the correct and expected 1-rail/25 GB/s behaviour for a TP=1 worker,
#      not a degraded topology. (The separate experiment that DID try to
#      widen a single GPU's rail count, by pruning the hwloc XML, is
#      --mode solo-peak in the benchmark, and it is a documented negative:
#      the NIC split is per PCIe complex, so pruning cannot help.)
# To pin a specific GPU for a future run, set GPU_ID rather than editing this
# file, e.g.: GPU_ID=2 bash run_pd_tp4_N2.sh
#
# AT TP>1 THIS BECOMES A FOOTGUN, so it is now guarded. Setting GPU_ID at TP=4
# leaves vLLM one visible device and it fails at startup with "not enough GPUs"
# -- loud, but the message points at the GPU count rather than at the variable,
# and losing five minutes of engine init to that is avoidable. More insidiously,
# GPU_ID with TP=2 on a 4-GPU node would "work" while quietly defeating the one
# thing this script exists to measure. Refuse the combination outright.
GPU_PIN_LINE=""
if [ -n "${GPU_ID:-}" ]; then
    if [ "${TP}" -gt 1 ]; then
        echo "GPU_ID=${GPU_ID} and TP=${TP} are incompatible: TP=${TP} needs ${TP} visible"
        echo "GPUs, and restricting visibility is exactly what Q4 is measuring."
        echo "Use GPU_ID only with TP=1 (that is run_pd_full_test_N2_R1.sh)."
        exit 1
    fi
    GPU_PIN_LINE="export CUDA_VISIBLE_DEVICES=${GPU_ID}"
    echo "GPU_ID=${GPU_ID} set -- pinning CUDA_VISIBLE_DEVICES=${GPU_ID} for both P and D."
fi

# The common_env.sh body used to live inline here. It moved to
# emit_common_env.sh so that run_colocated_N1.sh -- the Figure 2 baseline arm --
# generates a byte-identical environment from the same source. The two arms are
# only comparable if they differ in the P/D split and nothing else; a second
# copy of this block is exactly how that stops being true, and it is the same
# drift the "inline duplicate of env_for_libfabric_topology_error.sh" comment
# inside the generated file is already complaining about. One copy.
#
# emit_common_env() reads NO_PROXY_LIST, ENV_SCRIPT, UCX_LINES and GPU_PIN_LINE
# out of this scope -- all four are set above -- and takes the destination path
# as its argument.
if [ ! -f "${SCRIPT_DIR}/emit_common_env.sh" ]; then
    echo "FATAL: ${SCRIPT_DIR}/emit_common_env.sh is missing."
    echo "It carries the math_libs CPATH repair and the Triton \$CC override."
    echo "Without them the engines do not fail -- they grind through a silent"
    echo "recompile until wait_healthy times out. Refusing to launch."
    exit 1
fi
source "${SCRIPT_DIR}/emit_common_env.sh"
emit_common_env "${SHARED}/common_env.sh"

# --no-enable-prefix-caching matters here specifically: without it, if the
# baseline query (direct-to-D) and the disagg query (via proxy) use the same
# prompt, D could serve the second one from ITS OWN local prefix cache
# instead of actually pulling KV from the producer -- which would look fast
# for the wrong reason and silently invalidate the Q1 timing comparison.
# NIXL backend selection lives in kv_connector_extra_config.backends (per vLLM's
# NixlConnector docs). Leaving this unset defaults NIXL to UCX -- which is the
# whole reason last session's run couldn't have passed Q3 regardless of UCX_TLS
# tuning. NIXL_BACKEND=UCX still works if you want an explicit A/B later.
#
# kv_lease_duration -- the third knob in extra_config and the one that will
# bite under load. base_scheduler.py:70 reads it from extra_config with a
# default of 30, and pull_scheduler.py:244-255 uses it as the TTL on the
# producer's blocks after request_finished(): P holds a finished request's KV
# for this many seconds so D can still read it. The TP=2 run logged exactly
# that -- "waiting for 30 seconds before releasing blocks".
#
# At one request that is invisible. At benchmark concurrency it is a hard cap
# on P's throughput: P's usable KV cache is not its capacity but its capacity
# divided by how many requests finish inside the lease window. Nemotron's
# blocks are large, so P can run out of blocks while nearly all of them are
# held for readers that finished long ago.
#
# 30 is left as the default for the single-shot bring-up because shortening it
# risks D losing its source mid-read, which fails the run for a reason that
# looks like a fabric problem. Drop it (10, then 5) for the Stage 4 sweep and
# watch P's preemption counters -- the transfer itself took 5.1 ms at TP=2, so
# even 5 s is three orders of magnitude of headroom. Raise it only if D starts
# reporting failed reads.
KV_LEASE_DURATION=${KV_LEASE_DURATION:-30}

KV_EXTRA="\"backends\":[\"${NIXL_BACKEND}\"],\"kv_lease_duration\":${KV_LEASE_DURATION}"
KV_XFER_CONFIG_P="{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_producer\",\"kv_connector_extra_config\":{${KV_EXTRA}}}"
KV_XFER_CONFIG_D="{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_consumer\",\"kv_connector_extra_config\":{${KV_EXTRA}}}"

# ONE role script for both nodes, selecting by hostname. See the VNI section
# in the header for why this cannot be two independent launches.
#
# LD_PRELOAD is set HERE and not in common_env.sh on purpose. common_env.sh is
# also sourced by launch_proxy.sh, and the shim has no business interposing
# fi_getinfo for a plain HTTP process. Setting it immediately before the exec
# also keeps it out of every helper command in this shell.
#
# It does need to survive into vLLM's children -- EngineCore is a separate
# process -- but LD_PRELOAD is a normal environment variable and is inherited
# across fork/exec, so no special handling is required. The shim announces
# itself on stderr, so `grep fi_getinfo_shim` on the log is free confirmation
# rather than an assumption; Q3 below does exactly that.
#
# SIDE-CHANNEL PORTS STAY AT 5600/5601 AT TP=4. The handoff doc says to respace
# them to 5600/5700 before raising TP, on the belief that vLLM allocates
# VLLM_NIXL_SIDE_CHANNEL_PORT + tp_rank and four workers would therefore claim
# 5600-5603 on P and collide with D. That belief is wrong for 0.27.1 and the
# respacing would be cargo cult. Read the source:
#
#   nixl/base_scheduler.py:65   self.side_channel_port = (
#                                   envs.VLLM_NIXL_SIDE_CHANNEL_PORT
#                                   + vllm_config.parallel_config.data_parallel_index)
#
# The offset is DATA parallel index, not tensor parallel rank, and it is 0 for
# both roles here. There is exactly ONE listener per engine: base_scheduler.py
# :306-341 binds a single zmq.ROUTER and serves encoded_data[(pp_rank, tp_rank)]
# for every rank over that one socket. Four TP workers do not open four ports.
#
# 5600 and 5601 are on different NODES in any case, so even a per-rank scheme
# could not have collided. Left as-is; the handoff item is retired, not deferred.
cat > ${SHARED}/launch_role.sh <<EOF
#!/bin/bash
source ${SHARED}/common_env.sh

# Short-vs-short. \$(hostname -s) is already unqualified on this system, but
# strip any domain anyway so the comparison holds either way -- getting this
# wrong is silent and total: both ranks fall through to the else branch, the
# first one to exit takes the whole PALS application down with it, and neither
# p.log nor d.log is ever written.
MY_HOST="\$(hostname -s)"
MY_HOST="\${MY_HOST%%.*}"
if [ "\${MY_HOST}" = "${NODE_P_SHORT}" ]; then
    ROLE=p
    LOG=${SHARED}/logs/p.log
    PORT=${P_PORT}
    SIDE_HOST=${P_IP}
    SIDE_PORT=5600
    KV_CFG='${KV_XFER_CONFIG_P}'
    # Prefill: compute-bound. Few sequences, many tokens per step.
    MAX_BATCHED_TOKENS=${P_MAX_NUM_BATCHED_TOKENS}
    MAX_SEQS=${P_MAX_NUM_SEQS}
elif [ "\${MY_HOST}" = "${NODE_D_SHORT}" ]; then
    ROLE=d
    LOG=${SHARED}/logs/d.log
    PORT=${D_PORT}
    SIDE_HOST=${D_IP}
    SIDE_PORT=5601
    KV_CFG='${KV_XFER_CONFIG_D}'
    # Decode: memory-bandwidth-bound. Many sequences, one token each per step.
    MAX_BATCHED_TOKENS=${D_MAX_NUM_BATCHED_TOKENS}
    MAX_SEQS=${D_MAX_NUM_SEQS}
else
    echo "Rank landed on unexpected host '\${MY_HOST}'." >&2
    echo "  expected one of: '${NODE_P_SHORT}' (prefill) or '${NODE_D_SHORT}' (decode)" >&2
    echo "  full node names from PBS_NODEFILE: ${NODE_P} / ${NODE_D}" >&2
    exit 1
fi

exec > "\${LOG}" 2>&1
echo "=== role=\${ROLE} host=\$(hostname -s) SLINGSHOT_VNIS=\${SLINGSHOT_VNIS:-<UNSET>} ==="
if [ -z "\${SLINGSHOT_VNIS:-}" ]; then
    echo "SLINGSHOT_VNIS is EMPTY. The cxi provider will fail fi_domain() with"
    echo "-FI_ENOSYS and the LIBFABRIC backend will not come up. This means the"
    echo "process was not launched under PALS -- check the mpiexec invocation."
fi

export VLLM_NIXL_SIDE_CHANNEL_HOST=\${SIDE_HOST}
export VLLM_NIXL_SIDE_CHANNEL_PORT=\${SIDE_PORT}
export NIXL_LOG_LEVEL=\${NIXL_LOG_LEVEL:-INFO}
export LD_PRELOAD=${SHIM}

# Hard requirement for any Mamba model under NixlConnector, asserted at engine
# init (base_worker.py:316). Exported rather than passed as a flag because it is
# an envs.py setting with no CLI equivalent. See the LOCKED section up top.
export VLLM_SSM_CONV_STATE_LAYOUT=${SSM_CONV_STATE_LAYOUT}

# Optional flags, appended only when their variable is non-empty, so that an
# unset knob produces vLLM's own default rather than this script's opinion of
# it. Written as an array so the empty case adds nothing at all -- an empty
# string in the command line would be parsed as a positional argument and
# vllm serve would reject it.
EXTRA=()
[ -n "${BLOCK_SIZE}" ]      && EXTRA+=(--block-size ${BLOCK_SIZE})
[ -n "${EXPERT_PARALLEL}" ] && EXTRA+=(--enable-expert-parallel)
[ -n "${ENFORCE_EAGER}" ]   && EXTRA+=(--enforce-eager)

exec vllm serve ${MODEL} --host 0.0.0.0 --port \${PORT} \\
    --tensor-parallel-size ${TP} \\
    --max-model-len ${MAX_MODEL_LEN} \\
    --gpu-memory-utilization ${GPU_MEM_UTIL} \\
    --dtype bfloat16 \\
    --kv-cache-dtype ${KV_CACHE_DTYPE} \\
    --mamba-ssm-cache-dtype ${MAMBA_SSM_CACHE_DTYPE} \\
    ${HMA_FLAG} \\
    ${PREFIX_CACHING_FLAG} \\
    --max-num-batched-tokens \${MAX_BATCHED_TOKENS} \\
    --max-num-seqs \${MAX_SEQS} \\
    \${EXTRA[@]+"\${EXTRA[@]}"} \\
    --kv-transfer-config "\${KV_CFG}"
EOF
echo "Model:                     ${MODEL}"
echo "NIXL backend for this run: ${NIXL_BACKEND}"
echo "Tensor parallel size:      ${TP} (both roles), gpu-memory-utilization ${GPU_MEM_UTIL}"
echo "Context / caches:          max-model-len ${MAX_MODEL_LEN}, kv ${KV_CACHE_DTYPE}, ssm ${MAMBA_SSM_CACHE_DTYPE}, conv layout ${SSM_CONV_STATE_LAYOUT}"
echo "Batching (asymmetric):     P ${P_MAX_NUM_BATCHED_TOKENS} tok / ${P_MAX_NUM_SEQS} seq   D ${D_MAX_NUM_BATCHED_TOKENS} tok / ${D_MAX_NUM_SEQS} seq"
echo "Optional:                  block-size '${BLOCK_SIZE:-<derived>}', expert-parallel '${EXPERT_PARALLEL:-off}', enforce-eager '${ENFORCE_EAGER:-off}'"

# --- Launch both, from ONE application so they share a VNI -------------------
# -ppn 1 with 2 nodes puts exactly one server on each. Rank-to-node assignment
# is PALS's business; the role script sorts itself out by hostname.
#
# The logs are opened by the role script itself rather than redirected here,
# because one mpiexec has one stdout and both servers would otherwise
# interleave into a single unsplittable stream.
#
# --cpu-bind none IS NEW HERE AND IS NOT COSMETIC. PALS binds each rank to a
# subset of the node's cores by default, and a child process inherits its
# parent's affinity mask. At TP=1 that cost one server some cores. At TP=${TP}
# it would confine ALL ${TP} worker processes -- plus their NCCL and NIXL
# progress threads -- to whatever slice rank 0 was given, which on a Grace node
# can be a single core. The symptom is not an error: it is an engine that takes
# many minutes to initialise and a transfer rate that looks like a fabric
# problem when it is a scheduling problem. Since this run exists specifically
# to measure fabric behaviour, that confound has to go.
#
# Overridable, because a launcher that does not accept the flag would abort the
# whole application: MPI_CPU_BIND= (empty) drops it and restores the exact
# baseline invocation.
MPI_CPU_BIND=${MPI_CPU_BIND-none}
CPU_BIND_ARGS=()
[ -n "${MPI_CPU_BIND}" ] && CPU_BIND_ARGS=(--cpu-bind "${MPI_CPU_BIND}")
# ${arr[@]+"${arr[@]}"} rather than "${arr[@]}": expanding an empty array under
# `set -u` is only safe from bash 4.4 on, and this has to work if MPI_CPU_BIND=
# is used to fall back to the baseline invocation.

touch ${SHARED}/logs/p.log ${SHARED}/logs/d.log   # avoid a race with the tails below
mpiexec -n 2 -ppn 1 ${CPU_BIND_ARGS[@]+"${CPU_BIND_ARGS[@]}"} bash ${SHARED}/launch_role.sh > ${SHARED}/logs/mpiexec.log 2>&1 &
MPIEXEC_PID=$!

# Stream both server logs live, prefixed by role so interleaved output stays
# readable. `sed -u` (unbuffered) matters here -- without it, output piped
# through sed gets block-buffered and shows up in silent bursts instead of
# line-by-line, which looks exactly like "nothing is happening" even when it is.
#
# disown after each: cleanup() kills these, and without disown bash reports
# every one as a job-status line ("PID Killed exit 1") mixed into the teardown
# output, which reads like a fresh error at exactly the moment you are trying
# to work out what the real one was. Disowning only drops them from the job
# table; kill -TERM on the saved PID still works.
tail -n +1 -f ${SHARED}/logs/p.log | sed -u 's/^/[P] /' &
TAIL_P_PID=$!; disown ${TAIL_P_PID}
tail -n +1 -f ${SHARED}/logs/d.log | sed -u 's/^/[D] /' &
TAIL_D_PID=$!; disown ${TAIL_D_PID}
# mpiexec's own stream carries PALS errors that never reach either role log --
# a launch that dies before the role script runs is otherwise completely silent.
tail -n +1 -f ${SHARED}/logs/mpiexec.log | sed -u 's/^/[MPI] /' &
TAIL_M_PID=$!; disown ${TAIL_M_PID}

# pgrep/pkill patterns are bracketed ('[v]llm serve', not 'vllm serve') for a
# specific reason. `ssh host "pkill -f 'vllm serve'"` runs the command through
# `bash -c "pkill -f 'vllm serve'"` on the far side, and THAT shell's command
# line contains the literal string being searched for -- so pgrep/pkill match
# their own parent. Observed: cleanup reported "Stray vllm process ... force
# killing" on a run where no server ever started, then killed the wrapper shell
# and took the local ssh child down with it ("Killed exit 1" in the teardown
# output). The bracket makes the regex match the real process but not the
# command line containing the regex.
PAT_VLLM='[v]llm serve'
PAT_ENGINE='[E]ngineCore'
PAT_PROXY='[t]oy_proxy_server.py'
# NEW AT TP>1, and not optional. At TP=1 the engine runs the model in-process,
# so EngineCore was the only extra process to reap. At TP=${TP} the executor
# forks ${TP} MORE processes per node (multiproc_executor.py:708) and renames
# each of them via setproctitle to "VLLM::Worker_TP<rank>"
# (multiproc_executor.py:1043-1057, system_utils.py:198, prefix "VLLM" from
# envs.py:1752). None of those strings contains "vllm serve" or "EngineCore",
# so BOTH existing patterns miss all ${TP} of them.
#
# That is not a cosmetic leak. Each survivor holds a CUDA context and its share
# of the KV cache, so the next run profiles less free memory than it expects,
# or fails allocation outright -- and the symptom lands on the NEXT run, which
# is the worst place for it. Same class of bug as the EngineCore/side-channel
# port leak above, just multiplied by ${TP} and by two nodes.
#
# Case matters: "Worker_TP" not "worker_tp". Bracketed like the others so the
# remote `bash -c` running the pkill does not match its own command line.
PAT_WORKER='[W]orker_TP'

cleanup() {
    echo "=== Cleaning up ==="
    # CXI poll loops are setsid'd specifically so they survive an SSH session
    # closing -- which means they ALSO survive this script dying early, same
    # orphan-process problem as EngineCore below. Stop marker is a no-op if
    # the poll never started (POLL_STOP_MARKER unset under `set -u` is guarded
    # with :-).
    touch "${POLL_STOP_MARKER:-}" 2>/dev/null
    # SIGTERM the actual remote processes first, so each engine gets a real
    # chance to release its NIXL registrations and GPU memory cleanly before
    # anything gets SIGKILLed. Killing the launcher alone is not enough and
    # never was: with the old two-ssh launch there was no pty for a signal to
    # travel through, and with mpiexec, PALS's own teardown is not guaranteed
    # to reach a grandchild EngineCore before this script exits.
    for n in "${NODE_P}" "${NODE_D}"; do
        ssh -n "$n" "pkill -TERM -f \"${PAT_VLLM}\"" 2>/dev/null
    done
    ssh -n "${NODE_P}" "pkill -TERM -f \"${PAT_PROXY}\"" 2>/dev/null
    # :- guards matter here: cleanup can fire (via the EXIT trap) before the
    # proxy block ever runs -- e.g. if P or D fails its health check and the
    # script exits early -- in which case PROXY_SSH_PID/TAIL_PROXY_PID were
    # never assigned, and under `set -u` referencing them bare would abort
    # cleanup() partway through, skipping everything after that line.
    kill -TERM ${MPIEXEC_PID:-} ${TAIL_P_PID:-} ${TAIL_D_PID:-} ${TAIL_M_PID:-} ${PROXY_SSH_PID:-} ${TAIL_PROXY_PID:-} 2>/dev/null
    sleep 5
    # Belt-and-suspenders: an orphaned EngineCore child process (spawned
    # via multiprocessing) does NOT match a 'vllm serve' name pattern --
    # CONFIRMED via `top`, its process name shows as 'VLLM::EngineCor',
    # not anything containing 'vllm serve'. This is why it can survive
    # the pkill above and be left holding the NIXL side-channel port,
    # which is exactly what caused an "Address already in use" on a
    # later run. Kill both patterns, and free the known ports directly
    # too rather than relying on process-name matching alone.
    for n in "${NODE_P}" "${NODE_D}"; do
        ssh -n "$n" "pkill -KILL -f \"${PAT_ENGINE}\"" 2>/dev/null
        # Workers before/alongside the engine, not after: an orphaned worker
        # with no parent to talk to just sits there holding GPU memory.
        ssh -n "$n" "pkill -KILL -f \"${PAT_WORKER}\"" 2>/dev/null
    done
    ssh -n "${NODE_P}" "pkill -KILL -f \"${PAT_PROXY}\"" 2>/dev/null
    ssh -n "${NODE_P}" "fuser -k 5600/tcp" 2>/dev/null
    ssh -n "${NODE_D}" "fuser -k 5601/tcp" 2>/dev/null
    ssh -n "${NODE_P}" "fuser -k ${PROXY_PORT}/tcp" 2>/dev/null
    for n in "${NODE_P}" "${NODE_D}"; do
        if ssh -n "$n" "pgrep -f \"${PAT_VLLM}\"" > /dev/null 2>&1; then
            echo "Stray vllm process on $n -- force killing."
            ssh -n "$n" "pkill -KILL -f \"${PAT_VLLM}\""
        fi
        # Same idiom as the vllm check above -- `pgrep -f` in the if, not
        # `pgrep -cf` captured into a variable: pgrep exits 1 when it finds
        # nothing AND still prints "0", so `$(pgrep -cf ... || echo 0)` yields
        # the two-line string "0\n0" and the subsequent -gt test errors out
        # mid-cleanup, silently skipping everything below it.
        if ssh -n "$n" "pgrep -f \"${PAT_WORKER}\"" > /dev/null 2>&1; then
            echo "Stray TP worker(s) on $n -- force killing."
            ssh -n "$n" "pkill -KILL -f \"${PAT_WORKER}\""
        fi
    done
    # Free GPU memory is the thing that actually has to be true before the next
    # run, and process names are only a proxy for it. Report it directly.
    for n in "${NODE_P}" "${NODE_D}"; do
        echo "  GPU memory still in use on $n:"
        ssh -n "$n" "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader" 2>/dev/null \
            | sed 's/^/    /' || echo "    (nvidia-smi unavailable)"
    done
}
trap cleanup EXIT INT TERM

wait_healthy() {
    local ip=$1 port=$2 name=$3 path=${4:-/health}
    for i in $(seq 1 ${HEALTH_TRIES}); do
        if curl -s -o /dev/null -w "%{http_code}" http://${ip}:${port}${path} 2>/dev/null | grep -q 200; then
            echo "${name} healthy after $((i*5))s"; return 0
        fi
        # Fail fast instead of burning the full timeout: if mpiexec is already
        # gone, PALS tore the application down and no amount of waiting will
        # produce a healthy server. Skipped for the proxy, which is ssh-launched
        # and has no relationship to MPIEXEC_PID.
        if [ "${name#Proxy}" = "${name}" ] && ! kill -0 ${MPIEXEC_PID} 2>/dev/null; then
            echo "mpiexec exited while waiting for ${name}."
            echo "Look at ${SHARED}/logs/mpiexec.log first -- a PALS-level failure"
            echo "(bad VNI, node not in the allocation, launcher error) never"
            echo "reaches p.log or d.log at all."
            return 1
        fi
        sleep 5
    done
    echo "${name} never became healthy -- check ${SHARED}/logs/"; return 1
}
wait_healthy ${P_IP} ${P_PORT} "Prefill (${NODE_P})" || exit 1
wait_healthy ${D_IP} ${D_PORT} "Decode (${NODE_D})"  || exit 1

# =============================================================================
# Q3 (startup half) -- did the LIBFABRIC path actually come up, with the shim?
#
# Checked HERE, before any request, because all three of these are decided at
# engine init. If the shim never loaded or the backend fell back, there is no
# point sending traffic and then puzzling over the counters.
# =============================================================================
echo "=== Q3: transport evidence at startup ==="
for role in p d; do
    log=${SHARED}/logs/${role}.log
    echo "  --- ${role} ---"
    vni=$(grep -m1 -o 'SLINGSHOT_VNIS=[^ ]*' "${log}" 2>/dev/null)
    echo "      ${vni:-SLINGSHOT_VNIS=<line not found>}"
    n_shim=$(grep -c 'fi_getinfo_shim' "${log}" 2>/dev/null || true)
    echo "      shim announcements: ${n_shim}   (0 means LD_PRELOAD did not reach the engine)"
    # No fixed expected count: the shim announces on every fi_getinfo call, and
    # how many of those each worker makes is NIXL's business, not ours. What
    # matters is that the number rises with TP -- if TP=4 shows the same count
    # as the TP=1 baseline, LD_PRELOAD reached the engine but not the workers,
    # which vLLM spawns via multiprocessing rather than fork+exec of a shell.
    grep -m3 -i 'System runtime:.*NVIDIA GPU\|rail\|LIBFABRIC' "${log}" 2>/dev/null \
        | sed 's/^/      /' || echo "      (no LIBFABRIC/rail lines -- is NIXL_LOG_LEVEL=INFO set?)"
done
echo "  A healthy LIBFABRIC run shows a non-empty VNI, a non-zero shim count, and"
echo "  the rail-manager's GPU/rail lines. Zero shim announcements with a working"
echo "  server means it came up on UCX, and Q2's counters will be misleading."
echo ""

# =============================================================================
# Q4 (CPU half) -- is each TP worker running on its own GPU's NUMA domain?
#
# Node topology, from `nvidia-smi topo -m` on a Tara compute node (2026-08-25,
# also captured per-run in provenance.txt):
#
#       GPU0  GPU1  GPU2  GPU3   CPU affinity  NUMA  GPU NUMA ID
#   GPU0   X   NV6   NV6   NV6      0-71         0        4
#   GPU1  NV6    X   NV6   NV6     72-143        1       12
#   GPU2  NV6   NV6    X   NV6    144-215        2       20
#   GPU3  NV6   NV6   NV6    X    216-287        3       28
#
# NV6 everywhere: all four GPUs are NVLink-connected, six links per pair. TP=4's
# all-reduce therefore stays on NVLink and never competes with the KV transfer
# for the fabric. That is one of the reasons TP=4 is the right choice here and
# not merely the forced one (the other being Mamba's lack of a heterogeneous-TP
# path -- see the LOCKED block at the top).
#
# The rest of the table says this is a QUAD-GH200 node: four Grace-Hopper
# modules, each its own NUMA domain with its own 72 Grace cores and its own
# PCIe complex. Not one big CPU with four GPUs attached. GPU i, NUMA i and
# cxi<i> are the same module -- which is why groupNicsWithAccel() finds four
# distinct complexes to partition, and why Q4's fan-out works.
#
# What --cpu-bind cannot fix: mpiexec runs -n 2 -ppn 1, ONE rank per node, and
# that rank forks all ${TP} workers itself, so any PALS mask hits the whole
# tree. `none` (all 288 cores) is the only setting that does not strangle
# ${TP} workers into one rank's slice. But it grants freedom, not placement --
# nothing pulls worker i toward NUMA i, and vLLM 0.27.1 has no hook to do it:
# the entire CPU story for GPU workers is multiproc_executor.py:1060-1088,
# which only forces OMP_NUM_THREADS=1.
#
# Cost is UNMEASURED. Expect little for Q1-Q4 -- the KV payload goes GPU-to-NIC
# over GPUDirect and never touches host memory, so what crosses NUMA is the
# control path against a multi-millisecond transfer. Expect more once this
# model is under benchmark concurrency, where the scheduler, sampler and
# detokeniser run hot every step. Nemotron adds a reason the Qwen rig did not
# have: the Mamba state is large and constant per sequence, so D's host-side
# bookkeeping per request is heavier.
#
# Report always, pin only on request. PIN_WORKERS=1 tasksets each worker to its
# module's cores. The honest limit: taskset moves THREADS, not already-
# allocated PAGES, and vLLM's large host allocations happen during model load,
# long before this runs -- so expect CPU locality, not memory locality. Real
# memory placement needs numactl around a spawn we do not control. Default 0 so
# bring-up runs stay unconfounded; turn it on for the Stage 4 sweep and compare.
CORES_PER_NUMA=${CORES_PER_NUMA:-72}
PIN_WORKERS=${PIN_WORKERS:-0}

report_worker_affinity() {
    local role=$1 node=$2
    echo "  --- ${role} (${node}) ---"
    # NOT `ssh -n` here, unlike everywhere else in this file: -n points stdin at
    # /dev/null, which would swallow the heredoc that carries the remote script.
    ssh "${node}" "TP=${TP} CPN=${CORES_PER_NUMA} PIN=${PIN_WORKERS} bash -s" <<'REMOTE' 2>&1 | sed 's/^/      /'
for i in $(seq 0 $((TP - 1))); do
    # Bracketed like every other pattern in this script so pgrep does not match
    # its own command line. No TP1/TP10 ambiguity at TP<=4.
    pid=$(pgrep -f "[W]orker_TP${i}" | head -1)
    if [ -z "${pid}" ]; then
        echo "TP${i}: no process found"
        continue
    fi
    lo=$(( i * CPN )); hi=$(( lo + CPN - 1 ))
    if [ "${PIN}" = "1" ]; then
        if taskset -apc ${lo}-${hi} ${pid} >/dev/null 2>&1; then
            echo "TP${i}: pinned pid ${pid} -> cores ${lo}-${hi}"
        else
            echo "TP${i}: pin FAILED for pid ${pid} (no taskset, or not permitted)"
        fi
    fi
    cpus=$(awk '/^Cpus_allowed_list/{print $2}' /proc/${pid}/status 2>/dev/null)
    mems=$(awk '/^Mems_allowed_list/{print $2}' /proc/${pid}/status 2>/dev/null)
    verdict="floating across all NUMA domains"
    [ "${cpus}" = "${lo}-${hi}" ] && verdict="matches GPU${i}/NUMA${i}"
    echo "TP${i}: pid ${pid} cpus=${cpus} mems=${mems} -- want ${lo}-${hi} (NUMA ${i}) -- ${verdict}"
done
REMOTE
}

echo "=== Q4 (CPU half): TP worker placement vs the GPU/NUMA partition ==="
echo "  PIN_WORKERS=${PIN_WORKERS} (1 to taskset each worker to its module's ${CORES_PER_NUMA} cores)"
report_worker_affinity "prefill" "${NODE_P}"
report_worker_affinity "decode"  "${NODE_D}"
echo ""

# =============================================================================
# Q4 (log half) -- did all ${TP} workers build a NIXL agent?
#
# Corrected 2026-08-25 against the TP=2 Qwen run. Two NIXL lines mention rails
# and they count DIFFERENT things:
#
#   libfabric_backend.cpp:456  "Rail Manager created with <N> rails"
#       Per agent construction. N is how many rails that agent HOLDS.
#   libfabric_backend.cpp:782  "Successfully created connection for agent
#                               <id> on <N> rails"
#       Per peer connection. N counts every NIC on the node, so on Tara it
#       reads "on 4 rails" regardless of TP. NOT a fan-out measurement.
#
# The TP=2 run printed "on 4 rails" while the wire showed each worker on
# exactly ONE device (cxi0 tx=31.5 MB, cxi1 tx=31.3 MB, cxi2=572 B, cxi3=0).
# Two workers striping over four rails would have shown four devices at ~15 MB
# each. So the connection spans all four NICs but rail SELECTION is by GPU
# affinity, one rail per worker.
#
# This section is therefore advisory. Agent count is all it establishes: fewer
# than ${TP} rail managers => some workers never built an agent. The rail
# COUNTS are not a verdict; the wire half below is.
# =============================================================================
echo "=== Q4 (log half): NIXL agent construction across the ${TP} TP workers ==="
for role in p d; do
    log=${SHARED}/logs/${role}.log
    n_rm=$(grep -c 'Rail Manager created with' "${log}" 2>/dev/null || true)
    echo "  --- ${role}: ${n_rm} rail managers, expected ${TP} (one per TP worker) ---"
    if [ "${n_rm}" -eq 0 ] 2>/dev/null; then
        echo "      No rail-manager lines. This string is not emitted by every NIXL"
        echo "      build at INFO -- absence is not by itself a failure. Fall back to"
        echo "      the connection lines below and to Q3's shim-announcement count."
    else
        grep -o 'Rail Manager created with [0-9]* rails' "${log}" 2>/dev/null \
            | sort | uniq -c | sed 's/^/      /'
    fi
    n_conn=$(grep -c 'Successfully created connection for agent' "${log}" 2>/dev/null || true)
    echo "      ${n_conn} peer connections established (spans all node NICs; not a fan-out signal)"
    # The accelerator->NIC map, printed at INFO by libfabric_topology.cpp:285.
    # Every agent prints the whole node's map, so these repeat ${TP} times;
    # dedupe. This is what a healthy 4-GPU/4-NIC partition looks like -- four
    # distinct PCI addresses, four distinct cxi devices, one each.
    echo "      accelerator -> NIC map as NIXL sees it:"
    grep -o 'Accelerator-PCI [^ ]* .*\[.*\]' "${log}" 2>/dev/null \
        | sort -u | sed 's/^/        /' \
        || echo "        (none logged)"
done
echo ""

# =============================================================================
# Q1 -- baseline: query D directly, bypassing the proxy entirely.
# This is D doing its own full local prefill, same as a standalone instance.
#
# NOTE the prompt size. The old 6-token prompt moved ~74 KB of KV, far below
# the noise floor of the CXI counters. See PROMPT_REPEAT at the top.
# =============================================================================
PROMPT=$(python3 -c "print(('The quick brown fox jumps over the lazy dog. ' * ${PROMPT_REPEAT}).strip())")
echo "Prompt: ${PROMPT_REPEAT} repeats, $(printf %s "${PROMPT}" | wc -c) chars"
EST_TOKENS=$(( $(printf %s "${PROMPT}" | wc -c) / 4 ))
# Two terms, printed separately on purpose. One scales with the prompt; the
# other does not, and for this model the constant one is usually the larger.
# Blending them into a single "MB transferred" would hide exactly the property
# that makes disaggregating a hybrid model different from disaggregating a
# transformer.
KV_MB_RANK=$(( EST_TOKENS * KV_BYTES_PER_TOKEN_PER_RANK / 1000000 ))
SSM_MB_RANK=$(( MAMBA_STATE_BYTES_PER_SEQ_PER_RANK / 1000000 ))
echo "  ~${EST_TOKENS} tokens, per rank:"
echo "    attention KV   ~${KV_MB_RANK} MB   (scales with length, ${KV_BYTES_PER_TOKEN_PER_RANK} B/token)"
echo "    Mamba state    ~${SSM_MB_RANK} MB   (CONSTANT per request, length-independent)"
echo "    node-wide      ~$(( (KV_MB_RANK + SSM_MB_RANK) * TP )) MB across ${TP} rails"
if [ "${KV_MB_RANK}" -lt "${SSM_MB_RANK}" ]; then
    echo "  -> Below the ~10k-token crossover: state dominates KV, so this request is"
    echo "     the WORST case for disaggregation. Expect Q1 to favour co-located."
else
    echo "  -> Above the ~10k-token crossover: KV dominates state, the regime where"
    echo "     disaggregation is supposed to pay."
fi

# Both servers' /metrics, as a name->value map. Diffed around the disagg
# request to answer Q1 without hardcoding metric names: vLLM renames these
# between releases, and printing everything that MOVED is both more robust and
# more informative than asserting on one counter we guessed at.
snapshot_metrics() {
    local ip=$1 port=$2 out=$3
    curl -s "http://${ip}:${port}/metrics" 2>/dev/null \
        | grep -v '^#' | awk '{print $1" "$2}' | sort > "${out}"
}
# Grouped, and the vllm: counters are UNCAPPED. This used to be a flat
# `head -40` over everything that moved, which was actively harmful: /metrics
# is alphabetical, so http_* and process_* sort ahead of vllm:*, and 40 lines
# of http_request_duration histogram buckets consumed the entire budget. On the
# first passing run that hid vllm:prompt_tokens_total -- the single counter Q1
# is actually about -- while printing 20 near-identical bucket lines. Histogram
# buckets are the noise here and named counters are the signal, so split them.
diff_metrics() {
    local before=$1 after=$2 label=$3
    local moved="${after}.moved"
    join "${before}" "${after}" 2>/dev/null \
        | awk '$2 != $3 {printf "%-58s %s -> %s\n", $1, $2, $3}'  > "${moved}"
    join -v2 "${before}" "${after}" 2>/dev/null \
        | awk '{printf "%-58s (new) %s\n", $1, $2}'              >> "${moved}"

    echo "  -- ${label} --"
    echo "     vllm counters that moved (uncapped -- these are what Q1 turns on):"
    grep '^vllm:' "${moved}" | grep -v '_bucket' | sed 's/^/       /'
    echo "     (plus $(grep -c '^vllm:.*_bucket' "${moved}") vllm histogram buckets)"
    echo "     non-vllm counters that moved (first 12 of $(grep -cv '^vllm:' "${moved}")):"
    grep -v '^vllm:' "${moved}" | head -12 | sed 's/^/       /'
    echo "     full list: ${moved}"
}

# Body built by python3 -- json.dump, not shell interpolation. At 500 repeats
# the prompt is ~22 KB, well past the point where hand-escaping it into a
# -d "{...}" string is worth the risk, and python3 is guaranteed present here
# (the conda env is already active).
BODY_JSON=${SHARED}/req_body.json
python3 -c "
import json, sys
json.dump({'model': sys.argv[1], 'prompt': sys.argv[2], 'max_tokens': 10, 'stream': True},
          open(sys.argv[3], 'w'))
" "${MODEL}" "${PROMPT}" "${BODY_JSON}"

send_and_time() {
    local ip=$1 port=$2 label=$3
    local out=${SHARED}/logs/resp_${label}.txt
    local t
    t=$(curl -s -o ${out} -w "%{time_starttransfer}" http://${ip}:${port}/v1/completions \
        -H "Content-Type: application/json" --data-binary @${BODY_JSON})
    echo "${label}: time_starttransfer=${t}s  (raw response: ${out})" \
        | tee -a ${SHARED}/logs/timings.txt
}

M=${SHARED}/logs/metrics
echo "=== Q1a: baseline -- direct to decode instance, no proxy ==="
echo "  (P is NOT in this path at all. Whatever P's counters do here is the"
echo "   background floor that Q1b has to beat.)"
snapshot_metrics ${P_IP} ${P_PORT} ${M}_p_a_before
snapshot_metrics ${D_IP} ${D_PORT} ${M}_d_a_before
send_and_time ${D_IP} ${D_PORT} "baseline_decode_alone"
snapshot_metrics ${P_IP} ${P_PORT} ${M}_p_a_after
snapshot_metrics ${D_IP} ${D_PORT} ${M}_d_a_after
diff_metrics ${M}_p_a_before ${M}_p_a_after "Q1a PREFILL node -- expected: nothing moves"
diff_metrics ${M}_d_a_before ${M}_d_a_after "Q1a DECODE node -- expected: a full local prefill+decode"

# =============================================================================
# Direct P hop -- replicates the proxy's own first request to P by hand,
# BEFORE the proxy is even involved. Checks whether P's completion response
# actually contains real transfer coordinates (remote_engine_id/
# remote_block_ids/remote_host/remote_port) or leaves them null. Isolates
# whether a problem lives in P's own NixlConnector response versus
# downstream in the proxy's relay logic -- the proxy's own
# send_request_to_service() sends exactly this shape.
# =============================================================================
echo "=== Direct P hop: replicating the proxy's first request by hand ==="
# Same python-built-body reason as send_and_time: the prompt is now ~22 KB and
# has no business being pasted into a shell-quoted JSON literal.
PHOP_JSON=${SHARED}/req_body_phop.json
python3 -c "
import json, sys
json.dump({
    'model': sys.argv[1], 'prompt': sys.argv[2], 'max_tokens': 1, 'stream': False,
    'kv_transfer_params': {
        'do_remote_decode': True, 'do_remote_prefill': False,
        'remote_engine_id': None, 'remote_block_ids': None,
        'remote_host': None, 'remote_port': None,
    },
}, open(sys.argv[3], 'w'))
" "${MODEL}" "${PROMPT}" "${PHOP_JSON}"

curl -s http://${P_IP}:${P_PORT}/v1/completions \
    -H "Content-Type: application/json" \
    --data-binary @${PHOP_JSON} \
    | tee ${SHARED}/logs/resp_direct_p_hop.json | python3 -m json.tool 2>/dev/null \
    || echo "(response wasn't valid JSON -- see ${SHARED}/logs/resp_direct_p_hop.json raw)"
echo "=== Check kv_transfer_params above: real remote_engine_id/remote_block_ids/"
echo "remote_host/remote_port means P is populating it correctly; any of"
echo "those still null/None means the gap is upstream of the proxy entirely."
echo ""

# =============================================================================
# PROXY -- wired to the real reference implementation
# (tests/v1/kv_connector/nixl_integration/toy_proxy_server.py). Its health
# endpoint is /healthcheck, not /health like vLLM's own servers -- confirmed
# by reading the script, not assumed. It also sends
# "Authorization: Bearer $OPENAI_API_KEY" on every request to P and D; our
# vllm serve instances were never launched with --api-key so this shouldn't
# be validated either way, but set a real value rather than leave it to
# "Bearer None" on an unvalidated assumption. --host 0.0.0.0 is explicit
# rather than relying on the script's default 127.0.0.1 (loopback-only) --
# same "bound somewhere unreachable by default" pattern that bit the NIXL
# side-channel host earlier.
# =============================================================================
if [ ! -f "${PROXY_SCRIPT}" ]; then
    echo "PROXY_SCRIPT not found at ${PROXY_SCRIPT}."
    echo "Set PROXY_SCRIPT=/actual/path and rerun."
    exit 1
fi

cat > ${SHARED}/launch_proxy.sh <<EOF
#!/bin/bash
source ${SHARED}/common_env.sh
export OPENAI_API_KEY=smoke-test-dummy-key
python3 ${PROXY_SCRIPT} \
    --host 0.0.0.0 --port ${PROXY_PORT} \
    --prefiller-host ${P_IP} --prefiller-port ${P_PORT} \
    --decoder-host ${D_IP} --decoder-port ${D_PORT}
EOF

ssh -n "${NODE_P}" "bash ${SHARED}/launch_proxy.sh" > ${SHARED}/logs/proxy.log 2>&1 &
PROXY_SSH_PID=$!
touch ${SHARED}/logs/proxy.log
tail -n +1 -f ${SHARED}/logs/proxy.log | sed -u 's/^/[PROXY] /' &
TAIL_PROXY_PID=$!
wait_healthy ${P_IP} ${PROXY_PORT} "Proxy (${NODE_P})" "/healthcheck" || exit 1

echo ""
echo "=== Proxy wired in -- running Q1b/Q2 for real ==="
echo ""

# --- Q2 (legacy, kept as a cheap secondary signal) ---------------------------
# CAVEAT (confirmed architecture fact, not just a small-model noise problem):
# CXI RDMA is exposed via a kernel-bypass character device, separate from the
# hsn0 netdevice/classical-Ethernet path these /proc/net/dev counters read.
# A real CXI RDMA transfer may not register here at all, regardless of model
# size -- this is why the CXI sysfs polling below is now the primary check,
# not this one.
snapshot_counters() {
    local tag=$1
    for n in "${NODE_P}" "${NODE_D}"; do
        ssh -n "$n" "echo '--- $n ($tag) ---'; grep -E 'hsn0|bond0' /proc/net/dev"
    done
}

# --- Q2, real signal: CXI hardware octet counters, sampled DURING the request,
# not just before/after. Confirmed real path + counter names via
# check_cxi_libfabric.sh probe on both nodes (2026-08-19):
#   /sys/class/cxi/cxi<0-3>/device/telemetry/hni_sts_{tx,rx}_ok_octets
# Polls all 4 CXI devices per node (don't yet know which one is rail-aligned
# to CUDA_VISIBLE_DEVICES=0 on this system -- cheaper to poll all 4 and let
# the data show which moved than to assume cxi0<->GPU0).
cat > ${SHARED}/poll_cxi.sh <<'POLLEOF'
#!/bin/bash
STOP_MARKER="$1"
OUT="$2"
while [ ! -f "${STOP_MARKER}" ]; do
    ts=$(date +%s.%N)
    for i in 0 1 2 3; do
        tx=$(cat /sys/class/cxi/cxi${i}/device/telemetry/hni_sts_tx_ok_octets 2>/dev/null)
        rx=$(cat /sys/class/cxi/cxi${i}/device/telemetry/hni_sts_rx_ok_octets 2>/dev/null)
        echo "${ts},cxi${i},tx=${tx},rx=${rx}" >> "${OUT}"
    done
    sleep 0.05
done
POLLEOF

POLL_STOP_MARKER="${SHARED}/poll_stop_${STAMP}"
start_cxi_poll() {
    local node=$1 outfile=$2
    : > "${outfile}"
    # setsid+nohup+disown so the loop survives this ssh session closing --
    # same class of issue the handoff doc already hit with EngineCore cleanup
    # (signaling the local ssh client PID doesn't reach the remote process).
    ssh -n "$node" "setsid nohup bash ${SHARED}/poll_cxi.sh ${POLL_STOP_MARKER} ${outfile} > /dev/null 2>&1 < /dev/null &" 
}
stop_cxi_poll() {
    touch "${POLL_STOP_MARKER}"
    sleep 0.3   # >1 poll interval, let both remote loops notice and exit
}

# Wait for the counters to stop moving before stopping the pollers. A 200 ms
# fixed sleep (what this used to do) truncates the transfer: the KV pull is
# asynchronous with respect to the HTTP response, and the sysfs telemetry is
# itself sampled by firmware on its own cadence, so the last octets of a
# transfer land well after curl returns. The benchmark hit exactly this and
# solved it the same way (read_cxi_counters_settled in
# repro_nixl_2rank_transfer.py) -- sample until quiescent, don't guess a sleep.
#
# "Quiet" is a THRESHOLD, not equality. The first version required the counters
# to be byte-identical across three consecutive samples, which can never happen:
# these are node-wide NIC counters, and this very script is curling /metrics and
# ssh-ing across the same NIC while it waits. Observed on the first passing run
# -- it reported "STILL MOVING ... deltas are a lower bound" at the cap, when
# the transfer had in fact completed and the deltas were accurate to 2.2%.
# Crying wolf on a good run is worse than not checking at all.
#
# The threshold separates cleanly: background chatter is a few hundred KB per
# 250 ms window, while the transfer itself is ~60 MB in ~10 ms -- any window
# overlapping it is orders of magnitude above 1 MB.
#
# Summing in awk is safe despite these being 64-bit counters: eight values of
# ~1e12 sum to ~1e13, and a double is exact to 9e15.
#
# CAVEAT: the pollers run on the remote nodes and append to ${SHARED}; if that
# filesystem doesn't propagate appends promptly, "quiescent" here can mean
# "not yet visible". The fixed drain below runs first for exactly that reason
# -- it guarantees a real post-request sampling window regardless.
cxi_total_bytes() {
    { tail -n 4 "${SHARED}/logs/cxi_poll_p.csv" 2>/dev/null
      tail -n 4 "${SHARED}/logs/cxi_poll_d.csv" 2>/dev/null; } \
      | awk -F'[,=]' '{ s += $4 + $6 } END { printf "%.0f\n", s+0 }'
}
settle_cxi_poll() {
    local drain=${CXI_DRAIN_S:-2}                  # unconditional post-request window
    local quiet_bytes=${CXI_QUIET_BYTES:-1000000}  # per 250ms sample; ~4 MB/s of noise
    local quiet=0 tries=0 cur prev delta
    sleep "${drain}"
    prev=$(cxi_total_bytes)
    while [ ${quiet} -lt 3 ] && [ ${tries} -lt 40 ]; do   # cap ~10s past the drain
        sleep 0.25
        cur=$(cxi_total_bytes)
        delta=$(( ${cur:-0} - ${prev:-0} ))
        [ ${delta} -lt 0 ] && delta=$(( -delta ))
        if [ ${delta} -lt ${quiet_bytes} ]; then quiet=$((quiet + 1)); else quiet=0; fi
        prev=${cur}
        tries=$((tries + 1))
    done
    if [ ${quiet} -ge 3 ]; then
        echo "  CXI counters quiet (<${quiet_bytes} B/sample) after ${drain}s drain + $(awk "BEGIN{printf \"%.1f\", ${tries} * 0.25}")s"
    else
        echo "  CXI counters STILL MOVING at the settle cap (>${quiet_bytes} B/sample)."
        echo "  Deltas below are a lower bound. Either the transfer is genuinely still"
        echo "  running, or something else is saturating hsn0 -- check the time series"
        echo "  in cxi_poll_{p,d}.csv before trusting the totals."
    fi
}
summarize_cxi_poll() {
    local outfile=$1 label=$2
    echo "  -- ${label} --"
    if [ ! -s "${outfile}" ]; then
        echo "    (no samples captured -- poll loop may not have started; check permissions on the telemetry files)"
        return
    fi
    awk -F'[,=]' '
        { dev=$2
          if (!(dev in txfirst)) { txfirst[dev]=$4; rxfirst[dev]=$6 }
          txlast[dev]=$4; rxlast[dev]=$6; n[dev]++
        }
        END {
          for (d in txlast) printf "    %s: %d samples, tx_delta=%d bytes, rx_delta=%d bytes\n", d, n[d], txlast[d]-txfirst[d], rxlast[d]-rxfirst[d]
        }' "${outfile}" | sort
}

echo "=== Q2 (legacy/secondary): interface counters BEFORE disagg request ==="
snapshot_counters "before"

echo "=== Q2 (real signal): starting CXI octet-counter polling on both nodes ==="
start_cxi_poll "${NODE_P}" "${SHARED}/logs/cxi_poll_p.csv"
start_cxi_poll "${NODE_D}" "${SHARED}/logs/cxi_poll_d.csv"
sleep 0.2   # let both pollers get at least one sample before the request fires

echo "=== Q1b: disagg request, through the proxy ==="
snapshot_metrics ${P_IP} ${P_PORT} ${M}_p_b_before
snapshot_metrics ${D_IP} ${D_PORT} ${M}_d_b_before
send_and_time ${P_IP} ${PROXY_PORT} "disagg_via_proxy"

settle_cxi_poll
stop_cxi_poll

# Snapshot AFTER settling, not immediately after curl returns: the KV pull
# outlives the HTTP response, and so do the counters that record it.
snapshot_metrics ${P_IP} ${P_PORT} ${M}_p_b_after
snapshot_metrics ${D_IP} ${D_PORT} ${M}_d_b_after

echo "=== Q2 (legacy/secondary): interface counters AFTER disagg request ==="
snapshot_counters "after"

echo "=== Q2 (real signal): CXI octet deltas during the request window ==="
echo "  A non-zero tx/rx delta on ONE specific device, on both P and D, during"
echo "  this exact window is the actual evidence -- not the /proc/net/dev numbers"
echo "  above. Full time series in ${SHARED}/logs/cxi_poll_{p,d}.csv if you want"
echo "  to look at the shape rather than just the delta."
summarize_cxi_poll "${SHARED}/logs/cxi_poll_p.csv" "Prefill (${NODE_P})"
summarize_cxi_poll "${SHARED}/logs/cxi_poll_d.csv" "Decode (${NODE_D})"

# =============================================================================
# Q4 (wire half) -- how many CXI devices actually carried the transfer?
#
# The log half above reports what each worker BELIEVES it holds. This reports
# what moved. At TP=${TP} with one rail per GPU, ${TP} devices should light up
# on each node; one device carrying everything means the workers collapsed onto
# a single NIC and the node is running at 1/${TP} of its fabric bandwidth
# whatever the logs say.
#
# The threshold matters and is not cosmetic. These are node-wide NIC counters
# and this script is itself curling /metrics and ssh-ing across the same fabric
# while it polls, so every device shows a nonzero delta on every run. Counting
# "devices with delta > 0" would report 4/4 unconditionally and mean nothing.
# The separation is wide enough to be safe: background chatter is hundreds of
# KB over the whole window, while each rail's share of this transfer is tens of
# MB. 5 MB sits two orders of magnitude clear of the noise and an order below
# the signal.
#
# Direction: P is the source (RDMA READ, so P's NIC transmits the payload while
# D's receives it) and D issues the reads. tx is the signal on P and rx on D,
# but taking max(tx, rx) per device is correct on both sides and does not
# require the caller to know which role a file came from.
# =============================================================================
RAIL_ACTIVE_BYTES=${RAIL_ACTIVE_BYTES:-5000000}
count_active_rails() {
    local outfile=$1 label=$2
    if [ ! -s "${outfile}" ]; then
        echo "  ${label}: no samples -- cannot judge rail fan-out"
        return
    fi
    awk -F'[,=]' -v thr="${RAIL_ACTIVE_BYTES}" -v lbl="${label}" -v tp="${TP}" '
        { dev=$2
          if (!(dev in txfirst)) { txfirst[dev]=$4; rxfirst[dev]=$6 }
          txlast[dev]=$4; rxlast[dev]=$6
        }
        END {
          n=0; list=""
          for (d in txlast) {
            dt=txlast[d]-txfirst[d]; dr=rxlast[d]-rxfirst[d]
            m = (dt > dr ? dt : dr)
            if (m >= thr) { n++; list = list " " d "(" int(m/1048576) "MiB)" }
          }
          printf "  %s: %d of 4 devices above %.0f MB --%s\n", lbl, n, thr/1e6, (list == "" ? " none" : list)
          if (n >= tp)      print "    -> fan-out looks right: at least one device per TP worker moved payload."
          else if (n <= 1)  print "    -> COLLAPSED. All workers appear to be sharing one NIC; expect ~1/" tp " of node fabric bandwidth."
          else              print "    -> PARTIAL. Fewer active devices than TP workers; some workers are sharing a NIC."
        }' "${outfile}"
}
echo ""
echo "=== Q4 (wire half): how many CXI rails carried the transfer ==="
count_active_rails "${SHARED}/logs/cxi_poll_p.csv" "Prefill (${NODE_P})"
count_active_rails "${SHARED}/logs/cxi_poll_d.csv" "Decode  (${NODE_D})"
echo "  Cross-check against the log half above. Log says '${TP} x 1 rails' and wire"
echo "  says ${TP} active devices -> the one-rail-per-GPU partition holds under vLLM,"
echo "  and the Nemotron runs can assume ~88 GB/s of node fabric rather than ~24."
echo "  Log agrees but wire says 1 -> the rail managers each hold one NIC, but it"
echo "  is the SAME NIC; that is a topology-grouping problem, not a vLLM one."

# =============================================================================
# Q1 verdict -- read the metric diffs, not the clock.
#
# The old verdict here claimed disagg TTFT "should be noticeably LOWER than
# baseline". For a single request through 1P1D that is simply false, and
# treating it as the pass criterion would have failed a perfectly healthy run.
# Disagg does the SAME prefill work, then adds a KV transfer over the fabric
# and an extra proxy hop. One request in isolation is expected to be SLOWER.
# Disagg wins on aggregate throughput under load -- prefill and decode stop
# contending for the same GPU -- which is `vllm bench serve`'s job, not this
# script's. Timing is printed here only as a sanity check that neither path
# fell off a cliff.
#
# What actually decides Q1 is the pair of metric diffs.
# =============================================================================
echo ""
# NixlConnector's own telemetry, and the best single piece of evidence in the
# whole run -- better than anything this script computes. It reports transfer
# count, bytes, descriptor count and achieved throughput straight from the
# connector, so it distinguishes "a transfer happened" from "two servers each
# returned 200" with no inference at all. It is logged on the CONSUMER (D),
# since that is the side that issues the READ. Found by reading a passing run's
# log, not by design; surfaced here so it is never buried again.
#
# READ THIS LINE DIFFERENTLY AT TP>1. Every field changes meaning, and three
# of the four cross-checks that held at TP=1 no longer do.
#
# The stats from all ${TP} workers are aggregated before they are logged, so
# the line covers the whole engine, not rank 0 --
#   kv_connector/utils.py:120-130   accumulate kv_output.kv_connector_stats
#                                   across every worker's ModelRunnerOutput
# and then reduced to the printed fields at nixl/stats.py:86-116. Therefore:
#
#   Num successful transfers   ${TP}x what TP=1 showed for the same request.
#                              One record per worker, not one per request. A
#                              value that is not a multiple of ${TP} means some
#                              worker did not transfer -- which is a finding.
#   Avg MB per transfer        PER WORKER (total_mb / n, stats.py:111), so it
#                              is the per-rank payload. For THIS model that is
#                              dominated by the ~42.7 MB of constant Mamba
#                              state, not by KV: a short-prompt request should
#                              land near 45 MB and a long-prompt one should
#                              exceed it by 4,096 B per token. Anything close
#                              to zero means the SSM state is not moving even
#                              though attention KV is -- see the byte
#                              arithmetic at the top of the file.
#   Throughput (MB/s)          total_mb / SUM of the per-transfer durations
#                              (stats.py:113-114). The ${TP} workers transfer
#                              CONCURRENTLY, so summing their times divides the
#                              real aggregate rate by roughly ${TP}. This field
#                              UNDERSTATES achieved bandwidth at TP>1 and is not
#                              a bandwidth measurement. Use the CXI octet deltas
#                              over the wall-clock window for that.
#   Avg number of descriptors  NOT simply the layer count for this model, and
#                              this is the field to read most carefully. A
#                              dense transformer registers one descriptor per
#                              layer. Here the 8 attention layers contribute FA
#                              descriptors over physical blocks, the 40 Mamba
#                              layers contribute SSM descriptors over logical
#                              blocks, and each conv state is decomposed into
#                              THREE contiguous x/B/C sub-projections
#                              (base_worker.py, the reason DS layout is
#                              mandatory). Expect it well above 48 and roughly
#                              stable across requests. A count that matches the
#                              attention layer count alone would mean the Mamba
#                              half is not being registered at all -- which is
#                              exactly the failure this run exists to rule out.
#
# The one cross-check that survives unchanged: the CXI tx delta should exceed
# (Num successful transfers x Avg MB per transfer) by a few percent of wire
# framing overhead.
echo "=== vLLM's own NIXL transfer telemetry ==="
# POLL, do not grep once. This line is NOT emitted when the transfer completes.
# It comes from vLLM's PERIODIC stats logger on its own interval (10s by
# default), so it lands seconds after the KV has already moved, and
# settle_cxi_poll returns as soon as the fabric goes quiet -- reliably inside
# that window. Measured on the Qwen TP=4 rig, run 18:42:05: transfer finished
# 18:45:15, this check ran ~18:45:18, the metrics line appeared 18:45:23. A run
# whose KV transfer was perfect in every other respect printed "NONE FOUND".
#
# Expect the gap to be WIDER here, not narrower: a 120B transfer moves far more
# per request, so the fabric takes longer to go quiet and the stats tick you are
# waiting for is correspondingly further out. Raise KV_METRICS_WAIT_S before
# concluding anything from a bare absence.
_kvm=""
_kvm_tries=0
while [ ${_kvm_tries} -lt ${KV_METRICS_WAIT_S:-20} ]; do
    _kvm=$(grep -h 'KV Transfer metrics' ${SHARED}/logs/p.log ${SHARED}/logs/d.log 2>/dev/null | tail -5)
    [ -n "${_kvm}" ] && break
    _kvm_tries=$(( _kvm_tries + 1 ))
    sleep 1
done
if [ -n "${_kvm}" ]; then
    printf '%s\n' "${_kvm}" | sed 's/^/  /'
    [ ${_kvm_tries} -gt 0 ] && echo "  (appeared ${_kvm_tries}s after the fabric went quiet -- stats interval, not a problem)"
else
    echo "  NONE FOUND after ${KV_METRICS_WAIT_S:-20}s of waiting. NixlConnector logs this"
    echo "  line once a transfer completes and the next stats interval ticks, so its"
    echo "  absence THIS LONG after the request means no KV moved -- regardless of"
    echo "  what the HTTP status codes and the metric diffs below suggest."
fi

echo ""
echo "=== Q1 verdict: read the metric diffs above ==="
diff_metrics ${M}_p_b_before ${M}_p_b_after "Q1b PREFILL node -- expected: a full prefill of the prompt"
diff_metrics ${M}_d_b_before ${M}_d_b_after "Q1b DECODE node  -- expected: request served, prefill work much smaller than Q1a"
echo ""
echo "  PASS looks like:"
echo "    * Q1a moved D's counters and left P's flat  (P is not in that path)."
echo "    * Q1b moved BOTH. P participating at all is the single most robust"
echo "      signal -- it is name-independent and it is the whole claim: the"
echo "      proxy really did route prefill to the other node."
echo "    * D's prefill-side work in Q1b is much smaller than in Q1a, because"
echo "      D received the KV instead of computing it."
echo "  FAIL looks like:"
echo "    * Q1b's D diff resembles Q1a's D diff and P stayed flat -- D quietly"
echo "      did its own prefill, i.e. the connector fell back and 200s mean"
echo "      nothing."
echo ""
echo "  Timing (sanity only -- disagg being slower here is EXPECTED):"
grep -h 'time_starttransfer' ${SHARED}/logs/timings.txt 2>/dev/null || true

# =============================================================================
# Q3 (transfer half) -- the startup half ran before the requests. This is the
# part that can only be checked afterwards: did LIBFABRIC actually carry a
# transfer, and did the shim stay in the loop?
# =============================================================================
echo ""
echo "=== Q3: transport evidence after the transfer ==="
for role in p d; do
    log=${SHARED}/logs/${role}.log
    echo "  -- ${role}.log --"
    echo "     LIBFABRIC/CXI mentions:  $(grep -ci 'libfabric\|cxi' "${log}" 2>/dev/null || echo 0)"
    echo "     shim announcements:      $(grep -c 'fi_getinfo_shim' "${log}" 2>/dev/null || echo 0)"
    # A raw `grep -ci ucx` is useless as a fallback detector: a healthy
    # LIBFABRIC run has exactly two benign UCX lines on each side, confirmed by
    # reading them on the first passing run --
    #   nixl_utils.py:32  vLLM sets UCX_RCACHE_MAX_UNRELEASED unconditionally,
    #                     whatever backend was requested. Env var, not usage.
    #   nixl_plugin_manager.cpp:621  "Discovered backend plugin: UCX" is the
    #                     plugin manager enumerating what is loadable.
    # Counting those as failures would fire on every good run, which just
    # teaches you to ignore the check. Subtract the two known-benign patterns
    # and report what is left -- that residue is what would need explaining.
    ucx_other=$(grep -i 'ucx' "${log}" 2>/dev/null \
        | grep -v 'UCX_RCACHE_MAX_UNRELEASED' \
        | grep -vc 'Discovered backend plugin: UCX' || true)
    echo "     unexplained UCX lines (want 0 on a LIBFABRIC run): ${ucx_other}"
    if [ "${ucx_other}" -ne 0 ] 2>/dev/null; then
        grep -i 'ucx' "${log}" 2>/dev/null \
            | grep -v 'UCX_RCACHE_MAX_UNRELEASED' \
            | grep -v 'Discovered backend plugin: UCX' | head -5 | sed 's/^/       ! /'
    fi
    grep -i 'nixl\|libfabric\|fi_getinfo_shim' "${log}" 2>/dev/null | tail -20 | sed 's/^/       /'
done
echo ""
echo "  A LIBFABRIC run with UCX mentions > 0 is the failure mode to watch for:"
echo "  a missing backend is a WARNING in NIXL's python API, not an error, so"
echo "  the agent comes up on UCX and everything downstream still returns 200s."

echo ""
echo "Full logs at: ${SHARED}/logs/"

# =============================================================================
# KEEP_ALIVE -- hold both servers up so an external client can benchmark them.
#
# Everything above is the bring-up evidence chain and runs unchanged. This block
# only decides whether the script exits (and so tears the servers down) or
# parks.
#
# Why the INT/TERM trap is swapped for the duration of the hold: the standing
# `trap cleanup EXIT INT TERM` runs cleanup() on SIGINT and then RESUMES, because
# the handler never exits. During a blocking loop that is the wrong shape twice
# over -- the loop would keep spinning after the user asked to stop, and when it
# eventually did fall through, the EXIT trap would run cleanup() a SECOND time.
# So for the hold we install a handler that only raises a flag, let the loop
# notice it and fall out, and let the normal EXIT trap do the one teardown. Do
# not "simplify" this back to a single trap.
# =============================================================================
if [ "${KEEP_ALIVE:-0}" = "1" ]; then
    echo ""
    echo "============================================================"
    echo "KEEP_ALIVE=1 -- servers held up. Ctrl-C here is the teardown."
    echo "============================================================"
    echo "  Disagg (1P:1D) entry point -- bench this arm:"
    echo "      http://${P_IP}:${PROXY_PORT}"
    echo "  Decode server, direct -- the proxy-ceiling control:"
    echo "      http://${D_IP}:${D_PORT}"
    echo "  Prefill server, direct:"
    echo "      http://${P_IP}:${P_PORT}"
    echo ""
    echo "  Model string for --model:  ${MODEL}"
    echo "  Run dir:                   ${SHARED}"
    echo "  max-model-len:             ${MAX_MODEL_LEN}"
    echo ""
    # Print the invocations rather than the endpoints alone. Reconstructing them
    # from memory is how 2026-09-11 got benched against the PREVIOUS run's IP
    # from a shell that had never sourced common_env.sh -- two mistakes that
    # both surface as a Squid error page and read as a dead server.
    echo "  From another shell, on either node:"
    echo ""
    echo "      source ${SHARED}/common_env.sh"
    echo ""
    echo "      # the disagg arm, through the proxy"
    echo "      ARM=disagg \\"
    echo "      BASE_URL=http://${P_IP}:${PROXY_PORT} \\"
    echo "      METRICS_URLS=http://${P_IP}:${P_PORT},http://${D_IP}:${D_PORT} \\"
    echo "      RUN_DIR=${SHARED} \\"
    echo "          bash ${SCRIPT_DIR}/bench_arm.sh"
    echo ""
    echo "      # the proxy-ceiling control, D direct"
    echo "      ARM=d-direct \\"
    echo "      BASE_URL=http://${D_IP}:${D_PORT} \\"
    echo "      METRICS_URLS=http://${D_IP}:${D_PORT} \\"
    echo "      RUN_DIR=${SHARED} \\"
    echo "          bash ${SCRIPT_DIR}/bench_arm.sh"
    echo ""
    echo "  METRICS_URLS points at P and D directly because the proxy serves no"
    echo "  /metrics (CLOSED.md). The source line is NOT optional: without it a"
    echo "  fresh shell has no conda env, no HF_HOME/HF_TOKEN for the tokenizer,"
    echo "  and no no_proxy for these HSN addresses -- so curl reaches"
    echo "  proxy.alcf.anl.gov and every failure looks like a dead server."
    echo "  Take these addresses from THIS banner; no_proxy is generated per run,"
    echo "  so an IP from an earlier allocation is never in it."
    echo ""
    # The proxy-ceiling control is not optional book-keeping -- a single-process
    # asyncio proxy fronting a 120B disagg pair is the most likely thing to cap
    # measured throughput, and a cap there looks exactly like "disaggregation
    # does not help". Benching D directly puts a number on the proxy before the
    # comparison is drawn, not after someone disputes it.
    echo "  Bench the proxy AND D-direct. If both arms plateau at the same"
    echo "  number, the proxy is the ceiling and neither number is about"
    echo "  disaggregation."
    echo ""

    _ka_stop=0
    # shellcheck disable=SC2064
    trap '_ka_stop=1; echo ""; echo "Interrupt received -- releasing servers."' INT TERM

    _ka_t0=$(date +%s)
    while [ "${_ka_stop}" -eq 0 ]; do
        # mpiexec is the liveness source of truth: if PALS tore the application
        # down, the endpoints above are dead no matter what the last heartbeat
        # said, and holding the terminal open is just misleading.
        if ! kill -0 "${MPIEXEC_PID}" 2>/dev/null; then
            echo ""
            echo "mpiexec (${MPIEXEC_PID}) exited -- both servers are gone."
            echo "Check ${SHARED}/logs/mpiexec.log; a PALS-level failure never"
            echo "reaches p.log or d.log."
            break
        fi
        sleep "${KEEP_ALIVE_POLL_S}"
        # A bare "still alive" is worth little. Report elapsed time, whether each
        # server still answers /health, and D's ENGINE line, so a wedged engine
        # and a busy one stop looking identical from the outside.
        #
        # Not `tail -n 1 d.log`. That was reporting this loop's own /health probe
        # back to itself -- the access-log entry is almost always the last line,
        # because the loop writes one every KEEP_ALIVE_POLL_S. On 2026-09-11 the
        # colocated arm printed `health=200` once a minute for twelve minutes
        # while the engine behind it had not stepped once.
        #
        # /health cannot substitute: APIServer is a separate process from
        # EngineCore and keeps answering 200 long after the engine has stopped.
        # loggers.py:310 is the engine's own periodic stats line -- it advances
        # about every 10 s while work is in flight and stops the instant it is
        # not, which is the signal we actually want.
        #
        # D is the right single watchpoint for BOTH failure directions: if P
        # wedges, D starves and its line freezes too.
        _ka_el=$(( $(date +%s) - _ka_t0 ))
        _ka_hp=$(curl -s -o /dev/null -w '%{http_code}' "http://${P_IP}:${P_PORT}/health" 2>/dev/null || echo "---")
        _ka_hd=$(curl -s -o /dev/null -w '%{http_code}' "http://${D_IP}:${D_PORT}/health" 2>/dev/null || echo "---")
        _ka_eng=$(grep -F 'loggers.py:310' "${SHARED}/logs/d.log" 2>/dev/null | tail -n 1 \
                    | sed 's/.*Engine 000: //' | cut -c1-100)
        printf '  [keep-alive %02d:%02d:%02d] P=%s D=%s | %s\n' \
            $(( _ka_el / 3600 )) $(( (_ka_el % 3600) / 60 )) $(( _ka_el % 60 )) \
            "${_ka_hp}" "${_ka_hd}" \
            "${_ka_eng:-<D idle -- no requests yet; both servers are up>}"

        # An engine with nothing to do also stops logging, and that is NOT a
        # stall -- it is the normal state between benches. Only an unchanged
        # line that still claims in-flight work is evidence of a wedge.
        case "${_ka_eng}" in
            ""|*"Running: 0 reqs, Waiting: 0 reqs"*) _ka_busy=0 ;;
            *)                                       _ka_busy=1 ;;
        esac
        if [ "${_ka_busy}" -eq 1 ] && [ "${_ka_eng}" = "${_ka_last:-}" ]; then
            _ka_stall=$(( ${_ka_stall:-0} + 1 ))
        else
            _ka_stall=0
        fi
        _ka_last=${_ka_eng}

        # Two identical polls is >=2*KEEP_ALIVE_POLL_S with no engine step,
        # against a line that normally advances every 10 s. Not slowness.
        if [ "${_ka_stall}" -ge 2 ]; then
            echo "      ^^ STALLED: no D engine step in $(( _ka_stall * KEEP_ALIVE_POLL_S ))s."
            echo "         health=200 proves nothing -- APIServer is a separate pid."
            echo "         Capture the stacks BEFORE Ctrl-C, or the cause dies with it:"
            echo "             pgrep -a -f 'EngineCore|VLLM::Worker_TP'"
            echo "             py-spy dump --pid <EngineCore pid>"
            echo "             py-spy dump --pid <Worker_TP0 pid>"
            echo "         Do this on BOTH nodes -- a stalled D can be a starved D."
            echo "         Then nvidia-smi, then Ctrl-C."
        fi
    done

    # Restore the original disposition before falling through, so the EXIT trap
    # is the single teardown path exactly as it is on a KEEP_ALIVE=0 run.
    trap cleanup EXIT INT TERM
    echo "Releasing servers -- cleanup() follows."
fi
