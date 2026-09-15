#!/bin/bash
set -uo pipefail

# =============================================================================
# run_inkling_colocated_N2.sh -- the COLOCATED arm for INKLING-SMALL.
#
# TWO serving nodes, one role, one HTTP endpoint, no NIXL, no proxy. Plus a
# THIRD node that serves nothing and runs the benchmark client. It brings the
# pair up, waits for /health, starts the frontend samplers, and parks so that
# bench_arm.sh can drive it from the client node:
#
#     ARM=inkling-colocated MODEL=<repo id> GPUS=8 \
#         BASE_URL=http://<HEAD_IP>:8100 RUN_DIR=<run dir> bash bench_arm.sh
#
# MODEL and GPUS are not optional: bench_arm.sh defaults MODEL to Nemotron and
# GPUS to 4. The keep-alive banner prints the whole command with both filled in
# -- copy it from there rather than from here.
#
# WHY THE COLOCATED BASELINE IS NOW MULTI-NODE, AND WHY THAT IS GOOD NEWS.
# Inkling-Small is 266B at BF16 = 532 GB of weights against ~345 GB usable per
# node. (265,956,439,090 parameters, all BF16 but for 10,280 in F32, read off
# the HF model index on 2026-09-14; earlier notes in this project said 276B /
# 552 GB from the model card, which was rounded up. The conclusion is unmoved.)
# It does not fit on one node, so the BASELINE is forced across the fabric
# too (MEMORY_2026-09-12b.md). That removes the confound in Nemotron's 7.1%
# per-GPU deficit for free: there, disagg crossed Slingshot and colocated did
# not, so part of the deficit was the fabric rather than disaggregation. Here
# both arms cross it and the comparison is clean.
#
# THE SHAPE: one role = two nodes, DP=2 x TP=4, expert-parallel across both.
#
#     --data-parallel-size 2 --tensor-parallel-size 4 --enable-expert-parallel
#
# Attention tensor-parallelism stays inside a node, on NVLink (NV6 between all
# four GH200 modules, provenance.txt records the table). Only the expert
# all-to-all crosses Slingshot. NEVER --tensor-parallel-size 8: that puts the
# attention all-reduce on the fabric and cripples the baseline for a reason
# that has nothing to do with disaggregation -- the d-direct trap in a new
# costume (MEMORY_2026-09-12b.md, CLOSED.md).
#
# MULTI-NODE DP IS HEAD + HEADLESS, NOT TWO SYMMETRIC RANKS. The Nemotron
# pattern of `mpiexec -n 2 -ppn 1` launching one IDENTICAL role per node does
# not carry over. One node runs the API servers and data-parallel rank 0; the
# other runs `--headless` and hosts rank 1 only. Both are launched from ONE
# mpiexec (see the VNI section below) and sort themselves out by hostname.
#
# NO RAY, AND NO NEW LAUNCHER. This was settled by source read against the local
# vllm-v0.27.1/ tree (MEMORY_2026-09-14b.md); it is recorded here so it is not
# re-derived. The worry is that `mp` classically meant single-node. It does not
# bind here, for two independent reasons:
#
#   1. IN THIS SHAPE `mp` IS NEVER ASKED TO CROSS A NODE. Head + headless is
#      TWO SEPARATE ENGINES, each with world_size = TP = 4. --nnodes is not
#      passed, so it stays 1, nnodes_within_dp = 1 (config/parallel.py:714-720)
#      and local_world_size = 4 (:722-724). MultiprocExecutor then spawns
#      `for local_rank in range(self.local_world_size)` -- four workers, all
#      node-local (v1/executor/multiproc_executor.py:176). The cross-node work
#      is carried by the DP ZMQ handshake on --data-parallel-address /
#      --data-parallel-rpc-port and by the NCCL expert all-to-all. NEITHER is
#      the executor backend.
#   2. `mp` IS GENUINELY MULTI-NODE IN 0.27.1 ANYWAY. multiproc_executor.py:165
#      offsets local ranks by local_world_size * node_rank_within_dp, :262
#      asserts world_size % nnodes_within_dp == 0, and :135,:209 branch on a
#      node_rank_within_dp == 0 DP-group leader. That is the machinery behind
#      the documented `--nnodes 2 --node-rank 1 -d-e-b mp --headless` form
#      (docs/deployment/integrations/kthena.md:220).
#
# The principle worth keeping: RAY'S JOB WAS REMOTE PROCESS *PLACEMENT*, NOT
# COMMUNICATION. mpiexec/PALS already does placement -- one rank per node, one
# application, one VNI -- so each `mp` executor only ever fans out locally and
# NCCL carries what crosses the fabric. "No Ray" and "multi-node model" are not
# in tension, and this is the same reason Nemotron's `-n 2 -ppn 1` worked.
#
# --distributed-executor-backend mp IS THEREFORE PASSED EXPLICITLY BELOW, on
# both roles. Leaving it unset happens to yield mp today, but only because Ray
# is not *initialized*: ray>=2.55.0 IS installed in the Minerva env, so
# ray_found is true and the sole remaining guard is
# `placement_group is None and not ray_is_initialized()` (parallel.py:941-951).
# Anything that touches Ray earlier in a future session flips the backend
# silently, mid-study. Explicit makes it structural. Inside the ONE RULE.
#
# DO NOT REACH FOR external_launcher OR vllm_external_wrapper.sh. Considered and
# rejected: in this tree external_launcher appears only in entrypoints/llm.py
# and v1/engine/llm_engine.py -- the OFFLINE `LLM` class -- plus tests and two
# torchrun_*_offline.py examples. No hits under entrypoints/openai/ or
# cli/serve.py. It cannot serve an HTTP endpoint, so it forecloses
# `vllm bench serve`, the /metrics scrape and the ss sampler in one move. It
# would also make --numa-bind a no-op (no worker subprocesses left for
# numa_utils to wrap in numactl) and force VLLM_ENABLE_V1_MULTIPROCESSING=0
# (parallel.py:907-909), which breaks comparability with the whole Nemotron
# frontend study. The wrapper is Intel/Aurora machinery besides (CCL_*,
# NEO_CACHE_*); on GH200 those exports are inert.
#
# PIPELINE PARALLELISM was priced and not taken: PP=2 x TP=4 is also Ray-free
# (parallel.py:921 forces mp on CUDA when nnodes > 1, :963-969 rejects Ray
# outright), but it puts pipeline bubbles into inter-token latency, and the ITL
# distribution is the strongest result in the report. Revisit only if DP+EP
# fails to fit.
#
# IGNORE docs/serving/parallelism_scaling.md:33 in the vendored tree. It still
# says "Ray for multi-node inference and native Python multiprocessing for
# single-node". The code and kthena.md both contradict it.
#
# THE ONE RULE. This must issue the SAME `vllm serve` command as
# run_inkling_1p1d_N4.sh, differing by exactly three things:
#
#     1. no --kv-transfer-config
#     2. no VLLM_NIXL_SIDE_CHANNEL_HOST / _PORT / NIXL_LOG_LEVEL
#     3. batching -- this arm takes the per-knob MAXIMUM of P's and D's
#        settings, argued below; the disagg arm tunes them in opposite
#        directions, which is the entire point of disaggregation.
#
# Every other flag, env var and default is identical. If you change a serve flag
# here, change it there in the same commit, or the comparison silently starts
# measuring the difference in flags. Both launchers write their generated
# launcher into the run dir precisely so the two can be diffed after the fact:
#
#     diff <(sed -n '/^exec vllm serve/,$p' <colo-run>/launch_role.sh) \
#          <(sed -n '/^exec vllm serve/,$p' <disagg-run>/launch_role.sh)
#
# THE POOL GATE IS PART OF THE RULE, BUT ITS THRESHOLD IS NOT SHARED. Both
# launchers must gate on the KV pool before anything is benched (see the gate
# near the end of this file, and the Nemotron disagg launcher that had none).
# What does NOT carry across is the NUMBER. The pool is whatever is left after
# weights and the activation profile, and the activation profile is driven by
# --max-num-batched-tokens, which is exactly the one knob the ONE RULE lets the
# arms differ on:
#
#     colocated   16384 tok / 256 seq     (per-knob max of the two below)
#     P           16384 tok /  32 seq
#     D            2048 tok / 256 seq
#
# Three configurations, three pools. So EXPECT_POOL from this arm is meaningless
# on either disagg role, and a single EXPECT_POOL across P and D would be
# meaningless on at least one of them. run_inkling_1p1d_N4.sh takes
# EXPECT_POOL_P and EXPECT_POOL_D as separate gates, each established from that
# role's own first healthy launch, and takes the minimum WITHIN a role across
# that role's ranks -- never across roles. This arm gets away with one number
# only because head and headless are configured identically here, which is what
# makes a single minimum mean anything.
#
# ENVIRONMENT COMES FROM emit_common_env.sh AND THE SAMPLER FROM
# emit_conn_sampler.sh, both sourced, not copied. Those are the single copies
# both arms share; a second copy is how the arms quietly stop being comparable.
# workload_profile.sh is the third.
#
# USAGE  (run this ON THE CLIENT NODE -- see "Resolve the nodes" below)
#   bash run_inkling_colocated_N2.sh                        # holds the pair up
#   WORKLOAD=iter32k MAX_MODEL_LEN=43008 bash run_inkling_colocated_N2.sh
#   EXPECT_POOL=<tokens> bash run_inkling_colocated_N2.sh   # gate on the pool
#   KEEP_ALIVE=0 bash run_inkling_colocated_N2.sh           # up, verify, down
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The workload point this server is being provisioned FOR. Read from the same
# file bench_arm.sh reads, so the context budget set here and the ISL/OSL sent
# there come from ONE definition. They are decided hours apart, on different
# nodes, by different commands; when they disagree the symptom is an HTTP 400
# per request, which is instant and therefore reads on the progress bar as a
# fast request rather than as a fault (2026-09-11).
WORKLOAD=${WORKLOAD:-iter32k}
# shellcheck source=workload_profile.sh
source "${SCRIPT_DIR}/workload_profile.sh"
load_workload_profile "${WORKLOAD}" || exit 1

# =============================================================================
# LOCKED -- must match run_inkling_1p1d_N4.sh exactly
# =============================================================================
# Every default in this block is shared with the disagg launcher and is a
# same-commit change if either moves. Rationale for the ones that are genuinely
# Inkling decisions is here; the ones inherited from the Nemotron pair are
# marked as such and argued there.

# --- MODEL: resolved from the cache, not guessed ------------------------------
# THE REPO ID IS thinkingmachines/Inkling-Small -- confirmed against the HF API
# on 2026-09-14, public and ungated, apache-2.0, 32 shards plus a separate
# mtp.safetensors. The scale-up is thinkingmachines/Inkling (952B, 108 shards).
# Neither is a BF16 *variant* repo: these ARE the BF16 originals, and the
# quantized ones live at the parallel *-NVFP4 ids.
#
# It is still resolved from the cache rather than baked in, because the scan
# answers a second question a hardcoded string cannot: whether the weights are
# actually THERE. `bash download_model.sh` has to have completed first; 532 GB
# does not arrive inside a health-check window, and the failure mode of finding
# out late is an abandoned partial download plus a spent allocation.
# MODEL=<repo-id> overrides and skips the scan.
HF_CACHE_ROOT=${HF_CACHE_ROOT:-/vast/draco/tara/projects/Tara_Deployment/software/model-weights/hub}
MODEL=${MODEL:-}
if [ -z "${MODEL}" ]; then
    mapfile -t _cands < <(find "${HF_CACHE_ROOT}" -maxdepth 1 -type d \
        -name 'models--*Inkling*' 2>/dev/null | sort)
    # Exclude thinkingmachines/Inkling, the 952B scale-up: it is not this run,
    # it matches a bare *Inkling* glob, and it is expected to be sitting in the
    # same cache (both are downloaded together). HANDOFF item 4 -- 1,905 GB of
    # BF16 weights against ~345 GB usable per node is 7-8 nodes per replica
    # before any KV, and NVFP4-on-SM90 has to be resolved first.
    _small=()
    for _c in "${_cands[@]+"${_cands[@]}"}"; do
        case "${_c}" in *[Ss]mall*) _small+=("${_c}") ;; esac
    done
    if [ "${#_small[@]}" -eq 1 ]; then
        MODEL=$(basename "${_small[0]}")
        MODEL=${MODEL#models--}
        MODEL=${MODEL//--/\/}
        echo "Resolved MODEL from the cache: ${MODEL}"
    else
        echo "FATAL: could not resolve the Inkling-Small repo id." >&2
        echo "  Looked for models--*Inkling*Small* under ${HF_CACHE_ROOT}" >&2
        echo "  and found ${#_small[@]} match(es):" >&2
        printf '    %s\n' "${_small[@]+"${_small[@]}"}" >&2
        echo "" >&2
        echo "  Either the weights are not downloaded --" >&2
        echo "      MODEL=<repo-id> bash ${SCRIPT_DIR}/download_model.sh" >&2
        echo "  -- or the glob is wrong for this checkpoint. Pass it directly:" >&2
        echo "      MODEL=<repo-id> bash ${0##*/}" >&2
        echo "" >&2
        echo "  Do NOT let this be guessed. A wrong repo id fails after the" >&2
        echo "  health-check window opens, which costs an allocation, and a" >&2
        echo "  RIGHT-LOOKING wrong one silently serves a different model." >&2
        exit 1
    fi
fi

# --- Parallelism. Never TP=8. -------------------------------------------------
DP=${DP:-2}                 # one data-parallel rank per node
TP=${TP:-4}                 # four GPUs per node, all-reduce stays on NVLink
DP_LOCAL=${DP_LOCAL:-1}     # one rank per node: head hosts 0, headless hosts 1
if [ "${TP}" -gt 4 ]; then
    echo "FATAL: TP=${TP}. Tensor parallelism must not leave the node." >&2
    echo "  TP=8 spans two nodes, which puts the attention all-reduce on" >&2
    echo "  Slingshot and cripples the baseline for a reason unrelated to" >&2
    echo "  disaggregation (MEMORY_2026-09-12b.md). Use DP to add nodes." >&2
    exit 1
fi

GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-auto}

# EXPERT PARALLELISM IS ON, and unlike Nemotron it is not a knob to A/B later.
# Inkling-Small has 256 routed experts plus 2 shared. DP=2 x TP=4 builds an
# expert group of 8 ranks spanning BOTH data groups (parallel_state.py:1918-1950,
# mirrored at models/inkling/nvidia/moe.py:264-276, confirmed by source read in
# MEMORY_2026-09-12d.md), and 256 divides by 8 exactly so there is no expert
# padding (moe.py:440). Without it each rank would hold a quarter of all 256
# experts and the model would not fit the way the sizing assumes.
#
# The expert counts are no longer inferred: the checkpoint's own config.json
# reads n_routed_experts 256, n_shared_experts 2, num_experts_per_tok 6
# (text_config, read 2026-09-14). Note that CLOSED.md parks EXPERT_PARALLEL=1
# as a NEMOTRON A/B override -- 512 experts, top-k 22, off by default there.
# That entry is about a different model and does not apply to this script.
# EXPERT_PARALLEL=0 is now honoured (it was not before -- see the EXTRA block),
# but it is a debugging escape hatch, not an arm.
EXPERT_PARALLEL=${EXPERT_PARALLEL:-1}

# --- HMA: explicitly ON, and for Inkling it is load-bearing, not a tripwire ---
# Inkling's sconv short-convolution state emits a SlidingWindowSpec, not a
# MambaSpec (models/inkling/nvidia/sconv_swa_attn.py, and see the dtype note
# below). NixlConnector's _is_hma_required is
#
#     not disable_hybrid_kv_cache_manager AND any group that is not
#     FullAttentionSpec                        (base_scheduler.py:83-90,
#                                               base_worker.py:285-291)
#
# so with 35 sliding-window layers and 7 global ones, HMA on means the connector
# CLIPS each local layer's transfer to cdiv(window, block) + 1 blocks
# (base_scheduler.py:123-136). Turn HMA off and that clipping is skipped
# (base_scheduler.py:255 returns early when not _is_hma_required), so the local
# layers are treated as if they had unbounded context. It must stay false --
# i.e. this flag must stay as written -- on BOTH roles.
#
# CHECKPOINT GEOMETRY, no longer open. MEMORY_2026-09-12d.md carried three
# values as download-time unknowns; all three are now read off
# thinkingmachines/Inkling-Small config.json (text_config, 2026-09-14):
#
#     sconv_kernel_size        4
#     swa_head_dim             128      swa_num_attention_heads   32
#     swa_num_key_value_heads  8        sliding_window_size       512
#
# and local_layer_ids lists 35 of the 42 layers, which CONFIRMS the 35 local /
# 7 global split this block already assumed. The clip above is therefore
# cdiv(512, block) + 1 per local layer -- 33 blocks at block_size 16, against
# whatever the global layers carry at full context. That ratio is the whole
# reason a mixed-spec transfer is worth measuring, and it is the number to
# check the first disagg smoke run against.
#
# Passed explicitly for the same reason the Nemotron pair passes it: unset means
# vLLM decides and can silently turn it OFF with a warning, after which a hybrid
# model fails at startup for a reason that does not mention HMA (vllm.py:1605-
# 1676). Explicit ON raises at config time instead. A tripwire AND a setting.
HMA_FLAG=${HMA_FLAG:---no-disable-hybrid-kv-cache-manager}

# Inherited from the Nemotron pair unchanged, for the measurement reason rather
# than the legality one: if two requests share a prompt prefix, a cache hit
# looks fast for the wrong reason and, on the disagg arm, lets the consumer skip
# pulling state entirely. Off on both arms or neither.
PREFIX_CACHING_FLAG=${PREFIX_CACHING_FLAG:---no-enable-prefix-caching}

# --- NOT SET, and each absence is a decision --------------------------------
#
# --mamba-ssm-cache-dtype / --mamba-cache-dtype / VLLM_SSM_CONV_STATE_LAYOUT
#     All three are Nemotron machinery and NONE of them reaches Inkling.
#     InklingConvState takes model_config.dtype and hard-asserts bfloat16
#     (sconv_swa_attn.py:194-198); the config hard-codes bf16 for conv and
#     temporal state as well (configs.py:196). Because the sconv emits a
#     SlidingWindowSpec rather than a MambaSpec, NixlConnector's DS-layout
#     assert is gated on `_has_mamba` (base_worker.py:305-317) and never fires,
#     so VLLM_SSM_CONV_STATE_LAYOUT has no effect here at all.
#
#     THE CONSEQUENCE IS WORTH STATING PLAINLY, because it inverts the Nemotron
#     hazard: there, a P/D conv dtype mismatch would have produced GARBAGE
#     rather than an error. Here the only lever is --dtype, and anything but
#     bfloat16 aborts at layer construction on both roles. This hazard fails
#     loudly (MEMORY_2026-09-12d.md, check 1). Setting these anyway would be
#     cargo cult carried across from a different model.
#
# --block-size
#     Left derived, same argument as the Nemotron pair: under HMA vLLM picks an
#     attention block size that makes the page sizes line up, and a hand-picked
#     value overrides half of a derived pairing. Read what it chose out of the
#     startup log first. BLOCK_SIZE=<n> to experiment; applies to both roles.
#
# --speculative-config
#     InklingMTPModel exists and speculative.py:568 only rewrites model_type to
#     inkling_mtp when a draft model is configured, so MTP is off by leaving
#     this unset. Keep it that way: with MTP on,
#     inter_token_latency_seconds stops being per-token and the ITL
#     distribution figure -- the strongest result in the report -- loses its
#     meaning (MEMORY_2026-09-12b.md).
BLOCK_SIZE=${BLOCK_SIZE:-}
ENFORCE_EAGER=${ENFORCE_EAGER:-}

# --- API_SERVER_COUNT: explicit, head node only, and never empty --------------
# TWO LANDMINES, both in entrypoints/cli/serve.py, both confirmed by source read
# (MEMORY_2026-09-12d.md) and re-checked against the local tree:
#
#   * On the HEADLESS node it is a HARD ERROR: serve.py:66-71 raises if
#     api_server_count is set with --headless, and run_headless() raises again
#     at :178. It is passed on the head node only, which is why it is appended
#     inside the role branch below rather than to the shared command.
#   * On the HEAD node, LEFT UNSET IT SILENTLY DEFAULTS TO data_parallel_size,
#     i.e. 2 -- not 1, and not 16 (serve.py:121). Two arms that both "left it
#     alone" would therefore both get 2, which happens to match, but the moment
#     one arm sets it the other is measuring a different frontend. It stays
#     explicit, and this script refuses an empty value rather than inheriting
#     a default that depends on DP.
#
# 16 carried over from the Nemotron study so the two model families are
# comparable. EXPECT IT TO BIND HARDER HERE, and do not spend a run
# rediscovering that: sixteen API server processes now front EIGHT GPUs per role
# instead of four, and all sixteen live on the HEAD node -- the headless node
# runs no frontend at all. The head node is therefore doing frontend work and
# engine work at once while the headless node does only engine work, and an
# expert all-to-all completes at the speed of its slowest rank. The standing
# prediction from that, recorded here so it can be checked rather than
# reconstructed: the head node's data-parallel rank is the slow one.
API_SERVER_COUNT=${API_SERVER_COUNT:-16}
if [ -z "${API_SERVER_COUNT}" ]; then
    echo "FATAL: API_SERVER_COUNT is empty. Omitting --api-server-count does" >&2
    echo "  not mean 1 -- serve.py:121 silently defaults it to" >&2
    echo "  data_parallel_size (${DP}). Pass a number." >&2
    exit 1
fi

# --- MAX_MODEL_LEN ------------------------------------------------------------
# Defaults to the selected profile's value, unlike the Nemotron pair. There, the
# colocated launcher deliberately defaulted to the FROZEN disagg script's 32768
# so that running both arms bare gave two comparable servers. Both Inkling
# launchers are new and neither is frozen, so both default to WL_MAX_MODEL_LEN
# and a bare run of each is comparable by construction -- which is strictly
# better than two matching wrong numbers. Pass it explicitly to both arms
# anyway if you override the profile.
#
# 8192 of slack over ISL+OSL, and it is not superstition: `vllm bench serve
# --dataset-name random` synthesises prompts that land NEAR --random-input-len,
# not on it, so an exactly-fitting budget rejects whichever prompts round up.
# Tried on 2026-09-11; the rejection is instant, so it registers on the progress
# bar as a completed request and reads as progress.
MAX_MODEL_LEN=${MAX_MODEL_LEN:-${WL_MAX_MODEL_LEN}}

# =============================================================================
# THE COLOCATED BATCHING DECISION -- the third and last legal difference
# =============================================================================
# Disagg runs two engines tuned in opposite directions:
#
#     P (prefill)  16384 tokens / 32 seqs    compute-bound, few big batches
#     D (decode)    2048 tokens / 256 seqs   bandwidth-bound, many sequences
#
# A colocated server does both jobs and can only pick one setting per knob, so
# copying either engine wholesale is a rigged baseline -- CLOSED.md lists "the
# Q1a arm as a co-located baseline" among the numbers that must never be quoted
# for exactly that reason.
#
# PER-KNOB MAXIMUM, unchanged from the Nemotron pair: for each knob
# independently, this arm gets whatever the disagg engine that OWNS that job
# gets, so it is never tighter than either on any axis and no reviewer can point
# at a flag and say the baseline was starved.
#
#     max-num-batched-tokens  16384 = max(P 16384, D 2048)
#     max-num-seqs              256 = max(P 32,    D 256)
#
# INHERITED, NOT RE-DERIVED FOR INKLING, and the difference matters in one
# place. On Nemotron, D's 256 was argued from the 41.9 MB/sequence constant
# Mamba state. Inkling has no such constant: its per-request cost is dominated
# by global-attention KV at 28,672 B/token unsharded, so at ISL 32k a request
# costs ~0.94 GB of global KV plus ~0.073 GB of clipped local window, ~1.01 GB
# (MEMORY_2026-09-12d.md). Against a per-node KV budget in the tens of GB that
# is order-100 resident sequences per data-parallel rank, so 256 is above the KV
# ceiling and KV binds first -- as it does on every run this rig has produced.
# 256 is therefore non-binding rather than generous. Leave it; the number that
# actually controls concurrency is the client's --max-concurrency.
#
# 16384 is kept for the reason it was kept on Nemotron, which does carry over: a
# 131072-token prompt chunks into exactly 8 pieces in BOTH arms, so prefill
# chunking is held constant and the only remaining difference is whether decode
# has to share the step. That is precisely the variable Figure 2 isolates.
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-16384}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}

# --- NUMA binding: first-class, explicit, never auto-detected -----------------
# --numa-bind wraps each worker subprocess spawn in `numactl --physcpubind=<cpus>
# --membind=<node>` (numa_utils.py:350-370) and binds the EngineCore too
# (core.py:1292). The mapping is passed EXPLICITLY because on Grace-Hopper the
# GPU's own HBM is itself a NUMA node -- these nodes report GPU NUMA IDs
# 4/12/20/28 against CPU NUMA affinity 0/1/2/3 -- and an auto-detect that picks
# the HBM node would --membind host allocations onto GPU memory (CLOSED.md).
#
# IT DOES NOT BIND THE API SERVERS. numa_utils' _get_numactl_worker_args covers
# TP/PP worker subprocesses and the EngineCore, nothing else (CLOSED.md). So on
# the head node the sixteen API server processes still float across all 288
# cores, including the ones the four workers are now bound to, and nothing
# excludes them. That is not a bug in this script and it is not fixable from
# here -- it is recorded because it is the leading unconfirmed explanation for
# the 2026-09-14 result that a local benchmark client moved ENGINE-side numbers
# by 14.4%. The sampler records each API server's last-run core every interval
# so the question can be answered from a run rather than from memory.
#
# THE SEPARATOR IS A SPACE, NOT A COMMA, AND THE LIST IS INDEXED BY GPU INDEX.
# `--numa-bind-nodes` is nargs='+' of int, so the comma form dies instantly at
# argparse with "Value 0,1,2,3 cannot be converted to <class 'int'>" -- rank 0
# exits code 2, PALS signal-15s rank 1, and the whole application is gone before
# a single weight is read (2026-09-15, the first Inkling launch, which is the
# first time this flag has EVER actually run in this project: on Nemotron it
# lives only inside a comment block at run_pd_nemotron_1p1d_N2.sh:1076-1087 and
# was never passed. A form that has only ever been written down is not a form
# that has been tested -- same trap as EXPERT_PARALLEL in MEMORY_2026-09-14c.md).
#
# The semantics are NOT "the set of NUMA nodes available". numa_utils.py:261-263
# indexes this list BY GPU INDEX and raises if gpu_index >= len(numa_bind_nodes),
# so it needs one entry per VISIBLE GPU, in GPU order. vLLM's own test uses
# [0, 0, 1, 1] for four GPUs across two NUMA nodes
# (tests/engine/test_arg_utils.py:605-621). Here the four GH200 modules are four
# separate Grace NUMA nodes, one per GPU, so the identity map is correct and the
# VALUES are unchanged from the comma version -- only the separator moves.
#
# Unquoted on purpose where it is consumed, so the four words split into four
# array elements. Verify the mapping on a role node if these numbers are ever in
# doubt: `numactl -H` for the CPU nodes, and
# `nvidia-smi --query-gpu=index,pci.bus_id --format=csv` against
# /sys/bus/pci/devices/<id>/numa_node for the GPU-to-node association. CPU
# affinity is 0/1/2/3; 4/12/20/28 are the HBM nodes and must NOT appear here.
NUMA_BIND=${NUMA_BIND:-1}
NUMA_BIND_NODES="${NUMA_BIND_NODES:-0 1 2 3}"

KEEP_ALIVE=${KEEP_ALIVE:-1}
KEEP_ALIVE_POLL_S=${KEEP_ALIVE_POLL_S:-60}
HEALTH_TRIES=${HEALTH_TRIES:-480}   # x5s = 40 min; 552 GB over 8 workers is slow
SAMPLE_INTERVAL_S=${SAMPLE_INTERVAL_S:-10}

PORT=${PORT:-8100}
DP_RPC_PORT=${DP_RPC_PORT:-29550}
RUNS_ROOT=${RUNS_ROOT:-/vast/draco/tara/projects/Tara_Deployment/software/testing/RUNS}
STAMP=$(date +%Y%m%d_%H%M%S)
SHARED=${SHARED:-${RUNS_ROOT}/inkling_colocated_dp${DP}tp${TP}_${STAMP}}

# =============================================================================
# Resolve the nodes. THIS SCRIPT RUNS ON THE CLIENT NODE.
# =============================================================================
# This is the structural change from every launcher before it, and it is the
# whole reason the 2026-09-14 session's result is built in rather than written
# down: moving the benchmark client off the serving node was worth +37.7%
# throughput and -14.4% engine work per token on an otherwise identical run
# (TECHNICAL_REPORT_1.tex S7). Engine-side quantities are immune to HOW
# connections were distributed. They are NOT immune to a client sharing a node.
#
# A rule written in a handoff gets followed until the evening it does not. So
# the rule is made structural instead: the node this script is running on is the
# CLIENT node, the remaining allocated nodes are the SERVING nodes, and the
# script refuses to give a serving role to its own node. There is then no way to
# run the bench on a serving node without deliberately defeating the launcher.
#
# THE ALLOCATION IS N+1. "N2" in the filename counts SERVING nodes, matching the
# convention of the Nemotron pair. This arm needs THREE:
#
#     head + headless   the role, 8 GPUs
#     client            this script, bench_arm.sh, nothing else
#
# The client node's GPUs are idle and that is correct: Figure 2 normalizes per
# GPU over the SERVING GPUs, so an idle node costs allocation and changes no
# number. CLOSED.md notes the login node is worth testing once for this job -- if
# it can reach the compute nodes' HSN addresses it costs no allocation at all.
if [ -z "${PBS_NODEFILE:-}" ] || [ ! -r "${PBS_NODEFILE}" ]; then
    echo "FATAL: no readable PBS_NODEFILE. This script does not run on a login node." >&2
    exit 1
fi
mapfile -t ALL_NODES < <(sort -u "${PBS_NODEFILE}")
CLIENT_HOST=$(hostname -s); CLIENT_HOST="${CLIENT_HOST%%.*}"

ROLE_NODES=()
for n in "${ALL_NODES[@]}"; do
    [[ "${n%%.*}" == "${CLIENT_HOST}" ]] && continue
    ROLE_NODES+=("$n")
done

NEED_ROLE_NODES=$(( DP ))
if [ "${#ROLE_NODES[@]}" -lt "${NEED_ROLE_NODES}" ]; then
    echo "FATAL: need ${NEED_ROLE_NODES} serving nodes PLUS this client node." >&2
    echo "  Allocation has ${#ALL_NODES[@]} node(s); this one (${CLIENT_HOST}) is the" >&2
    echo "  client, leaving ${#ROLE_NODES[@]} for serving." >&2
    echo "" >&2
    echo "  Ask for $(( NEED_ROLE_NODES + 1 )). The extra node serves nothing and is not" >&2
    echo "  waste: with the client on a serving node, 2026-09-14 measured 37.7%" >&2
    echo "  less throughput AND 14.4% more engine work per token on an" >&2
    echo "  otherwise identical run. The client node is the cheapest correction" >&2
    echo "  available (TECHNICAL_REPORT_1.tex S7)." >&2
    exit 1
fi
if [ "${#ROLE_NODES[@]}" -gt "${NEED_ROLE_NODES}" ]; then
    echo "NOTE: ${#ROLE_NODES[@]} non-client nodes allocated; this arm serves on ${NEED_ROLE_NODES}."
    echo "      The rest idle. Correct -- per-GPU normalization counts serving GPUs."
fi

NODE_HEAD="${ROLE_NODES[0]}"
NODE_TAIL="${ROLE_NODES[1]}"
NODE_HEAD_SHORT="${NODE_HEAD%%.*}"
NODE_TAIL_SHORT="${NODE_TAIL%%.*}"

# PBS_NODEFILE carries FQDNs but `hostname -s` on a compute node returns only
# the leading label. Keep BOTH forms: ssh wants the FQDN, and launch_role.sh's
# hostname comparison has to be short-vs-short or it never matches. Getting that
# wrong is silent and total -- both ranks fall through to the else branch, the
# first to exit takes the whole PALS application down, and neither log is ever
# written (2026-08-25, the Nemotron pair).
echo "Client node (serves nothing): ${CLIENT_HOST}"
echo "Role head     (API + DP0):    ${NODE_HEAD_SHORT}"
echo "Role headless (DP1):          ${NODE_TAIL_SHORT}"

get_hsn_ip() { ssh -n "$1" "ip -4 -o addr show hsn0" 2>/dev/null | awk '{print $4}' | cut -d/ -f1; }
HEAD_IP=$(get_hsn_ip "${NODE_HEAD}")
TAIL_IP=$(get_hsn_ip "${NODE_TAIL}")
CLIENT_IP=$(ip -4 -o addr show hsn0 2>/dev/null | awk '{print $4}' | cut -d/ -f1)
if [ -z "${HEAD_IP}" ] || [ -z "${TAIL_IP}" ] || [ -z "${CLIENT_IP}" ]; then
    echo "FATAL: could not resolve an hsn0 address on one of the three nodes." >&2
    echo "  head=${HEAD_IP:-<none>} tail=${TAIL_IP:-<none>} client=${CLIENT_IP:-<none>}" >&2
    exit 1
fi
echo "  hsn0: head ${HEAD_IP}   headless ${TAIL_IP}   client ${CLIENT_IP}"
echo "  HTTP endpoint will be http://${HEAD_IP}:${PORT} (head only -- headless serves none)"

mkdir -p "${SHARED}/logs" || { echo "Cannot create ${SHARED}" >&2; exit 1; }
ssh -n "${NODE_TAIL}" "test -d ${SHARED}" || {
    echo "FATAL: ${SHARED} is not visible on ${NODE_TAIL_SHORT} -- is /vast/draco mounted?" >&2
    exit 1
}

# =============================================================================
# Preflight -- fail before a 552 GB weight load, not during it
# =============================================================================
# Promoted from the deferred backlog after it cost an allocation on the Nemotron
# disagg launcher, which had no gate at all and discovered a dirty node ~40
# minutes in at KV-cache init as "No available memory for the cache blocks".
#
# Bracketed patterns ('[v]llm serve') because `ssh host "pkill -f 'vllm serve'"`
# runs through `bash -c` on the far side and THAT shell's command line contains
# the string being searched for, so pgrep matches its own parent.
#
# THE PATTERNS MUST TRACK vLLM'S PROCESS TITLES, AND THEY DRIFTED. 'Worker_TP'
# was written against the Nemotron-era naming. Under DP + EP the workers are
# titled Worker_DP0_TP1_EP1 / Worker_DP1_TP0_EP4, in which the substring
# "Worker_TP" DOES NOT OCCUR -- so the pattern matched nothing, in BOTH places
# it is used. Preflight therefore declared a node clean while a previous run's
# workers were still resident on it, and worse, cleanup()'s
# `pkill -KILL -f 'Worker_TP'` never killed them in the first place: a failed
# launch left eight live workers holding GPU memory and reported a tidy
# teardown. Observed 2026-09-15 as 15.64 GiB occupied on head cuda:1 at
# init_device, which fails the startup snapshot check before any weight loads.
#
# Same drift on the engine title: the logs show EngineCore_DP0, not the older
# VLLM::EngineCore. Both spellings are kept -- an over-broad kill pattern costs
# nothing here because these nodes run nothing else of ours, while a pattern
# that silently matches zero processes is indistinguishable from a clean node.
# WHEN vLLM IS UPGRADED, RE-CHECK THESE AGAINST `pgrep -af` ON A LIVE RUN.
# THE LIST LIVES IN gpu_cleanup.sh AND IS SOURCED, NOT COPIED -- the same rule
# emit_common_env.sh, emit_conn_sampler.sh and workload_profile.sh follow, and
# for the same reason. A second copy is how 'Worker_TP' survived a naming change
# in two places at once. gpu_cleanup.sh is also what teardown executes on each
# role node, so the patterns screened here and the patterns killed there are the
# same array by construction rather than by review.
if [ ! -f "${SCRIPT_DIR}/gpu_cleanup.sh" ]; then
    echo "FATAL: missing ${SCRIPT_DIR}/gpu_cleanup.sh -- preflight and teardown" >&2
    echo "  both read the process-pattern list from it." >&2
    exit 1
fi
# shellcheck source=./gpu_cleanup.sh
source "${SCRIPT_DIR}/gpu_cleanup.sh"
PAT_ALL=("${VLLM_PROC_PATTERNS[@]}")
PAT_VLLM="${VLLM_PROC_PATTERNS[0]}"
ssh -n "${NODE_TAIL}" "test -f ${SCRIPT_DIR}/gpu_cleanup.sh" || {
    echo "FATAL: ${SCRIPT_DIR}/gpu_cleanup.sh is not visible from ${NODE_TAIL_SHORT}." >&2
    exit 1
}

# GPU MEMORY IS ADVISORY, NOT A GATE, and the bar is deliberately loose.
# MEASURED across several fresh PBS allocations: GPU 0 at 1026-1027 MiB, GPU 1
# at 1 MiB, GPU 2 at 2049 MiB, GPU 3 at 2049-2050 MiB. Same pattern, different
# jobs, different nodes -- a system baseline held by root-owned daemons (DCGM's
# nv-hostengine, fabric manager, the IMEX daemon), not our leak. nvidia-smi
# prints "No running processes found" underneath because a non-root user cannot
# see root's pids. The standing rule: 0-2 GiB is baseline, GiB-scale memory
# SURVIVING A REALLOCATION is a leaked context and the answer is another node,
# not a higher bar.
GPU_DIRTY_MIB=${GPU_DIRTY_MIB:-4096}
# The per-node inspection is gpu_cleanup.sh's report mode, run over ssh on each
# role node and gated on its exit status. It is the same file teardown executes
# and the same array sourced above, so a launch cannot screen for one set of
# processes and kill a different one. It also prints the kernel-side HBM view
# and names the owning pid, which is what turns "GPU 1 holds 15.6 GiB" from a
# reason to abandon the node into a kill command.
_preflight_fail=0
for n in "${NODE_HEAD}" "${NODE_TAIL}"; do
    _short="${n%%.*}"
    echo "PREFLIGHT ${_short}:"
    ssh -n "$n" "GPU_DIRTY_MIB=${GPU_DIRTY_MIB} bash ${SCRIPT_DIR}/gpu_cleanup.sh report" \
        2>&1 | sed 's/^/  /'
    # PIPESTATUS, not $?, which would be sed's. Getting this wrong makes the
    # gate pass unconditionally, which is worse than having no gate: it reads
    # as a node that was checked.
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        _preflight_fail=1
    fi
    if [ "$n" = "${NODE_HEAD}" ] && ssh -n "$n" "fuser ${PORT}/tcp" >/dev/null 2>&1; then
        echo "PREFLIGHT ${_short}: port ${PORT} is already bound." >&2
        _preflight_fail=1
    fi
done
if [ "${_preflight_fail}" -ne 0 ]; then
    echo "" >&2
    echo "Refusing to start on a dirty node. Clear it with the same file the" >&2
    echo "report above came from, so the sweep and the screen agree:" >&2
    echo "" >&2
    echo "  for n in ${NODE_HEAD_SHORT} ${NODE_TAIL_SHORT}; do" >&2
    echo "    ssh \$n \"bash ${SCRIPT_DIR}/gpu_cleanup.sh kill ${PORT} ${DP_RPC_PORT}\"" >&2
    echo "  done" >&2
    echo "" >&2
    echo "Then re-run this launcher. If the report still shows GPU memory AND" >&2
    echo "'no process you own holds HBM pages', it belongs to a pid you cannot" >&2
    echo "see or kill. That is a leaked context: TAKE ANOTHER NODE." >&2
    exit 1
fi

# =============================================================================
# Locate the shim and the shared helpers, IN PLACE in the repo
# =============================================================================
for f in fi_getinfo_shim.so env_for_libfabric_topology_error.sh gpu_cleanup.sh \
         emit_common_env.sh emit_conn_sampler.sh workload_profile.sh; do
    if [ ! -f "${SCRIPT_DIR}/${f}" ]; then
        echo "FATAL: missing ${SCRIPT_DIR}/${f}." >&2
        exit 1
    fi
done
SHIM="${SCRIPT_DIR}/fi_getinfo_shim.so"
ENV_SCRIPT="${SCRIPT_DIR}/env_for_libfabric_topology_error.sh"

# fi_getinfo_shim.so is a TRACKED BINARY that goes stale silently against its
# own .c. Nothing in THIS arm should reach libfabric through it, but a stale
# shim in one arm's LD_PRELOAD and a fresh one in the other is exactly the
# asymmetry the ONE RULE exists to prevent.
_shim_patches=$(strings "${SHIM}" 2>/dev/null | grep -c 'PATCH [234]' || true)
if [ "${_shim_patches}" -lt 1 ]; then
    echo "WARNING: ${SHIM} has no PATCH [234] strings -- it looks stale." >&2
    echo "         Rebuild it before trusting a cross-arm comparison." >&2
fi
ssh -n "${NODE_TAIL}" "test -f ${SHIM}" || {
    echo "FATAL: ${SHIM} is not visible from ${NODE_TAIL_SHORT}." >&2
    exit 1
}

# =============================================================================
# Generate common_env.sh and the sampler via the SHARED emitters
# =============================================================================
# NO_PROXY_LIST covers all three nodes, and the CLIENT node is in it because the
# bench runs there: without it, curl to an hsn0 address goes to
# proxy.alcf.anl.gov, comes back a Squid error page, and reads as a dead server
# rather than a misrouted client (2026-09-11, twice).
#
#   UCX_LINES    empty. The disagg arm only sets UCX_* when NIXL_BACKEND=UCX and
#                the default there is LIBFABRIC, so empty is what the default
#                disagg run also emits. Matching the disagg default is the
#                reason, not "colocated needs no UCX".
#   GPU_PIN_LINE empty. TP=4 wants all four GPUs visible on every role node.
NO_PROXY_LIST="localhost,127.0.0.1,${HEAD_IP},${TAIL_IP},${CLIENT_IP},${NODE_HEAD},${NODE_TAIL},${NODE_HEAD_SHORT},${NODE_TAIL_SHORT},${CLIENT_HOST}"
UCX_LINES=""
GPU_PIN_LINE=""
# shellcheck source=./emit_common_env.sh
source "${SCRIPT_DIR}/emit_common_env.sh"
emit_common_env "${SHARED}/common_env.sh"
# shellcheck source=./emit_conn_sampler.sh
source "${SCRIPT_DIR}/emit_conn_sampler.sh"
emit_conn_sampler "${SHARED}/sample_conns.sh"
echo "Wrote ${SHARED}/common_env.sh and sample_conns.sh (shared emitters, not copies)."

# =============================================================================
# Generate the role script -- ONE file, both nodes, selecting by hostname
# =============================================================================
# Same reason the Nemotron pair does it: CXI needs a VNI, the VNI arrives only
# in SLINGSHOT_VNIS provisioned by PALS, and PALS allocates a VNI per
# APPLICATION. Two independent `mpiexec -n 1` launches get two DIFFERENT VNIs --
# three separate launches were observed getting 1726, 1825 and 1873 -- and
# endpoints on different VNIs cannot reach each other at all. Plain `ssh`
# carries no PALS environment whatsoever.
#
# THAT ARGUMENT NOW APPLIES TO THE COLOCATED ARM TOO, which is new. On Nemotron
# this arm was one node and plain `vllm serve` with no mpiexec: nothing it did
# crossed the fabric. Inkling-Small's baseline spans two nodes and its expert
# all-to-all crosses Slingshot, so it gets the same one-application launch. It
# is also the shape the ONE RULE wants: two arms launched through different
# mechanisms are two arms that can differ in ways nobody enumerated.
#
# Unquoted heredoc, so values resolve now; \${...} escapes are the handful that
# must survive to runtime. An unescaped $ here bakes a launcher-time value into
# a runtime file.
cat > "${SHARED}/launch_role.sh" <<EOF
#!/bin/bash
source ${SHARED}/common_env.sh

# THE COLOCATED ARM SETS NO NIXL ENVIRONMENT. The disagg arm sets exactly three
# variables here -- VLLM_NIXL_SIDE_CHANNEL_HOST, _PORT and NIXL_LOG_LEVEL -- and
# adds --kv-transfer-config below. Together with the batching split, that is the
# COMPLETE list of differences between the two launch_role.sh files. Anything
# else that differs is a bug in one of them.

MY_HOST="\$(hostname -s)"
MY_HOST="\${MY_HOST%%.*}"
if [ "\${MY_HOST}" = "${NODE_HEAD_SHORT}" ]; then
    ROLE=head
    LOG=${SHARED}/logs/head.log
    # Head hosts data-parallel rank 0 AND all the API servers.
    DP_ARGS=(--data-parallel-size ${DP}
             --data-parallel-size-local ${DP_LOCAL}
             --data-parallel-address ${HEAD_IP}
             --data-parallel-rpc-port ${DP_RPC_PORT})
    # --api-server-count is legal ONLY here. serve.py:66-71 raises with
    # --headless. Unset it and serve.py:121 silently gives you ${DP}.
    DP_ARGS+=(--api-server-count ${API_SERVER_COUNT})
    SERVE_ARGS=(--host 0.0.0.0 --port ${PORT})
elif [ "\${MY_HOST}" = "${NODE_TAIL_SHORT}" ]; then
    ROLE=headless
    LOG=${SHARED}/logs/headless.log
    # Headless hosts data-parallel rank 1 and serves no HTTP. Same address and
    # rpc port as the head -- that pair is how the two find each other.
    DP_ARGS=(--headless
             --data-parallel-size ${DP}
             --data-parallel-size-local ${DP_LOCAL}
             --data-parallel-start-rank 1
             --data-parallel-address ${HEAD_IP}
             --data-parallel-rpc-port ${DP_RPC_PORT})
    SERVE_ARGS=()
else
    echo "Rank landed on unexpected host '\${MY_HOST}'." >&2
    echo "  expected '${NODE_HEAD_SHORT}' (head) or '${NODE_TAIL_SHORT}' (headless)" >&2
    exit 1
fi

exec > "\${LOG}" 2>&1
echo "=== role=\${ROLE} host=\$(hostname -s) SLINGSHOT_VNIS=\${SLINGSHOT_VNIS:-<UNSET>} ==="
if [ -z "\${SLINGSHOT_VNIS:-}" ]; then
    echo "SLINGSHOT_VNIS is EMPTY -- this process was not launched under PALS."
    echo "Anything that reaches the cxi provider will fail fi_domain() with"
    echo "-FI_ENOSYS. Check the mpiexec invocation."
fi

# Held constant across arms for the same reason the Nemotron pair holds it: the
# shim is a no-op unless something calls fi_getinfo, but an LD_PRELOAD present
# in one arm and absent in the other is a difference in the thing being
# compared. Cheap to hold constant, so hold it.
export LD_PRELOAD=${SHIM}

EXTRA=()
# BLOCK_SIZE and ENFORCE_EAGER default to EMPTY and are tested with -n: unset
# means "do not pass the flag". EXPERT_PARALLEL and NUMA_BIND default to a
# VALUE and are tested with = "1": for those, -n would make EXPERT_PARALLEL=0
# still pass --enable-expert-parallel, i.e. the one spelling a reader would
# reach for to turn it off would be the one spelling that cannot.
[ -n "${BLOCK_SIZE}" ]           && EXTRA+=(--block-size ${BLOCK_SIZE})
[ "${EXPERT_PARALLEL}" = "1" ]   && EXTRA+=(--enable-expert-parallel)
[ -n "${ENFORCE_EAGER}" ]        && EXTRA+=(--enforce-eager)
[ "${NUMA_BIND}" = "1" ]         && EXTRA+=(--numa-bind --numa-bind-nodes ${NUMA_BIND_NODES})

# --distributed-executor-backend mp is EXPLICIT, on both roles, and the header
# argues why at length: unset yields mp only because Ray is not initialized,
# ray>=2.55.0 is installed in this env, and the guard is one
# ray_is_initialized() call away from flipping the backend mid-study
# (parallel.py:941-951). It never crosses a node here -- each role is its own
# engine at world_size = TP = 4.
exec vllm serve ${MODEL} \\
    \${SERVE_ARGS[@]+"\${SERVE_ARGS[@]}"} \\
    \${DP_ARGS[@]+"\${DP_ARGS[@]}"} \\
    --distributed-executor-backend mp \\
    --tensor-parallel-size ${TP} \\
    --max-model-len ${MAX_MODEL_LEN} \\
    --gpu-memory-utilization ${GPU_MEM_UTIL} \\
    --dtype bfloat16 \\
    --kv-cache-dtype ${KV_CACHE_DTYPE} \\
    ${HMA_FLAG} \\
    ${PREFIX_CACHING_FLAG} \\
    --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} \\
    --max-num-seqs ${MAX_NUM_SEQS} \\
    \${EXTRA[@]+"\${EXTRA[@]}"}
EOF
chmod +x "${SHARED}/launch_role.sh"

# --- Provenance, recorded rather than hoarded --------------------------------
# Report what was actually PASSED, not what was configured: EXPERT_PARALLEL is
# always set, so the old `${EXPERT_PARALLEL:-off}` form could never print "off"
# and would have reported EP as on for a run that had it off.
if [ "${EXPERT_PARALLEL}" = "1" ]; then _ep_state="on"; else _ep_state="OFF"; fi
{
    echo "stamp             ${STAMP}"
    echo "arm               inkling-colocated  (one role, ${DP} nodes, $(( DP * TP )) GPUs)"
    echo "model             ${MODEL}"
    echo "CLIENT NODE       ${CLIENT_HOST}  hsn0 ${CLIENT_IP}   <- serves nothing; bench runs here"
    echo "role head         ${NODE_HEAD_SHORT}  hsn0 ${HEAD_IP}  http ${PORT}  dp-rank 0  api servers ${API_SERVER_COUNT}"
    echo "role headless     ${NODE_TAIL_SHORT}  hsn0 ${TAIL_IP}  no http       dp-rank 1  api servers 0"
    echo "parallelism       DP ${DP} x TP ${TP}, expert-parallel ${_ep_state}  (EP group ${DP} x ${TP} = $(( DP * TP )))"
    echo "executor          mp, passed explicitly on both roles  (each role is one engine at world_size ${TP}; mp never crosses a node)"
    echo "max-model-len     ${MAX_MODEL_LEN}"
    echo "provisioned for   ${WORKLOAD}  (ISL ${WL_ISL} / OSL ${WL_OSL}, needs ${WL_MAX_MODEL_LEN})"
    echo "batching          ${MAX_NUM_BATCHED_TOKENS} tok / ${MAX_NUM_SEQS} seq  (per-knob max of P 16384/32 and D 2048/256)"
    echo "numa-bind         ${NUMA_BIND} nodes ${NUMA_BIND_NODES}  (workers + EngineCore only; NOT the API servers)"
    echo "kv dtype          ${KV_CACHE_DTYPE}   model dtype bfloat16 (hard-asserted by InklingConvState)"
    echo "shim              ${SHIM}  sha256 $(sha256sum "${SHIM}" 2>/dev/null | awk '{print $1}')"
    echo "git               $(git -C "${SCRIPT_DIR}" rev-parse --short HEAD 2>/dev/null || echo '(not a work tree)')"
    echo ""
    echo "=== nvidia-smi topo -m on ${NODE_HEAD_SHORT} ==="
    ssh -n "${NODE_HEAD}" "nvidia-smi topo -m" 2>&1
} > "${SHARED}/run_config.txt"
sed -n '1,14p' "${SHARED}/run_config.txt"

# =============================================================================
# Teardown
# =============================================================================
cleanup() {
    echo "=== Cleaning up ==="
    touch "${SAMPLE_STOP_MARKER:-}" 2>/dev/null
    for n in "${NODE_HEAD}" "${NODE_TAIL}"; do
        ssh -n "$n" "pkill -TERM -f \"${PAT_VLLM}\"" 2>/dev/null
    done
    kill -TERM ${MPIEXEC_PID:-} ${TAIL_H_PID:-} ${TAIL_T_PID:-} ${TAIL_M_PID:-} 2>/dev/null
    sleep 5
    # Hand the sweep to gpu_cleanup.sh, which is the SAME file preflight ran in
    # report mode and the same array it screened on. An orphaned EngineCore is
    # 'EngineCore_DP<n>' (older builds: 'VLLM::EngineCor') and matches no
    # 'vllm serve' pattern; the workers are 'Worker_DP<n>_TP<n>_EP<n>' and match
    # neither. Each survivor holds a CUDA context and its share of the KV pool,
    # and the symptom lands on the NEXT run, a stage removed from its cause.
    # Routing both through one file is what stops "cleanup reported success" and
    # "preflight sees nothing" from meaning different things.
    #
    # kill mode ends in a report, so teardown states what it actually left
    # behind rather than asserting that it left nothing.
    for n in "${NODE_HEAD}" "${NODE_TAIL}"; do
        ssh -n "$n" "bash ${SCRIPT_DIR}/gpu_cleanup.sh kill ${PORT} ${DP_RPC_PORT}" \
            2>&1 | sed 's/^/  /'
    done
    # The per-node nvidia-smi dump that used to live here is gone: the report at
    # the end of `gpu_cleanup.sh kill` above prints it, plus the kernel HBM view
    # and the owning pid, for both nodes. Read the VERDICT lines. A node that
    # ends DIRTY is the next run's failure, and this is the last moment it can
    # still be handed back.
    echo "  Run dir: ${SHARED}"
}
trap cleanup EXIT INT TERM

# =============================================================================
# Launch -- ONE mpiexec, restricted to the SERVING nodes
# =============================================================================
# --hosts, not a bare -n 2: the allocation contains the client node and `-ppn 1`
# over the whole allocation would start a rank on it. The client node must run
# no engine, which is the entire point of having it.
#
# --cpu-bind none IS NOT COSMETIC. PALS binds each rank to a subset of the
# node's cores by default and children inherit the mask, so all four TP workers
# plus their NCCL and NIXL progress threads would be confined to whatever slice
# rank 0 was given -- on a Grace node, possibly a single core. The symptom is
# not an error: it is an engine that takes many minutes to initialise and a
# transfer rate that looks like a fabric problem. MPI_CPU_BIND= (empty) drops
# the flag if a launcher ever refuses it.
MPI_CPU_BIND=${MPI_CPU_BIND-none}
CPU_BIND_ARGS=()
[ -n "${MPI_CPU_BIND}" ] && CPU_BIND_ARGS=(--cpu-bind "${MPI_CPU_BIND}")
MPI_HOSTS="${NODE_HEAD},${NODE_TAIL}"

# mpiexec.log IS IN THIS touch DELIBERATELY. `cmd > file &` creates the file in
# the FORKED CHILD, so the parent can reach `tail -f` first and lose the race --
# tail prints "cannot open ... No such file or directory / no files remaining"
# and exits, and the ONE stream that carries PALS-level failures is silently not
# being followed for the rest of the run. Observed on the first Inkling launch,
# 2026-09-15: head.log and headless.log tailed fine purely because they were in
# this touch and mpiexec.log was not.
touch "${SHARED}/logs/head.log" "${SHARED}/logs/headless.log" \
      "${SHARED}/logs/mpiexec.log"
echo ""
echo "Launching: mpiexec -n ${DP} -ppn 1 --hosts ${MPI_HOSTS}"
mpiexec -n "${DP}" -ppn 1 --hosts "${MPI_HOSTS}" \
    ${CPU_BIND_ARGS[@]+"${CPU_BIND_ARGS[@]}"} \
    bash "${SHARED}/launch_role.sh" > "${SHARED}/logs/mpiexec.log" 2>&1 &
MPIEXEC_PID=$!

# `sed -u` (unbuffered): without it, output piped through sed is block-buffered
# and arrives in silent bursts, which looks exactly like "nothing is happening".
# disown so cleanup's kills do not print "PID Killed exit 1" job-status lines
# into the middle of the teardown output.
tail -n +1 -f "${SHARED}/logs/head.log"     | sed -u 's/^/[HEAD] /' &
TAIL_H_PID=$!; disown ${TAIL_H_PID}
tail -n +1 -f "${SHARED}/logs/headless.log" | sed -u 's/^/[TAIL] /' &
TAIL_T_PID=$!; disown ${TAIL_T_PID}
# mpiexec's own stream carries PALS errors that never reach either role log -- a
# launch that dies before the role script runs is otherwise completely silent.
tail -n +1 -f "${SHARED}/logs/mpiexec.log"  | sed -u 's/^/[MPI]  /' &
TAIL_M_PID=$!; disown ${TAIL_M_PID}

wait_healthy() {
    local ip=$1 port=$2 name=$3
    local t0; t0=$(date +%s)
    for i in $(seq 1 "${HEALTH_TRIES}"); do
        if curl -s -o /dev/null -w "%{http_code}" "http://${ip}:${port}/health" 2>/dev/null | grep -q 200; then
            echo "${name} healthy after $(( $(date +%s) - t0 ))s"; return 0
        fi
        # mpiexec is the liveness source of truth. Without this the script burns
        # the full 40-minute timeout on an application PALS already tore down --
        # a dead startup and a slow one look identical from the outside.
        if ! kill -0 "${MPIEXEC_PID}" 2>/dev/null; then
            echo "mpiexec exited while waiting for ${name}." >&2
            echo "  Read ${SHARED}/logs/mpiexec.log FIRST: a PALS-level failure" >&2
            echo "  (bad VNI, node not in the allocation, launcher error) never" >&2
            echo "  reaches head.log or headless.log at all." >&2
            # DUMP THE LOGS HERE RATHER THAN NAMING THEM. cleanup() kills the
            # `tail -f | sed` pipelines within a second of this return, and a
            # `vllm serve` that dies on an argparse error writes its reason and
            # exits faster than that pipeline flushes -- so the operator sees
            # the role BANNER, then nothing, then teardown, and has to go read
            # files by hand to learn anything at all (2026-09-15, first Inkling
            # launch). The reason is on disk; print it while we still can.
            for _l in mpiexec head headless; do
                echo "" >&2
                echo "  --- last 40 lines of logs/${_l}.log ---" >&2
                tail -n 40 "${SHARED}/logs/${_l}.log" 2>/dev/null \
                    | sed 's/^/    /' >&2 \
                    || echo "    (no ${_l}.log)" >&2
            done
            return 1
        fi
        if [ $(( i % 12 )) -eq 0 ]; then
            printf '  [startup %4ds] %s\n' "$(( $(date +%s) - t0 ))" \
                "$(tail -n 1 "${SHARED}/logs/head.log" 2>/dev/null | cut -c1-100)"
        fi
        sleep 5
    done
    echo "${name} never became healthy -- check ${SHARED}/logs/" >&2
    return 1
}
# Only the head answers HTTP. The headless node's liveness is implied: the head
# cannot become healthy until every data-parallel rank has registered, so a
# healthy head is proof both engines came up.
wait_healthy "${HEAD_IP}" "${PORT}" "Inkling colocated head (${NODE_HEAD_SHORT})" || exit 1
kill -TERM "${TAIL_H_PID}" "${TAIL_T_PID}" 2>/dev/null   # stop mirroring

# =============================================================================
# Gate on the KV pool -- before anything is benched against it
# =============================================================================
# "One rank can come up GiB-heavier than its peers on a clean node, dying at KV
# sizing or silently shrinking the pool. Relaunch; do not tune." The pool is
# sized by the WORST worker, and it has come up short twice on Nemotron -- once
# from rank 3's CUDA-graph estimate, once from TP1 consuming 8 GiB more of
# weights+non-torch than its peers, giving 4,023,996 tokens against the healthy
# 5,975,848 (MEMORY_2026-09-12h.md). Both times nothing about the model or the
# workload had changed, and both times the remedy was a relaunch.
#
# On the FIRST Inkling run there is no expected value yet, so this records the
# number and says so. On every run after that, pass EXPECT_POOL=<tokens> from a
# healthy launch and the gate binds. Do not skip it: a run benched against a
# short pool is comparable to nothing.
_pool=$(grep -h -o 'GPU KV cache size: [0-9,]* tokens' "${SHARED}/logs/head.log" \
        "${SHARED}/logs/headless.log" 2>/dev/null | grep -o '[0-9,]*' | tr -d ',' \
        | grep -v '^$' | sort -n | head -1)
echo ""
echo "=== KV pool ==="
if [ -z "${_pool}" ]; then
    echo "  Could not read 'GPU KV cache size' from either log. Find it before"
    echo "  benching -- every comparison is gated on it:"
    echo "      grep -h 'GPU KV cache size' ${SHARED}/logs/*.log"
else
    echo "  ${_pool} tokens (minimum across data-parallel ranks -- the pool is"
    echo "  sized by the worst worker, so the minimum is the pool)."
    if [ -n "${EXPECT_POOL:-}" ]; then
        _delta=$(( _pool - EXPECT_POOL ))
        _pct=$(( _delta * 100 / EXPECT_POOL ))
        if [ "${_pct}" -lt -2 ] || [ "${_pct}" -gt 2 ]; then
            echo "  *** POOL GATE FAILED: expected ${EXPECT_POOL}, got ${_pool} (${_pct}%). ***" >&2
            echo "  This is the pool lottery. RELAUNCH, do not tune -- the cause is" >&2
            echo "  one rank allocating more during init than its peers, and the" >&2
            echo "  pool is sized by the worst one. Nothing about the model or the" >&2
            echo "  workload changed. Ctrl-C and try again." >&2
        else
            echo "  Pool gate PASSED against EXPECT_POOL=${EXPECT_POOL} (${_pct}%)."
        fi
    else
        echo "  No EXPECT_POOL set -- this run establishes it. Record this number"
        echo "  and pass EXPECT_POOL=${_pool} on every subsequent COLOCATED launch."
        echo "  It does NOT transfer to the disagg arm: P and D have different"
        echo "  --max-num-batched-tokens (16384 / 2048) and therefore different"
        echo "  activation profiles and different pools. Each disagg role"
        echo "  establishes its own number -- see THE ONE RULE in the header."
    fi
fi
echo "${_pool:-unknown}" > "${SHARED}/kv_pool_tokens.txt"

# --- Confirm the arm is actually free of the connector -----------------------
# Cheap, and it catches the one mistake this whole file exists to prevent: a
# colocated arm that still has a KV connector attached is not a baseline, it is
# half a disagg pair with a confusing name.
if grep -qiE 'NixlConnector|kv_transfer_config|KVConnector' "${SHARED}/logs/head.log"; then
    echo "" >&2
    echo "WARNING: the head log mentions a KV connector. This arm must have NONE." >&2
    grep -inE 'NixlConnector|kv_transfer_config|KVConnector' "${SHARED}/logs/head.log" \
        | head -5 | sed 's/^/    /' >&2
fi

# =============================================================================
# Start the frontend samplers -- BEFORE the bench, or the distribution is lost
# =============================================================================
# The connection distribution is decided in the first seconds of a bench and
# cannot be recovered afterwards; `ss` reports live sockets and nothing records
# what the split was. Started here, held for the life of the run, stopped by
# cleanup's marker. setsid+nohup so the loops survive the ssh session closing --
# the same orphan problem the CXI pollers had.
SAMPLE_STOP_MARKER="${SHARED}/sample_stop_${STAMP}"
start_sampler() {
    local node=$1 port=$2 out=$3 label=$4 kind=${5:-serving}
    : > "${out}"
    ssh -n "${node}" "setsid nohup bash ${SHARED}/sample_conns.sh \
        ${SAMPLE_STOP_MARKER} ${out} ${port} ${SAMPLE_INTERVAL_S} ${label} ${kind} \
        > /dev/null 2>&1 < /dev/null &"
}
start_sampler "${NODE_HEAD}" "${PORT}" "${SHARED}/logs/conns_head.log"     "head"     serving
start_sampler "${NODE_TAIL}" "-"       "${SHARED}/logs/conns_headless.log" "headless" serving
# The client node gets one too, and it is not padding: it is the CONTROL for
# the disagg arm's client node, which also hosts the proxy. Without a colocated
# measurement of the client alone there is nothing to subtract the proxy from.
: > "${SHARED}/logs/conns_client.log"
setsid nohup bash "${SHARED}/sample_conns.sh" \
    "${SAMPLE_STOP_MARKER}" "${SHARED}/logs/conns_client.log" "-" \
    "${SAMPLE_INTERVAL_S}" "client" client > /dev/null 2>&1 < /dev/null &
sleep 2
echo ""
echo "=== frontend samplers started (every ${SAMPLE_INTERVAL_S}s) ==="
echo "  head:     ${SHARED}/logs/conns_head.log"
echo "  headless: ${SHARED}/logs/conns_headless.log   (no socket -- by design)"
echo "  client:   ${SHARED}/logs/conns_client.log     (bench CPU -- the control)"
echo "  The LISTEN header is the whole Appendix A proof: ONE row means all"
echo "  ${API_SERVER_COUNT} API servers share one accept queue and --api-server-count"
echo "  balances nothing."
grep -m4 '^# ' "${SHARED}/logs/conns_head.log" 2>/dev/null | sed 's/^/    /'

# =============================================================================
# Hold the pair up for bench_arm.sh -- WHICH RUNS ON THIS NODE
# =============================================================================
if [ "${KEEP_ALIVE}" = "1" ]; then
    echo ""
    echo "============================================================"
    echo "KEEP_ALIVE=1 -- pair held up. Ctrl-C here is the teardown."
    echo "============================================================"
    echo "  Colocated arm endpoint (head node only):"
    echo "      http://${HEAD_IP}:${PORT}"
    echo ""
    echo "  Model string for --model:  ${MODEL}"
    echo "  Run dir:                   ${SHARED}"
    echo "  max-model-len:             ${MAX_MODEL_LEN}"
    echo "  Serving GPUs:              $(( DP * TP ))  (passed as GPUS below; bench_arm.sh defaults to 4)"
    echo ""
    echo "  From another shell ON THIS NODE (${CLIENT_HOST}) -- the client node,"
    echo "  which serves nothing. Do NOT bench from ${NODE_HEAD_SHORT} or"
    echo "  ${NODE_TAIL_SHORT}: 2026-09-14 measured -37.7% throughput and +14.4%"
    echo "  engine work per token with the client on a serving node, and the"
    echo "  sampler will flag the run as contaminated if you do."
    echo ""
    echo "      source ${SHARED}/common_env.sh"
    echo ""
    echo "      ARM=inkling-colocated \\"
    echo "      MODEL=${MODEL} \\"
    echo "      GPUS=$(( DP * TP )) \\"
    echo "      BASE_URL=http://${HEAD_IP}:${PORT} \\"
    echo "      METRICS_URLS=http://${HEAD_IP}:${PORT} \\"
    echo "      WORKLOAD=${WORKLOAD} \\"
    echo "      RUN_DIR=${SHARED} \\"
    echo "          bash ${SCRIPT_DIR}/bench_arm.sh"
    echo ""
    echo "  MODEL AND GPUS ARE BOTH REQUIRED HERE, and both fail quietly:"
    echo "    * bench_arm.sh defaults MODEL to the Nemotron repo id (:32). A"
    echo "      wrong --model is an HTTP 400 PER REQUEST, which returns instantly"
    echo "      and therefore reads on the progress bar as a fast run rather"
    echo "      than as a fault -- the same trap as an undersized max-model-len."
    echo "    * GPUS defaults to 4 for every ARM that is not literally 'disagg'"
    echo "      (:77-80), sized for the Nemotron colocated arm. This arm serves"
    echo "      $(( DP * TP )) (DP ${DP} x TP ${TP}), so the default would report every"
    echo "      per-GPU number at $(( DP * TP / 4 ))x -- in the favourable direction, in the"
    echo "      figure the whole study turns on."
    echo ""
    echo "  The source line is NOT optional, and skipping it fails in three ways"
    echo "  that all look like a broken server rather than a bare shell:"
    echo "    * no conda env      -> 'vllm: command not found'"
    echo "    * no HF_HOME/TOKEN  -> --dataset-name random cannot load the"
    echo "                           tokenizer and reaches for the network"
    echo "    * no no_proxy       -> curl to ${HEAD_IP} goes to"
    echo "                           proxy.alcf.anl.gov and the preflight"
    echo "                           one-token completion fails as if nothing"
    echo "                           were listening"
    echo "  Take these addresses from THIS banner: no_proxy is generated per"
    echo "  run, so an IP from an earlier allocation is never in it."
    echo ""
    echo "  RECORD, for every bench against this launch:"
    echo "    * client node        ${CLIENT_HOST}   (already in run_config.txt)"
    echo "    * connection split   tail -5 ${SHARED}/logs/conns_head.log"
    echo "    * KV pool            $(cat "${SHARED}/kv_pool_tokens.txt" 2>/dev/null)"
    echo "  Do NOT sample kv_cache_usage_perc -- stale at ASC>1, CLOSED.md."
    echo ""
    echo "  Nothing is proven until the identical config has run TWICE."
    echo ""

    _ka_stop=0
    trap '_ka_stop=1; echo ""; echo "Interrupt received -- releasing servers."' INT TERM

    # The heartbeat reports the sampler, not the log's last line. Blind-tailing
    # was actively misleading on the Nemotron rig: the last line is almost
    # always the GET /health entry this very loop just produced, so the
    # heartbeat reported itself, at 200, once a minute, while the engine behind
    # it had not stepped in twelve minutes (2026-09-11).
    #
    # And at --api-server-count ${API_SERVER_COUNT} there is no engine line to read at
    # all: vLLM disables the periodic stats logger outright above client_count 1
    # (loggers.py:1341). So the watchpoint is the sampler's running/waiting pair
    # scraped from /metrics, which survives multiprocess mode as a sum.
    _ka_t0=$(date +%s)
    while [ "${_ka_stop}" -eq 0 ]; do
        if ! kill -0 "${MPIEXEC_PID}" 2>/dev/null; then
            echo ""
            echo "mpiexec (${MPIEXEC_PID}) exited -- both engines are gone."
            echo "Check ${SHARED}/logs/mpiexec.log; a PALS-level failure never"
            echo "reaches head.log or headless.log."
            break
        fi
        sleep "${KEEP_ALIVE_POLL_S}"
        _ka_el=$(( $(date +%s) - _ka_t0 ))
        _ka_h=$(curl -s -o /dev/null -w '%{http_code}' "http://${HEAD_IP}:${PORT}/health" 2>/dev/null || echo "---")
        _ka_s=$(grep 'n_conn=' "${SHARED}/logs/conns_head.log" 2>/dev/null | tail -n 1 \
                  | cut -d' ' -f2-6)
        printf '  [keep-alive %02d:%02d:%02d] health=%s | %s\n' \
            $(( _ka_el / 3600 )) $(( (_ka_el % 3600) / 60 )) $(( _ka_el % 60 )) \
            "${_ka_h}" "${_ka_s:-<sampler has produced no line yet>}"
    done

    echo ""
    echo "=== frontend summary for this launch ==="
    summarize_conn_sampler "${SHARED}/logs/conns_head.log"     "head     (${NODE_HEAD_SHORT})"
    summarize_conn_sampler "${SHARED}/logs/conns_headless.log" "headless (${NODE_TAIL_SHORT})"
    summarize_conn_sampler "${SHARED}/logs/conns_client.log"   "client   (${CLIENT_HOST})"

    # Restore the standing disposition before falling through, so the EXIT trap
    # is the single teardown path. The standing handler RETURNS rather than
    # exits, so leaving it installed during the loop would both spin the loop
    # after the user asked to stop and run cleanup() twice. Do not "simplify"
    # this back to one trap.
    trap cleanup EXIT INT TERM
    echo "Releasing servers -- cleanup() follows."
fi
