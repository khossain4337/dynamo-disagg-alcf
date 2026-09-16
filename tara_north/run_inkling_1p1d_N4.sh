#!/bin/bash
set -uo pipefail

# =============================================================================
# run_inkling_1p1d_N4.sh -- the DISAGGREGATED arm for INKLING-SMALL.
#
# FOUR serving nodes in two roles, plus a proxy node, plus a client node. It
# brings both roles up, waits for /health on each, starts the proxy, starts the
# frontend samplers, and parks so that bench_arm.sh can drive it from the
# client node:
#
#     ARM=disagg MODEL=<repo id> GPUS=16 \
#         BASE_URL=http://<PROXY_IP>:8000 \
#         METRICS_URLS=http://<P_HEAD_IP>:8100,http://<D_HEAD_IP>:8200 \
#         RUN_DIR=<run dir> bash bench_arm.sh
#
# MODEL and GPUS are not optional, and GPUS is 16 here, not bench_arm.sh's
# disagg default of 8 (:77-80, written for Nemotron's 1P+1D at TP=4 on two
# nodes). The keep-alive banner prints the whole command with both filled in --
# copy it from there rather than from here.
#
# =============================================================================
# WHY TWO NODES PER ROLE. IT IS THE WEIGHTS, NOT COMPARABILITY.
# =============================================================================
# Inkling-Small is 266B at BF16 = 532 GB of weights against ~345 GB usable per
# node (run_inkling_colocated_N2.sh:19-23; 265,956,439,090 parameters read off
# the HF model index, 2026-09-14). ONE NODE CANNOT LOAD THE MODEL. So every
# role is at least two nodes, and the minimum disaggregated configuration is
# 2P + 2D.
#
# DECISIONS_2026-09-15c.md framed the one-node-per-role shape as "NOT
# COMPARABLE" because DP1 x TP4 gives EP width 4 and E=64 against colocated's
# EP 8 and E=32. That understates it: at one node per role the engine does not
# come up at all. The EP-width match is a CONSEQUENCE of the weight constraint,
# not a second constraint that had to be traded against it.
#
# THE CONSEQUENCE FOR THE FIGURE, STATED PLAINLY. This arm serves on 16 GPUs
# against the colocated arm's 8. There is no configuration in which the two
# match, so per-GPU throughput is structurally unkind to this arm: it must
# double absolute throughput merely to tie. That is physics rather than a
# riggable choice, and the study already has the right primary metric for it --
# ENGINE WORK PER TOKEN AT MATCHED LOAD (TECHNICAL_REPORT_1.tex S6, where the
# Nemotron pair showed disagg doing 49.0% less), which is GPU-count neutral.
# The Nemotron disagg arm was already 8 GPUs against colocated's 4, so this is
# the same structure and the same argument, not a new framing invented here.
# Lead with engine work per token and the ITL tail share; report per-GPU
# throughput as the cost.
#
# ONE CONFOUND DISAPPEARS FOR FREE. Nemotron's 7.1% per-GPU deficit was partly
# fabric: there disagg crossed Slingshot and colocated did not. Here the
# colocated baseline is itself multi-node, so BOTH arms cross it and the
# difference is disaggregation alone (run_inkling_colocated_N2.sh:24-28).
#
# =============================================================================
# THE SHAPE: each role is head + headless, DP=2 x TP=4, expert-parallel.
# =============================================================================
#
#     node 0   P head       API servers + data-parallel rank 0   http 8100
#     node 1   P headless   data-parallel rank 1, no http
#     node 2   D head       API servers + data-parallel rank 0   http 8200
#     node 3   D headless   data-parallel rank 1, no http
#     node 4   PROXY        toy_proxy_server.py, nothing else    http 8000
#     node 5   CLIENT       this script, bench_arm.sh, nothing else
#
# MULTI-NODE DP IS HEAD + HEADLESS, NOT TWO SYMMETRIC RANKS. The Nemotron
# pattern of `mpiexec -n 2 -ppn 1` launching one IDENTICAL role per node does
# not carry over, and this file therefore has FOUR hostname branches rather
# than two. The full argument -- why no Ray, why `mp` is genuinely multi-node
# in 0.27.1, why --distributed-executor-backend is passed explicitly -- is in
# run_inkling_colocated_N2.sh:41-102 and is not repeated here. Read it before
# changing anything in the launch block.
#
# NEVER --tensor-parallel-size 8. TP must not leave the node: TP=8 puts the
# attention all-reduce on Slingshot. THE vLLM INKLING BLOG'S EXAMPLE USES TP=8
# AND THAT DOES NOT TRANSFER -- on GB200 NVL, eight GPUs are one NVLink domain;
# here they are two nodes. Do not "fix" this launcher to match the blog.
#
# =============================================================================
# THE ONE RULE (the colocated launcher's header states the other half)
# =============================================================================
# This must issue the SAME `vllm serve` command as run_inkling_colocated_N2.sh,
# differing by exactly three things:
#
#     1. --kv-transfer-config, producer on P and consumer on D
#     2. VLLM_NIXL_SIDE_CHANNEL_HOST / _PORT, and NIXL_LOG_LEVEL
#     3. batching -- P and D are tuned in opposite directions; the colocated
#        arm takes the per-knob MAXIMUM of the two, which is what makes it a
#        baseline nobody can call starved
#
# Every other flag, env var and default is identical. If you change a serve
# flag here, change it there in the same commit. Both launchers write their
# generated launcher into the run dir precisely so the two can be diffed:
#
#     diff <(sed -n '/^exec vllm serve/,$p' <colo-run>/launch_role.sh) \
#          <(sed -n '/^exec vllm serve/,$p' <disagg-run>/launch_role.sh)
#
# THE FOUR BATCHING NUMBERS BELOW ARE FROZEN BY WORK ALREADY PUBLISHED. The
# colocated arm's 16384 / 256 is the per-knob maximum of exactly P 16384/32 and
# D 2048/256. Changing any of the four voids the c=16 / c=32 / c=64 colocated
# sweep, which has already been run and recorded. They are not free parameters.
#
# =============================================================================
# USAGE  (run this ON THE CLIENT NODE)
#   bash run_inkling_1p1d_N4.sh                            # holds the pair up
#   WORKLOAD=iter32k bash run_inkling_1p1d_N4.sh
#   ROLE_STAGGER_SEC=0 bash run_inkling_1p1d_N4.sh         # simultaneous load
#   EXPECT_POOL_P=<tok> EXPECT_POOL_D=<tok> bash run_inkling_1p1d_N4.sh
#   KEEP_ALIVE=0 bash run_inkling_1p1d_N4.sh               # up, verify, down
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The workload point these servers are being provisioned FOR. Read from the
# same file bench_arm.sh reads, so the context budget set here and the ISL/OSL
# sent there come from ONE definition. They are decided hours apart, on
# different nodes; when they disagree the symptom is an HTTP 400 per request,
# which is instant and therefore reads on the progress bar as a fast request
# rather than as a fault (2026-09-11).
WORKLOAD=${WORKLOAD:-iter32k}
# shellcheck source=workload_profile.sh
source "${SCRIPT_DIR}/workload_profile.sh"
load_workload_profile "${WORKLOAD}" || exit 1

# =============================================================================
# LOCKED -- must match run_inkling_colocated_N2.sh exactly
# =============================================================================
# Every default in this block is shared with the colocated launcher and is a
# same-commit change if either moves. The Inkling rationale lives in that file;
# what is repeated here is only what a reader of THIS file needs in order not
# to break it.

# --- MODEL: resolved from the cache, not guessed ------------------------------
# Same scan as the colocated arm, and for the same second reason: it answers
# whether the weights are actually THERE. 532 GB does not arrive inside a
# health-check window, and finding out late costs an allocation.
HF_CACHE_ROOT=${HF_CACHE_ROOT:-/vast/draco/tara/projects/Tara_Deployment/software/model-weights/hub}
MODEL=${MODEL:-}
if [ -z "${MODEL}" ]; then
    mapfile -t _cands < <(find "${HF_CACHE_ROOT}" -maxdepth 1 -type d \
        -name 'models--*Inkling*' 2>/dev/null | sort)
    # Exclude thinkingmachines/Inkling, the 952B scale-up: it matches a bare
    # *Inkling* glob and is expected to be sitting in the same cache.
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
        echo "  A RIGHT-LOOKING wrong repo id silently serves a different" >&2
        echo "  model, and on this arm it would do so on both roles at once." >&2
        exit 1
    fi
fi

# --- Parallelism. Never TP=8. Identical on both roles. ------------------------
# HOMOGENEITY IS NOT A STYLE CHOICE HERE. base_worker.py's handshake validation
# ends with "Mamba doesn't support heterogeneous TP", and the tp_mapping
# machinery that makes P_TP != D_TP work for dense models has no hybrid path.
# NixlConnector additionally hashes vllm version, model, dtype, total KV heads,
# head size, layer count, attention backend, kv cache dtype and whether HMA is
# on into a compatibility hash exchanged at handshake, enforced by default
# (base_worker.py:554, metadata.py:113-127); block size, TP size and
# kv_cache_layout are checked separately (metadata.py:96-98). So an illegal
# asymmetry is a loud error rather than quiet corruption -- but DP is NOT in
# that hash, so keep DP equal by construction rather than by trust.
DP=${DP:-2}                 # one data-parallel rank per node, per role
TP=${TP:-4}                 # four GPUs per node, all-reduce stays on NVLink
DP_LOCAL=${DP_LOCAL:-1}     # one rank per node: head hosts 0, headless hosts 1
if [ "${TP}" -gt 4 ]; then
    echo "FATAL: TP=${TP}. Tensor parallelism must not leave the node." >&2
    echo "  TP=8 spans two nodes and puts the attention all-reduce on" >&2
    echo "  Slingshot. The vLLM Inkling blog's TP=8 example is GB200 NVL," >&2
    echo "  where eight GPUs are one NVLink domain. Use DP to add nodes." >&2
    exit 1
fi
if [ "${DP}" -ne 2 ]; then
    echo "FATAL: DP=${DP}. This launcher is the N4 shape: two nodes per role." >&2
    echo "  DP=1 does not load -- 532 GB of weights against ~345 GB per node." >&2
    echo "  DP>2 would need a wider allocation and a wider EP group, which" >&2
    echo "  would no longer match the colocated baseline's E=32." >&2
    exit 1
fi

GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-auto}

# EXPERT PARALLELISM IS ON, and it is what makes this arm comparable at all.
# 256 routed experts / (DP 2 x TP 4) = E=32 per rank, matching the colocated
# arm exactly. The heuristic fused-MoE kernel picks block sizes per E, so an
# arm at E=64 would run different kernels and the comparison would be measuring
# kernel selection (DECISIONS_2026-09-15c.md). EXPERT_PARALLEL=0 is a debugging
# escape hatch, not an arm.
EXPERT_PARALLEL=${EXPERT_PARALLEL:-1}

# --- HMA: explicitly ON, and on THIS arm it is load-bearing twice -------------
# In the colocated arm HMA governs allocation. Here it ALSO governs how much
# state crosses the fabric, which is the quantity this arm exists to measure.
#
# NixlConnector's _is_hma_required is `not disable_hybrid_kv_cache_manager AND
# any group that is not FullAttentionSpec` (base_scheduler.py:83-90,
# base_worker.py:285-291). With HMA on, the connector CLIPS each local layer's
# transfer to cdiv(window, block) + 1 blocks (base_scheduler.py:123-136). Turn
# it off and base_scheduler.py:255 returns early, the clipping is skipped, and
# every sliding-window layer is transferred as if it had unbounded context.
#
# THE SMOKE RUN HAS A COMPUTABLE PREDICTION, AND THIS IS IT. At ISL 32768,
# sliding_window_size 512 and block_size 16:
#
#     7 global layers  x 32768/16          = 14,336 blocks
#    35 local  layers  x (cdiv(512,16)+1)  =  1,155 blocks   (33 each)
#                                           ---------------
#                                            15,491 blocks
#
# against an unclipped 42 x 2048 = 86,016. The transfer should be ~18% of
# unclipped. Check nixl_bytes_transferred_sum against that ratio on the first
# smoke run; a number near 86,016 blocks' worth means HMA clipping is not
# happening and the arm is moving five times the state it should.
#
# Passed explicitly because unset means vLLM decides and can silently turn it
# OFF with a warning, after which a hybrid model fails at startup for a reason
# that does not mention HMA (vllm.py:1605-1676).
HMA_FLAG=${HMA_FLAG:---no-disable-hybrid-kv-cache-manager}

# Off on both arms. On this arm it has a second edge: if two requests share a
# prompt prefix, a cache hit lets D skip pulling state entirely, which looks
# fast for exactly the wrong reason.
PREFIX_CACHING_FLAG=${PREFIX_CACHING_FLAG:---no-enable-prefix-caching}

# --- NOT SET, and each absence is a decision ---------------------------------
#
# VLLM_SSM_CONV_STATE_LAYOUT
#     DO NOT SET IT. This is the one place the Nemotron disagg launcher is
#     actively misleading: there it is LOCKED, because base_worker.py:316
#     hard-asserts the DS layout for any model with a MambaSpec. Inkling's
#     sconv emits a SlidingWindowSpec, not a MambaSpec, so NixlConnector's
#     DS-layout assert is gated on `_has_mamba` (base_worker.py:305-317) and
#     NEVER FIRES. Setting it here would be cargo cult carried across from a
#     different model (run_inkling_colocated_N2.sh:302-316).
#
# --mamba-ssm-cache-dtype / --mamba-cache-dtype
#     Same reason. InklingConvState takes model_config.dtype and hard-asserts
#     bfloat16 (sconv_swa_attn.py:194-198). The Nemotron hazard -- a P/D conv
#     dtype mismatch producing GARBAGE rather than an error -- is INVERTED
#     here: the only lever is --dtype and anything but bfloat16 aborts at layer
#     construction, on both roles, loudly.
#
# --block-size
#     Left derived, and on this arm that matters more: under HMA vLLM picks an
#     attention block size that makes the page sizes line up across specs, and
#     the clip arithmetic above is a function of it. Read what it chose out of
#     the startup log before quoting the transfer ratio. BLOCK_SIZE=<n> to
#     experiment; it applies to both roles or the handshake rejects it.
#
# --speculative-config
#     MTP stays OFF, and on this arm the reason is sharper than on the
#     baseline. The vLLM Inkling blog reports 8 MTP heads, mean acceptance
#     length 4.5, and 380 vs 140 tok/s/user -- a 2.7x lever, by far the largest
#     available. But with MTP on, inter_token_latency_seconds stops being
#     per-token and the ITL distribution loses its meaning, and the ITL
#     distribution is what this arm exists to move. Run MTP afterwards as its
#     own arm; it cannot share a run with the disaggregation result.
BLOCK_SIZE=${BLOCK_SIZE:-}
ENFORCE_EAGER=${ENFORCE_EAGER:-}

# --- API_SERVER_COUNT: explicit, head nodes only, and never empty -------------
# Two landmines, both in entrypoints/cli/serve.py:
#   * On a HEADLESS node it is a HARD ERROR (serve.py:66-71, and run_headless()
#     raises again at :178). Appended inside the head branches only.
#   * On a HEAD node, LEFT UNSET IT SILENTLY DEFAULTS TO data_parallel_size
#     (serve.py:121) -- here 2, not 1.
#
# 16 matches the colocated arm and the Nemotron study. IT NOW FRONTS TWO
# SEPARATE ROLES: sixteen API servers on P head and sixteen on D head, 32
# processes across the arm against the baseline's 16. That is a real asymmetry
# in the frontend, it is forced by the shape, and it is why the samplers run on
# every node. Appendix A's finding -- that all sixteen share one listening
# socket -- applies to each head independently.
API_SERVER_COUNT=${API_SERVER_COUNT:-16}
if [ -z "${API_SERVER_COUNT}" ]; then
    echo "FATAL: API_SERVER_COUNT is empty. Omitting --api-server-count does" >&2
    echo "  not mean 1 -- serve.py:121 silently defaults it to" >&2
    echo "  data_parallel_size (${DP}). Pass a number." >&2
    exit 1
fi

MAX_MODEL_LEN=${MAX_MODEL_LEN:-${WL_MAX_MODEL_LEN}}

# --- NIXL ---------------------------------------------------------------------
# LIBFABRIC, not NIXL's own default of UCX: plain UCX has no native
# Slingshot/CXI transport. Confirmed by check_cxi_libfabric.sh on both nodes
# (2026-08-19): fi_info -p cxi works, libfabric 2.3.1, and NIXL's plugin
# manager lists LIBFABRIC as loadable in this conda env.
NIXL_BACKEND=${NIXL_BACKEND:-LIBFABRIC}

# kv_lease_duration -- base_scheduler.py:70 reads it from extra_config with a
# default of 30, and pull_scheduler.py:244-255 uses it as the TTL on the
# producer's blocks after request_finished(): P holds a finished request's KV
# for this many seconds so D can still read it.
#
# AT BENCHMARK CONCURRENCY THIS IS A HARD CAP ON P'S THROUGHPUT. P's usable KV
# is not its capacity but its capacity divided by how many requests finish
# inside the lease window. THE INKLING NUMBERS MAKE THIS WORSE THAN IT WAS ON
# NEMOTRON: the colocated arm measured prefill at ~17.9 s per request at c=64
# and the engine-resident set pinned near 31, so a 30 s lease can hold more
# than a full wave of finished requests' blocks. Left at 30 for bring-up
# because shortening it risks D losing its source mid-read -- which fails the
# run for a reason that looks like a fabric problem -- but 10 and then 5 are
# the first things to sweep once the arm is healthy. Watch P's preemption
# counter, which must stay at 0 for the run to be comparable.
KV_LEASE_DURATION=${KV_LEASE_DURATION:-30}

KV_EXTRA="\"backends\":[\"${NIXL_BACKEND}\"],\"kv_lease_duration\":${KV_LEASE_DURATION}"
KV_XFER_CONFIG_P="{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_producer\",\"kv_connector_extra_config\":{${KV_EXTRA}}}"
KV_XFER_CONFIG_D="{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_consumer\",\"kv_connector_extra_config\":{${KV_EXTRA}}}"

# --- Side-channel ports: RESPACED, and the old comment's reasoning is stale ---
# run_pd_nemotron_1p1d_N2.sh:738-755 argues at length that 5600/5601 need no
# respacing, quoting:
#
#   nixl/base_scheduler.py:65   self.side_channel_port = (
#                                   envs.VLLM_NIXL_SIDE_CHANNEL_PORT
#                                   + vllm_config.parallel_config.data_parallel_index)
#
# and concluding "the offset is DATA parallel index, not tensor parallel rank,
# and it is 0 for both roles here." THAT CONCLUSION WAS CORRECT AT DP=1 AND IS
# NOT CORRECT HERE. At DP=2 the offset is live: a base of 5600 on P yields 5600
# on the head and 5601 on the headless, and a base of 5601 on D would yield
# 5601 and 5602 -- two engines in different roles both advertising 5601.
#
# They sit on different NODES, so this is not an OS-level bind collision and it
# would probably work. It is respaced anyway, because "probably works, and the
# port number no longer tells you which role you are looking at" is not a
# property worth having in the one subsystem whose failures look like fabric
# faults. 100 apart leaves room for DP up to 100 without revisiting this.
#
# THE HOST IS PER-NODE, NOT PER-ROLE. Each engine is its own NIXL agent on its
# own node, so every one of the four advertises its OWN hsn0 address. Setting
# the role head's address on the headless node would have the headless engine
# advertise an endpoint it does not own -- and the symptom is a transfer that
# hangs rather than an error.
SIDE_PORT_P=${SIDE_PORT_P:-5600}
SIDE_PORT_D=${SIDE_PORT_D:-5700}

# --- ASYMMETRIC: the entire point of disaggregation --------------------------
# P is compute-bound and wants few, large batches. D is bandwidth-bound and
# wants as many concurrent sequences as the KV budget allows.
#
# THE NUMBERS ARE INHERITED FROM THE NEMOTRON PAIR BUT THE REASONING IS NOT,
# AND THE DIFFERENCE MATTERS. Nemotron argued D's 256 from a 41.9 MB/sequence
# constant Mamba state, giving ~1,100 sequences of headroom before attention KV
# was counted at all. INKLING HAS NO SUCH CONSTANT: a request costs ~1.01 GB
# dominated by global-attention KV at 32k (MEMORY_2026-09-12d.md), and the
# colocated sweep measured the engine-resident set pinning near 31 across both
# c=32 and c=64. So on Inkling D's 256 is NON-BINDING rather than generous --
# KV binds first, exactly as it does on every run this rig has produced. Leave
# it; the number that actually controls concurrency is the client's
# --max-concurrency.
#
# AND THESE FOUR NUMBERS ARE FROZEN. The colocated baseline's 16384 / 256 is
# max(P 16384, D 2048) and max(P 32, D 256). Change any of them and the
# published c=16 / c=32 / c=64 colocated sweep has to be re-run.
P_MAX_NUM_BATCHED_TOKENS=${P_MAX_NUM_BATCHED_TOKENS:-16384}
P_MAX_NUM_SEQS=${P_MAX_NUM_SEQS:-32}
D_MAX_NUM_BATCHED_TOKENS=${D_MAX_NUM_BATCHED_TOKENS:-2048}
D_MAX_NUM_SEQS=${D_MAX_NUM_SEQS:-256}

# --- NUMA binding: first-class, explicit, never auto-detected ----------------
# THE SEPARATOR IS A SPACE, NOT A COMMA, AND THE LIST IS INDEXED BY GPU INDEX.
# --numa-bind-nodes is nargs='+' of int; the comma form dies at argparse with
# "Value 0,1,2,3 cannot be converted to <class 'int'>", rank 0 exits code 2,
# PALS signal-15s the rest, and the application is gone before a weight is read
# (2026-09-15). numa_utils.py:261-263 indexes the list BY GPU INDEX and raises
# if gpu_index >= len(numa_bind_nodes), so it needs one entry per visible GPU
# in GPU order. On these nodes each GH200 module is its own Grace NUMA node, so
# the identity map is correct. CPU affinity is 0/1/2/3; 4/12/20/28 are the HBM
# nodes and must NOT appear here.
#
# It does NOT bind the API servers -- numa_utils covers TP/PP worker
# subprocesses and the EngineCore only (CLOSED.md). On the two head nodes the
# sixteen API servers still float across all 288 cores.
NUMA_BIND=${NUMA_BIND:-1}
NUMA_BIND_NODES="${NUMA_BIND_NODES:-0 1 2 3}"

# --- Stagger: the roles do NOT load simultaneously ---------------------------
# Two 532 GB weight loads off /vast at once is a filesystem contention problem
# that presents as a slow startup, and the colocated arm already budgets 40
# minutes for ONE of them.
#
# D STARTS FIRST. In a real deployment D is the long-lived role and P is what
# scales elastically, so this is the order production wants -- which means the
# benchmark script and a production script agree rather than diverge.
#
# THE SLEEP IS INSIDE launch_role.sh, NOT AROUND mpiexec, AND THAT IS LOAD-
# BEARING. CXI needs a VNI, the VNI arrives only in SLINGSHOT_VNIS provisioned
# by PALS, and PALS allocates a VNI PER APPLICATION. Two `mpiexec` calls get
# two different VNIs -- three separate launches were observed getting 1726,
# 1825 and 1873 -- and endpoints on different VNIs cannot reach each other at
# all. Staggering by splitting the launch would therefore break NIXL between
# the roles, which is the one thing this arm cannot survive. One application,
# four ranks, and P sleeps after the application already exists.
#
# BOTH RANKS OF A ROLE SHARE A STAGGER, and that is load-bearing rather than
# tidy: the two ranks of one role rendezvous over --data-parallel-address /
# --data-parallel-rpc-port at startup, so staggering one rank against its own
# partner would hang the DP handshake rather than the role.
#
# UNVERIFIED, AND THE FIRST THING TO WATCH ON THE SMOKE RUN: this assumes
# NixlConnector's P<->D handshake is LAZY -- that D, the consumer, only reaches
# for the producer when a request arrives carrying its address from the proxy,
# and therefore does not care that P does not exist for the first ~300 s of its
# life. If the handshake turns out to be EAGER, D will fail at startup with
# something that looks like a fabric error.
#
# IF THAT HAPPENS, REVERSE THE STAGGER (P first), DO NOT DISABLE IT. Setting
# ROLE_STAGGER_SEC=0 puts two 532 GB loads on /vast simultaneously, which is
# the problem this exists to avoid. P-first costs nothing here -- the
# production argument for D-first is about which role you keep alive, not which
# one you start.
ROLE_STAGGER_SEC=${ROLE_STAGGER_SEC:-300}

KEEP_ALIVE=${KEEP_ALIVE:-1}
KEEP_ALIVE_POLL_S=${KEEP_ALIVE_POLL_S:-60}
# 480 x 5s = 40 min covers ONE 532 GB load. P does not begin loading until the
# stagger has elapsed, so its budget has to cover the stagger too -- otherwise
# a 300 s stagger silently eats 12.5% of the window and a launch that would
# have succeeded times out. Derived rather than left as a constant that drifts
# out of step with ROLE_STAGGER_SEC.
HEALTH_TRIES=${HEALTH_TRIES:-$(( 480 + (ROLE_STAGGER_SEC + 4) / 5 ))}
SAMPLE_INTERVAL_S=${SAMPLE_INTERVAL_S:-10}

P_PORT=${P_PORT:-8100}
D_PORT=${D_PORT:-8200}
PROXY_PORT=${PROXY_PORT:-8000}
# One RPC port per role. They land on different nodes so equal values would
# bind cleanly, but the DP handshake is addressed by (address, port) and a
# reader tracing a hang should not have to check which node a port belongs to.
DP_RPC_PORT_P=${DP_RPC_PORT_P:-29550}
DP_RPC_PORT_D=${DP_RPC_PORT_D:-29560}
PROXY_SCRIPT=${PROXY_SCRIPT:-/vast/draco/tara/projects/Tara_Deployment/software/testing/vllm_0.27.1_08_18_2026/vllm/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py}

RUNS_ROOT=${RUNS_ROOT:-/vast/draco/tara/projects/Tara_Deployment/software/testing/RUNS}
STAMP=$(date +%Y%m%d_%H%M%S)
SHARED=${SHARED:-${RUNS_ROOT}/inkling_1p1d_dp${DP}tp${TP}_${STAMP}}

# =============================================================================
# Resolve the nodes. THIS SCRIPT RUNS ON THE CLIENT NODE.
# =============================================================================
# Structural, not written down: the node this script runs on is the CLIENT
# node, and it is refused a serving role. Moving the client off a serving node
# was worth +37.7% throughput and -14.4% engine work per token on an otherwise
# identical run (TECHNICAL_REPORT_1.tex S7).
#
# THE ALLOCATION IS SIX. "N4" counts SERVING nodes, matching the convention of
# the colocated arm's "N2":
#
#     P head + P headless     8 GPUs
#     D head + D headless     8 GPUs
#     proxy                   toy_proxy_server.py, nothing else
#     client                  this script, bench_arm.sh, nothing else
#
# THE PROXY GETS ITS OWN NODE, which is a change from the Nemotron pair where
# it shared P's head. It is on the critical path of every request AND every
# streamed token, its CPU cost is a known concern, and on P's head it would
# compete with sixteen API servers and four TP workers.
#
# THE COST OF THAT CHOICE, PRE-REGISTERED SO IT IS CHECKED RATHER THAN
# DISCOVERED: it adds one fabric hop to the PER-TOKEN path, D -> proxy ->
# client instead of D -> client. Against a ~67 ms steady-state ITL that should
# be well under 1%. bench_arm.sh reports frontend tax per arm (the colocated
# arm measured 4.553 s/req, 1.3% at c=32, and 4.206 s/req, 0.6% at c=64). If
# this arm's frontend tax is materially worse than that, move the proxy back
# onto P's head and eat the CPU contention instead -- but record which was
# used, because it is a difference between arms that is not the P/D split.
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

NEED_ENGINE_NODES=$(( DP * 2 ))     # two roles
NEED_ROLE_NODES=$(( NEED_ENGINE_NODES + 1 ))   # plus the proxy
if [ "${#ROLE_NODES[@]}" -lt "${NEED_ROLE_NODES}" ]; then
    echo "FATAL: need ${NEED_ENGINE_NODES} engine nodes + 1 proxy node PLUS this client node." >&2
    echo "  Allocation has ${#ALL_NODES[@]} node(s); this one (${CLIENT_HOST}) is the" >&2
    echo "  client, leaving ${#ROLE_NODES[@]}." >&2
    echo "" >&2
    echo "  Ask for $(( NEED_ROLE_NODES + 1 )). Two nodes per role is not a tuning" >&2
    echo "  choice: Inkling-Small is 532 GB of BF16 weights against ~345 GB" >&2
    echo "  usable per node, so DP=1 does not load at all." >&2
    echo "" >&2
    echo "  If only $(( NEED_ENGINE_NODES + 1 )) are available, PROXY_ON_P_HEAD=1 drops the" >&2
    echo "  proxy node and colocates it with P's head -- at the cost of a" >&2
    echo "  frontend-tax difference against the colocated arm. Record it." >&2
    exit 1
fi

NODE_P_HEAD="${ROLE_NODES[0]}"
NODE_P_TAIL="${ROLE_NODES[1]}"
NODE_D_HEAD="${ROLE_NODES[2]}"
NODE_D_TAIL="${ROLE_NODES[3]}"
NODE_P_HEAD_SHORT="${NODE_P_HEAD%%.*}"
NODE_P_TAIL_SHORT="${NODE_P_TAIL%%.*}"
NODE_D_HEAD_SHORT="${NODE_D_HEAD%%.*}"
NODE_D_TAIL_SHORT="${NODE_D_TAIL%%.*}"

# PROXY_ON_P_HEAD=1 is the degraded five-node fallback, not the default.
PROXY_ON_P_HEAD=${PROXY_ON_P_HEAD:-0}
if [ "${PROXY_ON_P_HEAD}" = "1" ]; then
    NODE_PROXY="${NODE_P_HEAD}"
else
    NODE_PROXY="${ROLE_NODES[4]}"
fi
NODE_PROXY_SHORT="${NODE_PROXY%%.*}"

ENGINE_NODES=("${NODE_P_HEAD}" "${NODE_P_TAIL}" "${NODE_D_HEAD}" "${NODE_D_TAIL}")

if [ "${#ROLE_NODES[@]}" -gt "${NEED_ROLE_NODES}" ]; then
    echo "NOTE: ${#ROLE_NODES[@]} non-client nodes allocated; this arm uses ${NEED_ROLE_NODES}."
    echo "      The rest idle. Correct -- per-GPU normalization counts serving GPUs."
fi

# PBS_NODEFILE carries FQDNs but `hostname -s` on a compute node returns only
# the leading label. Keep BOTH forms: ssh wants the FQDN, launch_role.sh's
# hostname comparison has to be short-vs-short. Getting that wrong is silent
# and total -- every rank falls through to the else branch, the first to exit
# takes the whole PALS application down, and no role log is ever written
# (2026-08-25, the Nemotron pair).
echo "Client node (serves nothing): ${CLIENT_HOST}"
echo "P head     (API + DP0):       ${NODE_P_HEAD_SHORT}"
echo "P headless (DP1):             ${NODE_P_TAIL_SHORT}"
echo "D head     (API + DP0):       ${NODE_D_HEAD_SHORT}"
echo "D headless (DP1):             ${NODE_D_TAIL_SHORT}"
echo "Proxy:                        ${NODE_PROXY_SHORT}$([ "${PROXY_ON_P_HEAD}" = "1" ] && echo '  (COLOCATED WITH P HEAD -- degraded, record it)')"

get_hsn_ip() { ssh -n "$1" "ip -4 -o addr show hsn0" 2>/dev/null | awk '{print $4}' | cut -d/ -f1; }
P_HEAD_IP=$(get_hsn_ip "${NODE_P_HEAD}")
P_TAIL_IP=$(get_hsn_ip "${NODE_P_TAIL}")
D_HEAD_IP=$(get_hsn_ip "${NODE_D_HEAD}")
D_TAIL_IP=$(get_hsn_ip "${NODE_D_TAIL}")
PROXY_IP=$(get_hsn_ip "${NODE_PROXY}")
CLIENT_IP=$(ip -4 -o addr show hsn0 2>/dev/null | awk '{print $4}' | cut -d/ -f1)
for _pair in "P_HEAD:${P_HEAD_IP}" "P_TAIL:${P_TAIL_IP}" "D_HEAD:${D_HEAD_IP}" \
             "D_TAIL:${D_TAIL_IP}" "PROXY:${PROXY_IP}" "CLIENT:${CLIENT_IP}"; do
    if [ -z "${_pair#*:}" ]; then
        echo "FATAL: could not resolve an hsn0 address for ${_pair%%:*}." >&2
        exit 1
    fi
done
echo "  hsn0: P ${P_HEAD_IP} / ${P_TAIL_IP}   D ${D_HEAD_IP} / ${D_TAIL_IP}"
echo "        proxy ${PROXY_IP}   client ${CLIENT_IP}"
echo "  Bench endpoint will be http://${PROXY_IP}:${PROXY_PORT} (the proxy)"
echo "  Metrics come from the two HEADS directly -- the proxy has no /metrics."

mkdir -p "${SHARED}/logs" || { echo "Cannot create ${SHARED}" >&2; exit 1; }
for n in "${ENGINE_NODES[@]}" "${NODE_PROXY}"; do
    ssh -n "$n" "test -d ${SHARED}" || {
        echo "FATAL: ${SHARED} is not visible on ${n%%.*} -- is /vast/draco mounted?" >&2
        exit 1
    }
done

# =============================================================================
# Preflight -- fail before two 532 GB weight loads, not during them
# =============================================================================
# The Nemotron disagg launcher had no gate at all and discovered a dirty node
# ~40 minutes in at KV-cache init as "No available memory for the cache
# blocks". On this arm that would be ~40 minutes plus the stagger, on four
# nodes at once.
#
# THE PROCESS PATTERNS LIVE IN gpu_cleanup.sh AND ARE SOURCED, NOT COPIED. They
# have drifted once already: 'Worker_TP' was written against Nemotron-era
# naming, but under DP + EP the workers are titled Worker_DP0_TP1_EP1, in which
# that substring does not occur -- so preflight declared nodes clean while a
# previous run's workers held GPU memory, and cleanup's kill matched nothing.
# WHEN vLLM IS UPGRADED, RE-CHECK THEM AGAINST `pgrep -af` ON A LIVE RUN.
if [ ! -f "${SCRIPT_DIR}/gpu_cleanup.sh" ]; then
    echo "FATAL: missing ${SCRIPT_DIR}/gpu_cleanup.sh -- preflight and teardown" >&2
    echo "  both read the process-pattern list from it." >&2
    exit 1
fi
# shellcheck source=./gpu_cleanup.sh
source "${SCRIPT_DIR}/gpu_cleanup.sh"
PAT_VLLM="${VLLM_PROC_PATTERNS[0]}"
# Bracketed so `ssh host "pkill -f '[t]oy_proxy_server.py'"` does not match the
# far-side `bash -c` whose own command line contains the pattern.
PAT_PROXY='[t]oy_proxy_server.py'

GPU_DIRTY_MIB=${GPU_DIRTY_MIB:-4096}
_preflight_fail=0
for n in "${ENGINE_NODES[@]}"; do
    _short="${n%%.*}"
    echo "PREFLIGHT ${_short}:"
    ssh -n "$n" "test -f ${SCRIPT_DIR}/gpu_cleanup.sh" || {
        echo "FATAL: ${SCRIPT_DIR}/gpu_cleanup.sh not visible from ${_short}." >&2
        exit 1
    }
    ssh -n "$n" "GPU_DIRTY_MIB=${GPU_DIRTY_MIB} bash ${SCRIPT_DIR}/gpu_cleanup.sh report" \
        2>&1 | sed 's/^/  /'
    # PIPESTATUS, not $?, which would be sed's. Getting this wrong makes the
    # gate pass unconditionally -- worse than no gate, because it reads as a
    # node that was checked.
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        _preflight_fail=1
    fi
done
if ssh -n "${NODE_P_HEAD}" "fuser ${P_PORT}/tcp" >/dev/null 2>&1; then
    echo "PREFLIGHT ${NODE_P_HEAD_SHORT}: port ${P_PORT} is already bound." >&2
    _preflight_fail=1
fi
if ssh -n "${NODE_D_HEAD}" "fuser ${D_PORT}/tcp" >/dev/null 2>&1; then
    echo "PREFLIGHT ${NODE_D_HEAD_SHORT}: port ${D_PORT} is already bound." >&2
    _preflight_fail=1
fi
if ssh -n "${NODE_PROXY}" "fuser ${PROXY_PORT}/tcp" >/dev/null 2>&1; then
    echo "PREFLIGHT ${NODE_PROXY_SHORT}: proxy port ${PROXY_PORT} is already bound." >&2
    _preflight_fail=1
fi
if [ "${_preflight_fail}" -ne 0 ]; then
    echo "" >&2
    echo "Refusing to start on a dirty node. Clear with the same file the" >&2
    echo "report came from, so the sweep and the screen agree:" >&2
    echo "" >&2
    echo "  for n in ${NODE_P_HEAD_SHORT} ${NODE_P_TAIL_SHORT} ${NODE_D_HEAD_SHORT} ${NODE_D_TAIL_SHORT}; do" >&2
    echo "    ssh \$n \"bash ${SCRIPT_DIR}/gpu_cleanup.sh kill ${P_PORT} ${D_PORT} ${DP_RPC_PORT_P} ${DP_RPC_PORT_D}\"" >&2
    echo "  done" >&2
    echo "" >&2
    echo "Then re-run. If the report still shows GPU memory AND 'no process you" >&2
    echo "own holds HBM pages', it is a leaked context: TAKE ANOTHER NODE." >&2
    echo "" >&2
    echo "THIS HAS NOW HAPPENED ON 2 OF 5 INKLING LAUNCHES -- a node passes" >&2
    echo "preflight and then fails init_device seconds later. The remedy both" >&2
    echo "times was to swap which node holds the head role and relaunch. Never" >&2
    echo "lower --gpu-memory-utilization to get past it: that changes the KV" >&2
    echo "pool and voids the repeat, and it is what the error itself suggests." >&2
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
# own .c. On THIS arm it is not decorative -- LIBFABRIC is the NIXL backend and
# the shim is the thing that gets fi_getinfo past the topology error.
_shim_patches=$(strings "${SHIM}" 2>/dev/null | grep -c 'PATCH [234]' || true)
if [ "${_shim_patches}" -lt 1 ]; then
    echo "WARNING: ${SHIM} has no PATCH [234] strings -- it looks stale." >&2
    echo "         This arm actually reaches libfabric. Rebuild before trusting it." >&2
fi
for n in "${ENGINE_NODES[@]}"; do
    ssh -n "$n" "test -f ${SHIM}" || {
        echo "FATAL: ${SHIM} is not visible from ${n%%.*}." >&2
        exit 1
    }
done
if [ ! -f "${PROXY_SCRIPT}" ]; then
    echo "FATAL: PROXY_SCRIPT not found at ${PROXY_SCRIPT}." >&2
    echo "  Set PROXY_SCRIPT=/actual/path and rerun." >&2
    exit 1
fi

# =============================================================================
# Generate common_env.sh and the sampler via the SHARED emitters
# =============================================================================
# NO_PROXY_LIST covers all six nodes. Without it, curl to an hsn0 address goes
# to proxy.alcf.anl.gov, comes back a Squid error page, and reads as a dead
# server rather than a misrouted client (2026-09-11, twice).
#
#   UCX_LINES    set only when NIXL_BACKEND=UCX, so the default LIBFABRIC run
#                emits exactly what the colocated arm emits.
#   GPU_PIN_LINE empty. TP=4 wants all four GPUs visible on every role node.
NO_PROXY_LIST="localhost,127.0.0.1,${P_HEAD_IP},${P_TAIL_IP},${D_HEAD_IP},${D_TAIL_IP},${PROXY_IP},${CLIENT_IP},${NODE_P_HEAD},${NODE_P_TAIL},${NODE_D_HEAD},${NODE_D_TAIL},${NODE_PROXY},${NODE_P_HEAD_SHORT},${NODE_P_TAIL_SHORT},${NODE_D_HEAD_SHORT},${NODE_D_TAIL_SHORT},${NODE_PROXY_SHORT},${CLIENT_HOST}"
UCX_LINES=""
if [ "${NIXL_BACKEND}" = "UCX" ]; then
    UCX_LINES=$'export UCX_TLS=rc,ud,sm,self\nexport UCX_NET_DEVICES=all'
fi
GPU_PIN_LINE=""
# shellcheck source=./emit_common_env.sh
source "${SCRIPT_DIR}/emit_common_env.sh"
emit_common_env "${SHARED}/common_env.sh"
# shellcheck source=./emit_conn_sampler.sh
source "${SCRIPT_DIR}/emit_conn_sampler.sh"
emit_conn_sampler "${SHARED}/sample_conns.sh"
echo "Wrote ${SHARED}/common_env.sh and sample_conns.sh (shared emitters, not copies)."

# =============================================================================
# Generate the role script -- ONE file, FOUR nodes, selecting by hostname
# =============================================================================
# One application, one VNI. See the ROLE_STAGGER_SEC block above for why the
# stagger lives in here rather than around mpiexec.
#
# Unquoted heredoc, so values resolve now; \${...} escapes are the handful that
# must survive to runtime. An unescaped $ here bakes a launcher-time value into
# a runtime file.
cat > "${SHARED}/launch_role.sh" <<EOF
#!/bin/bash
source ${SHARED}/common_env.sh

# THE DISAGG ARM SETS EXACTLY THREE NIXL VARIABLES -- VLLM_NIXL_SIDE_CHANNEL_HOST,
# _PORT and NIXL_LOG_LEVEL -- and adds --kv-transfer-config below. Together with
# the batching split that is the COMPLETE list of differences from
# run_inkling_colocated_N2.sh's launch_role.sh. Anything else that differs is a
# bug in one of them.

MY_HOST="\$(hostname -s)"
MY_HOST="\${MY_HOST%%.*}"
if [ "\${MY_HOST}" = "${NODE_P_HEAD_SHORT}" ]; then
    ROLE=p-head
    LOG=${SHARED}/logs/p-head.log
    STAGGER=${ROLE_STAGGER_SEC}
    DP_ARGS=(--data-parallel-size ${DP}
             --data-parallel-size-local ${DP_LOCAL}
             --data-parallel-address ${P_HEAD_IP}
             --data-parallel-rpc-port ${DP_RPC_PORT_P}
             --api-server-count ${API_SERVER_COUNT})
    SERVE_ARGS=(--host 0.0.0.0 --port ${P_PORT})
    SIDE_HOST=${P_HEAD_IP}
    SIDE_PORT=${SIDE_PORT_P}
    KV_CFG='${KV_XFER_CONFIG_P}'
    MAX_BATCHED_TOKENS=${P_MAX_NUM_BATCHED_TOKENS}
    MAX_SEQS=${P_MAX_NUM_SEQS}
elif [ "\${MY_HOST}" = "${NODE_P_TAIL_SHORT}" ]; then
    ROLE=p-headless
    LOG=${SHARED}/logs/p-headless.log
    STAGGER=${ROLE_STAGGER_SEC}
    # --api-server-count is a HARD ERROR with --headless (serve.py:66-71, and
    # run_headless() raises again at :178). Absent here, deliberately.
    DP_ARGS=(--headless
             --data-parallel-size ${DP}
             --data-parallel-size-local ${DP_LOCAL}
             --data-parallel-start-rank 1
             --data-parallel-address ${P_HEAD_IP}
             --data-parallel-rpc-port ${DP_RPC_PORT_P})
    SERVE_ARGS=()
    # Its OWN address: this engine is its own NIXL agent on its own node.
    # Advertising the head's address here produces a transfer that hangs.
    SIDE_HOST=${P_TAIL_IP}
    SIDE_PORT=${SIDE_PORT_P}
    KV_CFG='${KV_XFER_CONFIG_P}'
    MAX_BATCHED_TOKENS=${P_MAX_NUM_BATCHED_TOKENS}
    MAX_SEQS=${P_MAX_NUM_SEQS}
elif [ "\${MY_HOST}" = "${NODE_D_HEAD_SHORT}" ]; then
    ROLE=d-head
    LOG=${SHARED}/logs/d-head.log
    STAGGER=0                      # D loads first
    DP_ARGS=(--data-parallel-size ${DP}
             --data-parallel-size-local ${DP_LOCAL}
             --data-parallel-address ${D_HEAD_IP}
             --data-parallel-rpc-port ${DP_RPC_PORT_D}
             --api-server-count ${API_SERVER_COUNT})
    SERVE_ARGS=(--host 0.0.0.0 --port ${D_PORT})
    SIDE_HOST=${D_HEAD_IP}
    SIDE_PORT=${SIDE_PORT_D}
    KV_CFG='${KV_XFER_CONFIG_D}'
    MAX_BATCHED_TOKENS=${D_MAX_NUM_BATCHED_TOKENS}
    MAX_SEQS=${D_MAX_NUM_SEQS}
elif [ "\${MY_HOST}" = "${NODE_D_TAIL_SHORT}" ]; then
    ROLE=d-headless
    LOG=${SHARED}/logs/d-headless.log
    STAGGER=0
    DP_ARGS=(--headless
             --data-parallel-size ${DP}
             --data-parallel-size-local ${DP_LOCAL}
             --data-parallel-start-rank 1
             --data-parallel-address ${D_HEAD_IP}
             --data-parallel-rpc-port ${DP_RPC_PORT_D})
    SERVE_ARGS=()
    SIDE_HOST=${D_TAIL_IP}
    SIDE_PORT=${SIDE_PORT_D}
    KV_CFG='${KV_XFER_CONFIG_D}'
    MAX_BATCHED_TOKENS=${D_MAX_NUM_BATCHED_TOKENS}
    MAX_SEQS=${D_MAX_NUM_SEQS}
else
    echo "Rank landed on unexpected host '\${MY_HOST}'." >&2
    echo "  expected one of:" >&2
    echo "    ${NODE_P_HEAD_SHORT} (p-head)     ${NODE_P_TAIL_SHORT} (p-headless)" >&2
    echo "    ${NODE_D_HEAD_SHORT} (d-head)     ${NODE_D_TAIL_SHORT} (d-headless)" >&2
    exit 1
fi

exec > "\${LOG}" 2>&1
echo "=== role=\${ROLE} host=\$(hostname -s) SLINGSHOT_VNIS=\${SLINGSHOT_VNIS:-<UNSET>} ==="
if [ -z "\${SLINGSHOT_VNIS:-}" ]; then
    echo "SLINGSHOT_VNIS is EMPTY -- this process was not launched under PALS."
    echo "Anything that reaches the cxi provider will fail fi_domain() with"
    echo "-FI_ENOSYS, and on THIS arm that means no KV transfer at all."
    echo "Check the mpiexec invocation."
fi

# The stagger. Inside the application, so the VNI already exists and is shared
# with the role that is already loading.
if [ "\${STAGGER}" -gt 0 ]; then
    echo "=== staggered start: sleeping \${STAGGER}s so D loads its 532 GB first ==="
    echo "    (two simultaneous loads off /vast contend; ROLE_STAGGER_SEC=0 disables)"
    sleep "\${STAGGER}"
    echo "=== stagger elapsed at \$(date -Is) -- starting ==="
fi

export LD_PRELOAD=${SHIM}
export VLLM_NIXL_SIDE_CHANNEL_HOST=\${SIDE_HOST}
export VLLM_NIXL_SIDE_CHANNEL_PORT=\${SIDE_PORT}
export NIXL_LOG_LEVEL=\${NIXL_LOG_LEVEL:-INFO}

EXTRA=()
# BLOCK_SIZE and ENFORCE_EAGER default to EMPTY and are tested with -n: unset
# means "do not pass the flag". EXPERT_PARALLEL and NUMA_BIND default to a
# VALUE and are tested with = "1", so that EXPERT_PARALLEL=0 -- the one
# spelling a reader would reach for -- actually turns it off.
[ -n "${BLOCK_SIZE}" ]           && EXTRA+=(--block-size ${BLOCK_SIZE})
[ "${EXPERT_PARALLEL}" = "1" ]   && EXTRA+=(--enable-expert-parallel)
[ -n "${ENFORCE_EAGER}" ]        && EXTRA+=(--enforce-eager)
[ "${NUMA_BIND}" = "1" ]         && EXTRA+=(--numa-bind --numa-bind-nodes ${NUMA_BIND_NODES})

# --distributed-executor-backend mp is EXPLICIT on all four ranks. Unset yields
# mp only because Ray is not initialized; ray>=2.55.0 IS installed in this env
# and the sole guard is one ray_is_initialized() call (parallel.py:941-951).
# Each role is its own engine at world_size = TP = 4; mp never crosses a node.
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
    --max-num-batched-tokens \${MAX_BATCHED_TOKENS} \\
    --max-num-seqs \${MAX_SEQS} \\
    \${EXTRA[@]+"\${EXTRA[@]}"} \\
    --kv-transfer-config "\${KV_CFG}"
EOF
chmod +x "${SHARED}/launch_role.sh"

# =============================================================================
# Generate the proxy launcher
# =============================================================================
# toy_proxy_server.py is the reference implementation from
# tests/v1/kv_connector/nixl_integration/. Its health endpoint is /healthcheck,
# NOT /health -- confirmed by reading the script. It sends
# "Authorization: Bearer $OPENAI_API_KEY" on every request to P and D; neither
# is launched with --api-key so it is not validated either way, but set a real
# value rather than send "Bearer None" on an unvalidated assumption.
#
# --host 0.0.0.0 is explicit rather than relying on the script's 127.0.0.1
# default -- the same "bound somewhere unreachable by default" pattern that bit
# the NIXL side-channel host.
#
# NO LD_PRELOAD HERE. The shim has no business interposing fi_getinfo for a
# plain HTTP process, and common_env.sh does not set it.
#
# IT POINTS AT THE TWO HEADS. Only the heads serve HTTP; the headless nodes
# have no frontend at all, so there is nothing for the proxy to address there.
cat > "${SHARED}/launch_proxy.sh" <<EOF
#!/bin/bash
source ${SHARED}/common_env.sh
export OPENAI_API_KEY=inkling-1p1d-dummy-key
python3 ${PROXY_SCRIPT} \\
    --host 0.0.0.0 --port ${PROXY_PORT} \\
    --prefiller-host ${P_HEAD_IP} --prefiller-port ${P_PORT} \\
    --decoder-host ${D_HEAD_IP} --decoder-port ${D_PORT}
EOF
chmod +x "${SHARED}/launch_proxy.sh"

# --- Provenance, recorded rather than hoarded --------------------------------
if [ "${EXPERT_PARALLEL}" = "1" ]; then _ep_state="on"; else _ep_state="OFF"; fi
{
    echo "stamp             ${STAMP}"
    echo "arm               inkling-disagg 1P:1D  (two roles x ${DP} nodes, $(( DP * TP * 2 )) GPUs)"
    echo "model             ${MODEL}"
    echo "CLIENT NODE       ${CLIENT_HOST}  hsn0 ${CLIENT_IP}   <- serves nothing; bench runs here"
    echo "PROXY NODE        ${NODE_PROXY_SHORT}  hsn0 ${PROXY_IP}  http ${PROXY_PORT}  on-p-head=${PROXY_ON_P_HEAD}"
    echo "P head            ${NODE_P_HEAD_SHORT}  hsn0 ${P_HEAD_IP}  http ${P_PORT}  dp-rank 0  api ${API_SERVER_COUNT}  side ${SIDE_PORT_P}"
    echo "P headless        ${NODE_P_TAIL_SHORT}  hsn0 ${P_TAIL_IP}  no http       dp-rank 1  api 0   side $(( SIDE_PORT_P + 1 ))"
    echo "D head            ${NODE_D_HEAD_SHORT}  hsn0 ${D_HEAD_IP}  http ${D_PORT}  dp-rank 0  api ${API_SERVER_COUNT}  side ${SIDE_PORT_D}"
    echo "D headless        ${NODE_D_TAIL_SHORT}  hsn0 ${D_TAIL_IP}  no http       dp-rank 1  api 0   side $(( SIDE_PORT_D + 1 ))"
    echo "parallelism       DP ${DP} x TP ${TP} PER ROLE, expert-parallel ${_ep_state}  (EP group ${DP} x ${TP} = $(( DP * TP )), E=$(( 256 / (DP * TP) )))"
    echo "executor          mp, explicit on all four ranks"
    echo "nixl              backend ${NIXL_BACKEND}, kv_lease_duration ${KV_LEASE_DURATION}"
    echo "side-channel      P base ${SIDE_PORT_P}, D base ${SIDE_PORT_D}  (base + data_parallel_index, base_scheduler.py:65)"
    echo "stagger           ${ROLE_STAGGER_SEC}s, D first  (health budget ${HEALTH_TRIES} x 5s)"
    echo "max-model-len     ${MAX_MODEL_LEN}"
    echo "provisioned for   ${WORKLOAD}  (ISL ${WL_ISL} / OSL ${WL_OSL}, needs ${WL_MAX_MODEL_LEN})"
    echo "batching          P ${P_MAX_NUM_BATCHED_TOKENS} tok / ${P_MAX_NUM_SEQS} seq   D ${D_MAX_NUM_BATCHED_TOKENS} tok / ${D_MAX_NUM_SEQS} seq  (colocated takes the per-knob max)"
    echo "numa-bind         ${NUMA_BIND} nodes ${NUMA_BIND_NODES}  (workers + EngineCore only; NOT the API servers)"
    echo "kv dtype          ${KV_CACHE_DTYPE}   model dtype bfloat16 (hard-asserted by InklingConvState)"
    echo "shim              ${SHIM}  sha256 $(sha256sum "${SHIM}" 2>/dev/null | awk '{print $1}')"
    echo "git               $(git -C "${SCRIPT_DIR}" rev-parse --short HEAD 2>/dev/null || echo '(not a work tree)')"
    echo ""
    echo "=== nvidia-smi topo -m on ${NODE_P_HEAD_SHORT} ==="
    ssh -n "${NODE_P_HEAD}" "nvidia-smi topo -m" 2>&1
} > "${SHARED}/run_config.txt"
sed -n '1,22p' "${SHARED}/run_config.txt"

# =============================================================================
# Teardown
# =============================================================================
cleanup() {
    echo "=== Cleaning up ==="
    touch "${SAMPLE_STOP_MARKER:-}" 2>/dev/null
    ssh -n "${NODE_PROXY}" "pkill -TERM -f \"${PAT_PROXY}\"" 2>/dev/null
    for n in "${ENGINE_NODES[@]}"; do
        ssh -n "$n" "pkill -TERM -f \"${PAT_VLLM}\"" 2>/dev/null
    done
    kill -TERM ${MPIEXEC_PID:-} ${TAIL_PH_PID:-} ${TAIL_PT_PID:-} \
               ${TAIL_DH_PID:-} ${TAIL_DT_PID:-} ${TAIL_M_PID:-} \
               ${PROXY_SSH_PID:-} ${TAIL_PROXY_PID:-} 2>/dev/null
    sleep 5
    ssh -n "${NODE_PROXY}" "pkill -KILL -f \"${PAT_PROXY}\"; fuser -k ${PROXY_PORT}/tcp" 2>/dev/null
    # Hand the sweep to gpu_cleanup.sh -- the SAME file preflight ran in report
    # mode and the same array it screened on. An orphaned EngineCore is
    # 'EngineCore_DP<n>' and the workers are 'Worker_DP<n>_TP<n>_EP<n>'; neither
    # matches a 'vllm serve' pattern, each holds a CUDA context and its share of
    # the KV pool, and the symptom lands on the NEXT run.
    for n in "${ENGINE_NODES[@]}"; do
        ssh -n "$n" "bash ${SCRIPT_DIR}/gpu_cleanup.sh kill ${P_PORT} ${D_PORT} ${DP_RPC_PORT_P} ${DP_RPC_PORT_D}" \
            2>&1 | sed 's/^/  /'
    done
    echo "  Read the VERDICT lines. A node that ends DIRTY is the next run's"
    echo "  failure, and this is the last moment it can still be handed back."
    echo "  Run dir: ${SHARED}"
}
trap cleanup EXIT INT TERM

# =============================================================================
# Launch -- ONE mpiexec, restricted to the four ENGINE nodes
# =============================================================================
# --hosts, not a bare -n 4: the allocation also contains the proxy node and the
# client node, and `-ppn 1` over the whole allocation would start an engine on
# both. The client node must run no engine, which is the entire point of having
# it, and the proxy node must stay free for the proxy.
#
# --cpu-bind none IS NOT COSMETIC. PALS binds each rank to a subset of the
# node's cores by default and children inherit the mask, so all four TP workers
# plus their NCCL and NIXL progress threads would be confined to whatever slice
# rank 0 was given -- on a Grace node, possibly a single core. The symptom is
# not an error: it is an engine that takes many minutes to initialise and a
# transfer rate that looks like a fabric problem. On THIS arm that confound is
# fatal to the result, because fabric behaviour is what is being measured.
MPI_CPU_BIND=${MPI_CPU_BIND-none}
CPU_BIND_ARGS=()
[ -n "${MPI_CPU_BIND}" ] && CPU_BIND_ARGS=(--cpu-bind "${MPI_CPU_BIND}")
MPI_HOSTS="${NODE_P_HEAD},${NODE_P_TAIL},${NODE_D_HEAD},${NODE_D_TAIL}"

# mpiexec.log IS IN THIS touch DELIBERATELY. `cmd > file &` creates the file in
# the FORKED CHILD, so the parent can reach `tail -f` first and lose the race --
# tail exits, and the ONE stream that carries PALS-level failures is silently
# not being followed for the rest of the run (observed 2026-09-15).
touch "${SHARED}/logs/p-head.log" "${SHARED}/logs/p-headless.log" \
      "${SHARED}/logs/d-head.log" "${SHARED}/logs/d-headless.log" \
      "${SHARED}/logs/mpiexec.log" "${SHARED}/logs/proxy.log"
echo ""
echo "Launching: mpiexec -n ${NEED_ENGINE_NODES} -ppn 1 --hosts ${MPI_HOSTS}"
echo "  D loads first; P sleeps ${ROLE_STAGGER_SEC}s inside the role script."
mpiexec -n "${NEED_ENGINE_NODES}" -ppn 1 --hosts "${MPI_HOSTS}" \
    ${CPU_BIND_ARGS[@]+"${CPU_BIND_ARGS[@]}"} \
    bash "${SHARED}/launch_role.sh" > "${SHARED}/logs/mpiexec.log" 2>&1 &
MPIEXEC_PID=$!

# `sed -u` (unbuffered): without it, output piped through sed is block-buffered
# and arrives in silent bursts, which looks exactly like "nothing is happening".
# disown so cleanup's kills do not print job-status lines into the teardown.
tail -n +1 -f "${SHARED}/logs/p-head.log"     | sed -u 's/^/[P-HEAD] /' &
TAIL_PH_PID=$!; disown ${TAIL_PH_PID}
tail -n +1 -f "${SHARED}/logs/p-headless.log" | sed -u 's/^/[P-TAIL] /' &
TAIL_PT_PID=$!; disown ${TAIL_PT_PID}
tail -n +1 -f "${SHARED}/logs/d-head.log"     | sed -u 's/^/[D-HEAD] /' &
TAIL_DH_PID=$!; disown ${TAIL_DH_PID}
tail -n +1 -f "${SHARED}/logs/d-headless.log" | sed -u 's/^/[D-TAIL] /' &
TAIL_DT_PID=$!; disown ${TAIL_DT_PID}
tail -n +1 -f "${SHARED}/logs/mpiexec.log"    | sed -u 's/^/[MPI]    /' &
TAIL_M_PID=$!; disown ${TAIL_M_PID}

wait_healthy() {
    local ip=$1 port=$2 name=$3 path=${4:-/health}
    local t0; t0=$(date +%s)
    for i in $(seq 1 "${HEALTH_TRIES}"); do
        if curl -s -o /dev/null -w "%{http_code}" "http://${ip}:${port}${path}" 2>/dev/null | grep -q 200; then
            echo "${name} healthy after $(( $(date +%s) - t0 ))s"; return 0
        fi
        # mpiexec is the liveness source of truth. Without this the script burns
        # the full timeout on an application PALS already tore down -- a dead
        # startup and a slow one look identical from the outside.
        if ! kill -0 "${MPIEXEC_PID}" 2>/dev/null; then
            echo "mpiexec exited while waiting for ${name}." >&2
            echo "  Read ${SHARED}/logs/mpiexec.log FIRST: a PALS-level failure" >&2
            echo "  (bad VNI, node not in the allocation, launcher error) never" >&2
            echo "  reaches any role log at all." >&2
            # DUMP THE LOGS HERE RATHER THAN NAMING THEM. cleanup() kills the
            # tail pipelines within a second, and a `vllm serve` that dies on an
            # argparse error writes its reason and exits faster than the pipeline
            # flushes -- so the operator sees the banner, then nothing, then
            # teardown (2026-09-15).
            for _l in mpiexec d-head d-headless p-head p-headless; do
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
                "$(tail -n 1 "${SHARED}/logs/d-head.log" 2>/dev/null | cut -c1-100)"
        fi
        sleep 5
    done
    echo "${name} never became healthy -- check ${SHARED}/logs/" >&2
    return 1
}

# D FIRST, because D starts first. Only the heads answer HTTP; each headless
# node's liveness is implied, since a head cannot become healthy until every
# data-parallel rank in its role has registered.
wait_healthy "${D_HEAD_IP}" "${D_PORT}" "D head (${NODE_D_HEAD_SHORT})" || exit 1
wait_healthy "${P_HEAD_IP}" "${P_PORT}" "P head (${NODE_P_HEAD_SHORT})" || exit 1
kill -TERM "${TAIL_PH_PID}" "${TAIL_PT_PID}" "${TAIL_DH_PID}" "${TAIL_DT_PID}" 2>/dev/null

# =============================================================================
# Gate on the KV pool -- PER ROLE, and never across roles
# =============================================================================
# "One rank can come up GiB-heavier than its peers on a clean node, dying at KV
# sizing or silently shrinking the pool. Relaunch; do not tune." The pool is
# sized by the WORST worker; it has come up short twice on Nemotron, both times
# cured by a relaunch with nothing about model or workload changed
# (MEMORY_2026-09-12h.md). On Inkling the two DP ranks of one role came up 6.5%
# apart (16.74 vs 15.72 GiB, MEMORY_2026-09-15b.md).
#
# THE MINIMUM IS TAKEN WITHIN A ROLE, NEVER ACROSS ROLES. P and D have
# different --max-num-batched-tokens (16384 vs 2048), therefore different
# activation profiles, therefore different pools. A single number across both
# would be meaningless on at least one of them, and the colocated arm's
# EXPECT_POOL is meaningless on either.
read_pool() {
    grep -h -o 'GPU KV cache size: [0-9,]* tokens' "$@" 2>/dev/null \
        | grep -o '[0-9,]*' | tr -d ',' | grep -v '^$' | sort -n | head -1
}
gate_pool() {
    local role=$1 pool=$2 expect=$3
    if [ -z "${pool}" ]; then
        echo "  ${role}: could not read 'GPU KV cache size' from either rank's log."
        echo "     grep -h 'GPU KV cache size' ${SHARED}/logs/${role,,}*.log"
        return
    fi
    echo "  ${role}: ${pool} tokens (minimum across this role's DP ranks)."
    if [ -n "${expect}" ]; then
        local d=$(( pool - expect )) p
        p=$(( d * 100 / expect ))
        if [ "${p}" -lt -2 ] || [ "${p}" -gt 2 ]; then
            echo "  *** ${role} POOL GATE FAILED: expected ${expect}, got ${pool} (${p}%). ***" >&2
            echo "  This is the pool lottery. RELAUNCH, do not tune." >&2
        else
            echo "     gate PASSED against ${p}% of ${expect}."
        fi
    else
        echo "     No EXPECT_POOL_${role} set -- this run establishes it. Record it"
        echo "     and pass EXPECT_POOL_${role}=${pool} on every subsequent launch."
    fi
}
_pool_p=$(read_pool "${SHARED}/logs/p-head.log" "${SHARED}/logs/p-headless.log")
_pool_d=$(read_pool "${SHARED}/logs/d-head.log" "${SHARED}/logs/d-headless.log")
echo ""
echo "=== KV pool (per role -- these numbers do NOT transfer between roles) ==="
gate_pool "P" "${_pool_p}" "${EXPECT_POOL_P:-}"
gate_pool "D" "${_pool_d}" "${EXPECT_POOL_D:-}"
echo "${_pool_p:-unknown}" > "${SHARED}/kv_pool_tokens_p.txt"
echo "${_pool_d:-unknown}" > "${SHARED}/kv_pool_tokens_d.txt"

# --- Confirm the connector is actually attached, on BOTH roles ---------------
# The mirror image of the colocated arm's check. A disagg arm whose connector
# failed to attach is not a disagg arm -- it is two independent servers with a
# proxy in front, and it would produce plausible-looking numbers.
for _r in p-head d-head; do
    if ! grep -qiE 'NixlConnector|kv_transfer_config|KVConnector' "${SHARED}/logs/${_r}.log"; then
        echo "" >&2
        echo "WARNING: ${_r}.log never mentions a KV connector. This arm must have ONE." >&2
        echo "  Without it there is no KV transfer and the 'disagg' result is" >&2
        echo "  two servers behind a proxy. Do not bench this." >&2
    fi
done

# --- Record the KV cache group structure, which is what makes this arm novel --
# Nemotron moved two spec types (full attention + Mamba). Inkling moves three:
# full attention, sliding-window attention, and the sconv state, which vLLM
# manages as the KV cache of a VIRTUAL sliding-window attention layer (the vLLM
# Inkling blog states this outright).
#
# THE GROUP COUNT IS AN OPEN QUESTION AND THIS IS WHERE IT GETS ANSWERED.
# DECISIONS_2026-09-15c.md recorded ~77 groups (42 attention + 35 sconv). The
# blog says each layer carries FOUR sconv modules, which would make it
# 42 + 35x4 = 182. Do not assume either; read it off the log below and record
# which it is, because it sets how many groups NIXL has to move.
echo ""
echo "=== KV cache groups (the mixed-spec transfer this arm exists to test) ==="
grep -hiE 'kv_cache_group|KVCacheGroupSpec|SlidingWindowSpec|FullAttentionSpec|num_kv_cache_groups' \
    "${SHARED}/logs/d-head.log" 2>/dev/null | head -12 | sed 's/^/  /' \
    || echo "  (nothing matched -- grep the log by hand before the smoke run)"
echo "  Block size chosen by HMA (the clip arithmetic depends on it):"
grep -hoiE 'block_size[ =:]+[0-9]+' "${SHARED}/logs/d-head.log" 2>/dev/null \
    | head -3 | sed 's/^/    /'

# =============================================================================
# Start the proxy
# =============================================================================
# After both roles are healthy, so a proxy health failure means the proxy and
# not a server still loading.
ssh -n "${NODE_PROXY}" "bash ${SHARED}/launch_proxy.sh" > "${SHARED}/logs/proxy.log" 2>&1 &
PROXY_SSH_PID=$!
tail -n +1 -f "${SHARED}/logs/proxy.log" | sed -u 's/^/[PROXY]  /' &
TAIL_PROXY_PID=$!; disown ${TAIL_PROXY_PID}
# /healthcheck, not /health -- toy_proxy_server.py's own endpoint name.
wait_healthy "${PROXY_IP}" "${PROXY_PORT}" "Proxy (${NODE_PROXY_SHORT})" "/healthcheck" || exit 1
kill -TERM "${TAIL_PROXY_PID}" 2>/dev/null

# =============================================================================
# Start the frontend samplers -- BEFORE the bench, or the distribution is lost
# =============================================================================
# The connection distribution is decided in the first seconds of a bench and
# cannot be recovered afterwards; `ss` reports live sockets and nothing records
# what the split was. setsid+nohup so the loops survive the ssh session closing.
#
# SIX SAMPLERS, and the proxy's is the one that is new. The colocated arm runs
# a client sampler explicitly as the CONTROL for this arm's proxy node -- with
# the proxy on its own node here, that subtraction is now clean rather than
# entangled with P's head.
SAMPLE_STOP_MARKER="${SHARED}/sample_stop_${STAMP}"
start_sampler() {
    local node=$1 port=$2 out=$3 label=$4 kind=${5:-serving}
    : > "${out}"
    ssh -n "${node}" "setsid nohup bash ${SHARED}/sample_conns.sh \
        ${SAMPLE_STOP_MARKER} ${out} ${port} ${SAMPLE_INTERVAL_S} ${label} ${kind} \
        > /dev/null 2>&1 < /dev/null &"
}
start_sampler "${NODE_P_HEAD}" "${P_PORT}"     "${SHARED}/logs/conns_p-head.log"     "p-head"     serving
start_sampler "${NODE_P_TAIL}" "-"             "${SHARED}/logs/conns_p-headless.log" "p-headless" serving
start_sampler "${NODE_D_HEAD}" "${D_PORT}"     "${SHARED}/logs/conns_d-head.log"     "d-head"     serving
start_sampler "${NODE_D_TAIL}" "-"             "${SHARED}/logs/conns_d-headless.log" "d-headless" serving
start_sampler "${NODE_PROXY}"  "${PROXY_PORT}" "${SHARED}/logs/conns_proxy.log"      "proxy"      serving
: > "${SHARED}/logs/conns_client.log"
setsid nohup bash "${SHARED}/sample_conns.sh" \
    "${SAMPLE_STOP_MARKER}" "${SHARED}/logs/conns_client.log" "-" \
    "${SAMPLE_INTERVAL_S}" "client" client > /dev/null 2>&1 < /dev/null &
sleep 2
echo ""
echo "=== frontend samplers started (every ${SAMPLE_INTERVAL_S}s) ==="
echo "  p-head / d-head:  the two frontends, ${API_SERVER_COUNT} API servers each"
echo "  p-headless / d-headless: no socket -- by design"
echo "  proxy:            ${SHARED}/logs/conns_proxy.log   <- the proxy's own CPU"
echo "  client:           ${SHARED}/logs/conns_client.log  <- the bench's CPU"
echo "  CATCH THE PROXY'S CPU WITH 'pgrep -f toy_proxy_server', NOT 'pgrep -f proxy'."

# =============================================================================
# Hold everything up for bench_arm.sh -- WHICH RUNS ON THIS NODE
# =============================================================================
if [ "${KEEP_ALIVE}" = "1" ]; then
    echo ""
    echo "============================================================"
    echo "KEEP_ALIVE=1 -- 1P:1D held up. Ctrl-C here is the teardown."
    echo "============================================================"
    echo "  Bench endpoint (the PROXY):  http://${PROXY_IP}:${PROXY_PORT}"
    echo "  P direct:                    http://${P_HEAD_IP}:${P_PORT}"
    echo "  D direct:                    http://${D_HEAD_IP}:${D_PORT}"
    echo ""
    echo "  Model string for --model:    ${MODEL}"
    echo "  Run dir:                     ${SHARED}"
    echo "  max-model-len:               ${MAX_MODEL_LEN}"
    echo "  Serving GPUs:                $(( DP * TP * 2 ))  (passed as GPUS below)"
    echo ""
    echo "  From another shell ON THIS NODE (${CLIENT_HOST}) -- the client node."
    echo ""
    echo "      source ${SHARED}/common_env.sh"
    echo ""
    echo "      ARM=disagg \\"
    echo "      MODEL=${MODEL} \\"
    echo "      GPUS=$(( DP * TP * 2 )) \\"
    echo "      BASE_URL=http://${PROXY_IP}:${PROXY_PORT} \\"
    echo "      METRICS_URLS=http://${P_HEAD_IP}:${P_PORT},http://${D_HEAD_IP}:${D_PORT} \\"
    echo "      WORKLOAD=${WORKLOAD} \\"
    echo "      MAX_CONCURRENCY=64 NUM_PROMPTS=256 \\"
    echo "      RUN_DIR=${SHARED} \\"
    echo "          bash ${SCRIPT_DIR}/bench_arm.sh"
    echo ""
    echo "  MAX_CONCURRENCY=64 / NUM_PROMPTS=256 IS THE BASELINE POINT. The"
    echo "  colocated arm's c=64 measured 178.50 tok/s, 22.31 tok/s/GPU, TPOT"
    echo "  164.76 ms. c=32 is colocated's throughput MAXIMUM (186.21 tok/s),"
    echo "  but c=64 is where colocated is queue-bound -- resident set pinned"
    echo "  near 31, 47.7% of e2e in the admission queue -- so it is where"
    echo "  lifting that ceiling should show. Run c=16 and c=32 too, for the"
    echo "  curve."
    echo ""
    echo "  GPUS=$(( DP * TP * 2 )) IS REQUIRED AND FAILS QUIETLY. bench_arm.sh:78"
    echo "  defaults ARM=disagg to 8, written for Nemotron's 1P+1D at TP=4 on"
    echo "  two nodes. This arm serves $(( DP * TP * 2 )), so the default would report every"
    echo "  per-GPU number at 2x -- in the favourable direction, in the figure"
    echo "  the whole study turns on. MODEL defaults to the Nemotron repo id"
    echo "  (:32), which is an HTTP 400 per request -- instant, so it reads on"
    echo "  the progress bar as a fast run rather than as a fault."
    echo ""
    echo "  BASE_URL IS THE PROXY, METRICS_URLS ARE THE TWO HEADS. The proxy has"
    echo "  no /metrics of its own, and the counters that matter (nixl_*,"
    echo "  prompt_tokens_by_source_total) live on the engines."
    echo ""
    echo "  WHAT THE SMOKE RUN MUST ASSERT, before any number is quoted:"
    echo "    1. coherence gate THROUGH THE PROXY -- one /v1/completions call"
    echo "       that returns sensible text. The benchmark sends random tokens"
    echo "       with --ignore-eos and is structurally blind to noise."
    echo "    2. nixl_bytes_transferred_count > 0 on P. Zero means no transfer"
    echo "       happened and D prefilled locally."
    echo "    3. the transfer is ~18% of unclipped. At ISL 32768, window 512,"
    echo "       block 16: 7x2048 + 35x33 = 15,491 blocks against 42x2048 ="
    echo "       86,016. Near 86,016 means HMA clipping is not happening."
    echo "    4. all three spec types moved -- full attention, sliding window,"
    echo "       sconv. Two out of three is a silent correctness bug."
    echo "    5. num_preemptions_total = 0 on BOTH roles."
    echo "    6. generation_tokens_total moved by exactly n x OSL."
    echo "    7. frontend tax < 1% worse than colocated's (1.3% at c=32, 0.6%"
    echo "       at c=64). Worse means the proxy's own node is costing more"
    echo "       than predicted -- see PROXY_ON_P_HEAD."
    echo ""
    echo "  workload_transfer_bytes_per_rank() DOES NOT APPLY HERE. It is the"
    echo "  closed Figure 1 model for Nemotron's 40 Mamba + 8 attention layers."
    echo "  Record nixl_bytes_transferred_sum; do not check it against that."
    echo ""
    echo "  RECORD, for every bench against this launch:"
    echo "    * client node        ${CLIENT_HOST}"
    echo "    * proxy node         ${NODE_PROXY_SHORT}  (on-p-head=${PROXY_ON_P_HEAD})"
    echo "    * connection split   tail -5 ${SHARED}/logs/conns_*.log"
    echo "    * KV pools           P $(cat "${SHARED}/kv_pool_tokens_p.txt" 2>/dev/null)  D $(cat "${SHARED}/kv_pool_tokens_d.txt" 2>/dev/null)"
    echo "  Do NOT sample kv_cache_usage_perc -- stale at ASC>1, CLOSED.md."
    echo ""
    echo "  Nothing is proven until the identical config has run TWICE."
    echo ""

    _ka_stop=0
    trap '_ka_stop=1; echo ""; echo "Interrupt received -- releasing servers."' INT TERM

    # The heartbeat reports the samplers, not a log's last line. Blind-tailing
    # was actively misleading on the Nemotron rig: the last line is almost
    # always the GET /health entry this very loop produced. And at
    # --api-server-count ${API_SERVER_COUNT} there is no engine line to read at all --
    # vLLM disables the periodic stats logger above client_count 1
    # (loggers.py:1341).
    _ka_t0=$(date +%s)
    while [ "${_ka_stop}" -eq 0 ]; do
        if ! kill -0 "${MPIEXEC_PID}" 2>/dev/null; then
            echo ""
            echo "mpiexec (${MPIEXEC_PID}) exited -- all four engines are gone."
            echo "Check ${SHARED}/logs/mpiexec.log; a PALS-level failure never"
            echo "reaches any role log."
            break
        fi
        if ! ssh -n "${NODE_PROXY}" "pgrep -f '${PAT_PROXY}' >/dev/null" 2>/dev/null; then
            echo ""
            echo "PROXY IS GONE on ${NODE_PROXY_SHORT}. The engines are still up, but"
            echo "BASE_URL is dead. Check ${SHARED}/logs/proxy.log."
        fi
        sleep "${KEEP_ALIVE_POLL_S}"
        _ka_el=$(( $(date +%s) - _ka_t0 ))
        _ka_p=$(curl -s -o /dev/null -w '%{http_code}' "http://${P_HEAD_IP}:${P_PORT}/health" 2>/dev/null || echo "---")
        _ka_d=$(curl -s -o /dev/null -w '%{http_code}' "http://${D_HEAD_IP}:${D_PORT}/health" 2>/dev/null || echo "---")
        _ka_x=$(curl -s -o /dev/null -w '%{http_code}' "http://${PROXY_IP}:${PROXY_PORT}/healthcheck" 2>/dev/null || echo "---")
        _ka_s=$(grep 'n_conn=' "${SHARED}/logs/conns_d-head.log" 2>/dev/null | tail -n 1 \
                  | cut -d' ' -f2-6)
        printf '  [keep-alive %02d:%02d:%02d] P=%s D=%s proxy=%s | %s\n' \
            $(( _ka_el / 3600 )) $(( (_ka_el % 3600) / 60 )) $(( _ka_el % 60 )) \
            "${_ka_p}" "${_ka_d}" "${_ka_x}" "${_ka_s:-<sampler has produced no line yet>}"
    done

    echo ""
    echo "=== frontend summary for this launch ==="
    summarize_conn_sampler "${SHARED}/logs/conns_p-head.log"     "P head   (${NODE_P_HEAD_SHORT})"
    summarize_conn_sampler "${SHARED}/logs/conns_d-head.log"     "D head   (${NODE_D_HEAD_SHORT})"
    summarize_conn_sampler "${SHARED}/logs/conns_proxy.log"      "proxy    (${NODE_PROXY_SHORT})"
    summarize_conn_sampler "${SHARED}/logs/conns_client.log"     "client   (${CLIENT_HOST})"

    # Restore the standing disposition before falling through, so the EXIT trap
    # is the single teardown path. The standing handler RETURNS rather than
    # exits, so leaving it installed during the loop would both spin the loop
    # after the user asked to stop and run cleanup() twice.
    trap cleanup EXIT INT TERM
    echo "Releasing servers -- cleanup() follows."
fi
