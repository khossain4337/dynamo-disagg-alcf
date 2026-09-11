#!/bin/bash
set -uo pipefail

# =============================================================================
# run_colocated_N1.sh -- the COLOCATED arm of the Figure 2 comparison.
#
# One node, TP=4, one plain `vllm serve`. No proxy, no NIXL, no mpiexec.
# It brings the server up, waits for /health, and parks so that bench_arm.sh
# can drive it from outside:
#
#     ARM=colocated BASE_URL=http://<IP>:8100 RUN_DIR=<this run dir> \
#         bash bench_arm.sh
#
# WHAT THIS ARM IS FOR. Figure 2 is SLO-constrained goodput normalized PER GPU
# (CLOSED.md, 2026-09-11), so the baseline does not have to match the disagg
# arm's GPU count: one colocated TP=4 instance divided by 4 is a valid and in
# fact harder baseline than two replicas. Disagg gets 8 GPUs across two nodes,
# this gets 4 on one, and both numbers are divided by their own GPU count.
#
# THE ONE RULE. This must issue the SAME `vllm serve` command as
# run_pd_nemotron_1p1d_N2.sh, differing by exactly two things:
#
#     1. no --kv-transfer-config
#     2. no VLLM_NIXL_SIDE_CHANNEL_HOST / _PORT / NIXL_LOG_LEVEL
#
# Every other flag, env var and default is identical, and the ones that were
# NOT obvious are argued individually below (batching, conv state layout, the
# shim). If you change a serve flag here, change it there in the same commit,
# or the comparison silently starts measuring the difference in flags. The
# generated launcher is written to the run dir precisely so the two can be
# diffed after the fact:
#
#     diff <(grep -A14 'exec vllm serve' <run>/launch_colocated.sh) \
#          <(grep -A14 'exec vllm serve' <disagg-run>/launch_role.sh)
#
# ENVIRONMENT COMES FROM emit_common_env.sh, sourced, not copied. That file is
# the single copy both arms share; a second copy is how the arms quietly stop
# being comparable.
#
# USAGE
#   bash run_colocated_N1.sh                       # 32k context, holds the server
#   MAX_MODEL_LEN=147456 bash run_colocated_N1.sh  # the settled Figure 2 point
#   MAX_MODEL_LEN=43008  bash run_colocated_N1.sh  # the iter32k point
#   KEEP_ALIVE=0 bash run_colocated_N1.sh          # bring up, verify, tear down
#
# WORKLOAD=<profile> does not change --max-model-len (see the block below) but
# it does tell the guard near the end of this file WHICH workload to check the
# context budget against, so the advice it prints is the advice you need:
#
#   WORKLOAD=iter32k bash run_colocated_N1.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The workload point this server is being provisioned FOR. Read from the same
# file bench_arm.sh reads, so the context budget set here and the ISL/OSL sent
# there come from ONE definition. They are decided hours apart, on different
# nodes, by different commands; when they disagree the symptom is an HTTP 400
# per request, which is instant and therefore reads on the progress bar as a
# fast request rather than as a fault (2026-09-11).
WORKLOAD=${WORKLOAD:-pilot128k}
# shellcheck source=workload_profile.sh
source "${SCRIPT_DIR}/workload_profile.sh"
load_workload_profile "${WORKLOAD}" || exit 1

# =============================================================================
# LOCKED -- must match run_pd_nemotron_1p1d_N2.sh exactly
# =============================================================================
# Every default in this block is copied from the disagg launcher and is a
# same-commit change if either moves. The reasoning for each lives there (see
# its LOCKED section, lines ~185-250) and is not duplicated here -- duplicated
# rationale drifts the same way duplicated code does. Only the ones where the
# colocated arm had a real CHOICE are argued in this file.

MODEL=${MODEL:-nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16}
TP=${TP:-4}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-auto}
MAMBA_SSM_CACHE_DTYPE=${MAMBA_SSM_CACHE_DTYPE:-float32}
HMA_FLAG=${HMA_FLAG:---no-disable-hybrid-kv-cache-manager}
PREFIX_CACHING_FLAG=${PREFIX_CACHING_FLAG:---no-enable-prefix-caching}
BLOCK_SIZE=${BLOCK_SIZE:-}
EXPERT_PARALLEL=${EXPERT_PARALLEL:-}
ENFORCE_EAGER=${ENFORCE_EAGER:-}

# --- MAX_MODEL_LEN: defaults to the disagg script's default ON PURPOSE -------
# 32768, matching run_pd_nemotron_1p1d_N2.sh:262, so that running both arms
# with no environment at all gives two comparable servers. Each workload profile
# carries the value it needs (WL_MAX_MODEL_LEN), and that number must be passed
# to BOTH arms in the same breath:
#
#     MAX_MODEL_LEN=147456 bash run_colocated_N1.sh        # pilot128k
#     MAX_MODEL_LEN=147456 bash run_pd_nemotron_1p1d_N2.sh
#
# It is deliberately NOT defaulted to WL_MAX_MODEL_LEN. The disagg launcher is
# frozen and cannot learn about profiles, so defaulting this one would make the
# two arms differ when both are run bare -- exactly the failure this block
# exists to prevent. The profile informs the guard, it does not silently move
# the server. The guard at the end of this file prints the right number.
#
# 147456, not 139264, and the 8192 of slack is not superstition. Setting it to
# exactly ISL+OSL was tried on 2026-09-11 and one warmup request still came back
# HTTP 400: `vllm bench serve --dataset-name random` synthesises prompts that
# land NEAR --random-input-len, not on it, so an exactly-fitting budget rejects
# whichever prompts round up. The rejection is instant, so it registers on the
# progress bar as a completed request -- 1/32 in under four seconds -- and reads
# as progress. Give it a block of headroom and the ambiguity goes away.
#
# It is NOT changed as a default here, because the disagg script's
# in-file default is frozen and a default that differs between the two arms is
# worse than one that is merely inconvenient. The preflight below warns when
# this is too small for the settled workload -- at 32768 a 128k request is
# rejected with HTTP 400 before any work happens, which is handoff item 2.
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}

# =============================================================================
# THE COLOCATED BATCHING DECISION -- read this before changing the two numbers
# =============================================================================
# Disagg runs two engines tuned in opposite directions:
#
#     P (prefill)  16384 tokens / 32 seqs    compute-bound, few big batches
#     D (decode)    2048 tokens / 256 seqs   bandwidth-bound, many sequences
#
# A colocated server does both jobs and can only pick one setting per knob, so
# copying either engine wholesale is a rigged baseline. Copying D's 2048 is the
# specific failure already on record: CLOSED.md lists "the Q1a arm as a
# co-located baseline" among the numbers that must never be quoted, because a
# decode-tuned server doing a prefill measured 2.76x slower than P at the same
# work. That is a measurement of a bad flag, not of colocation.
#
# THE RULE USED HERE IS PER-KNOB MAXIMUM. For each knob independently, the
# colocated server gets whatever the disagg engine that OWNS that job gets:
#
#     max-num-batched-tokens  16384 = max(P 16384, D 2048)
#     max-num-seqs              256 = max(P 32,    D 256)
#
# So this arm is the UNION of both tuned configs, never tighter than either on
# any axis. That is the property worth having: no reviewer can point at a flag
# and say the baseline was starved. It is not a split-the-difference compromise
# and should not be described as one.
#
# WHAT PER-KNOB MAXIMUM DOES NOT BUY, and do not claim it does:
#
#   * max-num-seqs is INERT at the pilot point. bench_arm.sh drives with
#     --max-concurrency 32, so at most 32 requests are ever in flight and both
#     32 and 256 are non-binding; the KV budget (~60 sequences at 136k tokens)
#     would bind long before 256 anyway. 256 is here for the later concurrency
#     sweep, not for the pilot. The pilot's only live knob is the token budget.
#
#   * 16384 is the generous choice for TTFT and the PUNITIVE one for p95 ITL.
#     max-num-batched-tokens is a per-scheduler-step budget shared between
#     prefill chunks and decode tokens, so a bigger chunk means a longer stall
#     for every decode riding in that step -- and p95 ITL is the metric Figure 2
#     turns on. A colocated server tuned to PASS p95 ITL <= 25 ms would pick a
#     smaller chunk and pay for it in TTFT.
#
# It is kept at 16384 anyway, for a reason that is not generosity: at 16384 a
# 131072-token prompt chunks into exactly 8 pieces in BOTH arms, so prefill
# chunking is held constant and the only remaining difference is whether decode
# has to share the step. That is precisely the variable Figure 2 exists to
# isolate.
#
# THE DEBT THIS LEAVES. Because the knob cuts both ways, the pilot number is
# not yet the Figure 2 baseline. Before the comparison is quoted, run this arm
# at 4096 and 8192 as well and take the best SLO-feasible goodput -- under an
# SLO-constrained-goodput framing the baseline is legitimately the best
# colocated configuration, and measuring it is what makes the claim
# unattackable. It is cheap: one node, no proxy, no NIXL, no second allocation.
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-16384}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}

# =============================================================================
# SSM_CONV_STATE_LAYOUT -- held at DS, and that is a decision, not a copy-paste
# =============================================================================
# vLLM's default is SD, not DS: get_conv_state_layout() returns "SD" when the
# env is unset (mamba_utils.py:43). The disagg arm has no choice -- DS is a hard
# assert under NixlConnector (base_worker.py:316) because the connector splits
# each conv state into contiguous x/B/C sub-projections and needs (dim,
# state_len) ordering.
#
# So "drop the NIXL env" would have silently flipped the conv state layout in
# all 40 Mamba layers of this arm. That is not a side-channel setting; it is a
# memory layout the causal_conv1d kernels index through on every decode step,
# which lands directly on ITL -- again, the metric being compared. Let it differ
# and a colocated ITL win cannot be told apart from "SD is a faster layout".
#
# Held at DS for the pilot so the layout is constant across arms. The honest
# follow-up is a colocated-only SD-vs-DS A/B (one node, no fabric, cheap): if
# SD is materially faster, the Figure 2 baseline should be quoted at SD and the
# delta reported, since a real colocated deployment would never set DS.
SSM_CONV_STATE_LAYOUT=${SSM_CONV_STATE_LAYOUT:-DS}

# --- KEEP_ALIVE: defaults to 1, same as the disagg script --------------------
# Holding a server up for bench_arm.sh is the ONLY thing this script does, so 0
# would make the default invocation useless. The disagg launcher defaulted to 0
# until 2026-09-11 -- it has a bring-up evidence chain and holding the pair up
# was framed as the exception -- but the asymmetry between the two launchers
# cost an allocation, so both now default to 1. KEEP_ALIVE=0 is still supported
# and does the sensible thing: bring up, confirm healthy, tear down.
KEEP_ALIVE=${KEEP_ALIVE:-1}
KEEP_ALIVE_POLL_S=${KEEP_ALIVE_POLL_S:-60}
HEALTH_TRIES=${HEALTH_TRIES:-360}   # x5s = 30 min; a 240 GB cold load is slow

PORT=${PORT:-8100}
RUNS_ROOT=${RUNS_ROOT:-/vast/draco/tara/projects/Tara_Deployment/software/testing/RUNS}
STAMP=$(date +%Y%m%d_%H%M%S)
SHARED=${SHARED:-${RUNS_ROOT}/colocated_tp${TP}_${STAMP}}

# =============================================================================
# Resolve the node. No ssh, no PBS_NODEFILE parsing, no mpiexec.
# =============================================================================
# Everything runs on the node this script is running on. That is the entire
# structural difference from the disagg launcher, and it is why none of the
# two-node machinery (VNI sharing, hostname-matching role scripts, remote
# pkill) appears below.
#
# The hsn0 IP rather than 127.0.0.1: the server binds 0.0.0.0 exactly as the
# disagg engines do, and bench_arm.sh may be driven from another node. A
# baseline reachable only from localhost would quietly force the bench client
# onto this node and change what is being measured.
THIS_HOST=$(hostname -s); THIS_HOST="${THIS_HOST%%.*}"
IP=$(ip -4 -o addr show hsn0 2>/dev/null | awk '{print $4}' | cut -d/ -f1)
if [ -z "${IP}" ]; then
    echo "Could not resolve an hsn0 address on ${THIS_HOST}." >&2
    echo "  Are you on a compute node? This script does not run on a login node." >&2
    exit 1
fi
echo "Colocated node: ${THIS_HOST}   hsn0: ${IP}   port: ${PORT}"

if [ -n "${PBS_NODEFILE:-}" ] && [ -r "${PBS_NODEFILE}" ]; then
    _n_alloc=$(sort -u "${PBS_NODEFILE}" | grep -c . )
    if [ "${_n_alloc}" -gt 1 ]; then
        echo "NOTE: allocation has ${_n_alloc} nodes; this arm uses 1 (${THIS_HOST})."
        echo "      That is correct -- Figure 2 normalizes per GPU, so the baseline"
        echo "      does not match the disagg arm's GPU count. The others idle."
    fi
fi

# PALS binds each rank to a subset of cores and children inherit the mask, which
# is why the disagg launcher passes --cpu-bind none. Launched without mpiexec
# there is nothing to bind, but if this is ever wrapped in an mpiexec the TP
# workers would be confined to a fraction of the node and the baseline would
# lose for a reason that has nothing to do with colocation. nproc honours the
# affinity mask; nproc --all does not.
_cpus_avail=$(nproc 2>/dev/null || echo 0)
_cpus_total=$(nproc --all 2>/dev/null || echo 0)
if [ "${_cpus_avail}" -gt 0 ] && [ "${_cpus_avail}" -lt "${_cpus_total}" ]; then
    echo "WARNING: affinity mask allows ${_cpus_avail} of ${_cpus_total} CPUs." >&2
    echo "         All ${TP} TP workers will inherit it. If this is under mpiexec," >&2
    echo "         add --cpu-bind none; the disagg arm already does." >&2
fi

# =============================================================================
# Preflight -- fail before a 240 GB weight load, not during it
# =============================================================================
# On the deferred backlog for the disagg script and cheap enough to just do
# here. The scenario is concrete: this arm gets run on a node that just hosted
# a disagg engine, an orphaned VLLM::EngineCor is still holding GPU memory, and
# the failure surfaces minutes later as an OOM or a wedged startup.
#
# Bracketed patterns ('[v]llm serve') for the same reason the disagg script
# brackets them -- a pattern that matches its own killer is a self-inflicted
# outage. See run_pd_nemotron_1p1d_N2.sh:849-866 for the full anatomy.
PAT_VLLM='[v]llm serve'
PAT_ENGINE='[V]LLM::EngineCor'
PAT_WORKER='[W]orker_TP'

_preflight_fail=0
for _pat in "${PAT_VLLM}" "${PAT_ENGINE}" "${PAT_WORKER}"; do
    if pgrep -f "${_pat}" >/dev/null 2>&1; then
        echo "PREFLIGHT: surviving process matching ${_pat}:" >&2
        pgrep -af "${_pat}" | sed 's/^/    /' >&2
        _preflight_fail=1
    fi
done
if fuser "${PORT}/tcp" >/dev/null 2>&1; then
    echo "PREFLIGHT: port ${PORT} is already bound." >&2
    _preflight_fail=1
fi
# GPU MEMORY IS ADVISORY, NOT A GATE. This bar used to be 1024 MiB and it
# refused every node on the system.
#
# MEASURED 2026-09-11 across several fresh PBS allocations, including two
# separate allocations of x4820c7s6b1n0 hours apart:
#     GPU 0: 1026-1027 MiB    GPU 1: 1 MiB
#     GPU 2: 2049      MiB    GPU 3: 2049-2050 MiB
# Same pattern, different jobs, different nodes. It is a system baseline, not
# our leak -- root-owned daemons (DCGM's nv-hostengine, fabric manager, the
# IMEX daemon) hold it permanently. `nvidia-smi` prints "No running processes
# found" underneath because a non-root user cannot see root's pids in its
# process table; `ps -eo pid,user,comm | grep -i dcgm` can.
#
# WHY THIS MATTERS BEYOND THE GATE. The baseline is UNEVEN across GPUs, and it
# is what makes the per-rank KV pools differ: on 2026-09-11, TP0/TP1 came up
# with 23.43 GiB of KV and TP2/TP3 with 21.43 GiB, and the hybrid allocator
# sizes the pool to the MINIMUM rank. That is a real tax, but it is the same
# tax on every node, so both Figure 2 arms pay it equally and the comparison
# survives. What would NOT survive is one arm running on a node that also has
# a few GB of somebody's orphaned process on top -- hence the loud print. The
# per-GPU numbers go into the launch log so any two runs can be compared after
# the fact instead of argued about.
#
# The gate that actually protects a run is the pgrep/fuser block above: our own
# leftover processes take the port and contend for the GPU. Raw MiB does not.
GPU_DIRTY_MIB=${GPU_DIRTY_MIB:-4096}
_gpu_note=""
while read -r _idx _used; do
    _gpu_note="${_gpu_note}    GPU ${_idx}: ${_used} MiB\n"
    if [ "${_used:-0}" -gt "${GPU_DIRTY_MIB}" ]; then
        echo "PREFLIGHT: GPU ${_idx} holds ${_used} MiB, over the ${GPU_DIRTY_MIB} MiB bar." >&2
        _preflight_fail=1
    fi
done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ',')
echo "PREFLIGHT: GPU memory at launch (~1-2 GiB per GPU is this system's"
echo "           permanent baseline -- see the comment above, not a problem):"
printf '%b' "${_gpu_note}"
if [ "${_preflight_fail}" -ne 0 ]; then
    echo "" >&2
    echo "Refusing to start on a dirty node -- clear the above and re-run." >&2
    echo "  pkill -KILL -f 'vllm serve'; pkill -KILL -f 'VLLM::EngineCor'" >&2
    echo "  pkill -KILL -f 'Worker_TP';  fuser -k ${PORT}/tcp" >&2
    echo "" >&2
    echo "If pgrep and fuser found nothing and it is only the GPU MiB line, the" >&2
    echo "memory belongs to a process you cannot see or kill. A fresh allocation" >&2
    echo "is the only lever without root. If every allocation looks like this," >&2
    echo "the baseline has moved: confirm with ps, then raise the bar --" >&2
    echo "  GPU_DIRTY_MIB=8192 bash ${0##*/}" >&2
    echo "-- and record the new measured baseline in the comment above." >&2
    exit 1
fi

# =============================================================================
# Locate the shim and the shared env script, IN PLACE in the repo
# =============================================================================
for f in fi_getinfo_shim.so env_for_libfabric_topology_error.sh emit_common_env.sh; do
    if [ ! -f "${SCRIPT_DIR}/${f}" ]; then
        echo "Missing ${SCRIPT_DIR}/${f} -- cannot continue." >&2
        [ "${f}" = "fi_getinfo_shim.so" ] && echo "  Build it: gcc -shared -fPIC -o fi_getinfo_shim.so fi_getinfo_shim.c -ldl \$(pkg-config --cflags libfabric)" >&2
        exit 1
    fi
done
SHIM="${SCRIPT_DIR}/fi_getinfo_shim.so"
ENV_SCRIPT="${SCRIPT_DIR}/env_for_libfabric_topology_error.sh"

# fi_getinfo_shim.so is a TRACKED BINARY that goes stale silently against its
# own .c. Not fatal for this arm -- nothing here should reach libfabric at all
# -- but a stale shim in the LD_PRELOAD of one arm and a fresh one in the other
# is exactly the asymmetry this file exists to prevent.
_shim_patches=$(strings "${SHIM}" 2>/dev/null | grep -c 'PATCH [234]')
if [ "${_shim_patches}" -lt 1 ]; then
    echo "WARNING: ${SHIM} has no PATCH [234] strings -- it looks stale." >&2
    echo "         Rebuild it before trusting a cross-arm comparison." >&2
fi

mkdir -p "${SHARED}/logs" || { echo "Cannot create ${SHARED}" >&2; exit 1; }
LOG="${SHARED}/logs/colocated.log"

# =============================================================================
# Generate common_env.sh via the SHARED emitter
# =============================================================================
# emit_common_env() reads these four out of this scope. They are the only
# arm-specific inputs the shared environment takes.
#
#   UCX_LINES    empty. The disagg arm only sets UCX_* when NIXL_BACKEND=UCX,
#                and the default there is LIBFABRIC -- so empty is what the
#                default disagg run also emits. Not "colocated needs no UCX";
#                matching the default disagg run is the reason.
#   GPU_PIN_LINE empty. TP=4 wants all four GPUs visible; the disagg script
#                only pins when it is running a single-GPU role.
NO_PROXY_LIST="localhost,127.0.0.1,${IP},${THIS_HOST}"
UCX_LINES=""
GPU_PIN_LINE=""
# shellcheck source=./emit_common_env.sh
source "${SCRIPT_DIR}/emit_common_env.sh"
emit_common_env "${SHARED}/common_env.sh"
echo "Wrote ${SHARED}/common_env.sh (from the shared emitter, not a copy)."

# =============================================================================
# Generate the launcher
# =============================================================================
# Written to a file rather than exec'd inline for two reasons: it records the
# exact serve command next to the run it produced, and it gives the cross-arm
# diff at the top of this file something to diff against.
#
# Unquoted heredoc, so the values resolve now; \${...} escapes are the handful
# that must survive to runtime. Same convention as launch_role.sh -- an
# unescaped $ here bakes a launcher-time value into a runtime file.
cat > "${SHARED}/launch_colocated.sh" <<EOF
#!/bin/bash
source ${SHARED}/common_env.sh

# THE TWO NIXL ENV VARS THE DISAGG ARM SETS ARE DELIBERATELY ABSENT HERE:
#     VLLM_NIXL_SIDE_CHANNEL_HOST / _PORT   (no connector, no side channel)
#     NIXL_LOG_LEVEL                        (nothing to log)
# Together with the missing --kv-transfer-config below, that is the complete
# list of differences from launch_role.sh. Anything else that differs is a bug.

# KEPT, and it is a decision. The shim is a no-op unless something calls
# fi_getinfo, and this arm should never reach libfabric. But intra-node NCCL at
# TP=4 MAY probe it, and if it does, we want it behaving exactly as it does
# inside P and D -- an LD_PRELOAD present in one arm and absent in the other is
# a difference in the thing being compared. Cheap to hold constant, so hold it.
export LD_PRELOAD=${SHIM}

# NOT a NIXL side-channel variable, despite arriving with them. vLLM's default
# is SD (mamba_utils.py:43); the disagg arm is forced to DS by the assert at
# base_worker.py:316. Held at DS here so the conv state layout is identical
# across arms -- see the argument in run_colocated_N1.sh.
export VLLM_SSM_CONV_STATE_LAYOUT=${SSM_CONV_STATE_LAYOUT}

# Same array idiom as launch_role.sh: an unset knob must contribute NOTHING to
# the command line, because an empty string would be parsed as a positional
# argument and vllm serve would reject it.
EXTRA=()
[ -n "${BLOCK_SIZE}" ]      && EXTRA+=(--block-size ${BLOCK_SIZE})
[ -n "${EXPERT_PARALLEL}" ] && EXTRA+=(--enable-expert-parallel)
[ -n "${ENFORCE_EAGER}" ]   && EXTRA+=(--enforce-eager)

exec vllm serve ${MODEL} --host 0.0.0.0 --port ${PORT} \\
    --tensor-parallel-size ${TP} \\
    --max-model-len ${MAX_MODEL_LEN} \\
    --gpu-memory-utilization ${GPU_MEM_UTIL} \\
    --dtype bfloat16 \\
    --kv-cache-dtype ${KV_CACHE_DTYPE} \\
    --mamba-ssm-cache-dtype ${MAMBA_SSM_CACHE_DTYPE} \\
    ${HMA_FLAG} \\
    ${PREFIX_CACHING_FLAG} \\
    --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} \\
    --max-num-seqs ${MAX_NUM_SEQS} \\
    \${EXTRA[@]+"\${EXTRA[@]}"}
EOF
chmod +x "${SHARED}/launch_colocated.sh"

# --- Provenance, recorded rather than hoarded --------------------------------
{
    echo "stamp            ${STAMP}"
    echo "host             ${THIS_HOST}  hsn0 ${IP}  port ${PORT}"
    echo "model            ${MODEL}"
    echo "tp               ${TP}   gpu-memory-utilization ${GPU_MEM_UTIL}"
    echo "max-model-len    ${MAX_MODEL_LEN}"
    echo "provisioned for  ${WORKLOAD}  (ISL ${WL_ISL} / OSL ${WL_OSL}, needs ${WL_MAX_MODEL_LEN})"
    echo "batching         ${MAX_NUM_BATCHED_TOKENS} tok / ${MAX_NUM_SEQS} seq  (per-knob max of P 16384/32 and D 2048/256)"
    echo "conv layout      ${SSM_CONV_STATE_LAYOUT}   kv ${KV_CACHE_DTYPE}   ssm ${MAMBA_SSM_CACHE_DTYPE}"
    echo "shim             ${SHIM}  sha256 $(sha256sum "${SHIM}" 2>/dev/null | awk '{print $1}')"
    echo "git              $(git -C "${SCRIPT_DIR}" rev-parse --short HEAD 2>/dev/null || echo '(not a work tree)')"
} > "${SHARED}/run_config.txt"
cat "${SHARED}/run_config.txt"

# --- The workload guard that handoff item 2 exists to fix --------------------
# The bar is the selected profile's WL_MAX_MODEL_LEN: ISL + OSL plus 8192 of
# slack, and the guard wants at least that. Slack, not an exact fit, because
# `vllm bench serve --dataset-name random` synthesises prompts NEAR
# --random-input-len rather than on it, so a budget of exactly ISL+OSL rejects
# whichever prompts round up -- tried on 2026-09-11, one warmup request still
# came back HTTP 400. Below the bar requests are rejected before a single token
# is processed, instantly, which is why it reads as a fast request rather than
# as a harness fault.
#
# Raising --max-model-len is close to free: it sizes the per-request position
# budget, not the KV pool, which comes from --gpu-memory-utilization. The one
# thing it does change is cosmetic and easy to misread -- vLLM's startup line
# "Maximum concurrency for N tokens per request" divides the pool by
# max-model-len, not by the ISL you will actually send.
if [ "${MAX_MODEL_LEN}" -lt "${WL_MAX_MODEL_LEN}" ]; then
    echo ""
    echo "NOTE: max-model-len ${MAX_MODEL_LEN} < ${WL_MAX_MODEL_LEN}, so the ${WORKLOAD}"
    echo "      workload (ISL ${WL_ISL} / OSL ${WL_OSL}, plus dataset jitter) risks"
    echo "      being REJECTED with HTTP 400."
    echo "      Fine for a smoke run. To bench it, pass it to BOTH arms:"
    echo "          MAX_MODEL_LEN=${WL_MAX_MODEL_LEN} bash run_colocated_N1.sh"
    echo "          MAX_MODEL_LEN=${WL_MAX_MODEL_LEN} bash run_pd_nemotron_1p1d_N2.sh"
    echo "      and confirm the model really serves ${WL_MAX_MODEL_LEN} positions --"
    echo "      config.json declaring 262144 is a claim, not a measurement."
fi

# =============================================================================
# Teardown
# =============================================================================
# Local only -- no ssh, one node. Same three patterns as the disagg cleanup and
# for the same reason: an orphaned EngineCore shows up in `top` as
# 'VLLM::EngineCor' and matches no 'vllm serve' pattern, so killing the server
# by name alone leaves it holding GPU memory and the port.
cleanup() {
    echo "=== Cleaning up ==="
    kill -TERM "${SERVE_PID:-}" 2>/dev/null
    pkill -TERM -f "${PAT_VLLM}" 2>/dev/null
    sleep 5
    pkill -KILL -f "${PAT_ENGINE}" 2>/dev/null
    pkill -KILL -f "${PAT_WORKER}" 2>/dev/null
    pkill -KILL -f "${PAT_VLLM}" 2>/dev/null
    fuser -k "${PORT}/tcp" 2>/dev/null
    # Free GPU memory is what actually has to be true before the next run;
    # process names are only a proxy for it. Report it directly.
    echo "  GPU memory still in use:"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null \
        | sed 's/^/    /' || echo "    (nvidia-smi unavailable)"
    echo "  Log: ${LOG}"
}
trap cleanup EXIT INT TERM

# =============================================================================
# Launch
# =============================================================================
echo ""
echo "Launching: ${SHARED}/launch_colocated.sh"
echo "  log -> ${LOG}"
bash "${SHARED}/launch_colocated.sh" > "${LOG}" 2>&1 &
SERVE_PID=$!
tail -f "${LOG}" &
TAIL_PID=$!

wait_healthy() {
    local ip=$1 port=$2 name=$3
    local t0; t0=$(date +%s)
    for i in $(seq 1 "${HEALTH_TRIES}"); do
        if curl -s -o /dev/null -w "%{http_code}" "http://${ip}:${port}/health" 2>/dev/null | grep -q 200; then
            echo "${name} healthy after $(( $(date +%s) - t0 ))s"; return 0
        fi
        # The server process is the liveness source of truth. Without this the
        # script burns the full 30-minute timeout on a server that died in the
        # first ten seconds -- a dead startup and a slow one look identical from
        # the outside, which is the whole complaint the heartbeat below fixes.
        if ! kill -0 "${SERVE_PID}" 2>/dev/null; then
            echo "vllm serve exited while waiting for ${name}." >&2
            echo "  Last lines of ${LOG}:" >&2
            tail -n 30 "${LOG}" | sed 's/^/    /' >&2
            return 1
        fi
        # Heartbeat every 60s so a slow 240 GB load is visibly progressing.
        if [ $(( i % 12 )) -eq 0 ]; then
            printf '  [startup %4ds] %s\n' "$(( $(date +%s) - t0 ))" \
                "$(tail -n 1 "${LOG}" 2>/dev/null | cut -c1-100)"
        fi
        sleep 5
    done
    echo "${name} never became healthy -- check ${LOG}" >&2
    return 1
}

wait_healthy "${IP}" "${PORT}" "Colocated (${THIS_HOST})" || exit 1
kill -TERM "${TAIL_PID}" 2>/dev/null   # stop mirroring; the banner follows

# --- Confirm the arm is actually free of the connector -----------------------
# Cheap, and it catches the one mistake this whole file is built to prevent: a
# colocated arm that somehow still has a KV connector attached is not a
# baseline, it is a second disagg engine with a confusing name.
if grep -qiE 'NixlConnector|kv_transfer_config|KVConnector' "${LOG}"; then
    echo "" >&2
    echo "WARNING: the log mentions a KV connector. This arm must have NONE." >&2
    grep -inE 'NixlConnector|kv_transfer_config|KVConnector' "${LOG}" | head -5 | sed 's/^/    /' >&2
fi

# =============================================================================
# Hold the server up for bench_arm.sh
# =============================================================================
# The trap swap is the same shape as the disagg script's, and for the same
# reason: the standing `trap cleanup EXIT INT TERM` RETURNS rather than exits,
# so during a blocking loop it would both spin the loop after the user asked to
# stop and run cleanup() a second time via EXIT. Install a flag-raiser for the
# hold, let the loop notice, restore the standing trap before falling through.
# Do not "simplify" this back to a single trap.
if [ "${KEEP_ALIVE}" = "1" ]; then
    echo ""
    echo "============================================================"
    echo "KEEP_ALIVE=1 -- server held up. Ctrl-C here is the teardown."
    echo "============================================================"
    echo "  Colocated arm -- bench this endpoint:"
    echo "      http://${IP}:${PORT}"
    echo ""
    echo "  Model string for --model:  ${MODEL}"
    echo "  Run dir:                   ${SHARED}"
    echo "  max-model-len:             ${MAX_MODEL_LEN}"
    echo ""
    echo "  From another shell, ON THIS NODE (${THIS_HOST}):"
    echo ""
    echo "      source ${SHARED}/common_env.sh"
    echo ""
    echo "      ARM=colocated \\"
    echo "      BASE_URL=http://${IP}:${PORT} \\"
    echo "      RUN_DIR=${SHARED} \\"
    echo "          bash ${SCRIPT_DIR}/bench_arm.sh"
    echo ""
    echo "  The source line is NOT optional, and skipping it fails in three"
    echo "  ways that all look like a broken server rather than a bare shell:"
    echo "    * no conda env      -> 'vllm: command not found'"
    echo "    * no HF_HOME/TOKEN  -> --dataset-name random cannot load the"
    echo "                           tokenizer and reaches for the network"
    echo "    * no no_proxy       -> curl to ${IP} goes to"
    echo "                           proxy.alcf.anl.gov and the preflight"
    echo "                           one-token completion fails as if nothing"
    echo "                           were listening"
    echo "  It is the same file this server sourced, so client and server"
    echo "  cannot disagree. It leaves nounset on; 'set +u' afterwards if that"
    echo "  bothers an interactive shell."
    echo ""
    echo "  Divide this arm's throughput by ${TP} GPUs before comparing it to"
    echo "  the disagg arm's, which spans 8. Read p95/p99 ITL, not the mean."
    echo ""

    _ka_stop=0
    trap '_ka_stop=1; echo ""; echo "Interrupt received -- releasing server."' INT TERM

    # The heartbeat reports the ENGINE, not the log's last line.
    #
    # Blind-tailing the log was actively misleading: the last line is almost
    # always the `GET /health` access log entry that this very loop just
    # produced, so the heartbeat reported itself, at 200, once a minute, while
    # the engine behind it had not stepped in twelve minutes (2026-09-11).
    #
    # loggers.py:310 is vLLM's periodic stats line. It prints roughly every 10 s
    # for as long as the engine is stepping and stops the instant it is not, so
    # "unchanged since the last poll" is a direct read on whether work is
    # moving -- which /health cannot give, because the API server is a separate
    # process from EngineCore and answers 200 long after the engine has stopped.
    _ka_last=""
    _ka_stall=0
    _ka_t0=$(date +%s)
    while [ "${_ka_stop}" -eq 0 ]; do
        if ! kill -0 "${SERVE_PID}" 2>/dev/null; then
            echo ""
            echo "vllm serve (${SERVE_PID}) exited -- the endpoint is gone."
            echo "Check ${LOG}."
            break
        fi
        sleep "${KEEP_ALIVE_POLL_S}"
        _ka_el=$(( $(date +%s) - _ka_t0 ))
        _ka_h=$(curl -s -o /dev/null -w '%{http_code}' "http://${IP}:${PORT}/health" 2>/dev/null || echo "---")
        _ka_eng=$(grep -F 'loggers.py:310' "${LOG}" 2>/dev/null | tail -n 1 \
                    | sed 's/.*Engine 000: //' | cut -c1-110)
        printf '  [keep-alive %02d:%02d:%02d] health=%s | %s\n' \
            $(( _ka_el / 3600 )) $(( (_ka_el % 3600) / 60 )) $(( _ka_el % 60 )) \
            "${_ka_h}" \
            "${_ka_eng:-<idle -- no requests yet; the server is up, health says so>}"

        # An engine with nothing to do also stops logging, and that is NOT a
        # stall -- it is the normal state between benches, which is most of what
        # a held-up server does. Only an unchanged line that still claims
        # in-flight work is evidence of a wedge.
        case "${_ka_eng}" in
            ""|*"Running: 0 reqs, Waiting: 0 reqs"*) _ka_busy=0 ;;
            *)                                       _ka_busy=1 ;;
        esac
        if [ "${_ka_busy}" -eq 1 ] && [ "${_ka_eng}" = "${_ka_last}" ]; then
            _ka_stall=$(( _ka_stall + 1 ))
        else
            _ka_stall=0
        fi
        _ka_last=${_ka_eng}

        # Two identical polls is >=2*KEEP_ALIVE_POLL_S with no engine step, against
        # a line that normally advances every 10 s. That is not slowness.
        if [ "${_ka_stall}" -ge 2 ]; then
            echo "      ^^ STALLED: no engine step in $(( _ka_stall * KEEP_ALIVE_POLL_S ))s."
            echo "         health=200 proves nothing here -- APIServer is a separate pid."
            echo "         Capture the stacks BEFORE Ctrl-C, or the cause dies with it:"
            echo "             pgrep -a -f 'EngineCore|VLLM::Worker_TP'"
            echo "             py-spy dump --pid <EngineCore pid>"
            echo "             py-spy dump --pid <Worker_TP0 pid>"
            echo "         Then nvidia-smi, then Ctrl-C."
        fi
    done

    trap cleanup EXIT INT TERM
    echo "Releasing server -- cleanup() follows."
fi
