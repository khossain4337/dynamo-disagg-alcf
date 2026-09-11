#!/bin/bash
set -uo pipefail

# =============================================================================
# bench_arm.sh -- run ONE benchmark arm and emit a normalized result.
#
# Called identically by both arms of the Figure 2 comparison:
#
#   disagg     BASE_URL=http://<P_IP>:8000   (the proxy)
#              METRICS_URLS=http://<P_IP>:8100,http://<D_IP>:8200
#   colocated  BASE_URL=http://<IP>:8100
#              METRICS_URLS=http://<IP>:8100
#   control    BASE_URL=http://<D_IP>:8200   (D direct -- the proxy-ceiling test)
#              METRICS_URLS=http://<D_IP>:8200
#
# The whole point of this file existing is that the measurement is written ONCE.
# A bench harness written separately for each arm is a harness that diverges,
# and every divergence lands as a difference between arms that looks like a
# result. If you need arm-specific behaviour, add a flag here -- do not fork it.
#
# BASE_URL and METRICS_URLS are separate because the disagg arm is driven
# through the proxy, and the proxy has no /metrics of its own. The counters that
# matter (nixl_*, prompt_tokens_by_source_total) live on P and D, so they have to
# be scraped from the engines directly while traffic goes through the front door.
# =============================================================================

ARM=${ARM:?set ARM to a label, e.g. disagg|colocated|d-direct}
BASE_URL=${BASE_URL:?set BASE_URL to the endpoint under test}
METRICS_URLS=${METRICS_URLS:-${BASE_URL}}
MODEL=${MODEL:-nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16}

# --- Locked workload. Do not vary these between arms. -------------------------
# (128k, 8k) was chosen because prefill pressure per output token is ISL/OSL:
# at ratio 16 the decode stream sees 16x the prefill interference ShareGPT's
# ~1.7 produces, while OSL stays long enough that ITL is what the user actually
# experiences. See MEMORY_2026-09-10.md.
ISL=${ISL:-131072}
OSL=${OSL:-8192}
NUM_PROMPTS=${NUM_PROMPTS:-128}
# Client-side offered load, and the knob that actually controls concurrency.
# --max-num-seqs is a server-side CAP, but it is NOT the binding one: KV is.
#
# MEASURED on the colocated arm 2026-09-11 (run colocated_tp4_20260911_052433),
# TP=4, gpu-memory-utilization 0.90, MAX_MODEL_LEN=139264:
#     Available KV cache memory:  23.43 GiB per rank   (NOT the ~48 GB the
#                                 launcher comments estimate -- weights plus
#                                 non-torch came to 58.7-60.7 GiB, peak
#                                 activation 3.35 GiB, cudagraph pool 0.89 GiB)
#     GPU KV cache size:          5,222,400 tokens
#     Maximum concurrency for 139,264 tokens per request:  37.50x
#
# So ~37 is the hard ceiling at this ISL on 4 GPUs, and it applies to the disagg
# arm too -- D holds the same sequences' KV on the same 4 GPUs. Any Figure 2
# concurrency sweep has to live below it in BOTH arms.
#
# Leaving this unset makes p95 TTFT a measurement of the queue, not of prefill.
MAX_CONCURRENCY=${MAX_CONCURRENCY:-32}

# --- Warmup ------------------------------------------------------------------
# jit_monitor warns that Triton compiles _causal_conv1d_fwd_kernel and
# fused_moe_kernel DURING inference. Those compiles land in the first requests
# of any sweep and go straight into the p99 that this comparison turns on.
# vllm bench serve has no warmup-discard of its own, so it is a separate,
# thrown-away run. Set WARMUP_PROMPTS=0 only to measure the compile itself.
WARMUP_PROMPTS=${WARMUP_PROMPTS:-${MAX_CONCURRENCY}}
#
# The warmup runs at the FULL ISL but a SHORT OSL, and the asymmetry is
# deliberate. Kernel selection is a function of shape, and the shapes that
# matter are the prefill chunk (set by ISL and --max-num-batched-tokens) and the
# steady-state decode batch (set by concurrency). Neither depends on how many
# tokens each request goes on to emit. Warming at OSL=8192 therefore compiles
# nothing that OSL=128 does not, and costs 64x the decode steps to do it.
#
# 2026-09-11: warming at the full 8192 was not merely wasteful, it never
# finished. At concurrency 32 the observed aggregate decode rate was ~34 tok/s,
# so 8192 tokens x 32 sequences is ~2 hours of warmup before the measured run
# starts. The run was killed at 16 minutes still inside warmup.
#
# 128 is enough to (a) drain every prefill, (b) reach a pure-decode batch, and
# (c) let requests actually COMPLETE and release their KV -- which is itself a
# test the 8192 warmup never got far enough to perform.
WARMUP_OSL=${WARMUP_OSL:-128}

# --- Where results land -------------------------------------------------------
# NOT the current directory. This script is run from the checkout, and results
# written relative to $PWD land in the repo -- untracked clutter at best, and
# a 128k-context sweep's result.json committed by accident at worst. Measurement
# output belongs on /vast next to the run that produced it, on a filesystem both
# nodes mount.
#
# RUN_DIR is the launcher's ${SHARED} -- the KEEP_ALIVE banner prints it as
# "Run dir:". Set it and the bench is filed with the servers it measured,
# alongside p.log, d.log and provenance.txt, which is what makes a result
# reconstructable six weeks later. Without it, benches collect under
# ${RUNS_ROOT}/bench/ and you have to match them up by timestamp.
RUNS_ROOT=${RUNS_ROOT:-/vast/draco/tara/projects/Tara_Deployment/software/testing/RUNS}
if [ -n "${RUN_DIR:-}" ]; then
    OUT_DIR=${OUT_DIR:-${RUN_DIR}/bench/$(date +%Y%m%d_%H%M%S)_${ARM}}
else
    OUT_DIR=${OUT_DIR:-${RUNS_ROOT}/bench/$(date +%Y%m%d_%H%M%S)_${ARM}}
fi

# Refuse to write into a git work tree even if OUT_DIR was set explicitly.
_out_parent=$(dirname "${OUT_DIR}")
mkdir -p "${_out_parent}" || { echo "FATAL: cannot create ${_out_parent}" >&2; exit 1; }
_out_parent=$(cd "${_out_parent}" && pwd)
if git -C "${_out_parent}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "FATAL: ${OUT_DIR} is inside a git work tree (${_out_parent})." >&2
    echo "  Bench output does not belong in the repo. Set RUN_DIR to the" >&2
    echo "  launcher's run directory, or OUT_DIR to somewhere under /vast." >&2
    exit 1
fi
mkdir -p "${OUT_DIR}"
OUT_DIR=$(cd "${OUT_DIR}" && pwd)

echo "=== bench_arm: ${ARM} ==="
echo "  endpoint    ${BASE_URL}"
echo "  metrics     ${METRICS_URLS}"
echo "  workload    ISL=${ISL} OSL=${OSL} n=${NUM_PROMPTS} concurrency=${MAX_CONCURRENCY}"
echo "  warmup      ${WARMUP_PROMPTS} prompts at OSL=${WARMUP_OSL} (discarded)"
echo "  out         ${OUT_DIR}"
echo ""

# --- Preflight ----------------------------------------------------------------
# Fail here rather than 40 minutes in. A dead endpoint and a slow one look the
# same to a bench client that has already started its clock.
#
# The probe is a real one-token completion, NOT /health or /v1/models. The
# disagg arm is driven through toy_proxy_server.py, which implements
# /v1/completions and /v1/chat/completions and nothing else -- probing for a
# sidecar endpoint failed the disagg arm while the servers were perfectly
# healthy. A completion is also the stronger test: on the proxy it forces a
# full P prefill -> NIXL pull -> D decode round trip, so a fabric fault surfaces
# here instead of as a mysterious first-request outlier in the p99.
#
# This runs BEFORE snapshot_metrics, so the probe's own tokens are outside the
# measured diff.
# Preflight 0 runs before the probe, because the probe's failure mode when the
# proxy is in the way is a 40-line Squid HTML page that says nothing useful.
#
# BOTH of 2026-09-11's operator errors land here, and they compound: the run was
# benched with the PREVIOUS run's BASE_URL (a different node, after a node swap)
# from a shell that had conda active but had never sourced common_env.sh. Either
# alone produces a proxied request; together they produce a Squid error page
# that reads as "the server is broken".
#
# This is also why a stale IP cannot survive even in a correctly-sourced shell:
# no_proxy is generated per run from THAT run's node, so last run's address is
# not in it. The check turns a confusing HTML dump into the actual sentence.
_bu_host=${BASE_URL#*://}; _bu_host=${_bu_host%%/*}; _bu_host=${_bu_host%%:*}
case "${no_proxy:-}" in
    '*') : ;;                                   # curl's blanket bypass
    *)
        case ",${no_proxy:-}," in
            *",${_bu_host},"*) : ;;
            *)
                echo "FATAL: ${_bu_host} is not in \$no_proxy." >&2
                echo "" >&2
                echo "  Every request -- this script's curl probe and vllm bench" >&2
                echo "  serve's own HTTP client alike -- goes to" >&2
                echo "  ${http_proxy:-the site proxy}, which cannot route to a" >&2
                echo "  compute node's HSN address. The reply is a Squid error" >&2
                echo "  page, not a refused connection, so it looks like a dead" >&2
                echo "  server rather than a misrouted client." >&2
                echo "" >&2
                echo "    no_proxy   ${no_proxy:-<unset>}" >&2
                echo "    BASE_URL   ${BASE_URL}" >&2
                echo "" >&2
                echo "  Two causes, and they look identical from here:" >&2
                echo "    * the shell never sourced the run's environment:" >&2
                echo "          source ${RUN_DIR:-<RUN_DIR>}/common_env.sh" >&2
                echo "    * BASE_URL came from an EARLIER run's banner. no_proxy" >&2
                echo "      is generated per run from that run's node, so an" >&2
                echo "      address from a previous allocation is never in it." >&2
                echo "      Take BASE_URL from the banner of the server that is" >&2
                echo "      up right now." >&2
                exit 1
                ;;
        esac
        ;;
esac

_probe=$(curl -s -S --max-time 120 -X POST "${BASE_URL}/v1/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${MODEL}\",\"prompt\":\"ping\",\"max_tokens\":1,\"temperature\":0}" 2>&1)
if ! printf '%s' "${_probe}" | grep -q '"choices"'; then
    echo "FATAL: ${BASE_URL} did not answer a one-token completion." >&2
    echo "  model: ${MODEL}" >&2
    # An HTML body is never vLLM. Say so in one line instead of dumping 800
    # bytes of Squid stylesheet and burying the diagnosis under it.
    if printf '%s' "${_probe}" | grep -qiE '<!DOCTYPE html|<html|squid'; then
        echo "  The reply is an HTML error page, so it came from an HTTP proxy," >&2
        echo "  not from vLLM. Preflight 0 should have caught this -- if it did" >&2
        echo "  not, \$no_proxy names ${_bu_host} but something else is" >&2
        echo "  intercepting: check \$http_proxy/\$https_proxy and that you are" >&2
        echo "  on the HSN." >&2
    else
        printf '%s\n' "${_probe}" | head -c 800 >&2
        echo "" >&2
    fi
    echo "  A 404 here means the front end does not serve /v1/completions." >&2
    echo "  An empty reply means nothing is listening on that host:port --" >&2
    echo "  check you are on the HSN (these are get_hsn_ip addresses) and that" >&2
    echo "  no_proxy covers them; a login-node shell needs curl --noproxy '*'." >&2
    exit 1
fi
echo "  preflight   one-token completion OK"

# --- Preflight 2: does ISL + OSL actually fit --max-model-len? -----------------
# 2026-09-11: MAX_MODEL_LEN was set to exactly 131072 + 8192 = 139264, and the
# warmup still took one "POST /v1/completions HTTP/1.1" 400 Bad Request. The
# random dataset does not hit the requested input length to the token, so an
# exactly-fitting budget is a coin flip per prompt. The rejected request costs
# more than a retry: it dies instantly, so the progress bar jumps to 1/N in
# three seconds and reads as progress while nothing is progressing.
#
# Scrape from METRICS_URLS, not BASE_URL -- the disagg arm's BASE_URL is
# toy_proxy_server.py, which serves no /v1/models (see CLOSED.md).
_first_metrics=${METRICS_URLS%%,*}
_mml=$(curl -sf --max-time 10 "${_first_metrics}/v1/models" 2>/dev/null \
        | tr ',' '\n' | grep -o '"max_model_len"[[:space:]]*:[[:space:]]*[0-9]\+' \
        | grep -o '[0-9]\+$' | head -1)
if [ -n "${_mml}" ]; then
    _need=$(( ISL + OSL ))
    echo "  preflight   max_model_len=${_mml}, ISL+OSL=${_need}"
    if [ "${_need}" -ge "${_mml}" ]; then
        echo "FATAL: ISL + OSL = ${_need} does not leave headroom under" >&2
        echo "  --max-model-len=${_mml}. Requests will be rejected with HTTP 400" >&2
        echo "  before any KV moves, and a rejected request looks like a fast one." >&2
        echo "  Relaunch with MAX_MODEL_LEN=$(( (_need + 8192 + 4095) / 4096 * 4096 ))." >&2
        exit 1
    fi
else
    echo "  preflight   WARN: could not read max_model_len from ${_first_metrics}" >&2
fi

# --- Config fingerprint -------------------------------------------------------
# The comparison is only valid if the two arms differ ONLY in the P/D split and
# the NIXL connector. Everything observable is recorded per arm so that a diff
# of two runs' config.json surfaces drift instead of hiding it inside a
# throughput number. Recorded, not asserted, because vLLM exposes only part of
# its effective config over HTTP -- the rest has to be read off the launcher.
for u in ${METRICS_URLS//,/ }; do
    tag=$(echo "${u}" | tr -c 'A-Za-z0-9' '_')
    curl -sf --max-time 10 "${u}/v1/models" -o "${OUT_DIR}/models_${tag}.json" 2>/dev/null
done
cat > "${OUT_DIR}/config.json" <<EOF
{
  "arm": "${ARM}",
  "base_url": "${BASE_URL}",
  "metrics_urls": "${METRICS_URLS}",
  "model": "${MODEL}",
  "isl": ${ISL},
  "osl": ${OSL},
  "num_prompts": ${NUM_PROMPTS},
  "max_concurrency": ${MAX_CONCURRENCY},
  "warmup_prompts": ${WARMUP_PROMPTS},
  "warmup_osl": ${WARMUP_OSL},
  "server_max_model_len": ${_mml:-null},
  "started": "$(date -Is)"
}
EOF

# --- /metrics snapshots -------------------------------------------------------
# vllm bench serve reports latency and throughput from the client side, which is
# what the SLOs are defined against. It knows nothing about nixl_bytes_transferred
# or prompt_tokens_by_source_total -- the disaggregation evidence -- so those come
# from a before/after diff of the engines' own counters.
snapshot_metrics() {
    local phase=$1
    for u in ${METRICS_URLS//,/ }; do
        local tag; tag=$(echo "${u}" | tr -c 'A-Za-z0-9' '_')
        curl -sf --max-time 20 "${u}/metrics" -o "${OUT_DIR}/metrics_${phase}_${tag}.txt" 2>/dev/null \
            || echo "  WARN: could not scrape ${u}/metrics at ${phase}" >&2
    done
}

run_bench() {
    local label=$1 n=$2 osl=$3 outfile=$4
    vllm bench serve \
        --backend openai \
        --base-url "${BASE_URL}" \
        --endpoint /v1/completions \
        --model "${MODEL}" \
        --dataset-name random \
        --random-input-len "${ISL}" \
        --random-output-len "${osl}" \
        --ignore-eos \
        --temperature 0 \
        --num-prompts "${n}" \
        --max-concurrency "${MAX_CONCURRENCY}" \
        --percentile-metrics ttft,tpot,itl,e2el \
        --metric-percentiles 95,99 \
        --save-result --result-filename "${outfile}" \
        2>&1 | tee "${OUT_DIR}/${label}.log"
    return "${PIPESTATUS[0]}"
}

# --- Warmup (discarded) -------------------------------------------------------
if [ "${WARMUP_PROMPTS}" -gt 0 ]; then
    echo "--- warmup (${WARMUP_PROMPTS} prompts, OSL=${WARMUP_OSL}, results discarded) ---"
    run_bench warmup "${WARMUP_PROMPTS}" "${WARMUP_OSL}" "${OUT_DIR}/warmup_result.json"
    _warm_rc=$?
    echo ""
    # A warmup that did not finish means the measured run is about to start on a
    # cold engine at best, and into a wedged one at worst. Stopping here costs a
    # relaunch; continuing costs the node-hour AND produces percentiles nobody
    # can quote. 2026-09-11: the first colocated attempt died inside warmup and
    # the harness had no opinion about it.
    if [ "${_warm_rc}" -ne 0 ]; then
        echo "FATAL: warmup exited ${_warm_rc}. Not starting the measured run." >&2
        echo "  Warmup log: ${OUT_DIR}/warmup.log" >&2
        echo "  Check the server log for an HTTP 400 (ISL+OSL over --max-model-len)" >&2
        echo "  or a stalled engine (no loggers.py:310 line for minutes)." >&2
        exit 1
    fi
fi

# --- Measured run -------------------------------------------------------------
snapshot_metrics before
echo "--- measured run (${NUM_PROMPTS} prompts) ---"
run_bench measured "${NUM_PROMPTS}" "${OSL}" "${OUT_DIR}/result.json"
BENCH_RC=$?
snapshot_metrics after

# --- Evidence the client cannot see -------------------------------------------
# --ignore-eos is on, so every request must emit exactly OSL tokens. If
# generation_tokens_total did not move by num_prompts*OSL, requests were
# truncated or rejected and the latency percentiles describe a different
# workload than the one on the label.
echo ""
echo "=== engine counter deltas ==="
for u in ${METRICS_URLS//,/ }; do
    tag=$(echo "${u}" | tr -c 'A-Za-z0-9' '_')
    b="${OUT_DIR}/metrics_before_${tag}.txt"; a="${OUT_DIR}/metrics_after_${tag}.txt"
    [ -f "${b}" ] && [ -f "${a}" ] || { echo "  ${u}: snapshots missing"; continue; }
    echo "  -- ${u} --"
    for k in \
        'vllm:prompt_tokens_by_source_total.*external_kv_transfer' \
        'vllm:prompt_tokens_by_source_total.*local_compute' \
        'vllm:generation_tokens_total' \
        'vllm:nixl_bytes_transferred_sum' \
        'vllm:nixl_bytes_transferred_count' \
        'vllm:nixl_num_descriptors_sum' \
        'vllm:nixl_xfer_time_seconds_sum' \
        'vllm:nixl_post_time_seconds_sum' \
        'vllm:num_preemptions_total' ; do
        # Bare metric name for display; the regex may carry a label selector.
        name=${k%%.*}
        vb=$(grep -E "^${k}" "${b}" 2>/dev/null | awk '{s+=$NF} END{printf "%.6g", s+0}')
        va=$(grep -E "^${k}" "${a}" 2>/dev/null | awk '{s+=$NF} END{printf "%.6g", s+0}')
        printf '     %-58s %14s -> %-14s\n' "${name##vllm:}" "${vb:-0}" "${va:-0}"
    done
done

echo ""
echo "  Sanity: with --ignore-eos, generation_tokens_total must move by"
echo "  ${NUM_PROMPTS} x ${OSL} = $(( NUM_PROMPTS * OSL )). A short delta means requests were"
echo "  truncated or rejected and the percentiles are not comparable across arms."
echo "  num_preemptions_total must stay at 0. Any preemption means the KV budget"
echo "  was exceeded and you measured recompute, not prefill interference --"
echo "  lower MAX_CONCURRENCY and re-run."

echo ""
echo "Results: ${OUT_DIR}/result.json   (bench_rc=${BENCH_RC})"
exit "${BENCH_RC}"
