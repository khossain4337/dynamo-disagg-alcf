#!/bin/bash
# Hang watch for the sweep runners (DECISIONS -11-09f). Run under setsid; stop with kill -- -PID.
# Usage: hang_watch.sh SHARED METRICS_URLS LOG...
# Fires on a vLLM ERROR line or rank death in LOG..., or on no progress for WINDOW_S with requests
# in flight. On fire: writes SHARED/dead_stage.txt, kills the process group in SHARED/bench.pid.
SHARED=$1 URLS=${2//,/ }; shift 2
WINDOW_S=1800 STEP_S=60
DEAD="${SHARED}/dead_stage.txt"

fire() {
    [ -f "${DEAD}" ] && return
    echo "$(date -u +%FT%TZ) $1" > "${DEAD}"
    echo "hang watch fired: $1" >&2
    [ -f "${SHARED}/bench.pid" ] && kill -TERM -- "-$(cat "${SHARED}/bench.pid")" 2>/dev/null
}

# vllm never exits while serving, so any rank exit counts.
tail -n0 -F "$@" 2>/dev/null | grep -m1 -E ' ERROR [0-9]{2}-[0-9]{2} [0-9:]{8} \[|rank [0-9]+ (died|exited)' \
    | while read -r line; do fire "log: ${line}"; done &

# KV usage moves on every prefill chunk; the token counters only on request output.
last= still=0
while sleep "${STEP_S}"; do
    [ -f "${DEAD}" ] && exit 0
    m=$(for u in ${URLS}; do curl --noproxy '*' -s -m 30 "${u}/metrics"; done)
    sig=$(grep -E '^vllm:(generation_tokens_total|prompt_tokens_total|kv_cache_usage_perc)\{' <<< "${m}")
    busy=$(grep -E '^vllm:num_requests_(running|waiting)\{' <<< "${m}" | awk '{s+=$NF} END{print (s>0)}')
    if [ -z "${sig}" ]; then still=$(( still + STEP_S ))
    elif [ "${busy}" = 0 ] || [ "${sig}" != "${last}" ]; then still=0
    else still=$(( still + STEP_S )); fi
    [ -n "${sig}" ] && last=${sig}
    [ "${still}" -ge "${WINDOW_S}" ] && fire "no progress for ${still} s with requests in flight"
done
