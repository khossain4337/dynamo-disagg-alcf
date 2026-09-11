# shellcheck shell=bash
# =============================================================================
# workload_profile.sh -- the single definition of a Figure 2 workload point.
#
# SOURCE this file, do not execute it. It defines one function:
#
#     load_workload_profile <name>
#
# which sets the WL_* variables below in the caller's scope, and returns 1 on an
# unknown name after listing the valid ones.
#
# WHY THIS IS A SEPARATE FILE. The same numbers are needed in two places that
# run at different times on different nodes: bench_arm.sh needs ISL/OSL/
# concurrency to drive the client, and run_colocated_N1.sh needs the matching
# --max-model-len hours earlier, at server launch, when nothing about the client
# exists yet. Those two numbers have to agree or every request comes back HTTP
# 400 -- instantly, which reads on the progress bar as a fast request rather
# than as a fault (2026-09-11). This is the same argument emit_common_env.sh
# makes about the environment, applied to the workload. One copy.
#
# THE PROFILE SETS DEFAULTS, THE ENVIRONMENT WINS. Every consumer applies these
# as ${VAR:-${WL_VAR}}, so a one-off override still works:
#
#     WORKLOAD=iter32k MAX_CONCURRENCY=64 bash bench_arm.sh
#
# WL_MAX_MODEL_LEN IS NOT A KV BUDGET. It sizes the per-request position budget,
# not the KV pool -- the pool comes from --gpu-memory-utilization. Raising it is
# close to free, which is why every profile carries a full 8192 of slack. The
# one thing it does change is cosmetic and misleading: vLLM's startup line
# "Maximum concurrency for N tokens per request: X" divides the pool by
# max-model-len, NOT by the ISL you are about to send. At iter32k it will print
# ~121x against 43008; the real ceiling at 34816 tokens/request is 150.0x.
# Measured pool 2026-09-11: 5,222,400 tokens (MEMORY_2026-09-11c.md).
# =============================================================================

load_workload_profile() {
    local _name=${1:?load_workload_profile: profile name required}

    # Common to every profile. Overridden per-profile below where they differ.
    WL_NAME=${_name}
    WL_WARMUP_OSL=128
    WL_SLO_TTFT_MS=""
    WL_SLO_ITL_P95_MS=""

    case "${_name}" in

    # -- The settled Figure 2 point. MEMORY_2026-09-10.md, do not re-litigate. --
    # Prefill pressure per output token is ISL/OSL; at ratio 16 the decode
    # stream sees ~9x the interference ShareGPT's ~1.7 produces.
    pilot128k)
        WL_DESC="settled Figure 2 point -- ISL/OSL ratio 16, long context"
        WL_ISL=131072
        WL_OSL=8192
        # 16, not 32. Measured 2026-09-11: 32/37.5 = 85% of the KV ceiling
        # leaves no room for the prefill working set, KV climbed to 97.4% and
        # prompt throughput collapsed 25,027 -> 71.8 tok/s. 16 is 43% and never
        # approaches the preemption path. It is also the CONSERVATIVE choice for
        # the hypothesis: less decode competing with prefill makes the colocated
        # baseline look better, so a disagg win at 16 is the stronger claim.
        WL_MAX_CONCURRENCY=16
        # PROVISIONAL, pending the uncontended decode floor (handoff item 1).
        # 128 does not fit -- at OSL 8192 it is hours per arm. 32 is two full
        # waves at concurrency 16; n is not the binding constraint for ITL
        # percentiles anyway, since each request contributes ~8,191 intervals.
        WL_NUM_PROMPTS=32
        WL_WARMUP_PROMPTS=16
        WL_MAX_MODEL_LEN=147456          # 131072 + 8192 + 8192 slack
        WL_SLO_TTFT_MS=8000
        WL_SLO_ITL_P95_MS=25
        ;;

    # -- The iteration point. Same ratio, ~4-7x cheaper per run. ---------------
    # EQUIVALENT BECAUSE THE RATIO IS PRESERVED, not because it is smaller:
    #
    #     decode steps per prefill chunk = OSL / (ISL / B) = B / (ISL/OSL)
    #                                    = 16384 / 16 = 1024
    #
    # identical at (131072, 8192) and (32768, 2048). A 16384-token chunk-step
    # costs the same wall time either way, so both the length of a prefill stall
    # and its arrival rate per decode step carry over -- and that is the
    # mechanism Figure 2 exists to measure.
    #
    # The ITL SLO carries over too, because decode here is WEIGHT-bound rather
    # than KV-bound: ~537 MB of attention KV per rank per token at 131k against
    # ~6 GB of active MoE weights, so KV is ~8% of the per-step read at 131k and
    # ~2% at 32k. The uncontended floor should move by well under 10%. THAT IS
    # THE GATE -- measure floor128k and floor32k before trusting this profile.
    #
    # TTFT DOES NOT CARRY OVER AND IS NOT INHERITED. Prefill is 32768/20747 ~
    # 1.6 s here against 6.3 s at 128k, so the 8 s bar would never bind -- and a
    # non-binding SLO turns SLO-constrained goodput back into raw throughput,
    # which is the comparison we already know disagg loses.
    #
    # KNOWN, PREDICTABLE PENALTY TO DISAGG: the Mamba floor does not scale with
    # context. Per request the transfer drops 587.8 -> 178.9 MB/rank, but the
    # 42.56 MB fixed floor goes from 7.2% of the payload to 23.8%. Conservative
    # direction, exactly computable from the closed Figure 1 model. State it in
    # the writeup rather than letting a reviewer find it.
    iter32k)
        WL_DESC="iteration point -- same ISL/OSL ratio 16, ~4-7x cheaper"
        WL_ISL=32768
        WL_OSL=2048
        # Same 16 as the pilot, so the per-request interference is identical.
        # Headroom is much larger here (34,816 tokens/request -> 150.0x rather
        # than 37.50x), so 32/64/96 are all reachable for the goodput CURVE that
        # the 128k point could never produce. Override MAX_CONCURRENCY to sweep.
        WL_MAX_CONCURRENCY=16
        # 4 waves. 64 x 2,047 = ~131k ITL samples, ample for p99.
        WL_NUM_PROMPTS=64
        WL_WARMUP_PROMPTS=16
        WL_MAX_MODEL_LEN=43008           # 32768 + 2048 + 8192 slack
        WL_SLO_TTFT_MS=2000              # re-derived, NOT inherited from 8000
        WL_SLO_ITL_P95_MS=25
        ;;

    # -- Handoff item 1: the uncontended decode floor. ~15 seconds. ------------
    # One request, no competition, full context. Its TPOT is the denominator for
    # everything else: it sizes the pilot, it separates "decode is slow" from
    # "prefill is stealing decode" (disagg can only remove the second), and it
    # says whether 25 ms p95 ITL is a knife-edge or unreachable by any arm.
    #
    # RUN BOTH. The pair is the gate on iter32k: if the two floors agree to
    # within ~10%, decode is weight-bound as predicted, the ratio argument holds
    # and the 25 ms ITL SLO transfers unchanged. If floor32k is much faster,
    # decode is more KV-bound than the arithmetic says and the ITL bar has to be
    # tightened for the small workload or the comparison is not equivalent.
    #
    # Both fit under pilot128k's 147456, so the pair runs against a server that
    # is ALREADY UP at the 128k profile -- no relaunch, no second allocation.
    #
    # No SLOs: n=1 uncontended is the reference these are judged against, not a
    # candidate to judge.
    floor128k)
        WL_DESC="uncontended decode floor at 128k -- handoff item 1"
        WL_ISL=131072
        WL_OSL=512
        WL_MAX_CONCURRENCY=1
        WL_NUM_PROMPTS=1
        WL_WARMUP_PROMPTS=0
        WL_MAX_MODEL_LEN=147456
        ;;

    floor32k)
        WL_DESC="uncontended decode floor at 32k -- the gate on iter32k"
        WL_ISL=32768
        WL_OSL=512
        WL_MAX_CONCURRENCY=1
        WL_NUM_PROMPTS=1
        WL_WARMUP_PROMPTS=0
        WL_MAX_MODEL_LEN=43008
        ;;

    *)
        echo "FATAL: unknown WORKLOAD '${_name}'." >&2
        echo "  Valid profiles:" >&2
        echo "    pilot128k   ISL 131072 / OSL 8192  conc 16   the settled Figure 2 point" >&2
        echo "    iter32k     ISL  32768 / OSL 2048  conc 16   same ratio, cheap iteration" >&2
        echo "    floor128k   ISL 131072 / OSL  512  conc  1   handoff item 1" >&2
        echo "    floor32k    ISL  32768 / OSL  512  conc  1   the gate on iter32k" >&2
        return 1
        ;;
    esac
}

# -----------------------------------------------------------------------------
# workload_transfer_bytes_per_rank <isl>
#
# The closed Figure 1 model, echoed at runtime so the disagg arm's measured
# nixl_bytes_transferred_sum has something to be checked against instead of
# being merely recorded. Exact to the byte on all four measured points
# (MEMORY_2026-09-10.md); any deviation is new physics, not noise.
#
#     bytes/rank = (40 + 8 * ceil(tokens / 2080)) * 1,064,960 - 40,960
#
# 40 Mamba layers at one 1,064,960 B page each (a 42,557,440 B floor, 0.3% from
# the analytic 42.7 MB) plus 8 attention layers at one 2,080-token page each,
# which is 512 B/token/layer x 8 = 4,096 B/token/rank exactly.
# -----------------------------------------------------------------------------
workload_transfer_bytes_per_rank() {
    local _tok=${1:?workload_transfer_bytes_per_rank: token count required}
    local _attn_pages=$(( 8 * ( (_tok + 2079) / 2080 ) ))
    echo $(( (40 + _attn_pages) * 1064960 - 40960 ))
}
