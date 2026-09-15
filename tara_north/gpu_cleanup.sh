#!/bin/bash
# shellcheck shell=bash
# =============================================================================
# gpu_cleanup.sh -- inspect, attribute and clear a serving node's GPUs.
#
# TWO WAYS TO USE IT, and it is the same file both times on purpose:
#
#   SOURCE it, to get the canonical process-pattern list and the functions:
#       source gpu_cleanup.sh          # defines VLLM_PROC_PATTERNS + functions
#
#   EXECUTE it ON A NODE, to report or to clear:
#       bash gpu_cleanup.sh report [port...]   read-only. exit 1 if dirty.
#       bash gpu_cleanup.sh kill   [port...]   kill, settle, re-report.
#
# Run it by hand before a launch, and the launchers call it over ssh during
# teardown. Exit status is the gate: 0 = clean, 1 = something is still resident.
#
# WHY THIS FILE EXISTS.
#
# 1. THE PATTERN LIST HAD TWO COPIES AND THEY WENT STALE TOGETHER. The launcher
#    screened for 'Worker_TP', a Nemotron-era process title. Under DP + EP the
#    workers are Worker_DP0_TP1_EP1 / Worker_DP1_TP0_EP4 and the engine is
#    EngineCore_DP0 -- none of which contain that substring. The pattern matched
#    zero processes in BOTH places it was used, so cleanup() reported a tidy
#    teardown while leaving eight workers holding GPU memory, and the next run's
#    preflight then called the node clean. It surfaced a stage later as
#    "Free memory on device cuda:1 (79.36/95.0 GiB) on startup is less than
#    desired GPU memory utilization (0.9, 85.5 GiB)" -- 15.64 GiB of someone
#    else's run, one stage and eight minutes away from the cause (2026-09-15).
#    A pattern that matches nothing is indistinguishable from a clean node.
#    ONE list now, here, sourced by everything that screens or kills.
#
# 2. nvidia-smi ANSWERS "HOW MUCH" BUT REFUSES TO ANSWER "WHO". It prints
#    "No running processes found" whenever the holder is a pid the caller cannot
#    see, which on these nodes includes every root-owned daemon. That is the
#    whole reason the standing rule has to end in "take another node": there was
#    no way to tell a leak you own from one you do not.
#
#    So this file reads GPU memory a SECOND way, from the kernel rather than
#    from CUDA. On Grace-Hopper each GPU's HBM is a CPU-visible NUMA node -- a
#    node WITH memory and NO cpus -- so /sys/devices/system/node/node<N>/meminfo
#    accounts it independently of the CUDA runtime and of pid ownership. The
#    nodes are discovered by empty cpulist rather than hardcoded; on these nodes
#    they come out as 4/12/20/28 against CPU NUMA affinity 0/1/2/3.
#
#    CALIBRATED 2026-09-15, AT IDLE, AND IT IS AN EXACT 1:1. Same node, same
#    instant, four for four, agreeing to under 1 MiB:
#
#        node  4  14682240 kB = 14338.1 MiB   GPU 0  nvidia-smi 14339 MiB
#        node 12  14157824 kB = 13826.0 MiB   GPU 1  nvidia-smi 13826 MiB
#        node 20  13108160 kB = 12800.9 MiB   GPU 2  nvidia-smi 12801 MiB
#        node 28  15729088 kB = 15360.4 MiB   GPU 3  nvidia-smi 15361 MiB
#
#    So the mapping is GPU i <-> NUMA node 4 + 8i, and MemUsed on that node IS
#    nvidia-smi's memory.used. That is a GPU-memory reading obtainable with
#    nothing but grep -- no NVIDIA tooling, no CUDA context, no pid visibility.
#
#    STILL OWED: the same comparison UNDER LOAD, which is the case that would
#    distinguish "tracks device memory" from "happens to agree at idle". Take it
#    mid-run with weights resident at a known GiB/GPU; at Inkling-Small's 64.16
#    GiB/GPU expect roughly 67.8 M kB per node. Until that reading exists,
#    nvidia-smi stays the gate and the HBM column is corroboration. The report
#    prints both, always, so it accrues from normal use rather than as an errand.
#
# 3. ATTRIBUTION, which is the part that turns a dead end into an action.
#    /proc/<pid>/numa_maps carries per-NUMA-node page counts per mapping, so
#    scanning it for pages on the HBM nodes names the PROCESS holding GPU
#    memory. Only for processes you own -- but a leak from your own previous
#    run is exactly the case that keeps costing allocations. "GPU 1 holds 15.6
#    GiB, held by pid 397992 Worker_DP0_TP1_EP1" is a kill command. "GPU 1 holds
#    15.6 GiB" is a new PBS job.
#
# The standing rule is unchanged and this file enforces its LAST step rather
# than replacing it: 0-2 GiB is this system's baseline, and GiB-scale memory
# that survives a kill sweep AND has no owner you can name is a leaked context.
# Take another node. What changes is that you now find that out in ten seconds,
# before the launch, instead of eight minutes in.
# =============================================================================

# The canonical list. Bracketed first characters ('[v]llm serve') because
# `ssh host "pkill -f 'vllm serve'"` runs through `bash -c` on the far side and
# THAT shell's command line contains the string being searched for, so an
# unbracketed pattern matches its own parent and the sweep kills its own ssh.
#
# Deliberately over-broad: these nodes run nothing else of ours, so a pattern
# with no false negatives beats one with no false positives. Both the current
# and the legacy spellings are kept -- an upgrade that renames processes must
# not silently disarm the sweep.
#
# WHEN vLLM IS UPGRADED, RE-CHECK THIS AGAINST `pgrep -af` DURING A LIVE RUN.
VLLM_PROC_PATTERNS=(
    '[v]llm serve'
    '[V]LLM::EngineCor'       # legacy engine title
    '[E]ngineCore_DP'         # observed 2026-09-15, vLLM 0.27.1 with DP
    '[W]orker_'               # Worker_TP0 and Worker_DP0_TP1_EP1 both
    '[V]LLM_DP_Coordinator'
)

# 0-2 GiB is this system's baseline, held by root-owned daemons (DCGM's
# nv-hostengine, the fabric manager, the IMEX daemon). The bar is loose on
# purpose: it is here to catch a whole leaked context, not to police MiB.
GPU_DIRTY_MIB=${GPU_DIRTY_MIB:-4096}

# --- HBM NUMA nodes: discovered, never hardcoded -----------------------------
# A GPU HBM node is a NUMA node that has memory and no CPUs. Hardcoding
# 4/12/20/28 would be right on today's nodes and wrong the first time this runs
# anywhere else, silently and in the reassuring direction.
gpu_cleanup_hbm_nodes() {
    local d n
    for d in /sys/devices/system/node/node*; do
        [ -r "${d}/cpulist" ] || continue
        [ -s "${d}/meminfo" ] || continue
        if [ -z "$(tr -d '[:space:]' < "${d}/cpulist")" ]; then
            n=${d##*/node}
            printf '%s\n' "${n}"
        fi
    done
}

# --- Attribution: which OWNED process holds pages on the HBM nodes -----------
# Second argument is whether anything was actually found over the bar. Without
# it this printed "TAKE ANOTHER NODE" at the end of every CLEAN report, which
# is how a gate teaches you to stop reading it.
gpu_cleanup_attribute() {
    local nodes_re=$1 is_dirty=$2 m p cmd found=0
    for m in /proc/[0-9]*/numa_maps; do
        [ -r "${m}" ] || continue
        grep -qE " N(${nodes_re})=" "${m}" 2>/dev/null || continue
        p=${m#/proc/}; p=${p%/numa_maps}
        cmd=$(tr '\0' ' ' < "/proc/${p}/cmdline" 2>/dev/null | cut -c1-90)
        [ -z "${cmd}" ] && cmd="[$(cat "/proc/${p}/comm" 2>/dev/null)]"
        printf '    pid %-8s %s\n' "${p}" "${cmd}"
        found=1
    done
    if [ "${found}" -eq 0 ]; then
        if [ "${is_dirty}" -ne 0 ]; then
            echo "    NONE -- memory is reported above but no process you own"
            echo "    holds HBM pages, so it belongs to a pid you cannot see or"
            echo "    kill. That is a leaked context: TAKE ANOTHER NODE."
        else
            echo "    none"
        fi
    fi
}

# --- Report. Returns 0 clean, 1 dirty ----------------------------------------
gpu_cleanup_report() {
    local host dirty=0 idx used pat hits nodes nodes_re n kb _apps _holders _h
    host=$(hostname -s)
    echo "=== gpu_cleanup report: ${host} ==="

    echo "  -- surviving processes --"
    hits=0
    for pat in "${VLLM_PROC_PATTERNS[@]}"; do
        if pgrep -f "${pat}" >/dev/null 2>&1; then
            printf '    pattern %s\n' "${pat}"
            pgrep -af "${pat}" 2>/dev/null | cut -c1-110 | sed 's/^/      /'
            hits=1; dirty=1
        fi
    done
    [ "${hits}" -eq 0 ] && echo "    none"

    # THE ATTRIBUTION THAT ACTUALLY WORKS FOR DEVICE MEMORY. `top` cannot answer
    # this -- it reports host RSS, and a cudaMalloc'd buffer never appears
    # there, so a node holding 15 GiB per GPU looks completely idle. Neither can
    # /proc/<pid>/numa_maps below: device allocations are not mapped into the
    # process's CPU address space, which is why the HBM nodes report
    # `Mapped: 0 kB` while 15 GiB is unmistakably allocated. The only instrument
    # that ties device memory to a pid is the driver's own compute-apps table.
    #
    # It lists pids regardless of owner, so a root-owned holder shows up here
    # even though you cannot kill it -- which is exactly the distinction the
    # standing rule turns on: a pid you own is a kill, a pid you do not own is
    # another node. If it prints nothing while memory is held, the allocation
    # has no live process behind it at all and no sweep will recover it.
    echo "  -- GPU memory by process (driver compute-apps table) --"
    if _apps=$(nvidia-smi --query-compute-apps=pid,used_memory,process_name \
                   --format=csv,noheader 2>/dev/null) && [ -n "${_apps}" ]; then
        printf '%s\n' "${_apps}" | cut -c1-110 | sed 's/^/    /'
        # A process in this table is holding device memory by definition, even
        # if its command line no longer matches any pattern above.
        dirty=1
    else
        echo "    none"
    fi

    # THE GAP BETWEEN "compute-apps IS EMPTY" AND "55 GiB IS HELD". The driver
    # table above enumerates live, healthy CUDA contexts. A process wedged in
    # the driver (state D, uninterruptible) or already reaped to a zombie does
    # not appear there and still owns its allocation, and neither does a
    # nvidia-cuda-mps server holding contexts for clients that already exited.
    # Observed 2026-09-15: four GPUs at 12-15 GiB each, compute-apps empty, no
    # pattern match, nothing in top -- 55 GiB with no visible owner.
    #
    # /dev/nvidia* holders is the widest net available without root: it lists
    # anything with the device open, healthy context or not.
    #
    # READ-ONLY, DELIBERATELY. This section diagnoses and never acts. Recovering
    # memory from a D-state holder needs a GPU reset, which needs root and is
    # disruptive on a shared node -- it must never fire from an automated
    # teardown. If this section is empty too and the memory persists, no
    # user-space action will recover it: the allocation clears when the PBS job
    # ends, and until then the standing remedy is to give that node the CLIENT
    # role, whose GPUs are idle by design, and serve from the clean ones.
    echo "  -- /dev/nvidia* holders (catches D-state, zombies, MPS) --"
    if command -v fuser >/dev/null 2>&1 \
       && _holders=$(fuser /dev/nvidia* 2>/dev/null | tr -s ' ' '\n' | grep -E '^[0-9]+$' | sort -u) \
       && [ -n "${_holders}" ]; then
        for _h in ${_holders}; do
            printf '    pid %-8s %-18s state=%s\n' "${_h}" \
                "$(cat "/proc/${_h}/comm" 2>/dev/null || echo unknown)" \
                "$(awk '{print $3}' "/proc/${_h}/stat" 2>/dev/null || echo '?')"
        done
        echo "    state D = wedged in the driver, unkillable; Z = zombie."
        dirty=1
    else
        echo "    none"
    fi
    if pgrep -f '[n]vidia-cuda-mps' >/dev/null 2>&1; then
        echo "    MPS server present -- holds contexts for clients that exited:"
        pgrep -af '[n]vidia-cuda-mps' 2>/dev/null | sed 's/^/      /'
        dirty=1
    fi

    echo "  -- GPU memory, via CUDA (nvidia-smi) --"
    while read -r idx used; do
        [ -z "${idx}" ] && continue
        printf '    GPU %-3s %8s MiB' "${idx}" "${used}"
        if [ "${used:-0}" -gt "${GPU_DIRTY_MIB}" ]; then
            printf '   OVER the %s MiB bar\n' "${GPU_DIRTY_MIB}"; dirty=1
        else
            printf '\n'
        fi
    done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ',')

    # The second, independent view. Printed unconditionally so that every run
    # contributes a calibration point against the column above it.
    #
    # A read loop rather than mapfile: this file is also read by hand on
    # whatever shell is in front of you, and mapfile is bash 4+. On bash 3.2 it
    # fails, leaves the array in an ill-defined state, and the report then
    # claims to have checked NUMA nodes it never found -- a gate reporting a
    # check it did not perform.
    nodes=()
    while IFS= read -r n; do
        [ -n "${n}" ] && nodes+=("${n}")
    done < <(gpu_cleanup_hbm_nodes)
    if [ "${#nodes[@]}" -gt 0 ]; then
        echo "  -- GPU memory, via kernel NUMA accounting (HBM nodes ${nodes[*]}) --"
        for n in "${nodes[@]}"; do
            kb=$(awk '/MemUsed/ {print $4}' "/sys/devices/system/node/node${n}/meminfo" 2>/dev/null)
            [ -z "${kb}" ] && continue
            awk -v n="${n}" -v kb="${kb}" \
                'BEGIN {printf "    node %-4s %8.2f GiB used\n", n, kb/1048576}'
        done
        nodes_re=$(IFS='|'; printf '%s' "${nodes[*]}")
        echo "  -- HBM pages by owned process (CPU-side mappings only) --"
        gpu_cleanup_attribute "${nodes_re}" "${dirty}"
    else
        echo "  -- no memoryless-CPU NUMA nodes found; HBM cross-check unavailable --"
    fi

    if [ "${dirty}" -eq 0 ]; then
        echo "  VERDICT: clean"
    else
        echo "  VERDICT: DIRTY"
    fi
    return "${dirty}"
}

# --- Kill sweep --------------------------------------------------------------
gpu_cleanup_kill() {
    local pat port
    echo "=== gpu_cleanup kill: $(hostname -s) ==="
    # TERM first so a live server gets to close its sockets, then KILL what is
    # left. Anything still up after a SIGTERM during teardown is wedged.
    for pat in "${VLLM_PROC_PATTERNS[@]}"; do
        pkill -TERM -f "${pat}" 2>/dev/null
    done
    sleep 5
    for pat in "${VLLM_PROC_PATTERNS[@]}"; do
        pkill -KILL -f "${pat}" 2>/dev/null
    done
    for port in "$@"; do
        [ -n "${port}" ] && fuser -k "${port}/tcp" 2>/dev/null
    done

    # THEN KILL BY WHAT THE DRIVER SAYS IS HOLDING MEMORY, not by name. The
    # pattern sweep above is a guess about process titles, and this project has
    # now been bitten twice by a title changing underneath it. The compute-apps
    # table is ground truth: anything still in it after the sweep is holding
    # device memory whatever it calls itself. This is also the step that answers
    # "how do I know which pid to kill" without anyone reading a table -- `top`
    # cannot see device memory, so there was no manual answer to that question.
    #
    # Processes owned by others fail with EPERM and are left alone, which is the
    # correct outcome: an unkillable holder is the leaked-context case, and the
    # report that follows will still show the memory and say so.
    local _pid _left
    _left=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
    if [ -n "${_left}" ]; then
        echo "  still holding device memory after the pattern sweep:"
        for _pid in ${_left}; do
            case "${_pid}" in ''|*[!0-9]*) continue ;; esac
            printf '    kill -KILL %s  (%s)\n' "${_pid}" \
                "$(cat "/proc/${_pid}/comm" 2>/dev/null || echo unknown)"
            kill -KILL "${_pid}" 2>/dev/null \
                || echo "      EPERM -- not yours. If memory survives, take another node."
        done
    fi
    # CUDA contexts are reaped asynchronously; reporting immediately after
    # SIGKILL reads memory that is already on its way out and calls a clean
    # node dirty.
    sleep 5
}

gpu_cleanup_main() {
    local mode=${1:-report}
    shift 2>/dev/null || true
    case "${mode}" in
        report)
            gpu_cleanup_report
            ;;
        kill)
            gpu_cleanup_kill "$@"
            gpu_cleanup_report
            ;;
        *)
            echo "usage: bash gpu_cleanup.sh [report|kill] [port...]" >&2
            echo "  report  read-only inspection; exit 1 if dirty" >&2
            echo "  kill    TERM, KILL, free ports, settle, then report" >&2
            return 2
            ;;
    esac
}

# Sourced -> definitions only. Executed -> run. This is what lets the launchers
# screen on the SAME array this file kills on, which is the defect that made
# the file necessary.
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    gpu_cleanup_main "$@"
fi
