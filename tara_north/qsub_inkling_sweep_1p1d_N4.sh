#!/bin/bash
#PBS -l select=6:tier1=x4820c0 -l place=scatter:group=tier1
#PBS -l walltime=11:59:00
#PBS -A inference_service
#PBS -q workq
#PBS -k doe
#PBS -N d-73728
#PBS -o /vast/draco/tara/projects/Tara_Deployment/software/testing/RUNS/outdir_tara
#PBS -e /vast/draco/tara/projects/Tara_Deployment/software/testing/RUNS/errordir_tara
#PBS -j oe
#
TARA_NORTH=/vast/draco/tara/projects/Tara_Deployment/software/dynamo-disagg-alcf/tara_north
NODEFILE_DIR=/vast/draco/tara/projects/Tara_Deployment/software/testing/RUNS
export TARA_NORTH NODEFILE_DIR

MAX_ATTEMPTS=3
[ "$MAX_ATTEMPTS" -ge 1 ] || { echo "MAX_ATTEMPTS must be >= 1" >&2; exit 1; }
API_SERVER_COUNT=4
ISL=1024
OSL=256
CONCURRENCIES="4 8 16 32 64 128"
MAX_MODEL_LEN=73728

# PBS_NODEFILE is set only on the mother superior and lives under its local
# /var/spool. Staging it to /vast is what lets the run script be launched from
# any node in the job.
export JOB_NODEFILE="${NODEFILE_DIR}/${PBS_JOBID}.txt"
cat "$PBS_NODEFILE" > "$JOB_NODEFILE"

# The run script takes its client node from `hostname -s` and filters it out of
# the role list, so "park a node" means "launch from it".
launch() {
    local host=$1 conc=$2
    local cmd="export PBS_NODEFILE=${JOB_NODEFILE}; \
        API_SERVER_COUNT=${API_SERVER_COUNT} ISL=${ISL} OSL=${OSL} \
        MAX_MODEL_LEN=${MAX_MODEL_LEN} \
        CONCURRENCIES='${conc}' bash ${TARA_NORTH}/run_inkling_sweep_1p1d_N4.sh"
    if [ "$host" = "$(hostname -s)" ]; then
        bash -c "$cmd"
    else
        ssh -n "$host" "$cmd"
    fi
}

CLIENT=$(hostname -s)
PARKED=
TODO="$CONCURRENCIES"
# RC is the RETURN CODE (exit code) of run_inkling_sweep_1p1d_N4.sh on the last
# attempt. It is NOT a counter -- `attempt` is the counter. Its values name the
# fault types we have actually seen, and that file's line 11 is the list:
#   0 ladder finished | 1 config or boot failure | 10 dirty node
#   11 pool gate      | 30 a server died mid-ladder
# Named RC because the run script calls it RC too. Every branch below tests it.
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    LOG="${NODEFILE_DIR}/qsub_${PBS_JOBID}_isl${ISL}_osl${OSL}_mml${MAX_MODEL_LEN}_a${attempt}.log"
    echo "=== attempt ${attempt}/${MAX_ATTEMPTS}  client=${CLIENT}  concurrencies=${TODO} ==="
    launch "$CLIENT" "$TODO" 2>&1 | tee "$LOG"
    RC=${PIPESTATUS[0]}        # $? is tee's
    [ "$RC" -eq 0 ] && break

    # 30 alone leaves part of the ladder done. Every other code means nothing
    # ran, and the glob would then match a previous job's run dir.
    if [ "$RC" -eq 30 ]; then
        _last=$(ls -dt "${NODEFILE_DIR}"/inkling_1p1d_dp2tp4_isl${ISL}_osl${OSL}_mml${MAX_MODEL_LEN}_* 2>/dev/null | head -1)
        _ok=$(awk -F'\t' 'NR>1 && $4=="ok" {print $3}' "${_last}/bench_status.tsv" 2>/dev/null)
        _rem=
        for c in $TODO; do printf '%s\n' "$_ok" | grep -qx "$c" || _rem="${_rem} ${c}"; done
        TODO="${_rem# }"
        [ -z "$TODO" ] && { RC=0; break; }
    fi

    _dirty=$(grep -m1 '^DIRTY_NODE:' "$LOG" | cut -d: -f2)
    if [ "$RC" -eq 10 ] && [ -n "$_dirty" ]; then
        # Parked, not cleaned: the holder never appears in --query-compute-apps,
        # so gpu_cleanup.sh is not known to free it (-10-08b).
        if [ -n "$PARKED" ]; then
            echo "FATAL: ${_dirty} dirty and ${PARKED} already parked. select=6 has no spare." >&2
            break
        fi
        PARKED="$_dirty"
        CLIENT="$_dirty"
    fi
done

# TODO still holds the list the last attempt ran; only exit 30 trims it.
[ "$RC" -eq 0 ] && TODO=
echo "=== exit_code=${RC}  parked=${PARKED:-none}  unfinished=${TODO:-none} ==="
exit "$RC"
