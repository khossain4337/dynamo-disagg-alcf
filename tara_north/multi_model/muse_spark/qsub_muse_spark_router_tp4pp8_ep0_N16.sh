#!/bin/bash
#PBS -l select=19 -l place=scatter:excl
#PBS -l walltime=06:00:00
#PBS -A inference_service
#PBS -q workq
#PBS -k doe
#PBS -N ms-rt-ep0
#PBS -o /vast/draco/tara/projects/Tara_Deployment/software/testing/RUNS/outdir_tara
#PBS -e /vast/draco/tara/projects/Tara_Deployment/software/testing/RUNS/errordir_tara
#PBS -j oe
# Muse-Spark router arm, EP off: 2 x TP4 x PP8 + vllm-router. 16 serving + router + client + 1 spare.
# c=32 through the router = 16 per instance, next to the pilot's c=16.
TARA_NORTH=/vast/draco/tara/projects/Tara_Deployment/software/dynamo-disagg-alcf/tara_north
MUSE_SPARK=${TARA_NORTH}/multi_model/muse_spark
RUNS=/vast/draco/tara/projects/Tara_Deployment/software/testing/RUNS
EP=0
ISL=1024
OSL=256
CONCURRENCIES="32"
MAX_ATTEMPTS=3

# PBS_NODEFILE is local to the mother superior; staged so any node can launch.
JOB_NODEFILE=${RUNS}/${PBS_JOBID}.txt
cat "${PBS_NODEFILE}" > "${JOB_NODEFILE}"

# A dirty node is parked, never cleaned: the first becomes the client, later ones sit out.
CLIENT=$(hostname -s)
PARKED=
TODO=${CONCURRENCIES}
for attempt in $(seq 1 "${MAX_ATTEMPTS}"); do
    LOG=${RUNS}/qsub_${PBS_JOBID}_muse_spark_router_ep${EP}_isl${ISL}_osl${OSL}_a${attempt}.log
    echo "=== attempt ${attempt}/${MAX_ATTEMPTS} client=${CLIENT} parked=${PARKED:-none} concurrencies=${TODO} ==="
    cmd="export PBS_NODEFILE=${JOB_NODEFILE} EP=${EP} ISL=${ISL} OSL=${OSL} CONCURRENCIES='${TODO}' EXCLUDE_NODES='${PARKED}'; \
        bash ${MUSE_SPARK}/run_muse_spark_router_tp4pp8_N16.sh"
    if [ "${CLIENT}" = "$(hostname -s)" ]; then bash -c "${cmd}"; else ssh -n "${CLIENT}" "${cmd}"; fi 2>&1 | tee "${LOG}"
    RC=${PIPESTATUS[0]}
    [ "${RC}" -eq 0 ] && break
    if [ "${RC}" -eq 30 ]; then
        _dir=$(grep -m1 '^RUN_DIR:' "${LOG}" | cut -d: -f2)
        _ok=$(awk -F'\t' 'NR>1 && $4=="ok" {print $3}' "${_dir}/bench_status.tsv" 2>/dev/null)
        _rem=
        for c in ${TODO}; do printf '%s\n' "${_ok}" | grep -qx "${c}" || _rem="${_rem} ${c}"; done
        TODO=${_rem# }
        [ -z "${TODO}" ] && { RC=0; break; }
    fi
    if [ "${RC}" -eq 10 ]; then
        for d in $(grep '^DIRTY_NODE:' "${LOG}" | cut -d: -f2 | sort -u); do PARKED="${PARKED} ${d}"; done
        PARKED=${PARKED# }
        CLIENT=${PARKED%% *}
    fi
done
[ "${RC}" -eq 0 ] && TODO=
echo "=== exit_code=${RC} parked=${PARKED:-none} unfinished=${TODO:-none} ==="
exit "${RC}"
