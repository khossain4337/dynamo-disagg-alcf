#!/bin/bash
#PBS -l select=10 -l place=scatter:excl
#PBS -l walltime=04:00:00
#PBS -A inference_service
#PBS -q workq
#PBS -k doe
#PBS -N ms-nat-noplug
#PBS -o /vast/draco/tara/projects/Tara_Deployment/software/testing/RUNS/outdir_tara
#PBS -e /vast/draco/tara/projects/Tara_Deployment/software/testing/RUNS/errordir_tara
#PBS -j oe
# Muse-Spark pilot: Nathan's serve args verbatim, TP4 x PP8, EP off, (1024,256), MR cache monitors on (userfaultfd), no NCCL net plugin (Socket). 8 serving + client + 1 spare.
TARA_NORTH=/vast/draco/tara/projects/Tara_Deployment/software/dynamo-disagg-alcf/tara_north
MUSE_SPARK=${TARA_NORTH}/multi_model/muse_spark
RUNS=/vast/draco/tara/projects/Tara_Deployment/software/testing/RUNS
ISL=1024
OSL=256
CONCURRENCIES="1 2 4 8 16"
MR_CACHE=on
MAX_ATTEMPTS=1

# PBS_NODEFILE is local to the mother superior; staged so any node can launch.
JOB_NODEFILE=${RUNS}/${PBS_JOBID}.txt
cat "${PBS_NODEFILE}" > "${JOB_NODEFILE}"

# A dirty node is parked, never cleaned: the first becomes the client, later ones sit out.
CLIENT=$(hostname -s)
PARKED=
TODO=${CONCURRENCIES}
for attempt in $(seq 1 "${MAX_ATTEMPTS}"); do
    LOG=${RUNS}/qsub_${PBS_JOBID}_muse_spark_nathan_mrc${MR_CACHE}_no_ncclplugin_isl${ISL}_osl${OSL}_a${attempt}.log
    echo "=== attempt ${attempt}/${MAX_ATTEMPTS} client=${CLIENT} parked=${PARKED:-none} concurrencies=${TODO} ==="
    cmd="export PBS_NODEFILE=${JOB_NODEFILE} ISL=${ISL} OSL=${OSL} CONCURRENCIES='${TODO}' MR_CACHE=${MR_CACHE} EXCLUDE_NODES='${PARKED}'; \
        bash ${MUSE_SPARK}/run_muse_spark_pilot_tp4pp8_nathan_no_ncclplugin_N8.sh"
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
