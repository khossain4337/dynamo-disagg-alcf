#!/bin/bash
#PBS -l select=19 -l place=scatter:excl
#PBS -l walltime=06:00:00
#PBS -A inference_service
#PBS -q workq
#PBS -k doe
#PBS -N ms-par-pd
#PBS -o /vast/draco/tara/projects/Tara_Deployment/software/testing/RUNS/outdir_tara
#PBS -e /vast/draco/tara/projects/Tara_Deployment/software/testing/RUNS/errordir_tara
#PBS -j oe
# Muse-Spark greedy parity, P:D PP8 -> PP8. 16 serving + proxy + client + 1 spare. EP=0 at qsub -v for EP off.
TARA_NORTH=/vast/draco/tara/projects/Tara_Deployment/software/dynamo-disagg-alcf/tara_north
MUSE_SPARK=${TARA_NORTH}/multi_model/muse_spark
RUNS=/vast/draco/tara/projects/Tara_Deployment/software/testing/RUNS
EP=${EP:-1}
PROXY_SCRIPT=${PROXY_SCRIPT:?qsub -v PROXY_SCRIPT=<toy_proxy_server.py from the 9e3db48873 tree>}
MAX_ATTEMPTS=3

# PBS_NODEFILE is local to the mother superior; staged so any node can launch.
JOB_NODEFILE=${RUNS}/${PBS_JOBID}.txt
cat "${PBS_NODEFILE}" > "${JOB_NODEFILE}"

# A dirty node is parked, never cleaned: the first becomes the client, later ones sit out.
CLIENT=$(hostname -s)
PARKED=
for attempt in $(seq 1 "${MAX_ATTEMPTS}"); do
    LOG=${RUNS}/qsub_${PBS_JOBID}_muse_spark_parity_pd_ep${EP}_a${attempt}.log
    echo "=== attempt ${attempt}/${MAX_ATTEMPTS} client=${CLIENT} parked=${PARKED:-none} ==="
    cmd="export PBS_NODEFILE=${JOB_NODEFILE} EP=${EP} PROXY_SCRIPT=${PROXY_SCRIPT} EXCLUDE_NODES='${PARKED}'; \
        bash ${MUSE_SPARK}/run_muse_spark_parity_pd_tp4pp8_N16.sh"
    if [ "${CLIENT}" = "$(hostname -s)" ]; then bash -c "${cmd}"; else ssh -n "${CLIENT}" "${cmd}"; fi 2>&1 | tee "${LOG}"
    RC=${PIPESTATUS[0]}
    [ "${RC}" -eq 10 ] || break
    for d in $(grep '^DIRTY_NODE:' "${LOG}" | cut -d: -f2 | sort -u); do PARKED="${PARKED} ${d}"; done
    PARKED=${PARKED# }
    CLIENT=${PARKED%% *}
done
echo "=== exit_code=${RC} parked=${PARKED:-none} ==="
exit "${RC}"
