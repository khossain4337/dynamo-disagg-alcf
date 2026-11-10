#!/bin/bash
#PBS -l select=10 -l place=scatter:excl
#PBS -l walltime=03:00:00
#PBS -A inference_service
#PBS -q workq
#PBS -k doe
#PBS -N ms-tp16short
#PBS -o /vast/draco/tara/projects/Tara_Deployment/software/testing/RUNS/outdir_tara
#PBS -e /vast/draco/tara/projects/Tara_Deployment/software/testing/RUNS/errordir_tara
#PBS -j oe
# Muse-Spark TP16 x PP2 pilot, one wave: (1024, 256) c 1 4 16. 8 serving + client + 1 spare.
TARA_NORTH=/vast/draco/tara/projects/Tara_Deployment/software/dynamo-disagg-alcf/tara_north
MUSE_SPARK=${TARA_NORTH}/multi_model/muse_spark
RUNS=/vast/draco/tara/projects/Tara_Deployment/software/testing/RUNS
RUN_TAG=tp16short
MNBT=8192
CELLS="1024:256:1 1024:256:4 1024:256:16"
CELL_TIMEOUT_S=10800
MAX_ATTEMPTS=3

JOB_NODEFILE=${RUNS}/${PBS_JOBID}.txt
cat "${PBS_NODEFILE}" > "${JOB_NODEFILE}"

CLIENT=$(hostname -s)
PARKED=
for attempt in $(seq 1 "${MAX_ATTEMPTS}"); do
    LOG=${RUNS}/qsub_${PBS_JOBID}_${RUN_TAG}_a${attempt}.log
    echo "=== attempt ${attempt}/${MAX_ATTEMPTS} client=${CLIENT} parked=${PARKED:-none} ==="
    cmd="export PBS_NODEFILE=${JOB_NODEFILE} RUN_TAG=${RUN_TAG} MNBT=${MNBT} CELLS='${CELLS}' CELL_TIMEOUT_S=${CELL_TIMEOUT_S} EXCLUDE_NODES='${PARKED}'; \
        bash ${MUSE_SPARK}/run_muse_spark_pilot_tp16pp2_N8.sh"
    if [ "${CLIENT}" = "$(hostname -s)" ]; then bash -c "${cmd}"; else ssh -n "${CLIENT}" "${cmd}"; fi 2>&1 | tee "${LOG}"
    RC=${PIPESTATUS[0]}
    # Retry only a dirty node; an OOM repeats itself.
    [ "${RC}" -eq 10 ] || break
    for d in $(grep '^DIRTY_NODE:' "${LOG}" | cut -d: -f2 | sort -u); do PARKED="${PARKED} ${d}"; done
    PARKED=${PARKED# }
    CLIENT=${PARKED%% *}
done
echo "=== exit_code=${RC} parked=${PARKED:-none} ==="
exit "${RC}"
