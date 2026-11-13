#!/bin/bash
#PBS -l select=19 -l place=scatter:excl
#PBS -l walltime=54:00:00
#PBS -A inference_service
#PBS -q workq
#PBS -k doe
#PBS -N grd256_8
#PBS -o /vast/draco/tara/projects/Tara_Deployment/software/testing/RUNS/outdir_tara
#PBS -e /vast/draco/tara/projects/Tara_Deployment/software/testing/RUNS/errordir_tara
#PBS -j oe
# Grid decode-heavy (8k, 256k) (DECISIONS -11-13), router arm, 2 x TP4 x PP8 + vllm-router, EP off. 16 serving + 1 + client + 1 spare.
TARA_NORTH=/vast/draco/tara/projects/Tara_Deployment/software/dynamo-disagg-alcf/tara_north
MUSE_SPARK=${TARA_NORTH}/multi_model/muse_spark
RUNS=/vast/draco/tara/projects/Tara_Deployment/software/testing/RUNS
EP=0
RUN_TAG=grid-rt-decode-osl256k-isl8k
CELLS=(8192:262144:4 8192:262144:8 8192:262144:16 8192:262144:32 8192:262144:64)
PROMPTS_PER_STREAM=2
MAX_ATTEMPTS=3

# PBS_NODEFILE is local to the mother superior; staged so any node can launch.
JOB_NODEFILE=${RUNS}/${PBS_JOBID}.txt
cat "${PBS_NODEFILE}" > "${JOB_NODEFILE}"

# A dirty node is parked, never cleaned: the first becomes the client, later ones sit out.
CLIENT=$(hostname -s)
PARKED=
TODO="${CELLS[*]}"
for attempt in $(seq 1 "${MAX_ATTEMPTS}"); do
    LOG=${RUNS}/qsub_${PBS_JOBID}_${RUN_TAG}_a${attempt}.log
    echo "=== attempt ${attempt}/${MAX_ATTEMPTS} client=${CLIENT} parked=${PARKED:-none} cells=${TODO} ==="
    cmd="export PBS_NODEFILE=${JOB_NODEFILE} EP=${EP} RUN_TAG=${RUN_TAG} CELLS='${TODO}' \
        PROMPTS_PER_STREAM=${PROMPTS_PER_STREAM} EXCLUDE_NODES='${PARKED}'; \
        bash ${MUSE_SPARK}/run_muse_spark_router_sweep_tp4pp8_N16.sh"
    if [ "${CLIENT}" = "$(hostname -s)" ]; then bash -c "${cmd}"; else ssh -n "${CLIENT}" "${cmd}"; fi 2>&1 | tee "${LOG}"
    RC=${PIPESTATUS[0]}
    # Retry only a dirty node (10) or a server death (30); anything else repeats itself.
    case "${RC}" in 10|30) ;; *) break ;; esac
    if [ "${RC}" -eq 30 ]; then
        _dir=$(grep -m1 '^RUN_DIR:' "${LOG}" | cut -d: -f2)
        _ok=$(awk -F'\t' 'NR>1 && $4=="ok" {print $1":"$2":"$3}' "${_dir}/bench_status.tsv" 2>/dev/null)
        _rem=
        for x in ${TODO}; do printf '%s\n' "${_ok}" | grep -qx "${x}" || _rem="${_rem} ${x}"; done
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
