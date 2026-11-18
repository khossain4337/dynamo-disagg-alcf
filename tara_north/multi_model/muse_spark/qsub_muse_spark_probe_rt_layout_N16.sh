#!/bin/bash
#PBS -l select=19 -l place=scatter:excl
#PBS -l walltime=03:00:00
#PBS -A inference_service
#PBS -q workq
#PBS -k doe
#PBS -N rprobe64
#PBS -o /vast/draco/tara/projects/Tara_Deployment/software/testing/RUNS/outdir_tara
#PBS -e /vast/draco/tara/projects/Tara_Deployment/software/testing/RUNS/errordir_tara
#PBS -j oe
# Router per-token probe (DECISIONS -11-18a): two boots, teardown between. Boot 1 api servers 1, default
# KV layout; boot 2 the same + LBHNC. (56k, 16k, 8) crosses context 65,536 mid-decode.
TARA_NORTH=/vast/draco/tara/projects/Tara_Deployment/software/dynamo-disagg-alcf/tara_north
MUSE_SPARK=${TARA_NORTH}/multi_model/muse_spark
RUNS=/vast/draco/tara/projects/Tara_Deployment/software/testing/RUNS
EP=0
CELLS=(65536:8192:8 65536:8192:16 57344:16384:8)
PROMPTS_PER_STREAM=2
MAX_ATTEMPTS=3

JOB_NODEFILE=${RUNS}/${PBS_JOBID}.txt
cat "${PBS_NODEFILE}" > "${JOB_NODEFILE}"

CLIENT=$(hostname -s)
PARKED=
boot() {
    local RUN_TAG=$1 API=$2 KV=$3 TODO="${CELLS[*]}" RC
    for attempt in $(seq 1 "${MAX_ATTEMPTS}"); do
        LOG=${RUNS}/qsub_${PBS_JOBID}_${RUN_TAG}_a${attempt}.log
        echo "=== ${RUN_TAG} attempt ${attempt}/${MAX_ATTEMPTS} client=${CLIENT} parked=${PARKED:-none} cells=${TODO} ==="
        cmd="export PBS_NODEFILE=${JOB_NODEFILE} EP=${EP} RUN_TAG=${RUN_TAG} CELLS='${TODO}' \
            PROMPTS_PER_STREAM=${PROMPTS_PER_STREAM} EXCLUDE_NODES='${PARKED}' \
            API_SERVER_COUNT=${API} KV_LAYOUT=${KV}; \
            bash ${MUSE_SPARK}/run_muse_spark_router_probe_tp4pp8_N16.sh"
        if [ "${CLIENT}" = "$(hostname -s)" ]; then bash -c "${cmd}"; else ssh -n "${CLIENT}" "${cmd}"; fi 2>&1 | tee "${LOG}"
        RC=${PIPESTATUS[0]}
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
    echo "=== ${RUN_TAG} exit_code=${RC} parked=${PARKED:-none} unfinished=${TODO:-none} ==="
}
boot probe-rt-api1 1 ""
boot probe-rt-api1-lbhnc 1 LBHNC
