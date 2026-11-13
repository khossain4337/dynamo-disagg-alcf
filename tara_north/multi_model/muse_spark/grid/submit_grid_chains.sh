#!/bin/bash
# Submits the grid as two afterany chains, router and P:D: one job of each runs at a time (38 nodes).
# Run once. The stamp file blocks a second run and logs "jobid chain file".
set -u
RUNS=/vast/draco/tara/projects/Tara_Deployment/software/testing/RUNS
STAMP=${RUNS}/grid_chains_submitted.txt
[ -e "${STAMP}" ] && { echo "already submitted; see ${STAMP}" >&2; exit 1; }
cd "$(dirname "$0")" || exit 1

chain() {
    local name=$1 prev= id f
    shift
    for f in "$@"; do
        id=$(qsub ${prev:+-W depend=afterany:${prev}} "${f}") || { echo "FAILED ${f}; chain ${name} stopped" >&2; return 1; }
        echo "${id} ${name} ${f}" | tee -a "${STAMP}"
        prev=${id}
    done
}

RT=(
    qsub_muse_spark_grid_rt_prefill_N16_osl8k.sh
    qsub_muse_spark_grid_rt_prefill_N16_osl16k.sh
    qsub_muse_spark_grid_rt_prefill_N16_osl32k.sh
    qsub_muse_spark_grid_rt_prefill_N16_osl64k.sh
    qsub_muse_spark_grid_rt_prefill_N16_osl128k-256k.sh
    qsub_muse_spark_grid_rt_decode_N16_osl8k-16k_isl8k-16k.sh
    qsub_muse_spark_grid_rt_decode_N16_osl32k_isl8k.sh
    qsub_muse_spark_grid_rt_decode_N16_osl32k_isl16k.sh
    qsub_muse_spark_grid_rt_decode_N16_osl32k_isl32k.sh
    qsub_muse_spark_grid_rt_decode_N16_osl64k_isl8k.sh
    qsub_muse_spark_grid_rt_decode_N16_osl64k_isl16k.sh
    qsub_muse_spark_grid_rt_decode_N16_osl64k_isl32k.sh
    qsub_muse_spark_grid_rt_decode_N16_osl64k_isl64k.sh
    qsub_muse_spark_grid_rt_decode_N16_osl128k_isl8k.sh
    qsub_muse_spark_grid_rt_decode_N16_osl128k_isl16k.sh
    qsub_muse_spark_grid_rt_decode_N16_osl128k_isl32k.sh
    qsub_muse_spark_grid_rt_decode_N16_osl128k_isl64k.sh
    qsub_muse_spark_grid_rt_decode_N16_osl128k_isl128k.sh
    qsub_muse_spark_grid_rt_decode_N16_osl256k_isl8k.sh
    qsub_muse_spark_grid_rt_decode_N16_osl256k_isl16k.sh
    qsub_muse_spark_grid_rt_decode_N16_osl256k_isl32k.sh
    qsub_muse_spark_grid_rt_decode_N16_osl256k_isl64k.sh
    qsub_muse_spark_grid_rt_decode_N16_osl256k_isl128k.sh
    qsub_muse_spark_grid_rt_decode_N16_osl256k_isl256k.sh
    qsub_muse_spark_grid_rt_decode_N16_osl512k_isl8k.sh
    qsub_muse_spark_grid_rt_decode_N16_osl512k_isl16k.sh
    qsub_muse_spark_grid_rt_decode_N16_osl512k_isl32k.sh
    qsub_muse_spark_grid_rt_decode_N16_osl512k_isl64k.sh
    qsub_muse_spark_grid_rt_decode_N16_osl512k_isl128k.sh
    qsub_muse_spark_grid_rt_decode_N16_osl512k_isl256k.sh
)
PD=(
    qsub_muse_spark_grid_pd_prefill_N16_osl8k.sh
    qsub_muse_spark_grid_pd_prefill_N16_osl16k.sh
    qsub_muse_spark_grid_pd_prefill_N16_osl32k.sh
    qsub_muse_spark_grid_pd_prefill_N16_osl64k.sh
    qsub_muse_spark_grid_pd_prefill_N16_osl128k-256k.sh
    qsub_muse_spark_grid_pd_decode_N16_osl8k-16k_isl8k.sh
    qsub_muse_spark_grid_pd_decode_N16_osl16k_isl16k.sh
    qsub_muse_spark_grid_pd_decode_N16_osl32k_isl8k.sh
    qsub_muse_spark_grid_pd_decode_N16_osl32k_isl16k.sh
    qsub_muse_spark_grid_pd_decode_N16_osl32k_isl32k.sh
    qsub_muse_spark_grid_pd_decode_N16_osl64k_isl8k.sh
    qsub_muse_spark_grid_pd_decode_N16_osl64k_isl16k.sh
    qsub_muse_spark_grid_pd_decode_N16_osl64k_isl32k.sh
    qsub_muse_spark_grid_pd_decode_N16_osl64k_isl64k.sh
    qsub_muse_spark_grid_pd_decode_N16_osl128k_isl8k.sh
    qsub_muse_spark_grid_pd_decode_N16_osl128k_isl16k.sh
    qsub_muse_spark_grid_pd_decode_N16_osl128k_isl32k.sh
    qsub_muse_spark_grid_pd_decode_N16_osl128k_isl64k.sh
    qsub_muse_spark_grid_pd_decode_N16_osl128k_isl128k.sh
    qsub_muse_spark_grid_pd_decode_N16_osl256k_isl8k.sh
    qsub_muse_spark_grid_pd_decode_N16_osl256k_isl16k.sh
    qsub_muse_spark_grid_pd_decode_N16_osl256k_isl32k.sh
    qsub_muse_spark_grid_pd_decode_N16_osl256k_isl64k.sh
    qsub_muse_spark_grid_pd_decode_N16_osl256k_isl128k.sh
    qsub_muse_spark_grid_pd_decode_N16_osl256k_isl256k.sh
    qsub_muse_spark_grid_pd_decode_N16_osl512k_isl8k.sh
    qsub_muse_spark_grid_pd_decode_N16_osl512k_isl16k.sh
    qsub_muse_spark_grid_pd_decode_N16_osl512k_isl32k-64k.sh
    qsub_muse_spark_grid_pd_decode_N16_osl512k_isl128k-256k.sh
)

chain router "${RT[@]}"
chain pd "${PD[@]}"
wc -l < "${STAMP}"
