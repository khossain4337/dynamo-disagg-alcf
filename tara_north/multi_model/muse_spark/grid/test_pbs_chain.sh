#!/bin/bash
# Three 1-node jobs chained with afterany; the middle one fails on purpose.
# Pass: all three end in state F, the second with Exit_status 1, the third with 0.
# Usage: bash test_pbs_chain.sh        (submit)
#        bash test_pbs_chain.sh check  (after ~3 min)
RUNS=/vast/draco/tara/projects/Tara_Deployment/software/testing/RUNS
IDS=${RUNS}/chaintest_ids.txt
if [ "$1" = check ]; then
    for j in $(cat "${IDS}"); do qstat -xf "$j" | grep -E "Job Id|job_state|Exit_status"; done
    exit
fi
Q="-l select=1 -l walltime=00:05:00 -A inference_service -q workq -j oe -o ${RUNS}/outdir_tara -N chaintest"
a=$(qsub $Q -- /bin/sleep 60) || exit 1
b=$(qsub $Q -W depend=afterany:$a -- /bin/false) || exit 1
c=$(qsub $Q -W depend=afterany:$b -- /bin/sleep 30) || exit 1
echo "$a $b $c" | tee "${IDS}"
qstat "$a" "$b" "$c"
