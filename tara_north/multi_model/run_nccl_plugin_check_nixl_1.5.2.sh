#!/bin/bash
# HANDOFF item 2C: aws-ofi-nccl plugin vs socket, same 2 nodes. Run inside the allocation, no args.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env_vllm_0.32.1rc1_nixl_1.5.2.sh"
SP=$(python -c 'import site; print(site.getsitepackages()[0])')
B=$SOFT_ROOT/repositories/nccl-tests/build_cu13.0_nccl2.30.7
OFI=$SOFT_ROOT/repositories/aws_nccl_ofi_v1.22.0/v1.22.0_lf2.3.1_cu13.0
export LD_LIBRARY_PATH=$SP/nvidia/nccl/lib:$SP/nvidia/cu13/lib:$LD_LIBRARY_PATH
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,ENV,NET
export FI_MR_CACHE_MONITOR=userfaultfd FI_CXI_DISABLE_HOST_REGISTER=1 FI_CXI_DEFAULT_CQ_SIZE=131072 \
  FI_CXI_RDZV_PROTO=alt_read FI_CXI_RDZV_EAGER_SIZE=0 FI_CXI_RDZV_THRESHOLD=0 FI_CXI_RDZV_GET_MIN=0 \
  FI_CXI_DEFAULT_TX_SIZE=2048
export NCCL_CROSS_NIC=1 NCCL_NET_GDR_LEVEL=PHB NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
OUT=$SOFT_ROOT/testing/RUNS/nccl_plugin_check_$(date +%Y%m%d-%H%M%S); mkdir -p $OUT; echo $OUT
# One mpiexec per arm: both nodes need the same PALS VNI.
for t in all_reduce all_gather reduce_scatter alltoall; do
  for arm in ofi socket; do
    [ $arm = ofi ] && P=$OFI/lib/libnccl-net-ofi.so || P=none
    NCCL_NET_PLUGIN=$P mpiexec -n 8 --ppn 4 --envall --cpu-bind list:0-71:72-143:144-215:216-287 \
      $B/${t}_perf -b 8 -e 4G -f 2 -g 1 -n 50 -w 10 > $OUT/${t}_$arm.log 2>&1
  done
done
for f in $OUT/*.log; do echo "== $(basename $f .log)"; grep -m1 'NCCL version' $f
  grep -c 'Loaded net plugin' $f; grep -m2 -E 'Using network|Selected provider' $f
  grep -E '^ +4294967296 |Avg bus bandwidth' $f; done | tee $OUT/summary.txt
