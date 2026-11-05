#!/bin/bash
# Checks the libfabric v2.7.0 build, then runs the cq_data_write reproducer.
# Compute node inside a PBS job, no args.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
S=/vast/draco/tara/projects/Tara_Deployment/software
P=$S/repositories/libfabric_v2.7.0/v2.7.0_cu13.0
OUT=$S/testing/RUNS/libfabric_write_error_v2.7.0/cq_data_write_$(date +%Y%m%d_%H%M%S)
# The libfabric/2.3.1 module prepends its own lib.
export LD_LIBRARY_PATH=$P/lib:$LD_LIBRARY_PATH
mkdir -p $OUT && cd $OUT
exec > >(tee $OUT/check.log) 2>&1

$P/bin/fi_info --version
ldd $P/lib/libfabric.so.1 | grep -E 'cxi|json|curl|not found'
mpiexec -n 1 --single-node-vni $P/bin/fi_info -p cxi | grep -E 'provider|domain'
$P/bin/fi_info -e -p cxi | grep -aoE 'FI_CXI_(DEFAULT_CQ_SIZE|DEFAULT_TX_SIZE|DISABLE_HOST_REGISTER|RDZV_EAGER_SIZE|RDZV_GET_MIN|RDZV_PROTO|RDZV_THRESHOLD|ENABLE_WRITEDATA)' | sort -u
FI_CXI_ENABLE_WRITEDATA=1 mpiexec -n 1 --single-node-vni $P/bin/fi_info -p cxi -v | grep -a cq_data_size | sort -u
mpiexec -n 1 --single-node-vni $P/bin/fi_info -p cxi -v | grep -a cq_data_size | sort -u

gcc -O2 -o cq_data_write $HERE/nixl_libfabric_write_reproducer/cq_data_write.c -I$P/include -L$P/lib -lfabric
mpiexec -n 1 --single-node-vni ./cq_data_write; echo rc=$?
FI_CXI_ENABLE_WRITEDATA=1 mpiexec -n 1 --single-node-vni ./cq_data_write; echo rc=$?
