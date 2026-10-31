#!/bin/bash
# nccl-tests against the 1.5.2 env's wheel NCCL. Run on a compute node, no args.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env_vllm_0.32.1rc1_nixl_1.5.2.sh"
source ~/proxies.sh
set -e
NCCL_WHL=$(python -c 'import site; print(site.getsitepackages()[0])')/nvidia/nccl
NT=$SOFT_ROOT/repositories/nccl-tests
NH=$NT/nccl_wheel_2.30.7
mkdir -p $NT
exec > >(tee $NT/build_cu13.0_nccl2.30.7.log) 2>&1
git clone https://github.com/NVIDIA/nccl-tests.git $NT/src
git -C $NT/src rev-parse HEAD
mkdir -p $NH/lib
ln -s $NCCL_WHL/include $NH/include
# Link name for the build only; runs load the wheel's libnccl.so.2.
ln -s $NCCL_WHL/lib/libnccl.so.2 $NH/lib/libnccl.so
make -C $NT/src -j 32 MPI=1 MPI_HOME=$CRAY_MPICH_DIR CUDA_HOME=$CUDA_HOME NCCL_HOME=$NH \
  NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90" BUILDDIR=$NT/build_cu13.0_nccl2.30.7
