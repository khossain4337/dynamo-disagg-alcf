#!/bin/bash
#
source /vast/draco/tara/projects/Tara_Deployment/software/miniforge3/bin/activate
conda activate /vast/draco/tara/projects/Tara_Deployment/software/envs/conda_envs/vllm_0.27.1_nixl_1.4.0_python_3.12.12

echo "### pkg-config libfabric (cflags + libs) ###"
pkg-config --cflags --libs libfabric 2>/dev/null || echo "(pkg-config has no libfabric)"

echo ""
echo "### rdma/fabric.h locations ###"
find /usr/include /usr/local/include "$CONDA_PREFIX" -name 'fabric.h' -path '*rdma*' 2>/dev/null | head

echo ""
echo "### fi_domain.h locations ###"
find /usr/include /usr/local/include "$CONDA_PREFIX" -name 'fi_domain.h' -path '*rdma*' 2>/dev/null | head

echo ""
echo "### libfabric.so location + version ###"
find /usr/lib* /usr/local/lib* "$CONDA_PREFIX/lib" -name 'libfabric.so*' 2>/dev/null | head
fi_info --version 2>/dev/null | head -3
