#!/bin/bash
# Source with no positional args: conda's activate reads $1.
# No CC/CXX: the JITs use gcc. Torch cu132 loads its own CUDA libs.
SOFT_ROOT=/vast/draco/tara/projects/Tara_Deployment/software
source $SOFT_ROOT/miniforge3/bin/activate
conda activate $SOFT_ROOT/envs/conda_envs/vllm_0.32.1rc1_disaggpp_nixl_1.5.2_python_3.12.12
module add gcc-native
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/25.9/cuda/13.0
export PATH=$CUDA_HOME/bin:$PATH
# libfabric's CUDA support needs libcudart.so (see get_env).
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/cudart_for_libfabric${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
# libfabric 2.7.0 (cxi writedata); must precede the libfabric/2.3.1 module.
export LD_LIBRARY_PATH=$SOFT_ROOT/repositories/libfabric_v2.7.0/v2.7.0_cu13.0/lib:$LD_LIBRARY_PATH
