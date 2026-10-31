#!/bin/bash
# Source with no positional args: conda's activate reads $1.
# No CC/CXX and no LD_LIBRARY_PATH: the JITs use gcc, torch cu132 loads its own CUDA libs.
SOFT_ROOT=/vast/draco/tara/projects/Tara_Deployment/software
source $SOFT_ROOT/miniforge3/bin/activate
conda activate $SOFT_ROOT/envs/conda_envs/vllm_0.32.1rc1_nixl_1.5.2_python_3.12.12
module add gcc-native
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/25.9/cuda/13.0
export PATH=$CUDA_HOME/bin:$PATH
