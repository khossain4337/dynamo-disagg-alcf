#!/bin/bash -x
set -euo pipefail
# Time Stamp
tstamp() {
     date +"%Y-%m-%d-%H%M%S"
}
## Proxies to clone from a compute node
export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128
#
SOFT_ROOT=/vast/draco/tara/projects/Tara_Deployment/software
VENV_ROOT=${SOFT_ROOT}/envs/pip_venvs
VENV_NAME=aisimulate_0.12.0
VENV_DIR=${VENV_ROOT}/${VENV_NAME}
CONDA_ENV_INSTALL_DIR=${SOFT_ROOT}/envs/conda_envs
CONDA_ENV_NAME=vllm_0.27.1_nixl_1.4.0_python_3.12.12
TMPDIR_FOR_ENVS=${SOFT_ROOT}/envs/tmpdir_for_envs
MINIFORGE3_ROOT=${SOFT_ROOT}/miniforge3
ENVPREFIX=$CONDA_ENV_INSTALL_DIR/$CONDA_ENV_NAME

MODEL_NAME=/vast/draco/tara/projects/Tara_Deployment/software/model-weights/hub/models--thinkingmachines--Inkling-Small
MODEL_PATH_FOR_CONFIG=${MODEL_NAME}/snapshots/8cc5877b44d343f88b92086aa1fb72897950f06a

#MODEL_PATH_FOR_CONFIG=/vast/draco/tara/projects/Tara_Deployment/software/model-weights/hub/models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-BF16/snapshots/2dc98e2afe4face0e4ce40972a915c45368bd34a

source $MINIFORGE3_ROOT/bin/activate
conda activate ${ENVPREFIX}

source ${VENV_DIR}/bin/activate

python - "$MODEL_PATH_FOR_CONFIG" <<'EOF'
import sys
from aiconfigurator.sdk.utils import get_model_config_from_model_path
print(get_model_config_from_model_path(sys.argv[1]))
EOF
grep -A3 architectures "$MODEL_PATH_FOR_CONFIG/config.json"

#AICONFIGURATOR_LOG_LEVEL=DEBUG aiconfigurator cli support --model-path ${MODEL_PATH_FOR_CONFIG} \
#    --system all --backend all

#aiconfigurator cli support --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 \
#    --system h200_sxm --backend vllm


