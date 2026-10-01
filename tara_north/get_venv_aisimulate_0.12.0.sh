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

source $MINIFORGE3_ROOT/bin/activate

ENVPREFIX=$CONDA_ENV_INSTALL_DIR/$CONDA_ENV_NAME

conda activate ${ENVPREFIX}
echo "Conda is coming from $(which conda)"

## CHANGE HERE!!!
TMP_WORK=${VENV_ROOT}
cd $TMP_WORK

mkdir -p ${VENV_DIR}

module add gcc-native
export CXX=$(which g++)
export CC=$(which gcc)

python3 -m venv ${VENV_DIR} --system-site-packages
source ${VENV_DIR}/bin/activate

python3 -m pip install aisimulate==0.12.0
aisimulate --help
