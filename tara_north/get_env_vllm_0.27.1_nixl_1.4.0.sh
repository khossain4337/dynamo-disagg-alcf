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
CONDA_ENV_INSTALL_DIR=${SOFT_ROOT}/envs/conda_envs
CONDA_ENV_NAME=vllm_0.27.1_nixl_1.4.0_python_3.12.12
TMPDIR_FOR_ENVS=${SOFT_ROOT}/envs/tmpdir_for_envs
MINIFORGE3_ROOT=${SOFT_ROOT}/miniforge3

source $MINIFORGE3_ROOT/bin/activate

ENVPREFIX=$CONDA_ENV_INSTALL_DIR/$CONDA_ENV_NAME
CONDA_ENV_MANIFEST=${CONDA_ENV_INSTALL_DIR}/manifests/${CONDA_ENV_NAME}

rm -rf ${ENVPREFIX}
mkdir -p ${ENVPREFIX}

rm -rf ${CONDA_ENV_MANIFEST}
mkdir -p ${CONDA_ENV_MANIFEST}

export CONDA_PKGS_DIRS=${ENVPREFIX}/../.conda/pkgs
export PIP_CACHE_DIR=${ENVPREFIX}/../.pip

echo "Creating Conda environment with Python 3.12.12"
conda create python=3.12.12 --prefix ${ENVPREFIX} --override-channels \
           --channel conda-forge \
           --strict-channel-priority \
           --yes

conda activate ${ENVPREFIX}
echo "Conda is coming from $(which conda)"

## CHANGE HERE!!!
TMP_WORK=${TMPDIR_FOR_ENVS}
cd $TMP_WORK

mkdir -p ${CONDA_ENV_NAME}

LOG_FILE=${TMP_WORK}/${CONDA_ENV_NAME}/module-$(tstamp).log

touch ${LOG_FILE}

module add gcc-native
export CXX=$(which g++)
export CC=$(which gcc)

module -t list 2>&1 | tee ${LOG_FILE}

pip install conda-pack ipython uv
uv pip install vllm==0.27.1 --torch-backend auto
uv pip install nixl==1.4.0
uv pip install aiperf==0.12.0

echo ""
echo "Writing the package lists"
conda list > $CONDA_ENV_MANIFEST/${CONDA_ENV_NAME}_conda_env.list 2>&1
pip list > $CONDA_ENV_MANIFEST/${CONDA_ENV_NAME}_pip.list 2>&1
echo "Package list writing finished"

echo "Writing $CONDA_ENV_MANIFEST/${CONA_ENV_NAME}_all.list"
cat $CONDA_ENV_MANIFEST/${CONDA_ENV_NAME}_conda_env.list | grep -v '^#' | grep 'pypi$' | perl -pe 's/^(\S+)\s+(\S+).*/$1==$2/' >  $CONDA_ENV_MANIFEST/${CONDA_ENV_NAME}_all.list

