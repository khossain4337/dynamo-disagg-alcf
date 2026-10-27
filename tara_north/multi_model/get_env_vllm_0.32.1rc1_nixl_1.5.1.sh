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
export https_proxy=http://proxy.alcf.anl.gov:3128
#
SOFT_ROOT=/vast/draco/tara/projects/Tara_Deployment/software
CONDA_ENV_INSTALL_DIR=${SOFT_ROOT}/envs/conda_envs
CONDA_ENV_NAME=vllm_0.32.1rc1_nixl_1.5.1_python_3.12.12
TMPDIR_FOR_ENVS=${SOFT_ROOT}/envs/tmpdir_for_envs
MINIFORGE3_ROOT=${SOFT_ROOT}/miniforge3
AWS_NCCL_OFI_CLONE_DIR=${SOFT_ROOT}/repositories/aws_nccl_ofi_v1.22.0
CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/25.9/cuda/13.0
LIBFABRIC_DIR=/opt/cray/libfabric/2.3.1
OFI_PLUGIN_DIR=${AWS_NCCL_OFI_CLONE_DIR}/v1.22.0_lf2.3.1_cu13.0
#
export CUDA_HOME LIBFABRIC_DIR OFI_PLUGIN_DIR
#
source $MINIFORGE3_ROOT/bin/activate

## CHANGE HERE!!!
TMP_WORK=${TMPDIR_FOR_ENVS}
cd $TMP_WORK
#
mkdir -p ${CONDA_ENV_NAME}
#
LOG_FILE=${TMP_WORK}/${CONDA_ENV_NAME}/${CONDA_ENV_NAME}_build-$(tstamp).log
#
touch ${LOG_FILE}
exec > >(tee ${LOG_FILE}) 2>&1
#
ENVPREFIX=$CONDA_ENV_INSTALL_DIR/$CONDA_ENV_NAME
CONDA_ENV_MANIFEST=${CONDA_ENV_INSTALL_DIR}/manifests/${CONDA_ENV_NAME}

rm -rf ${ENVPREFIX}
mkdir -p ${ENVPREFIX}

rm -rf ${CONDA_ENV_MANIFEST}
mkdir -p ${CONDA_ENV_MANIFEST}

export CONDA_PKGS_DIRS=${ENVPREFIX}/../.conda/pkgs
export PIP_CACHE_DIR=${ENVPREFIX}/../.pip
export UV_CACHE_DIR=${ENVPREFIX}/../.uv

echo "Creating Conda environment with Python 3.12.12"
conda create python=3.12.12 --prefix ${ENVPREFIX} --override-channels \
           --channel conda-forge \
           --strict-channel-priority \
           --yes

conda activate ${ENVPREFIX}
echo "Conda is coming from $(which conda)"


module add gcc-native
export CXX=$(which g++)
export CC=$(which gcc)

module -t list 

## Changing to the AWS_NCCL_OFI plugin clone dir
cd ${AWS_NCCL_OFI_CLONE_DIR}/aws-ofi-nccl
#git clone https://github.com/aws/aws-ofi-nccl.git
#git checkout v1.22.0
#git submodule sync && git submodule update --init --recursive
#
pip install conda-pack ipython uv
uv pip install "vllm==0.32.1rc1.dev562+g9e3db4887" \
    nixl==1.5.1 \
    aiperf==0.13.0 \
    aisimulate==0.13.0 \
    py-spy==0.4.2 \
    nvitop==1.7.2 \
    --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/9e3db48873b87a3c232f16991675d33abb49b72b
#
python -c "import vllm, torch; print(vllm.__version__, torch.__version__, torch.version.cuda, torch.cuda.nccl.version())"
#
# Building AWS_NCCL_OFI plugin
ls -d ${CUDA_HOME}
bash ${AWS_NCCL_OFI_CLONE_DIR}/aws-ofi-nccl/autogen.sh
CC=$(which gcc) CXX=$(which g++) ./configure --prefix=${OFI_PLUGIN_DIR} \
  --with-libfabric=${LIBFABRIC_DIR} --with-cuda=$CUDA_HOME \
  --disable-picky-compiler LDFLAGS="-Wl,-rpath,${LIBFABRIC_DIR}/lib64"
make -j 32 && make install
ldd $OFI_PLUGIN_DIR/lib/libnccl-net.so | grep -E 'fabric|hwloc'


echo ""
echo "Writing the package lists"
conda list > $CONDA_ENV_MANIFEST/${CONDA_ENV_NAME}_conda_env.list 2>&1
pip list > $CONDA_ENV_MANIFEST/${CONDA_ENV_NAME}_pip.list 2>&1
echo "Package list writing finished"

echo "Writing $CONDA_ENV_MANIFEST/${CONDA_ENV_NAME}_all.list"
cat $CONDA_ENV_MANIFEST/${CONDA_ENV_NAME}_conda_env.list | grep -v '^#' | grep 'pypi$' | perl -pe 's/^(\S+)\s+(\S+).*/$1==$2/' >  $CONDA_ENV_MANIFEST/${CONDA_ENV_NAME}_all.list

