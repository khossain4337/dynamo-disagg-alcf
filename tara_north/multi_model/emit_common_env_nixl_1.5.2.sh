#!/bin/bash
# Writes the run's common_env.sh: every variable the vLLM ranks and the bench client see.
# Usage: bash emit_common_env_nixl_1.5.2.sh <dest> <no_proxy list> <NCCL_DEBUG_FILE>
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOFT_ROOT=/vast/draco/tara/projects/Tara_Deployment/software
P=http://proxy.alcf.anl.gov:3128
cat > "${1:?dest}" <<EOF
export HTTP_PROXY=$P HTTPS_PROXY=$P http_proxy=$P https_proxy=$P
export NO_PROXY=${2:?no_proxy} no_proxy=$2
set +u
# mpiexec and ssh start non-login shells with no 'module'; the env file needs it.
type module >/dev/null 2>&1 || source /etc/profile >/dev/null 2>&1
source $HERE/env_vllm_0.32.1rc1_nixl_1.5.2.sh
set -u
# HPC SDK keeps curand headers out of CUDA_HOME; FlashInfer's sampling JIT needs them.
export CPATH=/opt/nvidia/hpc_sdk/Linux_aarch64/25.9/math_libs/13.0/targets/sbsa-linux/include\${CPATH:+:\$CPATH}
export PYTHONNOUSERSITE=1
export HF_HOME=$SOFT_ROOT/model-weights HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export TMPDIR=/tmp TRITON_CACHE_DIR=/tmp/triton_cache_\${USER} VLLM_CACHE_ROOT=/tmp/vllm_cache_\${USER}
export FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer_\${USER}
mkdir -p "\${TRITON_CACHE_DIR}" "\${VLLM_CACHE_ROOT}" "\${FLASHINFER_WORKSPACE_BASE}"
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export NCCL_NET_PLUGIN=$SOFT_ROOT/repositories/aws_nccl_ofi_v1.22.0/v1.22.0_lf2.3.1_cu13.0/lib/libnccl-net-ofi.so
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,ENV,NET NCCL_DEBUG_FILE=${3:?nccl debug file}
# Tested set: run_nccl_plugin_check_nixl_1.5.2.sh.
export FI_MR_CACHE_MONITOR=userfaultfd FI_CXI_DISABLE_HOST_REGISTER=1 FI_CXI_DEFAULT_CQ_SIZE=131072 \\
  FI_CXI_RDZV_PROTO=alt_read FI_CXI_RDZV_EAGER_SIZE=0 FI_CXI_RDZV_THRESHOLD=0 FI_CXI_RDZV_GET_MIN=0 \\
  FI_CXI_DEFAULT_TX_SIZE=2048
export NCCL_CROSS_NIC=1 NCCL_NET_GDR_LEVEL=PHB NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3 GLOO_SOCKET_IFNAME=hsn0
EOF
