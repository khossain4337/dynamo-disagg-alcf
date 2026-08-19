#!/bin/bash -x
set -uo pipefail
#
ml add gcc-native
source /vast/draco/tara/projects/Tara_Deployment/software/miniforge3/bin/activate
conda activate /vast/draco/tara/projects/Tara_Deployment/software/envs/conda_envs/vllm_0.27.1_nixl_1.4.0_python_3.12.12
#
run_test() {
    local desc="$1"
    shift
    echo "=== ${desc} ==="
    "$@"
    local rc=$?
    if [ ${rc} -eq 0 ]; then
        echo "--- PASS: ${desc} ---"
    elif [ ${rc} -eq 139 ]; then
        echo "--- SEGFAULT (signal 11): ${desc} ---"
    else
        echo "--- FAIL (exit ${rc}): ${desc} ---"
    fi
}
#
ENV_SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
#
echo "=== pip packages ==="
pip list | grep -i nixl
#
echo "=== torch cuda version ==="
TORCH_CUDA=$(python3 -c "import torch; print(torch.version.cuda)")
echo "${TORCH_CUDA}"
#
CUDA_MAJOR=$(echo "${TORCH_CUDA}" | cut -d. -f1)
#
NIXL_UCX_DIR=$(find "${ENV_SITE_PACKAGES}" -maxdepth 2 -type d -path "*nixl_cu${CUDA_MAJOR}*.libs/ucx" 2>/dev/null | head -n1)
#
if [ -z "${NIXL_UCX_DIR}" ]; then
    echo "Could not locate nixl_cu${CUDA_MAJOR}.libs/ucx directory under ${ENV_SITE_PACKAGES}"
    exit 1
fi
#
echo "=== Using UCX_MODULE_DIR: ${NIXL_UCX_DIR} ==="
#
export UCX_TLS=cuda_copy,cuda_ipc,sm,tcp,self
export UCX_MODULE_DIR="${NIXL_UCX_DIR}"
#
run_test "Baseline NIXL agent init (UCX_TLS + UCX_MODULE_DIR set)" \
    python3 -c "import nixl; a = nixl.nixl_agent('agent1')"
#
run_test "NIXL agent init with CUDA_VISIBLE_DEVICES= isolation" \
    env CUDA_VISIBLE_DEVICES= python3 -c "import nixl; a = nixl.nixl_agent('agent1')"
