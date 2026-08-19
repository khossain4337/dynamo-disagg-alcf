#!/bin/bash
#
# Minimal env for reproducing the LIBFABRIC/CXI topology bug -- NOT
# dependent on any prior run_pd_full_test_N2_R1.sh output. Two real steps:
# conda activation, and the one fixup confirmed load-bearing by the A/B/D
# ldd comparison (libcudart.so.13 unresolved without it, resolved with it,
# and that's the only thing that changed between repro failing and passing).
#
# The fixup itself is several lines because NVHPC's CUDA_HOME
# auto-detection is genuinely broken on this system (lands in the
# compiler-only tree, not the real CUDA toolkit dir) -- that's a confirmed
# fact from the original handoff doc, not something we're adding.
#
# Usage: source this, then run the repro directly:
#   source env_min.sh && python3 repro_libfabric_topology.py

source /vast/draco/tara/projects/Tara_Deployment/software/miniforge3/bin/activate
conda activate /vast/draco/tara/projects/Tara_Deployment/software/envs/conda_envs/vllm_0.27.1_nixl_1.4.0_python_3.12.12

NVCXX=$(which nvc++ 2>/dev/null || true)
if [ -n "${NVCXX}" ]; then
    export CC="${NVCXX}"
    export CXX="${NVCXX}"
    export CUDAHOSTCXX="${NVCXX}"
    NVHPC_COMPILERS_DIR=$(dirname "$(dirname "${NVCXX}")")   # .../<ver>/compilers
    NVHPC_VER_ROOT=$(dirname "${NVHPC_COMPILERS_DIR}")        # .../<ver>
    CUDA_VER_DIR=$(find "${NVHPC_VER_ROOT}/cuda" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort -V | tail -1)
    REAL_CUDA_LIB_DIR=""
    if [ -n "${CUDA_VER_DIR}" ]; then
        REAL_CUDA_LIB_DIR=$(find "${CUDA_VER_DIR}" -name "libcudart.so" -printf '%h\n' 2>/dev/null | head -1)
    fi
    if [ -n "${REAL_CUDA_LIB_DIR}" ]; then
        export CUDA_HOME="${CUDA_VER_DIR}"
        export LIBRARY_PATH="${REAL_CUDA_LIB_DIR}:${LIBRARY_PATH:-}"
        export LD_LIBRARY_PATH="${REAL_CUDA_LIB_DIR}:${LD_LIBRARY_PATH:-}"
        TARGET_ROOT=$(dirname "${REAL_CUDA_LIB_DIR}")
        [ -d "${TARGET_ROOT}/include" ] && export CPATH="${TARGET_ROOT}/include:${CPATH:-}"
        CUDA_VER=$(basename "${CUDA_VER_DIR}")
        ARCH_NAME=$(basename "${TARGET_ROOT}")
        MATH_LIBS_TARGET="${NVHPC_VER_ROOT}/math_libs/${CUDA_VER}/targets/${ARCH_NAME}"
        if [ -d "${MATH_LIBS_TARGET}/lib" ]; then
            export LIBRARY_PATH="${MATH_LIBS_TARGET}/lib:${LIBRARY_PATH}"
            export LD_LIBRARY_PATH="${MATH_LIBS_TARGET}/lib:${LD_LIBRARY_PATH}"
        fi
    else
        echo "WARNING: couldn't resolve real CUDA lib dir -- libcudart.so.13 will likely be unresolved." >&2
    fi
else
    echo "WARNING: nvc++ not found on PATH -- CUDA_HOME fixup skipped." >&2
fi
