# shellcheck shell=bash
# =============================================================================
# emit_common_env.sh -- generate the per-run common_env.sh both arms source.
#
# SOURCE this file, do not execute it. It defines one function:
#
#     emit_common_env <dest-path>
#
# and writes nothing until that function is called.
#
# WHY THIS IS A SEPARATE FILE. The Figure 2 comparison has two launchers --
# run_pd_nemotron_1p1d_N2.sh (disagg, mpiexec, NIXL) and run_colocated_N1.sh
# (one node, plain vllm serve). The comparison is only meaningful if the two
# arms run in the SAME environment, and the way that guarantee dies is by
# copy-paste: one launcher gets a CPATH fix or a TRITON_CC candidate and the
# other does not, and six weeks later the difference reads as a result. This
# is the same drift that the inline-duplicate-of-env_for_libfabric_topology_error
# comment below is already complaining about; the fix is the same one. One copy.
#
# REQUIRED IN THE CALLER'S SCOPE (all read at generation time):
#   NO_PROXY_LIST   comma list for NO_PROXY/no_proxy
#   ENV_SCRIPT      absolute path to env_for_libfabric_topology_error.sh
#   UCX_LINES       zero or more `export UCX_*=...` lines, or ""
#   GPU_PIN_LINE    `export CUDA_VISIBLE_DEVICES=N`, or ""
#
# THE TWO HEREDOCS DIFFER ON PURPOSE. The first is unquoted (<<EOF) so the four
# variables above are baked in here, on the launcher's node. The second is
# QUOTED (<<'EOF') so its paths resolve when common_env.sh RUNS on the compute
# node. Preserve that distinction and the `\$`/`\${` escapes with it -- an
# unescaped $ in the first heredoc silently evaluates a login-node value into a
# compute-node file.
# =============================================================================

emit_common_env() {
    local _dest=${1:?emit_common_env: destination path required}

cat > ${_dest} <<EOF
export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
export NO_PROXY=${NO_PROXY_LIST}
export no_proxy=${NO_PROXY_LIST}

# Conda activation and the NVHPC CUDA_HOME fixup both come from
# env_for_libfabric_topology_error.sh in the repo -- the SAME file, at the same
# path, that the working NIXL benchmark sources. Previously this script carried
# its own inline duplicate of that logic, which is exactly how the two drift
# apart and how a run that "uses the same environment" quietly stops doing so.
# Sourced from SCRIPT_DIR rather than a staged copy for the same reason.
#
# set +u across the source for the same reason the benchmark does it: conda's
# activation machinery reads unset variables, which is fatal under nounset.
set +u
source ${ENV_SCRIPT}
set -u

export HF_TOKEN=\$(cat ~/.hf_token)
export PYTHONNOUSERSITE=1
export HF_HOME=/vast/draco/tara/projects/Tara_Deployment/software/model-weights
export HF_DATASETS_CACHE=\${HF_HOME}
export HF_MODULES_CACHE=\${HF_HOME}
export RAY_TMPDIR=/tmp
export TMPDIR=/tmp
export VLLM_LOGGING_LEVEL=INFO
${UCX_LINES}
${GPU_PIN_LINE}
EOF

# Appended with a QUOTED heredoc delimiter ('EOF') so the paths below are
# resolved when common_env.sh actually RUNS on the compute node, not now on
# whichever node this launcher script happens to be running on -- login and
# compute node module environments aren't guaranteed to match on Cray systems.
#
# Everything the old inline block did -- CC/CXX/CUDAHOSTCXX=nvc++, CUDA_HOME,
# LIBRARY_PATH, LD_LIBRARY_PATH, CPATH and the math_libs LIB dir -- now comes
# from env_for_libfabric_topology_error.sh, sourced above. This appendix keeps
# the ONE thing that file does not do.
cat >> ${_dest} <<'EOF'

# CONFIRMED (not guessed) from an actual "curandStatePhilox4_32_10_t
# undefined" failure: the CUDA math-library HEADERS -- curand, and by strong
# implication cublas/cusparse/cusolver/cufft -- live in a SEPARATE sibling
# tree, math_libs/<ver>/targets/<arch>/include, not inside cuda/<ver>/ at all.
# env_for_libfabric_topology_error.sh puts the math_libs LIB dir on
# LIBRARY_PATH/LD_LIBRARY_PATH but never adds its INCLUDE dir to CPATH, so a
# JIT compile that needs curand_kernel.h still fails without this block.
#
# Derived from CUDA_HOME (which that file exports) rather than from its
# internal shell variables, so this stays correct if it is ever refactored.
if [ -n "${CUDA_HOME:-}" ]; then
    _REAL_CUDA_LIB_DIR=$(find "${CUDA_HOME}" -name "libcudart.so" -printf '%h\n' 2>/dev/null | head -1)
    if [ -n "${_REAL_CUDA_LIB_DIR}" ]; then
        _TARGET_ROOT=$(dirname "${_REAL_CUDA_LIB_DIR}")           # .../targets/<arch>
        _NVHPC_VER_ROOT=$(dirname "$(dirname "${CUDA_HOME}")")    # .../<ver>
        _MATH_INC="${_NVHPC_VER_ROOT}/math_libs/$(basename "${CUDA_HOME}")/targets/$(basename "${_TARGET_ROOT}")/include"
        if [ -d "${_MATH_INC}" ]; then
            export CPATH="${_MATH_INC}:${CPATH:-}"
        else
            echo "WARNING: math_libs include dir not found at ${_MATH_INC}" >&2
            echo "         A JIT build needing curand_kernel.h will fail." >&2
        fi
    fi
fi

# TRITON NEEDS A GCC-COMPATIBLE $CC AND env_for_libfabric_topology_error.sh
# HANDS IT nvc++. This block is the difference between a server that starts and
# one that grinds until the health check gives up. Read before removing.
#
# That file (lines 22-24) exports CC=CXX=CUDAHOSTCXX=nvc++. Right for the
# nvcc/curand path directly above, wrong for Triton, which builds its
# cuda_utils.c helper by invoking $CC with GCC flags. Measured, not inferred:
#
#   $ $CC   /tmp/t.c -O3 -shared -fPIC -Wno-psabi -o /tmp/t1.so
#   nvc++-Error-Unknown switch: -Wno-psabi
#   $ /opt/cray/pe/gcc-native/14/bin/gcc  <same flags>
#   (silent)
#
# WHY THIS STAYED HIDDEN FOR SO LONG. Two different code paths:
#   cold cache -> vLLM compiles from scratch; Triton's JIT reuses a
#                 cuda_utils.so already cached in ~/.triton, so $CC is never
#                 invoked and nvc++ is never tested.
#   warm cache -> vLLM LOADS the binary torch_aot_compile artifact, which
#                 rebuilds the launcher stub in a fresh /tmp dir with no
#                 ~/.triton reuse. $CC runs for real. nvc++ fails.
# The cache key covers model + TP + compilation config, so every run that
# changed any of those was cold. The bug only fires on the second run of a
# byte-identical config -- i.e. success is what arms it. Diagnosed on the Qwen
# TP=4 rig; this script had never completed a run, so it had never written a
# cache, and would have hit it on its SECOND run -- after a ~240 GB weight load.
#
# AND IT DOES NOT LOOK LIKE AN ERROR. vLLM catches the failure, logs it at
# WARNING (compilation/decorators.py:321), and falls back to a full recompile.
# The workers stop servicing the shm_broadcast ring while they grind, so the
# operator sees EngineCore repeating "No available shared memory broadcast
# block found in 60 seconds" until wait_healthy times out: no traceback, no
# OOM, no port conflict, and a node that looks perfectly clean.
#
# CC ONLY. CXX and CUDAHOSTCXX stay nvc++ -- CUDAHOSTCXX is the CUDA host
# compiler the curand block above exists to serve, and changing it undoes that.
#
# Do NOT solve this with `module load gcc-native`: that modulefile is
# family("compiler") and does load("PrgEnv-gnu"), so it evicts the NVHPC module
# and tears down the CUDA_HOME/LIBRARY_PATH/CPATH setup this whole file depends
# on. We want one binary, not a programming-environment switch.
#
# Candidates in order: explicit TRITON_CC override, the confirmed Cray gcc, then
# whatever `gcc` resolves to on PATH (covers a future gcc-native/15).
_TRITON_CC=""
for _cand in "${TRITON_CC:-}" /opt/cray/pe/gcc-native/14/bin/gcc gcc; do
    [ -n "${_cand}" ] || continue
    if _resolved=$(command -v "${_cand}" 2>/dev/null) && [ -n "${_resolved}" ]; then
        _TRITON_CC="${_resolved}"
        break
    fi
done
if [ -n "${_TRITON_CC}" ]; then
    export CC="${_TRITON_CC}"
    # Printed on purpose, and it lands in mpiexec.log because common_env.sh is
    # sourced before launch_role.sh redirects to p.log/d.log. Silent success is
    # precisely what made this cost an afternoon; one line per rank is cheap.
    echo "Triton CC override: CC=${CC} (CXX/CUDAHOSTCXX left at nvc++)"
else
    echo "WARNING: no gcc found for Triton; leaving CC=${CC:-<unset>}." >&2
    echo "         If that is nvc++, Triton's cuda_utils build will fail, vLLM" >&2
    echo "         will silently fall back to recompiling, and the engine will" >&2
    echo "         hang during startup instead of reporting an error." >&2
    echo "         Set TRITON_CC=/path/to/gcc to fix." >&2
fi
EOF
}
