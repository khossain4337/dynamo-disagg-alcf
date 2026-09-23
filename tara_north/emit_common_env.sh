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
# NO OUTBOUND CALLS AT ENGINE START. Same argument as the node-local compiler
# caches below: a launch must not depend on anything off the node that it does
# not already have. The weights are wholly cached under HF_HOME on /vast, so
# every remote call here is a liveness check with nothing to fetch -- and one of
# them took an API server down. On 2026-09-23 huggingface_hub list_repo_tree
# (hf_api.py:4100 -> _pagination.py:36) raised
# httpcore.RemoteProtocolError: Server disconnected: the ALCF proxy dropped a
# paginated remote directory crawl. --api-server-count 16 is the amplifier --
# sixteen servers each crawling -- which is why it had never bitten before and
# is intermittent now. The proxy is configured correctly above and NO_PROXY_LIST
# covers every node; the call was genuinely external, so the fix is to not make
# it (DECISIONS_2026-09-23.md).
#
# If a model is ever NOT fully cached, these turn a slow download into a clean
# immediate failure -- which is the behaviour you want on a timed allocation.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Same class, not the same incident: an outbound usage POST at engine start, and
# a \$HOME write under ~/.config/vllm, which the "never write to \$HOME" rule
# below forbids on its own.
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export RAY_TMPDIR=/tmp
export TMPDIR=/tmp
# COMPILER CACHES MUST BE NODE-LOCAL. Unset, these default under \$HOME, which
# is a shared network filesystem here, and that combination kills a multi-rank
# launch outright: eight workers across two nodes JIT the same MoE kernels at
# the same moment during determine_available_memory(), Triton writes each cache
# entry as temp-file-then-rename, and on a network filesystem a rename can
# invalidate another rank's already-open handle server-side. The reader then
# gets OSError: [Errno 116] Stale file handle out of
# compiler.py:412 metadata_path.read_text(), one worker dies, and the engine
# takes the whole application down AFTER a full 8-minute weight load
# (2026-09-15, Inkling colocated, twice). On a local filesystem the inode
# survives the rename and the race is simply not expressible.
#
# This is also the standing "never write output to \$HOME" rule: a compiler
# cache is output. TMPDIR and RAY_TMPDIR above already establish /tmp as the
# node-local scratch on these nodes.
#
# Per-USER rather than per-run, deliberately: Triton keys cache entries by a
# hash that includes the Triton version and the kernel source, so reuse across
# runs is safe, and it buys a warm cache on relaunch. Each node has its own
# /tmp, so there is no cross-node contention to reuse INTO. Steady-state
# throughput is unaffected either way -- compilation lands in startup and in
# bench_arm.sh's warmup, never in the measured window.
export TRITON_CACHE_DIR=/tmp/triton_cache_\${USER}
export VLLM_CACHE_ROOT=/tmp/vllm_cache_\${USER}
mkdir -p "\${TRITON_CACHE_DIR}" "\${VLLM_CACHE_ROOT}" 2>/dev/null || true
# Expanded HERE, on the client node at emit time (this heredoc is unquoted), so
# a one-off VLLM_LOGGING_LEVEL=DEBUG in the launching shell reaches every rank.
# Written bare as INFO it did not: common_env.sh is sourced on each node AFTER
# the caller's environment, so the hardcoded value silently won and a DEBUG
# launch produced an INFO log (2026-09-15). INFO stays the default -- DEBUG is
# a diagnostic, never a measured run.
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}
# Engine-core ready timeout. vLLM defaults to 600 s (envs.py:27, :788); the API
# servers raise TimeoutError from core_client.py:653-666 when any rank has not
# sent its ready message by then. The Inkling colocated launch of 2026-09-16
# died there at 715 s with the weights ALREADY RESIDENT -- 532 GB of weights,
# the audio encoder profile and CUDA-graph capture do not fit in 600 s on this
# rig, and a head cannot come up until BOTH DP ranks have registered, so the
# slower rank sets the clock. Startup only: it changes no measured quantity and
# does not touch the KV pool, so unlike --gpu-memory-utilization it does not
# void a repeat. Expanded at emit time, so a one-off from the launching shell
# still wins.
export VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-2400}
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
