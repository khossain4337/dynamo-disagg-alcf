#!/bin/bash -x
#
# env_build_dynamo_vllm_1.3.0_dev1_06102026.sh
#
# Builds ai-dynamo v1.3.0-dev.1 from source into a fresh conda environment.
# Designed for bare metal, no sudo, no Docker.
# Target: RHEL 9.7, CUDA 13.2, 8x B200, miniforge3
#
# IDEMPOTENT: Each major step is guarded by a sentinel file.
# Re-running after a failure skips completed steps and resumes from
# the failure point. Delete a sentinel to force that step to re-run.
#
# SENTINELS live at: $ENVPREFIX/.build_sentinels/
#
# USAGE:
#   mkdir -p build_logs
#   bash env_build_dynamo_vllm_1.3.0_dev1_06102026.sh 2>&1 | tee build_logs/build_$(date +%Y%m%d_%H%M%S).log
#
set -euo pipefail

# ─── Helpers ─────────────────────────────────────────────────────────────────
tstamp() { date +"%Y-%m-%d %H:%M:%S"; }
log()    { echo "[$(tstamp)] $*"; }
die()    { echo "[$(tstamp)] FATAL: $*" >&2; exit 1; }

# ─── Configuration ────────────────────────────────────────────────────────────
REPO_DIR=/home/hossainm/software/repositories/dynamo_1.3.0_dev1/dynamo
CONDA_ENV_INSTALL_DIR=/home/hossainm/software/envs/conda_envs
CONDA_ENV_NAME=dynamo_vllm_1.3.0_dev1
ENVPREFIX=${CONDA_ENV_INSTALL_DIR}/${CONDA_ENV_NAME}
MINIFORGE=/home/hossainm/miniforge3

# Wheel output directory — for multi-node deployment
WHEEL_DIR=/home/hossainm/software/wheels/dynamo_1.3.0_dev1

# Rust toolchain — kept under ~/software/ for consistency.
# MUST be exported before rustup install AND before every cargo invocation.
# If unset, cargo falls back to ~/.cargo — split installation, hard to debug.
export CARGO_HOME=/home/hossainm/software/rust/cargo
export RUSTUP_HOME=/home/hossainm/software/rust/rustup

# Binary versions
ETCD_VER=v3.5.17
NATS_VER=v2.10.24
PROTOC_VER=29.3

# Sentinel files — presence means step completed successfully
SENTINEL_DIR=${ENVPREFIX}/.build_sentinels
SENTINEL_CONDA=${SENTINEL_DIR}/01_conda_created
SENTINEL_BUILDDEPS=${SENTINEL_DIR}/01b_builddeps_installed
SENTINEL_CARGO=${SENTINEL_DIR}/02_cargo_built
SENTINEL_MATURIN=${SENTINEL_DIR}/03_maturin_built
SENTINEL_DYNAMO=${SENTINEL_DIR}/04_dynamo_installed
SENTINEL_WHEEL=${SENTINEL_DIR}/05_wheel_built

# ─── Proxy ───────────────────────────────────────────────────────────────────
export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"

# ─── Verify repo exists ───────────────────────────────────────────────────────
[[ -d "$REPO_DIR" ]] || die "Repo not found at $REPO_DIR — clone first:
  git clone --branch v1.3.0-dev.1 https://github.com/ai-dynamo/dynamo.git \\
      $REPO_DIR"

# ─── STEP 0: Rust (user-level, outside conda) ────────────────────────────────
# Installs to $CARGO_HOME / $RUSTUP_HOME — intentionally outside conda,
# same pattern as miniforge3 itself.
mkdir -p "$CARGO_HOME" "$RUSTUP_HOME"

if [[ ! -f "$CARGO_HOME/bin/rustc" ]]; then
    log "=== STEP 0: Installing Rust via rustup ==="
    log "    CARGO_HOME  = $CARGO_HOME"
    log "    RUSTUP_HOME = $RUSTUP_HOME"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
else
    log "=== STEP 0: Rust already installed at $CARGO_HOME, skipping ==="
fi

source "$CARGO_HOME/env"
log "Rust:  $(rustc --version)"
log "Cargo: $(cargo --version)"
log "Toolchain pinned in repo: $(grep channel $REPO_DIR/rust-toolchain.toml)"

# ─── Conda setup ──────────────────────────────────────────────────────────────
export CONDA_PKGS_DIRS=${CONDA_ENV_INSTALL_DIR}/../.conda/pkgs
export PIP_CACHE_DIR=${CONDA_ENV_INSTALL_DIR}/../.pip

source "${MINIFORGE}/bin/activate"
eval "$(conda shell.bash hook)"

# ─── STEP 1: Create conda env ─────────────────────────────────────────────────
if [[ ! -f "$SENTINEL_CONDA" ]]; then
    log "=== STEP 1: Creating conda environment ==="
    mkdir -p "$ENVPREFIX"
    conda create python=3.12.12 icu=73 \
        --prefix "$ENVPREFIX" \
        --override-channels \
        --channel conda-forge \
        --strict-channel-priority \
        --yes
    touch "$SENTINEL_CONDA"
    log "Conda env created: $ENVPREFIX"
else
    log "=== STEP 1: Conda env already exists, skipping ==="
fi

conda activate "$ENVPREFIX"
# Always ensure sentinel dir exists — conda create makes $ENVPREFIX but not subdirs
mkdir -p "$SENTINEL_DIR"
log "Python: $(which python) — $(python --version)"

# ─── Install uv and maturin (fast, always idempotent) ────────────────────────
log "=== Installing uv and maturin ==="
pip install --quiet uv maturin

# ─── STEP 1b: Install build-time system deps via conda-forge ─────────────────
# cmake, libhwloc, clangdev missing on RHEL 9.7.
# NOTE: conda-forge protobuf does NOT ship the protoc binary — see Step 1c.
if [[ ! -f "$SENTINEL_BUILDDEPS" ]]; then
    log "=== STEP 1b: Installing build-time system deps via conda-forge ==="
    conda install --prefix "$ENVPREFIX" \
        --override-channels --channel conda-forge \
        --strict-channel-priority --yes \
        cmake libhwloc clangdev
    touch "$SENTINEL_BUILDDEPS"
    log "Build deps installed."
else
    log "=== STEP 1b: Build deps already installed, skipping ==="
fi

# ─── STEP 1c: Install protoc binary via curl ─────────────────────────────────
# conda-forge protobuf does NOT include the protoc compiler binary.
# Required by cargo build scripts for: etcd-client, velo-transports, dynamo-llm.
# Must be installed BEFORE cargo build (Step 2).
if [[ ! -f "$CONDA_PREFIX/bin/protoc" ]]; then
    log "=== STEP 1c: Installing protoc ${PROTOC_VER} ==="
    curl -L \
        "https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOC_VER}/protoc-${PROTOC_VER}-linux-x86_64.zip" \
        -o /tmp/protoc.zip
    unzip -o /tmp/protoc.zip -d /tmp/protoc_extracted bin/protoc
    cp /tmp/protoc_extracted/bin/protoc "$CONDA_PREFIX/bin/protoc"
    chmod +x "$CONDA_PREFIX/bin/protoc"
    rm -rf /tmp/protoc.zip /tmp/protoc_extracted
    log "protoc: $(protoc --version)"
else
    log "=== STEP 1c: protoc already present, skipping ==="
fi

# ─── Point cargo build scripts at conda-installed libs ───────────────────────
# Needed at build time. Whether needed at runtime is checked via ldd in Step 7.
export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export CPATH="$CONDA_PREFIX/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CONDA_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Explicit env vars for cargo build scripts that don't use pkg-config:
# cudarc v0.19.3 allowlist stops at CUDA 13.1 — 13.2 not listed yet.
# Safe: cudarc uses dynamic-loading, CUDA 13.2 is backward compatible with 13.1.
export CUDARC_CUDA_VERSION=13010
# protoc binary — prost-build canonical env var.
export PROTOC="$CONDA_PREFIX/bin/protoc"

log "PKG_CONFIG_PATH:     $PKG_CONFIG_PATH"
log "PROTOC:              $PROTOC"
log "CUDARC_CUDA_VERSION: $CUDARC_CUDA_VERSION"

# ─── STEP 2: cargo build --release ────────────────────────────────────────────
cd "$REPO_DIR"

if [[ ! -f "$SENTINEL_CARGO" ]]; then
    log "=== STEP 2: Running cargo build --release ==="
    log "Rust will auto-download toolchain 1.93.1 on first run (~5 min extra)."
    log "Full build: expect 20-40 minutes. Live output below."
    cargo build --release
    touch "$SENTINEL_CARGO"
    log "cargo build complete."
else
    log "=== STEP 2: cargo already built, skipping ==="
    log "    (delete $SENTINEL_CARGO to force rebuild)"
fi

# ─── STEP 3: Build Python bindings via maturin ───────────────────────────────
# http, llmctl, dynamo-run no longer exist in v1.3.0-dev.1.
# Serving is entirely via: python -m dynamo.frontend / python -m dynamo.vllm
if [[ ! -f "$SENTINEL_MATURIN" ]]; then
    log "=== STEP 3: Building Python bindings via maturin ==="
    cd "$REPO_DIR/lib/bindings/python"
    maturin develop --uv
    cd "$REPO_DIR"
    touch "$SENTINEL_MATURIN"
    log "maturin develop complete."
else
    log "=== STEP 3: maturin already built, skipping ==="
    log "    (delete $SENTINEL_MATURIN to force rebuild)"
fi

# ─── STEP 4: Install ai-dynamo[vllm] from source ─────────────────────────────
# [vllm] extra pulls in:
#   nixl[cu12]==1.1.0 (pip selects cu13 wheel on CUDA 13.x — correct),
#   vllm[flashinfer,runai,otel]==0.22.0, ray>=2.55.0, blake3, uvloop,
#   soundfile, librosa
# nixl and vllm[runai] live on pypi.nvidia.com — extra-index-url required.
if [[ ! -f "$SENTINEL_DYNAMO" ]]; then
    log "=== STEP 4: Installing ai-dynamo[vllm] from source ==="
    cd "$REPO_DIR"
    uv pip install --python "$CONDA_PREFIX/bin/python" \
        --extra-index-url https://pypi.nvidia.com \
        -e ".[vllm]"
    touch "$SENTINEL_DYNAMO"
    log "ai-dynamo[vllm] installed."
else
    log "=== STEP 4: dynamo already installed, skipping ==="
    log "    (delete $SENTINEL_DYNAMO to force reinstall)"
fi

# ─── STEP 4b: Pin transformers ────────────────────────────────────────────────
# Done after dynamo install so vllm's resolver runs first.
log "=== STEP 4b: Installing transformers==5.10.2 ==="
uv pip install --python "$CONDA_PREFIX/bin/python" transformers==5.10.2

# ─── STEP 4c: Build dynamo wheel for multi-node deployment ───────────────────
# maturin build produces: ai_dynamo_runtime-1.3.0.dev1-cp312-*.whl
# This is the Rust extension only (dynamo._core). It does NOT include
# vllm, nixl, ray, or other Python dependencies.
#
# TO INSTALL ON ANOTHER NODE (e.g. minerva-dgx-02):
#   Step 1 — copy wheel to the node (shared FS makes this automatic here)
#   Step 2 — create a fresh conda env (same as Step 1 above)
#   Step 3 — install the wheel + all Python dependencies:
#
#   uv pip install --python $CONDA_PREFIX/bin/python \
#       --extra-index-url https://pypi.nvidia.com \
#       $WHEEL_DIR/ai_dynamo_runtime-1.3.0.dev1-*.whl \
#       "vllm[flashinfer,runai,otel]==0.22.0" \
#       "nixl[cu12]==1.1.0" \
#       "ray>=2.55.0" \
#       "blake3>=1.0.0,<2.0.0" \
#       uvloop soundfile librosa transformers==5.10.2
#
#   Step 4 — also install the ai-dynamo Python package (components/):
#   uv pip install --python $CONDA_PREFIX/bin/python \
#       --extra-index-url https://pypi.nvidia.com \
#       -e "$REPO_DIR[vllm]"
#   (this is fast — pure Python, no Rust compilation)
#
#   Step 5 — curl install etcd, nats-server, protoc as above
#
# NOTE: The shared NFS filesystem means all nodes can see the same wheel
# and the same repo — so "another node" install may just be another conda env.
if [[ ! -f "$SENTINEL_WHEEL" ]]; then
    log "=== STEP 4c: Building dynamo wheel for multi-node deployment ==="
    mkdir -p "$WHEEL_DIR"
    cd "$REPO_DIR/lib/bindings/python"
    maturin build --release \
        --out "$WHEEL_DIR"
    cd "$REPO_DIR"
    touch "$SENTINEL_WHEEL"
    log "Wheel built. Contents of $WHEEL_DIR:"
    ls -lah "$WHEEL_DIR/"
else
    log "=== STEP 4c: Wheel already built, skipping ==="
    log "    (delete $SENTINEL_WHEEL to force rebuild)"
    ls -lah "$WHEEL_DIR/" 2>/dev/null || true
fi

# ─── STEP 5: etcd server binary ───────────────────────────────────────────────
# etcd-client in Cargo.toml is the Rust client *library* crate — does NOT
# produce a standalone etcd server binary. We still need this curl install.
if [[ ! -f "$CONDA_PREFIX/bin/etcd" ]]; then
    log "=== STEP 5: Installing etcd ${ETCD_VER} ==="
    curl -L \
        "https://github.com/etcd-io/etcd/releases/download/${ETCD_VER}/etcd-${ETCD_VER}-linux-amd64.tar.gz" \
        | tar xz -C "$CONDA_PREFIX/bin" --strip-components=1 \
          --wildcards '*/etcd' '*/etcdctl'
    log "etcd: $(etcd --version | head -1)"
else
    log "=== STEP 5: etcd already present, skipping ==="
fi

# ─── STEP 6: nats-server binary ───────────────────────────────────────────────
# async-nats in Cargo.toml is the Rust client *library* crate — same story.
if [[ ! -f "$CONDA_PREFIX/bin/nats-server" ]]; then
    log "=== STEP 6: Installing nats-server ${NATS_VER} ==="
    curl -L \
        "https://github.com/nats-io/nats-server/releases/download/${NATS_VER}/nats-server-${NATS_VER}-linux-amd64.tar.gz" \
        | tar xz -C "$CONDA_PREFIX/bin" --strip-components=1 \
          --wildcards '*/nats-server'
    log "nats-server: $(nats-server --version)"
else
    log "=== STEP 6: nats-server already present, skipping ==="
fi

# ─── STEP 7: Verification ─────────────────────────────────────────────────────
log "=== STEP 7: Verifying installation ==="
VERIFY_FAILED=0

verify_python() {
    local stmt=$1 label=$2
    if python -c "$stmt" 2>/dev/null; then
        log "  OK : $label"
    else
        log "  FAIL: $label"
        VERIFY_FAILED=1
    fi
}

verify_bin() {
    local bin=$1
    if command -v "$bin" &>/dev/null; then
        log "  OK : $bin — $($bin --version 2>&1 | head -1)"
    else
        log "  FAIL: $bin not found in PATH"
        VERIFY_FAILED=1
    fi
}

verify_python "import vllm; print('         vllm', vllm.__version__)"              "vllm"
verify_python "import dynamo._core; print('  dynamo._core', dynamo._core.__file__)" "dynamo._core"
verify_python "import dynamo.runtime; print(' dynamo.runtime OK')"                 "dynamo.runtime"
verify_python "import dynamo.vllm; print('   dynamo.vllm', dynamo.vllm.__file__)"  "dynamo.vllm"
verify_python "from dynamo.frontend import main; print('  frontend OK')"           "dynamo.frontend"
verify_python "import nixl; print('         nixl', nixl.__file__)"                 "nixl"
verify_python "import ray; print('          ray', ray.__version__)"                "ray"
verify_python "import transformers; print('transformers', transformers.__version__)" "transformers"

verify_bin etcd
verify_bin nats-server
verify_bin protoc

# ── ldd check on dynamo._core.so — tells us if conda libs needed at runtime ───
# If libhwloc.so or libclang.so appear with "not found":
#   add export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH to job script.
# Empty = build-time only, no runtime action needed.
log "=== ldd check on dynamo._core extension module ==="
SO_FILE=$(python -c "import dynamo._core; print(dynamo._core.__file__)" 2>/dev/null || true)
if [[ -n "$SO_FILE" && -f "$SO_FILE" ]]; then
    log "  Checking: $SO_FILE"
    LDD_OUT=$(ldd "$SO_FILE" 2>/dev/null | grep -E "hwloc|clang|not found" || true)
    if [[ -n "$LDD_OUT" ]]; then
        log "  ATTENTION — runtime library dependencies found:"
        echo "$LDD_OUT" | while IFS= read -r line; do log "    $line"; done
        log "  ACTION: add 'export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH' to job script"
    else
        log "  Clean — no conda libs needed at runtime"
    fi
else
    log "  WARNING: could not locate dynamo._core.so"
fi

if [[ $VERIFY_FAILED -eq 0 ]]; then
    log ""
    log "══════════════════════════════════════════════════════════"
    log "  BUILD COMPLETE — all checks passed"
    log "  Conda env: $ENVPREFIX"
    log "  Wheel dir: $WHEEL_DIR"
    log "  Activate with:"
    log "    source /home/hossainm/miniforge3/bin/activate"
    log "    conda activate $ENVPREFIX"
    log "══════════════════════════════════════════════════════════"
else
    die "Build finished but verification failed — see FAIL lines above"
fi
