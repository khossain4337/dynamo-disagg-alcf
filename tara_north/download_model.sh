#!/bin/bash
set -uo pipefail

# =============================================================================
# Pre-stage a HuggingFace model into the shared HF cache, ahead of any run.
#
#   bash download_model.sh                    # Nemotron-3-Super-120B-A12B-BF16
#   MODEL=<repo-id> bash download_model.sh    # anything else
#   FOREGROUND=1 bash download_model.sh       # watch it instead of backgrounding
#
# WHY A SEPARATE STEP AND NOT JUST `vllm serve <repo>`
#
# Two reasons, both learned the expensive way on other systems:
#
#   1. 247 GB does not download inside a health-check window. Letting the
#      server pull it means HEALTH_TRIES expires, the driver script calls
#      cleanup(), and the partial download is abandoned mid-shard.
#   2. At TP=4 there are FOUR worker processes per node and TWO nodes. They
#      would all race for the same cache directory on first touch. HF's
#      per-file locking mostly survives this, but "mostly" across 8 processes
#      and 50 shards on a shared filesystem is not a bet worth taking.
#
# Download once, from one process, on one node. Then every later run is a
# cache hit and starts in seconds.
#
# RESUMABLE. huggingface_hub writes to <blob>.incomplete and picks up where it
# left off, so re-running this after a dropped connection is safe and cheap --
# it re-verifies what is present and fetches only what is missing. That is also
# why this doubles as the verification step: run it again at the end and a
# complete download prints "nothing to fetch" almost immediately.
# =============================================================================

MODEL=${MODEL:-nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16}

# MUST match HF_HOME in the run scripts' common_env.sh, or the servers will
# look in one cache and find an empty one while 247 GB sits in another. The
# cache layout under $HF_HOME/hub/ is what `vllm serve <repo-id>` resolves
# against, which is why this uses the cache and NOT --local-dir: --local-dir
# writes a flat directory that vllm can only find by full path, silently
# breaking the repo-id form every other script here uses.
export HF_HOME=${HF_HOME:-/vast/draco/tara/projects/Tara_Deployment/software/model-weights}

# ALCF compute and login nodes have no direct route out; everything external
# goes through the site proxy. Set both cases -- some tools read the lowercase
# names only, some the uppercase only.
export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

# Default is 10s, which is fine on a LAN and marginal through a proxy under
# load. A timeout mid-shard costs a retry, not a restart, but 50 shards times
# a few retries each is an hour of nothing.
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-60}

# hf_transfer (the Rust fast path) is DELIBERATELY left off. It does not honour
# HTTP_PROXY/HTTPS_PROXY in the versions shipped so far -- it opens sockets
# directly and hangs with no error on a proxied network, which reads as "the
# download is slow" rather than "the download is never going to happen".
# Set HF_HUB_ENABLE_HF_TRANSFER=1 yourself if you confirm the route is direct.
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-0}

MAX_WORKERS=${MAX_WORKERS:-8}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG=${LOG:-${HF_HOME}/download_$(basename "${MODEL}")_${STAMP}.log}

# --- Token -------------------------------------------------------------------
# nvidia/* repos are public but gated behind an accepted licence, so an
# anonymous fetch returns 401 on the weights while still serving config.json.
# That failure mode is confusing: the tokenizer and config arrive, the shards
# do not, and it looks like a network problem.
if [ -z "${HF_TOKEN:-}" ]; then
    if [ -f ~/.hf_token ]; then
        HF_TOKEN=$(cat ~/.hf_token)
        export HF_TOKEN
    else
        echo "No HF_TOKEN and no ~/.hf_token. Gated nvidia/* weights will 401."
        echo "  echo <token> > ~/.hf_token && chmod 600 ~/.hf_token"
        exit 1
    fi
fi

# --- Conda env, for a huggingface_hub new enough to have the `hf` CLI --------
set +u
source "${SCRIPT_DIR}/env_for_libfabric_topology_error.sh"
set -u

# The CLI was renamed from `huggingface-cli` to `hf` in huggingface_hub 0.34.
# Both are present in some versions and the old one warns; prefer the new name
# and fall back rather than pinning to whichever this env happens to ship.
if command -v hf >/dev/null 2>&1; then
    HF_CLI=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then
    HF_CLI=(huggingface-cli download)
else
    echo "Neither 'hf' nor 'huggingface-cli' on PATH after activating the env."
    echo "  pip install -U huggingface_hub"
    exit 1
fi

# --- Space check -------------------------------------------------------------
# 247 GB of shards. Ask for 300 GB: HF stages each file as <blob>.incomplete
# and renames on completion, so peak usage is roughly the final size plus one
# in-flight shard, and filling a shared /vast is everyone's problem, not just
# this run's.
NEED_GB=${NEED_GB:-300}
mkdir -p "${HF_HOME}"
AVAIL_GB=$(df -BG --output=avail "${HF_HOME}" 2>/dev/null | tail -1 | tr -dc '0-9')
if [ -n "${AVAIL_GB}" ] && [ "${AVAIL_GB}" -lt "${NEED_GB}" ]; then
    echo "Only ${AVAIL_GB} GB free at ${HF_HOME}, want >= ${NEED_GB} GB."
    echo "Free space or point HF_HOME somewhere with room, then rerun."
    exit 1
fi
echo "Space at ${HF_HOME}: ${AVAIL_GB:-unknown} GB free"

# --- Fetch -------------------------------------------------------------------
# .eval_results/ and the accuracy chart are documentation, not weights. Tiny,
# but excluding them keeps `find $HF_HOME -name '*.safetensors' | wc -l`
# honest as a completeness check.
echo "Model:   ${MODEL}"
echo "Cache:   ${HF_HOME}/hub"
echo "Log:     ${LOG}"
echo "Workers: ${MAX_WORKERS}   hf_transfer: ${HF_HUB_ENABLE_HF_TRANSFER}"
echo ""

CMD=("${HF_CLI[@]}" "${MODEL}"
     --max-workers "${MAX_WORKERS}"
     --exclude ".eval_results/*" "accuracy_chart.png")

if [ -n "${FOREGROUND:-}" ]; then
    "${CMD[@]}" 2>&1 | tee "${LOG}"
    echo ""
    echo "Done. Verify with: bash ${BASH_SOURCE[0]}   (a complete cache returns immediately)"
else
    # setsid so it survives the ssh session that started it. This is hours of
    # wall clock; a dropped VPN should not cost the whole transfer. Same
    # pattern as start_cxi_poll() in run_pd_full_test_N2_R1.sh, for the same
    # reason -- signalling the local ssh client never reaches the remote job.
    #
    # Everything the child needs (PATH from the conda activation, HF_HOME,
    # HF_TOKEN, the proxy vars) is exported above, so the command can be
    # setsid'd directly rather than reconstructed inside a `bash -c` string.
    setsid nohup "${CMD[@]}" > "${LOG}" 2>&1 < /dev/null &
    DL_PID=$!
    disown "${DL_PID}" 2>/dev/null || true
    echo "Downloading in the background, pid ${DL_PID}."
    echo ""
    echo "  watch:    tail -f ${LOG}"
    echo "  progress: du -sh ${HF_HOME}/hub/models--${MODEL//\//--}"
    echo "  verify:   bash ${BASH_SOURCE[0]}"
    echo ""
    echo "Expect ~247 GB across 50 safetensors shards. When it finishes, rerun"
    echo "this script -- a complete cache exits in seconds having fetched"
    echo "nothing, which is the only completeness check worth trusting."
fi
