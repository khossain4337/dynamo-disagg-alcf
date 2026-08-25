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

# --- Completeness check --------------------------------------------------------
# "The command exited 0" is not evidence, as the --exclude incident documented
# below proved: a run that fetched 0.03% of the model printed a tick and
# returned. Neither is "du looks about right" -- a truncated shard set can be
# most of the bytes and still be unloadable.
#
# The only trustworthy check is the model's own manifest.
# model.safetensors.index.json maps every tensor to the shard holding it, so the
# set of shard names in weight_map IS the definition of complete. Compare the
# cache against that, and look for leftover .incomplete blobs while we are here
# (huggingface_hub stages each download as <blob>.incomplete and renames on
# success, so one left behind is a shard interrupted mid-flight).
#
# Runs automatically after a foreground download, and standalone via
#   VERIFY=1 bash download_model.sh
verify_cache() {
    python3 - "${MODEL}" "${HF_HOME}" <<'PY'
import json, os, sys

repo, home = sys.argv[1], sys.argv[2]
root = os.path.join(home, "hub", "models--" + repo.replace("/", "--"))
snaps = os.path.join(root, "snapshots")

if not os.path.isdir(snaps):
    print("  INCOMPLETE: no snapshot directory. Nothing has been downloaded.")
    sys.exit(1)

# Resolve the SAME revision vLLM will load, not merely the newest one on disk.
# A repo can pick up a new commit between runs, leaving two snapshot
# directories -- one complete, one a stub. Verifying by mtime and loading by
# ref would then check one revision and serve another, and the failure would
# arrive 30 minutes into a Nemotron load looking nothing like "wrong snapshot".
# refs/main is what huggingface_hub resolves a bare repo-id against, so read it.
ref = os.path.join(root, "refs", os.environ.get("HF_REVISION", "main"))
if os.path.exists(ref):
    with open(ref) as f:
        rev = os.path.join(snaps, f.read().strip())
    if not os.path.isdir(rev):
        print(f"  INCOMPLETE: refs/main names {os.path.basename(rev)},")
        print("  but no such snapshot directory exists.")
        sys.exit(1)
else:
    # No ref (rare -- a partial or hand-assembled cache). Fall back to newest.
    rev = max((os.path.join(snaps, d) for d in os.listdir(snaps)),
              key=os.path.getmtime)
    print("  (no refs/main; falling back to the newest snapshot on disk)")

stale = [d for d in os.listdir(snaps) if os.path.join(snaps, d) != rev]
if stale:
    print(f"  note: {len(stale)} other snapshot(s) present: {', '.join(d[:12] for d in stale)}")
    print("        harmless -- snapshots hold only symlinks -- but `du` counts them.")
idx = os.path.join(rev, "model.safetensors.index.json")

if os.path.exists(idx):
    with open(idx) as f:
        want = sorted(set(json.load(f)["weight_map"].values()))
elif os.path.exists(os.path.join(rev, "model.safetensors")):
    # Unsharded. Models below roughly 5 GB ship one model.safetensors and NO
    # index, so an index-only check would call every small model INCOMPLETE --
    # including the Qwen rigs this harness is brought up on. There is no
    # manifest to compare against in this case, so "the single weight file is
    # present" is the strongest statement available; say so rather than imply
    # the same rigour as the sharded path.
    want = ["model.safetensors"]
    print("  (unsharded model: no index to verify against, checking the one file)")
else:
    print("  INCOMPLETE: neither model.safetensors.index.json nor")
    print("  model.safetensors is present. No weights have been fetched.")
    sys.exit(1)

# os.path.exists() follows symlinks, so a snapshot entry pointing at a blob
# that was never written counts as missing -- which is exactly right here.
missing, total = [], 0
for name in want:
    p = os.path.join(rev, name)
    if os.path.exists(p):
        total += os.path.getsize(os.path.realpath(p))
    else:
        missing.append(name)

blobs = os.path.join(root, "blobs")
partial = ([f for f in os.listdir(blobs) if f.endswith(".incomplete")]
           if os.path.isdir(blobs) else [])

print(f"  snapshot: {rev}")
print(f"  shards:   {len(want) - len(missing)} of {len(want)} present")
# Both units, because `du -h` reports GiB and the model card reports GB, and a
# reader comparing the two otherwise sees a 7% "discrepancy" that is only ever
# 2^30 vs 10^9. Shards only -- du counts the tokenizer, configs and any stale
# snapshot as well, so du will always read slightly higher than this line.
print(f"  weights:  {total / 1e9:.1f} GB  ({total / 2**30:.1f} GiB, shards only)")
if partial:
    print(f"  in-flight: {len(partial)} .incomplete blob(s) -- download was interrupted")
if missing:
    show = ", ".join(missing[:4]) + (" ..." if len(missing) > 4 else "")
    print(f"  INCOMPLETE: {len(missing)} shard(s) missing: {show}")
    sys.exit(1)
if partial:
    sys.exit(1)
print("  COMPLETE. Safe to serve.")
PY
}

# Standalone verification, before the space check so it costs nothing and works
# on a full filesystem.
if [ -n "${VERIFY:-}" ]; then
    echo "Verifying ${MODEL} in ${HF_HOME}/hub"
    verify_cache
    exit $?
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
# NO --exclude, deliberately, and this is scar tissue: the first run of this
# script downloaded 78 KB and reported success.
#
#     --exclude ".eval_results/*" "accuracy_chart.png"
#
# huggingface_hub 1.x's CLI is typer-based, where --exclude takes exactly ONE
# value per occurrence. So that parsed as --exclude=".eval_results/*" plus a
# POSITIONAL filename "accuracy_chart.png" -- and `hf download <repo> <files>`
# means "fetch only these files". The CLI warned ("Ignoring --exclude since
# filenames have been explicitly set"), fetched the one png, exited 0, and
# printed a tick.
#
# The typer-correct form is a repeated flag (--exclude A --exclude B), but the
# argparse CLI in huggingface_hub 0.x reads --exclude as nargs="*", where a
# repeated flag means last-one-wins. No single spelling is right for both
# generations, and the entire prize is skipping ~1 MB of eval JSON and a chart.
# Not worth a footgun whose failure mode is a successful-looking exit: fetch
# everything, and let verify_cache() below decide what "complete" means.
echo "Model:   ${MODEL}"
echo "Cache:   ${HF_HOME}/hub"
echo "Log:     ${LOG}"
echo "Workers: ${MAX_WORKERS}   hf_transfer: ${HF_HUB_ENABLE_HF_TRANSFER}"
echo ""

CMD=("${HF_CLI[@]}" "${MODEL}" --max-workers "${MAX_WORKERS}")

if [ -n "${FOREGROUND:-}" ]; then
    "${CMD[@]}" 2>&1 | tee "${LOG}"
    echo ""
    # NOT gated on the CLI's exit status, on purpose. The whole lesson of the
    # --exclude incident is that a zero exit says nothing about what arrived.
    echo "=== Completeness check (manifest, not exit code) ==="
    verify_cache
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
    echo "  verify:   VERIFY=1 bash ${BASH_SOURCE[0]}"
    echo ""
    echo "Expect ~247 GB across ~50 safetensors shards. VERIFY=1 checks the"
    echo "cache against the model's own weight manifest and prints COMPLETE or"
    echo "names the missing shards -- do that before serving, not du."
fi
