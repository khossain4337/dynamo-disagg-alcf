#!/bin/bash
# Fetch a model into the shared HF cache, in the background. VERIFY=1 checks only.
# MODEL=<repo> NEED_GB=<GiB> to override the Muse-Spark defaults.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env_vllm_0.32.1rc1_nixl_1.5.2.sh"
MODEL=${MODEL:-meta-models/Muse-Spark-Open-816B-A42B}
NEED_GB=${NEED_GB:-1650}
# Must match HF_HOME in emit_common_env.sh.
export HF_HOME=/vast/draco/tara/projects/Tara_Deployment/software/model-weights
export HF_TOKEN=$(cat ~/.hf_token)
export HTTP_PROXY=http://proxy.alcf.anl.gov:3128 HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=$HTTP_PROXY https_proxy=$HTTPS_PROXY
# hf_transfer ignores the proxy and hangs silently.
export HF_HUB_OFFLINE=0 HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DOWNLOAD_TIMEOUT=60

# Complete = every shard in the index exists and no .incomplete blob is left.
verify() {
python3 - "$MODEL" "$HF_HOME" <<'PY'
import json, os, sys
root = os.path.join(sys.argv[2], "hub", "models--" + sys.argv[1].replace("/", "--"))
rev = os.path.join(root, "snapshots", open(os.path.join(root, "refs", "main")).read().strip())
want = sorted(set(json.load(open(os.path.join(rev, "model.safetensors.index.json")))["weight_map"].values()))
missing = [w for w in want if not os.path.exists(os.path.join(rev, w))]
partial = [b for b in os.listdir(os.path.join(root, "blobs")) if b.endswith(".incomplete")]
gb = sum(os.path.getsize(os.path.join(rev, w)) for w in want if w not in missing) / 2**30
print(f"{rev}\nshards {len(want) - len(missing)}/{len(want)}, {gb:.1f} GiB, {len(partial)} .incomplete")
if missing or partial: sys.exit(1)
print("COMPLETE")
PY
}
[ -n "${VERIFY:-}" ] && { verify; exit $?; }

AVAIL=$(df -BG --output=avail "$HF_HOME" | tail -1 | tr -dc '0-9')
[ "$AVAIL" -lt "$NEED_GB" ] && { echo "$AVAIL GiB free, need $NEED_GB"; exit 1; }
LOG=$HF_HOME/download_$(basename "$MODEL")_$(date +%Y%m%d_%H%M%S).log
# setsid: survives the ssh session.
setsid nohup hf download "$MODEL" --max-workers 8 > "$LOG" 2>&1 < /dev/null &
echo "pid $! log $LOG"
echo "verify: VERIFY=1 MODEL=$MODEL bash ${BASH_SOURCE[0]}"
