#!/bin/bash
set -euo pipefail

# ============================================================
# patch_vllm_nixl_ssm_assert.sh
#
# Patches vLLM 0.22.0 NIXL connector to fix crash with 2+
# prefill workers on hybrid Mamba/Attention models (Nemotron).
#
# BUG: _apply_prefix_caching in nixl/worker.py asserts
#   assert num_local_blocks == num_remote_blocks
# for SSM (Mamba) groups. With 2+ prefill workers, SSM block
# counts can differ due to logical block rounding, causing
# AssertionError and EngineDeadError on every request.
#
# FIX: Replace the assert with trim-to-minimum, same logic
# as the HeteroTP attention path in the same function.
#
# USAGE:
#   bash patch_vllm_nixl_ssm_assert.sh
#
# TO REVERT:
#   cp worker.py.bak worker.py
# ============================================================

tstamp() { date +"%Y-%m-%d %H:%M:%S"; }
log()    { echo "[$(tstamp)] $*"; }
die()    { echo "[$(tstamp)] FATAL: $*" >&2; exit 1; }

WORKER_PY="/home/hossainm/software/envs/conda_envs/dynamo_vllm_1.3.0_dev1/lib/python3.12/site-packages/vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py"
BACKUP="${WORKER_PY}.bak"

# ── Step 1: Verify the target line exists ────────────────────
log "=== Step 1: Verifying target line exists ==="
MATCHES=$(grep -c "assert num_local_blocks == num_remote_blocks" "$WORKER_PY" || true)

if [[ "$MATCHES" -eq 0 ]]; then
    die "Target line not found — already patched, or wrong file?"
elif [[ "$MATCHES" -gt 1 ]]; then
    die "Multiple matches found ($MATCHES) — unexpected, aborting"
fi

LINE_NO=$(grep -n "assert num_local_blocks == num_remote_blocks" "$WORKER_PY" | cut -d: -f1)
log "Found target at line $LINE_NO — exactly one match. Good."

# ── Step 2: Backup original ──────────────────────────────────
log "=== Step 2: Backing up original ==="
cp "$WORKER_PY" "$BACKUP"
log "Backup: $BACKUP"
ls -lh "$WORKER_PY" "$BACKUP"

# ── Step 3: Apply patch ──────────────────────────────────────
log "=== Step 3: Applying patch ==="

python3 << 'PYEOF'
import sys

path = "/home/hossainm/software/envs/conda_envs/dynamo_vllm_1.3.0_dev1/lib/python3.12/site-packages/vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py"

old = ("                if _is_ssm_spec(self._group_spec_types[i]):\n"
       "                    assert num_local_blocks == num_remote_blocks")

new = ("                if _is_ssm_spec(self._group_spec_types[i]):\n"
       "                    # PATCH: trim to minimum instead of asserting equality.\n"
       "                    # Original assert crashes with 2+ prefill workers on\n"
       "                    # hybrid Mamba/Attention models (e.g. Nemotron Nano FP8).\n"
       "                    # SSM block counts differ due to logical block rounding\n"
       "                    # in multi-prefill topology. Same fix as HeteroTP\n"
       "                    # attention path below.\n"
       "                    # Filed: github.com/vllm-project/vllm — see NOTES.\n"
       "                    num_blocks = min(num_local_blocks, num_remote_blocks)\n"
       "                    local_block_ids[i] = local_block_ids[i][:num_blocks]\n"
       "                    remote_block_ids[i] = remote_group[:num_blocks]")

content = open(path).read()
if old not in content:
    print("ERROR: Pattern not found — check indentation or already patched")
    sys.exit(1)

content = content.replace(old, new, 1)
open(path, 'w').write(content)
print("Patch applied successfully")
PYEOF

# ── Step 4: Verify patch landed correctly ────────────────────
log "=== Step 4: Verifying patch ==="

# Should be GONE
ASSERT_COUNT=$(grep -c "assert num_local_blocks == num_remote_blocks" "$WORKER_PY" || true)
if [[ "$ASSERT_COUNT" -ne 0 ]]; then
    die "Assert still present after patch — something went wrong"
fi
log "  OK: assert line is gone"

# Should be PRESENT
PATCH_COUNT=$(grep -c "trim to minimum instead of asserting equality" "$WORKER_PY" || true)
if [[ "$PATCH_COUNT" -ne 1 ]]; then
    die "Patch comment not found — something went wrong"
fi
log "  OK: patch comment present"

# Show the patched region for visual confirmation
log "=== Patched region (context) ==="
grep -n -A 12 "if _is_ssm_spec" "$WORKER_PY" | head -30

log ""
log "══════════════════════════════════════════════"
log "  PATCH COMPLETE"
log "  To revert: cp $BACKUP $WORKER_PY"
log "══════════════════════════════════════════════"
