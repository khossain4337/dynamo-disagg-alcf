#!/bin/bash
set -uo pipefail

# =============================================================================
# CXI / Libfabric / NIXL-plugin environment probe.
#
# Run this on an ALLOCATED COMPUTE NODE (interactive PBS session), with the
# conda env active, e.g.:
#   qsub -A inference_service -q workq -I -l walltime=00:30:00 -l select=2:ngpus=4
#   source /vast/draco/tara/projects/Tara_Deployment/software/miniforge3/bin/activate
#   conda activate /vast/draco/tara/projects/Tara_Deployment/software/envs/conda_envs/vllm_0.27.1_nixl_1.4.0_python_3.12.12
#   bash check_cxi_libfabric.sh
#
# Login-node fi_info/python3 will NOT reflect the compute-node module/conda
# environment -- same caveat the handoff doc already flagged for nvc++.
#
# Answers, with real output instead of assumption:
#   A. Is libfabric's cxi provider actually usable here?      (fi_info -p cxi)
#   B. Which sysfs counter tree exists -- device/telemetry/ or
#      device/ports/0/counters/ -- and what byte/octet counters does it expose?
#   C. Does THIS conda env's NIXL install actually ship a LIBFABRIC plugin,
#      or only UCX? No point setting backends:[LIBFABRIC] if the .so was
#      never built/installed for this env.
#   D. libfabric version -- NIXL's Libfabric backend requires >=1.21.0
#      (breaking-change note in NIXL 0.8.0 release notes).
#
# Runs on both P and D nodes if PBS_NODEFILE gives two distinct nodes;
# otherwise just the current node.
#
# FIXED from the previous version: `ssh host "command"` runs a non-interactive,
# non-login shell, which does NOT source ~/.bashrc or otherwise inherit your
# local `conda activate`. The remote branch below now explicitly re-sources
# conda inside the ssh command string itself -- same pattern your own
# common_env.sh already uses for launch_p.sh/launch_d.sh. Override via env
# var if your paths differ.
# =============================================================================

CONDA_ACTIVATE_SNIPPET=${CONDA_ACTIVATE_SNIPPET:-'source /vast/draco/tara/projects/Tara_Deployment/software/miniforge3/bin/activate && conda activate /vast/draco/tara/projects/Tara_Deployment/software/envs/conda_envs/vllm_0.27.1_nixl_1.4.0_python_3.12.12'}

run_checks() {
    echo "############################################"
    echo "### Node: $(hostname -s)"
    echo "############################################"

    echo "--- A. fi_info: all providers visible on this node ---"
    fi_info 2>&1 | grep -E '^provider:' || echo "  fi_info not on PATH or returned nothing -- is libfabric installed/loaded here?"

    echo ""
    echo "--- A. fi_info: cxi provider specifically ---"
    fi_info -p cxi 2>&1 || echo "  fi_info -p cxi failed -- cxi provider not usable from this environment/conda env."

    echo ""
    echo "--- D. libfabric version ---"
    fi_info -v 2>&1 | head -5
    pkg-config --modversion libfabric 2>&1 || echo "  (pkg-config libfabric not found -- not necessarily fatal, just fewer confirmations)"

    echo ""
    echo "--- B. sysfs CXI device presence + counter files ---"
    if ! ls -d /sys/class/cxi/cxi* >/dev/null 2>&1; then
        echo "  No /sys/class/cxi/cxi* -- cxi_core driver not loaded on this node, or no read permission."
    fi
    for dev in /sys/class/cxi/cxi*; do
        [ -d "$dev" ] || continue
        echo ""
        echo "  device: $dev"
        echo "  -- device/telemetry/ (FULL listing this time -- previous head -20 cut off before"
        echo "     reaching the byte/octet counters alphabetically; confirmed real path on this system) --"
        if [ -d "$dev/device/telemetry" ]; then
            echo "  total files: $(ls "$dev/device/telemetry" | wc -l)"
            echo "  candidates matching octet/byte (the ones we actually want):"
            ls "$dev/device/telemetry" | grep -iE 'octet|byte' | sed 's/^/    /'
            [ -z "$(ls "$dev/device/telemetry" | grep -iE 'octet|byte')" ] && echo "    (none matched octet/byte -- see full listing below to find the right name)"
            echo "  full listing:"
            ls "$dev/device/telemetry" | sort | sed 's/^/    /'
        else
            echo "  (no telemetry/ dir here)"
        fi
        # Dropped the ports/0/counters/ check -- confirmed absent on this system,
        # no need to keep probing for it.
    done

    echo ""
    echo "--- C. NIXL plugin inventory: is LIBFABRIC actually built into THIS env? ---"
    # Anchored on the actual 'import nixl' location instead of a site-packages-wide
    # find -- the conda env DIRECTORY NAME itself contains the substring "nixl"
    # (vllm_0.27.1_nixl_1.4.0_...), so a path-substring filter over the whole
    # site-packages tree false-matches everything in the env (PIL, pydantic,
    # sqlalchemy, ...). That's what happened last run -- not a real signal.
    NIXL_FILE=$(python3 -c "import nixl; print(nixl.__file__)" 2>&1)
    if [ -f "${NIXL_FILE}" ] 2>/dev/null; then
        echo "  import nixl -> ${NIXL_FILE}"
        SITE_PACKAGES_DIR=$(dirname "$(dirname "${NIXL_FILE}")")
        echo "  searching for libplugin_LIBFABRIC.so under ${SITE_PACKAGES_DIR} ..."
        find "${SITE_PACKAGES_DIR}" -iname "libplugin_LIBFABRIC.so" 2>/dev/null | sed 's/^/    /'
    else
        echo "  'import nixl' failed: ${NIXL_FILE}"
        echo "  (conda env not active in this shell, or nixl genuinely not installed here)"
    fi
    echo "  NIXL's own plugin-manager report (best-effort -- exact API may need adjusting to your NIXL version):"
    python3 - <<'PYEOF' 2>&1 | sed 's/^/    /'
try:
    from nixl._api import nixl_agent, nixl_agent_config
    a = nixl_agent("cxi_probe", nixl_agent_config(backends=[]))
    print("available plugins:", a.get_plugin_list())
except Exception as e:
    print("NIXL plugin probe raised:", repr(e))
    print("(non-fatal for this script -- just means the API call above doesn't match this NIXL version;")
    print(" the find/ls output above is the authoritative signal either way)")
PYEOF
}

run_checks

if [ -n "${PBS_NODEFILE:-}" ] && [ -f "${PBS_NODEFILE}" ]; then
    mapfile -t NODES < <(sort -u "${PBS_NODEFILE}")
    THIS_HOST=$(hostname -s)
    for n in "${NODES[@]}"; do
        [[ "$n" == "${THIS_HOST}"* ]] && continue
        echo ""
        echo "=== Remote check on $n ==="
        # Conda activation explicitly re-sourced HERE, inside the remote command
        # string -- this is the fix. Function declared first, activation second,
        # so run_checks executes with the right PATH already in place.
        ssh -n "$n" "$(declare -f run_checks); ${CONDA_ACTIVATE_SNIPPET} && run_checks"
    done
fi
