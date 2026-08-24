/*
 * fi_getinfo_shim.c
 * -----------------
 * LD_PRELOAD workarounds for two NIXL 1.4.0 LIBFABRIC/Slingshot-CXI bugs.
 * Both are interposed on fi_getinfo alone -- no C++ symbols, no mangled
 * names, no dlsym into the plugin.
 *
 *
 * BLOCKER 1 (mr_mode) -- fixed by PATCH 1 below
 * --------------------------------------------
 * nixlLibfabricTopology::buildPcieToLibfabricMapping() calls fi_getinfo()
 * with hints that set ONLY fabric_attr->prov_name and leave
 * domain_attr->mr_mode == 0. The CXI provider needs a real mr_mode, so it
 * returns -FI_ENODATA ("No data available"), which NIXL treats as fatal and
 * aborts createBackend("LIBFABRIC"). The sibling call
 * getAvailableNetworkDevices() DOES set mr_mode (=~3) and works -- they
 * fixed one fi_getinfo and missed this one. Upstream `main` fixes it by
 * setting, for cxi, the exact mr_mode mask used below.
 *
 *
 * BLOCKER 4 (no VRAM_SEG on cxi) -- fixed by PATCHES 2+3+4 below
 * --------------------------------------------------------------
 * `--mem cuda` fails with:
 *     registerMem: no available backends for mem type 'VRAM_SEG'
 *
 * That is not a registration failure. NIXL never even tried: the mem-type ->
 * backend map has an EMPTY list for VRAM_SEG, because the LIBFABRIC backend
 * never advertised it. The chain, all in NIXL 1.4.0 (and still in `main`):
 *
 *   libfabric_topology.cpp:92-123   discoverTopology()
 *       if (provider_name == "efa") { ...discoverHwlocTopology()... }
 *       else {
 *           // "For TCP/sockets devices, bypass complex topology discovery"
 *           num_nvidia_accel = 0;      <-- HARDCODED for EVERY non-EFA provider
 *       }
 *   libfabric_rail_manager.cpp:195  if (getNumNvidiaAccel() > 0)
 *                                       runtime_ = FI_HMEM_CUDA;
 *                                   else runtime_ = FI_HMEM_SYSTEM;
 *   libfabric_backend.cpp:421       mems.push_back(DRAM_SEG);
 *                                   if (runtime_ == FI_HMEM_CUDA)
 *                                       mems.push_back(VRAM_SEG);
 *
 * The author considered exactly two worlds, EFA and TCP/sockets. On cxi the
 * else-branch fires, the GPUs are never looked for, runtime_ becomes
 * FI_HMEM_SYSTEM, and the backend offers DRAM only. Same species of bug as
 * BLOCKER 1: an EFA-centric path that never anticipated a third provider.
 *
 * The single root variable is num_nvidia_accel, and its single input is the
 * string comparison provider_name == "efa". So: make NIXL believe the
 * provider is "efa" during discovery, and let it keep using the real cxi
 * devices underneath.
 *
 * WHY THIS IS SAFE TO DO FROM fi_getinfo -- every step verified in the 1.4.0
 * source, not assumed:
 *
 *   a) provider_name is not read from a config or a device node. It comes
 *      from LibfabricUtils::getAvailableNetworkDevices() (libfabric_common.cpp:36),
 *      which builds a provider->devices map by reading
 *      cur->fabric_attr->prov_name off the fi_info list that fi_getinfo
 *      RETURNS, then picks by preference: cxi > efa > tcp > sockets.
 *      Rewriting the returned name to "efa" removes the "cxi" key and makes
 *      it return {"efa", ["cxi0".."cxi3"]} -- the spoofed provider label with
 *      the REAL device list. That is the whole trick (PATCH 2).
 *
 *   b) Taking the EFA branch then runs the hwloc accelerator scan, so
 *      num_nvidia_accel becomes the true GPU count. Everything downstream
 *      falls out: runtime_ = FI_HMEM_CUDA, VRAM_SEG advertised, and
 *      getMrAttrIface() (libfabric_topology.h:234) returns FI_HMEM_CUDA
 *      instead of FI_HMEM_NEURON -- it is `device_id < num_nvidia_accel`,
 *      the same variable.
 *
 *   c) Nothing on that branch can hard-fail for a machine with no real EFA
 *      NICs. discoverEfaDevicesWithHwloc() (topology.cpp:457) is purely
 *      diagnostic: it counts, logs a DEBUG on mismatch, and unconditionally
 *      returns NIXL_SUCCESS. buildAccelToEfaMapping() (topology.cpp) falls
 *      back to buildFallbackMapping(), which also returns NIXL_SUCCESS.
 *
 *   d) VRAM rail selection does not dead-end either. selectRailsForMemory()
 *      asks getEfaDevicesForPci(gpu_bdf); when the GPU's BDF is not in the
 *      affinity map that function returns all_devices (topology.cpp:196),
 *      i.e. the real cxi NIC list, which createRails() has already keyed
 *      into efa_device_to_rail_map. So the GPU gets rails either way.
 *
 *   e) Nothing ever compares the returned prov_name against the requested
 *      one, so the spoof is not detected.
 *
 * THE COST OF (a) is that provider_name is not just a label -- downstream
 * code uses it as a BEHAVIOURAL SWITCH, in two different ways, so the spoof
 * has to be undone in two different ways:
 *
 *   COST 1: buildPcieToLibfabricMapping() (topology.cpp:496) copies
 *      provider_name straight into hints->fabric_attr->prov_name. Left alone
 *      it would request a provider this machine does not have and fail.
 *      PATCH 3 rewrites that "efa" back to "cxi" on the way in.
 *
 *   COST 2: the nixlLibfabricRail ctor (libfabric_rail.cpp:420-436) does NOT
 *      set prov_name at all -- it string-switches on the provider to pick an
 *      mr_mode PROFILE, then narrows by domain_attr->name:
 *          provider == "cxi" -> mr_mode 0x674 (has FI_MR_ENDPOINT),
 *                               caps |= FI_RMA_EVENT, mr_key_size left 0
 *          else ("EFA and    -> mr_mode 0x474 (NO FI_MR_ENDPOINT),
 *           other providers")   mr_key_size = 2
 *      There is no prov_name for PATCH 3 to catch, so the spoof silently
 *      downgrades every rail to the EFA profile, and CXI answers -FI_ENODATA
 *      ("fi_getinfo failed for rail 0: No data available"). The ctor's one
 *      retry only drops FI_HMEM from caps and leaves the wrong mr_mode, so it
 *      fails too. PATCH 4 restores the cxi profile.
 *
 * "cxi" and "efa" are both exactly 3 bytes, so every rewrite here is an
 * in-place memcpy. No allocation, no free, nothing for fi_freeinfo to get
 * wrong.
 *
 * SCOPE: PATCH 2 fires on ONE precisely fingerprinted call (no prov_name AND
 * mr_mode == ~3 -- that is getAvailableNetworkDevices() and nothing else).
 * PATCH 3 fires only on hints that literally say "efa". PATCH 4 fires only on
 * hints naming a "cxi*" domain, which is the rail ctor and nothing else --
 * the discovery call and buildPcieToLibfabricMapping() both leave
 * domain_attr->name NULL. On a real EFA machine this shim would be inert for
 * PATCH 2 (the returned name is already "efa") and for PATCH 4 (no cxi
 * domains), while PATCH 3 would rewrite efa->cxi wrongly -- so do not use it
 * there. It is a Slingshot-only tool.
 *
 *
 * KILL SWITCH
 * -----------
 * PATCHES 2+3+4 are on by default. To get exactly the old behaviour (BLOCKER
 * 1 fix only, DRAM only, no VRAM):
 *     export NIXL_CXI_VRAM_SHIM=0
 *
 * Worth A/B-ing: riding the EFA branch also makes hasPcieDevices() true, so
 * the DRAM rail-selection policy can switch from "all rails" to NUMA-aware
 * and pick a SUBSET of rails. That may move the measured DRAM bandwidth in
 * either direction. Compare `--mem dram` with the switch on and off.
 *
 *
 * Symbol resolution note: NIXL dlopen's libplugin_LIBFABRIC.so, which loads
 * libfabric into a LOCAL scope. RTLD_NEXT only searches the GLOBAL scope, so
 * it does NOT find libfabric's fi_getinfo here. We therefore resolve the real
 * symbol by dlopen'ing libfabric explicitly (idempotent: returns the
 * already-loaded instance).
 *
 * Build (compute node, conda env active):
 *   gcc -shared -fPIC -o fi_getinfo_shim.so fi_getinfo_shim.c -ldl \
 *       $(pkg-config --cflags libfabric)
 *
 * Run against the reproducer:
 *   source env_for_libfabric_topology_error.sh
 *   LD_PRELOAD=$PWD/fi_getinfo_shim.so python3 repro_nixl_2rank_transfer.py --mem cuda
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>

/* Exact CXI mr_mode mask used by upstream nixl `main` in
 * buildPcieToLibfabricMapping() for provider "cxi". */
#define CXI_MR_MODE                                                          \
    (FI_MR_LOCAL | FI_MR_HMEM | FI_MR_VIRT_ADDR | FI_MR_ALLOCATED |          \
     FI_MR_PROV_KEY | FI_MR_ENDPOINT)

/* The mr_mode getAvailableNetworkDevices() uses, and no other NIXL call
 * does. Half of the fingerprint that identifies the discovery call. */
#define DISCOVERY_MR_MODE (~3)

/* Prefix of the libfabric domain names for Slingshot NICs ("cxi0".."cxi3").
 * nixlLibfabricRail puts the device name in domain_attr->name, which is how
 * PATCH 4 recognises a rail call. */
#define CXI_DOMAIN_PREFIX "cxi"

typedef int (*fi_getinfo_fn)(uint32_t version,
                             const char *node,
                             const char *service,
                             uint64_t flags,
                             const struct fi_info *hints,
                             struct fi_info **info);

static fi_getinfo_fn real_fi_getinfo = NULL;

/* -1 = not yet read, 0 = disabled, 1 = enabled. */
static int vram_shim_enabled = -1;

static int
vram_shim_on(void) {
    if (vram_shim_enabled < 0) {
        const char *e = getenv("NIXL_CXI_VRAM_SHIM");
        vram_shim_enabled = (e && strcmp(e, "0") == 0) ? 0 : 1;
        fprintf(stderr,
                "[fi_getinfo_shim] cxi->efa VRAM spoof %s"
                " (NIXL_CXI_VRAM_SHIM=0 to disable)\n",
                vram_shim_enabled ? "ENABLED" : "disabled");
    }
    return vram_shim_enabled;
}

static fi_getinfo_fn
resolve_real_fi_getinfo(void) {
    /* Try the cheap path first (works if libfabric happens to be global). */
    fi_getinfo_fn fn = (fi_getinfo_fn)dlsym(RTLD_NEXT, "fi_getinfo");
    if (fn) {
        fprintf(stderr, "[fi_getinfo_shim] resolved real fi_getinfo via RTLD_NEXT\n");
        return fn;
    }

    /* RTLD_NEXT missed it (libfabric is in a local/plugin scope). dlopen libfabric
     * explicitly -- this returns the already-loaded instance if present. */
    const char *cands[] = {
        "libfabric.so.1",
        "libfabric.so",
        "/opt/cray/libfabric/2.3.1/lib64/libfabric.so.1",
        "/opt/cray/libfabric/2.3.1/lib64/libfabric.so",
        NULL,
    };
    for (int i = 0; cands[i]; ++i) {
        void *h = dlopen(cands[i], RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
        if (!h) {
            /* NOLOAD failed (not already loaded under that name); try a real load. */
            h = dlopen(cands[i], RTLD_NOW | RTLD_GLOBAL);
        }
        if (h) {
            fn = (fi_getinfo_fn)dlsym(h, "fi_getinfo");
            if (fn) {
                fprintf(stderr, "[fi_getinfo_shim] resolved real fi_getinfo via dlopen(\"%s\")\n",
                        cands[i]);
                return fn;
            }
        }
    }

    fprintf(stderr, "[fi_getinfo_shim] FATAL: could not resolve real fi_getinfo (%s)\n",
            dlerror() ? dlerror() : "no candidate matched");
    return NULL;
}

int
fi_getinfo(uint32_t version,
           const char *node,
           const char *service,
           uint64_t flags,
           const struct fi_info *hints,
           struct fi_info **info) {

    if (!real_fi_getinfo) {
        real_fi_getinfo = resolve_real_fi_getinfo();
        if (!real_fi_getinfo) {
            return -FI_ENODATA;
        }
    }

    /* hints is const in the API; we deliberately mutate the caller's struct
     * for this throwaway test. NIXL owns these hints and uses them once. */
    struct fi_info *h = (struct fi_info *)hints;

    /* Is this the one discovery call whose RESULT decides provider_name?
     * Fingerprint: no provider requested AND the distinctive mr_mode that
     * only getAvailableNetworkDevices() sets. Captured BEFORE any rewrite. */
    int is_discovery = h && h->fabric_attr && h->fabric_attr->prov_name == NULL &&
        h->domain_attr && h->domain_attr->mr_mode == DISCOVERY_MR_MODE;

    /* PATCH 3: undo the spoof on the way in. Once provider_name is "efa",
     * NIXL copies it into the hints of buildPcieToLibfabricMapping() and of
     * every nixlLibfabricRail ctor. Send those to the real cxi provider.
     * Runs BEFORE the mr_mode patch below so the cxi rule still applies. */
    if (vram_shim_on() && h && h->fabric_attr && h->fabric_attr->prov_name &&
        strcmp(h->fabric_attr->prov_name, "efa") == 0) {

        memcpy(h->fabric_attr->prov_name, "cxi", 3); /* same length, in place */
        fprintf(stderr, "[fi_getinfo_shim] PATCH 3: hints prov_name efa -> cxi\n");
    }

    /* PATCH 1 (BLOCKER 1): target ONLY the broken call -- provider is cxi and
     * mr_mode was left at 0. The working discovery call already sets mr_mode,
     * so it is skipped here. */
    if (h && h->fabric_attr && h->fabric_attr->prov_name &&
        strcmp(h->fabric_attr->prov_name, "cxi") == 0 && h->domain_attr &&
        h->domain_attr->mr_mode == 0) {

        h->domain_attr->mr_mode = CXI_MR_MODE;
        fprintf(stderr,
                "[fi_getinfo_shim] PATCH 1: patched cxi hints, set domain_attr->mr_mode = 0x%x\n",
                (unsigned)CXI_MR_MODE);
    }

    /* PATCH 4 (BLOCKER 4, second half): repair the per-rail hints.
     *
     * nixlLibfabricRail's ctor (libfabric_rail.cpp:420-436) never sets
     * prov_name at all -- it uses the provider STRING as a behavioural
     * switch to choose an mr_mode profile, then filters by
     * domain_attr->name. So PATCH 3 has nothing to rewrite there, and our
     * spoof silently sends it down the "EFA and other providers"
     * else-branch:
     *
     *   cxi branch  : mr_mode 0x674 (incl. FI_MR_ENDPOINT), caps |= FI_RMA_EVENT,
     *                 mr_key_size untouched (0)
     *   else branch : mr_mode 0x474 (NO FI_MR_ENDPOINT), mr_key_size = 2
     *
     * CXI rejects the else-branch combination with -FI_ENODATA, which is
     * exactly the "fi_getinfo failed for rail 0: No data available" we saw.
     * Restore the profile the cxi branch would have produced.
     *
     * Fires only on rail calls: buildPcieToLibfabricMapping() and the
     * discovery call both leave domain_attr->name NULL. Idempotent, so the
     * ctor's retry-without-FI_HMEM path is handled too. We deliberately do
     * NOT re-add FI_RMA_EVENT on that retry -- NIXL's own retry resets caps
     * unconditionally for cxi as well, so this matches real cxi behaviour. */
    if (vram_shim_on() && h && h->domain_attr && h->domain_attr->name &&
        strncmp(h->domain_attr->name, CXI_DOMAIN_PREFIX, 3) == 0 &&
        (((h->domain_attr->mr_mode & FI_MR_ENDPOINT) == 0) ||
         h->domain_attr->mr_key_size != 0)) {

        h->domain_attr->mr_mode |= FI_MR_ENDPOINT;
        h->domain_attr->mr_key_size = 0; /* the cxi branch never sets this */
        h->caps |= FI_RMA_EVENT;
        fprintf(stderr,
                "[fi_getinfo_shim] PATCH 4: restored cxi rail hints for domain %s"
                " (mr_mode = 0x%x, mr_key_size = 0, +FI_RMA_EVENT)\n",
                h->domain_attr->name, (unsigned)h->domain_attr->mr_mode);
    }

    int ret = real_fi_getinfo(version, node, service, flags, hints, info);

    /* PATCH 2 (BLOCKER 4): relabel the discovered provider cxi -> efa, so that
     * getAvailableNetworkDevices() reports "efa" and discoverTopology() takes
     * the branch that actually counts the GPUs. The device names in
     * domain_attr->name are untouched, so the real cxi NICs are still what
     * gets used. */
    if (vram_shim_on() && is_discovery && ret == 0 && info && *info) {
        int renamed = 0;
        for (struct fi_info *cur = *info; cur; cur = cur->next) {
            if (cur->fabric_attr && cur->fabric_attr->prov_name &&
                strcmp(cur->fabric_attr->prov_name, "cxi") == 0) {

                memcpy(cur->fabric_attr->prov_name, "efa", 3); /* same length */
                renamed++;
            }
        }
        if (renamed) {
            fprintf(stderr,
                    "[fi_getinfo_shim] PATCH 2: relabelled %d discovered cxi info(s) as efa"
                    " -- NIXL will now count GPUs and advertise VRAM_SEG\n",
                    renamed);
        } else {
            fprintf(stderr,
                    "[fi_getinfo_shim] PATCH 2: discovery call saw no cxi provider;"
                    " VRAM_SEG will NOT be advertised\n");
        }
    }

    return ret;
}
