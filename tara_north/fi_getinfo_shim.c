/*
 * fi_getinfo_shim.c
 * -----------------
 * LD_PRELOAD proof-of-concept for the NIXL LIBFABRIC / Slingshot-CXI bug.
 *
 * Root cause (NIXL 1.4.0): nixlLibfabricTopology::buildPcieToLibfabricMapping()
 * calls fi_getinfo() with hints that set ONLY fabric_attr->prov_name and leave
 * domain_attr->mr_mode == 0. The CXI provider needs a real mr_mode, so it returns
 * -FI_ENODATA ("No data available"), which NIXL treats as fatal and aborts
 * createBackend("LIBFABRIC"). The sibling call getAvailableNetworkDevices()
 * DOES set mr_mode (=~3) and works -- they fixed one fi_getinfo and missed this one.
 * Upstream `main` fixes it by setting, for cxi, the exact mr_mode mask used below.
 *
 * This shim interposes fi_getinfo and, ONLY for the broken call
 * (prov_name == "cxi" AND mr_mode == 0), injects main's blessed CXI mr_mode mask,
 * then forwards to the real fi_getinfo.
 *
 * Symbol resolution note: NIXL dlopen's libplugin_LIBFABRIC.so, which loads libfabric
 * into a LOCAL scope. RTLD_NEXT only searches the GLOBAL scope, so it does NOT find
 * libfabric's fi_getinfo here. We therefore resolve the real symbol by dlopen'ing
 * libfabric explicitly (idempotent: returns the already-loaded instance).
 *
 * Build (compute node, conda env active):
 *   gcc -shared -fPIC -o fi_getinfo_shim.so fi_getinfo_shim.c -ldl \
 *       $(pkg-config --cflags libfabric)
 *
 * Run against the existing reproducer:
 *   source env_for_libfabric_topology_error.sh
 *   LD_PRELOAD=$PWD/fi_getinfo_shim.so python3 repro_libfabric_topology.py
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>

/* Exact CXI mr_mode mask used by upstream nixl `main` in
 * buildPcieToLibfabricMapping() for provider "cxi". */
#define CXI_MR_MODE                                                          \
    (FI_MR_LOCAL | FI_MR_HMEM | FI_MR_VIRT_ADDR | FI_MR_ALLOCATED |          \
     FI_MR_PROV_KEY | FI_MR_ENDPOINT)

typedef int (*fi_getinfo_fn)(uint32_t version,
                             const char *node,
                             const char *service,
                             uint64_t flags,
                             const struct fi_info *hints,
                             struct fi_info **info);

static fi_getinfo_fn real_fi_getinfo = NULL;

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

    /* Target ONLY the broken call: provider is cxi and mr_mode was left at 0.
     * The working discovery call already sets mr_mode, so it is skipped here. */
    if (hints && hints->fabric_attr && hints->fabric_attr->prov_name &&
        strcmp(hints->fabric_attr->prov_name, "cxi") == 0 && hints->domain_attr &&
        hints->domain_attr->mr_mode == 0) {

        /* hints is const in the API; we deliberately mutate the caller's struct
         * for this throwaway test. NIXL owns this hints object and uses it once. */
        struct fi_info *h = (struct fi_info *)hints;
        h->domain_attr->mr_mode = CXI_MR_MODE;
        fprintf(stderr,
                "[fi_getinfo_shim] patched cxi hints: set domain_attr->mr_mode = 0x%x\n",
                (unsigned)CXI_MR_MODE);
    }

    return real_fi_getinfo(version, node, service, flags, hints, info);
}
