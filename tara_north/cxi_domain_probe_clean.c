/*
 * cxi_domain_probe.c
 * ------------------
 * Standalone libfabric probe for the SECOND NIXL/CXI blocker:
 *   fi_domain() returns -FI_ENOSYS ("Function not implemented") for the cxi
 *   provider, even though NIXL 1.4.0 already builds cxi-correct hints in
 *   libfabric_rail.cpp (the `provider == "cxi"` branch, lines ~426-437) and
 *   fi_getinfo + fi_fabric both succeed.
 *
 * This program replicates NIXL's EXACT rail hints for cxi, then sweeps the
 * attributes most likely to cause fi_domain ENOSYS (threading model, progress
 * model, FI_HMEM cap) and reports which combination lets fi_domain succeed.
 *
 *   - If SOME combination succeeds  -> NIXL-fixable (set that attr for cxi).
 *   - If NONE succeed               -> genuine CXI-provider / SHS limitation.
 *
 * Build:  gcc -o cxi_domain_probe cxi_domain_probe.c $(pkg-config --cflags --libs libfabric)
 * Run:    ./cxi_domain_probe        (on a compute node where fi_info -p cxi works)
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <rdma/fabric.h>
#include <rdma/fi_domain.h>

static const char *
thr_name(int t) {
    switch (t) {
    case FI_THREAD_UNSPEC: return "UNSPEC";
    case FI_THREAD_SAFE: return "SAFE";
    case FI_THREAD_FID: return "FID";
    case FI_THREAD_DOMAIN: return "DOMAIN";
    case FI_THREAD_COMPLETION: return "COMPLETION";
    case FI_THREAD_ENDPOINT: return "ENDPOINT";
    default: return "?";
    }
}
static const char *
prog_name(int p) {
    switch (p) {
    case FI_PROGRESS_UNSPEC: return "UNSPEC";
    case FI_PROGRESS_AUTO: return "AUTO";
    case FI_PROGRESS_MANUAL: return "MANUAL";
    default: return "?";
    }
}

/* Build hints matching NIXL's cxi rail branch, with knobs to vary. */
static struct fi_info *
build_hints(const char *dev, int with_hmem, int threading, int set_progress, int progress) {
    struct fi_info *h = fi_allocinfo();
    h->caps = FI_MSG | FI_RMA | FI_LOCAL_COMM | FI_REMOTE_COMM | FI_RMA_EVENT;
    if (with_hmem) h->caps |= FI_HMEM;
    h->mode = FI_CONTEXT;
    h->ep_attr->type = FI_EP_RDM;
    h->domain_attr->mr_mode = FI_MR_LOCAL | FI_MR_HMEM | FI_MR_VIRT_ADDR | FI_MR_ALLOCATED |
        FI_MR_PROV_KEY | FI_MR_ENDPOINT;
    if (dev) h->domain_attr->name = strdup(dev);
    h->domain_attr->threading = threading;
    if (set_progress) {
        h->domain_attr->control_progress = progress;
        h->domain_attr->data_progress = progress;
    }
    h->fabric_attr->prov_name = strdup("cxi");
    return h;
}

/* fi_getinfo -> fi_fabric -> fi_domain, reporting where it fails. */
static int
try_domain(const char *label, struct fi_info *hints) {
    struct fi_info *info = NULL;
    int ret = fi_getinfo(FI_VERSION(1, 18), NULL, NULL, 0, hints, &info);
    if (ret || !info) {
        printf("  [%-14s] fi_getinfo FAILED: %s\n", label, fi_strerror(-ret));
        fi_freeinfo(hints);
        return -1;
    }
    printf("  [%-14s] getinfo OK dom=%s thr=%s ctrl=%s data=%s mr_mode=0x%x caps=0x%lx\n",
           label,
           info->domain_attr->name ? info->domain_attr->name : "(null)",
           thr_name(info->domain_attr->threading),
           prog_name(info->domain_attr->control_progress),
           prog_name(info->domain_attr->data_progress),
           info->domain_attr->mr_mode,
           (unsigned long)info->caps);

    struct fid_fabric *fab = NULL;
    ret = fi_fabric(info->fabric_attr, &fab, NULL);
    if (ret) {
        printf("  [%-14s] fi_fabric FAILED: %s\n", label, fi_strerror(-ret));
        fi_freeinfo(info);
        fi_freeinfo(hints);
        return -1;
    }

    struct fid_domain *dom = NULL;
    ret = fi_domain(fab, info, &dom, NULL);
    if (ret) {
        printf("  [%-14s] fi_domain FAILED: %s\n", label, fi_strerror(-ret));
    } else {
        printf("  [%-14s] fi_domain SUCCESS  <=== works\n", label);
        fi_close(&dom->fid);
    }
    fi_close(&fab->fid);
    fi_freeinfo(info);
    fi_freeinfo(hints);
    return ret;
}

int
main(void) {
    /* 0. Discover cxi domain names using the KNOWN-WORKING discovery hints
     *    (mr_mode = ~3, the mask NIXL uses in getAvailableNetworkDevices). */
    struct fi_info *dh = fi_allocinfo();
    dh->caps = FI_MSG | FI_RMA | FI_LOCAL_COMM | FI_REMOTE_COMM;
    dh->mode = FI_CONTEXT;
    dh->ep_attr->type = FI_EP_RDM;
    dh->domain_attr->mr_mode = ~3;
    dh->fabric_attr->prov_name = strdup("cxi");

    struct fi_info *di = NULL;
    int ret = fi_getinfo(FI_VERSION(1, 18), NULL, NULL, 0, dh, &di);
    char dev[128] = "";
    if (!ret && di) {
        printf("Discovered cxi domains:\n");
        for (struct fi_info *c = di; c; c = c->next) {
            printf("  - %s (fabric=%s)\n", c->domain_attr->name, c->fabric_attr->name);
            if (!dev[0] && c->domain_attr->name) strncpy(dev, c->domain_attr->name, sizeof(dev) - 1);
        }
    } else {
        printf("discovery fi_getinfo failed: %s\n", fi_strerror(-ret));
    }
    if (di) fi_freeinfo(di);
    fi_freeinfo(dh);
    printf("Using device: %s\n\n", dev[0] ? dev : "(none / all)");
    const char *d = dev[0] ? dev : NULL;

    printf("== Baseline: exact NIXL rail hints (thr=COMPLETION, +HMEM, +RMA_EVENT) ==\n");
    try_domain("baseline", build_hints(d, 1, FI_THREAD_COMPLETION, 0, 0));

    printf("\n== Vary threading (keep +HMEM) ==\n");
    try_domain("thr=SAFE", build_hints(d, 1, FI_THREAD_SAFE, 0, 0));
    try_domain("thr=DOMAIN", build_hints(d, 1, FI_THREAD_DOMAIN, 0, 0));
    try_domain("thr=ENDPOINT", build_hints(d, 1, FI_THREAD_ENDPOINT, 0, 0));
    try_domain("thr=UNSPEC", build_hints(d, 1, FI_THREAD_UNSPEC, 0, 0));

    printf("\n== thr=COMPLETION + explicit progress=MANUAL ==\n");
    try_domain("compl+manual", build_hints(d, 1, FI_THREAD_COMPLETION, 1, FI_PROGRESS_MANUAL));

    printf("\n== drop FI_HMEM (thr=COMPLETION) ==\n");
    try_domain("no-hmem", build_hints(d, 0, FI_THREAD_COMPLETION, 0, 0));

    printf("\n== kitchen sink: thr=DOMAIN + progress=MANUAL + no HMEM ==\n");
    try_domain("sink", build_hints(d, 0, FI_THREAD_DOMAIN, 1, FI_PROGRESS_MANUAL));

    printf("\nDone. A line ending in '<=== works' identifies a viable domain_attr combo.\n");
    printf("If none work, fi_domain cannot create a cxi domain with NIXL's feature set\n");
    printf("on this libfabric/SHS build -> genuine provider-side limitation.\n");
    return 0;
}
