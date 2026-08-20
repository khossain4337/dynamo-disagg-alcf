/*
 * cxi_domain_probe2.c
 * -------------------
 * Follow-up to cxi_domain_probe.c. Probe v1 showed fi_domain returns -FI_ENOSYS
 * for cxi across ALL threading/progress/HMEM combos, while fi_getinfo always
 * succeeds. This narrows it to two remaining suspects:
 *
 *   (a) NIXL's hint set itself (test truly minimal, discovery-style hints), and
 *   (b) the requested libfabric API VERSION -- NIXL hardcodes FI_VERSION(1,18),
 *       but this system is libfabric 2.3. A too-old requested API version is a
 *       classic cause of fi_domain -FI_ENOSYS on a newer provider.
 *
 * This sweeps both. If a minimal-hints or higher-version call makes fi_domain
 * SUCCEED, the fix is NIXL-side (change hints and/or bump the version). If every
 * variant still ENOSYS, it's a genuine CXI-provider / SHS limitation.
 *
 * Build: gcc -o cxi_domain_probe2 cxi_domain_probe2.c $(pkg-config --cflags --libs libfabric)
 * Run:   ./cxi_domain_probe2
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <rdma/fabric.h>
#include <rdma/fi_domain.h>

/* "minimal / discovery-style" hints == what NIXL's getAvailableNetworkDevices uses
 * (mr_mode = ~3), which is the call that WORKS at the fi_getinfo layer. */
static struct fi_info *
build_discovery(const char *dev) {
    struct fi_info *h = fi_allocinfo();
    h->caps = FI_MSG | FI_RMA | FI_LOCAL_COMM | FI_REMOTE_COMM;
    h->mode = FI_CONTEXT;
    h->ep_attr->type = FI_EP_RDM;
    h->domain_attr->mr_mode = ~3;
    if (dev) h->domain_attr->name = strdup(dev);
    h->fabric_attr->prov_name = strdup("cxi");
    return h;
}

/* Full NIXL cxi rail hints (matches libfabric_rail.cpp cxi branch). */
static struct fi_info *
build_rail(const char *dev) {
    struct fi_info *h = fi_allocinfo();
    h->caps = FI_MSG | FI_RMA | FI_HMEM | FI_LOCAL_COMM | FI_REMOTE_COMM | FI_RMA_EVENT;
    h->mode = FI_CONTEXT;
    h->ep_attr->type = FI_EP_RDM;
    h->domain_attr->mr_mode = FI_MR_LOCAL | FI_MR_HMEM | FI_MR_VIRT_ADDR | FI_MR_ALLOCATED |
        FI_MR_PROV_KEY | FI_MR_ENDPOINT;
    if (dev) h->domain_attr->name = strdup(dev);
    h->domain_attr->threading = FI_THREAD_COMPLETION;
    h->fabric_attr->prov_name = strdup("cxi");
    return h;
}

static int
try_domain(const char *label, uint32_t version, struct fi_info *hints) {
    struct fi_info *info = NULL;
    int ret = fi_getinfo(version, NULL, NULL, 0, hints, &info);
    if (ret || !info) {
        printf("  [%-22s] fi_getinfo FAILED: %s (%d)\n", label, fi_strerror(-ret), ret);
        fi_freeinfo(hints);
        return -1;
    }
    printf("  [%-22s] getinfo OK (api=%u.%u) dom=%s mr_mode=0x%x\n",
           label,
           FI_MAJOR(info->fabric_attr->api_version),
           FI_MINOR(info->fabric_attr->api_version),
           info->domain_attr->name ? info->domain_attr->name : "(null)",
           info->domain_attr->mr_mode);

    struct fid_fabric *fab = NULL;
    ret = fi_fabric(info->fabric_attr, &fab, NULL);
    if (ret) {
        printf("  [%-22s] fi_fabric FAILED: %s (%d)\n", label, fi_strerror(-ret), ret);
        fi_freeinfo(info);
        fi_freeinfo(hints);
        return -1;
    }

    struct fid_domain *dom = NULL;
    ret = fi_domain(fab, info, &dom, NULL);
    if (ret) {
        printf("  [%-22s] fi_domain FAILED: %s (%d)\n", label, fi_strerror(-ret), ret);
    } else {
        printf("  [%-22s] fi_domain SUCCESS  <=== works\n", label);
        fi_close(&dom->fid);
    }
    fi_close(&fab->fid);
    fi_freeinfo(info);
    fi_freeinfo(hints);
    return ret;
}

int
main(void) {
    /* discover a device name */
    struct fi_info *di = NULL;
    char dev[128] = "";
    if (fi_getinfo(FI_VERSION(1, 18), NULL, NULL, 0, build_discovery(NULL), &di) == 0 && di) {
        for (struct fi_info *c = di; c; c = c->next)
            if (!dev[0] && c->domain_attr->name) strncpy(dev, c->domain_attr->name, sizeof(dev) - 1);
        fi_freeinfo(di);
    }
    printf("Using device: %s\n\n", dev[0] ? dev : "(none/all)");
    const char *d = dev[0] ? dev : NULL;

    printf("== (a) Minimal discovery-style hints (mr_mode=~3), various versions ==\n");
    try_domain("disc no-name v1.18", FI_VERSION(1, 18), build_discovery(NULL));
    try_domain("disc cxi0   v1.18", FI_VERSION(1, 18), build_discovery(d));
    try_domain("disc cxi0   v2.3", FI_VERSION(2, 3), build_discovery(d));

    printf("\n== (b) Full NIXL rail hints, VERSION sweep ==\n");
    try_domain("rail cxi0   v1.15", FI_VERSION(1, 15), build_rail(d));
    try_domain("rail cxi0   v1.18", FI_VERSION(1, 18), build_rail(d)); /* NIXL's value */
    try_domain("rail cxi0   v1.20", FI_VERSION(1, 20), build_rail(d));
    try_domain("rail cxi0   v1.21", FI_VERSION(1, 21), build_rail(d));
    try_domain("rail cxi0   v2.0", FI_VERSION(2, 0), build_rail(d));
    try_domain("rail cxi0   v2.3", FI_VERSION(2, 3), build_rail(d));

    printf("\nDone. Any '<=== works' line => NIXL-side fix (adopt those hints/version).\n");
    printf("All ENOSYS => fi_domain on cxi is unavailable to plain libfabric here\n");
    printf("(environmental/provider) -- cross-check with: fi_pingpong -p cxi\n");
    return 0;
}
