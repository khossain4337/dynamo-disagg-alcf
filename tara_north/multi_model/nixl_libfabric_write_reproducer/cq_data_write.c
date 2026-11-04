// RMA write with remote CQ data over cxi: fi_writemsg(FI_REMOTE_CQ_DATA) as NIXL issues it, then
// fi_writedata. One process writes to its own endpoint. Exit code = number of failed cases.
// Build: LF=/opt/cray/libfabric/2.3.1; gcc -O2 -o cq_data_write cq_data_write.c -I$LF/include -L$LF/lib64 -lfabric
// Run inside a PBS job (cxi needs the job VNI): mpiexec -n 1 ./cq_data_write
#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_rma.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define LEN 65536
#define CK(x) do { int r_ = (x); if (r_) { fprintf(stderr, "%s: %d (%s)\n", #x, r_, fi_strerror(-r_)); exit(100); } } while (0)

static struct fid_domain *dom;
static struct fid_ep *ep;
static struct fid_cq *cq;
static struct fi_context ctx;
static char *src, *dst;
static void *src_desc;
static uint64_t raddr, rkey;
static fi_addr_t self;

static struct fid_mr *reg(void *buf, uint64_t access)
{
    struct fid_mr *mr;
    CK(fi_mr_reg(dom, buf, LEN, access, 0, 0, 0, &mr, NULL));
    CK(fi_mr_bind(mr, &ep->fid, 0));
    CK(fi_mr_enable(mr));
    return mr;
}

// Both the initiator completion and the target's CQ-data completion land in the one CQ.
static int wait_both(uint64_t data)
{
    struct fi_cq_data_entry e;
    int local = 0, remote = 0;
    time_t t0 = time(NULL);
    while (!(local && remote) && time(NULL) - t0 < 5) {
        ssize_t n = fi_cq_read(cq, &e, 1);
        if (n == -FI_EAGAIN)
            continue;
        if (n == -FI_EAVAIL) {
            struct fi_cq_err_entry err = {0};
            fi_cq_readerr(cq, &err, 0);
            printf(", cq error %d (%s)", err.err, fi_strerror(err.err));
            return 1;
        }
        if (n < 0) {
            printf(", fi_cq_read %zd", n);
            return 1;
        }
        if (e.flags & FI_REMOTE_CQ_DATA) {
            remote = 1;
            if (e.data != data) {
                printf(", cq data 0x%lx want 0x%lx", (unsigned long)e.data, (unsigned long)data);
                return 1;
            }
        } else {
            local = 1;
        }
    }
    if (!(local && remote))
        printf(", timeout (local %d, remote %d)", local, remote);
    return !(local && remote);
}

static int run(const char *name, int use_msg, uint64_t data)
{
    ssize_t ret;
    memset(src, (int)data, LEN);
    memset(dst, 0, LEN);
    if (use_msg) {
        struct iovec iov = {src, LEN};
        struct fi_rma_iov rma = {raddr, LEN, rkey};
        struct fi_msg_rma msg = {.msg_iov = &iov, .desc = &src_desc, .iov_count = 1, .addr = self,
                                 .rma_iov = &rma, .rma_iov_count = 1, .context = &ctx, .data = data};
        ret = fi_writemsg(ep, &msg, FI_REMOTE_CQ_DATA);
    } else {
        ret = fi_writedata(ep, src, LEN, src_desc, data, self, raddr, rkey, &ctx);
    }
    printf("%-32s post %zd (%s)", name, ret, ret ? fi_strerror((int)-ret) : "ok");
    int bad = ret != 0;
    if (!bad) {
        bad = wait_both(data) || memcmp(src, dst, LEN) != 0;
        printf(bad ? ", FAIL" : ", payload and cq data delivered");
    }
    printf("\n");
    return bad;
}

int main(void)
{
    struct fi_info *hints = fi_allocinfo(), *info;
    struct fid_fabric *fab;
    struct fid_av *av;
    // Same caps and mr_mode NIXL 1.5.2 asks of cxi (libfabric_rail.cpp:463-466).
    hints->caps = FI_RMA | FI_RMA_EVENT | FI_LOCAL_COMM | FI_REMOTE_COMM;
    hints->mode = FI_CONTEXT;
    hints->ep_attr->type = FI_EP_RDM;
    hints->domain_attr->mr_mode = FI_MR_LOCAL | FI_MR_VIRT_ADDR | FI_MR_ALLOCATED | FI_MR_PROV_KEY | FI_MR_ENDPOINT;
    hints->fabric_attr->prov_name = strdup("cxi");
    CK(fi_getinfo(FI_VERSION(1, 18), NULL, NULL, 0, hints, &info));
    printf("libfabric api %u.%u, provider %s %u.%u, domain %s\n", FI_MAJOR(fi_version()), FI_MINOR(fi_version()),
           info->fabric_attr->prov_name, FI_MAJOR(info->fabric_attr->prov_version),
           FI_MINOR(info->fabric_attr->prov_version), info->domain_attr->name);

    struct fi_cq_attr cqa = {.format = FI_CQ_FORMAT_DATA};
    struct fi_av_attr ava = {0};
    CK(fi_fabric(info->fabric_attr, &fab, NULL));
    CK(fi_domain(fab, info, &dom, NULL));
    CK(fi_cq_open(dom, &cqa, &cq, NULL));
    CK(fi_av_open(dom, &ava, &av, NULL));
    CK(fi_endpoint(dom, info, &ep, NULL));
    CK(fi_ep_bind(ep, &cq->fid, FI_TRANSMIT | FI_RECV));
    CK(fi_ep_bind(ep, &av->fid, 0));
    CK(fi_enable(ep));

    char name[256];
    size_t len = sizeof(name);
    CK(fi_getname(&ep->fid, name, &len));
    if (fi_av_insert(av, name, 1, &self, 0, NULL) != 1) {
        fprintf(stderr, "fi_av_insert failed\n");
        return 100;
    }

    src = aligned_alloc(4096, LEN);
    dst = aligned_alloc(4096, LEN);
    src_desc = fi_mr_desc(reg(src, FI_WRITE));
    rkey = fi_mr_key(reg(dst, FI_REMOTE_WRITE));
    raddr = (info->domain_attr->mr_mode & FI_MR_VIRT_ADDR) ? (uint64_t)(uintptr_t)dst : 0;

    return run("fi_writemsg + FI_REMOTE_CQ_DATA", 1, 0x5a) + run("fi_writedata", 0, 0xa5);
}
