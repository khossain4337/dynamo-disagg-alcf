#!/usr/bin/env python3
"""
Two-rank NIXL bulk-transfer test over Slingshot/CXI.

WHY THIS EXISTS
---------------
repro_libfabric_topology.py proves only that createBackend("LIBFABRIC")
succeeds -- i.e. that fi_getinfo + fi_domain get past the mr_mode bug (shim)
and the FI_ENOSYS wall (PALS-provisioned SLINGSHOT_VNIS). It creates two
agents that never speak to each other. It cannot answer Q3.

This script closes that gap: two agents on two DIFFERENT nodes exchange
metadata, register a large buffer, and push multiple GiB through NIXL with
the LIBFABRIC backend. It answers three things a vllm serve run answers
only expensively and ambiguously:

  1. Does a cross-node NIXL transfer complete at all under a shared VNI?
  2. Is the data correct (byte-exact), not just "the call returned DONE"?
  3. Did it actually traverse Slingshot? -> per-NIC CXI hardware octet
     counters are sampled before/after on BOTH ranks. CXI RDMA is
     kernel-bypass and never shows in /proc/net/dev, so these sysfs
     counters are the only honest signal. A tx delta on the initiator and
     an rx delta on the target, each >= the payload size, is direct
     evidence -- not an inference from timing or from "we never configured
     bond0".

HOW TO RUN
----------
Always through the wrapper, which handles env + LD_PRELOAD + mpiexec:

    bash run_repro_nixl_2rank_transfer.sh                 # self-launches mpiexec
    mpiexec -n 2 -ppn 1 bash run_repro_nixl_2rank_transfer.sh   # equivalent

It MUST run under a SINGLE `mpiexec` spanning both ranks. Three measured
facts, in the order they matter:

  - No VNI means fi_domain() returns -FI_ENOSYS for the cxi provider. That
    is the wall this whole exercise is about.
  - A bare `mpiexec -n 1` yields an EMPTY SLINGSHOT_VNIS. Adding
    `--single-node-vni` DOES provision one for a single-rank launch, so
    "one mpiexec per engine" looks viable at first glance.
  - It is not. The VNI is allocated per APPLICATION, and a VNI is a fabric
    traffic-isolation domain: endpoints on different VNIs cannot exchange
    traffic at all. Three separate launches were observed to get three
    different VNIs (1726, 1825, and 1873 -- the last from `-n 1
    --single-node-vni`). So two independently-launched ranks would each
    create their LIBFABRIC agent successfully and then never reach each
    other -- trading today's loud, immediate ENOSYS for a silent hang.

`--single-node-vni` therefore has exactly one legitimate use here:
exercising this script's plumbing inside a 1-node allocation. The transfer
will still abort on the distinct-hosts guard below, by design -- a
same-node "success" proves nothing, since it could be served by shm or cxi
loopback rather than the fabric.

MODES
-----
NIXL partitions a node's NICs among its GPUs 1:1. groupNicsWithAccel() in
libfabric_topology.cpp computes `nics.size() / num_groups`, and Tara has 4
GPUs and 4 Slingshot NICs, so a VRAM transfer gets exactly ONE rail per GPU.
There is no knob for this -- the only LIBFABRIC tunable, max_bw_per_dram_seg,
applies to DRAM_SEG. It is the EFA design assumption (a p5 has 8 GPUs and 32
NICs, i.e. 4 each, and expects one rank per GPU) meeting a 4:4 machine. The
three modes measure around it rather than fight it:

  --mode pair       (default) 1 rank per node, GPU 0 -> 1 rail.
                    MEASURED: 23.87 GB/s, which is 95.5% of one 200 Gbps
                    NIC. This is the honest per-GPU number, and the low
                    total is rail count, not transport inefficiency.

  --mode aggregate  N ranks per node, rank i on GPU i, every pair in flight
                    at once. This is the shape vLLM actually runs (TP=N, one
                    rank per GPU), so the node total is the figure that
                    matters for the deployment. Launch with -n 2N -ppn N.

  --mode solo-peak  1 rank per node, but the wrapper hands hwloc an XML
                    topology with all but one GPU deleted. num_groups
                    becomes 1, so that GPU inherits all 4 NICs. Diagnostic
                    only: it shows a single GPU can saturate the fabric. It
                    is not a configuration anyone should ship.

DRAM is unaffected by any of this. Its rail policy is separate, and on this
node it falls back to "all rails" because NUMA detection fails -- which is
why DRAM reads ~87 GB/s from a single rank while VRAM reads ~24.

DESIGN NOTES
------------
Metadata and descriptor exchange go through plain files on /vast, not
NIXL's TCP side channel (send_local_metadata/fetch_remote_metadata). That
is deliberate: it keeps every socket out of the picture so nothing can
quietly carry payload over bond0, and it avoids depending on the
listener-thread constructor signature, which differs across NIXL versions.
The only non-CXI traffic here is a few KiB of filesystem metadata.
"""

import argparse
import os
import socket
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------------
# CXI hardware counters
# --------------------------------------------------------------------------
# Kernel-bypass RDMA never touches /proc/net/dev, so these per-NIC sysfs
# telemetry files are the only place the bytes show up. Format is
# "<value>@<timestamp>" on this platform; parse defensively anyway.
#
# The value is NOT read straight off the hardware -- it comes from a per-NIC
# cache that the driver refreshes on its own schedule, and the timestamp is
# when that refresh happened, not when we opened the file. Everything below
# about settling exists because of that one fact.

CXI_TELEMETRY = "/sys/class/cxi/cxi{i}/device/telemetry/hni_sts_{dir}_ok_octets"


def read_cxi_counters(max_nics=8):
    """One instantaneous sample: ({key: octets}, {key: raw_timestamp}).

    The timestamp's units are not documented on this platform, so it is
    carried through as an opaque string and only logged -- nothing branches
    on it. It is here so the next run tells us what it is, and so a stale
    sample is visible rather than inferred.
    """
    counters, stamps = {}, {}
    for i in range(max_nics):
        for direction in ("tx", "rx"):
            path = CXI_TELEMETRY.format(i=i, dir=direction)
            try:
                with open(path) as fh:
                    raw = fh.read().strip()
            except OSError:
                continue
            head, _, stamp = raw.partition("@")
            fields = head.split()
            if not fields:
                continue
            try:
                counters[f"cxi{i}.{direction}"] = int(fields[0])
            except ValueError:
                continue
            if stamp.strip():
                stamps[f"cxi{i}.{direction}"] = stamp.strip()
    return counters, stamps


def read_cxi_counters_settled(window, poll=0.25, timeout=30.0):
    """Sample until the counters hold still for `window` seconds.

    A fixed sleep is a guess, and "two consecutive reads agree" is worse than
    a guess: two reads landing inside one refresh interval agree trivially
    while the value is still stale. Only stillness across a window LONGER
    than the refresh interval shows the cache has caught up. `window` must
    therefore over-estimate that interval, which is why it defaults high.

    This is not hypothetical. Sampling the target the instant the transfer
    ended gave tx totals of 3.05 / 3.09 / 3.11 / 3.12 GiB across cxi0..cxi3
    -- ascending in read order, a ~75 MB staleness gradient -- summing to
    0.77x of payload. The initiator's four NICs, which happened to have
    settled, agreed with each other to within 512 bytes and summed to 1.02x.
    Four settled NICs agree; four unsettled ones fan out in read order.

    Returns (counters, stamps, info); info carries waited/reads/settled.
    """
    # Exact equality is too strict to ever be reached on a shared node: /vast
    # rides the same fabric, so barrier polling and other background chatter
    # keep the counters ticking and nothing would ever settle. The tolerance
    # below accepts drift of ~4 MiB/s at the default poll (so <= 8 MiB per
    # counter across a 2s window) -- four orders of magnitude under the
    # ~60 GB/s the transfer itself moves, so it cannot mistake a transfer
    # tail for quiet, and it cannot hide a missing pass.
    tol = 1 << 20

    def _quiet(a, b):
        return all(abs(a.get(k, 0) - b.get(k, 0)) <= tol for k in set(a) | set(b))

    t0 = time.perf_counter()
    prev, stamps = read_cxi_counters()
    stable_since, reads = t0, 1
    while True:
        time.sleep(poll)
        cur, stamps = read_cxi_counters()
        reads += 1
        now = time.perf_counter()
        if not _quiet(cur, prev):
            stable_since = now
        prev = cur
        if now - stable_since >= window:
            return cur, stamps, {"waited": now - t0, "reads": reads, "settled": True}
        if now - t0 >= timeout:
            return cur, stamps, {"waited": now - t0, "reads": reads, "settled": False}


def diff_counters(before, after):
    return {k: after[k] - before.get(k, 0) for k in sorted(after) if after[k] - before.get(k, 0) > 0}


# --------------------------------------------------------------------------
# Filesystem rendezvous
# --------------------------------------------------------------------------
# /vast is shared, but treat it like a network filesystem: fsync on write,
# and probe with listdir() rather than Path.exists() so we are not defeated
# by cached negative dentries.


def _publish(sync_dir: Path, name: str, payload: bytes = b"1"):
    tmp = sync_dir / f".{name}.tmp.{os.getpid()}"
    with open(tmp, "wb") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, sync_dir / name)


def _await(sync_dir: Path, name: str, timeout: float, poll=0.05, on_poll=None):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if name in os.listdir(sync_dir):
            with open(sync_dir / name, "rb") as fh:
                return fh.read()
        if on_poll is not None:
            on_poll()
        time.sleep(poll)
    raise TimeoutError(f"timed out after {timeout:.0f}s waiting for {sync_dir/name}")


def barrier(sync_dir: Path, tag: str, rank: int, size: int, timeout: float = 600.0):
    _publish(sync_dir, f"{tag}.{rank}")
    deadline = time.time() + timeout
    want = {f"{tag}.{r}" for r in range(size)}
    while time.time() < deadline:
        if want <= set(os.listdir(sync_dir)):
            return
        time.sleep(0.05)
    raise TimeoutError(f"barrier '{tag}' timed out on rank {rank}")


# --------------------------------------------------------------------------

SYNC_ROOT = "/vast/draco/tara/projects/Tara_Deployment/software/testing"


def human(nbytes):
    return f"{nbytes / 2**30:.2f} GiB"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gib", type=float, default=4.0,
                    help="buffer size per rank, in GiB (default: 4)")
    ap.add_argument("--chunk-mib", type=int, default=64,
                    help="descriptor granularity in MiB; more chunks give the "
                         "rail manager more to stripe across the 4 NICs (default: 64)")
    ap.add_argument("--iters", type=int, default=3,
                    help="timed iterations after the warmup (default: 3)")
    ap.add_argument("--warmup", type=int, default=1,
                    help="untimed warmup iterations, for connection setup (default: 1)")
    ap.add_argument("--op", choices=("READ", "WRITE"), default="READ",
                    help="READ = initiator pulls (default), which is both the direction "
                         "vLLM's NixlConnector uses and the only one that works on this "
                         "stack. WRITE = initiator pushes; NIXL posts it with "
                         "fi_writedata(), which this CXI/libfabric build rejects with "
                         "-FI_EBADFLAGS and NIXL 1.4.0 has no fallback for. Kept "
                         "selectable so the failure stays reproducible on demand.")
    ap.add_argument("--mem", choices=("dram", "cuda"), default="dram",
                    help="where the buffer lives. dram first -- it isolates the fabric "
                         "from HMEM/GPUDirect concerns. cuda is the real KV-cache path.")
    ap.add_argument("--mode", choices=("pair", "aggregate", "solo-peak"), default="pair",
                    help="pair = 1 rank per node on GPU 0, which on Slingshot gets ONE of "
                         "the 4 rails (see MODES in the module docstring). aggregate = N "
                         "ranks per node, rank i on GPU i, all transferring concurrently; "
                         "reports the node total, and is what vLLM TP=N actually does. "
                         "solo-peak = 1 rank per node with HWLOC_XMLFILE pruned to a single "
                         "GPU so that GPU inherits all 4 rails (diagnostic only). "
                         "(default: pair)")
    ap.add_argument("--backend", default="LIBFABRIC")
    ap.add_argument("--sync-dir", default=None,
                    help="shared rendezvous dir. Default is derived from PALS_APID under "
                         "the /vast testing tree, which makes it unique per run; if you "
                         "override it, give each run a fresh empty directory")
    ap.add_argument("--settle", type=float, default=2.0,
                    help="seconds the CXI counters must hold STILL before a sample is "
                         "accepted. Must over-estimate the per-NIC telemetry refresh "
                         "interval, or a stale value passes the stability test trivially "
                         "(default: 2.0)")
    ap.add_argument("--settle-timeout", type=float, default=30.0,
                    help="stop waiting for the counters to settle after this long and "
                         "report the sample as unsettled (default: 30)")
    ap.add_argument("--telemetry", action="store_true",
                    help="call agent.get_xfer_telemetry(); off by default because NIXL "
                         "logs a loud error unless telemetry was enabled at agent creation")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip byte-exactness check (it is not free on a multi-GiB buffer)")
    ap.add_argument("--dump-api", action="store_true",
                    help="print the nixl_agent method list and exit -- use this if a call "
                         "below turns out not to exist in this NIXL build")
    args = ap.parse_args()

    rank_env = os.environ.get("PMI_RANK", os.environ.get("PALS_RANKID"))
    size_env = os.environ.get("PMI_SIZE", os.environ.get("PALS_NRANKS"))
    if rank_env is None:
        sys.exit("ERROR: no PMI_RANK in env -- this must run under mpiexec.\n"
                 "       Outside PALS there is no SLINGSHOT_VNIS, and fi_domain() will "
                 "return -FI_ENOSYS for cxi.")
    rank, size = int(rank_env), int(size_env or 2)
    if size % 2 != 0:
        sys.exit(f"ERROR: need an even rank count -- pairs across two nodes -- got {size}.")

    host = socket.gethostname().split(".")[0]

    # role/pair_id are derived from the hostname exchange below, not from the
    # rank number; until that has happened, identify by rank alone.
    label = f"r{rank}"

    def log(msg):
        print(f"[{label} {host}] {msg}", flush=True)

    vni = os.environ.get("SLINGSHOT_VNIS", "")
    log(f"SLINGSHOT_VNIS={vni or '<EMPTY>'} DEVICES={os.environ.get('SLINGSHOT_DEVICES', '?')} "
        f"SVC_IDS={os.environ.get('SLINGSHOT_SVC_IDS', '?')} TCS={os.environ.get('SLINGSHOT_TCS', '?')}")
    if not vni:
        log("WARNING: SLINGSHOT_VNIS is empty. fi_domain() will almost certainly fail.")

    fi_env = {k: v for k, v in sorted(os.environ.items()) if k.startswith("FI_")}
    log(f"FI_* env: {fi_env or '<none>'}")
    if os.environ.get("FI_CXI_ENABLE_WRITEDATA") not in ("1", "true", "yes"):
        log("WARNING: FI_CXI_ENABLE_WRITEDATA is not set. NIXL posts its data path with "
            "fi_writedata(), which CXI disables by default -- expect "
            "'fi_writedata failed ...: Flags not supported'.")

    # ---- Rendezvous directory ---------------------------------------------
    # This MUST be unique per run. If two runs share it, every barrier passes
    # instantly on the previous run's files and add_remote_agent() gets handed
    # the PREVIOUS run's metadata -- a silent wrong-answer failure, not a
    # crash. PALS hands out a per-application id; with no id there is nothing
    # safe to derive, so refuse instead of guessing.
    if args.sync_dir:
        sync_dir = Path(args.sync_dir)
    else:
        apid = os.environ.get("PALS_APID")
        if not apid:
            sys.exit("ERROR: PALS_APID is unset, so the default rendezvous directory "
                     "would be shared by every run: barriers would pass on stale files "
                     "and the metadata exchange would wire up a dead agent from a "
                     "previous launch. Pass --sync-dir <fresh empty dir> explicitly.")
        sync_dir = Path(SYNC_ROOT) / f"nixl_2rank_{apid}"
    sync_dir.mkdir(parents=True, exist_ok=True)

    # Belt and braces for an explicit --sync-dir: anything in here that
    # predates this launch is debris. The peer's files are seconds old at
    # this point -- both ranks come from one mpiexec -- so age separates the
    # two cleanly. stat() can lose a race against the peer's atomic replace;
    # a file that vanishes mid-scan is by definition not stale.
    now = time.time()
    stale = []
    for p in sync_dir.iterdir():
        try:
            if now - p.stat().st_mtime > 300.0:
                stale.append(p.name)
        except OSError:
            continue
    if stale:
        sys.exit(f"ERROR: {sync_dir} holds {len(stale)} file(s) older than this launch "
                 f"(e.g. {sorted(stale)[:3]}). Barriers would pass immediately on them. "
                 f"Delete them or pass a fresh --sync-dir.")
    log(f"rendezvous: {sync_dir}")

    # ---- Guard + pairing: ranks must span exactly two nodes ----------------
    # A same-node pass could be served by shm or cxi loopback and would be
    # worthless as evidence for a cross-node KV transfer.
    #
    # Pairing is derived from the EXCHANGED HOSTNAMES, not from arithmetic on
    # the rank number. `peer = rank +/- npairs` would silently assume PALS
    # hands out ranks block-wise (0..N-1 on node A, N..2N-1 on node B); if a
    # launcher ever round-robins instead, that assumption pairs ranks on the
    # SAME node and the distinct-hosts guard below would be the only thing
    # catching it. Grouping by hostname holds under any ordering.
    _publish(sync_dir, f"host.{rank}", host.encode())
    barrier(sync_dir, "hosts", rank, size)
    hosts = [(sync_dir / f"host.{r}").read_bytes().decode() for r in range(size)]

    distinct = sorted(set(hosts))
    if len(distinct) != 2:
        sys.exit(f"ERROR: need exactly 2 distinct nodes, saw {len(distinct)}: {distinct}. "
                 f"Use -ppn <N> with -n <2N> across 2 nodes; a same-node transfer proves "
                 f"nothing about the fabric.")

    init_host = hosts[0]                       # the node holding global rank 0
    targ_host = next(h for h in distinct if h != init_host)
    init_ranks = [r for r in range(size) if hosts[r] == init_host]
    targ_ranks = [r for r in range(size) if hosts[r] == targ_host]
    if len(init_ranks) != len(targ_ranks):
        sys.exit(f"ERROR: uneven ranks per node -- {len(init_ranks)} on {init_host}, "
                 f"{len(targ_ranks)} on {targ_host}. Pairing requires -ppn <N> with "
                 f"-n <2N>.")
    npairs = len(init_ranks)

    is_initiator = (host == init_host)
    pair_id = (init_ranks if is_initiator else targ_ranks).index(rank)
    peer_rank = (targ_ranks if is_initiator else init_ranks)[pair_id]

    role = "initiator" if is_initiator else "target"
    peer_role = "target" if is_initiator else "initiator"
    label = f"{role} r{rank} p{pair_id}"

    # Exactly one rank per node samples the CXI counters. They live in
    # /sys/class/cxi/cxiN/ and are per-NIC, i.e. NODE-WIDE -- every rank on
    # the node reads the same numbers. If all N sampled, each would report
    # the whole node's traffic as its own and the totals would be N x too
    # big. pair 0 samples, and what it reports is the NODE total.
    samples_counters = (pair_id == 0)

    # Mode/launch-shape consistency.
    if args.mode == "aggregate":
        if npairs == 1:
            log("WARNING: --mode aggregate with 1 rank per node is identical to --mode "
                "pair. Relaunch with -ppn <#GPUs> -n <2 x #GPUs> to exercise all rails.")
    elif npairs != 1:
        sys.exit(f"ERROR: --mode {args.mode} expects ONE rank per node, got {npairs}. "
                 f"Use -ppn 1 -n 2, or switch to --mode aggregate.")

    if args.mode == "solo-peak":
        xmlfile = os.environ.get("HWLOC_XMLFILE")
        if not xmlfile:
            sys.exit("ERROR: --mode solo-peak needs HWLOC_XMLFILE pointing at a topology "
                     "with all but one GPU deleted -- that is what makes num_groups 1 so "
                     "the surviving GPU inherits all 4 NICs. Run through the wrapper, "
                     "which builds it.")
        log(f"solo-peak: HWLOC_XMLFILE={xmlfile}")

    log(f"pair {pair_id} of {npairs}: peer is {peer_role} r{peer_rank} on {targ_host if is_initiator else init_host}"
        f" -- distinct nodes, good")

    # ---- Agent ------------------------------------------------------------
    import torch
    from nixl._api import nixl_agent, nixl_agent_config

    # The name has to be unique across the WHOLE job, not just the pair --
    # in aggregate mode there are npairs initiators, and two agents sharing a
    # name is a silent mis-routing waiting to happen. pair_id disambiguates.
    agent = nixl_agent(f"{role}-p{pair_id}", nixl_agent_config(backends=[args.backend]))
    log(f"NIXL agent up with backend {args.backend}")
    if args.dump_api:
        log("agent API: " + ", ".join(sorted(m for m in dir(agent) if not m.startswith("_"))))
        return

    # ---- Buffer -----------------------------------------------------------
    # int64 elements so the verification pattern (value == index) is exact
    # and regenerable without keeping a second copy around.
    nbytes = int(args.gib * 2**30)
    chunk_bytes = args.chunk_mib * 2**20
    nbytes -= nbytes % chunk_bytes          # keep chunks uniform
    nelem = nbytes // 8
    nchunks = nbytes // chunk_bytes

    # In aggregate mode pair i drives GPU i, and that is precisely what lights
    # up rail i: NIXL's 1:1 GPU->NIC partition sends each GPU's traffic down
    # its own NIC, so N concurrent pairs occupy N rails. Every other mode
    # stays on GPU 0 and therefore on one rail.
    gpu_index = pair_id if args.mode == "aggregate" else 0
    device = f"cuda:{gpu_index}" if args.mem == "cuda" else "cpu"
    log(f"allocating {human(nbytes)} on {device} as {nchunks} x {args.chunk_mib} MiB descriptors")
    buf = torch.empty(nelem, dtype=torch.int64, device=device)

    # Source side gets the pattern, destination side gets zeros, so a
    # no-op transfer cannot masquerade as success.
    src_is_local = ((args.op == "WRITE" and is_initiator) or
                    (args.op == "READ" and not is_initiator))
    if src_is_local:
        torch.arange(nelem, out=buf)
    else:
        buf.zero_()
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    reg = agent.register_memory(buf)
    if not reg:
        sys.exit("ERROR: register_memory failed.")
    log("memory registered")

    # ---- Descriptors ------------------------------------------------------
    dev_id = buf.get_device()
    if dev_id == -1:
        dev_id = 0
    base = buf.data_ptr()
    tuples = [(base + i * chunk_bytes, chunk_bytes, dev_id) for i in range(nchunks)]

    # agent.nixl_mems is the authoritative table of accepted mem_type
    # spellings in this build -- consult it rather than guessing. Upstream
    # examples use "cuda"/"cpu"; older builds only know "VRAM"/"DRAM".
    accepted = getattr(agent, "nixl_mems", None) or {}
    wanted = ("cuda", "VRAM") if device.startswith("cuda") else ("cpu", "DRAM")
    mem_type = next((m for m in wanted if m in accepted), wanted[-1])
    log(f"mem_type='{mem_type}' (agent.nixl_mems knows: {sorted(accepted) or 'unavailable'})")

    local_descs = agent.get_xfer_descs(tuples, mem_type=mem_type)
    if not local_descs:
        sys.exit("ERROR: get_xfer_descs returned nothing.")

    # ---- Metadata + descriptor exchange over the filesystem ---------------
    _publish(sync_dir, f"meta.{rank}", agent.get_agent_metadata())
    _publish(sync_dir, f"descs.{rank}", agent.get_serialized_descs(local_descs))
    barrier(sync_dir, "exchange", rank, size)

    peer_meta = (sync_dir / f"meta.{peer_rank}").read_bytes()
    peer_name = agent.add_remote_agent(peer_meta)
    if isinstance(peer_name, bytes):
        peer_name = peer_name.decode()
    log(f"added remote agent '{peer_name}'")

    remote_descs = agent.deserialize_descs((sync_dir / f"descs.{peer_rank}").read_bytes())

    # Some backends want an explicit connect; others do it lazily on first
    # transfer. Harmless either way.
    if hasattr(agent, "make_connection"):
        try:
            agent.make_connection(peer_name)
            log("make_connection() ok")
        except Exception as exc:
            log(f"make_connection() raised ({exc}) -- continuing, transfer may connect lazily")

    barrier(sync_dir, "connected", rank, size)

    # ---- Transfer ---------------------------------------------------------
    # Settle the baseline too, not just the post-transfer sample. A stale
    # baseline reads LOW, which inflates the delta -- the opposite error, and
    # the one that would manufacture a false pass.
    before, before_stamps = {}, {}
    if samples_counters:
        before, before_stamps, before_info = read_cxi_counters_settled(
            args.settle, timeout=args.settle_timeout)
        log(f"baseline counters settled in {before_info['waited']:.2f}s over "
            f"{before_info['reads']} reads"
            + ("" if before_info["settled"] else "  -- NOT STABLE; baseline may be stale"))

    # Neither rank may move data until BOTH hold a baseline. Without this the
    # initiator finishes settling first and starts transferring while the
    # target is still inside its settle loop -- the target's counters would
    # never hold still, so its "before" would land on the far side of the
    # transfer and its delta would collapse to near zero.
    barrier(sync_dir, "armed", rank, size)

    if is_initiator:
        handle = agent.initialize_xfer(args.op, local_descs, remote_descs, peer_name, b"xfer_done")
        if not handle:
            sys.exit("ERROR: initialize_xfer returned no handle.")

        # Q3, at the NIXL layer. This asks NIXL which backend it actually
        # bound this transfer to -- attribution independent of the hardware
        # counters below. If this says LIBFABRIC *and* the CXI octets move,
        # there is no room left for a silent fallback.
        try:
            chosen = agent.query_xfer_backend(handle)
            log(f"query_xfer_backend -> {chosen}")
            if chosen and args.backend.upper() not in str(chosen).upper():
                log(f"WARNING: transfer bound to '{chosen}', not {args.backend}.")
        except Exception as exc:
            log(f"query_xfer_backend unavailable ({exc}) -- relying on CXI counters alone")

        def one_pass():
            t0 = time.perf_counter()
            if agent.transfer(handle) == "ERR":
                sys.exit("ERROR: posting the transfer failed.")
            while True:
                state = agent.check_xfer_state(handle)
                if state == "ERR":
                    sys.exit("ERROR: transfer entered the Error state.")
                if state == "DONE":
                    return time.perf_counter() - t0

        for i in range(args.warmup):
            dt = one_pass()
            log(f"warmup {i}: {dt:.3f}s  ({nbytes / dt / 1e9:.2f} GB/s)")

        times = []
        for i in range(args.iters):
            dt = one_pass()
            times.append(dt)
            log(f"iter {i}: {dt:.3f}s  {nbytes / dt / 1e9:.2f} GB/s")

        _publish(sync_dir, f"xfer_complete.p{pair_id}")

        if args.telemetry:
            try:
                tele = agent.get_xfer_telemetry(handle)
                if tele:
                    log(f"NIXL transfer telemetry: {tele}")
            except Exception as exc:
                log(f"get_xfer_telemetry unavailable: {exc}")

        best = min(times)
        mean_s = sum(times) / len(times)
        log("")
        log(f"=== {args.op} {human(nbytes)} x{args.iters} timed ({args.mem}, {nchunks} descriptors) ===")
        log(f"best  {best:.3f}s -> {nbytes / best / 1e9:.2f} GB/s")
        log(f"mean  {mean_s:.3f}s -> {nbytes / mean_s / 1e9:.2f} GB/s")

        # Every initiator publishes; pair 0 sums them after the quiesce
        # barrier. Written here rather than at the end so the roll-up cannot
        # race a slow pair still finishing its last iteration.
        _publish(sync_dir, f"result.p{pair_id}",
                 f"{best} {mean_s} {nbytes} {gpu_index}".encode())

        if args.mode == "aggregate":
            log(f"  (pair {pair_id} alone, on GPU {gpu_index}; node total below)")
        elif args.mem == "cuda":
            expect = ("all 4 rails, so 100 GB/s" if args.mode == "solo-peak"
                      else "ONE rail, so 25 GB/s -- see MODES")
            log(f"line rate reference: VRAM in --mode {args.mode} should get {expect}")
        else:
            log("line rate reference: DRAM uses all rails -- 4 x 200 Gbps = 100 GB/s peak")
    else:
        # The target is passive for the data movement, but polling notifs
        # drives NIXL's progress engine in builds without a progress thread,
        # and tells us whether the notification path works at all.
        notif_seen = []

        def poll():
            try:
                got = agent.get_new_notifs()
            except Exception:
                return
            for who, msgs in (got or {}).items():
                for m in msgs:
                    notif_seen.append((who, m))

        log("waiting for the initiator to finish...")
        _await(sync_dir, f"xfer_complete.p{pair_id}", timeout=3600.0, on_poll=poll)
        # Expect ONE FEWER notification than passes, and do not read anything
        # into it: _await checks for xfer_complete BEFORE calling on_poll, and
        # the initiator publishes that file the instant its last pass returns
        # DONE, so the final notification usually arrives with nobody left to
        # poll for it. A measured run showed 3 of 4. Only zero is interesting.
        log(f"initiator reported complete. notifications received: {len(notif_seen)}"
            + (f" (e.g. {notif_seen[0]})" if notif_seen else " -- none; the notification "
               "path may need a progress thread, this does not invalidate the transfer"))

    # ---- Counter snapshot, taken on BOTH ranks after everything is quiet --
    # The first version sampled each rank the moment it locally believed the
    # transfer was over, and produced an asymmetry: initiator rx read 1.02x of
    # payload while target tx read only 0.77x. The per-NIC breakdown showed
    # what that was -- the target's four NICs disagreed by ~75 MB in ascending
    # read order while the initiator's agreed to 512 bytes -- so it was the
    # sampling instrument, not the transport. Both ranks now sample behind a
    # common barrier AND wait for the telemetry caches to stop moving. If an
    # asymmetry survives that, it is real and worth chasing.
    # The counters are NODE-wide, and in aggregate mode every pair on this
    # node contributed, so the denominator is the node's payload -- not this
    # rank's. Getting this wrong would divide by 1/npairs of the real traffic
    # and report a passing 4.0x instead of 1.0x.
    total = nbytes * npairs * (args.iters + args.warmup)
    barrier(sync_dir, "quiesce", rank, size)
    if samples_counters:
        after, after_stamps, after_info = read_cxi_counters_settled(
            args.settle, timeout=args.settle_timeout)

    # ---- Q3: did the bytes actually go over Slingshot? --------------------
    if samples_counters:
        deltas = diff_counters(before, after)
        scope = "this NODE" if npairs > 1 else "this rank"
        log("")
        log(f"--- CXI hardware octet deltas ({human(total)} of payload crossed {scope}"
            + (f", {npairs} pairs" if npairs > 1 else "") + ") ---")
        log(f"  sample settled in {after_info['waited']:.2f}s over {after_info['reads']} reads"
            + ("" if after_info["settled"] else
               f"  -- NOT STABLE after {args.settle_timeout:.0f}s; the counters were still "
               "moving, so everything below is a LOWER BOUND"))

        if after_stamps:
            def _span(stamps):
                vals = list(stamps.values())
                try:
                    vals.sort(key=float)
                except ValueError:
                    vals.sort()
                return vals[0], vals[-1]

            b_lo, b_hi = _span(before_stamps) if before_stamps else ("?", "?")
            a_lo, a_hi = _span(after_stamps)
            log(f"  telemetry stamps: baseline [{b_lo} .. {b_hi}]  sample [{a_lo} .. {a_hi}]")
            log("  (opaque units. If the sample span sits clear of the baseline span, every "
                "NIC cache refreshed after the transfer and the deltas are complete.)")

        if not deltas:
            log("NO CXI COUNTER MOVEMENT. Either the sysfs telemetry path is wrong on this "
                "node, or the payload did not go over Slingshot.")
        else:
            for k, v in deltas.items():
                log(f"  {k:12s} {v:>18,d} octets  ({human(v)})")
            # Which direction should move depends on the op, not just the role.
            # WRITE: initiator pushes  -> initiator tx, target rx.
            # READ:  initiator pulls   -> initiator rx, target tx.
            if args.op == "WRITE":
                interesting = "tx" if is_initiator else "rx"
            else:
                interesting = "rx" if is_initiator else "tx"
            moved = sum(v for k, v in deltas.items() if k.endswith(interesting))
            log(f"  total {interesting}: {human(moved)} vs {human(total)} payload "
                f"-> {moved / total:.2f}x")
            log("  >= 1.0x on the expected direction is the Q3 pass condition; the excess "
                "is protocol overhead. Near 0 means a silent fallback.")

            # RAIL COUNT -- the whole point of the mode split. A NIC carrying
            # a few hundred octets is connection chatter, not payload, so
            # count only those above 1% of the expected per-rail share.
            floor = total / 100.0
            carried = sorted(k for k, v in deltas.items()
                             if k.endswith(interesting) and v > floor)
            log(f"  RAILS CARRYING PAYLOAD: {len(carried)} "
                f"({', '.join(c.split('.')[0] for c in carried) or 'none'})"
                f"  [1 rail = 25 GB/s, 4 rails = 100 GB/s]")
    else:
        log(f"CXI counters sampled by pair 0 on this node (they are node-wide; "
            f"{npairs} ranks reading them would each claim the whole node's traffic)")

    # ---- Node roll-up (aggregate mode) ------------------------------------
    # One printer, not npairs of them. The result.p* files were all written
    # before the quiesce barrier we just passed, so they are complete.
    if args.mode == "aggregate" and is_initiator and pair_id == 0:
        rows = []
        for p in range(npairs):
            f = sync_dir / f"result.p{p}"
            if not f.exists():
                log(f"WARNING: result.p{p} missing -- roll-up is incomplete.")
                continue
            b, m, nb, gi = f.read_bytes().decode().split()
            rows.append((p, float(b), float(m), int(nb), int(gi)))

        if rows:
            log("")
            log(f"=== NODE AGGREGATE: {len(rows)} pairs, {args.mem}, {args.op} ===")
            for p, b, m, nb, gi in rows:
                log(f"  pair {p} (GPU {gi}): best {nb / b / 1e9:6.2f} GB/s   "
                    f"mean {nb / m / 1e9:6.2f} GB/s")

            node_bytes = sum(r[3] for r in rows)
            slowest = max(r[2] for r in rows)
            sum_rates = sum(r[3] / r[2] for r in rows)

            # Two numbers because the pairs are NOT barriered against each
            # other -- they start together and then drift, so neither bound is
            # the whole truth.
            #   envelope: node payload / slowest pair's mean. Assumes perfect
            #             overlap; the honest LOWER bound on node throughput.
            #   sum:      adds the per-pair rates. Assumes each pair had the
            #             node to itself for its own window; UPPER bound.
            # If they are close, the pairs really did overlap and either is
            # fine to quote. If they are far apart, the pairs serialised and
            # only the envelope means anything.
            log(f"  node payload per pass: {human(node_bytes)}")
            log(f"  ENVELOPE (lower bound): {node_bytes / slowest / 1e9:.2f} GB/s")
            log(f"  SUM OF RATES  (upper) : {sum_rates / 1e9:.2f} GB/s")
            log(f"  line rate reference: {len(rows)} GPUs x 1 rail = "
                f"{len(rows) * 25} GB/s, node ceiling 100 GB/s (4 x 200 Gbps)")
            log("  cross-check against the CXI octet deltas above -- those are "
                "measured on the wire and do not depend on this arithmetic.")

    # ---- Verification -----------------------------------------------------
    barrier(sync_dir, "counters", rank, size)
    dst_is_local = not src_is_local
    if dst_is_local and not args.no_verify:
        log("verifying byte-exactness...")
        stride = 64 * 2**20 // 8
        bad = 0
        for start in range(0, nelem, stride):
            end = min(start + stride, nelem)
            want = torch.arange(start, end, dtype=torch.int64, device=buf.device)
            if not torch.equal(buf[start:end], want):
                bad += 1
                if bad == 1:
                    log(f"MISMATCH in elements [{start}, {end})")
        if bad:
            sys.exit(f"FAIL: {bad} mismatching slice(s) -- the transfer corrupted data.")
        log("PASS: destination buffer is byte-exact.")

    barrier(sync_dir, "done", rank, size)
    if is_initiator:
        # Only initiators ever built a handle. Guarding on role, rather than
        # letting the targets raise NameError into a bare except, keeps a
        # genuine release failure visible instead of silently swallowed.
        try:
            agent.release_xfer_handle(handle)
        except Exception as exc:
            log(f"release_xfer_handle warning: {exc}")
    try:
        agent.remove_remote_agent(peer_name)
        agent.deregister_memory(reg)
    except Exception as exc:
        log(f"teardown warning: {exc}")

    log("COMPLETE.")


if __name__ == "__main__":
    main()
