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

CXI_TELEMETRY = "/sys/class/cxi/cxi{i}/device/telemetry/hni_sts_{dir}_ok_octets"


def read_cxi_counters(max_nics=8):
    counters = {}
    for i in range(max_nics):
        for direction in ("tx", "rx"):
            path = CXI_TELEMETRY.format(i=i, dir=direction)
            try:
                with open(path) as fh:
                    raw = fh.read().strip()
            except OSError:
                continue
            token = raw.split("@")[0].split()[0]
            try:
                counters[f"cxi{i}.{direction}"] = int(token)
            except ValueError:
                continue
    return counters


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
    ap.add_argument("--op", choices=("WRITE", "READ"), default="WRITE",
                    help="WRITE = initiator pushes (default). READ = initiator pulls, "
                         "which is the direction vLLM's NixlConnector actually uses.")
    ap.add_argument("--mem", choices=("dram", "cuda"), default="dram",
                    help="where the buffer lives. dram first -- it isolates the fabric "
                         "from HMEM/GPUDirect concerns. cuda is the real KV-cache path.")
    ap.add_argument("--backend", default="LIBFABRIC")
    ap.add_argument("--sync-dir", default=None,
                    help="shared rendezvous dir (default: derived from PALS_APID under "
                         "the /vast testing tree)")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip byte-exactness check (it is not free on a multi-GiB buffer)")
    ap.add_argument("--dump-api", action="store_true",
                    help="print the nixl_agent method list and exit -- use this if a call "
                         "below turns out not to exist in this NIXL build")
    args = ap.parse_args()

    rank_env = os.environ.get("PMI_RANK", os.environ.get("PALS_RANKID"))
    size_env = os.environ.get("PMI_SIZE", os.environ.get("PALS_NRANKS"))
    if rank_env is None:
        sys.exit("ERROR: no PMI_RANK in env -- this must run under `mpiexec -n 2 -ppn 1`.\n"
                 "       Outside PALS there is no SLINGSHOT_VNIS, and fi_domain() will "
                 "return -FI_ENOSYS for cxi.")
    rank, size = int(rank_env), int(size_env or 2)
    if size != 2:
        sys.exit(f"ERROR: expected exactly 2 ranks, got {size}.")

    host = socket.gethostname().split(".")[0]
    role = "initiator" if rank == 0 else "target"
    peer_role = "target" if rank == 0 else "initiator"

    def log(msg):
        print(f"[{role} r{rank} {host}] {msg}", flush=True)

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

    sync_dir = Path(args.sync_dir or
                    f"/vast/draco/tara/projects/Tara_Deployment/software/testing/"
                    f"nixl_2rank_{os.environ.get('PALS_APID', 'noapid')}")
    sync_dir.mkdir(parents=True, exist_ok=True)

    # ---- Guard: the two ranks must be on different nodes ------------------
    # A same-node pass could be served by shm or cxi loopback and would be
    # worthless as evidence for a cross-node KV transfer.
    _publish(sync_dir, f"host.{rank}", host.encode())
    barrier(sync_dir, "hosts", rank, size)
    peer_host = (sync_dir / f"host.{1 - rank}").read_bytes().decode()
    if peer_host == host:
        sys.exit(f"ERROR: both ranks landed on {host}. Use -ppn 1 across 2 nodes; a "
                 f"same-node transfer proves nothing about the fabric.")
    log(f"peer is {peer_role} on {peer_host} -- distinct nodes, good")

    # ---- Agent ------------------------------------------------------------
    import torch
    from nixl._api import nixl_agent, nixl_agent_config

    agent = nixl_agent(f"{role}", nixl_agent_config(backends=[args.backend]))
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

    device = "cuda:0" if args.mem == "cuda" else "cpu"
    log(f"allocating {human(nbytes)} on {device} as {nchunks} x {args.chunk_mib} MiB descriptors")
    buf = torch.empty(nelem, dtype=torch.int64, device=device)

    # Source side gets the pattern, destination side gets zeros, so a
    # no-op transfer cannot masquerade as success.
    src_is_local = (args.op == "WRITE" and rank == 0) or (args.op == "READ" and rank == 1)
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

    peer_meta = (sync_dir / f"meta.{1 - rank}").read_bytes()
    peer_name = agent.add_remote_agent(peer_meta)
    if isinstance(peer_name, bytes):
        peer_name = peer_name.decode()
    log(f"added remote agent '{peer_name}'")

    remote_descs = agent.deserialize_descs((sync_dir / f"descs.{1 - rank}").read_bytes())

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
    before = read_cxi_counters()

    if rank == 0:
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

        after = read_cxi_counters()
        _publish(sync_dir, "xfer_complete")

        try:
            tele = agent.get_xfer_telemetry(handle)
            if tele:
                log(f"NIXL transfer telemetry: {tele}")
        except Exception:
            pass

        best = min(times)
        total = nbytes * (args.iters + args.warmup)
        log("")
        log(f"=== {args.op} {human(nbytes)} x{args.iters} timed ({args.mem}, {nchunks} descriptors) ===")
        log(f"best  {best:.3f}s -> {nbytes / best / 1e9:.2f} GB/s")
        log(f"mean  {sum(times)/len(times):.3f}s -> {nbytes*len(times) / sum(times) / 1e9:.2f} GB/s")
        log(f"line rate reference: 4 x 200 Gbps Slingshot = 100 GB/s aggregate peak")
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
        _await(sync_dir, "xfer_complete", timeout=3600.0, on_poll=poll)
        after = read_cxi_counters()
        total = nbytes * (args.iters + args.warmup)
        log(f"initiator reported complete. notifications received: {len(notif_seen)}"
            + (f" (e.g. {notif_seen[0]})" if notif_seen else " -- none; the notification "
               "path may need a progress thread, this does not invalidate the transfer"))

    # ---- Q3: did the bytes actually go over Slingshot? --------------------
    deltas = diff_counters(before, after)
    log("")
    log(f"--- CXI hardware octet deltas ({human(total)} of payload crossed this rank) ---")
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
            interesting = "tx" if rank == 0 else "rx"
        else:
            interesting = "rx" if rank == 0 else "tx"
        moved = sum(v for k, v in deltas.items() if k.endswith(interesting))
        log(f"  total {interesting}: {human(moved)} vs {human(total)} payload "
            f"-> {moved / total:.2f}x")
        log("  >= 1.0x on the expected direction is the Q3 pass condition; the excess "
             "is protocol overhead. Near 0 means a silent fallback.")

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
    try:
        agent.release_xfer_handle(handle)  # noqa: F821  (initiator only)
    except Exception:
        pass
    try:
        agent.remove_remote_agent(peer_name)
        agent.deregister_memory(reg)
    except Exception as exc:
        log(f"teardown warning: {exc}")

    log("COMPLETE.")


if __name__ == "__main__":
    main()
