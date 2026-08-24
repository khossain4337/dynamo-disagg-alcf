#!/usr/bin/env python3
"""
Delete every NVIDIA GPU but one from an hwloc XML topology.

WHY THIS EXISTS
---------------
NIXL's LIBFABRIC backend partitions NICs across accelerators in
libfabric_topology.cpp, groupNicsWithAccel(), Step 4:

    const int nics_per_group = nics.size() / num_groups;

`num_groups` is the number of accelerators hwloc found. On a Tara node that
is 4 GPUs and 4 Slingshot NICs, so nics_per_group == 1 and each GPU is
permanently bound to exactly ONE rail -- 25 GB/s, no matter what you ask
for. There is no knob: `max_bw_per_dram_seg` is the only LIBFABRIC tunable
and it is DRAM-only.

The assumption is an AWS p5 one (8 GPUs, 32 EFA NICs, 4 rails each, one rank
per GPU) and it is not wrong there. It is just arithmetic that lands badly
on a 4:4 machine.

The only lever that reaches `num_groups` from outside the process is what
hwloc reports, and hwloc reads HWLOC_XMLFILE in hwloc_topology_load(). Hand
it a topology with ONE GPU and num_groups becomes 1, so nics_per_group
becomes 4 and that GPU inherits all four rails.

This is a DIAGNOSTIC. It measures what a single GPU could do if NIXL let it
spread; it is not a configuration anyone should run a real workload under,
because the other three GPUs no longer exist as far as NIXL is concerned.

WHY NOT HWLOC_COMPONENTS=-pci
-----------------------------
Tempting -- it empties nic_info_map, buildTopologyAwareGrouping() fails,
buildFallbackMapping() leaves the accel->NIC map EMPTY, and an empty map
means getEfaDevicesForPci() returns all_devices, i.e. all 4 rails. But it
also zeroes discoverAccelWithHwloc(), so num_nvidia_accel becomes 0, so
VRAM_SEG is never advertised and `--mem cuda` dies in registerMem before it
gets anywhere near a rail. Pruning the XML keeps one GPU visible, which is
what keeps the VRAM path alive.

WHICH GPU SURVIVES
------------------
The lowest PCI bus id, unless --keep says otherwise. The caller must then
point CUDA at that same physical device -- see the GPU-UUID line this prints
on stdout -- or torch's cuda:0 and NIXL's one visible accelerator will be
different pieces of silicon and the rail mapping will be a lie.

USAGE
    lstopo-no-graphics --of xml > full.xml
    python3 prune_hwloc_xml.py full.xml solo.xml        # keep lowest BDF
    python3 prune_hwloc_xml.py full.xml solo.xml --keep 0000:0f:00.0
"""

import argparse
import re
import sys
import xml.etree.ElementTree as ET

NVIDIA_VENDOR = 0x10DE


def is_nvidia_gpu(elem):
    """Mirror nixlLibfabricTopology::isNvidiaAccel exactly.

    That function tests vendor_id == 0x10de and class_id in [0x300, 0x400).
    In hwloc XML both live in pci_type, whose first field is the class and
    whose first bracketed pair is vendor:device --

        pci_type="0302 [10de:2330] [10de:16c1] a1"

    Matching NIXL's predicate rather than something looser matters: an
    NVIDIA display adapter or an audio function on the same card would be
    kept by a vendor-only test and would still count towards num_groups.
    """
    if elem.get("type") != "PCIDev":
        return False
    pci_type = elem.get("pci_type", "")
    m = re.match(r"^([0-9a-fA-F]{4})\s+\[([0-9a-fA-F]{4}):", pci_type)
    if not m:
        return False
    class_id = int(m.group(1), 16)
    vendor_id = int(m.group(2), 16)
    return vendor_id == NVIDIA_VENDOR and 0x300 <= class_id < 0x400


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="hwloc XML from `lstopo-no-graphics --of xml`")
    ap.add_argument("dst", help="where to write the pruned topology")
    ap.add_argument("--keep", metavar="BDF",
                    help="PCI bus id of the GPU to keep (default: lowest)")
    args = ap.parse_args()

    with open(args.src, "r", encoding="utf-8", errors="replace") as fh:
        raw = fh.read()

    # ElementTree discards the XML declaration and the <!DOCTYPE topology
    # SYSTEM "hwloc2.dtd"> line. hwloc's importer wants them, so keep the
    # prolog verbatim and only rewrite the tree underneath it.
    cut = raw.find("<topology")
    if cut < 0:
        sys.exit(f"ERROR: {args.src} has no <topology> element -- is it hwloc XML?")
    prolog = raw[:cut]

    root = ET.fromstring(raw)

    # Parent map: ElementTree has no upward links, and removal is done
    # through the parent.
    parents = {c: p for p in root.iter() for c in p}

    gpus = [e for e in root.iter() if is_nvidia_gpu(e)]
    if not gpus:
        sys.exit("ERROR: no NVIDIA GPUs found in the topology. Either lstopo ran "
                 "without PCI discovery (needs to run on the compute node, not the "
                 "login node) or this really is a CPU-only node.")

    by_bdf = {}
    for g in gpus:
        bdf = g.get("pci_busid", "?")
        by_bdf.setdefault(bdf, g)
    ordered = sorted(by_bdf)

    if args.keep:
        if args.keep not in by_bdf:
            sys.exit(f"ERROR: --keep {args.keep} is not one of the GPUs found: {ordered}")
        keep_bdf = args.keep
    else:
        keep_bdf = ordered[0]

    if len(ordered) == 1:
        print(f"NOTE: only one GPU ({keep_bdf}) in the topology -- nothing to prune. "
              f"num_groups was already 1, so this node should already give a single "
              f"GPU all its rails.", file=sys.stderr)

    removed = []
    for bdf in ordered:
        if bdf == keep_bdf:
            continue
        elem = by_bdf[bdf]
        parent = parents.get(elem)
        if parent is None:
            sys.exit(f"ERROR: GPU {bdf} has no parent element -- refusing to guess.")
        parent.remove(elem)
        removed.append(bdf)

    with open(args.dst, "w", encoding="utf-8") as fh:
        fh.write(prolog)
        fh.write(ET.tostring(root, encoding="unicode"))
        fh.write("\n")

    # stdout is the machine-readable channel: the wrapper eval's these.
    print(f"SOLO_GPU_BDF={keep_bdf}")
    print(f"SOLO_GPU_REMOVED={len(removed)}")
    print(f"pruned {len(removed)} of {len(ordered)} GPUs, kept {keep_bdf} "
          f"(removed {', '.join(removed) or 'none'})", file=sys.stderr)


if __name__ == "__main__":
    main()
