#!/usr/bin/env python3
"""Merge per-batch-size benchmark_moe_inkling.py shards into one tuned config.

Point VLLM_TUNED_CONFIG_FOLDER at --out. See DECISIONS_2026-10-02c.md.
"""

import argparse
import json
import os
import sys
from pathlib import Path


def write(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=4) + "\n")
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shards", required=True, type=Path, help="epoch shards dir")
    p.add_argument("--out", required=True, type=Path, help="epoch config dir")
    p.add_argument("--expect", required=True, type=int, nargs="+")
    args = p.parse_args()

    files = sorted(args.shards.glob("*/*.json"), key=lambda f: f.stat().st_mtime)
    if not files:
        sys.exit(f"no shard json under {args.shards}")

    names = {f.name for f in files}
    if len(names) > 1:
        sys.exit(f"shards disagree on shape: {sorted(names)}")

    merged, source, triton = {}, {}, None
    for f in files:
        d = json.loads(f.read_text())
        v = d.pop("triton_version", None)
        if triton is None:
            triton = v
        elif v != triton:
            sys.exit(f"triton {v} in {f} != {triton}; configs are not comparable")
        for k, cfg in d.items():
            merged[k] = cfg  # latest mtime wins
            source[k] = str(f)

    missing = [m for m in args.expect if str(m) not in merged]
    if missing:
        sys.exit(f"missing {missing}: a gap snaps to the nearest key, silently")

    args.out.mkdir(parents=True, exist_ok=True)
    name = names.pop()
    write(
        args.out / name,
        {"triton_version": triton, **{k: merged[k] for k in sorted(merged, key=int)}},
    )
    write(args.out / "manifest.json", {"config": name, "source": source})
    print(f"{args.out / name}: {len(merged)} batch sizes, triton {triton}")


if __name__ == "__main__":
    main()
