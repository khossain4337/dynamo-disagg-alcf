"""First token divergence per prompt, with the top-2 logprob gap there in each run.
Usage: python parity_compare.py <ref.jsonl> <test.jsonl>
"""
import json
import sys

ref, test = ({r["idx"]: r for r in map(json.loads, open(p))} for p in sys.argv[1:3])
same = 0
print("idx  prompt_tok  first_div  gap_ref  gap_test")
for i in sorted(ref):
    a, b = ref[i], test.get(i, {})
    if "error" in a or "error" in b or not b:
        print(f"{i:3d}  missing or failed")
        continue
    d = next((k for k, (x, y) in enumerate(zip(a["tokens"], b["tokens"])) if x != y), None)
    if d is None:
        same += 1
        print(f"{i:3d}  {a['prompt_tokens']:10d}  -")
        continue
    gap = lambda r: r["top2"][d][0] - r["top2"][d][1] if len(r["top2"][d]) > 1 else float("nan")
    print(f"{i:3d}  {a['prompt_tokens']:10d}  {d:9d}  {gap(a):7.3f}  {gap(b):8.3f}")
print(f"identical {same}/{len(ref)}")
