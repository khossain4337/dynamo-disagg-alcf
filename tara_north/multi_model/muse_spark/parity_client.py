"""Greedy parity client: fixed prompts, temperature 0, 256 tokens, top-2 logprobs.
Usage: python parity_client.py <base_url> <model> <out.jsonl> <concurrency> <pass>
"""
import glob
import json
import sys
import sysconfig
import urllib.request
from concurrent.futures import ThreadPoolExecutor

url, model, out, conc, npass = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]

SHORT = [
    "The capital of France is",
    "Explain why the sky is blue in two sentences.",
    "Write a haiku about a lighthouse in winter.",
    "List the first ten prime numbers and explain how you found them.",
    "What is the difference between a virus and a bacterium?",
    "Translate into Spanish: The train leaves at seven tomorrow morning.",
    "Write a Python function that reverses a linked list.",
    "Summarize the causes of the French Revolution.",
    "A farmer has 17 sheep. All but 9 run away. How many are left? Explain.",
    "Describe how a transistor works to a ten-year-old.",
    "Give three arguments for and against nuclear power.",
    "What happens when you mix baking soda and vinegar, chemically?",
    "Write the opening paragraph of a mystery novel set in Lisbon.",
    "How does public-key cryptography let two strangers share a secret?",
    "Compute 37 times 43 step by step.",
    "Explain the rules of chess castling.",
    "What is a hash table, and why are lookups fast on average?",
    "Write a short dialogue between a cat and a robot.",
    "Why do leaves change color in autumn?",
    "Explain the difference between TCP and UDP.",
    "Describe the water cycle.",
    "Write a SQL query that finds the second highest salary in a table.",
    "What were the main achievements of the Apollo program?",
    "Explain gradient descent with a simple analogy.",
]
# Long prompts: stdlib source, fixed slices (~4 chars per token). Same Python in every env.
src = "".join(open(f).read() for f in sorted(glob.glob(sysconfig.get_paths()["stdlib"] + "/*.py")))
LONG = [
    src[off:off + n] + "\n\nSummarize what the code above does."
    for n in (4_000, 16_000, 64_000, 128_000)
    for off in (0, 1_000_000)
]
PROMPTS = SHORT + LONG


def run(i):
    body = {"model": model, "prompt": PROMPTS[i], "max_tokens": 256, "temperature": 0,
            "ignore_eos": True, "logprobs": 2, "return_tokens_as_token_ids": True}
    req = urllib.request.Request(url + "/v1/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=3600))
    except Exception as e:
        return {"idx": i, "error": repr(e)}
    lp = r["choices"][0]["logprobs"]
    return {"idx": i, "pass": npass, "c": conc, "prompt_tokens": r["usage"]["prompt_tokens"],
            "tokens": [int(t.split(":")[1]) for t in lp["tokens"]],
            "top2": [sorted(d.values(), reverse=True)[:2] for d in lp["top_logprobs"]],
            "text": r["choices"][0]["text"]}


with ThreadPoolExecutor(conc) as ex, open(out, "w") as f:
    rows = list(ex.map(run, range(len(PROMPTS))))
    for row in rows:
        f.write(json.dumps(row) + "\n")
bad = sum("error" in r for r in rows)
print(f"{out}: {len(rows)} prompts, {bad} failed")
sys.exit(1 if bad else 0)
