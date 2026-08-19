"""ShareGPT-derived 长度分布（报告 §8.1.2 协议）：
首两 turn 严格 human→gpt → K3 tokenizer 原文 tokenize（无 chat template）→
vLLM filter（prompt≥4, output≥4, prompt≤1024, prompt+output≤2048）→ lengths.json。
并对照报告钉死的统计量（N=52229, mean=108.20, P50=28…），不一致则如实报告。

VM: ~/venv-sgl/bin/python k3_sharegpt_lengths.py <sharegpt.json> <tokenizer_dir_or_repo> <out.json>
"""

import hashlib
import json
import os
import sys

import numpy as np

sg_path, tok_src, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

sha = hashlib.sha256(open(sg_path, "rb").read()).hexdigest()
print(f"dataset sha256: {sha}")
print(f"  期望: 35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4  匹配: {sha == '35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4'}")

from transformers import AutoTokenizer  # noqa: E402

kw = {"trust_remote_code": True}
if "/" in tok_src and not os.path.exists(os.path.expanduser(tok_src)):
    kw["revision"] = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
tok = AutoTokenizer.from_pretrained(os.path.expanduser(tok_src), **kw)

data = json.load(open(sg_path))
print(f"records: {len(data)}")
kept = []
n_two_turns = 0
for rec in data:
    conv = rec.get("conversations") or []
    if len(conv) < 2:
        continue
    n_two_turns += 1
    if conv[0].get("from") != "human" or conv[1].get("from") != "gpt":
        continue
    kept.append((conv[0].get("value") or "", conv[1].get("value") or ""))
print(f">=2 turns: {n_two_turns}, 首两 turn human->gpt: {len(kept)}")

lengths = []
for prompt, output in kept:
    lp = len(tok(prompt).input_ids)
    lo = len(tok(output).input_ids)
    if lp < 4 or lo < 4:
        continue
    if lp > 1024 or lp + lo > 2048:
        continue
    lengths.append(lp)
a = np.array(lengths)
print(f"最终 lengths: {len(a)}（报告钉死值 52229，匹配: {len(a) == 52229}）")
stats = {
    "count": int(len(a)), "mean": round(float(a.mean()), 2), "std": round(float(a.std(ddof=1)), 2),
    "min": int(a.min()), "max": int(a.max()),
    **{f"p{p}": float(np.percentile(a, p, method="linear")) for p in (10, 25, 50, 75, 90, 95, 99, 99.9)},
}
print("stats:", stats)
print("报告钉死: mean=108.20 std=189.81 P50=28 P90=336 P99=928 max=1024")
json.dump({"sha256": sha, "stats": stats, "lengths": [int(x) for x in a]}, open(out_path, "w"))
print(f"saved -> {out_path}")
