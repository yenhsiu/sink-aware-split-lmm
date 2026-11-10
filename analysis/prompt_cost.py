#!/usr/bin/env python3
"""
prompt_cost.py -- size of the text prompt per benchmark (UTF-8 bits), compared
with the smallest visual budget (paper, Sec. "Robustness of the Deadline Guarantee").

    python analysis/prompt_cost.py
"""
import json
import statistics as st
from pathlib import Path

import pandas as pd

EVAL = Path(__file__).resolve().parent.parent / "playground" / "data" / "eval"
SMALLEST_BUDGET = 25_600


def jsonl_prompts(rel):
    return [json.loads(line)["text"] for line in open(EVAL / rel)]


def mmbench_prompts():
    df = pd.read_csv(EVAL / "mmbench" / "mmbench_dev_20230712.tsv", sep="\t")
    out = []
    for _, r in df.iterrows():
        t = (r["hint"] + "\n" if isinstance(r.get("hint"), str) else "") + str(r["question"])
        for o in "ABCD":
            if isinstance(r.get(o), str):
                t += f"\n{o}. {r[o]}"
        out.append(t)
    return out


if __name__ == "__main__":
    prompts = {
        "MME": jsonl_prompts("MME/llava_mme.jsonl"),
        "POPE": jsonl_prompts("pope/llava_pope_test.jsonl"),
        "TextVQA": jsonl_prompts("textvqa/llava_textvqa_val_v051_ocr.jsonl"),
        "MMBench": mmbench_prompts(),
    }
    for name, ps in prompts.items():
        bits = [8 * len(p.encode("utf-8")) for p in ps]
        print(f"{name:8s} n={len(ps):5d}  mean {st.mean(bits):6.0f} bits  median {st.median(bits):6.0f}"
              f"  p95 {sorted(bits)[int(0.95 * len(bits))]:6d}  mean/smallest budget {100 * st.mean(bits) / SMALLEST_BUDGET:.1f}%")
