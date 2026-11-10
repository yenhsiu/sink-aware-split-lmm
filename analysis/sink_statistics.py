#!/usr/bin/env python3
"""
sink_statistics.py -- attention-sink statistics of CLIP ViT-L/14-336 on evaluation images.

Runs only the vision encoder (no LLM), so it needs about 1-2 GB of GPU memory.
For a random sample of images from each benchmark it reports:

  * how often the detector finds a sink (candidates: score > ratio x median;
    sinks: at most the two highest candidates), for several thresholds `ratio`,
    and without the cap of two;
  * the share of CLS-attention mass absorbed by the flagged tokens;
  * for each LUT anchor's token count N, how often PruneSID selects at least one
    flagged sink into the transmitted set (the case where sink-aware demotion acts).

The detector and the PruneSID selection are the exact code paths used by the
pipeline (llava/model/multimodal_encoder/clip_encoder.py).

    python analysis/sink_statistics.py --datasets mme pope textvqa --n-images 500
"""
import argparse
import json
import os
import random
import statistics as st
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "paper"))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

import llava.model.multimodal_encoder.clip_encoder as ce  # noqa: E402
from llava.mm_utils import process_images  # noqa: E402
from paper_data import ANCHOR_CONFIGS  # noqa: E402

EVAL = ROOT / "playground" / "data" / "eval"
DATASETS = {
    "mme":     ("MME/llava_mme.jsonl", "MME/MME_Benchmark_release_version"),
    "pope":    ("pope/llava_pope_test.jsonl", "pope/val2014"),
    "textvqa": ("textvqa/llava_textvqa_val_v051_ocr.jsonl", "textvqa/train_images"),
}
RATIOS = [25, 50, 100, 200]


def sample_images(ds, n, seed):
    qf, folder = DATASETS[ds]
    imgs = sorted({json.loads(l)["image"] for l in open(EVAL / qf)})
    random.Random(seed).shuffle(imgs)
    return [EVAL / folder / p for p in imgs[:n]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--n-images", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0, help="image sampling seed")
    ap.add_argument("--out", type=Path, default=ROOT / "analysis" / "sink_statistics.json")
    a = ap.parse_args()

    os.environ.update(LLAVA_USE_QUANT="false", LLAVA_STRAT_GROUPS="", LLAVA_SINK_FILTER="true",
                      LLAVA_SINK_MODE="pin", LLAVA_SINK_RATIO="100.0", LLAVA_SINK_MAX_CANDIDATES="2")
    tower = ce.CLIPVisionTower("openai/clip-vit-large-patch14-336",
                               SimpleNamespace(mm_vision_select_layer=-2, mm_vision_select_feature="patch"))
    tower = tower.to("cuda", dtype=torch.float16)
    proc_cfg = SimpleNamespace(image_aspect_ratio="pad")  # LLaVA-1.5 preprocessing

    # Record how many of the N selected tokens are sinks: the pipeline calls
    # _fit_tier_sizes(n4, n2, n1, n_total, n_non_sink) once per image.
    captured = {}
    orig_fit = ce._fit_tier_sizes

    def fit_spy(n4, n2, n1, n_total, n_non_sink):
        captured["sinks_selected"] = n_total - n_non_sink
        return orig_fit(n4, n2, n1, n_total, n_non_sink)
    ce._fit_tier_sizes = fit_spy

    anchors = {B: c["N"] for B, c in sorted(ANCHOR_CONFIGS.items())}
    report = {}
    for ds in a.datasets:
        per = {"n_images": 0, "count": {r: [] for r in RATIOS}, "count_uncapped": [], "mass": [],
               "selected": {B: [] for B in anchors}}
        for path in sample_images(ds, a.n_images, a.seed):
            img = process_images([Image.open(path).convert("RGB")], tower.image_processor, proc_cfg)
            img = img.to("cuda", dtype=torch.float16)
            with torch.no_grad():
                out = tower.vision_tower(img, output_hidden_states=True, output_attentions=True)
                attn = out.attentions[-2][0, :, 0, 1:].float()          # [heads, 576]
                score = attn.sum(0)                                      # same score as the pipeline
                srt, idx = torch.sort(score, descending=True)
                for r in RATIOS:
                    per["count"][r].append(ce._detect_sink_count(srt, 2, float(r)))
                per["count_uncapped"].append(ce._detect_sink_count(srt, 576, 100.0))
                k = per["count"][100][-1]
                per["mass"].append((attn[:, idx[:k]].sum() / attn.shape[0]).item())  # share of CLS attention
                for B, N in anchors.items():
                    os.environ["LLAVA_N_TOKENS"] = str(N)
                    tower.token_prunesid_quant_select(img)
                    per["selected"][B].append(captured["sinks_selected"])
            per["n_images"] += 1
        flagged = [m for m, c in zip(per["mass"], per["count"][100]) if c > 0]
        report[ds] = {
            "n_images": per["n_images"],
            "pct_images_with_sink": {r: 100 * sum(c > 0 for c in v) / len(v) for r, v in per["count"].items()},
            "mean_sinks_per_image": {r: st.mean(v) for r, v in per["count"].items()},
            "pct_images_with_sink_uncapped": 100 * sum(c > 0 for c in per["count_uncapped"]) / per["n_images"],
            "mean_sinks_uncapped": st.mean(per["count_uncapped"]),
            "sink_attention_share_median": st.median(flagged) if flagged else 0.0,
            "sink_attention_share_mean": st.mean(flagged) if flagged else 0.0,
            "pct_images_sink_selected": {B: 100 * sum(s > 0 for s in v) / len(v) for B, v in per["selected"].items()},
            "mean_sinks_selected": {B: st.mean(v) for B, v in per["selected"].items()},
        }
        print(ds, json.dumps(report[ds], indent=1), flush=True)
    a.out.write_text(json.dumps(report, indent=1))
    print("saved", a.out)


if __name__ == "__main__":
    main()
