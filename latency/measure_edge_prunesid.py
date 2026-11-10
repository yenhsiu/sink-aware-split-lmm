#!/usr/bin/env python3
"""
Edge-side latency measurement for the PruneSID clustering step (PCA-cluster +
intra-cluster similarity NMS diversity selection) -- the token-selection
method used by this pipeline (replacing PruMerge).

Measures ONLY the post-ViT clustering/selection body -- same convention as
measure_edge.py's T_prumerge_base/T_prumerge_plus: capture one ViT forward
pass's hidden_states + attentions once (outside the timed loop), then time
just the clustering computation repeatedly. T_edge (ViT-L/14-336 forward
itself) is measured separately by measure_edge.py (231.478099784581 ms on
the reference device) and is NOT included here. Quantization (T_quant) and
sink detection are also NOT included here -- those are separate components.

The clustering body is imported verbatim from AdaptQuant's clip_encoder.py
(_prunesid_batch_pca, _prunesid_similarity_nms) so this measures the actual
production code path, not a reimplementation; the connecting glue code
(tensor bookkeeping between those two calls) is copied 1:1 from
token_prunesid_quant_select's own body, up through building `important_indices`
(i.e. everything the real method does *before* its per-image sink-detection /
tiering / quantization loop).

Usage:
    python measure_edge_prunesid.py [--device "Jetson AGX Orin"]
Output:
    results_edge_prunesid_<device>.csv  (columns: component, N, mean_ms, std_ms)
"""

import argparse
import csv
import math
import platform
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

# ── Config ────────────────────────────────────────────────────────────────────

TOKEN_DIM      = 1024
N_MIN          = 1
N_MAX          = 576
DEFAULT_N_STEP = 8       # PCA+NMS body is much heavier than T_quant -- default to a coarser sweep
WARMUP         = 5
REPEAT_CLUSTER = 20

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    machine = platform.machine().lower()
    release = platform.uname().release.lower()
    is_agx = "aarch64" in machine or "arm64" in machine or "orin" in release or "tegra" in release
    if is_agx:
        raise RuntimeError(
            "CUDA is not available in this Python environment on AGX Orin.\n"
            "Please activate the JetPack/Conda environment with CUDA enabled and run:\n"
            "  python3 -c \"import torch; print(torch.cuda.is_available(), torch.cuda.device_count())\"\n"
            "  python3 measure_edge_prunesid.py --device 'AGX Orin'"
        )
    return torch.device("cpu")

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def timed_run(fn, n_warmup, n_repeat):
    for _ in range(n_warmup):
        fn()
    sync()
    times = []
    for _ in range(n_repeat):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - t0) * 1000.0)
    return sum(times) / len(times), statistics.stdev(times) if len(times) > 1 else 0.0

def detect_system_info():
    info = {}
    info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
    info["cuda_version"] = (torch.version.cuda or "N/A") if torch.cuda.is_available() else "N/A"
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    info["ram_gb"] = round(int(line.split()[1]) / 1024 / 1024, 1)
                    break
    except Exception:
        info["ram_gb"] = "unknown"
    info["python"] = platform.python_version()
    info["torch"] = torch.__version__
    info["machine"] = platform.machine()
    info["release"] = platform.uname().release
    return info

# ── Capture one ViT forward pass (random weights, matches measure_edge.py) ────

def _capture_vit_prunesid(device):
    """Build a random-weight CLIP ViT-L/14-336, run it once with
    output_hidden_states=True, output_attentions=True (the real
    token_prunesid_quant_select's own inputs -- proper multi-head attention,
    not a hand-rolled hook), return (hidden_states[-2], attentions[-2])."""
    from transformers import CLIPVisionModel, CLIPVisionConfig
    config = CLIPVisionConfig(
        hidden_size=1024, intermediate_size=4096,
        num_hidden_layers=24, num_attention_heads=16,
        image_size=336, patch_size=14, projection_dim=768,
    )
    print("  Building CLIPVisionModel (ViT-L/14-336, random weights, float32) ...", flush=True)
    # float32 here (not float16) to match token_prunesid_quant_select's own
    # dtype handling of hidden_states/attn_weights during the clustering body
    # (PCA/similarity ops are done in float32 regardless of model dtype).
    model = CLIPVisionModel(config).to(device=device, dtype=torch.float32).eval()
    pv = torch.randn(1, 3, 336, 336, device=device, dtype=torch.float32)

    with torch.no_grad():
        out = model(pixel_values=pv, output_hidden_states=True, output_attentions=True)
    hidden_states = out.hidden_states[-2].clone()   # [1, 577, 1024]
    attn_weights = out.attentions[-2].clone()        # [1, 16, 577, 577]
    del model
    return hidden_states, attn_weights

# ── PruneSID clustering body (post-ViT operations only) ──────────────────────
# Imported verbatim from AdaptQuant's clip_encoder.py; the glue code below is
# copied 1:1 from token_prunesid_quant_select (up to `important_indices`).

def _prunesid_cluster_body(hidden_states, attn_weights, need_token_num):
    from llava.model.multimodal_encoder.clip_encoder import _prunesid_batch_pca, _prunesid_similarity_nms

    cls_attn = attn_weights[:, :, 0, 1:]
    cls_attn_sum = cls_attn.sum(dim=1)

    projector_lengths, belong_components = _prunesid_batch_pca(
        hidden_states, min_components=max(1, int(need_token_num / 4))
    )
    projector_scores = cls_attn_sum.unsqueeze(-1).repeat(1, 1, projector_lengths.shape[-1]).to(projector_lengths.dtype)
    projector_mask = belong_components.unsqueeze(-1).repeat(1, 1, projector_lengths.shape[-1])
    index_map = torch.arange(projector_lengths.shape[-1], device=projector_lengths.device).unsqueeze(0).unsqueeze(0).repeat(projector_lengths.shape[0], projector_lengths.shape[1], 1)
    weights_mask = torch.where(projector_mask != index_map)
    projector_lengths[weights_mask] = 0
    projector_scores[weights_mask] = 0

    normalized_states = F.normalize(hidden_states[:, 1:, :], p=2, dim=-1)
    similarity = torch.bmm(normalized_states, normalized_states.transpose(2, 1)).to(torch.float32)
    triu_mask = torch.triu(torch.ones_like(similarity), diagonal=1).bool()
    sim_mean = (similarity * triu_mask).sum(-1).sum(-1) / triu_mask.sum(-1).sum(-1)
    ratio = need_token_num / 32

    group_similarity = similarity.clone().unsqueeze(1).repeat(1, projector_lengths.shape[-1], 1, 1)
    group_similarity_masks = torch.zeros_like(group_similarity)
    group_index = torch.arange(group_similarity.shape[1], device=group_similarity.device).unsqueeze(0).unsqueeze(-1).repeat(group_similarity.shape[0], 1, group_similarity.shape[-2])
    group_belong_components = belong_components.unsqueeze(1).repeat(1, group_similarity.shape[1], 1)
    group_similarity_masks[group_index == group_belong_components] = 1
    group_similarity_masks[torch.where(group_similarity_masks.transpose(3, 2) == 1)] = 1
    group_similarity[group_similarity_masks != 1] = 0
    group_ids, projector_scores = _prunesid_similarity_nms(group_similarity, projector_scores, ratio * sim_mean)
    group_ids_mask = (group_ids != -1)
    keep_nms_counts = group_ids_mask.sum(dim=-1)
    group_counts = (projector_mask == index_map).sum(dim=1)
    group_lower_bound = torch.ones_like(group_counts, device=group_counts.device)
    group_lower_bound = torch.min(torch.cat([group_lower_bound.unsqueeze(0), group_counts.unsqueeze(0)], dim=0), dim=0)[0]
    group_upper_bound = torch.ones_like(group_counts, device=group_counts.device) * 5 * math.ceil(need_token_num / 64)
    group_upper_bound = torch.min(torch.cat([group_upper_bound.unsqueeze(0), group_counts.unsqueeze(0)], dim=0), dim=0)[0]
    group_upper_bound = torch.min(torch.cat([group_upper_bound.unsqueeze(0), keep_nms_counts.unsqueeze(0)], dim=0), dim=0)[0]
    while torch.any(group_upper_bound.sum(-1) < need_token_num):
        group_upper_bound[torch.where(group_upper_bound.sum(-1) < need_token_num)] += 1
        group_upper_bound = torch.min(torch.cat([group_upper_bound.unsqueeze(0), group_counts.unsqueeze(0)], dim=0), dim=0)[0]

    other_token_nums = need_token_num - group_lower_bound.sum(-1) - 1
    other_token_nums[other_token_nums < 0] = 0
    norm_group_counts = keep_nms_counts / keep_nms_counts.sum(-1, keepdim=True)
    cumulative_sum = torch.cumsum(norm_group_counts, dim=-1)
    other_token_d = (cumulative_sum * other_token_nums.unsqueeze(-1).expand(-1, group_counts.shape[1])).round().int()
    other_token_d = other_token_d - torch.cat([torch.zeros((other_token_d.shape[0], 1), device=other_token_d.device), other_token_d[:, :-1]], dim=-1)
    group_token_d = other_token_d + group_lower_bound
    group_token_d = torch.min(torch.cat([group_token_d.unsqueeze(0), group_upper_bound.unsqueeze(0)], dim=0), dim=0)[0]
    group_sort_index = torch.argsort(keep_nms_counts, dim=-1, descending=True)
    filling_group = torch.zeros(group_counts.shape[0], device=group_counts.device).int()
    while torch.any(group_token_d.sum(-1) < other_token_nums + group_lower_bound.sum(-1)):
        need_filling_batch = torch.where(group_token_d.sum(-1) < other_token_nums + group_lower_bound.sum(-1))[0]
        filling_num = torch.min(torch.stack(
            [group_upper_bound[need_filling_batch, group_sort_index[need_filling_batch, filling_group[need_filling_batch]]] - group_token_d[need_filling_batch, group_sort_index[need_filling_batch, filling_group[need_filling_batch]]],
             other_token_nums[need_filling_batch] + group_lower_bound[need_filling_batch].sum(-1) - group_token_d[need_filling_batch].sum(-1)]
        ), dim=0)[0]
        group_token_d[need_filling_batch, group_sort_index[need_filling_batch, filling_group[need_filling_batch]]] += filling_num
        filling_group[need_filling_batch] += 1
    projector_sort_index = torch.argsort(projector_scores, dim=1, descending=True)
    projector_sort_index = projector_sort_index.transpose(1, 2).reshape(-1, group_similarity.shape[-1])
    group_token_d = group_token_d.reshape(-1)
    important_indices = []
    for i in range(len(group_token_d)):
        important_indices.append(projector_sort_index[i][:int(group_token_d[i])])
    important_indices = [important_indices[i:i + group_similarity.shape[1]] for i in range(0, len(important_indices), group_similarity.shape[1])]
    return important_indices

# ── Measure T_prunesid_cluster ────────────────────────────────────────────────

def measure_t_prunesid_cluster(device, n_values):
    print("  Capturing ViT features (random weights) for PruneSID clustering ...", flush=True)
    hidden_states, attn_weights = _capture_vit_prunesid(device)

    results = {}
    for i, n in enumerate(n_values):
        for _ in range(2):
            out = _prunesid_cluster_body(hidden_states, attn_weights, n)
        n_out = sum(t.numel() for t in out[0])
        mean, std = timed_run(lambda n=n: _prunesid_cluster_body(hidden_states, attn_weights, n),
                              WARMUP, REPEAT_CLUSTER)
        results[n] = {"mean_ms": mean, "std_ms": std, "n_out": n_out}
        if (i + 1) % 10 == 0 or i == len(n_values) - 1:
            print(f"  [{i+1:>4}/{len(n_values)}] target_N={n:<4}  {mean:.3f} ± {std:.3f} ms  (-> {n_out} tokens)", flush=True)

    return results

# ── Print & CSV ───────────────────────────────────────────────────────────────

SEP = "═" * 62

def hdr(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

def print_results(device_name, sys_info, cluster):
    hdr("Device info")
    print(f"  Device  : {device_name}")
    print(f"  GPU     : {sys_info['gpu']}")
    print(f"  CUDA    : {sys_info['cuda_version']}")
    print(f"  RAM     : {sys_info['ram_gb']} GB")
    print(f"  PyTorch : {sys_info['torch']}")

    hdr("T_prunesid_cluster  (post-ViT PCA-cluster + NMS diversity selection only)")
    if not cluster:
        print("  [FAILED]")
    else:
        ns = sorted(cluster.keys())
        print(f"  N range : {ns[0]}..{ns[-1]}  ({len(ns)} values)")
        landmarks = [ns[0]] + [n for n in [23, 40, 64, 87, 183, 305, 576] if n in ns] + [ns[-1]]
        for n in sorted(set(landmarks)):
            d = cluster[n]
            print(f"    target_N={n:<4}  {d['mean_ms']:.3f} ± {d['std_ms']:.3f} ms  (-> {d['n_out']} tokens)")
        print(f"\n  (full {len(ns)} rows saved to CSV)")
        print(f"\n  Reference: T_edge (ViT-L/14-336 forward) = 231.478099784581 ms")
        print(f"             measured separately by measure_edge.py")

def export_csv(device_name, cluster):
    rows = []
    for n in sorted(cluster.keys()):
        d = cluster[n]
        rows.append({"component": "T_prunesid_cluster", "N": n,
                     "mean_ms": d["mean_ms"], "std_ms": d["std_ms"], "n_out": d["n_out"]})

    safe  = device_name.replace(" ", "_").replace("/", "-")
    fname = str(Path(__file__).resolve().parent / "data" / f"results_edge_prunesid_{safe}.csv")
    fields = ["component", "N", "mean_ms", "std_ms", "n_out"]
    with open(fname, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {fname}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PruneSID Clustering Latency Measurement")
    parser.add_argument("--device",   type=str, default="Unknown Device")
    parser.add_argument("--n-step",   type=int, default=DEFAULT_N_STEP,
                        help=f"target-N sweep step (default {DEFAULT_N_STEP})")
    parser.add_argument("--n-values", type=int, nargs="+", default=None,
                        help="Explicit target-N values (overrides --n-step); "
                             "e.g. --n-values 23 40 64 79 87 183 305 295 576 "
                             "for the paper's 8 budgets + full N=576")
    args = parser.parse_args()

    if args.n_values:
        n_values = sorted(set(n for n in args.n_values if N_MIN <= n <= N_MAX))
    else:
        n_values = list(range(N_MIN, N_MAX + 1, args.n_step))
        if N_MAX not in n_values:
            n_values.append(N_MAX)

    device   = get_device()
    sys_info = detect_system_info()

    print(SEP)
    print("  PruneSID Clustering Latency Measurement  →  results_edge_prunesid_<device>.csv")
    print(SEP)
    print(f"  Device  : {args.device}  |  Torch: {device}  |  GPU: {sys_info['gpu']}")
    print(f"  target_N: {n_values[0]}..{n_values[-1]}  ({len(n_values)} values)")

    print(f"\n[1/1] T_prunesid_cluster  ({len(n_values)} target-N values) ...")
    cluster = measure_t_prunesid_cluster(device, n_values)

    print_results(args.device, sys_info, cluster)

    hdr("CSV Output")
    export_csv(args.device, cluster)

if __name__ == "__main__":
    main()
