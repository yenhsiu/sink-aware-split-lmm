#!/usr/bin/env python3
"""
Edge-side latency measurement.
Measures: T_edge (ViT only), T_prumerge_base (post-ViT), T_prumerge_plus (post-ViT), T_quant(N,B)

Usage:
    python measure_edge.py [--device "Jetson AGX Orin"]
Output:
    results_edge.csv  (columns: component, N, B, mean_ms, std_ms)
"""

import argparse
import csv
import platform
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Config ────────────────────────────────────────────────────────────────────

BIT_VALUES    = [4, 2, 1]
TOKEN_DIM     = 1024
N_MIN         = 1
N_MAX         = 576
DEFAULT_N_STEP = 1      # sweep every N by default (MSECompressor is fast)
WARMUP        = 10
REPEAT_EDGE   = 100
REPEAT_PM     = 100
REPEAT_QUANT  = 100

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    return info

# ── T_edge ────────────────────────────────────────────────────────────────────

def measure_t_edge(device):
    from transformers import CLIPVisionModel, CLIPVisionConfig
    config = CLIPVisionConfig(
        hidden_size=1024, intermediate_size=4096,
        num_hidden_layers=24, num_attention_heads=16,
        image_size=336, patch_size=14, projection_dim=768,
    )
    print("  Building CLIPVisionModel (ViT-L/14-336, random weights, float16) ...", flush=True)
    model = CLIPVisionModel(config).to(device=device, dtype=torch.float16).eval()
    pv = torch.randn(1, 3, 336, 336, device=device, dtype=torch.float16)

    def run():
        with torch.no_grad():
            model(pixel_values=pv, output_hidden_states=True)

    mean, std = timed_run(run, WARMUP, REPEAT_EDGE)
    del model
    return {"mean_ms": mean, "std_ms": std}

# ── PruMerge helpers ──────────────────────────────────────────────────────────

def _build_vit_tower(device):
    """Build CLIPVisionTower (from clip_encoder.py) with random weights."""
    from transformers import CLIPVisionModel, CLIPVisionConfig
    from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower

    config = CLIPVisionConfig(
        hidden_size=1024, intermediate_size=4096,
        num_hidden_layers=24, num_attention_heads=16,
        image_size=336, patch_size=14, projection_dim=768,
    )
    tower = CLIPVisionTower.__new__(CLIPVisionTower)
    nn.Module.__init__(tower)
    tower.is_loaded = False
    tower.vision_tower_name = "openai/clip-vit-large-patch14-336"
    tower.select_layer = -2
    tower.select_feature = "patch"
    tower.total_tokens = 0
    tower.image_processor = None
    tower.vision_tower = CLIPVisionModel(config)
    tower.vision_tower.requires_grad_(False)
    tower.is_loaded = True
    tower = tower.to(device=device, dtype=torch.float16).eval()
    return tower

def _capture_vit(tower, pv):
    """Run ViT once with hooks; return (image_features, k, q) — all cloned."""
    captured = {}

    def hk(m, i, o): captured['k'] = o.detach().clone()
    def hq(m, i, o): captured['q'] = o.detach().clone()

    h1 = tower.vision_tower.vision_model.encoder.layers[23].self_attn.k_proj.register_forward_hook(hk)
    h2 = tower.vision_tower.vision_model.encoder.layers[23].self_attn.q_proj.register_forward_hook(hq)

    with torch.no_grad():
        out = tower.vision_tower(pv, output_hidden_states=True)
    h1.remove(); h2.remove()

    img_feat = out.hidden_states[-2][:, 1:].to(pv.dtype).clone()  # [1, 576, 1024]
    return img_feat, captured['k'], captured['q']

# ── PruMerge base body (post-ViT operations only) ────────────────────────────

def _prumerge_base_body(img_feat, k_t, q_t):
    from llava.model.multimodal_encoder.clip_encoder import complement_idx, outlier_dectection

    B, N, C = img_feat.shape
    attn = (q_t @ k_t.transpose(-2, -1)) * C ** -0.5
    attn = F.softmax(attn, dim=-1)
    cls_attn = attn[:, 0, 1:]

    reduction_ratio = outlier_dectection(cls_attn)
    _, idx = torch.topk(cls_attn, int(N * reduction_ratio), dim=1, largest=True)
    index = idx.unsqueeze(-1).expand(-1, -1, C)

    Key_wo_cls    = k_t[:, 1:]
    x_others      = torch.gather(img_feat, dim=1, index=index)
    x_others_attn = torch.gather(cls_attn, dim=1, index=idx)
    Key_others    = torch.gather(Key_wo_cls, dim=1, index=index)
    compl         = complement_idx(idx, N)
    non_topk      = torch.gather(img_feat, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))
    non_topk_Key  = torch.gather(Key_wo_cls, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))
    non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)

    Key_others_norm   = F.normalize(Key_others, p=2, dim=-1)
    non_topk_Key_norm = F.normalize(non_topk_Key, p=2, dim=-1)

    B2, left_tokens, _ = x_others.size()
    updated = torch.zeros_like(x_others)

    for b in range(B2):
        for i in range(left_tokens):
            kn       = Key_others_norm[b, i].unsqueeze(0).unsqueeze(0)
            rest_k   = torch.cat([Key_others_norm[b, :i].unsqueeze(0),
                                   Key_others_norm[b, i+1:].unsqueeze(0),
                                   non_topk_Key_norm[b].unsqueeze(0)], dim=1)
            rest_x   = torch.cat([x_others[b, :i].unsqueeze(0),
                                   x_others[b, i+1:].unsqueeze(0),
                                   non_topk[b].unsqueeze(0)], dim=1)
            rest_a   = torch.cat([x_others_attn[b, :i].unsqueeze(0),
                                   x_others_attn[b, i+1:].unsqueeze(0),
                                   non_topk_attn[b].unsqueeze(0)], dim=1)
            sim      = torch.bmm(kn, rest_k.transpose(1, 2))
            _, ci    = torch.topk(sim, k=32, dim=2, largest=True)
            ct       = rest_x[:, ci.squeeze(), :]
            w        = rest_a[:, ci.squeeze()].unsqueeze(-1)
            updated[b, i] = torch.sum(ct * w, dim=1) + x_others[b, i]

    extra = torch.sum(non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True)
    return torch.cat([updated, extra], dim=1)

# ── PruMerge plus body (post-ViT operations only) ────────────────────────────

def _prumerge_plus_body(img_feat, k_t, q_t, device):
    from llava.model.multimodal_encoder.clip_encoder import complement_idx, outlier_dectection

    B, N, C = img_feat.shape
    attn = (q_t @ k_t.transpose(-2, -1)) * C ** -0.5
    attn = F.softmax(attn, dim=-1)
    cls_attn = attn[:, 0, 1:]

    reduction_ratio = outlier_dectection(cls_attn)
    _, idx = torch.topk(cls_attn, int(N * reduction_ratio), dim=1, largest=True)

    # Spatial supplementation
    step_length = int(1 / reduction_ratio)
    arith = torch.arange(0, 575, int(step_length / 3), device=device)
    orig_1d = idx.flatten()
    filtered = torch.tensor([x.item() for x in arith if x not in orig_1d],
                             dtype=torch.long, device=device)
    if filtered.numel() > 0:
        idx = torch.cat((idx, filtered.unsqueeze(0)), dim=1)

    index = idx.unsqueeze(-1).expand(-1, -1, C)

    Key_wo_cls    = k_t[:, 1:]
    x_others      = torch.gather(img_feat, dim=1, index=index)
    x_others_attn = torch.gather(cls_attn, dim=1, index=idx)
    Key_others    = torch.gather(Key_wo_cls, dim=1, index=index)
    compl         = complement_idx(idx, N)
    non_topk      = torch.gather(img_feat, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))
    non_topk_Key  = torch.gather(Key_wo_cls, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))
    non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)

    Key_others_norm   = F.normalize(Key_others, p=2, dim=-1)
    non_topk_Key_norm = F.normalize(non_topk_Key, p=2, dim=-1)

    B2, left_tokens, _ = x_others.size()
    updated = torch.zeros_like(x_others)

    for b in range(B2):
        for i in range(left_tokens):
            kn       = Key_others_norm[b, i].unsqueeze(0).unsqueeze(0)
            rest_k   = torch.cat([Key_others_norm[b, :i].unsqueeze(0),
                                   Key_others_norm[b, i+1:].unsqueeze(0),
                                   non_topk_Key_norm[b].unsqueeze(0)], dim=1)
            rest_x   = torch.cat([x_others[b, :i].unsqueeze(0),
                                   x_others[b, i+1:].unsqueeze(0),
                                   non_topk[b].unsqueeze(0)], dim=1)
            rest_a   = torch.cat([x_others_attn[b, :i].unsqueeze(0),
                                   x_others_attn[b, i+1:].unsqueeze(0),
                                   non_topk_attn[b].unsqueeze(0)], dim=1)
            sim      = torch.bmm(kn, rest_k.transpose(1, 2))
            _, ci    = torch.topk(sim, k=32, dim=2, largest=True)
            ct       = rest_x[:, ci.squeeze(), :]
            w        = rest_a[:, ci.squeeze()].unsqueeze(-1)
            updated[b, i] = x_others[b, i] + torch.sum(ct * w, dim=1)

    extra = torch.sum(non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True)
    return torch.cat([updated, extra], dim=1)

# ── Measure T_prumerge ────────────────────────────────────────────────────────

def measure_t_prumerge_base(device, tower, pv):
    print("  Capturing ViT features for PruMerge base ...", flush=True)
    img_feat, k_t, q_t = _capture_vit(tower, pv)

    for _ in range(3):
        out = _prumerge_base_body(img_feat, k_t, q_t)
    n_out = out.shape[1]
    print(f"  PruMerge base output: {n_out} tokens", flush=True)

    mean, std = timed_run(lambda: _prumerge_base_body(img_feat, k_t, q_t), WARMUP, REPEAT_PM)
    return {"mean_ms": mean, "std_ms": std, "n_out": n_out}

def measure_t_prumerge_plus(device, tower, pv):
    print("  Capturing ViT features for PruMerge plus ...", flush=True)
    img_feat, k_t, q_t = _capture_vit(tower, pv)

    for _ in range(3):
        out = _prumerge_plus_body(img_feat, k_t, q_t, device)
    n_out = out.shape[1]
    print(f"  PruMerge plus output: {n_out} tokens", flush=True)

    mean, std = timed_run(lambda: _prumerge_plus_body(img_feat, k_t, q_t, device), WARMUP, REPEAT_PM)
    return {"mean_ms": mean, "std_ms": std, "n_out": n_out}

# ── T_quant ───────────────────────────────────────────────────────────────────

def measure_t_quant(device, n_values):
    try:
        from turboquant.compressors_v3 import MSECompressor
    except ImportError as e:
        print(f"  [WARN] MSECompressor import failed: {e}")
        return {}

    dev_str = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}
    total   = len(n_values) * len(BIT_VALUES)
    done    = 0

    for bits in BIT_VALUES:
        c = MSECompressor(head_dim=TOKEN_DIM, bits=bits, seed=42, device=dev_str)
        for n in n_values:
            tok = torch.randn(1, 1, n, TOKEN_DIM, device=device)
            mean, std = timed_run(lambda t=tok, comp=c: comp.decompress(comp.compress(t)),
                                  WARMUP, REPEAT_QUANT)
            results[(n, bits)] = {"mean_ms": mean, "std_ms": std}
            done += 1
            if done % 50 == 0 or done == total:
                print(f"  [{done:>5}/{total}] bits={bits} N={n:<4}  {mean:.3f} ms", flush=True)

    return results

# ── Print & CSV ───────────────────────────────────────────────────────────────

SEP = "═" * 62

def hdr(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

def print_results(device_name, sys_info, t_edge, t_pm_base, t_pm_plus, quant):
    hdr("Device info")
    print(f"  Device  : {device_name}")
    print(f"  GPU     : {sys_info['gpu']}")
    print(f"  CUDA    : {sys_info['cuda_version']}")
    print(f"  RAM     : {sys_info['ram_gb']} GB")
    print(f"  PyTorch : {sys_info['torch']}")

    hdr("T_edge  (ViT-L/14-336, random weights, float16)")
    if t_edge["mean_ms"] is not None:
        print(f"  {t_edge['mean_ms']:.3f} ± {t_edge['std_ms']:.3f} ms")
    else:
        print("  skipped")

    hdr("T_prumerge_base  (post-ViT PruMerge base only)")
    if t_pm_base["mean_ms"] is not None:
        print(f"  {t_pm_base['mean_ms']:.3f} ± {t_pm_base['std_ms']:.3f} ms  (→ {t_pm_base.get('n_out','?')} tokens)")
    else:
        print("  skipped")

    hdr("T_prumerge_plus  (post-ViT PruMerge plus only)")
    if t_pm_plus["mean_ms"] is not None:
        print(f"  {t_pm_plus['mean_ms']:.3f} ± {t_pm_plus['std_ms']:.3f} ms  (→ {t_pm_plus.get('n_out','?')} tokens)")
    else:
        print("  skipped")

    hdr("T_quant  (MSECompressor, N x B)  — summary")
    if not quant:
        print("  [FAILED]")
    else:
        measured_ns = sorted(set(n for (n, _) in quant))
        print(f"  N range : {measured_ns[0]}..{measured_ns[-1]}  ({len(measured_ns)} values)")
        # Print a few representative N values only
        landmarks = [measured_ns[0]] + [n for n in [36, 145, 576] if n in measured_ns] + [measured_ns[-1]]
        show_ns = sorted(set(landmarks))
        for bits in BIT_VALUES:
            print(f"\n  {bits}bit:")
            for n in show_ns:
                d = quant.get((n, bits), {})
                m, s = d.get("mean_ms", 0), d.get("std_ms", 0)
                print(f"    N={n:<4}  {m:.3f} ± {s:.3f} ms")
        print(f"\n  (full {len(measured_ns)} × {len(BIT_VALUES)} rows saved to CSV)")

def export_csv(device_name, t_edge, t_pm_base, t_pm_plus, quant):
    rows = [
        {"component": "T_edge",          "N": "-", "B": "-",
         "mean_ms": t_edge["mean_ms"],    "std_ms": t_edge["std_ms"]},
        {"component": "T_prumerge_base", "N": "-", "B": "-",
         "mean_ms": t_pm_base["mean_ms"], "std_ms": t_pm_base["std_ms"]},
        {"component": "T_prumerge_plus", "N": "-", "B": "-",
         "mean_ms": t_pm_plus["mean_ms"], "std_ms": t_pm_plus["std_ms"]},
    ]
    for (n, bits) in sorted(quant.keys()):
        d = quant[(n, bits)]
        rows.append({"component": "T_quant", "N": n, "B": bits,
                     "mean_ms": d["mean_ms"], "std_ms": d["std_ms"]})

    safe  = device_name.replace(" ", "_").replace("/", "-")
    fname = str(Path(__file__).resolve().parent / "data" / f"results_edge_{safe}.csv")
    fields = ["component", "N", "B", "mean_ms", "std_ms"]
    with open(fname, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {fname}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Edge Latency Measurement")
    parser.add_argument("--device",        type=str, default="Unknown Device")
    parser.add_argument("--skip-edge",     action="store_true")
    parser.add_argument("--skip-prumerge", action="store_true")
    parser.add_argument("--skip-quant",    action="store_true")
    parser.add_argument("--n-step",  type=int, default=DEFAULT_N_STEP,
                        help=f"T_quant token count step (default {DEFAULT_N_STEP} = all 1..576)")
    parser.add_argument("--n-values", type=int, nargs="+", default=None,
                        help="Explicit N values for T_quant (overrides --n-step)")
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
    print("  Edge Latency Measurement  →  results_edge_<device>.csv")
    print(SEP)
    print(f"  Device  : {args.device}  |  Torch: {device}  |  GPU: {sys_info['gpu']}")
    print(f"  T_quant N: {n_values[0]}..{n_values[-1]}  ({len(n_values)} values)")

    if args.skip_edge:
        t_edge = {"mean_ms": None, "std_ms": None}
    else:
        print("\n[1/3] T_edge ...")
        t_edge = measure_t_edge(device)

    if args.skip_prumerge:
        t_pm_base = {"mean_ms": None, "std_ms": None, "n_out": None}
        t_pm_plus = {"mean_ms": None, "std_ms": None, "n_out": None}
    else:
        print("\n[2/3] Building CLIPVisionTower for PruMerge ...")
        tower = _build_vit_tower(device)
        pv    = torch.randn(1, 3, 336, 336, device=device, dtype=torch.float16)

        print("\n[2a/3] T_prumerge_base ...")
        t_pm_base = measure_t_prumerge_base(device, tower, pv)

        print("\n[2b/3] T_prumerge_plus ...")
        t_pm_plus = measure_t_prumerge_plus(device, tower, pv)
        del tower

    if args.skip_quant:
        quant = {}
    else:
        print(f"\n[3/3] T_quant  (N={n_values[0]}..{n_values[-1]}, {len(n_values)} × {len(BIT_VALUES)} combos) ...")
        quant = measure_t_quant(device, n_values)

    print_results(args.device, sys_info, t_edge, t_pm_base, t_pm_plus, quant)

    hdr("CSV Output")
    export_csv(args.device, t_edge, t_pm_base, t_pm_plus, quant)

if __name__ == "__main__":
    main()
