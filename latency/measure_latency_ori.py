#!/usr/bin/env python3
"""
measure_latency_ori.py — full unmodified pipeline on one device (edge-only / cloud-only baselines)
Full LLaVA-1.5 pipeline (no PruMerge) on a single device, random weights.

Pipeline: CLIPVisionModel  →  MLP Projector  →  concat text(40)  →  LlamaModel
Visual tokens: 576  |  Text tokens: 40  |  Total LLM input: 616 tokens

Usage:
    python measure_latency_ori.py [--device "Jetson AGX Orin"]
Output:
    results_latency_ori.csv  (columns: component, mean_ms, std_ms)
    Row: T_latency_ori
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Config ────────────────────────────────────────────────────────────────────

N_VISUAL    = 576   # ViT-L/14-336 patch tokens
TEXT_TOKENS = 40
VIT_DIM     = 1024
PROJ_DIM    = 4096
WARMUP      = 10
REPEAT      = 20

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

# ── Model building ────────────────────────────────────────────────────────────

def build_vit(device):
    from transformers import CLIPVisionModel, CLIPVisionConfig
    config = CLIPVisionConfig(
        hidden_size=1024, intermediate_size=4096,
        num_hidden_layers=24, num_attention_heads=16,
        image_size=336, patch_size=14, projection_dim=768,
    )
    print("  Building CLIPVisionModel (ViT-L/14-336, random weights) ...", flush=True)
    return CLIPVisionModel(config).to(device=device, dtype=torch.float16).eval()

def build_projector(device):
    """LLaVA-1.5 MLP projector: Linear(1024,4096) + GELU + Linear(4096,4096)."""
    proj = nn.Sequential(
        nn.Linear(VIT_DIM, PROJ_DIM),
        nn.GELU(),
        nn.Linear(PROJ_DIM, PROJ_DIM),
    ).to(device=device, dtype=torch.float16).eval()
    print("  Built MLP Projector (1024→4096, random weights).", flush=True)
    return proj

def build_llm(device):
    from transformers import LlamaConfig, LlamaModel
    config = LlamaConfig(
        hidden_size=4096, intermediate_size=11008,
        num_hidden_layers=32, num_attention_heads=32,
        num_key_value_heads=32, max_position_embeddings=4096,
        vocab_size=32000,
    )
    print("  Building LlamaModel (Vicuna-7B arch, random weights) ...", flush=True)
    return LlamaModel(config).to(device=device, dtype=torch.float16).eval()

# ── Measurement ───────────────────────────────────────────────────────────────

def measure_latency_ori(vit, projector, llm, device):
    pv          = torch.randn(1, 3, 336, 336, device=device, dtype=torch.float16)
    text_embeds = torch.randn(1, TEXT_TOKENS, PROJ_DIM, device=device, dtype=torch.float16)

    def run():
        with torch.no_grad():
            # ViT
            vit_out  = vit(pixel_values=pv, output_hidden_states=True)
            img_feat = vit_out.hidden_states[-2][:, 1:]        # [1, 576, 1024]
            # Projector
            proj_out = projector(img_feat)                      # [1, 576, 4096]
            # Concat visual + text
            inputs   = torch.cat([proj_out, text_embeds], dim=1)  # [1, 616, 4096]
            # LLM
            llm(inputs_embeds=inputs, use_cache=False)

    print(f"  Warm-up {WARMUP} + measure {REPEAT} runs ...", flush=True)
    mean, std = timed_run(run, WARMUP, REPEAT)
    return {"mean_ms": mean, "std_ms": std}

# ── Print & CSV ───────────────────────────────────────────────────────────────

SEP = "═" * 62

def hdr(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

def print_results(device_name, sys_info, result):
    hdr("Device info")
    print(f"  Device  : {device_name}")
    print(f"  GPU     : {sys_info['gpu']}")
    print(f"  CUDA    : {sys_info['cuda_version']}")
    print(f"  RAM     : {sys_info['ram_gb']} GB")
    print(f"  PyTorch : {sys_info['torch']}")

    hdr(f"T_latency_ori  (ViT + Projector + LLM, {N_VISUAL} visual + {TEXT_TOKENS} text tokens)")
    print(f"  {result['mean_ms']:.3f} ± {result['std_ms']:.3f} ms")
    print()

def export_csv(device_name, result):
    safe  = device_name.replace(" ", "_").replace("/", "-")
    fname = str(Path(__file__).resolve().parent / "data" / f"results_latency_ori_{safe}.csv")
    fields = ["component", "mean_ms", "std_ms"]
    with open(fname, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow({"component": "T_latency_ori",
                    "mean_ms": result["mean_ms"],
                    "std_ms":  result["std_ms"]})
    print(f"  Saved: {fname}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Full-pipeline latency (original LLaVA-1.5)")
    parser.add_argument("--device", type=str, default="Unknown Device",
                        help='Label, e.g. "Jetson AGX Orin" or "RTX 3090"')
    args = parser.parse_args()

    device   = get_device()
    sys_info = detect_system_info()

    print(SEP)
    print("  measure_latency_ori.py  →  results_latency_ori.csv")
    print(SEP)
    print(f"  Device: {args.device}  |  Torch: {device}  |  GPU: {sys_info['gpu']}")

    print("\n[1/4] ViT ...")
    vit = build_vit(device)

    print("\n[2/4] Projector ...")
    projector = build_projector(device)

    print("\n[3/4] LLM ...")
    llm = build_llm(device)

    print("\n[4/4] Measuring T_latency_ori ...")
    result = measure_latency_ori(vit, projector, llm, device)

    print_results(args.device, sys_info, result)

    hdr("CSV Output")
    export_csv(args.device, result)

if __name__ == "__main__":
    main()
