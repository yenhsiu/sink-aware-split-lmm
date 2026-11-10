#!/usr/bin/env python3
"""
Cloud-side latency measurement.
Measures: T_cloud(N) = Projector (MLP 1024→4096) + LLM prefill, sweeping N = 1..576

Input pipeline per N:
    torch.randn(1, N, 1024).half()  →  Projector  →  concat text(40 tokens)  →  LlamaModel

Usage:
    python measure_cloud.py [--device "RTX 3090"]              # default: step=8, 73 N values
    python measure_cloud.py --n-step 1                         # all 576 N values (slow)
    python measure_cloud.py --n-values 36 145 576              # specific N values only
Output:
    results_cloud_<device>.csv  (columns: component, N, mean_ms, std_ms)
"""

import argparse
import csv
import platform
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn

# ── Config ────────────────────────────────────────────────────────────────────

VIT_DIM       = 1024
PROJ_DIM      = 4096
TEXT_TOKENS   = 40
N_MIN         = 1
N_MAX         = 576
DEFAULT_N_STEP = 8    # default: 73 values; use --n-step 1 for all 576 (slow on LLM)
WARMUP        = 10
REPEAT_CLOUD  = 50

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

def build_projector(device):
    """LLaVA-1.5 MLP projector: Linear(1024,4096) + GELU + Linear(4096,4096)."""
    proj = nn.Sequential(
        nn.Linear(VIT_DIM, PROJ_DIM),
        nn.GELU(),
        nn.Linear(PROJ_DIM, PROJ_DIM),
    ).to(device=device, dtype=torch.float16).eval()
    return proj

def build_llama_model(device):
    """LlamaModel (Vicuna-7B architecture) with random weights."""
    from transformers import LlamaConfig, LlamaModel
    config = LlamaConfig(
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        max_position_embeddings=4096,
        vocab_size=32000,
    )
    print("  Building LlamaModel (Vicuna-7B arch, random weights, float16) ...", flush=True)
    model = LlamaModel(config).to(device=device, dtype=torch.float16).eval()
    return model

# ── T_cloud ───────────────────────────────────────────────────────────────────

def measure_t_cloud(projector, llm, device, n_values):
    """
    For each N: time Projector + LLM prefill together.
    Input:  vis_tokens [1, N, 1024]  →  proj [1, N, 4096]  +  text [1, 40, 4096]
    Concat: [1, N+40, 4096]  →  LlamaModel
    """
    text_embeds = torch.randn(1, TEXT_TOKENS, PROJ_DIM, device=device, dtype=torch.float16)
    results = {}
    total   = len(n_values)

    for idx, n in enumerate(n_values):
        vis = torch.randn(1, n, VIT_DIM, device=device, dtype=torch.float16)

        def run(v=vis):
            with torch.no_grad():
                proj_out = projector(v)
                inputs   = torch.cat([proj_out, text_embeds], dim=1)
                llm(inputs_embeds=inputs, use_cache=False)

        mean, std = timed_run(run, WARMUP, REPEAT_CLOUD)
        results[n] = {"mean_ms": mean, "std_ms": std}

        if (idx + 1) % 10 == 0 or idx == 0 or idx == total - 1:
            print(f"  [{idx+1:>4}/{total}] N={n:<4}  {mean:.1f} ± {std:.1f} ms", flush=True)

    return results

# ── Print & CSV ───────────────────────────────────────────────────────────────

SEP = "═" * 62

def hdr(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

def print_results(device_name, sys_info, results):
    hdr("Device info")
    print(f"  Device  : {device_name}")
    print(f"  GPU     : {sys_info['gpu']}")
    print(f"  CUDA    : {sys_info['cuda_version']}")
    print(f"  RAM     : {sys_info['ram_gb']} GB")
    print(f"  PyTorch : {sys_info['torch']}")

    measured_ns = sorted(results)
    hdr(f"T_cloud  (Projector + LLM prefill, text={TEXT_TOKENS} tokens)  — summary")
    print(f"  N range : {measured_ns[0]}..{measured_ns[-1]}  ({len(measured_ns)} values)")
    landmarks = sorted(set(
        [measured_ns[0]] + [n for n in [36, 145, 576] if n in results] + [measured_ns[-1]]
    ))
    print(f"  {'N':<6}  {'mean_ms':>10}  {'std_ms':>8}")
    print(f"  {'─'*30}")
    for n in landmarks:
        d = results[n]
        print(f"  N={n:<4}  {d['mean_ms']:>10.3f}  {d['std_ms']:>8.3f} ms")
    print(f"\n  (full {len(measured_ns)} rows saved to CSV)")

def export_csv(device_name, results):
    safe  = device_name.replace(" ", "_").replace("/", "-")
    fname = str(Path(__file__).resolve().parent / "data" / f"results_cloud_{safe}.csv")
    fields = ["component", "N", "mean_ms", "std_ms"]
    rows = [
        {"component": "T_cloud", "N": n,
         "mean_ms": results[n]["mean_ms"], "std_ms": results[n]["std_ms"]}
        for n in sorted(results)
    ]
    with open(fname, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {fname}  ({len(rows)} rows)")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cloud Latency Measurement")
    parser.add_argument("--device", type=str, default="Unknown GPU")
    parser.add_argument("--n-step", type=int, default=DEFAULT_N_STEP,
                        help=f"Token count step (default {DEFAULT_N_STEP}). Use 1 for all 576 values.")
    parser.add_argument("--n-values", type=int, nargs="+", default=None,
                        help="Explicit N values to test (overrides --n-step)")
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
    print("  Cloud Latency Measurement  →  results_cloud_<device>.csv")
    print(SEP)
    print(f"  Device  : {args.device}  |  Torch: {device}  |  GPU: {sys_info['gpu']}")
    print(f"  N range : {n_values[0]}..{n_values[-1]}  ({len(n_values)} values, step={args.n_step})")

    print("\n[1/3] Building Projector (MLP 1024→4096) ...")
    projector = build_projector(device)

    print("\n[2/3] Building LlamaModel (Vicuna-7B arch, random weights) ...")
    llm = build_llama_model(device)

    print(f"\n[3/3] Measuring T_cloud  (warmup={WARMUP}, repeat={REPEAT_CLOUD} per N) ...")
    results = measure_t_cloud(projector, llm, device, n_values)

    print_results(args.device, sys_info, results)

    hdr("CSV Output")
    export_csv(args.device, results)

if __name__ == "__main__":
    main()
