"""
Figure 1: Accuracy vs Transmission Budget (log scale x-axis), one panel per benchmark
Figure 2: Accuracy vs Latency (T_total at a fixed uplink bandwidth S), one panel per benchmark

Data verified against results/vqa_anchore/pipeline_vqav2/{MME,POPE,TEXTVQA,MMBENCH}_*.txt
-- every value in paper_data.py matches the corresponding table exactly.

Per-benchmark grid (not a normalized cross-benchmark average) per user request
2026-07-28. Labels renamed All B=1/2/4 -> SP-Q1/SP-Q2/SP-Q4 per user request.

Latency in Figure 2 uses each method's OWN T_cloud(N) (SP-Q1/Q2/Q4 use very
different token counts N at the same nominal budget, so they have different
cloud compute time -- see paper_data.T_CLOUD_BY_METHOD), not a single shared
per-budget value.

Usage:
    python plot_vqav2_figures.py             # both figures, S_fixed=1.0 Mbps
    python plot_vqav2_figures.py --s 2.5     # Figure 2 at a different bandwidth
"""
import argparse

import matplotlib
matplotlib.use("Agg")
# Embed real (subset) TrueType fonts instead of matplotlib's PDF default
# (Type 3 bitmap-ish fonts) -- journal PDF checkers flag Type 3 fonts as
# non-embeddable/non-searchable text.
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, ScalarFormatter, NullFormatter

from paper_data import (BUDGETS, DATA, METHODS, COLORS, STYLES, WIDTHS, LABELS,
                         N_BY_METHOD, T_CLOUD_BY_METHOD, t_edge_total,
                         MIDPOINT_DATA, MIDPOINT_METHOD_KEY, TIER_ID_BITS_PER_TOKEN,
                         NORM_BITS_PER_TOKEN,
                         ANCHOR_CONFIGS, BIT_BY_METHOD, t_quant_stratified)

PROPOSED = "VQAv2-opt"
REFILL_LABEL = {"VQAv2-opt": "Snap-and-refill (S1)"}  # SP-Q1/Q2/Q4 fall back to a generic label

# Per-benchmark metric name, matching the paper's main results table
# (sec:lut_effectiveness) instead of a generic "{bench} Score" -- just the
# metric itself, no benchmark name (already the panel title) or arrow.
# 2026-08-27.
METRIC_YLABEL = {
    "MME": "Perception+Cognition Score",
    "POPE": "F1 Score",
    "TextVQA": "Accuracy",
    "MMBench": "Accuracy",
}


def to_latency(method: str, idx: int, s_mbps: float) -> float:
    budget = BUDGETS[idx]
    n = N_BY_METHOD[method][idx]
    t_cloud = T_CLOUD_BY_METHOD[method][idx]
    # Only Proposed's tokens are genuinely mixed-tier (n4/n2/n1 all
    # potentially nonzero) -- the cloud needs 2 bits/token of tier-ID
    # metadata to unpack them, on top of the quantized payload.
    # SP-Q1/Q2/Q4 are always one uniform bit-width, no tier-ID overhead
    # needed. Every method's tokens carry a 16-bit fp16 L2 norm each
    # (NORM_BITS_PER_TOKEN), regardless of method -- all route through
    # MSECompressor, which always normalizes-then-quantizes.
    overhead = NORM_BITS_PER_TOKEN * n + (TIER_ID_BITS_PER_TOKEN * n if method == PROPOSED else 0)
    t_tx = (budget + overhead) / (s_mbps * 1000)  # ms
    # Quantization (MSECompressor.compress()) latency -- Proposed needs the
    # actual anchor's (n4, n2, n1) split (mixed-tier); SP-Q1/Q2/Q4 are one
    # uniform bit-width, so put all of N into that single tier.
    if method == PROPOSED:
        anchor = ANCHOR_CONFIGS[budget]
        t_quant = t_quant_stratified(anchor["n4"], anchor["n2"], anchor["n1"])
    else:
        bits = BIT_BY_METHOD[method]
        t_quant = t_quant_stratified(n if bits == 4 else 0, n if bits == 2 else 0, n if bits == 1 else 0)
    return t_edge_total(n) + t_quant + t_cloud + t_tx


def _midpoint_budget_series(bench_name: str, method_name: str):
    """Measured midpoint interpolation points for this benchmark/method, as
    (budget, score) -- only points that actually have data (POPE/TextVQA/
    MMBench are empty until those runs happen)."""
    strategy = MIDPOINT_METHOD_KEY[method_name]
    points = MIDPOINT_DATA.get(bench_name, {})
    xs, ys = [], []
    for budget_mid, entry in sorted(points.items()):
        row = entry.get(strategy)
        if row is not None:
            xs.append(entry["actual_budget"])
            ys.append(row["score"])
    return xs, ys


def _midpoint_latency_series(bench_name: str, method_name: str, s_mbps: float):
    """Same points as _midpoint_budget_series, x-axis converted to T_total at
    the given bandwidth using each point's own measured t_cloud(N). Only
    Proposed's midpoints (S1, mixed n4/n2/n1) get the tier-ID overhead --
    same reasoning as to_latency() above. Every strategy's midpoint row
    (S1/B1/B2/B4) already carries its own n4/n2/n1, so T_quant uses that
    directly instead of needing a method-based special case."""
    strategy = MIDPOINT_METHOD_KEY[method_name]
    points = MIDPOINT_DATA.get(bench_name, {})
    overhead_per_token = NORM_BITS_PER_TOKEN + (TIER_ID_BITS_PER_TOKEN if method_name == PROPOSED else 0)
    xs, ys = [], []
    for budget_mid, entry in sorted(points.items()):
        row = entry.get(strategy)
        if row is not None:
            tx_bits = entry["actual_budget"] + overhead_per_token * row["N"]
            t_quant = t_quant_stratified(row["n4"], row["n2"], row["n1"])
            t_total = t_edge_total(row["N"]) + t_quant + row["t_cloud"] + tx_bits / (s_mbps * 1000)
            xs.append(t_total)
            ys.append(row["score"])
    return xs, ys


# Font sizes tuned for a 2-column paper figure (default matplotlib sizes
# read as too small once shrunk to column width) -- 2026-08-17.
TITLE_FS = 22
LABEL_FS = 19
TICK_FS = 16
LEGEND_FS = 17


def plot_figure1(out_path="figure1_per_benchmark.pdf", show_midpoints=False):
    # 1x4 row spanning the full page width (meant for \figure* in a 2-column
    # paper) instead of a 2x2 grid -- plus one shared legend instead of 4
    # duplicate per-panel ones (all 4 panels plot the same series), so the
    # freed-up space goes to the plot area itself. 2026-08-18.
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, (bench_name, methods) in zip(axes, DATA.items()):
        for method_name in METHODS:
            scores = methods[method_name]
            # Midpoints are only drawn for Proposed -- that's the method that
            # actually does continuous snap-and-refill between anchors (the
            # paper's contribution); SP-Q1/Q2/Q4's own midpoint scores exist
            # in MIDPOINT_DATA for the S1-vs-uniform comparison tables, but
            # marking all 4 series here just clutters the plot without adding
            # to the figure's point.
            is_proposed = method_name == PROPOSED
            rx, ry = _midpoint_budget_series(bench_name, method_name) if (show_midpoints and is_proposed) else ([], [])
            if rx:
                # one continuous line through anchors + midpoint points, but the
                # midpoints get a hollow diamond marker -- a different shape
                # (not just hollow-vs-filled) so it stays legible even in
                # grayscale/print, and reads as "filled in", not as extra
                # searched anchors
                combined = sorted(zip(list(BUDGETS) + rx, list(scores) + ry))
                cx, cy = zip(*combined)
                ax.plot(cx, cy, color=COLORS[method_name], linestyle=STYLES[method_name],
                        linewidth=WIDTHS[method_name], zorder=4)
                ax.scatter(BUDGETS, scores, color=COLORS[method_name], marker='o',
                           s=16, zorder=5, label=LABELS[method_name])
                ax.scatter(rx, ry, facecolors='none', edgecolors=COLORS[method_name],
                           marker='D', s=55, linewidths=2.0, zorder=6,
                           label=REFILL_LABEL.get(method_name, f"{LABELS[method_name]} (midpoints)"))
            else:
                ax.plot(BUDGETS, scores,
                        label=LABELS[method_name],
                        color=COLORS[method_name],
                        linestyle=STYLES[method_name],
                        linewidth=WIDTHS[method_name],
                        marker='o', markersize=4)
        ax.set_xscale('log')
        ax.set_xlabel('Transmission Budget (bits, log scale)', fontsize=LABEL_FS)
        ax.set_ylabel(METRIC_YLABEL[bench_name], fontsize=LABEL_FS)
        ax.set_title(bench_name, fontsize=TITLE_FS)
        ax.tick_params(axis='both', labelsize=TICK_FS)
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels),
               bbox_to_anchor=(0.5, 1.08), fontsize=LEGEND_FS + 1, frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"saved {out_path}")


def plot_figure2(s_mbps: float, out_path="figure2_per_benchmark.pdf", show_midpoints=True):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, (bench_name, methods) in zip(axes, DATA.items()):
        for method_name in METHODS:
            scores = methods[method_name]
            latencies = [to_latency(method_name, i, s_mbps) for i in range(len(BUDGETS))]
            is_proposed = method_name == PROPOSED
            rx, ry = _midpoint_latency_series(bench_name, method_name, s_mbps) if (show_midpoints and is_proposed) else ([], [])
            if rx:
                combined = sorted(zip(latencies + rx, list(scores) + ry))
                cx, cy = zip(*combined)
                # Main line carries no marker (midpoints must stay hollow-
                # only, no filled dot underneath); anchors get a small
                # filled dot, midpoints get a larger hollow ring.
                ax.plot(cx, cy, color=COLORS[method_name], linestyle=STYLES[method_name],
                        linewidth=WIDTHS[method_name], zorder=4)
                ax.scatter(latencies, scores, color=COLORS[method_name], marker='o',
                           s=16, zorder=5)
                ax.scatter(rx, ry, facecolors='none', edgecolors=COLORS[method_name],
                           marker='o', s=40, linewidths=1.6, zorder=6)
                # Proxy (off-plot) handle purely for the legend swatch --
                # shows a line+marker like every comparison method's entry,
                # without drawing an actual filled marker at midpoints.
                ax.plot([], [], color=COLORS[method_name], linestyle=STYLES[method_name],
                        linewidth=WIDTHS[method_name], marker='o', markersize=4,
                        label=LABELS[method_name])
            else:
                ax.plot(latencies, scores,
                        label=LABELS[method_name],
                        color=COLORS[method_name],
                        linestyle=STYLES[method_name],
                        linewidth=WIDTHS[method_name],
                        marker='o', markersize=4)
        # Log-scale x-axis: at low S, T_total spans ~300-1100ms across the 8
        # budgets, and most of the accuracy-differentiating action happens in
        # the crowded low end (~300-500ms) -- linear scale squeezed those
        # points together against the few high-latency ones. 2026-08-26 per
        # user request.
        ax.set_xscale('log')
        # Default log-scale ticks land on powers of 10 with "3x10^2"-style
        # scientific labels within this narrow (~296-1087ms) range -- fixed
        # round-number ticks + plain ScalarFormatter read far more clearly.
        ax.xaxis.set_major_locator(FixedLocator([300, 400, 500, 700, 1000]))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_ylabel(METRIC_YLABEL[bench_name], fontsize=LABEL_FS)
        ax.set_title(bench_name, fontsize=TITLE_FS)
        ax.tick_params(axis='y', labelsize=TICK_FS)
        ax.tick_params(axis='x', labelsize=TICK_FS + 3)
        ax.grid(True, alpha=0.3)
    fig.supxlabel(rf'$T_{{\mathrm{{total}}}}$ (ms, log scale) at $C_{{\mathrm{{current}}}}$={s_mbps}Mbps', fontsize=LABEL_FS + 3, y=0.02)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels),
               bbox_to_anchor=(0.5, 1.08), fontsize=LEGEND_FS + 1, frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"saved {out_path}")


def print_win_counts():
    summary = {}
    for bench_name, methods in DATA.items():
        wins = 0
        for i in range(len(BUDGETS)):
            scores_at_i = {m: v[i] for m, v in methods.items()}
            best_method = max(scores_at_i, key=scores_at_i.get)
            if best_method == "VQAv2-opt" or scores_at_i["VQAv2-opt"] == scores_at_i[best_method]:
                wins += 1
        summary[bench_name] = f"{wins}/{len(BUDGETS)}"
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s", type=float, default=1.0, help="Uplink bandwidth in Mbps for Figure 2 (default: 1.0)")
    parser.add_argument("--show-midpoints-fig1", action="store_true",
                         help="Also draw the snap-and-refill/midpoint points on Figure 1 "
                              "(default: Figure 1 shows only the 8 LUT anchor points)")
    parser.add_argument("--hide-midpoints", action="store_true",
                         help="Hide the snap-and-refill/midpoint interpolation points on Figure 2 "
                              "(default: shown)")
    args = parser.parse_args()

    plot_figure1(show_midpoints=args.show_midpoints_fig1)
    plot_figure2(args.s, show_midpoints=not args.hide_midpoints)
    print_win_counts()
