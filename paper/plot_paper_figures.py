"""
Figure 3: Bandwidth vs End-to-End Latency.

Computation only -- no new inference. Up to three tiers of comparison (decided
with the user 2026-07-29), to separate "adaptation itself helps" from "mixed-
precision search helps on top of adaptation":

  1. Static-Q1/Q2/Q4  -- always drawn. No PruMerge, no bandwidth-adaptation:
     always send all N=576 tokens at a fixed bit-depth (1/2/4), ignoring
     T_deadline entirely.
  2. SP-Q1/Q2/Q4      -- shown by default (opt-out via show_spq=False / --hide-spq;
     re-added 2026-08-07 per user request). SAME definition as Figures 1/2
     (uniform bit-depth, N tuned per budget):
     here evaluated via "dynamic snap-down" over each method's own 8-anchor
     (budget, N, T_cloud(N)) LUT family -- at each bandwidth S, pick the
     largest budget whose T_total <= T_deadline, else the smallest/cheapest
     anchor. Adapts N, but NOT the quantization strategy (still uniform
     bit-depth).
  3. Proposed  -- always drawn. Dynamic snap-down over the CONTINUOUS
     snap-and-refill (S1) budget range, not just the 8 searched anchors:
     paper_data.refill_n_for_budget(B) gives N at ANY budget B by snapping to
     the largest anchor <= B and converting every extra TOKEN_DIM (1024)
     bits into one more 1-bit token (2026-08-08). The 7 REFILL_POINTS_S1
     budgets were only ever used to VALIDATE this rule against the S2
     alternative on real MME accuracy -- Figure 3 makes no accuracy claim,
     so it isn't limited to those tested checkpoints; every extra 1024 bits
     of budget already changes N and therefore latency. SP-Q1/Q2/Q4 (tier 2)
     still snap over their own 8-anchor-only LUT -- refill is a Proposed-only
     mechanism, they have no equivalent continuous rule here.

Tier 2 defaults to hidden because SP-Q1/Q2/Q4 and Proposed both avoid deadline
violations via the same snap-down mechanism -- only the static tier fails to --
so tier 2 doesn't sharpen this figure's "adaptation matters" claim. Proposed's
actual advantage (better accuracy at matched budget/latency) lives in Figures
1/2, not here; adding tier 2 back mainly helps preempt a reviewer asking why
SP-Q1/Q2/Q4 vanished between figures.

    Static:       T_total(S) = T_edge + T_cloud(576) + (576*1024*B)/(S*1000)
                  (no PruneSID selection -- sends everything, so T_edge
                  alone, no clustering term)
    SP-Q:         T_total(S) = T_edge_total(N) + T_cloud(N) + budget/(S*1000), with
                  N/budget chosen per-S by dynamic snap-down over that
                  method's own 8-anchor LUT
    Proposed:     same, but N/budget chosen by dynamic snap-down over the
                  continuous snap-and-refill budget range (any multiple of
                  TOKEN_DIM bits between the lowest and highest anchor)
                  T_edge_total(N) = T_edge + T_prunesid_cluster(N)
                  (2026-08-08: real Jetson AGX Orin measurement of the
                  PruneSID selection step, previously an acknowledged gap)

N=576 chosen for the static baselines (not an arbitrary point from the
8-anchor grid) because it's already a measured anchor -- paper_data.T_CLOUD_N576
equals T_CLOUD_BY_METHOD["SP-Q1"][7], the same N=576/B=1 measurement used
elsewhere.

Common settings:
    T_edge      = 231 ms (ViT-L/14-336 forward, fixed, measured)
    T_prunesid_cluster(N) = PruneSID selection latency, variable, measured on
                            Jetson AGX Orin (added to methods that select tokens)
    T_deadline  = 400 ms

Grid: 8 anchor points, 25,600-589,824 bits (spec calls for 9; only 8 exist in
results/vqa_anchore/pipeline_vqav2/ as of 2026-07-28 -- confirmed with the user
to proceed on these 8 until the 9th point is supplied/measured).

Legend labels: SP-Q1/Q2/Q4 and Proposed reuse paper_data.LABELS (same
definition as Figures 1/2); the static variants use paper_data.FIG3_STATIC_LABELS
(Static-Q1/Q2/Q4) since they're a different experiment, not just a renamed
series -- decided with the user 2026-07-29.

Known limitation: the static baselines transmit N=576 unconditionally, so at
low bandwidth they blow well past T_deadline (that's the point being made --
a fixed strategy has no way to back off). SP-Q1/Q2/Q4 and Proposed can both
drop below T_deadline at low S by snapping down to a cheaper anchor; only
Proposed can also trade quantization precision per token.

Usage:
    python plot_paper_figures.py
"""
import matplotlib
matplotlib.use("Agg")
# Embed real (subset) TrueType fonts instead of matplotlib's PDF default
# (Type 3 bitmap-ish fonts) -- journal PDF checkers flag Type 3 fonts as
# non-embeddable/non-searchable text.
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, ScalarFormatter, NullFormatter

from paper_data import (BUDGETS, DATA, METHODS, COLORS, LABELS, FIG3_STATIC_LABELS,
                         N_BY_METHOD, T_CLOUD_BY_METHOD, T_EDGE, t_edge_total, t_prunesid_cluster, T_DEADLINE,
                         N_FIXED_BASELINE, TOKEN_DIM, T_CLOUD_N576, BIT_BY_METHOD,
                         T_LATENCY_ORI_EDGE, T_LATENCY_ORI_CLOUD, RAW_IMAGE_BITS,
                         EDGE_ONLY_COLOR, CLOUD_ONLY_COLOR, EDGE_ONLY_LABEL, CLOUD_ONLY_LABEL,
                         ANCHOR_CONFIGS, refill_n_for_budget, refill_tiers_for_budget, t_cloud_lookup,
                         t_quant_lookup, t_quant_stratified,
                         MIDPOINT_DATA, MIDPOINT_METHOD_KEY, STATIC_ACCURACY, BASELINE_FP16,
                         TIER_ID_BITS_PER_TOKEN, NORM_BITS_PER_TOKEN)

PROPOSED = "VQAv2-opt"
STATIC_METHODS = [m for m in METHODS if m != PROPOSED]  # SP-Q1/Q2/Q4 only


def t_total_dynamic(method: str, idx: int, s_mbps: float) -> float:
    n = N_BY_METHOD[method][idx]
    quant = t_quant_lookup(n, BIT_BY_METHOD[method])
    tx_bits = BUDGETS[idx] + NORM_BITS_PER_TOKEN * n
    return t_edge_total(n) + quant + T_CLOUD_BY_METHOD[method][idx] + tx_bits / (s_mbps * 1000)


def dynamic_snap_down_idx(method: str, s_mbps: float) -> int:
    """Index of the largest budget in this method's own LUT whose T_total <= T_deadline; else the smallest (cheapest) anchor."""
    totals = [t_total_dynamic(method, i, s_mbps) for i in range(len(BUDGETS))]
    feasible = [i for i, t in enumerate(totals) if t <= T_DEADLINE]
    return max(feasible) if feasible else 0


def dynamic_snap_down(method: str, s_mbps: float) -> float:
    """Largest budget in this method's own LUT whose T_total <= T_deadline; else the smallest (cheapest) anchor."""
    idx = dynamic_snap_down_idx(method, s_mbps)
    return t_total_dynamic(method, idx, s_mbps)


# Continuous snap-and-refill candidates for Proposed: every TOKEN_DIM-bit
# step between the lowest and highest searched anchor, not just the 8
# anchors or the 7 midpoints that happened to get accuracy-tested. Figure 3
# only needs latency/feasibility (no accuracy claim), so it isn't limited to
# tested checkpoints -- refill_n_for_budget() is a closed-form rule, valid
# at any budget. Precomputed once (~550 points) and reused across the whole
# s_grid sweep instead of recomputing per-S.
_REFILL_BUDGETS = np.arange(min(ANCHOR_CONFIGS), max(ANCHOR_CONFIGS) + 1, TOKEN_DIM)
_REFILL_NS = np.array([refill_n_for_budget(int(b)) for b in _REFILL_BUDGETS])
_REFILL_EDGE = np.array([t_edge_total(int(n)) for n in _REFILL_NS])
_REFILL_CLOUD = np.array([t_cloud_lookup(int(n)) for n in _REFILL_NS])
# Proposed's tokens are genuinely mixed-tier (n4/n2/n1 all potentially
# nonzero at once), so the cloud needs 2 bits/token of tier-ID metadata on
# top of the quantized payload to know how to unpack each one -- see
# TIER_ID_BITS_PER_TOKEN. This is what's actually transmitted, not the bare
# LUT budget; every caller of dynamic_snap_down_refill uses its returned
# "budget" only for T_tx, never as the Figure 1/2-style nominal label, so
# folding the overhead in here (once) keeps every downstream T_tx correct
# automatically instead of needing every caller to remember it separately.
_REFILL_TX_BITS = _REFILL_BUDGETS + (TIER_ID_BITS_PER_TOKEN + NORM_BITS_PER_TOKEN) * _REFILL_NS
# Quantization latency: needs the actual (n4, n2, n1) tier split, not just
# total N (MSECompressor.compress() cost depends on n and bits both) --
# refill_tiers_for_budget mirrors refill_n_for_budget's exact snap+refill
# arithmetic (their sums always match), just also returns the split.
_REFILL_TIERS = [refill_tiers_for_budget(int(b)) for b in _REFILL_BUDGETS]
_REFILL_QUANT = np.array([t_quant_stratified(n4, n2, n1) for n4, n2, n1 in _REFILL_TIERS])


def dynamic_snap_down_refill(s_mbps: float):
    """Largest continuous snap-and-refill budget whose T_total <= T_deadline
    (searched over _REFILL_BUDGETS, latency computed on the true
    transmitted bits including tier-ID overhead); else the cheapest
    (lowest-anchor) point. Returns (tx_bits, N, t_edge, t_cloud, t_quant) --
    tx_bits already includes the tier-ID overhead, ready to divide by
    bandwidth; t_quant is separate from t_edge so callers that want a
    breakdown (Figure 3b) can show it as its own segment."""
    totals = _REFILL_EDGE + _REFILL_QUANT + _REFILL_CLOUD + _REFILL_TX_BITS / (s_mbps * 1000)
    feasible = np.nonzero(totals <= T_DEADLINE)[0]
    i = int(feasible.max()) if len(feasible) else 0
    return (int(_REFILL_TX_BITS[i]), int(_REFILL_NS[i]), float(_REFILL_EDGE[i]),
            float(_REFILL_CLOUD[i]), float(_REFILL_QUANT[i]))


# Continuous version of Tier 2 (PSID-Q1/Q2/Q4) for Figure 3's latency panel
# ONLY -- picks the largest N in [1, 576] (not restricted to the 8 measured
# anchors) whose T_total fits the deadline at that method's fixed bit-depth.
# Figure 3 makes no accuracy claim, so there's no reason to snap to
# specifically-measured checkpoints here the way dynamic_snap_down does --
# that discrete 8-point restriction was what produced the sawtooth jumps in
# the latency trend (each switch changes N by however much separates two
# anchors, tens to hundreds of tokens, instead of the ~1-token granularity
# Proposed's continuous refill already got). dynamic_snap_down/_idx are
# left untouched -- Figure 3b's bar chart and anything else using them
# still wants the actually-measured-accuracy anchor points. 2026-08-19.
_CONT_NS = np.arange(1, N_FIXED_BASELINE + 1)
_CONT_EDGE = np.array([t_edge_total(int(n)) for n in _CONT_NS])
_CONT_CLOUD = np.array([t_cloud_lookup(int(n)) for n in _CONT_NS])
# One quant-latency array per bit-depth (PSID-Qn is single-bit-depth, all N
# tokens at bits=BIT_BY_METHOD[method]), precomputed once.
_CONT_QUANT_BY_BITS = {bits: np.array([t_quant_lookup(int(n), bits) for n in _CONT_NS])
                        for bits in (1, 2, 4)}


def dynamic_snap_down_continuous(method: str, s_mbps: float) -> float:
    bits = BIT_BY_METHOD[method]
    tx = (_CONT_NS * TOKEN_DIM * bits + NORM_BITS_PER_TOKEN * _CONT_NS) / (s_mbps * 1000)
    totals = _CONT_EDGE + _CONT_QUANT_BY_BITS[bits] + _CONT_CLOUD + tx
    feasible = np.nonzero(totals <= T_DEADLINE)[0]
    i = int(feasible.max()) if len(feasible) else 0
    return float(totals[i])


def static_total(method: str, s_mbps: float) -> float:
    """Static-Q1/Q2/Q4: always send all N_FIXED_BASELINE tokens at a fixed bit-depth, regardless of S or T_deadline."""
    bits = BIT_BY_METHOD[method]
    budget = N_FIXED_BASELINE * TOKEN_DIM * bits
    quant = t_quant_lookup(N_FIXED_BASELINE, bits)
    tx_bits = budget + NORM_BITS_PER_TOKEN * N_FIXED_BASELINE
    return T_EDGE + quant + T_CLOUD_N576 + tx_bits / (s_mbps * 1000)


def edge_only_total(s_mbps: float) -> float:
    """Whole original pipeline (ViT+projector+LLM, full precision) runs on-device
    (Jetson AGX Orin) -- no network round-trip, so this is independent of S."""
    return T_LATENCY_ORI_EDGE


def cloud_only_total(s_mbps: float) -> float:
    """No on-device processing at all: transmit the raw 336x336x3, 8-bit/channel
    image uncompressed, then the cloud runs the whole original pipeline."""
    return RAW_IMAGE_BITS / (s_mbps * 1000) + T_LATENCY_ORI_CLOUD


# Proposed method's T_total breakdown, edge (device) -> cloud (server compute)
# -> transmission -- a grey-to-accent ramp so "device cost" reads as neutral
# hardware overhead and "network cost" reads as the accent-colored, S-dependent
# piece that adaptation is actually fighting.
BREAKDOWN_COLORS = {"edge": "#9aa0ac", "cloud": "#f3a19f", "tx": COLORS[PROPOSED],
                     "vit": "#6b7280", "prunesid": "#c7cbd1", "quant": "#e8c34a"}
BREAKDOWN_LABELS = {"edge": "T_edge (on-device)", "cloud": "T_cloud (server compute)", "tx": "T_tx (transmission)",
                     "vit": "T_edge: ViT forward", "prunesid": "T_edge: PruneSID select", "quant": "T_edge: Quantize"}


# Plain filled circles for every Tier-2 (PSID-Q1/Q2/Q4) line -- differentiated
# marker shapes (o/s/^) were tried and dropped per user request; color alone
# (COLORS[method]) already distinguishes the three lines.
MARKERS = {"SP-Q1": "o", "SP-Q2": "o", "SP-Q4": "o"}


def plot_figure3(out_path="figure3_bandwidth_vs_latency.pdf", s_min=0.1, s_max=20,
                  n_points=500, show_spq=True, breakdown=False, ax=None, s_grid=None):
    """show_spq=False: only Static-Q1/Q2/Q4 + Proposed are drawn -- SP-Q1/Q2/Q4's
    dynamic snap-down tier adds no discriminating information over Proposed for
    the "does adaptation matter" claim this figure makes (both avoid deadline
    violations; only the static tier fails to), so it's opt-in clutter.

    breakdown=False (default, 2026-08-18): Proposed's T_total is a plain
    line. breakdown=True instead draws it as a stacked area (T_edge +
    T_cloud + T_tx), showing WHERE the latency budget goes as bandwidth
    changes -- at low S transmission dominates (the accent band swells), at
    high S it's mostly fixed edge+cloud overhead (grey/pink bands take
    over). Turned off by default once the accuracy sub-panel
    (plot_figure3_with_accuracy) made "does Proposed hit the highest
    accuracy at the user's deadline" the figure's actual point -- the
    breakdown stack (3 extra fill colors + legend entries) competed with
    that story rather than supporting it.

    ax=None (default): creates its own standalone figure and saves out_path
    (unchanged old behavior). Pass an existing Axes to draw onto it instead
    (no figure creation, no save, no xlabel -- caller owns those) -- used by
    plot_figure3_with_accuracy() to stack this as the top panel of a 2-row figure.
    """
    if s_grid is None:
        s_grid = np.geomspace(s_min, s_max, n_points)
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 5))

    # Tier 1: static, no adaptation at all -- thin dashed
    for method in STATIC_METHODS:
        ys = [static_total(method, s) for s in s_grid]
        ax.plot(s_grid, ys, label=FIG3_STATIC_LABELS[method], color=COLORS[method],
                 linewidth=1.8, linestyle='--', alpha=0.6, zorder=2,
                 marker='o', markevery=45, markersize=5)

    # Tier 0: single-device baselines -- no split, no pruning/quant, no
    # bandwidth-adaptation at all. Edge-only ignores S entirely (nothing is
    # transmitted); Cloud-only sends the raw uncompressed image.
    ax.plot(s_grid, [edge_only_total(s) for s in s_grid], label=EDGE_ONLY_LABEL,
             color=EDGE_ONLY_COLOR, linewidth=1.6, linestyle='-', alpha=0.75, zorder=1)
    ax.plot(s_grid, [cloud_only_total(s) for s in s_grid], label=CLOUD_ONLY_LABEL,
             color=CLOUD_ONLY_COLOR, linewidth=1.6, linestyle=(0, (3, 1, 1, 1)), alpha=0.75, zorder=1,
             marker='o', markevery=45, markersize=5)

    # Tier 2 (opt-in): SP-Q1/Q2/Q4, dynamic snap-down but uniform bit-depth --
    # solid with sparse markers so it's distinguishable from Tier 1's dashed
    # static lines and from Proposed's thicker line where they run close
    # together. Continuous over N (not the 8 measured anchors, see
    # dynamic_snap_down_continuous) since this figure makes no accuracy
    # claim -- just the latency trend.
    if show_spq:
        for method in STATIC_METHODS:
            ys = [dynamic_snap_down_continuous(method, s) for s in s_grid]
            ax.plot(s_grid, ys, label=LABELS[method], color=COLORS[method],
                     linewidth=2.2, linestyle='-', zorder=3,
                     marker=MARKERS[method], markevery=45, markersize=6)

    # Tier 3: proposed, dynamic snap-down over the CONTINUOUS snap-and-refill
    # budget range (every TOKEN_DIM bits, not just the 8 anchors)
    entries = [dynamic_snap_down_refill(s) for s in s_grid]
    if breakdown:
        edge_ys = [e[2] for e in entries]
        quant_ys = [e[4] for e in entries]
        cloud_ys = [e[3] for e in entries]
        tx_ys = [e[0] / (s * 1000) for e, s in zip(entries, s_grid)]
        ax.stackplot(s_grid, edge_ys, quant_ys, cloud_ys, tx_ys,
                     colors=[BREAKDOWN_COLORS[k] for k in ("edge", "quant", "cloud", "tx")],
                     labels=[BREAKDOWN_LABELS[k] for k in ("edge", "quant", "cloud", "tx")],
                     alpha=0.65, zorder=3)
        # crisp outline on top of the stack, same values as the stack's total --
        # labeled so "Proposed" still gets its own legend entry like
        # every other method, on top of (not instead of) the breakdown labels
        ys = [e + q + c + t for e, q, c, t in zip(edge_ys, quant_ys, cloud_ys, tx_ys)]
        ax.plot(s_grid, ys, label=LABELS[PROPOSED], color=COLORS[PROPOSED],
                 linewidth=2.2, linestyle='-', zorder=5)
    else:
        ys = [e[2] + e[4] + e[3] + e[0] / (s * 1000) for e, s in zip(entries, s_grid)]
        ax.plot(s_grid, ys, label=LABELS[PROPOSED], color=COLORS[PROPOSED],
                 linewidth=3.2, linestyle='-', zorder=5,
                 marker='o', markevery=45, markersize=7)

    ax.axhline(y=T_DEADLINE, color="black", linewidth=1.5, linestyle=":",
               label=rf"$T_{{\mathrm{{deadline}}}}$ = {T_DEADLINE} ms")

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(s_min, s_max)
    ax.set_ylabel(r'$T_{\mathrm{total}}$ (ms, log scale)', fontsize=LABEL_FS + 3)
    ax.tick_params(axis='both', labelsize=TICK_FS)
    # Plain-number ticks (not 10^n) plus an explicit T_deadline tick, bolded
    # so a reader can find "does this line clear 350ms" at a glance instead
    # of interpolating between log-spaced gridlines.
    _y_ticks = sorted({200, T_DEADLINE, 500, 1000, 2000, 5000, 10000, 20000})
    ax.yaxis.set_major_locator(FixedLocator(_y_ticks))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    for tick_val, tick_label in zip(_y_ticks, ax.get_yticklabels()):
        if tick_val == T_DEADLINE:
            tick_label.set_fontweight("bold")
    ax.legend(fontsize=LEGEND_FS - 1, loc='upper right', ncol=1,
              handlelength=2.6, labelspacing=0.35, borderpad=0.4, columnspacing=0.8)
    ax.grid(True, which='major', alpha=0.3)
    if standalone:
        ax.set_xlabel(r'$C_{\mathrm{current}}$ (Mbps, log scale)', fontsize=LABEL_FS + 3)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        print(f"saved {out_path}")


# ============================================================
# Accuracy sub-panel: what accuracy does the deadline-feasible budget at
# each S actually buy? Figure 3's latency panel only ever answers "does
# adaptation avoid missing the deadline" -- this answers "and how good is
# the thing you get to send at that S", which neither this figure nor
# Figure 2 (fixed 8 anchors only, not the S-driven continuous selection)
# currently shows.
# ============================================================
def _measured_checkpoints(method: str, benchmark: str):
    """(budget, N, t_cloud, score, n4, n2, n1) for every real measurement of
    `method` on `benchmark`: the 8 anchors, plus the 7 snap-and-refill
    midpoints where that method's own MIDPOINT_DATA row exists. Sorted by
    budget ascending. n4/n2/n1 needed for T_quant (MSECompressor.compress()
    cost depends on n and bits, not just total N)."""
    points = [(BUDGETS[i], N_BY_METHOD[method][i], T_CLOUD_BY_METHOD[method][i], DATA[benchmark][method][i],
               ANCHOR_CONFIGS[BUDGETS[i]]["n4"] if method == PROPOSED else (N_BY_METHOD[method][i] if BIT_BY_METHOD.get(method) == 4 else 0),
               ANCHOR_CONFIGS[BUDGETS[i]]["n2"] if method == PROPOSED else (N_BY_METHOD[method][i] if BIT_BY_METHOD.get(method) == 2 else 0),
               ANCHOR_CONFIGS[BUDGETS[i]]["n1"] if method == PROPOSED else (N_BY_METHOD[method][i] if BIT_BY_METHOD.get(method) == 1 else 0))
              for i in range(len(BUDGETS))]
    strategy = MIDPOINT_METHOD_KEY[method]
    for entry in MIDPOINT_DATA.get(benchmark, {}).values():
        row = entry.get(strategy)
        if row is not None:
            points.append((entry["actual_budget"], row["N"], row["t_cloud"], row["score"],
                            row["n4"], row["n2"], row["n1"]))
    points.sort(key=lambda p: p[0])
    return points


def _accuracy_snap_down(method: str, benchmark: str, s_mbps: float) -> float:
    """Accuracy of the largest-budget measured checkpoint whose T_total <=
    T_deadline; else the cheapest one. Same snap-down rule as
    dynamic_snap_down_idx (max feasible index, else index 0), just over the
    anchor+midpoint checkpoint set instead of the 8-anchor-only LUT.

    Every token pays the per-token fp16 L2-norm overhead (NORM_BITS_PER_TOKEN,
    all methods route through MSECompressor). Only Proposed's checkpoints are
    genuinely mixed-tier (n4/n2/n1 all potentially nonzero) -- SP-Q1/Q2/Q4
    always use one uniform bit-width, so the extra tier-ID overhead
    (TIER_ID_BITS_PER_TOKEN) only applies when method == PROPOSED. T_quant
    uses the actual tier split either way."""
    points = _measured_checkpoints(method, benchmark)
    overhead = NORM_BITS_PER_TOKEN + (TIER_ID_BITS_PER_TOKEN if method == PROPOSED else 0)
    totals = [t_edge_total(n) + t_quant_stratified(n4, n2, n1) + t_cloud + (budget + overhead * n) / (s_mbps * 1000)
              for budget, n, t_cloud, _, n4, n2, n1 in points]
    feasible = [i for i, t in enumerate(totals) if t <= T_DEADLINE]
    idx = max(feasible) if feasible else 0
    return points[idx][3]


# Static-Q1/Q2/Q4: fixed N=576 single-bit-tier config, never adapts, never
# backs off to a cheaper config at low S -- so it has exactly one accuracy
# value (STATIC_ACCURACY's mean), shown only where its own (fixed) T_total
# meets the deadline; NaN elsewhere so the line simply doesn't draw there,
# same as an infeasible/unmet region rather than implying a fallback score
# that was never measured.
def _static_accuracy(method: str, benchmark: str, s_mbps: float) -> float:
    if static_total(method, s_mbps) <= T_DEADLINE:
        return STATIC_ACCURACY[benchmark][method]["mean"]
    return float("nan")


# Cloud-only: sends the raw uncompressed image, then the ORIGINAL model
# (no PruneSID, no quant) runs entirely in the cloud -- so its accuracy is
# exactly BASELINE_FP16, the same number Cloud-only's latency line in the
# top panel is implicitly promising. Same "flat value, shown only where
# feasible" treatment as Static. Edge-only is NOT added here: its latency
# (T_LATENCY_ORI_EDGE ~= 2008ms) exceeds T_DEADLINE regardless of S, so it
# would never once appear across the whole 0.1-20 Mbps range -- not a bug,
# just genuinely never a deadline-feasible option in this setup.
def _cloud_only_accuracy(benchmark: str, s_mbps: float) -> float:
    fp16 = BASELINE_FP16.get(benchmark)
    if fp16 is not None and cloud_only_total(s_mbps) <= T_DEADLINE:
        return fp16
    return float("nan")


def plot_figure3_accuracy_panel(ax, benchmark="MME", s_grid=None, s_min=0.1, s_max=20,
                                 n_points=500, compare_methods=("SP-Q1", "SP-Q2", "SP-Q4"),
                                 show_static=True, show_cloud_only=True):
    if s_grid is None:
        s_grid = np.geomspace(s_min, s_max, n_points)
    if show_static:
        for method in compare_methods:
            ys = [_static_accuracy(method, benchmark, s) for s in s_grid]
            ax.plot(s_grid, ys, color=COLORS[method], linestyle=':', linewidth=1.3,
                     alpha=0.6, label=FIG3_STATIC_LABELS[method], zorder=2)
    if show_cloud_only:
        ys = [_cloud_only_accuracy(benchmark, s) for s in s_grid]
        if not all(y != y for y in ys):  # only add a legend entry if it's ever visible
            ax.plot(s_grid, ys, color=CLOUD_ONLY_COLOR, linestyle=(0, (3, 1, 1, 1)),
                     linewidth=1.3, alpha=0.75, label=CLOUD_ONLY_LABEL, zorder=2)
    for method in [*compare_methods, PROPOSED]:
        ys = [_accuracy_snap_down(method, benchmark, s) for s in s_grid]
        is_proposed = method == PROPOSED
        ax.plot(s_grid, ys, color=COLORS[method], drawstyle='steps-post',
                 linewidth=1.8 if is_proposed else 1.4,
                 linestyle='-' if is_proposed else '--',
                 label=LABELS[method], zorder=5 if is_proposed else 3)
    ax.set_xscale('log')
    ax.set_xlim(s_grid[0], s_grid[-1])
    ax.set_ylabel(f'{benchmark} Score\n(at deadline-feasible budget)', fontsize=LABEL_FS)
    ax.legend(fontsize=LEGEND_FS, loc='lower right', ncol=2)
    ax.grid(True, which='major', alpha=0.3)


def plot_figure3_with_accuracy(out_path="figure3_bandwidth_vs_latency.pdf", s_min=0.1, s_max=20,
                                n_points=500, show_spq=True, breakdown=False,
                                accuracy_benchmark="MME", accuracy_compare=("SP-Q1", "SP-Q2", "SP-Q4"),
                                accuracy_show_static=True, accuracy_show_cloud_only=True):
    """Two-row version of Figure 3: same latency panel on top, plus a new
    accuracy-vs-bandwidth panel underneath sharing the same x-axis, so a
    reader can read off "at this S, latency looks like X and the accuracy
    you actually get is Y" directly instead of cross-referencing Figure 2's
    fixed-anchor accuracy-vs-latency plot."""
    s_grid = np.geomspace(s_min, s_max, n_points)
    fig, (ax_lat, ax_acc) = plt.subplots(
        2, 1, figsize=(6.5, 7.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.4], "hspace": 0.08})

    plot_figure3(show_spq=show_spq, breakdown=breakdown, ax=ax_lat, s_grid=s_grid)
    ax_lat.set_title(f'Figure 3: Bandwidth vs End-to-End Latency (+ {accuracy_benchmark} Accuracy)')

    plot_figure3_accuracy_panel(ax_acc, benchmark=accuracy_benchmark, s_grid=s_grid,
                                 compare_methods=accuracy_compare, show_static=accuracy_show_static,
                                 show_cloud_only=accuracy_show_cloud_only)
    ax_acc.set_xlabel('Bandwidth S (Mbps, log scale)')

    # stackplot's PolyCollection on ax_lat isn't tight_layout-compatible
    # (spurious warning, no visual issue) -- bbox_inches='tight' on save
    # achieves the same trim without touching the shared-x subplot spacing.
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"saved {out_path}")


# Font sizes matching Figures 1/2 (plot_vqav2_figures.py) so all paper
# figures read consistently at the same print size. 2026-08-19.
TITLE_FS = 15
LABEL_FS = 11
TICK_FS = 15
LEGEND_FS = 11


def _method_components(s_mbps: float):
    """(label, [vit_ms, prunesid_ms, quant_ms, cloud_ms, tx_ms]) for every
    method shown in the bar chart, at one fixed bandwidth -- same formulas
    as plot_figure3, just evaluated at a single S instead of swept over a
    grid, and with T_edge split into its two sub-components (ViT forward,
    always T_EDGE; PruneSID cluster+NMS selection, 0 for methods that don't
    run PruneSID at all -- Edge/Cloud inference) plus quantization as its
    own segment (0 for Edge/Cloud inference, the only methods that don't
    quantize at all). Edge inference has no cloud/tx component (nothing
    leaves the device); Cloud inference has no edge component (nothing runs
    on the device). Order: single-device baselines first (Edge/Cloud
    inference), then non-adaptive Static-Q1/2/4, then bandwidth-adaptive
    PSID-Q1/2/4, then Proposed -- least-to-most adaptive, 2026-08-19."""
    rows = []
    rows.append((EDGE_ONLY_LABEL, [T_LATENCY_ORI_EDGE, 0.0, 0.0, 0.0, 0.0]))
    rows.append((CLOUD_ONLY_LABEL, [0.0, 0.0, 0.0, T_LATENCY_ORI_CLOUD, RAW_IMAGE_BITS / (s_mbps * 1000)]))
    for method in STATIC_METHODS:
        bits = BIT_BY_METHOD[method]
        budget = N_FIXED_BASELINE * TOKEN_DIM * bits
        quant = t_quant_lookup(N_FIXED_BASELINE, bits)
        tx_bits = budget + NORM_BITS_PER_TOKEN * N_FIXED_BASELINE
        rows.append((FIG3_STATIC_LABELS[method],
                     [T_EDGE, 0.0, quant, T_CLOUD_N576, tx_bits / (s_mbps * 1000)]))
    for method in STATIC_METHODS:
        idx = dynamic_snap_down_idx(method, s_mbps)
        n = N_BY_METHOD[method][idx]
        quant = t_quant_lookup(n, BIT_BY_METHOD[method])
        tx_bits = BUDGETS[idx] + NORM_BITS_PER_TOKEN * n
        rows.append((LABELS[method],
                     [T_EDGE, t_prunesid_cluster(n), quant, T_CLOUD_BY_METHOD[method][idx], tx_bits / (s_mbps * 1000)]))
    budget, n, edge_ms, cloud_ms, quant_ms = dynamic_snap_down_refill(s_mbps)
    rows.append((LABELS[PROPOSED], [T_EDGE, edge_ms - T_EDGE, quant_ms, cloud_ms, budget / (s_mbps * 1000)]))
    return rows


def plot_figure3_bar(out_path="figure3b_latency_breakdown_bar.pdf", s_mbps=1.0):
    """Stacked bar version of the Figure 3 breakdown: same T_edge/T_cloud/T_tx
    decomposition, but at one fixed bandwidth so every method's composition
    can be read off and compared directly, side by side, rather than having
    to eyeball a single vertical slice through the continuous plot. T_edge
    is split into ViT forward + PruneSID selection, plus quantization as its
    own segment (5-segment stack total), 2026-08-19."""
    rows = _method_components(s_mbps)
    labels = [r[0] for r in rows]
    vit = np.array([r[1][0] for r in rows])
    prunesid = np.array([r[1][1] for r in rows])
    quant = np.array([r[1][2] for r in rows])
    cloud = np.array([r[1][3] for r in rows])
    tx = np.array([r[1][4] for r in rows])
    totals = vit + prunesid + quant + cloud + tx

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(labels))
    ax.bar(x, vit, color=BREAKDOWN_COLORS["vit"], label=BREAKDOWN_LABELS["vit"], zorder=3)
    ax.bar(x, prunesid, bottom=vit, color=BREAKDOWN_COLORS["prunesid"], label=BREAKDOWN_LABELS["prunesid"], zorder=3)
    ax.bar(x, quant, bottom=vit + prunesid, color=BREAKDOWN_COLORS["quant"], label=BREAKDOWN_LABELS["quant"], zorder=3)
    ax.bar(x, cloud, bottom=vit + prunesid + quant, color=BREAKDOWN_COLORS["cloud"], label=BREAKDOWN_LABELS["cloud"], zorder=3)
    ax.bar(x, tx, bottom=vit + prunesid + quant + cloud, color=BREAKDOWN_COLORS["tx"], label=BREAKDOWN_LABELS["tx"], zorder=3)

    for xi, total in zip(x, totals):
        # Angled instead of horizontal -- PSID-Q1/Q2/Q4 and Proposed all land
        # within ~50ms of each other near the deadline line, so horizontal
        # labels there collide; a 60-degree angle keeps them legible without
        # needing per-bar vertical offset bookkeeping.
        ax.text(xi, total + max(totals) * 0.015, f"{total:,.0f} ms", ha="left", va="bottom",
                 rotation=60, fontsize=TICK_FS - 3)

    ax.axhline(y=T_DEADLINE, color="black", linewidth=1.3, linestyle=":",
               label=rf"$T_{{\mathrm{{deadline}}}}$ = {T_DEADLINE} ms", zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=TICK_FS)
    ax.set_ylabel(r'$T_{\mathrm{total}}$ (ms)', fontsize=TICK_FS)
    ax.set_ylim(0, max(totals) * 1.18)  # headroom for the angled value labels
    # Explicit T_deadline tick, bolded, same "find 350ms at a glance"
    # treatment as Figure 3's log-scale y-axis.
    _y_ticks = sorted(set(ax.get_yticks().tolist()) | {T_DEADLINE})
    ax.set_yticks(_y_ticks)
    ax.tick_params(axis='y', labelsize=TICK_FS)
    for tick_val, tick_label in zip(_y_ticks, ax.get_yticklabels()):
        if tick_val == T_DEADLINE:
            tick_label.set_fontweight("bold")
    ax.legend(fontsize=LEGEND_FS + 1)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"saved {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hide-spq", action="store_true",
                         help="Hide the SP-Q1/Q2/Q4 dynamic snap-down tier (default: shown)")
    parser.add_argument("--breakdown", action="store_true",
                         help="Draw Proposed's latency as an edge/cloud/tx stacked breakdown "
                              "instead of a plain line (default: plain line -- the stack added "
                              "clutter once the accuracy sub-panel made the deadline-vs-accuracy "
                              "story the point of this figure, 2026-08-18)")
    parser.add_argument("--bar-s", type=float, default=1.0,
                         help="Bandwidth (Mbps) for the Figure 3b bar-chart breakdown (default: 1.0)")
    parser.add_argument("--with-accuracy-panel", action="store_true",
                         help="Draw the 2-row latency+accuracy version instead of the plain "
                              "single-panel Figure 3 (default: single panel, latency only -- "
                              "the accuracy sub-panel's step function inherits real 3-seed "
                              "measurement noise from the underlying checkpoints, sawtooth-shaped "
                              "rather than monotonic, which read as confusing/misleading in the "
                              "main figure; removed as the default 2026-08-19, kept available here)")
    parser.add_argument("--accuracy-benchmark", default="MME",
                         help="Benchmark for the accuracy sub-panel (default: MME)")
    args = parser.parse_args()
    if args.with_accuracy_panel:
        plot_figure3_with_accuracy(show_spq=not args.hide_spq, breakdown=args.breakdown,
                                    accuracy_benchmark=args.accuracy_benchmark)
    else:
        plot_figure3(show_spq=not args.hide_spq, breakdown=args.breakdown)
    plot_figure3_bar(s_mbps=args.bar_s)
