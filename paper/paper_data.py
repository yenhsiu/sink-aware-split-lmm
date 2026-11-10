"""
Shared data for the paper figures (plot_vqav2_figures.py / plot_paper_figures.py /
plot_figure4.py) -- single source of truth so the three scripts can't drift apart.

Grid: 8 anchor points, 25,600-589,824 bits (spec calls for 9; only 8 exist in
results/vqa_anchore/pipeline_vqav2/ as of 2026-07-28 -- confirmed with the user to
proceed on these 8 until the 9th point is supplied/measured).

Naming: "All B=1/2/4" renamed to SP-Q1/SP-Q2/SP-Q4 per user request (2026-07-28).
"VQAv2-opt" (the proposed method) is unchanged.

2026-08-07 UPDATE: token-selection method switched from PruMerge to PruneSID
(PCA-cluster + intra-cluster NMS diversity selection, plus sink-aware bit
tiering) in the new this repository pipeline -- vanilla LLaVA-1.5-7B,
training-free (no LoRA). All VQAv2-opt and SP-Q1/Q2/Q4 numbers below re-measured
on that pipeline (search on VQAv2 random-5000, validated on full MME/POPE).
SP-Q1/Q2/Q4's N_BY_METHOD entries are unchanged from the old PruMerge run --
the All-B1/B2/B4 anchor points are pure budget arithmetic (T, T//2, T//4),
independent of which token-selection algorithm produced them -- confirmed by
recomputation. BASELINE_FP16 also re-measured on this pipeline (true
--baseline path: no PruneSID, no quant, N=576, natural token order).
2026-08-08 UPDATE: real Jetson AGX Orin measurement of the PruneSID
PCA-cluster+NMS selection step landed (latency_collec_scripts/
results_edge_prunesid_AGX_Orin.csv, N=1..576 step 8) -- this was previously
an acknowledged gap (only an A100 stand-in existed, excluded from every
T_edge/T_total formula per an explicit decision with the user). It is now
folded into all latency calculations via t_edge_total(n) below, for every
series that actually runs PruneSID selection (SP-Q1/Q2/Q4, Proposed/
VQAv2-opt, and the snap-and-refill midpoints). Static-Q1/Q2/Q4 and the
Edge-only/Cloud-only single-device baselines in Figure 3 do NOT get this
addition -- they don't run PruneSID at all (Static sends everything
unselected; Edge/Cloud-only run the whole original, unpruned pipeline).

2026-08-17 UPDATE: SP-Q2/SP-Q4 ("All-B2"/"All-B4") redefined as *pure*
single-bit-width baselines. Previously, whenever a budget's T = S // 1024
didn't divide evenly by 2 (SP-Q2) or 4 (SP-Q4), get_anchor_points() packed
the leftover 1-3 tokens' worth of budget into an extra 1-bit token instead
of dropping it -- meaning these "uniform-bit-width" baselines were quietly
mixing in a second bit-width at 2/8 (SP-Q2) and 4/8 (SP-Q4) anchor budgets,
plus 3 and 6 of the 7 snap-and-refill midpoints respectively. Since the
budget-exact-match constraint (4n4+2n2+n1 = T) is by design only meant to
bind the search space Proposed draws from, not the definition of a
"single quantization width" baseline, SP-Q2/SP-Q4 now drop that leftover
token instead (wasting up to 3*1024 bits of the nominal budget) at every
affected point. All 15 affected (config, benchmark) triples were re-run
3-seed (42/43/44), 180 runs total, 2026-08-16/17 -- see results/validate_
{mme,pope,textvqa,mmbench}_n4*_n2*_n1*{,_pcaseed43,_pcaseed44}_*.txt where
n1 no longer has the old leftover value. N_BY_METHOD/T_CLOUD_BY_METHOD
updated to match (N drops by 1-3 at each affected budget).
Net effect on the paper's conclusions: NONE -- recomputing win/tie counts
under the new baseline definition gives byte-for-byte the same result as
before at both the main-table anchors (MME 6/8, POPE 5/8, TextVQA 6/8,
MMBench 3/8) and the snap-and-refill diverging midpoints (MME 4/4, POPE
1/4, TextVQA 3/4, MMBench 1/4). At every affected point either SP-Q1 was
already the deciding baseline (unaffected by this redefinition, since T
always divides evenly by 1), or the pure-vs-leftover delta was too small
to flip the win/loss margin. This was checked explicitly, not assumed.
"""

import csv as _csv
from pathlib import Path as _Path

BUDGETS = [25600, 41000, 66000, 106000, 170000, 273000, 439000, 589824]

METHODS = ["SP-Q1", "SP-Q2", "SP-Q4", "VQAv2-opt"]

# ============================================================
# Per-anchor benchmark scores: mean over 3 PCA seeds (42/43/44) of the
# PruneSID selection, full validation sets, measured with this repository
# (python reproduce.py main). Per-seed values and sd: DATA_SEEDED below.
# Verify against your own runs with: python compare_results.py main
# ============================================================
DATA = {
    "MME": {
        "SP-Q1": [1512.92, 1605.73, 1631.53, 1701.31, 1719.62, 1752.91, 1720.64, 1734.24],
        "SP-Q2": [1413.68, 1516.05, 1622.38, 1685.04, 1707.87, 1746.44, 1743.35, 1768.23],
        "SP-Q4": [1182.33, 1337.63, 1467.51, 1565.66, 1634.41, 1702.57, 1716.36, 1705.51],
        "VQAv2-opt": [1535.24, 1605.73, 1631.53, 1697.05, 1722.30, 1747.56, 1787.22, 1794.73],
    },
    "POPE": {
        "SP-Q1": [0.7566, 0.8064, 0.8333, 0.8410, 0.8505, 0.8547, 0.8543, 0.8550],
        "SP-Q2": [0.6106, 0.7221, 0.7993, 0.8338, 0.8485, 0.8414, 0.8594, 0.8579],
        "SP-Q4": [0.4569, 0.5432, 0.6795, 0.7728, 0.8181, 0.8456, 0.8416, 0.8450],
        "VQAv2-opt": [0.7367, 0.8064, 0.8333, 0.8420, 0.8470, 0.8557, 0.8591, 0.8587],
    },
    "TextVQA": {
        "SP-Q1": [50.20, 51.53, 52.17, 52.72, 53.41, 53.81, 53.72, 53.74],
        "SP-Q2": [47.91, 50.35, 51.95, 52.75, 54.02, 54.29, 54.70, 54.83],
        "SP-Q4": [44.22, 46.82, 49.49, 51.65, 52.68, 53.68, 54.31, 54.53],
        "VQAv2-opt": [50.31, 51.53, 52.17, 53.43, 54.08, 54.41, 54.74, 54.93],
    },
    "MMBench": {
        "SP-Q1": [66.03, 67.70, 69.20, 71.05, 71.32, 71.97, 71.95, 71.43],
        "SP-Q2": [61.24, 66.10, 68.62, 70.19, 71.66, 72.17, 72.42, 72.61],
        "SP-Q4": [53.21, 59.22, 64.82, 67.96, 69.46, 70.77, 71.97, 72.22],
        "VQAv2-opt": [65.57, 67.70, 69.20, 70.99, 71.42, 72.04, 72.34, 72.64],
    },
}

# Per-seed scores behind DATA (seeds 42/43/44), mean and sample sd.
DATA_SEEDED = {
    "MME": {
        "SP-Q1": [
            {"seeds": [1509.42, 1472.89, 1556.46], "mean": 1512.92, "sd": 41.89},
            {"seeds": [1599.03, 1608.04, 1610.14], "mean": 1605.73, "sd": 5.90},
            {"seeds": [1602.36, 1624.28, 1667.95], "mean": 1631.53, "sd": 33.39},
            {"seeds": [1718.88, 1692.39, 1692.66], "mean": 1701.31, "sd": 15.22},
            {"seeds": [1714.54, 1725.34, 1718.98], "mean": 1719.62, "sd": 5.43},
            {"seeds": [1759.74, 1748.12, 1750.88], "mean": 1752.91, "sd": 6.07},
            {"seeds": [1718.53, 1720.23, 1723.16], "mean": 1720.64, "sd": 2.34},
            {"seeds": [1730.37, 1739.49, 1732.87], "mean": 1734.24, "sd": 4.71},
        ],
        "SP-Q2": [
            {"seeds": [1414.34, 1437.60, 1389.11], "mean": 1413.68, "sd": 24.25},
            {"seeds": [1507.64, 1518.65, 1521.85], "mean": 1516.05, "sd": 7.45},
            {"seeds": [1633.07, 1598.78, 1635.29], "mean": 1622.38, "sd": 20.47},
            {"seeds": [1686.79, 1689.58, 1678.74], "mean": 1685.04, "sd": 5.63},
            {"seeds": [1704.28, 1680.26, 1739.07], "mean": 1707.87, "sd": 29.57},
            {"seeds": [1769.27, 1749.50, 1720.55], "mean": 1746.44, "sd": 24.50},
            {"seeds": [1739.14, 1735.27, 1755.65], "mean": 1743.35, "sd": 10.83},
            {"seeds": [1771.18, 1771.69, 1761.81], "mean": 1768.23, "sd": 5.56},
        ],
        "SP-Q4": [
            {"seeds": [1182.33, 1182.33, 1182.33], "mean": 1182.33, "sd": 0.00},
            {"seeds": [1335.23, 1340.05, 1337.61], "mean": 1337.63, "sd": 2.41},
            {"seeds": [1470.05, 1470.11, 1462.37], "mean": 1467.51, "sd": 4.45},
            {"seeds": [1553.74, 1571.04, 1572.19], "mean": 1565.66, "sd": 10.33},
            {"seeds": [1634.84, 1626.08, 1642.31], "mean": 1634.41, "sd": 8.12},
            {"seeds": [1685.33, 1689.79, 1732.59], "mean": 1702.57, "sd": 26.09},
            {"seeds": [1699.60, 1746.84, 1702.65], "mean": 1716.36, "sd": 26.44},
            {"seeds": [1728.82, 1709.55, 1678.16], "mean": 1705.51, "sd": 25.57},
        ],
        "VQAv2-opt": [
            {"seeds": [1555.87, 1552.71, 1497.15], "mean": 1535.24, "sd": 33.03},
            {"seeds": [1599.03, 1608.04, 1610.14], "mean": 1605.73, "sd": 5.90},
            {"seeds": [1602.36, 1624.28, 1667.95], "mean": 1631.53, "sd": 33.39},
            {"seeds": [1666.19, 1707.10, 1717.84], "mean": 1697.05, "sd": 27.25},
            {"seeds": [1713.90, 1715.61, 1737.39], "mean": 1722.30, "sd": 13.09},
            {"seeds": [1748.24, 1740.66, 1753.78], "mean": 1747.56, "sd": 6.59},
            {"seeds": [1784.96, 1790.61, 1786.11], "mean": 1787.22, "sd": 2.98},
            {"seeds": [1805.60, 1798.21, 1780.39], "mean": 1794.73, "sd": 12.96},
        ],
    },
    "POPE": {
        "SP-Q1": [
            {"seeds": [0.7564, 0.7526, 0.7607], "mean": 0.7566, "sd": 0.0040},
            {"seeds": [0.8041, 0.8077, 0.8073], "mean": 0.8064, "sd": 0.0020},
            {"seeds": [0.8366, 0.8330, 0.8303], "mean": 0.8333, "sd": 0.0032},
            {"seeds": [0.8400, 0.8419, 0.8410], "mean": 0.8410, "sd": 0.0010},
            {"seeds": [0.8500, 0.8499, 0.8514], "mean": 0.8505, "sd": 0.0008},
            {"seeds": [0.8553, 0.8552, 0.8536], "mean": 0.8547, "sd": 0.0009},
            {"seeds": [0.8545, 0.8543, 0.8541], "mean": 0.8543, "sd": 0.0002},
            {"seeds": [0.8552, 0.8550, 0.8547], "mean": 0.8550, "sd": 0.0002},
        ],
        "SP-Q2": [
            {"seeds": [0.6083, 0.6151, 0.6083], "mean": 0.6106, "sd": 0.0039},
            {"seeds": [0.7177, 0.7230, 0.7256], "mean": 0.7221, "sd": 0.0040},
            {"seeds": [0.7990, 0.7992, 0.7997], "mean": 0.7993, "sd": 0.0004},
            {"seeds": [0.8342, 0.8346, 0.8327], "mean": 0.8338, "sd": 0.0010},
            {"seeds": [0.8479, 0.8498, 0.8479], "mean": 0.8485, "sd": 0.0011},
            {"seeds": [0.8424, 0.8433, 0.8385], "mean": 0.8414, "sd": 0.0025},
            {"seeds": [0.8578, 0.8600, 0.8603], "mean": 0.8594, "sd": 0.0013},
            {"seeds": [0.8590, 0.8576, 0.8570], "mean": 0.8579, "sd": 0.0010},
        ],
        "SP-Q4": [
            {"seeds": [0.4569, 0.4569, 0.4569], "mean": 0.4569, "sd": 0.0000},
            {"seeds": [0.5474, 0.5431, 0.5392], "mean": 0.5432, "sd": 0.0041},
            {"seeds": [0.6753, 0.6831, 0.6799], "mean": 0.6795, "sd": 0.0039},
            {"seeds": [0.7747, 0.7689, 0.7748], "mean": 0.7728, "sd": 0.0034},
            {"seeds": [0.8195, 0.8171, 0.8179], "mean": 0.8181, "sd": 0.0012},
            {"seeds": [0.8457, 0.8463, 0.8448], "mean": 0.8456, "sd": 0.0007},
            {"seeds": [0.8401, 0.8439, 0.8409], "mean": 0.8416, "sd": 0.0020},
            {"seeds": [0.8447, 0.8461, 0.8441], "mean": 0.8450, "sd": 0.0011},
        ],
        "VQAv2-opt": [
            {"seeds": [0.7375, 0.7352, 0.7373], "mean": 0.7367, "sd": 0.0013},
            {"seeds": [0.8041, 0.8077, 0.8073], "mean": 0.8064, "sd": 0.0020},
            {"seeds": [0.8366, 0.8330, 0.8303], "mean": 0.8333, "sd": 0.0032},
            {"seeds": [0.8402, 0.8419, 0.8440], "mean": 0.8420, "sd": 0.0019},
            {"seeds": [0.8470, 0.8475, 0.8464], "mean": 0.8470, "sd": 0.0005},
            {"seeds": [0.8558, 0.8559, 0.8552], "mean": 0.8557, "sd": 0.0004},
            {"seeds": [0.8578, 0.8596, 0.8599], "mean": 0.8591, "sd": 0.0011},
            {"seeds": [0.8583, 0.8590, 0.8589], "mean": 0.8587, "sd": 0.0003},
        ],
    },
    "TextVQA": {
        "SP-Q1": [
            {"seeds": [50.35, 49.98, 50.27], "mean": 50.20, "sd": 0.19},
            {"seeds": [51.35, 51.98, 51.26], "mean": 51.53, "sd": 0.39},
            {"seeds": [52.22, 52.21, 52.08], "mean": 52.17, "sd": 0.08},
            {"seeds": [52.84, 52.84, 52.48], "mean": 52.72, "sd": 0.21},
            {"seeds": [53.49, 53.27, 53.47], "mean": 53.41, "sd": 0.12},
            {"seeds": [53.78, 53.69, 53.97], "mean": 53.81, "sd": 0.14},
            {"seeds": [53.75, 53.65, 53.77], "mean": 53.72, "sd": 0.06},
            {"seeds": [53.79, 53.72, 53.70], "mean": 53.74, "sd": 0.05},
        ],
        "SP-Q2": [
            {"seeds": [48.09, 47.62, 48.03], "mean": 47.91, "sd": 0.26},
            {"seeds": [50.14, 50.35, 50.55], "mean": 50.35, "sd": 0.21},
            {"seeds": [51.94, 51.95, 51.96], "mean": 51.95, "sd": 0.01},
            {"seeds": [52.90, 52.53, 52.81], "mean": 52.75, "sd": 0.19},
            {"seeds": [53.93, 54.15, 53.99], "mean": 54.02, "sd": 0.11},
            {"seeds": [54.47, 54.21, 54.19], "mean": 54.29, "sd": 0.16},
            {"seeds": [54.50, 54.80, 54.79], "mean": 54.70, "sd": 0.17},
            {"seeds": [55.07, 54.72, 54.71], "mean": 54.83, "sd": 0.21},
        ],
        "SP-Q4": [
            {"seeds": [44.22, 44.22, 44.22], "mean": 44.22, "sd": 0.00},
            {"seeds": [46.93, 46.82, 46.72], "mean": 46.82, "sd": 0.11},
            {"seeds": [49.66, 49.71, 49.10], "mean": 49.49, "sd": 0.34},
            {"seeds": [51.70, 51.41, 51.83], "mean": 51.65, "sd": 0.22},
            {"seeds": [52.45, 52.97, 52.61], "mean": 52.68, "sd": 0.27},
            {"seeds": [53.67, 53.78, 53.58], "mean": 53.68, "sd": 0.10},
            {"seeds": [54.25, 54.42, 54.26], "mean": 54.31, "sd": 0.10},
            {"seeds": [54.66, 54.47, 54.46], "mean": 54.53, "sd": 0.11},
        ],
        "VQAv2-opt": [
            {"seeds": [50.33, 50.48, 50.11], "mean": 50.31, "sd": 0.19},
            {"seeds": [51.35, 51.98, 51.26], "mean": 51.53, "sd": 0.39},
            {"seeds": [52.22, 52.21, 52.08], "mean": 52.17, "sd": 0.08},
            {"seeds": [53.92, 53.17, 53.21], "mean": 53.43, "sd": 0.42},
            {"seeds": [54.29, 54.19, 53.76], "mean": 54.08, "sd": 0.28},
            {"seeds": [54.19, 54.70, 54.35], "mean": 54.41, "sd": 0.26},
            {"seeds": [54.84, 54.61, 54.77], "mean": 54.74, "sd": 0.12},
            {"seeds": [54.92, 55.11, 54.76], "mean": 54.93, "sd": 0.18},
        ],
    },
    "MMBench": {
        "SP-Q1": [
            {"seeds": [66.12, 66.03, 65.94], "mean": 66.03, "sd": 0.09},
            {"seeds": [67.67, 67.67, 67.76], "mean": 67.70, "sd": 0.05},
            {"seeds": [69.29, 69.29, 69.02], "mean": 69.20, "sd": 0.16},
            {"seeds": [71.19, 71.14, 70.82], "mean": 71.05, "sd": 0.20},
            {"seeds": [71.51, 71.33, 71.12], "mean": 71.32, "sd": 0.20},
            {"seeds": [72.06, 71.90, 71.94], "mean": 71.97, "sd": 0.08},
            {"seeds": [71.88, 71.99, 71.97], "mean": 71.95, "sd": 0.06},
            {"seeds": [71.42, 71.44, 71.44], "mean": 71.43, "sd": 0.01},
        ],
        "SP-Q2": [
            {"seeds": [60.91, 61.66, 61.16], "mean": 61.24, "sd": 0.38},
            {"seeds": [66.26, 65.94, 66.10], "mean": 66.10, "sd": 0.16},
            {"seeds": [68.65, 68.72, 68.49], "mean": 68.62, "sd": 0.12},
            {"seeds": [70.19, 70.09, 70.30], "mean": 70.19, "sd": 0.11},
            {"seeds": [71.69, 71.56, 71.72], "mean": 71.66, "sd": 0.09},
            {"seeds": [72.40, 72.17, 71.94], "mean": 72.17, "sd": 0.23},
            {"seeds": [72.24, 72.47, 72.54], "mean": 72.42, "sd": 0.16},
            {"seeds": [72.56, 72.63, 72.63], "mean": 72.61, "sd": 0.04},
        ],
        "SP-Q4": [
            {"seeds": [53.21, 53.21, 53.21], "mean": 53.21, "sd": 0.00},
            {"seeds": [59.17, 59.38, 59.10], "mean": 59.22, "sd": 0.15},
            {"seeds": [65.02, 64.84, 64.59], "mean": 64.82, "sd": 0.22},
            {"seeds": [67.88, 67.72, 68.27], "mean": 67.96, "sd": 0.28},
            {"seeds": [69.16, 69.64, 69.59], "mean": 69.46, "sd": 0.26},
            {"seeds": [70.69, 70.85, 70.78], "mean": 70.77, "sd": 0.08},
            {"seeds": [71.76, 71.94, 72.20], "mean": 71.97, "sd": 0.22},
            {"seeds": [72.15, 72.40, 72.10], "mean": 72.22, "sd": 0.16},
        ],
        "VQAv2-opt": [
            {"seeds": [65.50, 65.89, 65.32], "mean": 65.57, "sd": 0.29},
            {"seeds": [67.67, 67.67, 67.76], "mean": 67.70, "sd": 0.05},
            {"seeds": [69.29, 69.29, 69.02], "mean": 69.20, "sd": 0.16},
            {"seeds": [70.66, 71.03, 71.28], "mean": 70.99, "sd": 0.31},
            {"seeds": [71.65, 71.46, 71.14], "mean": 71.42, "sd": 0.26},
            {"seeds": [72.01, 72.33, 71.78], "mean": 72.04, "sd": 0.28},
            {"seeds": [72.15, 72.56, 72.31], "mean": 72.34, "sd": 0.21},
            {"seeds": [72.65, 72.54, 72.72], "mean": 72.64, "sd": 0.09},
        ],
    },
}
# NOTE: SP-Q1 at budget=41,000/66,000 and SP-Q2/SP-Q4's own "best" are
# identical to VQAv2-opt at those budgets in the raw search output (TPE
# found no config beating the All-B1 anchor there) -- kept as separate,
# duplicate-valued entries here for consistent 4-series plotting.
# TextVQA/MMBench added 2026-08-07: full validation batch (all 8 budgets x
# {search-best, All-B1/B2/B4}) run on AdaptQuant (PruneSID pipeline), same
# --split all --save protocol as MME/POPE. Source:
# results/validate_{textvqa,mmbench}_n4*_n2*_n1*.txt
# (Accuracy / MMBench Dev Accuracy line), verified against the compiled
# comparison table in this conversation. Win/tie count: TextVQA 7/8, MMBench
# only 3/8 -- MMBench generalizes noticeably worse than the other three
# benchmarks (search-best loses to a uniform baseline at 5 of 8 budgets, by
# 0.07-0.92pp each time -- corrected 2026-08-08, an earlier draft of this
# comment understated the range as "0.06-0.42pp").

COLORS = {"SP-Q1": "#2a78d6", "SP-Q2": "#eb6834", "SP-Q4": "#1baf7a", "VQAv2-opt": "#e34948"}
STYLES = {"SP-Q1": "--", "SP-Q2": "-.", "SP-Q4": ":", "VQAv2-opt": "-"}
WIDTHS = {"SP-Q1": 1.5, "SP-Q2": 1.5, "SP-Q4": 1.5, "VQAv2-opt": 2.5}

# Display-only legend labels (data dict keys above stay "SP-Q1"/"SP-Q2"/
# "SP-Q4"/"VQAv2-opt" everywhere else so N_BY_METHOD / T_CLOUD_BY_METHOD /
# DATA don't need to change). Renamed SP-Qn -> PSID-Qn (PruneSID-Qn) for the
# figures 2026-08-17 per user request -- Static-Qn (FIG3_STATIC_LABELS
# below) is NOT renamed to match, since it explicitly does NOT run PruneSID
# selection at all (that's the whole point of the Static tier).
LABELS = {"SP-Q1": "PSID-Q1", "SP-Q2": "PSID-Q2", "SP-Q4": "PSID-Q4", "VQAv2-opt": "Proposed"}

# Figure 3 ONLY, fixed/non-adaptive variant: SP-Q1/Q2/Q4 always send N=576 at a
# fixed bit-depth, ignoring bandwidth/T_deadline entirely -- a different
# experiment from the budget-matched SP-Q1/Q2/Q4 swept in Figures 1/2
# (paper_data.N_BY_METHOD), so it gets a distinct label -- decided with the
# user 2026-07-29 -- to avoid implying it's the same setup. Figure 3 ALSO
# plots the budget-matched SP-Q1/Q2/Q4 (dynamic snap-down over their own
# 8-anchor family, same definition as Figures 1/2) using LABELS above, so
# Figure 3 shows both variants side by side. No static counterpart for the
# proposed method -- it only has the adaptive form.
FIG3_STATIC_LABELS = {"SP-Q1": "Static-Q1", "SP-Q2": "Static-Q2", "SP-Q4": "Static-Q4"}

# ============================================================
# Static-Q1/Q2/Q4 accuracy (2026-08-17): N=576 (every token, no PruneSID
# pruning), a single fixed bit-depth, no bandwidth adaptation -- the actual
# accuracy of the Figure 3 Static tier, which until now only had a latency
# model (Figure 3's docstring said "makes no accuracy claim"). Mean over 3
# PCA seeds (42/43/44), 3 configs x 4 benchmarks x 3 seeds = 36 runs, zero
# failures. Static-Q1 (n1=576) duplicates SP-Q1's own budget=589,824 anchor
# (DATA["..."]["SP-Q1"][7]) almost exactly -- same config, same result,
# confirms determinism. Static-Q2/Q4 are new: nothing in the adaptive
# SP-Q2/SP-Q4 sweep ever reaches N=576 (their own budget=589,824 anchor
# tops out at N=288/144 respectively -- see N_BY_METHOD), since T//2 and
# T//4 cap out well below 576 tokens even at the largest tested budget.
# NOTE: these scores are NOT budget-matched against anything else in this
# file -- Static-Q4 at N=576 costs 576*1024*4 = 2,359,296 bits, ~4x the
# largest budget (589,824) used everywhere else. Unsurprisingly they beat
# every budget-matched method's best score (e.g. MME: Static-Q4 1765.9 vs.
# Proposed's best 1790.3 stays ahead, but Static-Q2 1783.8 and TextVQA/
# MMBench/POPE Static-Q4 all edge past Proposed's best) -- that's expected,
# not a result worth reading as "Static beats Proposed", since nothing here
# is budget-matched. Source: results/validate_{mme,pope,textvqa,mmbench}_
# {n4576_n20_n10,n40_n2576_n10,n40_n20_n1576}{,_pcaseed43,_pcaseed44}_*.txt
# ============================================================
STATIC_ACCURACY = {
    "MME": {
        "SP-Q1": {"seeds": [1730.3699, 1739.4876, 1732.8699], "mean": 1734.2425, "sd": 3.8467},
        "SP-Q2": {"seeds": [1780.6593, 1792.1593, 1778.6235], "mean": 1783.8140, "sd": 5.9592},
        "SP-Q4": {"seeds": [1761.4105, 1769.7781, 1766.4105], "mean": 1765.8664, "sd": 3.4377},
    },
    "POPE": {
        "SP-Q1": {"seeds": [0.8552, 0.8550, 0.8547], "mean": 0.8550, "sd": 0.0002},
        "SP-Q2": {"seeds": [0.8575, 0.8575, 0.8572], "mean": 0.8574, "sd": 0.0002},
        "SP-Q4": {"seeds": [0.8601, 0.8601, 0.8598], "mean": 0.8600, "sd": 0.0002},
    },
    "TextVQA": {
        "SP-Q1": {"seeds": [53.79, 53.72, 53.70], "mean": 53.7367, "sd": 0.0386},
        "SP-Q2": {"seeds": [55.14, 55.20, 55.20], "mean": 55.1800, "sd": 0.0283},
        "SP-Q4": {"seeds": [55.27, 55.23, 55.32], "mean": 55.2733, "sd": 0.0368},
    },
    "MMBench": {
        "SP-Q1": {"seeds": [71.42, 71.44, 71.44], "mean": 71.4333, "sd": 0.0094},
        "SP-Q2": {"seeds": [72.33, 72.42, 72.38], "mean": 72.3767, "sd": 0.0368},
        "SP-Q4": {"seeds": [72.86, 72.81, 72.90], "mean": 72.8567, "sd": 0.0368},
    },
}

# ============================================================
# Per-method token count N at each budget.
# SP-Q1/Q2/Q4: the "anchors" field of results/search_vqav2_*.json (the pure
# single-bit-depth seed configs TPE search started from -- e.g. SP-Q2 at
# budget=25,600 is (n4=0, n2=12, n1=1): 12 tokens at 2-bit plus 1 leftover
# 1-bit token to hit the exact budget).
# VQAv2-opt: the best config found by the search ("results" field of the same files).
# ============================================================
N_BY_METHOD = {
    "SP-Q1":     [25, 40, 64, 103, 166, 266, 428, 576],
    "SP-Q2":     [12, 20, 32, 51, 83, 133, 214, 288],
    "SP-Q4":     [6, 10, 16, 25, 41, 66, 107, 144],
    "VQAv2-opt": [23, 40, 64, 79, 87, 183, 305, 295],
}

# ============================================================
# T_cloud(N) in ms, looked up per-method (not shared across methods -- SP-Q1 and
# SP-Q4 use very different N at the same budget, so they have different cloud
# compute time). Source: latency_collec_scripts/final_lat/results_cloud.csv,
# a direct N=1..576 sweep measured on the current cloud model/hardware.
# ============================================================
T_CLOUD_BY_METHOD = {
    "SP-Q1":     [21.39, 21.43, 22.40, 26.68, 31.69, 39.83, 54.63, 64.96],
    "SP-Q2":     [21.21, 20.98, 21.40, 22.21, 23.39, 28.24, 31.67, 40.01],
    "SP-Q4":     [21.16, 21.18, 21.01, 21.39, 21.45, 22.57, 26.76, 27.48],
    "VQAv2-opt": [21.10, 21.43, 22.40, 23.26, 23.43, 33.14, 42.29, 41.89],
}
# VQAv2-opt row re-looked-up from latency_collec_scripts/final_lat/results_cloud.csv
# at the new N_BY_METHOD["VQAv2-opt"] values (2026-08-07); SP-Q1/Q2/Q4 rows
# unchanged since their N didn't change (see module docstring).

T_EDGE = 231  # ViT-L/14-336 forward pass only (measure_edge.py), fixed regardless of N
T_DEADLINE = 350

# ============================================================
# PruneSID PCA-cluster + intra-cluster-NMS selection latency (ms), as a
# function of target token count N -- measured on the actual edge target
# (Jetson AGX Orin), NOT the earlier A100 stand-in. This is on-device cost
# that happens AFTER the ViT forward pass (T_EDGE above) and BEFORE
# transmission, for every method that runs PruneSID selection at all.
# Source: latency_collec_scripts/measure_edge_prunesid.py ->
# results_edge_prunesid_AGX_Orin.csv (N=1..576, step 8; linearly
# interpolated between sweep points).
# ============================================================
_PRUNESID_CSV = _Path(__file__).resolve().parent.parent / "latency" / "data" / "results_edge_prunesid_AGX_Orin.csv"


def _load_prunesid_lut(path):
    ns, ms = [], []
    with open(path) as f:
        for row in _csv.DictReader(f):
            ns.append(int(row["N"]))
            ms.append(float(row["mean_ms"]))
    return ns, ms


_PRUNESID_N, _PRUNESID_MS = _load_prunesid_lut(_PRUNESID_CSV)


def t_prunesid_cluster(n: int) -> float:
    """PruneSID selection latency (ms) at target token count n, linearly
    interpolated from the AGX Orin sweep. Clamped to the sweep's range
    ([1, 576]) at either end."""
    ns, ms = _PRUNESID_N, _PRUNESID_MS
    if n <= ns[0]:
        return ms[0]
    if n >= ns[-1]:
        return ms[-1]
    for i in range(1, len(ns)):
        if n <= ns[i]:
            lo_n, hi_n = ns[i - 1], ns[i]
            lo_m, hi_m = ms[i - 1], ms[i]
            return lo_m + (n - lo_n) / (hi_n - lo_n) * (hi_m - lo_m)
    return ms[-1]  # unreachable


def t_edge_total(n: int) -> float:
    """Total on-device latency for a PruneSID-selecting method targeting n
    tokens: ViT forward (T_EDGE, constant) + PruneSID selection (variable).
    Does NOT include quantization (T_quant below) -- add that separately at
    the call site, since it depends on the bit-depth split (n4/n2/n1), not
    just total N, and this function only knows N.
    Do NOT use this for Static-Q1/Q2/Q4 (N=576, no selection runs -- use
    T_EDGE alone) or Edge-only/Cloud-only (separately-measured whole-pipeline
    constants, T_LATENCY_ORI_EDGE/CLOUD below)."""
    return T_EDGE + t_prunesid_cluster(n)


# ============================================================
# Quantization (MSECompressor.compress()) latency (ms), measured on the
# actual edge target (Jetson AGX Orin) -- same CSV as T_EDGE
# (results_edge_Jetson_AGX_Orin.csv), component="T_quant", a dense N=1..576
# x bits={1,2,4} sweep (1,728 rows, no interpolation needed). This was
# missing from every latency calculation until 2026-08-19 -- confirmed via
# measure_edge_prunesid.py's own docstring ("Quantization (T_quant) ... NOT
# included here"), then found this sweep already existed in the raw CSV,
# just never wired into t_edge_total/plot_paper_figures.py/
# plot_vqav2_figures.py.
# ============================================================
_QUANT_CSV = _Path(__file__).resolve().parent.parent / "latency" / "data" / "results_edge_Jetson_AGX_Orin.csv"


def _load_quant_lut(path):
    lut = {}
    with open(path) as f:
        for row in _csv.DictReader(f):
            if row["component"] == "T_quant":
                lut[(int(row["N"]), int(row["B"]))] = float(row["mean_ms"])
    return lut


_QUANT_LUT = _load_quant_lut(_QUANT_CSV)


def t_quant_lookup(n: int, bits: int) -> float:
    """MSECompressor.compress() latency (ms) for n tokens at a given
    bit-depth. Exact lookup (dense sweep, no interpolation). Returns 0.0 for
    n<=0 (no tokens at that tier -- compress() is never called at all)."""
    if n <= 0:
        return 0.0
    return _QUANT_LUT[(min(n, N_FIXED_BASELINE), bits)]


def t_quant_stratified(n4: int, n2: int, n1: int) -> float:
    """Total quantization latency for a (possibly mixed-tier) config --
    StratifiedCompressor calls MSECompressor.compress() once per distinct
    bit-depth present, on exactly that tier's tokens, so the total is the
    sum of each tier's own cost. Works for uniform-bit-depth methods too
    (two of the three args are just 0, contributing nothing)."""
    return t_quant_lookup(n4, 4) + t_quant_lookup(n2, 2) + t_quant_lookup(n1, 1)

# ============================================================
# Figure 3 fixed baselines (2026-07-29): SP-Q1/Q2/Q4 there are NOT the dynamic
# 8-anchor families above -- they're a single fixed strategy (no PruMerge, no
# bandwidth-adaptation) so the comparison against the proposed method's dynamic
# snap-down is "adapts nothing" vs "adapts both N and bit-depth". Fixed at
# N=576 (send every token, no pruning) so the reference point is an
# already-measured anchor (T_CLOUD_BY_METHOD["SP-Q1"][7] == T_CLOUD_N576) rather
# than an arbitrary pick from the 8-point grid.
# ============================================================
N_FIXED_BASELINE = 576
TOKEN_DIM = 1024
T_CLOUD_N576 = 64.96
BIT_BY_METHOD = {"SP-Q1": 1, "SP-Q2": 2, "SP-Q4": 4}

# ============================================================
# Per-token tier-ID overhead (2026-08-19 fix): a mixed-precision transmission
# (n4 tokens at 4-bit, n2 at 2-bit, n1 at 1-bit all in the same payload) must
# also tell the cloud which tier each transmitted token is in, or it can't
# unpack the bit-packed indices correctly -- ceil(log2(3)) = 2 bits/token,
# on top of the quantized payload itself. Missing from every T_tx
# calculation until now. This ONLY applies to genuinely mixed-tier
# transmission (Proposed/VQAv2-opt, both anchors and snap-and-refill
# midpoints) -- SP-Q1/Q2/Q4 and Static-Q1/Q2/Q4 always use a single uniform
# bit-width for all N tokens, so the receiver already knows the tier from a
# one-time header, no per-token ambiguity, no overhead needed there.
TIER_ID_BITS_PER_TOKEN = 2  # ceil(log2(3)) -- 3 possible tiers: 4-bit/2-bit/1-bit

# Per-token vector-norm overhead (2026-08-26 fix): TurboQuant's MSECompressor
# normalizes each token's 1024-dim feature vector to the unit sphere before
# quantizing it, and stores the L2 norm separately (fp16) so decompress() can
# rescale the dequantized unit vector back to the original magnitude
# (turboquant/compressors_v3.py: MSECompressor.compress()'s "vec_norms" /
# decompress()'s "* vec_norms"). That norm is real payload, transmitted for
# every token regardless of its bit-tier -- missing from every T_tx
# calculation until now (the KV-cache-compression code elsewhere in this repo,
# turboquant/validate.py / validate_v3.py, already accounts for this same
# 16-bit/vector cost; our token-quantization budget math never did). Unlike
# TIER_ID_BITS_PER_TOKEN, this applies to EVERY method (Static/SP-Q.../
# Proposed all route through MSECompressor), not just mixed-tier Proposed.
NORM_BITS_PER_TOKEN = 16  # fp16 L2 norm, one per transmitted token

# ============================================================
# Two more "trivial" (non-adaptive, single-device) baselines for Figure 3,
# measured directly rather than derived from the adaptive pipeline: run the
# ENTIRE original, unquantized LLaVA-1.5 pipeline (ViT + MLP projector + LLM,
# all 576 visual + 40 text tokens, full precision) on ONE device only -- no
# split, no pruning, no quantization, no bandwidth-adaptation.
#   - Edge-only:  everything (including the LLM) runs on-device (Jetson AGX
#     Orin) -- no network round-trip at all, so T_total is a CONSTANT,
#     independent of bandwidth S.
#   - Cloud-only: the raw 336x336x3 image is transmitted uncompressed
#     (8 bits/channel, no ViT/pruning/quant on-device at all), then the
#     cloud runs the same full pipeline. T_total(S) = T_raw_tx(S) + T_latency_ori_cloud.
# Source: latency_collec_scripts/measure_latency_ori.py,
# final_lat/results_latency_ori_{Jetson_AGX_Orin,A100}.csv (2026-08-07).
# ============================================================
T_LATENCY_ORI_EDGE = 2008.445109485183    # ms, full pipeline on Jetson AGX Orin (edge-only)
T_LATENCY_ORI_CLOUD = 71.33370032534003   # ms, full pipeline on A100 (cloud-only, post-transmission)
RAW_IMAGE_BITS = 336 * 336 * 3 * 8         # uncompressed 336x336x3, 8 bits/channel

# Shared identity colors/labels for these two single-device baselines, so
# Figures 1/2/3/3b all draw "Edge-only"/"Cloud-only" the same color instead of
# each plotting script picking its own -- color follows the entity, not the
# figure. Neither collides with COLORS (blue/orange/green/red, all used by
# the 4 adaptive-method series above).
EDGE_ONLY_COLOR = "#4d4d4d"
CLOUD_ONLY_COLOR = "#7a5195"
EDGE_ONLY_LABEL = "Edge inference"
CLOUD_ONLY_LABEL = "Cloud inference"

# ============================================================
# FP16 (no-quant, N=576) baselines for normalization -- the TRUE
# original-model path (validate.py --baseline, routes to
# LLAVA_TOKEN_METHOD="original" -- vanilla CLIP forward, no PruneSID
# selection/reordering, no quant; see clip_encoder.py forward()'s else
# branch). This path has no PCA/randomness at all (PruneSID's selection
# step never runs), so it's fully deterministic -- one run each is enough,
# no seed averaging needed, unlike every other number in this file.
# MME/POPE measured 2026-08-07. TextVQA/MMBench measured 2026-08-18
# (results/validate_{textvqa,mmbench}_baseline_*.txt) -- TextVQA's old
# stale number (from the pre-pipeline MME-objective calibration,
# results/exp_results_summary.md, 2026-06-25) is now superseded.
# ============================================================
BASELINE_FP16 = {
    "MME": 1866.4551,
    "POPE": 0.8593,
    "TextVQA": 58.22,
    "MMBench": 73.70,
}

# ============================================================
# Ablation: sink-token handling (MME, 3 PCA seeds, mean and sample sd).
# Same PruneSID selection and the same (n4, n2, n1) as Proposed at every
# budget; only the treatment of detected sink candidates changes:
#   pin       -- detected sinks pinned to the lowest tier (the method)
#   no_sink   -- no sink detection (--no-sink-filter)
#   no_cap    -- like pin, but every token above 100x the median is a sink,
#                not only the top two (--sink-max-candidates 576)
#   capa_rank -- no sink detection; tokens ranked by CAPA contribution
# Reproduce: python reproduce.py sink
# ============================================================
ABLATION_SINK_MME_SEEDED = {
    25_600: {
        "n4": 0, "n2": 2, "n1": 21,
        "pin":       {"seeds": [1555.87, 1552.71, 1497.15], "mean": 1535.24, "stdev": 33.03},
        "no_sink":   {"seeds": [1508.53, 1518.73, 1445.51], "mean": 1490.92, "stdev": 39.66},
        "no_cap":    {"seeds": [1567.00, 1577.34, 1550.31], "mean": 1564.88, "stdev": 13.64},
        "capa_rank": {"seeds": [1528.80, 1511.30, 1502.68], "mean": 1514.26, "stdev": 13.31},
    },
    41_000: {
        "n4": 0, "n2": 0, "n1": 40,
        "pin":       {"seeds": [1599.03, 1608.04, 1610.14], "mean": 1605.73, "stdev": 5.90},
        "no_sink":   {"seeds": [1574.15, 1564.70, 1592.99], "mean": 1577.28, "stdev": 14.41},
        "no_cap":    {"seeds": [1588.91, 1620.13, 1588.12], "mean": 1599.05, "stdev": 18.26},
        "capa_rank": {"seeds": [1575.28, 1573.96, 1584.75], "mean": 1578.00, "stdev": 5.89},
    },
    66_000: {
        "n4": 0, "n2": 0, "n1": 64,
        "pin":       {"seeds": [1602.36, 1624.28, 1667.95], "mean": 1631.53, "stdev": 33.39},
        "no_sink":   {"seeds": [1610.97, 1611.14, 1623.67], "mean": 1615.26, "stdev": 7.28},
        "no_cap":    {"seeds": [1613.33, 1592.10, 1650.57], "mean": 1618.67, "stdev": 29.60},
        "capa_rank": {"seeds": [1632.96, 1649.87, 1646.80], "mean": 1643.21, "stdev": 9.01},
    },
    106_000: {
        "n4": 1, "n2": 21, "n1": 57,
        "pin":       {"seeds": [1666.19, 1707.10, 1717.84], "mean": 1697.05, "stdev": 27.25},
        "no_sink":   {"seeds": [1662.78, 1686.31, 1700.15], "mean": 1683.08, "stdev": 18.89},
        "no_cap":    {"seeds": [1675.56, 1710.81, 1714.75], "mean": 1700.37, "stdev": 21.58},
        "capa_rank": {"seeds": [1680.42, 1709.87, 1726.35], "mean": 1705.54, "stdev": 23.27},
    },
    170_000: {
        "n4": 8, "n2": 55, "n1": 24,
        "pin":       {"seeds": [1713.90, 1715.61, 1737.39], "mean": 1722.30, "stdev": 13.09},
        "no_sink":   {"seeds": [1708.54, 1722.14, 1722.46], "mean": 1717.72, "stdev": 7.94},
        "no_cap":    {"seeds": [1693.71, 1703.30, 1718.24], "mean": 1705.08, "stdev": 12.36},
        "capa_rank": {"seeds": [1722.91, 1704.67, 1712.32], "mean": 1713.30, "stdev": 9.16},
    },
    273_000: {
        "n4": 2, "n2": 77, "n1": 104,
        "pin":       {"seeds": [1748.24, 1740.66, 1753.78], "mean": 1747.56, "stdev": 6.59},
        "no_sink":   {"seeds": [1750.67, 1737.85, 1722.67], "mean": 1737.07, "stdev": 14.01},
        "no_cap":    {"seeds": [1746.90, 1701.78, 1737.14], "mean": 1728.61, "stdev": 23.74},
        "capa_rank": {"seeds": [1771.12, 1754.32, 1759.18], "mean": 1761.54, "stdev": 8.65},
    },
    439_000: {
        "n4": 4, "n2": 111, "n1": 190,
        "pin":       {"seeds": [1784.96, 1790.61, 1786.11], "mean": 1787.22, "stdev": 2.98},
        "no_sink":   {"seeds": [1782.07, 1791.39, 1798.77], "mean": 1790.74, "stdev": 8.37},
        "no_cap":    {"seeds": [1745.30, 1762.02, 1763.62], "mean": 1756.98, "stdev": 10.15},
        "capa_rank": {"seeds": [1808.59, 1816.05, 1836.75], "mean": 1820.46, "stdev": 14.59},
    },
    589_824: {
        "n4": 25, "n2": 206, "n1": 64,
        "pin":       {"seeds": [1805.60, 1798.21, 1780.39], "mean": 1794.73, "stdev": 12.96},
        "no_sink":   {"seeds": [1807.38, 1783.51, 1776.70], "mean": 1789.19, "stdev": 16.11},
        "no_cap":    {"seeds": [1798.18, 1784.77, 1780.39], "mean": 1787.78, "stdev": 9.27},
        "capa_rank": {"seeds": [1803.89, 1810.92, 1791.85], "mean": 1802.22, "stdev": 9.64},
    },
}

# ============================================================
# Sink-detector sensitivity (MME, 3 PCA seeds): threshold ratio tau
# (default 100) at a low, middle and high budget, with the Proposed tier
# sizes. tau=100, no sink handling and the no-cap variant at these budgets
# are in ABLATION_SINK_MME_SEEDED.
# Reproduce: python reproduce.py sink-threshold
# ============================================================
SINK_THRESHOLD_MME = {
    25_600: {
        "ratio25":  {"seeds": [1556.24, 1551.48, 1501.65], "mean": 1536.45, "stdev": 30.24},
        "ratio50":  {"seeds": [1556.24, 1551.48, 1496.65], "mean": 1534.79, "stdev": 33.12},
        "ratio200": {"seeds": [1527.70, 1551.48, 1486.27], "mean": 1521.82, "stdev": 33.00},
    },
    170_000: {
        "ratio25":  {"seeds": [1714.78, 1715.90, 1738.56], "mean": 1723.08, "stdev": 13.42},
        "ratio50":  {"seeds": [1715.53, 1715.90, 1738.56], "mean": 1723.33, "stdev": 13.19},
        "ratio200": {"seeds": [1707.17, 1709.25, 1738.37], "mean": 1718.26, "stdev": 17.44},
    },
    589_824: {
        "ratio25":  {"seeds": [1805.60, 1797.46, 1780.39], "mean": 1794.48, "stdev": 12.87},
        "ratio50":  {"seeds": [1805.60, 1797.46, 1780.39], "mean": 1794.48, "stdev": 12.87},
        "ratio200": {"seeds": [1804.65, 1796.49, 1780.16], "mean": 1793.77, "stdev": 12.47},
    },
}

# ============================================================
# Snap-and-refill anchor configs -- VQAv2-opt search-best per budget.
# Source: results/search_vqav2_S*.json, PruneSID
# pipeline, search on VQAv2 random-5000, re-measured 2026-08-07 (supersedes
# the old PruMerge-pipeline table confirmed 2026-07-06).
# ============================================================
ANCHOR_CONFIGS = {
    25_600:  {"n4":  0, "n2":   2, "n1":  21, "N":  23},
    41_000:  {"n4":  0, "n2":   0, "n1":  40, "N":  40},
    66_000:  {"n4":  0, "n2":   0, "n1":  64, "N":  64},
    106_000: {"n4":  1, "n2":  21, "n1":  57, "N":  79},
    170_000: {"n4":  8, "n2":  55, "n1":  24, "N":  87},
    273_000: {"n4":  2, "n2":  77, "n1": 104, "N": 183},
    439_000: {"n4":  4, "n2": 111, "n1": 190, "N": 305},
    589_824: {"n4": 25, "n2": 206, "n1":  64, "N": 295},
}

# 7 midpoint budgets (geometric mean of adjacent anchors)
# extra = B_mid // 1024 - B_anch // 1024
SNAP_REFILL_POINTS = [
    {"B_mid": 32_400,  "B_anch": 25_600,  "extra": 6},
    {"B_mid": 52_000,  "B_anch": 41_000,  "extra": 10},
    {"B_mid": 83_600,  "B_anch": 66_000,  "extra": 17},
    {"B_mid": 134_200, "B_anch": 106_000, "extra": 27},
    {"B_mid": 215_400, "B_anch": 170_000, "extra": 44},
    {"B_mid": 346_200, "B_anch": 273_000, "extra": 71},
    {"B_mid": 508_900, "B_anch": 439_000, "extra": 68},
]

# ============================================================
# Snap-and-refill -- FINAL strategy (decided 2026-08-07): S1, lower anchor's
# (n4, n2) held fixed, leftover budget converted into extra 1-bit tokens.
# An S2 alternative (proportional n4/n2/n1 scaling) was measured against S1
# on MME at all 7 points below and dropped -- S1 won 3/4 head-to-head, with
# larger margins when it won. This is what actually serves arbitrary
# real-world budgets between the 8 searched anchors, without re-searching.
#
# 7 geometric-midpoint budgets between adjacent anchors (budget_target =
# sqrt(B_lo * B_hi), same definition as plot_figure4.build_interpolation_points
# / refill_config -- these ARE that function's output, just also surfaced
# here so Figures 1/2 can plot them). Configs/MME scores verified against
# results/validate_mme_n4*_n2*_n1*.txt ("Combined
# Total Score" line), 2026-08-07. POPE/TextVQA/MMBench scores added
# 2026-08-08 (78-run validation batch across all 4 strategies x 7 midpoints,
# see MIDPOINT_DATA below for the full per-strategy breakdown).
# ============================================================
REFILL_POINTS_S1 = [
    {"budget_target": 32397.53077010654,  "lower_anchor": 25_600,  "upper_anchor": 41_000,
     "n4": 0, "n2": 2,   "n1": 27,  "N": 29,  "actual_budget": 31_744,  "t_cloud": 21.34},
    {"budget_target": 52019.22721455981,  "lower_anchor": 41_000,  "upper_anchor": 66_000,
     "n4": 0, "n2": 0,   "n1": 50,  "N": 50,  "actual_budget": 51_200,  "t_cloud": 22.11},
    {"budget_target": 83642.09466530594,  "lower_anchor": 66_000,  "upper_anchor": 106_000,
     "n4": 0, "n2": 0,   "n1": 81,  "N": 81,  "actual_budget": 82_944,  "t_cloud": 23.27},
    {"budget_target": 134238.59355639867, "lower_anchor": 106_000, "upper_anchor": 170_000,
     "n4": 1, "n2": 21,  "n1": 85,  "N": 107, "actual_budget": 134_144, "t_cloud": 26.76},
    {"budget_target": 215429.80295214494, "lower_anchor": 170_000, "upper_anchor": 273_000,
     "n4": 8, "n2": 55,  "n1": 68,  "N": 131, "actual_budget": 215_040, "t_cloud": 28.22},
    {"budget_target": 346189.2545992726,  "lower_anchor": 273_000, "upper_anchor": 439_000,
     "n4": 2, "n2": 77,  "n1": 176, "N": 255, "actual_budget": 346_112, "t_cloud": 39.93},
    {"budget_target": 508854.33672122716, "lower_anchor": 439_000, "upper_anchor": 589_824,
     "n4": 4, "n2": 111, "n1": 258, "N": 373, "actual_budget": 507_904, "t_cloud": 53.15},
]

# ============================================================
# Continuous snap-and-refill: N(B) for ANY total bit budget B, not just the
# 7 midpoint budgets in REFILL_POINTS_S1 above -- those were only ever
# picked to VALIDATE the S1 rule against S2 on real MME accuracy (a finite
# test), the rule itself is continuous. Figure 3 only needs latency, not
# accuracy, so it doesn't need to be restricted to those 7 tested points --
# every extra TOKEN_DIM (1024) bits above the lower anchor's own budget
# buys exactly one more 1-bit token (n1 += 1), up to the next anchor.
# ============================================================
def refill_n_for_budget(budget: float) -> int:
    """Token count N achievable at budget under the S1 rule: snap to the
    largest searched anchor <= budget, then convert every extra 1024 bits
    into one more 1-bit token. Capped at N_FIXED_BASELINE (576, all tokens)."""
    anchor_budgets = sorted(ANCHOR_CONFIGS)
    lower = anchor_budgets[0]
    for b in anchor_budgets:
        if b <= budget:
            lower = b
        else:
            break
    extra_tokens = int((budget - lower) // TOKEN_DIM)
    return min(ANCHOR_CONFIGS[lower]["N"] + extra_tokens, N_FIXED_BASELINE)


def refill_tiers_for_budget(budget: float):
    """(n4, n2, n1) at `budget` under the S1 rule -- same snap+refill logic
    as refill_n_for_budget, but returns the actual tier split instead of
    just the total N. Needed for anything that depends on WHICH bit-depth
    each token is at, not just how many tokens there are (e.g. T_quant,
    since MSECompressor.compress() cost depends on both n and bits)."""
    anchor_budgets = sorted(ANCHOR_CONFIGS)
    lower = anchor_budgets[0]
    for b in anchor_budgets:
        if b <= budget:
            lower = b
        else:
            break
    extra_tokens = int((budget - lower) // TOKEN_DIM)
    n4, n2, n1 = ANCHOR_CONFIGS[lower]["n4"], ANCHOR_CONFIGS[lower]["n2"], ANCHOR_CONFIGS[lower]["n1"]
    n1 += extra_tokens
    total = n4 + n2 + n1
    if total > N_FIXED_BASELINE:
        n1 -= (total - N_FIXED_BASELINE)  # same cap as refill_n_for_budget, applied to n1 (the tier refill adds to)
    return n4, n2, n1


_CLOUD_CSV = _Path(__file__).resolve().parent.parent / "latency" / "data" / "results_cloud.csv"


def _load_cloud_lut(path):
    lut = {}
    with open(path) as f:
        for row in _csv.DictReader(f):
            lut[int(row["N"])] = float(row["mean_ms"])
    return lut


_CLOUD_LUT = _load_cloud_lut(_CLOUD_CSV)


def t_cloud_lookup(n: int) -> float:
    """Exact T_cloud(N), N=1..576, no interpolation needed (results_cloud.csv
    is a dense sweep -- every N was actually measured, same source as
    T_CLOUD_BY_METHOD above, just not restricted to the 8 anchor N's)."""
    return _CLOUD_LUT[n]

# 2026-08-16 UPDATE: mean over 3 PCA seeds (42/43/44), same fix/rationale as
# DATA's 2026-08-15 update -- replaces the single-seed (seed=42) numbers.
# Raw per-seed scores are not transcribed into a *_SEEDED dict here (unlike
# DATA_SEEDED) -- they're recoverable from the raw result files:
# results/validate_{mme,pope,textvqa,mmbench}_n4*_n2*_n1*_pcaseed{43,44}_*.txt
# (seed=42 files have no _pcaseed suffix). MIDPOINT_DATA's own "score" field
# per strategy carries the same corrected mean as this dict.
REFILL_SCORES_S1 = {
    "MME":     [1583.7805, 1613.0922, 1688.4028, 1717.8306, 1737.6577, 1781.2080, 1778.1671],
    "POPE":    [0.7767, 0.8237, 0.8397, 0.8385, 0.8406, 0.8590, 0.8571],
    "TextVQA": [51.08, 51.98, 52.77, 53.74, 54.15, 54.45, 54.77],
    "MMBench": [67.20, 68.69, 70.48, 71.48, 71.75, 72.27, 72.49],
}

# ============================================================
# Full midpoint validation batch -- the same 7 geometric-midpoint budgets,
# but ALL FIVE strategies that were actually run there: the B1/B2/B4 uniform
# baselines (SP-Q1/Q2/Q4 continued past their 8 anchors) plus both
# snap-and-refill candidates, S1 (final choice, duplicates REFILL_POINTS_S1/
# REFILL_SCORES_S1 above) and S2 (proportional n4/n2/n1 scaling, dropped --
# kept here for reference, not plotted). This is what lets Figures 1/2 fill
# the gap for EVERY series, not just the proposed method's.
#
# 2026-08-18 UPDATE: S2 (MME only) is now also a 3-seed mean (42/43/44) --
# the missing seed43/44 runs at the 4 diverging midpoints (134,200/215,400/
# 346,200/508,900) landed, zero failures. At 32,400/52,000/83,600 S2's
# config is byte-identical to S1's (see note below), so its 3-seed mean
# there is just S1's own value, no separate run needed. Head-to-head at the
# 4 diverging points under 3-seed averaging: S1 wins 3/4 (+14.84 at
# 134,200; +13.52 at 215,400; +8.75 at 346,200), S2 wins the last one
# (-22.38 at 508,900, a wider margin than the old single-seed comparison
# showed). Every other strategy (S1/B1/B2/B4) at every midpoint was already
# 3-seed; S2 was the last remaining single-seed number in this file.
#
# At 32,400/52,000/83,600 the S1/S2 configs collapse to the same values as
# B1 (too few extra tokens for the strategies to diverge) -- kept as
# duplicate entries for a consistent per-budget schema. All 5 strategies at
# a given midpoint budget solve 4*n4+2*n2+n1 == T for the same T, so they
# share one "actual_budget" (bits) and there's a single x-position per
# midpoint per figure, not five.
#
# t_cloud(N) looked up per-config from the N=1..576 sweep in
# latency_collec_scripts/final_lat/results_cloud.csv (same source as
# T_CLOUD_BY_METHOD above); shared across all 4 benchmarks since it only
# depends on N, not the benchmark. Only S2 is MME-only -- it was already
# dropped in favor of S1 before POPE/TextVQA/MMBench were validated, so
# there was no reason to spend compute re-confirming it elsewhere; every
# other strategy (B1/B2/B4/S1) is now populated for all 4 benchmarks.
# Scores: results/validate_mme_n4*_n2*_n1*.txt
# ("Combined Total Score" line), 2026-08-07; POPE/TextVQA/MMBench from the
# 78-run validation batch, 2026-08-08 (F1 score / Accuracy / MMBench Dev
# Accuracy lines respectively).
# ============================================================
MIDPOINT_DATA = {
    "MME": {
        32_400: {"actual_budget": 31_744,
                 "B1": {"n4": 0, "n2": 0, "n1": 31, "N": 31, "t_cloud": 21.35, "score": 1583.05},
                 "B2": {"n4": 0, "n2": 15, "n1": 0, "N": 15, "t_cloud": 21.05, "score": 1450.07},
                 "B4": {"n4": 7, "n2": 0, "n1": 0, "N": 7, "t_cloud": 21.1, "score": 1209.25},
                 "S1": {"n4": 0, "n2": 2, "n1": 27, "N": 29, "t_cloud": 21.34, "score": 1575.34},
                 "S2": {"n4": 0, "n2": 2, "n1": 27, "N": 29, "t_cloud": 21.34, "score": 1575.34},
        },
        52_000: {"actual_budget": 51_200,
                 "B1": {"n4": 0, "n2": 0, "n1": 50, "N": 50, "t_cloud": 22.11, "score": 1615.39},
                 "B2": {"n4": 0, "n2": 25, "n1": 0, "N": 25, "t_cloud": 21.39, "score": 1574.45},
                 "B4": {"n4": 12, "n2": 0, "n1": 0, "N": 12, "t_cloud": 21.21, "score": 1429.02},
                 "S1": {"n4": 0, "n2": 0, "n1": 50, "N": 50, "t_cloud": 22.11, "score": 1615.39},
                 "S2": {"n4": 0, "n2": 0, "n1": 50, "N": 50, "t_cloud": 22.11, "score": 1615.39},
        },
        83_600: {"actual_budget": 82_944,
                 "B1": {"n4": 0, "n2": 0, "n1": 81, "N": 81, "t_cloud": 23.27, "score": 1694.45},
                 "B2": {"n4": 0, "n2": 40, "n1": 0, "N": 40, "t_cloud": 21.43, "score": 1639.41},
                 "B4": {"n4": 20, "n2": 0, "n1": 0, "N": 20, "t_cloud": 20.98, "score": 1508.69},
                 "S1": {"n4": 0, "n2": 0, "n1": 81, "N": 81, "t_cloud": 23.27, "score": 1694.45},
                 "S2": {"n4": 0, "n2": 0, "n1": 81, "N": 81, "t_cloud": 23.27, "score": 1694.45},
        },
        134_200: {"actual_budget": 134_144,
                 "B1": {"n4": 0, "n2": 0, "n1": 131, "N": 131, "t_cloud": 28.22, "score": 1676.07},
                 "B2": {"n4": 0, "n2": 65, "n1": 0, "N": 65, "t_cloud": 22.83, "score": 1698.16},
                 "B4": {"n4": 32, "n2": 0, "n1": 0, "N": 32, "t_cloud": 21.4, "score": 1631.68},
                 "S1": {"n4": 1, "n2": 21, "n1": 85, "N": 107, "t_cloud": 26.76, "score": 1707.0},
                 "S2": {"n4": 1, "n2": 27, "n1": 73, "N": 101, "t_cloud": 26.66, "score": 1701.42},
        },
        215_400: {"actual_budget": 215_040,
                 "B1": {"n4": 0, "n2": 0, "n1": 210, "N": 210, "t_cloud": 31.48, "score": 1740.93},
                 "B2": {"n4": 0, "n2": 105, "n1": 0, "N": 105, "t_cloud": 26.63, "score": 1713.43},
                 "B4": {"n4": 52, "n2": 0, "n1": 0, "N": 52, "t_cloud": 22.17, "score": 1657.1},
                 "S1": {"n4": 8, "n2": 55, "n1": 68, "N": 131, "t_cloud": 28.22, "score": 1724.06},
                 "S2": {"n4": 10, "n2": 70, "n1": 30, "N": 110, "t_cloud": 26.71, "score": 1727.86},
        },
        346_200: {"actual_budget": 346_112,
                 "B1": {"n4": 0, "n2": 0, "n1": 338, "N": 338, "t_cloud": 42.68, "score": 1751.85},
                 "B2": {"n4": 0, "n2": 169, "n1": 0, "N": 169, "t_cloud": 32.36, "score": 1735.33},
                 "B4": {"n4": 84, "n2": 0, "n1": 0, "N": 84, "t_cloud": 23.14, "score": 1703.2},
                 "S1": {"n4": 2, "n2": 77, "n1": 176, "N": 255, "t_cloud": 39.93, "score": 1784.55},
                 "S2": {"n4": 3, "n2": 98, "n1": 130, "N": 231, "t_cloud": 39.42, "score": 1774.19},
        },
        508_900: {"actual_budget": 507_904,
                 "B1": {"n4": 0, "n2": 0, "n1": 496, "N": 496, "t_cloud": 60.96, "score": 1751.98},
                 "B2": {"n4": 0, "n2": 248, "n1": 0, "N": 248, "t_cloud": 38.57, "score": 1760.71},
                 "B4": {"n4": 124, "n2": 0, "n1": 0, "N": 124, "t_cloud": 27.19, "score": 1742.13},
                 "S1": {"n4": 4, "n2": 111, "n1": 258, "N": 373, "t_cloud": 53.15, "score": 1772.71},
                 "S2": {"n4": 5, "n2": 129, "n1": 218, "N": 352, "t_cloud": 50.33, "score": 1806.96},
        },
    },
    "POPE": {
        32_400: {"actual_budget": 31_744,
                 "B1": {"n4": 0, "n2": 0, "n1": 31, "N": 31, "t_cloud": 21.35, "score": 0.7832},
                 "B2": {"n4": 0, "n2": 15, "n1": 0, "N": 15, "t_cloud": 21.05, "score": 0.6574},
                 "B4": {"n4": 7, "n2": 0, "n1": 0, "N": 7, "t_cloud": 21.1, "score": 0.4573},
                 "S1": {"n4": 0, "n2": 2, "n1": 27, "N": 29, "t_cloud": 21.34, "score": 0.7786},
        },
        52_000: {"actual_budget": 51_200,
                 "B1": {"n4": 0, "n2": 0, "n1": 50, "N": 50, "t_cloud": 22.11, "score": 0.823},
                 "B2": {"n4": 0, "n2": 25, "n1": 0, "N": 25, "t_cloud": 21.39, "score": 0.7704},
                 "B4": {"n4": 12, "n2": 0, "n1": 0, "N": 12, "t_cloud": 21.21, "score": 0.6103},
                 "S1": {"n4": 0, "n2": 0, "n1": 50, "N": 50, "t_cloud": 22.11, "score": 0.823},
        },
        83_600: {"actual_budget": 82_944,
                 "B1": {"n4": 0, "n2": 0, "n1": 81, "N": 81, "t_cloud": 23.27, "score": 0.8394},
                 "B2": {"n4": 0, "n2": 40, "n1": 0, "N": 40, "t_cloud": 21.43, "score": 0.8141},
                 "B4": {"n4": 20, "n2": 0, "n1": 0, "N": 20, "t_cloud": 20.98, "score": 0.7243},
                 "S1": {"n4": 0, "n2": 0, "n1": 81, "N": 81, "t_cloud": 23.27, "score": 0.8394},
        },
        134_200: {"actual_budget": 134_144,
                 "B1": {"n4": 0, "n2": 0, "n1": 131, "N": 131, "t_cloud": 28.22, "score": 0.8379},
                 "B2": {"n4": 0, "n2": 65, "n1": 0, "N": 65, "t_cloud": 22.83, "score": 0.8444},
                 "B4": {"n4": 32, "n2": 0, "n1": 0, "N": 32, "t_cloud": 21.4, "score": 0.7988},
                 "S1": {"n4": 1, "n2": 21, "n1": 85, "N": 107, "t_cloud": 26.76, "score": 0.8372},
        },
        215_400: {"actual_budget": 215_040,
                 "B1": {"n4": 0, "n2": 0, "n1": 210, "N": 210, "t_cloud": 31.48, "score": 0.8549},
                 "B2": {"n4": 0, "n2": 105, "n1": 0, "N": 105, "t_cloud": 26.63, "score": 0.842},
                 "B4": {"n4": 52, "n2": 0, "n1": 0, "N": 52, "t_cloud": 22.17, "score": 0.8363},
                 "S1": {"n4": 8, "n2": 55, "n1": 68, "N": 131, "t_cloud": 28.22, "score": 0.8401},
        },
        346_200: {"actual_budget": 346_112,
                 "B1": {"n4": 0, "n2": 0, "n1": 338, "N": 338, "t_cloud": 42.68, "score": 0.8535},
                 "B2": {"n4": 0, "n2": 169, "n1": 0, "N": 169, "t_cloud": 32.36, "score": 0.8536},
                 "B4": {"n4": 84, "n2": 0, "n1": 0, "N": 84, "t_cloud": 23.14, "score": 0.8493},
                 "S1": {"n4": 2, "n2": 77, "n1": 176, "N": 255, "t_cloud": 39.93, "score": 0.8596},
        },
        508_900: {"actual_budget": 507_904,
                 "B1": {"n4": 0, "n2": 0, "n1": 496, "N": 496, "t_cloud": 60.96, "score": 0.8543},
                 "B2": {"n4": 0, "n2": 248, "n1": 0, "N": 248, "t_cloud": 38.57, "score": 0.8577},
                 "B4": {"n4": 124, "n2": 0, "n1": 0, "N": 124, "t_cloud": 27.19, "score": 0.842},
                 "S1": {"n4": 4, "n2": 111, "n1": 258, "N": 373, "t_cloud": 53.15, "score": 0.8568},
        },
    },
    "TextVQA": {
        32_400: {"actual_budget": 31_744,
                 "B1": {"n4": 0, "n2": 0, "n1": 31, "N": 31, "t_cloud": 21.35, "score": 50.57},
                 "B2": {"n4": 0, "n2": 15, "n1": 0, "N": 15, "t_cloud": 21.05, "score": 48.84},
                 "B4": {"n4": 7, "n2": 0, "n1": 0, "N": 7, "t_cloud": 21.1, "score": 45.11},
                 "S1": {"n4": 0, "n2": 2, "n1": 27, "N": 29, "t_cloud": 21.34, "score": 50.94},
        },
        52_000: {"actual_budget": 51_200,
                 "B1": {"n4": 0, "n2": 0, "n1": 50, "N": 50, "t_cloud": 22.11, "score": 51.85},
                 "B2": {"n4": 0, "n2": 25, "n1": 0, "N": 25, "t_cloud": 21.39, "score": 51.41},
                 "B4": {"n4": 12, "n2": 0, "n1": 0, "N": 12, "t_cloud": 21.21, "score": 48.12},
                 "S1": {"n4": 0, "n2": 0, "n1": 50, "N": 50, "t_cloud": 22.11, "score": 51.85},
        },
        83_600: {"actual_budget": 82_944,
                 "B1": {"n4": 0, "n2": 0, "n1": 81, "N": 81, "t_cloud": 23.27, "score": 52.72},
                 "B2": {"n4": 0, "n2": 40, "n1": 0, "N": 40, "t_cloud": 21.43, "score": 52.44},
                 "B4": {"n4": 20, "n2": 0, "n1": 0, "N": 20, "t_cloud": 20.98, "score": 50.7},
                 "S1": {"n4": 0, "n2": 0, "n1": 81, "N": 81, "t_cloud": 23.27, "score": 52.72},
        },
        134_200: {"actual_budget": 134_144,
                 "B1": {"n4": 0, "n2": 0, "n1": 131, "N": 131, "t_cloud": 28.22, "score": 53.13},
                 "B2": {"n4": 0, "n2": 65, "n1": 0, "N": 65, "t_cloud": 22.83, "score": 53.37},
                 "B4": {"n4": 32, "n2": 0, "n1": 0, "N": 32, "t_cloud": 21.4, "score": 52.14},
                 "S1": {"n4": 1, "n2": 21, "n1": 85, "N": 107, "t_cloud": 26.76, "score": 53.57},
        },
        215_400: {"actual_budget": 215_040,
                 "B1": {"n4": 0, "n2": 0, "n1": 210, "N": 210, "t_cloud": 31.48, "score": 53.67},
                 "B2": {"n4": 0, "n2": 105, "n1": 0, "N": 105, "t_cloud": 26.63, "score": 54.05},
                 "B4": {"n4": 52, "n2": 0, "n1": 0, "N": 52, "t_cloud": 22.17, "score": 53.43},
                 "S1": {"n4": 8, "n2": 55, "n1": 68, "N": 131, "t_cloud": 28.22, "score": 54.36},
        },
        346_200: {"actual_budget": 346_112,
                 "B1": {"n4": 0, "n2": 0, "n1": 338, "N": 338, "t_cloud": 42.68, "score": 53.84},
                 "B2": {"n4": 0, "n2": 169, "n1": 0, "N": 169, "t_cloud": 32.36, "score": 54.51},
                 "B4": {"n4": 84, "n2": 0, "n1": 0, "N": 84, "t_cloud": 23.14, "score": 54.05},
                 "S1": {"n4": 2, "n2": 77, "n1": 176, "N": 255, "t_cloud": 39.93, "score": 54.51},
        },
        508_900: {"actual_budget": 507_904,
                 "B1": {"n4": 0, "n2": 0, "n1": 496, "N": 496, "t_cloud": 60.96, "score": 53.7},
                 "B2": {"n4": 0, "n2": 248, "n1": 0, "N": 248, "t_cloud": 38.57, "score": 54.79},
                 "B4": {"n4": 124, "n2": 0, "n1": 0, "N": 124, "t_cloud": 27.19, "score": 54.47},
                 "S1": {"n4": 4, "n2": 111, "n1": 258, "N": 373, "t_cloud": 53.15, "score": 54.76},
        },
    },
    "MMBench": {
        32_400: {"actual_budget": 31_744,
                 "B1": {"n4": 0, "n2": 0, "n1": 31, "N": 31, "t_cloud": 21.35, "score": 67.01},
                 "B2": {"n4": 0, "n2": 15, "n1": 0, "N": 15, "t_cloud": 21.05, "score": 63.66},
                 "B4": {"n4": 7, "n2": 0, "n1": 0, "N": 7, "t_cloud": 21.1, "score": 54.6},
                 "S1": {"n4": 0, "n2": 2, "n1": 27, "N": 29, "t_cloud": 21.34, "score": 67.18},
        },
        52_000: {"actual_budget": 51_200,
                 "B1": {"n4": 0, "n2": 0, "n1": 50, "N": 50, "t_cloud": 22.11, "score": 68.83},
                 "B2": {"n4": 0, "n2": 25, "n1": 0, "N": 25, "t_cloud": 21.39, "score": 67.65},
                 "B4": {"n4": 12, "n2": 0, "n1": 0, "N": 12, "t_cloud": 21.21, "score": 61.63},
                 "S1": {"n4": 0, "n2": 0, "n1": 50, "N": 50, "t_cloud": 22.11, "score": 68.83},
        },
        83_600: {"actual_budget": 82_944,
                 "B1": {"n4": 0, "n2": 0, "n1": 81, "N": 81, "t_cloud": 23.27, "score": 70.45},
                 "B2": {"n4": 0, "n2": 40, "n1": 0, "N": 40, "t_cloud": 21.43, "score": 69.39},
                 "B4": {"n4": 20, "n2": 0, "n1": 0, "N": 20, "t_cloud": 20.98, "score": 66.39},
                 "S1": {"n4": 0, "n2": 0, "n1": 81, "N": 81, "t_cloud": 23.27, "score": 70.45},
        },
        134_200: {"actual_budget": 134_144,
                 "B1": {"n4": 0, "n2": 0, "n1": 131, "N": 131, "t_cloud": 28.22, "score": 71.1},
                 "B2": {"n4": 0, "n2": 65, "n1": 0, "N": 65, "t_cloud": 22.83, "score": 70.78},
                 "B4": {"n4": 32, "n2": 0, "n1": 0, "N": 32, "t_cloud": 21.4, "score": 68.78},
                 "S1": {"n4": 1, "n2": 21, "n1": 85, "N": 107, "t_cloud": 26.76, "score": 71.5},
        },
        215_400: {"actual_budget": 215_040,
                 "B1": {"n4": 0, "n2": 0, "n1": 210, "N": 210, "t_cloud": 31.48, "score": 71.58},
                 "B2": {"n4": 0, "n2": 105, "n1": 0, "N": 105, "t_cloud": 26.63, "score": 72.11},
                 "B4": {"n4": 52, "n2": 0, "n1": 0, "N": 52, "t_cloud": 22.17, "score": 70.19},
                 "S1": {"n4": 8, "n2": 55, "n1": 68, "N": 131, "t_cloud": 28.22, "score": 71.62},
        },
        346_200: {"actual_budget": 346_112,
                 "B1": {"n4": 0, "n2": 0, "n1": 338, "N": 338, "t_cloud": 42.68, "score": 71.83},
                 "B2": {"n4": 0, "n2": 169, "n1": 0, "N": 169, "t_cloud": 32.36, "score": 72.51},
                 "B4": {"n4": 84, "n2": 0, "n1": 0, "N": 84, "t_cloud": 23.14, "score": 71.69},
                 "S1": {"n4": 2, "n2": 77, "n1": 176, "N": 255, "t_cloud": 39.93, "score": 72.23},
        },
        508_900: {"actual_budget": 507_904,
                 "B1": {"n4": 0, "n2": 0, "n1": 496, "N": 496, "t_cloud": 60.96, "score": 71.66},
                 "B2": {"n4": 0, "n2": 248, "n1": 0, "N": 248, "t_cloud": 38.57, "score": 72.72},
                 "B4": {"n4": 124, "n2": 0, "n1": 0, "N": 124, "t_cloud": 27.19, "score": 72.24},
                 "S1": {"n4": 4, "n2": 111, "n1": 258, "N": 373, "t_cloud": 53.15, "score": 72.49},
        },
    },
}

# Maps DATA's per-benchmark method keys -> MIDPOINT_DATA's per-budget strategy
# keys (B1/B2/B4 are the same uniform-bit-depth definition as SP-Q1/Q2/Q4;
# VQAv2-opt continues via S1, the final snap-and-refill strategy).
MIDPOINT_METHOD_KEY = {"SP-Q1": "B1", "SP-Q2": "B2", "SP-Q4": "B4", "VQAv2-opt": "S1"}
