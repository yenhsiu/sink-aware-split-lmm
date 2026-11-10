#!/usr/bin/env python3
"""
deadline_robustness.py -- robustness of the 350 ms deadline to bandwidth
over-estimation (paper, Sec. "Robustness of the Deadline Guarantee").

Uses the measured latency model (paper/paper_data.py) and the runtime
round-down-and-refill selection (paper/plot_paper_figures.py). The planner
sees C_est = (1 + eps) * C_true; the realized latency uses C_true. With a
safety margin the planner instead uses C_est / (1 + eps). CPU only.

    python analysis/deadline_robustness.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "paper"))
import paper_data as P  # noqa: E402
import plot_paper_figures as F  # noqa: E402

GRID = np.logspace(np.log10(0.35), np.log10(20), 400)  # Mbps range where the deadline is attainable


def violations(eps, margin=False):
    n_viol, worst = 0, 0.0
    for c in GRID:
        c_plan = c * (1 + eps) / ((1 + eps) if margin else 1.0)
        tx_bits, _, t_edge, t_cloud, t_quant = F.dynamic_snap_down_refill(c_plan)
        t = t_edge + t_quant + t_cloud + tx_bits / (c * 1000)
        if t > P.T_DEADLINE + 1e-9:
            n_viol += 1
            worst = max(worst, t - P.T_DEADLINE)
    return 100 * n_viol / len(GRID), worst


if __name__ == "__main__":
    print(f"deadline {P.T_DEADLINE} ms, bandwidth grid {GRID[0]:.2f}-{GRID[-1]:.0f} Mbps ({len(GRID)} points)")
    for eps in (0.0, 0.03, 0.05, 0.10, 0.20):
        v, w = violations(eps)
        vm, wm = violations(eps, margin=True)
        print(f"over-estimate {eps:4.0%}: violations {v:5.1f}% of range, worst +{w:5.1f} ms"
              f" | with margin: {vm:.1f}%, worst +{wm:.1f} ms")
    for c in (0.5, 1.0, 2.0):
        print(f"C={c} Mbps: N_tx without margin {F.dynamic_snap_down_refill(c)[1]},"
              f" with 10% margin {F.dynamic_snap_down_refill(0.9 * c)[1]}")
    print(f"IPv4+UDP headers per 1500-byte packet: {100 * 28 / 1472:.1f}%")
