#!/usr/bin/env python3
"""
compare_results.py -- compare your re-run results/ against the paper's numbers.

Reads the result files written by `validate.py --save` (via reproduce.py),
averages over PCA seeds, and prints each cell next to the value reported in
the paper (paper/paper_data.py). Cells that differ by more than the rounding
tolerance are marked with '!!'; cells with no result file yet are marked '--'.

    python compare_results.py                       # all experiments
    python compare_results.py main sink             # selected experiments
    python compare_results.py --results-dir other/  # a different results folder
"""

import argparse
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT / "paper"))
sys.path.insert(0, str(ROOT))
import paper_data as P  # noqa: E402
from quantization_search import SCORE_PARSERS  # noqa: E402
from reproduce import SEEDS, SINK_VARIANTS, config_for, exp_name  # noqa: E402

BENCH = {"mme": "MME", "pope": "POPE", "textvqa": "TextVQA", "mmbench": "MMBench"}
# Paper tables are rounded to these precisions.
TOL = {"mme": 0.01, "pope": 0.0001, "textvqa": 0.01, "mmbench": 0.01}
PAPER_METHOD = {"SP-Q1": "SP-Q1", "SP-Q2": "SP-Q2", "SP-Q4": "SP-Q4", "Proposed": "VQAv2-opt"}


def load_score(results_dir: Path, dataset: str, name: str):
    files = sorted(results_dir.glob(f"{name}_[0-9]*_[0-9]*.txt"))
    if not files:
        return None
    text = files[-1].read_text()  # latest run of this exact experiment
    return SCORE_PARSERS[dataset](text)


def seed_mean(results_dir, dataset, n4, n2, n1, seeds, tag=""):
    vals = [load_score(results_dir, dataset, exp_name(dataset, n4, n2, n1, s, tag)) for s in seeds]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, 0
    return statistics.mean(vals), len(vals)


class Table:
    def __init__(self, title):
        self.title, self.rows, self.n_bad, self.n_missing, self.n_ok = title, [], 0, 0, 0

    def add(self, label, dataset, ours, n, paper):
        if ours is None:
            mark, ours_s, diff_s = "--", "-", ""
            self.n_missing += 1
        else:
            d = ours - paper
            # One rounding unit: paper means were computed from per-seed values
            # already rounded to this precision, so they can differ by < 1 unit.
            bad = abs(d) > TOL[dataset] + 1e-9
            mark = "!!" if bad else "ok"
            self.n_bad += bad
            self.n_ok += not bad
            ours_s, diff_s = f"{ours:.4f}", f"{d:+.4f}"
        self.rows.append(f"  {mark}  {label:<44} paper={paper:<10.4f} ours={ours_s:<10} {diff_s:<10} (seeds={n})")

    def show(self):
        print(f"\n=== {self.title} ===  ok={self.n_ok}  mismatch={self.n_bad}  missing={self.n_missing}")
        print("\n".join(self.rows))
        return self.n_bad


def cmp_main(rd, seeds):
    t = Table("Main results (mean over PCA seeds)")
    for ds, bench in BENCH.items():
        for i, B in enumerate(P.BUDGETS):
            for m, pm in PAPER_METHOD.items():
                n4, n2, n1 = config_for(m, B)
                ours, n = seed_mean(rd, ds, n4, n2, n1, seeds)
                t.add(f"{bench:<7} B={B:<7} {m}", ds, ours, n, P.DATA[bench][pm][i])
    return t.show()


def cmp_fp16(rd, seeds):
    t = Table("Unmodified LLaVA-1.5 (576 tokens, FP16)")
    for ds, bench in BENCH.items():
        t.add(f"{bench}", ds, load_score(rd, ds, f"validate_{ds}_baseline"), 1, P.BASELINE_FP16[bench])
    return t.show()


def cmp_sink(rd, seeds):
    t = Table("Ablation: sink handling, MME (mean over PCA seeds)")
    for B, row in P.ABLATION_SINK_MME_SEEDED.items():
        for v in ("pin", "no_sink", "no_cap", "capa_rank"):
            ours, n = seed_mean(rd, "mme", row["n4"], row["n2"], row["n1"], seeds, SINK_VARIANTS[v][1])
            t.add(f"B={B:<7} {v}", "mme", ours, n, row[v]["mean"])
    return t.show()


def cmp_sink_threshold(rd, seeds):
    t = Table("Sink-detector sensitivity, MME (mean over PCA seeds)")
    for B, row in P.SINK_THRESHOLD_MME.items():
        n4, n2, n1 = config_for("Proposed", B)
        for v, cell in row.items():
            ours, n = seed_mean(rd, "mme", n4, n2, n1, seeds, SINK_VARIANTS[v][1])
            t.add(f"B={B:<7} {v}", "mme", ours, n, cell["mean"])
    return t.show()


def cmp_snap_refill(rd, seeds):
    t = Table("Snap-and-refill midpoints (mean over PCA seeds)")
    for bench_ds, bench in BENCH.items():
        for B_mid, row in P.MIDPOINT_DATA[bench].items():
            for key, cfg in row.items():
                if key == "actual_budget":
                    continue
                ours, n = seed_mean(rd, bench_ds, cfg["n4"], cfg["n2"], cfg["n1"], seeds)
                t.add(f"{bench:<7} B={B_mid:<7} {key} ({cfg['n4']},{cfg['n2']},{cfg['n1']})",
                      bench_ds, ours, n, cfg["score"])
    return t.show()


EXPERIMENTS = {"main": cmp_main, "fp16": cmp_fp16, "sink": cmp_sink, "sink-threshold": cmp_sink_threshold,
               "snap-refill": cmp_snap_refill}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiments", nargs="*", choices=[[], *EXPERIMENTS], default=[])
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    a = ap.parse_args()
    bad = sum(EXPERIMENTS[e](a.results_dir, a.seeds) for e in (a.experiments or EXPERIMENTS))
    print(f"\nTotal mismatching cells: {bad}")


if __name__ == "__main__":
    main()
