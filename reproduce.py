#!/usr/bin/env python3
"""
reproduce.py -- one entry point for every accuracy experiment in the paper.

Each experiment expands into a list of `validate.py` invocations (one per
benchmark x budget x method x PCA seed). Runs are skipped if a saved result
for the exact same experiment name already exists in results/, so an
interrupted sweep can simply be restarted.

    python reproduce.py main          # Main results: SP-Q1/Q2/Q4 vs Proposed, 4 benchmarks x 8 budgets x 3 seeds
    python reproduce.py fp16          # Unmodified LLaVA-1.5 (576 tokens, FP16) reference
    python reproduce.py sink          # Ablation: sink handling (pin / none / capa_rank), MME, 8 budgets x 3 seeds
    python reproduce.py sink-threshold  # Detector sensitivity: ratio 25/50/200 and no top-2 cap, MME, 3 budgets x 3 seeds
    python reproduce.py sink-explore  # Ablation: extra CAPA-style variants (capa / capa_full / value_rank / capa_rank_cap)
    python reproduce.py snap-refill   # Snap-and-refill at the 7 midpoint budgets (S1 vs S2 vs uniform)
    python reproduce.py search        # Re-run the offline VQAv2 LUT search (8 budgets)

Common options:
    --dry-run            print the commands instead of running them
    --cuda 0             GPU index passed to validate.py
    --seeds 42 43 44     PCA seeds (paper: mean over 42/43/44)
    --datasets mme pope  restrict benchmarks
    --budgets 25600 ...  restrict budgets

After a sweep finishes, compare against the paper's numbers with:
    python compare_results.py
"""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT / "paper"))
from paper_data import ANCHOR_CONFIGS, BUDGETS, SNAP_REFILL_POINTS  # noqa: E402

RESULTS_DIR = ROOT / "results"
SEEDS = [42, 43, 44]
MAIN_DATASETS = ["mme", "pope", "textvqa", "mmbench"]


# ── Method -> (n4, n2, n1) ────────────────────────────────────────────────────

def config_for(method: str, budget: int):
    """Tier sizes for a method at a transmission budget (bits).

    Budget accounting used throughout the paper: 4*n4 + 2*n2 + n1 = budget // 1024.
    SP-Qb are *pure* single-bit-width baselines (any leftover budget that
    does not fill a whole b-bit token is dropped, not padded with 1-bit).
    """
    T = budget // 1024
    if method == "SP-Q1":
        return 0, 0, min(T, 576)
    if method == "SP-Q2":
        return 0, min(T // 2, 576), 0
    if method == "SP-Q4":
        return min(T // 4, 576), 0, 0
    if method == "Proposed":
        c = ANCHOR_CONFIGS[budget]
        return c["n4"], c["n2"], c["n1"]
    raise ValueError(method)


def validate_cmd(dataset, n4, n2, n1, seed, extra=()):
    return [sys.executable, str(ROOT / "validate.py"), "--dataset", dataset,
            "--n4", str(n4), "--n2", str(n2), "--n1", str(n1),
            "--pca-seed", str(seed), "--split", "all", "--save", *extra]


def exp_name(dataset, n4, n2, n1, seed, sink_tag=""):
    """Mirror validate.py's result-file naming so finished runs are skipped."""
    seed_tag = f"_pcaseed{seed}" if seed != 42 else ""
    return f"validate_{dataset}_n4{n4}_n2{n2}_n1{n1}{sink_tag}{seed_tag}"


# ── Experiment definitions ────────────────────────────────────────────────────

def jobs_main(a):
    for ds in a.datasets or MAIN_DATASETS:
        for B in a.budgets or BUDGETS:
            for m in ["SP-Q1", "SP-Q2", "SP-Q4", "Proposed"]:
                n4, n2, n1 = config_for(m, B)
                for s in a.seeds:
                    yield exp_name(ds, n4, n2, n1, s), validate_cmd(ds, n4, n2, n1, s)


def jobs_fp16(a):
    for ds in a.datasets or MAIN_DATASETS:
        yield f"validate_{ds}_baseline", [sys.executable, str(ROOT / "validate.py"),
                                          "--dataset", ds, "--baseline", "--split", "all", "--save"]


SINK_VARIANTS = {
    # name: (extra validate.py args, filename tag)
    "pin":           ([], ""),
    "no_sink":       (["--no-sink-filter"], "_nosink"),
    "capa_rank":     (["--sink-mode", "capa_rank"], "_caparank"),
    "capa":          (["--sink-mode", "capa"], "_capa"),
    "capa_full":     (["--sink-mode", "capa_full"], "_capafull"),
    "value_rank":    (["--sink-mode", "value_rank"], "_valuerank"),
    "capa_rank_cap": (["--sink-mode", "capa_rank_cap"], "_caprankcap"),
    # detector sensitivity (paper: "Sink Statistics and Detector Sensitivity")
    "ratio25":       (["--sink-ratio", "25"], "_ratio25"),
    "ratio50":       (["--sink-ratio", "50"], "_ratio50"),
    "ratio200":      (["--sink-ratio", "200"], "_ratio200"),
    "no_cap":        (["--sink-max-candidates", "576"], "_cap576"),
}
THRESHOLD_BUDGETS = [25600, 170000, 589824]


def _jobs_sink(a, variants):
    for ds in a.datasets or ["mme"]:
        for B in a.budgets or BUDGETS:
            n4, n2, n1 = config_for("Proposed", B)
            for v in variants:
                extra, tag = SINK_VARIANTS[v]
                for s in a.seeds:
                    yield exp_name(ds, n4, n2, n1, s, tag), validate_cmd(ds, n4, n2, n1, s, extra)


def jobs_sink(a):
    return _jobs_sink(a, ["pin", "no_sink", "capa_rank"])


def jobs_sink_explore(a):
    return _jobs_sink(a, ["capa", "capa_full", "value_rank", "capa_rank_cap"])


def jobs_sink_threshold(a):
    """Detector sensitivity: threshold ratio and the top-2 candidate cap, at a
    low, middle and high budget (pin / no_sink at these budgets come from `sink`)."""
    if not a.budgets:
        a.budgets = THRESHOLD_BUDGETS
    return _jobs_sink(a, ["ratio25", "ratio50", "ratio200", "no_cap"])


def jobs_snap_refill(a):
    """S1 (snap-and-refill) and S2 (proportional scaling) at the 7 midpoints,
    plus pure uniform baselines at the same midpoint budgets."""
    for ds in a.datasets or MAIN_DATASETS:
        for pt in SNAP_REFILL_POINTS:
            B_mid, B_anch = pt["B_mid"], pt["B_anch"]
            if a.budgets and B_mid not in a.budgets:
                continue
            T_mid = B_mid // 1024
            # Refill uses the *token-budget* difference T(B_mid) - T(B_anch), so the
            # refilled config uses the full budget exactly. This is what the paper's
            # reported midpoint numbers were measured with. (paper_data.SNAP_REFILL_POINTS'
            # "extra" field uses floor((B_mid - B_anch)/1024), which is 1 smaller at
            # 134,200 and 346,200.)
            extra = T_mid - B_anch // 1024
            c = ANCHOR_CONFIGS[B_anch]
            cfgs = {
                "SP-Q1": (0, 0, T_mid),
                "SP-Q2": (0, T_mid // 2, 0),
                "SP-Q4": (T_mid // 4, 0, 0),
                "S1":    (c["n4"], c["n2"], c["n1"] + extra),
            }
            r = T_mid / (B_anch // 1024)
            n4s, n2s = round(c["n4"] * r), round(c["n2"] * r)
            n1s = T_mid - 4 * n4s - 2 * n2s
            if n1s < 0:
                n2s = (T_mid - 4 * n4s) // 2
                n1s = T_mid - 4 * n4s - 2 * n2s
            if ds == "mme":  # S2 was only evaluated on MME in the paper
                cfgs["S2"] = (n4s, n2s, n1s)
            for n4, n2, n1 in cfgs.values():
                for s in a.seeds:
                    yield exp_name(ds, n4, n2, n1, s), validate_cmd(ds, n4, n2, n1, s)


def jobs_search(a):
    budgets = a.budgets or BUDGETS
    # 5,000-question VQAv2 val subset (0.023326 x 214,354), split seed 42.
    yield None, [sys.executable, str(ROOT / "quantization_search.py"),
                 "--dataset", "vqav2", "--split", "search", "--split-ratio", "0.023326",
                 "--n-trials", "50", "--seed", "42",
                 "--budgets", *map(str, budgets)]


EXPERIMENTS = {
    "main": jobs_main, "fp16": jobs_fp16, "sink": jobs_sink, "sink-threshold": jobs_sink_threshold,
    "sink-explore": jobs_sink_explore,
    "snap-refill": jobs_snap_refill, "search": jobs_search,
}


def already_done(name):
    return name is not None and any(RESULTS_DIR.glob(f"{name}_[0-9]*_[0-9]*.txt"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiment", choices=EXPERIMENTS)
    p.add_argument("--cuda", default="0")
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--budgets", nargs="+", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true", help="re-run even if a saved result exists")
    a = p.parse_args()

    jobs, seen = [], set()
    for name, cmd in EXPERIMENTS[a.experiment](a):
        key = name or " ".join(cmd)
        if key in seen:  # e.g. SP-Q1 == Proposed at budgets where the LUT picked all-1-bit
            continue
        seen.add(key)
        jobs.append((name, cmd + ["--cuda", a.cuda]))

    todo = [(n, c) for n, c in jobs if a.force or not already_done(n)]
    print(f"[reproduce] {a.experiment}: {len(jobs)} runs, {len(jobs) - len(todo)} already done, {len(todo)} to run")
    failed = []
    for i, (name, cmd) in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] {shlex.join(cmd)}", flush=True)
        if a.dry_run:
            continue
        if subprocess.run(cmd, cwd=str(ROOT)).returncode != 0:
            failed.append(name or cmd)
    if failed:
        print(f"\n[reproduce] {len(failed)} run(s) failed:")
        for f in failed:
            print("  ", f)
        sys.exit(1)


if __name__ == "__main__":
    main()
