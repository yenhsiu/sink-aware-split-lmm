"""
quantization_search.py

Optuna ask-and-tell search for optimal (n4, n2, n1) token allocation.
Each trial runs evaluation via validate.py, then reports the score back.

Constraints:
  4·n4 + 2·n2 + n1 = B // 1024   (equality: always use full budget)
  n4 + n2 + n1 ≤ 576              (token cap)
  n4, n2, n1 ≥ 0

Sampling: (n4, n2) is searched directly as integers (n1 absorbs the exact
budget residual), instead of the old continuous two-cut-point parameterization.
The continuous version rounded (u1, u2) into integer (n4, n2), which is a
many-to-one mapping -- for small budgets in particular, large swaths of the
continuous space collapsed onto the same discrete point, so a large fraction
of TPE's trial budget was wasted re-sampling configs already tried (visible
as identical repeated entries in the Top-5 summary). Searching n4/n2 directly
means every trial proposes a genuinely distinct integer config. The three
uniform-bit-width corners (All B=1/2/4) are still guaranteed reachable because
they're explicitly enqueued as anchors (see get_anchor_points), independent of
whatever the sampler's own marginal distribution over (n4, n2) looks like.

Each budget also gets 3-4 anchor points (All B=1, All B=2, All B=4, and --
when the budget supports averaging >=1 bit/token across all 576 tokens --
the "N=576, no pruning" mixed-bit point) explicitly enqueued as the first
trials, so the search can never do worse than a baseline it never tried.

n1 is always the residual that exactly absorbs the budget equation:
n1 = (B // 1024) - 4·n4 - 2·n2

Supported datasets: mme, textvqa, pope, scienceqa, mmbench, vqav2

Usage:
    python quantization_search.py \
        --dataset mme \
        --budget 115500 \
        --n-trials 50 \
        --cuda 0
"""

import os
import re
import sys
import json
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

import optuna
VALIDATE_PY = Path(__file__).parent.resolve() / "validate.py"

# Score parsers per dataset — each returns a single float from validate.py stdout
SCORE_PARSERS = {
    "mme":        lambda s: float(re.search(r"Combined Total Score:\s*([\d.]+)", s).group(1))
                            if re.search(r"Combined Total Score:\s*([\d.]+)", s) else 0.0,
    "textvqa":    lambda s: float(re.search(r"Accuracy:\s*([\d.]+)%", s).group(1))
                            if re.search(r"Accuracy:\s*([\d.]+)%", s) else 0.0,
    "pope":       lambda s: sum(float(x) for x in re.findall(r"F1 score:\s*([\d.]+)", s)) /
                            max(len(re.findall(r"F1 score:\s*([\d.]+)", s)), 1),
    "scienceqa":  lambda s: float(re.search(r"IMG-Accuracy:\s*([\d.]+)%", s).group(1))
                            if re.search(r"IMG-Accuracy:\s*([\d.]+)%", s) else 0.0,
    "mmbench":    lambda s: float(re.search(r"MMBench Dev Accuracy:\s*([\d.]+)%", s).group(1))
                            if re.search(r"MMBench Dev Accuracy:\s*([\d.]+)%", s) else 0.0,
    "vqav2":      lambda s: float(re.search(r"Overall Accuracy:\s*([\d.]+)%", s).group(1))
                            if re.search(r"Overall Accuracy:\s*([\d.]+)%", s) else 0.0,
}

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Config ────────────────────────────────────────────────────────────────────

PYTHON      = sys.executable
PROJECT_ROOT = Path(__file__).parent.resolve()



# ── Evaluation via validate.py subprocess ────────────────────────────────────

def run_eval(dataset: str, n4: int, n2: int, n1: int, exp_name: str, cuda: str,
             split: str = "search", split_ratio: float = 0.3) -> float:
    """Run validate.py for (n4, n2, n1) and parse the scalar score."""
    cmd = [PYTHON, str(VALIDATE_PY),
           "--dataset", dataset,
           "--n4", str(n4), "--n2", str(n2), "--n1", str(n1),
           "--split", split, "--split-ratio", str(split_ratio),
           "--cuda", cuda]
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True, text=True,
    )
    output = result.stdout + result.stderr
    if result.returncode != 0:
        print(f"  [warn] validate.py failed for {exp_name}\n{output[-300:]}")
        return 0.0

    parser = SCORE_PARSERS.get(dataset)
    if parser is None:
        print(f"  [warn] no score parser for dataset={dataset}")
        return 0.0

    try:
        score = parser(output)
    except Exception as e:
        print(f"  [warn] score parse error: {e}\n{output[-300:]}")
        return 0.0

    if score == 0.0:
        print(f"  [warn] parsed score=0.0, raw output:\n{output[-300:]}")
    return score


# ── Optuna ask-and-tell ───────────────────────────────────────────────────────

N_MAX = 576  # CLIP patch count -- hard cap on n4+n2+n1


def sample_params(trial: optuna.Trial, S: int):
    """Directly search n4, n2 as integers; n1 absorbs the exact budget
    residual so 4n4+2n2+n1==T always holds. n2's range depends on n4 (less
    budget left after n4 is spent), which is fine -- Optuna supports
    dynamic/conditional integer ranges within a single trial."""
    T = S // 1024
    n4 = trial.suggest_int("n4", 0, T // 4)
    rem = T - 4 * n4
    n2 = trial.suggest_int("n2", 0, rem // 2)
    n1 = rem - 2 * n2
    return n4, n2, n1


def is_feasible(n4: int, n2: int, n1: int) -> bool:
    N = n4 + n2 + n1
    return n1 >= 0 and N > 0 and N <= N_MAX


def get_anchor_points(S: int) -> list:
    """All-B1 / All-B2 / All-B4, plus (when the budget allows an average of
    at least 1 bit/token across all N_MAX tokens) the "no pruning" point --
    each only included if it's actually feasible at this exact budget."""
    T = S // 1024
    anchors = []

    if T <= N_MAX:  # else n1=T > N_MAX tokens needed: infeasible, must skip
        anchors.append({"n4": 0, "n2": 0, "n1": T})
    if T <= 2 * N_MAX:
        anchors.append({"n4": 0, "n2": T // 2, "n1": T - 2 * (T // 2)})
    if T <= 4 * N_MAX:
        anchors.append({"n4": T // 4, "n2": 0, "n1": T - 4 * (T // 4)})

    # N=576 (no pruning), 1-bit/2-bit mix -- only exact for T in [N_MAX, 2*N_MAX]
    if N_MAX <= T <= 2 * N_MAX:
        n2 = T - N_MAX
        n1 = N_MAX - n2
        anchors.append({"n4": 0, "n2": n2, "n1": n1})
    # NOTE: for T > 2*N_MAX, the analogous "N=576, 2-bit/4-bit mix" point
    # would need a 3-way solver (not yet implemented) -- deliberately
    # skipped rather than emitting a point that silently underspends budget.

    feasible = [a for a in anchors if is_feasible(a["n4"], a["n2"], a["n1"])]
    deduped = list({(a["n4"], a["n2"], a["n1"]): a for a in feasible}.values())
    return deduped


def anchor_to_params(n4: int, n2: int) -> dict:
    """Param dict for study.enqueue_trial matching sample_params' own
    suggest_int names, so an anchor (n4, n2, n1) enqueues exactly."""
    return {"n4": n4, "n2": n2}


def enumerate_feasible(S: int) -> list:
    """Every feasible (n4, n2, n1) triple for budget S. For small budgets
    this is cheap enough to just try all of them instead of TPE-searching a
    space that's smaller than the trial budget anyway (see search())."""
    T = S // 1024
    result = []
    for n4 in range(T // 4 + 1):
        rem = T - 4 * n4
        for n2 in range(rem // 2 + 1):
            n1 = rem - 2 * n2
            if is_feasible(n4, n2, n1):
                result.append({"n4": n4, "n2": n2, "n1": n1})
    return result



def search(dataset: str, S: int, n_trials: int, cuda: str, split: str = "all",
           split_ratio: float = 0.3, seed: int = 42) -> dict:
    anchors = get_anchor_points(S)
    anchor_keys = {(a["n4"], a["n2"], a["n1"]) for a in anchors}

    feasible = enumerate_feasible(S)
    exhaustive = len(feasible) <= n_trials
    if exhaustive:
        print(f"  [exhaustive] feasible space = {len(feasible)} <= n_trials = {n_trials} "
              f"-> trying every feasible config once instead of TPE, guaranteed optimum")
        return _search_exhaustive(dataset, S, feasible, anchor_keys, cuda, split, split_ratio)

    # Always a fresh study -- direct (n4, n2) is a different parameter space
    # than the old (u1, u2) sampler, so there is nothing meaningful to resume.
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )

    for a in anchors:
        study.enqueue_trial(anchor_to_params(a["n4"], a["n2"]))
    print(f"  [anchors] {len(anchors)} enqueued: {anchors}")

    history = []  # full per-trial record, incl. pruned
    trial_count = 0
    while trial_count < n_trials:
        trial = study.ask()
        n4, n2, n1 = sample_params(trial, S)
        is_anchor = (n4, n2, n1) in anchor_keys

        if not is_feasible(n4, n2, n1):
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            history.append({"trial": trial_count,
                             "n4": n4, "n2": n2, "n1": n1, "N": n4 + n2 + n1,
                             "source": "anchor" if is_anchor else "tpe",
                             "score": None, "state": "pruned"})
            continue

        source = "anchor" if is_anchor else "tpe"
        exp_name = f"search_{dataset}_S{S}_t{trial_count:03d}_n4{n4}_n2{n2}_n1{n1}"
        print(f"\n  trial {trial_count:3d} [{source}]: n4={n4:3d}, n2={n2:3d}, n1={n1:3d}, "
              f"N={n4+n2+n1:3d}  →  {exp_name}")

        score = run_eval(dataset, n4, n2, n1, exp_name, cuda, split=split, split_ratio=split_ratio)
        print(f"           score = {score:.4f}")

        study.tell(trial, score)
        history.append({"trial": trial_count,
                         "n4": n4, "n2": n2, "n1": n1, "N": n4 + n2 + n1,
                         "source": source, "score": score, "state": "complete"})
        trial_count += 1

    return _summarize(dataset, S, anchors, history)


def _search_exhaustive(dataset: str, S: int, feasible: list, anchor_keys: set,
                        cuda: str, split: str, split_ratio: float) -> dict:
    """Try every feasible (n4, n2, n1) once, in a fixed order (anchors
    first, for a familiar log/history shape) -- no optimizer needed since
    we're covering the whole space anyway, so this is guaranteed to find
    the true optimum for the given eval split, not just a TPE estimate."""
    ordered = sorted(feasible, key=lambda a: (a["n4"], a["n2"], a["n1"]) not in anchor_keys)
    history = []
    for trial_count, cfg in enumerate(ordered):
        n4, n2, n1 = cfg["n4"], cfg["n2"], cfg["n1"]
        source = "anchor" if (n4, n2, n1) in anchor_keys else "exhaustive"
        exp_name = f"search_{dataset}_S{S}_t{trial_count:03d}_n4{n4}_n2{n2}_n1{n1}"
        print(f"\n  trial {trial_count:3d}/{len(ordered)} [{source}]: n4={n4:3d}, n2={n2:3d}, n1={n1:3d}, "
              f"N={n4+n2+n1:3d}  →  {exp_name}")

        score = run_eval(dataset, n4, n2, n1, exp_name, cuda, split=split, split_ratio=split_ratio)
        print(f"           score = {score:.4f}")

        history.append({"trial": trial_count,
                         "n4": n4, "n2": n2, "n1": n1, "N": n4 + n2 + n1,
                         "source": source, "score": score, "state": "complete"})

    anchors = [{"n4": a["n4"], "n2": a["n2"], "n1": a["n1"]} for a in feasible
               if (a["n4"], a["n2"], a["n1"]) in anchor_keys]
    return _summarize(dataset, S, anchors, history)


def _summarize(dataset: str, S: int, anchors: list, history: list) -> dict:
    """Shared by both TPE search and exhaustive search -- from our own
    history, not study.best_trial, since history already has n4/n2/n1
    recorded directly."""
    completed = [h for h in history if h["state"] == "complete"]
    best = max(completed, key=lambda h: h["score"])
    n4, n2, n1, N = best["n4"], best["n2"], best["n1"], best["N"]

    print(f"\n[Best] n4={n4}, n2={n2}, n1={n1}, N={N}, source={best['source']}, "
          f"top_ratio={n4/N:.3f}, score={best['score']:.4f}")
    print("[Top 5]")
    for h in sorted(completed, key=lambda h: h["score"], reverse=True)[:5]:
        print(f"  n4={h['n4']:3d}, n2={h['n2']:3d}, n1={h['n1']:3d}, N={h['N']:3d}, "
              f"source={h['source']:>6}, top_ratio={h['n4']/h['N']:.3f}, score={h['score']:.4f}")

    return {"dataset": dataset, "S": S, "n4": n4, "n2": n2, "n1": n1, "N": N,
            "top_ratio": n4 / N if N > 0 else 0.0, "score": best["score"],
            "anchors": anchors, "trials": history}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",     default="mme",
                        choices=["mme", "textvqa", "pope", "scienceqa", "mmbench", "vqav2"])
    parser.add_argument("--budgets",     nargs="+", type=int, default=[115500, 266240, 589824])
    parser.add_argument("--n-trials",   type=int,   default=50)
    parser.add_argument("--split",       default="all", choices=["all", "search", "val"],
                        help="Which subset of the dataset each trial evaluates on (default: all)")
    parser.add_argument("--split-ratio", type=float, default=0.3,
                        help="Fraction of data used as search set when --split=search/val (default: 0.3)")
    parser.add_argument("--cuda",        default="0")
    parser.add_argument("--output",      default=None,
                        help="Path to save results JSON (default: results/search_<dataset>_<timestamp>.json)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for Optuna TPE sampler (default: 42)")
    args = parser.parse_args()

    results = []
    for S in args.budgets:
        print(f"\n{'='*40}\nDataset: {args.dataset}  Budget: {S}\n{'='*40}")
        results.append(search(args.dataset, S, args.n_trials, args.cuda, split=args.split,
                               split_ratio=args.split_ratio, seed=args.seed))

    print("\n\n=== Search Summary ===")
    import pandas as pd
    summary_only = [{k: v for k, v in r.items() if k not in ("anchors", "trials")} for r in results]
    print(pd.DataFrame(summary_only).set_index(["dataset", "S"]).to_string())

    # ── Save results ──
    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    if args.output:
        out_path = Path(args.output)
    else:
        budgets_tag = "_".join(str(S) for S in args.budgets)
        seed_tag = f"_seed{args.seed}" if args.seed != 42 else ""
        out_path = out_dir / f"search_{args.dataset}_{budgets_tag}{seed_tag}.json"

    out_path.write_text(json.dumps({
        "dataset": args.dataset,
        "n_trials": args.n_trials,
        "split_ratio": args.split_ratio,
        "tpe_seed": args.seed,
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }, indent=2))
    print(f"\n[Saved] {out_path}")


if __name__ == "__main__":
    main()
