"""
validate_snap_refill.py

Snap-and-refill evaluation at 7 midpoint budgets between anchor points.

For each midpoint budget B_mid:
  - All B=1  : n1 = B_mid // 1024
  - All B=2  : n2 = B_mid // 2048
  - All B=4  : n4 = B_mid // 4096
  - Snap+refill: anchor (n4,n2,n1) + extra B=1 tokens
                 where extra = B_mid//1024 - B_anch//1024 (full budget used)

Anchor is the largest S with S <= B_mid. Anchors default to the paper's
VQAv2-searched LUT (paper/paper_data.py: ANCHOR_CONFIGS); pass
--search-results to use your own quantization_search.py output instead.

Usage:
    python validate_snap_refill.py \
        --datasets mme pope textvqa mmbench \
        --cuda 0 --pca-seed 42 \
        --output-dir results/snap_refill
"""

import re
import sys
import json
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
PYTHON       = sys.executable
VALIDATE_PY  = PROJECT_ROOT / "validate.py"
RESULTS_DIR  = PROJECT_ROOT / "results"

sys.path.insert(0, str(PROJECT_ROOT / "paper"))
from paper_data import ANCHOR_CONFIGS, SNAP_REFILL_POINTS

# ── Score parsers ─────────────────────────────────────────────────────────────

def parse_textvqa(text: str) -> str:
    m = re.search(r"Accuracy:\s*([\d.]+)%", text)
    return f"{float(m.group(1)):.2f}%" if m else "N/A"

def parse_pope(text: str) -> str:
    scores = re.findall(r"F1 score:\s*([\d.]+)", text)
    if not scores:
        return "N/A"
    avg = sum(float(x) for x in scores) / len(scores)
    return f"{avg:.4f}"

def parse_mmbench(text: str) -> str:
    m = re.search(r"MMBench Dev Accuracy:\s*([\d.]+)%", text)
    return f"{float(m.group(1)):.2f}%" if m else "N/A"

def parse_mme(text: str) -> str:
    m = re.search(r"Combined Total Score:\s*([\d.]+)", text)
    return f"{float(m.group(1)):.4f}" if m else "N/A"

PARSERS = {
    "textvqa": parse_textvqa,
    "pope":    parse_pope,
    "mmbench": parse_mmbench,
    "mme":     parse_mme,
}

# ── Runner ────────────────────────────────────────────────────────────────────

def run_validate(dataset: str, n4: int, n2: int, n1: int, cuda: str, pca_seed: int = 42) -> str:
    cmd = [PYTHON, str(VALIDATE_PY),
           "--dataset", dataset,
           "--n4", str(n4), "--n2", str(n2), "--n1", str(n1),
           "--split", "all",
           "--cuda", cuda,
           "--pca-seed", str(pca_seed)]
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    output = result.stdout + result.stderr
    if result.returncode != 0:
        print(f"  [warn] validate failed\n{output[-300:]}")
        return "ERR"
    return PARSERS[dataset](output)

# ── Table I/O ─────────────────────────────────────────────────────────────────

COLUMNS = ["B_mid", "anchor_S", "extra_B1", "All B=1", "All B=2", "All B=4", "Snap+refill", "Scaled-opt"]
SEP = "\t"

def load_table(path: Path) -> dict:
    table = {}
    if not path.exists():
        return table
    lines = path.read_text().strip().splitlines()
    if len(lines) < 2:
        return table
    headers = lines[0].split(SEP)
    for line in lines[1:]:
        cells = line.split(SEP)
        if not cells[0].strip():
            continue
        table[cells[0].strip()] = dict(zip(headers, cells))
    return table

def save_table(path: Path, table: dict):
    lines = [SEP.join(COLUMNS)]
    for b_str in sorted(table.keys(), key=lambda x: int(x.replace(",", ""))):
        row = table[b_str]
        lines.append(SEP.join(row.get(c, "") for c in COLUMNS))
    path.write_text("\n".join(lines) + "\n")
    print(f"  [saved] {path}")

def fmt(B: int) -> str:
    return f"{B:,}"

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-results", nargs="*", type=Path, default=None,
                        help="Search result JSON(s) — overrides paper_data.ANCHOR_CONFIGS if provided")
    parser.add_argument("--datasets", nargs="+", default=["mme"],
                        choices=["mme", "textvqa", "pope", "mmbench"])
    parser.add_argument("--cuda", default="0")
    parser.add_argument("--pca-seed", type=int, default=42,
                        help="PruneSID PCA seed forwarded to validate.py (paper reports mean over 42/43/44)")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--points", nargs="+", type=int, default=None,
                        help="Midpoint budgets to run (default: all 7). e.g. --points 32400 52000 83600")
    parser.add_argument("--scaled-only", action="store_true",
                        help="Run only Scaled-opt column, skip All B=1/2/4 and Snap+refill")
    args = parser.parse_args()

    # ── Build anchor lookup: {S: {"n4":..,"n2":..,"n1":..}} ──
    if args.search_results:
        anchors = {}
        for json_path in args.search_results:
            data = json.loads(json_path.read_text())
            for r in data.get("results", [data]):
                S = r["S"]
                anchors[S] = {"n4": r["n4"], "n2": r["n2"], "n1": r["n1"]}
        if not anchors:
            print("[error] No anchor configs found in search JSON(s).")
            sys.exit(1)
        print(f"[info] Loaded {len(anchors)} anchors from JSON")
    else:
        anchors = {S: {"n4": v["n4"], "n2": v["n2"], "n1": v["n1"]}
                   for S, v in ANCHOR_CONFIGS.items()}
        print(f"[info] Using paper_data.ANCHOR_CONFIGS ({len(anchors)} anchors)")

    sorted_anchors = sorted(anchors.keys())

    out_dir = args.output_dir if args.output_dir else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for dataset in args.datasets:
        out_path = out_dir / f"SNAP_REFILL_{dataset.upper()}_{timestamp}.txt"
        table = load_table(out_path)

        points = [pt for pt in SNAP_REFILL_POINTS
                  if args.points is None or pt["B_mid"] in args.points]
        for pt in points:
            B_mid, B_anch = pt["B_mid"], pt["B_anch"]
            # T(B_mid) - T(B_anch): uses the full budget exactly; matches the
            # midpoint configs reported in the paper (see reproduce.py).
            extra = B_mid // 1024 - B_anch // 1024
            b_str = fmt(B_mid)
            if b_str not in table:
                table[b_str] = {"B_mid": b_str}

            if B_anch not in anchors:
                print(f"[warn] Anchor {B_anch} not found, skipping B_mid={B_mid}.")
                continue
            anch_cfg = anchors[B_anch]
            n4_a, n2_a, n1_a = anch_cfg["n4"], anch_cfg["n2"], anch_cfg["n1"]
            T_mid    = B_mid // 1024
            N_refill = n4_a + n2_a + n1_a + extra

            table[b_str]["anchor_S"] = fmt(B_anch)
            table[b_str]["extra_B1"] = str(extra)

            print(f"\n[budget] B_mid={fmt(B_mid)}  anchor={fmt(B_anch)}  extra={extra}  N_refill={N_refill}")

            if not args.scaled_only:
                # All B=1
                if not table[b_str].get("All B=1"):
                    n1_u = T_mid
                    print(f"  [run] All B=1  (n1={n1_u}, N={n1_u})")
                    table[b_str]["All B=1"] = run_validate(dataset, 0, 0, n1_u, args.cuda, args.pca_seed)
                    save_table(out_path, table)
                else:
                    print(f"  [skip] All B=1 (already done)")

                # All B=2
                if not table[b_str].get("All B=2"):
                    n2_u = B_mid // 2048
                    if n2_u == 0:
                        table[b_str]["All B=2"] = "N/A"
                    else:
                        print(f"  [run] All B=2  (n2={n2_u}, N={n2_u})")
                        table[b_str]["All B=2"] = run_validate(dataset, 0, n2_u, 0, args.cuda, args.pca_seed)
                        save_table(out_path, table)
                else:
                    print(f"  [skip] All B=2 (already done)")

                # All B=4
                if not table[b_str].get("All B=4"):
                    n4_u = B_mid // 4096
                    if n4_u == 0:
                        table[b_str]["All B=4"] = "N/A"
                    else:
                        print(f"  [run] All B=4  (n4={n4_u}, N={n4_u})")
                        table[b_str]["All B=4"] = run_validate(dataset, n4_u, 0, 0, args.cuda, args.pca_seed)
                        save_table(out_path, table)
                else:
                    print(f"  [skip] All B=4 (already done)")

                # Snap+refill
                if not table[b_str].get("Snap+refill"):
                    print(f"  [run] Snap+refill  (n4={n4_a}, n2={n2_a}, n1={n1_a+extra}, N={N_refill})")
                    table[b_str]["Snap+refill"] = run_validate(dataset, n4_a, n2_a, n1_a + extra, args.cuda, args.pca_seed)
                    save_table(out_path, table)
                else:
                    print(f"  [skip] Snap+refill (already done)")

            # Scaled-opt: scale anchor (n4,n2,n1) proportionally to T_mid
            if not table[b_str].get("Scaled-opt"):
                T_anch = B_anch // 1024
                r = T_mid / T_anch
                n4_s = round(n4_a * r)
                n2_s = round(n2_a * r)
                n1_s = T_mid - 4 * n4_s - 2 * n2_s
                if n1_s < 0:
                    # reduce n2 so the budget equality still holds
                    n2_s = (T_mid - 4 * n4_s) // 2
                    n1_s = T_mid - 4 * n4_s - 2 * n2_s
                N_s = n4_s + n2_s + n1_s
                print(f"  [run] Scaled-opt  (n4={n4_s}, n2={n2_s}, n1={n1_s}, N={N_s}, r={r:.3f})")
                table[b_str]["Scaled-opt"] = run_validate(dataset, n4_s, n2_s, n1_s, args.cuda, args.pca_seed)
                save_table(out_path, table)
            else:
                print(f"  [skip] Scaled-opt (already done)")

        print(f"\n=== {dataset.upper()} ===")
        print(out_path.read_text())


if __name__ == "__main__":
    main()
