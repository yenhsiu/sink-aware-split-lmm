"""
validate.py

Run inference + scoring for a given dataset and quantization config.
Also used as the per-trial evaluator by quantization_search.py and
validate_snap_refill.py.

Modes:
  --baseline              : 576 tokens, no quant (original model)
  --no-quant [--n-tokens N]: N tokens (or 576), no quant
  --budget B              : lookup the paper's VQAv2-searched (n4, n2, n1) from configs/lut_vqav2.json
  --n4 X --n2 Y --n1 Z   : manual quantization config

Output:
  results/validate_{dataset}_{exp_name}.txt
"""

import os
import re
import csv
import sys
import json
import random
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

SPLIT_SEED = 42

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.resolve()
PYTHON       = sys.executable
# Vanilla, un-fine-tuned LLaVA-1.5 -- training-free, matching PruneSID's own
# paper setup (no LoRA; token selection + quantization are applied directly
# on top of the stock checkpoint). Override with --model-path or
# $LLAVA_MODEL_PATH to use a local copy.
MODEL_PATH   = os.environ.get("LLAVA_MODEL_PATH", "liuhaotian/llava-v1.5-7b")

CONFIGS_PATH = PROJECT_ROOT / "configs" / "lut_vqav2.json"
RESULTS_DIR  = PROJECT_ROOT / "results"

DATASET_CONFIG = {
    "mme": {
        "inference_module": "llava.eval.model_vqa_loader",
        "question_file":    "playground/data/eval/MME/llava_mme.jsonl",
        "image_folder":     "playground/data/eval/MME/MME_Benchmark_release_version",
        "answers_dir":      "playground/data/eval/MME/answers",
        "scoring":          "mme",
    },
    "textvqa": {
        "inference_module": "llava.eval.model_vqa_loader",
        "question_file":    "playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl",
        "image_folder":     "playground/data/eval/textvqa/train_images",
        "answers_dir":      "playground/data/eval/textvqa/answers",
        "annotation":       "playground/data/eval/textvqa/TextVQA_0.5.1_val.json",
        "scoring":          "textvqa",
    },
    "pope": {
        "inference_module": "llava.eval.model_vqa_loader",
        "question_file":    "playground/data/eval/pope/llava_pope_test.jsonl",
        "image_folder":     "playground/data/eval/pope/val2014",
        "answers_dir":      "playground/data/eval/pope/answers",
        "annotation_dir":   "playground/data/eval/pope/coco",
        "scoring":          "pope",
    },
    "scienceqa": {
        "inference_module": "llava.eval.model_vqa_science",
        "question_file":    "playground/data/eval/scienceqa/llava_test_CQM-A.json",
        "image_folder":     "playground/data/eval/scienceqa/images/test",
        "answers_dir":      "playground/data/eval/scienceqa/answers",
        "base_dir":         "playground/data/eval/scienceqa",
        "scoring":          "scienceqa",
        "extra_infer_args": ["--single-pred-prompt"],
    },
    "mmbench": {
        "inference_module": "llava.eval.model_vqa_mmbench",
        "question_file":    "playground/data/eval/mmbench/mmbench_dev_20230712.tsv",
        "answers_dir":      "playground/data/eval/mmbench/answers/mmbench_dev_20230712",
        "annotation_file":  "playground/data/eval/mmbench/mmbench_dev_20230712.tsv",
        "scoring":          "mmbench",
        "extra_infer_args": ["--single-pred-prompt"],
    },
    "vqav2": {
        "inference_module": "llava.eval.model_vqa_loader",
        "question_file":    "playground/data/eval/vqav2/llava_vqav2_mscoco_val2014.jsonl",
        "image_folder":     "playground/data/eval/vqav2/val2014",
        "answers_dir":      "playground/data/eval/vqav2/answers/llava_vqav2_mscoco_val2014",
        "annotation":       "playground/data/eval/vqav2/v2_mscoco_val2014_annotations.json",
        "scoring":          "vqav2",
    },
}

# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  required=True,
                        choices=["mme", "textvqa", "pope", "scienceqa", "mmbench", "vqav2"])
    parser.add_argument("--cuda",        default="0")
    parser.add_argument("--model-path",  default=MODEL_PATH,
                        help=f"LLaVA-1.5 checkpoint (HF id or local dir; default: {MODEL_PATH})")
    parser.add_argument("--split",       default="all", choices=["all", "search", "val"],
                        help="Use a subset: 'search' = first split-ratio%%, 'val' = remaining, 'all' = full dataset")
    parser.add_argument("--split-ratio", type=float, default=0.3,
                        help="Fraction of data used as search set (default: 0.3)")
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED,
                        help=f"Seed for the search/val split (default: {SPLIT_SEED}). "
                             "Override to draw an independent random subset, e.g. for repeated-resample noise checks.")
    parser.add_argument("--save", action="store_true",
                        help="Save results to a results/*.txt file (default: off)")
    parser.add_argument("--pca-seed", type=int, default=42,
                        help="Seed for PruneSID's PCA clustering (torch.pca_lowrank is a randomized algorithm, "
                             "unseeded it makes token selection non-deterministic across runs -- default 42 "
                             "matches this codebase's existing seed convention). Override to run a multi-seed "
                             "robustness check.")

    parser.add_argument("--no-sink-filter", action="store_true",
                        help="Disable sink-token detection (ablation): sink tokens compete for "
                             "high bit-tiers normally instead of being pinned to the lowest tier")
    parser.add_argument("--sink-ratio", type=float, default=100.0,
                        help="A candidate is flagged sink if its score > ratio * median score (default: 100.0)")
    parser.add_argument("--sink-max-candidates", type=int, default=2,
                        help="Only the top-K ranked patches are ever considered as sink candidates (default: 2)")
    parser.add_argument("--sink-mode", default="pin",
                        choices=["pin", "capa", "capa_rank", "capa_full", "value_rank", "capa_rank_cap"],
                        help="'pin' (default): every detected sink candidate is unconditionally pinned to the "
                             "lowest bit tier. 'capa': a sink candidate is only pinned if its CAPA "
                             "attention-contribution score is below the median contribution of the N selected "
                             "tokens -- spared candidates still rank by raw attention score. 'capa_rank': no sink "
                             "detection at all -- every selected token is ranked directly by CAPA contribution "
                             "for tier assignment. 'capa_full': CAPA also drives PruneSID's own cluster-NMS token "
                             "selection (not just post-selection tier ranking) -- the deepest integration. "
                             "'value_rank': like capa_rank but ranks by the *unweighted* value-vector magnitude "
                             "(attention-weight factor dropped) -- separates true attention-sink tokens (near-zero "
                             "intrinsic content) more cleanly than the attention-weighted CAPA score. "
                             "'capa_rank_cap': capa_rank's C_i ranking, but a detected sink candidate (same "
                             "_detect_sink_count heuristic as pin/capa) can never win a 4-bit slot -- capped at "
                             "2-bit max, still competes normally for 2-bit/1-bit by C_i rank. "
                             "capa_rank/capa_full/value_rank ignore --no-sink-filter / --sink-ratio / "
                             "--sink-max-candidates; capa_rank_cap uses --sink-ratio / --sink-max-candidates but "
                             "not --no-sink-filter.")

    # Mode flags
    parser.add_argument("--baseline",  action="store_true")
    parser.add_argument("--no-quant",  action="store_true")
    parser.add_argument("--n-tokens",  type=int, default=None)
    parser.add_argument("--budget",    type=int, default=None)
    parser.add_argument("--n4",        type=int, default=None)
    parser.add_argument("--n2",        type=int, default=None)
    parser.add_argument("--n1",        type=int, default=None)

    args = parser.parse_args()

    # Validate mode
    modes = [args.baseline, args.no_quant, args.budget is not None,
             any(x is not None for x in [args.n4, args.n2, args.n1])]
    if sum(modes) > 1:
        parser.error("Specify exactly one mode: --baseline / --no-quant / --budget / --n4 --n2 --n1")
    if sum(modes) == 0:
        parser.error("Specify a mode: --baseline / --no-quant [--n-tokens N] / --budget B / --n4 X --n2 Y --n1 Z")

    # Manual quant: all three required
    if any(x is not None for x in [args.n4, args.n2, args.n1]):
        if not all(x is not None for x in [args.n4, args.n2, args.n1]):
            parser.error("--n4, --n2, --n1 must all be specified together")

    return args


def resolve_config(args):
    """Return (n4, n2, n1, use_quant, exp_name) from parsed args."""
    if args.baseline:
        return 0, 0, 576, False, "baseline"

    if args.no_quant:
        n = args.n_tokens if args.n_tokens is not None else 576
        return 0, 0, n, False, f"noq_N{n}"

    if args.budget is not None:
        configs = json.loads(CONFIGS_PATH.read_text())
        key = str(args.budget)
        if key not in configs["budgets"]:
            available = ", ".join(configs["budgets"].keys())
            print(f"[error] Budget {args.budget} not in lookup table. Available: {available}")
            sys.exit(1)
        c = configs["budgets"][key]
        return c["n4"], c["n2"], c["n1"], True, f"budget{args.budget}"

    # Manual
    return args.n4, args.n2, args.n1, True, f"n4{args.n4}_n2{args.n2}_n1{args.n1}"


# ── Split logic ───────────────────────────────────────────────────────────────

def get_split_question_file(dataset_cfg: dict, split: str, ratio: float, seed: int = SPLIT_SEED) -> Path:
    """Return path to split question file, creating it if it doesn't exist yet.
    `seed` defaults to the module-wide SPLIT_SEED (reproducible, backward
    compatible); pass a different seed to draw an independent random subset
    (e.g. for repeated-resample noise estimation)."""
    question_file = PROJECT_ROOT / dataset_cfg["question_file"]
    if split == "all":
        return question_file

    split_dir = question_file.parent / "splits"
    split_dir.mkdir(exist_ok=True)

    ratio_tag  = f"r{int(ratio * 100)}"
    split_file = split_dir / f"{question_file.stem}_{split}_{ratio_tag}_seed{seed}{question_file.suffix}"

    if split_file.exists():
        return split_file

    suffix = question_file.suffix

    if suffix == ".jsonl":
        lines = question_file.read_text().strip().splitlines()
        # Group by question_id first so multi-question-per-image entries (e.g.
        # MME's 2 yes/no questions per image) never get split across search/val --
        # MME's official scorer asserts exactly 2 answers per image and crashes
        # (silently scoring 0.0) if a pair is split across the two subsets.
        groups = {}
        for i, line in enumerate(lines):
            qid = json.loads(line)["question_id"]
            groups.setdefault(qid, []).append(i)
        group_keys = list(groups.keys())
        rng = random.Random(seed)
        rng.shuffle(group_keys)
        n_search = int(len(group_keys) * ratio)
        chosen_keys = group_keys[:n_search] if split == "search" else group_keys[n_search:]
        chosen   = sorted(i for k in chosen_keys for i in groups[k])
        selected = [lines[i] for i in chosen]
        split_file.write_text("\n".join(selected) + "\n")
        print(f"[split] {split} ({len(selected)}/{len(lines)} items) → {split_file.name}")

    elif suffix == ".json":
        data = json.loads(question_file.read_text())
        if isinstance(data, dict):
            keys = list(data.keys())
            rng  = random.Random(seed)
            rng.shuffle(keys)
            n_search     = int(len(keys) * ratio)
            chosen_keys  = keys[:n_search] if split == "search" else keys[n_search:]
            selected_data = {k: data[k] for k in chosen_keys}
            split_file.write_text(json.dumps(selected_data, indent=2))
            print(f"[split] {split} ({len(chosen_keys)}/{len(keys)} items) → {split_file.name}")
        else:
            rng = random.Random(seed)
            shuffled = data[:]
            rng.shuffle(shuffled)
            n_search     = int(len(shuffled) * ratio)
            selected_data = shuffled[:n_search] if split == "search" else shuffled[n_search:]
            split_file.write_text(json.dumps(selected_data, indent=2))
            print(f"[split] {split} ({len(selected_data)}/{len(data)} items) → {split_file.name}")

    elif suffix == ".tsv":
        # Parse with csv (not naive splitlines()) -- MMBench's TSV has fields
        # with embedded newlines (e.g. multi-paragraph "hint"), and plain
        # line-splitting silently shreds those rows, corrupting the file
        # (breaks pandas.read_csv downstream) -- same class of bug as the
        # earlier MME question_id-pairing split issue, different dataset.
        with open(question_file, newline="") as f:
            reader = list(csv.reader(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL))
        header, rows = reader[0], reader[1:]
        rng    = random.Random(seed)
        indices = list(range(len(rows)))
        rng.shuffle(indices)
        n_search = int(len(indices) * ratio)
        chosen   = sorted(indices[:n_search] if split == "search" else indices[n_search:])
        selected = [rows[i] for i in chosen]
        with open(split_file, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
            writer.writerow(header)
            writer.writerows(selected)
        print(f"[split] {split} ({len(selected)}/{len(rows)} items) → {split_file.name}")

    else:
        raise ValueError(f"Unsupported question file format: {suffix}")

    return split_file


# ── Env builder ───────────────────────────────────────────────────────────────

def build_env(n4: int, n2: int, n1: int, use_quant: bool, cuda: str, original: bool = False) -> dict:
    N = n4 + n2 + n1
    # "original" = the true unmodified LLaVA-1.5 baseline: no PruneSID
    # selection, no reordering, natural spatial patch order, all 576 tokens,
    # no quantization. Falls into clip_encoder.py forward()'s plain `else`
    # branch (vanilla CLIP forward + feature_select) since this string
    # matches none of the special token_method cases.
    token_method = "original" if original else "prunesid_quant"

    if not use_quant:
        return {
            **os.environ,
            "CUDA_VISIBLE_DEVICES":  cuda,
            "LLAVA_TOKEN_METHOD":    token_method,
            "LLAVA_USE_QUANT":       "false",
            "LLAVA_N_TOKENS":        str(N),
        }

    active = [(n, b) for n, b in [(n4, 4), (n2, 2), (n1, 1)] if n > 0]
    if len(active) == 1:
        quant_mode   = "uniform"
        quant_bits   = str(active[0][1])
        strat_groups = ""
    else:
        quant_mode   = "stratified"
        quant_bits   = ""
        strat_groups = ",".join(f"{n}:{b}" for n, b in [(n4, 4), (n2, 2), (n1, 1)] if n > 0)

    return {
        **os.environ,
        "CUDA_VISIBLE_DEVICES":  cuda,
        "LLAVA_TOKEN_METHOD":    token_method,
        "LLAVA_USE_QUANT":       "true",
        "LLAVA_QUANT_MODE":      quant_mode,
        "LLAVA_N_TOKENS":        str(N),
        "LLAVA_QUANT_BITS":      quant_bits,
        "LLAVA_STRAT_GROUPS":    strat_groups,
    }


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_mme(answers_file: Path, exp_name: str, env: dict) -> str:
    mme_dir = PROJECT_ROOT / "playground/data/eval/MME"
    subprocess.run(
        [PYTHON, "convert_answer_to_mme.py", "--experiment", exp_name],
        env=env, cwd=str(mme_dir), check=True
    )
    result = subprocess.run(
        [PYTHON, "calculation.py", "--results_dir", f"answers/{exp_name}"],
        env=env, cwd=str(mme_dir / "eval_tool"),
        capture_output=True, text=True
    )
    totals = re.findall(r"total score:\s*([\d.]+)", result.stdout)
    total = sum(float(x) for x in totals) if totals else 0.0
    summary = result.stdout + f"\n=== Combined Total Score: {total:.4f} ===\n"
    return summary


def score_textvqa(answers_file: Path, dataset_cfg: dict) -> str:
    result = subprocess.run(
        [PYTHON, "-m", "llava.eval.eval_textvqa",
         "--annotation-file", str(PROJECT_ROOT / dataset_cfg["annotation"]),
         "--result-file",     str(answers_file)],
        cwd=str(PROJECT_ROOT),
        capture_output=True, text=True
    )
    return result.stdout + result.stderr


def score_pope(answers_file: Path, dataset_cfg: dict, question_file: Path) -> str:
    result = subprocess.run(
        [PYTHON, "llava/eval/eval_pope.py",
         "--annotation-dir",  dataset_cfg["annotation_dir"],
         "--question-file",   str(question_file),
         "--result-file",     str(answers_file)],
        cwd=str(PROJECT_ROOT),
        capture_output=True, text=True
    )
    return result.stdout + result.stderr


def score_scienceqa(answers_file: Path, dataset_cfg: dict) -> str:
    base_dir = str(PROJECT_ROOT / dataset_cfg["base_dir"])
    stem = answers_file.stem
    result = subprocess.run(
        [PYTHON, "llava/eval/eval_science_qa.py",
         "--base-dir",      base_dir,
         "--result-file",   str(answers_file),
         "--output-file",   str(answers_file.parent / f"{stem}_output.jsonl"),
         "--output-result", str(answers_file.parent / f"{stem}_result.json")],
        cwd=str(PROJECT_ROOT),
        capture_output=True, text=True
    )
    return result.stdout + result.stderr


def score_mmbench(answers_file: Path, dataset_cfg: dict) -> str:
    result = subprocess.run(
        [PYTHON, "scripts/eval_mmbench_local.py",
         "--annotation-file", str(PROJECT_ROOT / dataset_cfg["annotation_file"]),
         "--result-file",     str(answers_file)],
        cwd=str(PROJECT_ROOT),
        capture_output=True, text=True
    )
    return result.stdout + result.stderr


def score_vqav2(answers_file: Path, dataset_cfg: dict) -> str:
    result = subprocess.run(
        [PYTHON, "llava/eval/eval_vqav2.py",
         "--annotation-file", dataset_cfg["annotation"],
         "--result-file",     str(answers_file)],
        cwd=str(PROJECT_ROOT),
        capture_output=True, text=True
    )
    return result.stdout + result.stderr


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    n4, n2, n1, use_quant, mode_tag = resolve_config(args)
    N = n4 + n2 + n1

    dataset_cfg  = DATASET_CONFIG[args.dataset]
    split_tag    = f"_{args.split}" if args.split != "all" else ""
    seed_tag     = f"_seed{args.split_seed}" if args.split != "all" and args.split_seed != SPLIT_SEED else ""
    if args.sink_mode == "capa_rank_cap":
        sink_tag = "_caprankcap"
    elif args.sink_mode == "value_rank":
        sink_tag = "_valuerank"
    elif args.sink_mode == "capa_full":
        sink_tag = "_capafull"
    elif args.sink_mode == "capa_rank":
        sink_tag = "_caparank"
    elif args.no_sink_filter:
        sink_tag = "_nosink"
    elif args.sink_mode == "capa":
        sink_tag = "_capa"
    else:
        sink_tag = ""
    # pin/capa/capa_rank_cap all read --sink-ratio (capa_rank/capa_full/
    # value_rank ignore it entirely, see --sink-mode help) -- tag the
    # filename when it's not the 100.0 default so a non-default-ratio run
    # never collides with (or is silently indistinguishable from) the
    # baseline 100x run at the same config/seed.
    if args.sink_mode in ("pin", "capa", "capa_rank_cap") and args.sink_ratio != 100.0:
        sink_tag += f"_ratio{args.sink_ratio:g}"
    # same collision risk as sink_ratio above -- pin/capa/capa_rank_cap also
    # read --sink-max-candidates, so tag it too when non-default.
    if args.sink_mode in ("pin", "capa", "capa_rank_cap") and args.sink_max_candidates != 2:
        sink_tag += f"_cap{args.sink_max_candidates}"
    pca_seed_tag = f"_pcaseed{args.pca_seed}" if args.pca_seed != 42 else ""
    exp_name     = f"validate_{args.dataset}{split_tag}{seed_tag}_{mode_tag}{sink_tag}{pca_seed_tag}"
    answers_file = PROJECT_ROOT / dataset_cfg["answers_dir"] / f"{exp_name}.jsonl"
    env          = build_env(n4, n2, n1, use_quant, args.cuda, original=(mode_tag == "baseline"))
    env["LLAVA_SINK_FILTER"]         = "false" if args.no_sink_filter else "true"
    env["LLAVA_SINK_RATIO"]          = str(args.sink_ratio)
    env["LLAVA_SINK_MAX_CANDIDATES"] = str(args.sink_max_candidates)
    env["LLAVA_SINK_MODE"]           = args.sink_mode
    env["LLAVA_PCA_SEED"]            = str(args.pca_seed)

    question_file = get_split_question_file(dataset_cfg, args.split, args.split_ratio, args.split_seed)

    print(f"=== Validate: {args.dataset.upper()} ===")
    print(f"Mode:    {'baseline' if not use_quant and N==576 else ('no-quant' if not use_quant else 'quant')}")
    print(f"Config:  n4={n4}, n2={n2}, n1={n1}, N={N}")
    print(f"Split:   {args.split} (ratio={args.split_ratio}, seed={args.split_seed})")
    print(f"PCA seed: {args.pca_seed}")
    if args.sink_mode == "capa_rank_cap":
        sink_status = "on -- capa_rank ranking, sink candidates capped at 2-bit max (capa_rank_cap)"
    elif args.sink_mode == "value_rank":
        sink_status = "n/a -- all tokens ranked by unweighted value-vector magnitude (value_rank)"
    elif args.sink_mode == "capa_full":
        sink_status = "n/a -- CAPA drives both PruneSID selection and tier ranking (capa_full)"
    elif args.sink_mode == "capa_rank":
        sink_status = "n/a -- all tokens ranked by CAPA contribution (capa_rank)"
    elif args.no_sink_filter:
        sink_status = "off"
    else:
        sink_status = f"on (mode={args.sink_mode}, ratio={args.sink_ratio}, max_candidates={args.sink_max_candidates})"
    print(f"Sink:    {sink_status}")
    print(f"Exp:     {exp_name}")
    print("=" * 35)

    # ── Inference ──
    inference_module = dataset_cfg.get("inference_module", "llava.eval.model_vqa_loader")
    model_args = ["--model-path", args.model_path]

    cmd = [PYTHON, "-m", inference_module,
           *model_args,
           "--question-file", str(question_file),
           "--answers-file",  str(answers_file),
           "--temperature",   "0",
           "--conv-mode",     "vicuna_v1",
    ]
    if "image_folder" in dataset_cfg:
        cmd += ["--image-folder", dataset_cfg["image_folder"]]
    cmd += dataset_cfg.get("extra_infer_args", [])

    result = subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT))
    if result.returncode != 0 or not answers_file.exists():
        print("[error] Inference failed.")
        sys.exit(1)

    # ── Scoring ──
    scoring = dataset_cfg["scoring"]
    if scoring == "mme":
        output = score_mme(answers_file, exp_name, env)
    elif scoring == "textvqa":
        output = score_textvqa(answers_file, dataset_cfg)
    elif scoring == "pope":
        output = score_pope(answers_file, dataset_cfg, question_file)
    elif scoring == "scienceqa":
        output = score_scienceqa(answers_file, dataset_cfg)
    elif scoring == "mmbench":
        output = score_mmbench(answers_file, dataset_cfg)
    elif scoring == "vqav2":
        output = score_vqav2(answers_file, dataset_cfg)

    print(output)

    # ── Save results ──
    if args.save:
        RESULTS_DIR.mkdir(exist_ok=True)
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = RESULTS_DIR / f"{exp_name}_{timestamp}.txt"

        meta = {
            "dataset":      args.dataset,
            "split":        args.split,
            "split_ratio":  args.split_ratio,
            "mode":         mode_tag,
            "n4": n4, "n2": n2, "n1": n1, "N": N,
            "use_quant":    use_quant,
            "budget":       args.budget,
            "pca_seed":     args.pca_seed,
            "sink_mode":    args.sink_mode,
            "sink_filter":  not args.no_sink_filter,
            "sink_ratio":   args.sink_ratio,
            "sink_max_candidates": args.sink_max_candidates,
            "timestamp":    timestamp,
        }

        with open(result_file, "w") as f:
            f.write(f"=== Config ===\n{json.dumps(meta, indent=2)}\n\n")
            f.write(f"=== Score ===\n{output}\n")

        print(f"\n[saved] {result_file}")


if __name__ == "__main__":
    main()
