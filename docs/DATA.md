# Evaluation data

All benchmarks follow the LLaVA-1.5 evaluation layout. Start from LLaVA's
[`eval.zip`](https://drive.google.com/file/d/1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy/view?usp=sharing)
(see [LLaVA's Evaluation.md](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md)) and unzip it to
`playground/data/eval/`. Then add the images and annotations listed below. Every path is relative to the repository root.

```
playground/data/eval/
├── MME/
│   ├── llava_mme.jsonl                         # from eval.zip
│   ├── convert_answer_to_mme.py                # from eval.zip
│   ├── eval_tool/calculation.py                # official MME scorer (from eval.zip / MME release)
│   └── MME_Benchmark_release_version/          # MME images (request from the MME authors)
├── pope/
│   ├── llava_pope_test.jsonl                   # from eval.zip
│   ├── coco/coco_pope_{adversarial,popular,random}.json   # POPE annotations
│   └── val2014/                                # COCO val2014 images
├── textvqa/
│   ├── llava_textvqa_val_v051_ocr.jsonl        # from eval.zip
│   ├── TextVQA_0.5.1_val.json                  # https://textvqa.org/dataset
│   └── train_images/                           # TextVQA train_val images
├── mmbench/
│   └── mmbench_dev_20230712.tsv                # https://github.com/open-compass/MMBench
└── vqav2/                                      # used for the offline LUT search
    ├── v2_OpenEnded_mscoco_val2014_questions.json   # https://visualqa.org/download.html
    ├── v2_mscoco_val2014_annotations.json
    ├── llava_vqav2_mscoco_val2014.jsonl        # generated, see below
    └── val2014/                                # COCO val2014 images (same images as POPE, so a symlink works)
```

Generate the VQAv2-val question file:

```bash
python scripts/convert_vqav2_val_to_jsonl.py   # writes playground/data/eval/vqav2/llava_vqav2_mscoco_val2014.jsonl
```

Search/val subsets (`--split search|val`) are created automatically next to each question file,
in a `splits/` folder, with a fixed seed. MME's two questions per image are always kept in the same split.

Answer files go to `playground/data/eval/<benchmark>/answers/`. Scored results go to `results/`
when you pass `--save`.
