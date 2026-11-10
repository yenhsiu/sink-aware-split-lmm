import pandas as pd
import json
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--annotation-file', required=True)
parser.add_argument('--result-file', required=True)
args = parser.parse_args()

df = pd.read_csv(args.annotation_file, sep='\t')

if 'answer' not in df.columns:
    print("No 'answer' column found — this split requires submission to OpenCompass.")
    exit(1)

gt = {
    str(row['index']): row['answer']
    for _, row in df.iterrows()
    if pd.notna(row.get('answer'))
}

if len(gt) == 0:
    print("Answer column is empty — this split requires submission to OpenCompass.")
    exit(1)

results = [json.loads(line) for line in open(args.result_file)]

correct = 0
total = 0
category_scores = defaultdict(lambda: [0, 0])  # [correct, total]

cat_map = {str(row['index']): row.get('category', 'unknown') for _, row in df.iterrows()}

for r in results:
    qid = str(r['question_id'])
    if qid not in gt:
        continue
    pred = r['text'].strip().upper()
    label = gt[qid].strip().upper()
    is_correct = int(pred == label)
    correct += is_correct
    total += 1
    cat = cat_map.get(qid, 'unknown')
    category_scores[cat][0] += is_correct
    category_scores[cat][1] += 1

if total == 0:
    print("No matching question IDs found between annotation and result file.")
    exit(1)

print(f"MMBench Dev Accuracy: {100 * correct / total:.2f}% ({correct}/{total})")
print()
print("Per-category breakdown:")
for cat, (c, t) in sorted(category_scores.items(), key=lambda x: -x[1][1]):
    print(f"  {cat}: {100*c/t:.1f}% ({c}/{t})")
