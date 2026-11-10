import json
import argparse
from collections import defaultdict

def vqa_accuracy(gt_answers, pred_answer):
    pred = pred_answer.strip().lower().rstrip('.')
    count = sum(1 for a in gt_answers if a['answer'].lower() == pred)
    return min(1.0, count / 3.0)

parser = argparse.ArgumentParser()
parser.add_argument('--annotation-file', required=True)
parser.add_argument('--result-file', required=True)
args = parser.parse_args()

annotations = json.load(open(args.annotation_file))['annotations']
gt = {a['question_id']: a['answers'] for a in annotations}

results = [json.loads(line) for line in open(args.result_file)]

total, correct = 0, 0.0
answer_type_scores = defaultdict(list)

ann_map = {a['question_id']: a for a in annotations}

for r in results:
    qid = r['question_id']
    if qid not in gt:
        continue
    score = vqa_accuracy(gt[qid], r['text'])
    correct += score
    total += 1
    if qid in ann_map:
        atype = ann_map[qid].get('answer_type', 'other')
        answer_type_scores[atype].append(score)

print(f"Overall Accuracy: {100 * correct / total:.2f}% ({total} questions)")
for atype, scores in sorted(answer_type_scores.items()):
    print(f"  {atype}: {100 * sum(scores) / len(scores):.2f}% ({len(scores)} questions)")
