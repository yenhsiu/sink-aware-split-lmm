import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--questions', default='./playground/data/eval/vqav2/v2_OpenEnded_mscoco_val2014_questions.json')
parser.add_argument('--output', default='./playground/data/eval/vqav2/llava_vqav2_mscoco_val2014.jsonl')
args = parser.parse_args()

questions = json.load(open(args.questions))['questions']

with open(args.output, 'w') as f:
    for q in questions:
        image_id = q['image_id']
        image_file = f"COCO_val2014_{image_id:012d}.jpg"
        f.write(json.dumps({
            'question_id': q['question_id'],
            'image': image_file,
            'text': q['question'] + '\nAnswer the question using a single word or phrase.',
        }) + '\n')

print(f"Converted {len(questions)} questions to {args.output}")
