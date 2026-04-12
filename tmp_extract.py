import json
import random

def run():
    with open('explore_results.jsonl', 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]

    successful = []
    failed = []

    for item in data:
        if not item.get('responses'): continue
        resp = item['responses'][0]
        ext = item.get('extracted_answers', [])[0] if item.get('extracted_answers') else '[NO_ANSWER]'
        
        if ext != '[NO_ANSWER]':
            successful.append({'prob': item['problem'], 'resp': resp, 'ext': ext})
        else:
            failed.append({'prob': item['problem'], 'resp': resp, 'ext': ext})

    print(f'Found {len(successful)} successful extractions and {len(failed)} failed extractions.')

    print('\n' + '='*50 + '\n[SUCCESSFUL EXTRACTION EXAMPLE]\n' + '='*50)
    if successful:
        ex = successful[0]
        print(f'PROBLEM: {ex["prob"]}')
        print(f'EXTRACTED ANSWER: {ex["ext"]}')
        print(f'---\n{ex["resp"]}')

    print('\n' + '='*50 + '\n[FAILED EXTRACTION EXAMPLE]\n' + '='*50)
    if failed:
        fx = failed[0]
        print(f'PROBLEM: {fx["prob"]}')
        print(f'EXTRACTED ANSWER: {fx["ext"]}')
        print(f'---\n{fx["resp"]}')

if __name__ == '__main__':
    run()
