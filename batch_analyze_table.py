import json
import os
from collections import Counter
import re

def extract_answer(text):
    if not isinstance(text, str): return None
    idx = text.rfind('\\boxed')
    if idx < 0: return None
    if text[idx:].startswith('\\boxed{'):
        s = text[idx:]
        if '\\boxed ' in s: return s.replace('\\boxed ', '')
        if s.startswith('\\boxed{') and s.endswith('}'): return s[len('\\boxed{') : -1]
        return s
    match = re.search(r'\\boxed\s+([^\s$]+)', text[idx:])
    if match: return match.group(1).strip()
    return None

def is_valid_answer(s):
    if s is None: return False
    return str(s).strip().lower() not in ['[no_answer]', '[no answer]', 'none', 'n/a', '', 'unknown']

def strip_string(s):
    if not isinstance(s, str): return ''
    s = s.strip()
    if '\\boxed' in s:
        ext = extract_answer(s)
        if ext: s = ext
    s = s.replace(' ', '').lower().replace('\n', '').replace('\\!', '').replace('\\\\', '\\')
    s = s.replace('tfrac', 'frac').replace('dfrac', 'frac')
    s = s.replace('\\left', '').replace('\\right', '')
    s = s.replace('^{\\circ}', '').replace('^\\circ', '')
    s = s.replace('\\$', '')
    if s == '0.5': s = '1/2'
    if s == '1.0': s = '1'
    return s

def analyze_file(filepath):
    subset_total = 0
    subset_raw_corr = 0
    subset_filt_corr = 0
    subset_tp, subset_tn, subset_fp, subset_fn = 0, 0, 0, 0
    subset_verifs = 0
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                gt = strip_string(str(item.get('answer', '')))
                
                answers = item.get('extracted_answers', [])
                valid_candidates = [p for p in [strip_string(str(a)) for a in answers] if is_valid_answer(p)]
                freq = Counter(valid_candidates)
                if not freq: continue
                
                raw_top1 = freq.most_common(1)[0][0]
                top_freq = freq.most_common(1)[0][1]
                total_valid = len(valid_candidates)
                
                if total_valid > 0 and (top_freq / total_valid) > 0.3:
                    continue
                
                subset_total += 1
                if raw_top1 == gt: subset_raw_corr += 1
                
                verifs = item.get('candidate_verifications', {})
                for cand, res in verifs.items():
                    v_res = res.get('verif_res')
                    if v_res is None: continue
                    subset_verifs += 1
                    is_correct = (strip_string(cand) == gt)
                    if is_correct and v_res is True: subset_tp += 1
                    elif not is_correct and v_res is False: subset_tn += 1
                    elif not is_correct and v_res is True: subset_fp += 1
                    elif is_correct and v_res is False: subset_fn += 1
                    
                passed_cands = [cand for cand, res in verifs.items() if res.get('verif_res') is True]
                best_cand = sorted(passed_cands, key=lambda c: freq.get(c, 0), reverse=True)[0] if passed_cands else raw_top1
                if best_cand == gt: subset_filt_corr += 1
    except:
        return None

    if subset_total == 0: return None
    
    raw_acc = subset_raw_corr / subset_total * 100
    filt_acc = subset_filt_corr / subset_total * 100
    verif_acc = (subset_tp + subset_tn) / subset_verifs * 100 if subset_verifs > 0 else 0
    
    return {
        "file": os.path.basename(filepath),
        "subset_size": subset_total,
        "raw_acc": raw_acc,
        "filt_acc": filt_acc,
        "gain": filt_acc - raw_acc,
        "verif_acc": verif_acc
    }

dir_path = r'd:\学习\科研\DIV-TTRL-PR\verl\scripts\experiment_results'
results = []
for filename in os.listdir(dir_path):
    if filename.endswith('.jsonl'):
        res = analyze_file(os.path.join(dir_path, filename))
        if res: results.append(res)

results.sort(key=lambda x: x['gain'], reverse=True)

print(f"{'File Name':<40} | {'Subset':<6} | {'Raw Acc':<8} | {'Filt Acc':<8} | {'Gain':<8} | {'Verif Acc':<8}")
print("-" * 95)
for r in results:
    print(f"{r['file']:<40} | {r['subset_size']:<6} | {r['raw_acc']:>7.2f}% | {r['filt_acc']:>7.2f}% | {r['gain']:>+7.2f}% | {r['verif_acc']:>7.2f}%")
