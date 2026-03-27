import json
import argparse
from collections import Counter

def strip_string(string):
    if string is None:
        return ""
    string = str(string)
    string = string.replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "").replace(" ", "")
    if string == "0.5":
        string = "\\frac{1}{2}"
    return string

def main():
    parser = argparse.ArgumentParser(description="Analyze Rollout Accuracy (SC and Pass@k).")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSONL from rollouts.py")
    args = parser.parse_args()

    total = 0
    sc_correct_total = 0
    pass_k_total = 0
    has_gt = 0

    buckets = {
        "Low (0.0-0.3)": {"n": 0, "correct": 0},
        "Mid (0.3-0.7)": {"n": 0, "correct": 0},
        "High (0.7-1.0)": {"n": 0, "correct": 0}
    }

    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            total += 1
            
            gt_ans = item.get("answer") or item.get("solution")
            if not gt_ans:
                continue
            
            has_gt += 1
            gt_norm = strip_string(gt_ans)
            
            # 1. SC Accuracy
            voted_ans = strip_string(item.get("sc_answer", ""))
            sc_score = item.get("sc_score", 0.0)
            
            # Bucket assignment
            if sc_score < 0.3: b = "Low (0.0-0.3)"
            elif sc_score < 0.7: b = "Mid (0.3-0.7)"
            else: b = "High (0.7-1.0)"
            
            buckets[b]["n"] += 1
            
            is_sc_correct = (voted_ans == gt_norm)
            if is_sc_correct:
                sc_correct_total += 1
                buckets[b]["correct"] += 1
                
            # 2. Pass@k
            extracted_answers = item.get("extracted_answers", [])
            is_pass_any = any(strip_string(a) == gt_norm for a in extracted_answers)
            if is_pass_any:
                pass_k_total += 1

    print("\n" + "="*60)
    print("ROLLOUT ACCURACY ANALYSIS")
    print("="*60)
    print(f"Total problems           : {total}")
    print(f"Problems with GT         : {has_gt}")
    if has_gt > 0:
        print(f"Overall SC Accuracy      : {sc_correct_total/has_gt:.2%} ({sc_correct_total}/{has_gt})")
        print(f"Overall Pass@k Accuracy  : {pass_k_total/has_gt:.2%} ({pass_k_total}/{has_gt})")
        
        print("\n" + "-"*30)
        print("SC ACCURACY BY CONFIDENCE")
        print("-" * 30)
        print(f"{'Confidence Bucket':<16} | {'N':>5} | {'Accuracy':>10}")
        print("-" * 35)
        for b, data in buckets.items():
            acc = data["correct"] / data["n"] if data["n"] > 0 else 0.0
            print(f"{b:<16} | {data['n']:>5} | {acc:>10.1%}")
    print("="*60)

if __name__ == "__main__":
    main()
