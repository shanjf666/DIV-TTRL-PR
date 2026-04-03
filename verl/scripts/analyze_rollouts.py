import json
import argparse
from collections import Counter
from verl.utils.reward_score.ttrl.auto_verify import auto_verify

def main():
    parser = argparse.ArgumentParser(description="Analyze Rollout Accuracy (SC and Pass@k) using auto_verify.")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSONL from rollouts.py")
    parser.add_argument("--task", type=str, default="math", help="Task name for auto_verify (math, gpqa, simplerl_math)")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel workers for verification")
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

    # Load all items first to allow for batched auto_verify later
    items = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            items.append(json.loads(line))

    # To optimize processing, we collect all SC answers and all Pass@k answers for bulk verification
    sc_outputs = []
    sc_labels = []
    
    pass_k_outputs = []
    pass_k_labels = []
    pass_k_mapping = [] # (problem_index, sample_indices_in_pass_k_outputs)

    valid_problems = []

    for idx, item in enumerate(items):
        gt_ans = item.get("answer") or item.get("solution")
        if not gt_ans:
            continue
        
        valid_problems.append(item)
        has_gt += 1
        
        # Collect for SC
        sc_answer = item.get("sc_answer", "")
        sc_outputs.append(sc_answer)
        sc_labels.append(gt_ans)
        
        # Collect for Pass@k
        extracted_answers = item.get("extracted_answers", [])
        start_idx = len(pass_k_outputs)
        pass_k_outputs.extend(extracted_answers)
        pass_k_labels.extend([gt_ans] * len(extracted_answers))
        end_idx = len(pass_k_outputs)
        pass_k_mapping.append((start_idx, end_idx))

    if not valid_problems:
        print("No problems with ground truth found in the input file.")
        return

    print(f"Starting batch verification for {has_gt} problems using task '{args.task}'...")
    
    # Batch Verify SC
    sc_rewards, _ = auto_verify(task=args.task, all_outputs=sc_outputs, all_labels=sc_labels, num_workers=args.num_workers)
    
    # Batch Verify Pass@k
    if pass_k_outputs:
        pass_k_rewards, _ = auto_verify(task=args.task, all_outputs=pass_k_outputs, all_labels=pass_k_labels, num_workers=args.num_workers)
    else:
        pass_k_rewards = []

    # Process Results
    for i, item in enumerate(valid_problems):
        total += 1
        sc_score = item.get("sc_score", 0.0)
        
        # Bucket assignment
        if sc_score < 0.3: b = "Low (0.0-0.3)"
        elif sc_score < 0.7: b = "Mid (0.3-0.7)"
        else: b = "High (0.7-1.0)"
        
        buckets[b]["n"] += 1
        
        # 1. SC Accuracy
        if sc_rewards[i] > 0:
            sc_correct_total += 1
            buckets[b]["correct"] += 1
            
        # 2. Pass@k
        start, end = pass_k_mapping[i]
        group_rewards = pass_k_rewards[start:end]
        if any(r > 0 for r in group_rewards):
            pass_k_total += 1

    print("\n" + "="*60)
    print("ROLLOUT ACCURACY ANALYSIS (Calculated via auto_verify)")
    print("="*60)
    print(f"Total problems           : {len(items)}")
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
