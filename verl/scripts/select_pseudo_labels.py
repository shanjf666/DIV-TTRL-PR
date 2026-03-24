import json
import argparse
from tqdm import tqdm

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
    parser = argparse.ArgumentParser(description="Select pseudo-labels from rollouts for offline training.")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSONL file from rollouts.py")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL file for training")
    parser.add_argument("--strategy", type=str, choices=["max_logprob", "random", "first"], default="max_logprob",
                        help="Strategy to select the best response among those matching the voted answer.")
    parser.add_argument("--filter_gt", action="store_true", help="Only include samples where the voted answer matches ground truth.")
    args = parser.parse_args()

    with open(args.input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Processing {len(lines)} problems from {args.input_file}...")
    
    selected_count = 0
    correct_voted_count = 0
    has_gt_count = 0
    any_correct_count = 0 # Pass@N

    with open(args.output_file, "w", encoding="utf-8") as f_out:
        for line in tqdm(lines):
            item = json.loads(line)
            problem = item["problem"]
            ground_truth = item.get("answer") or item.get("solution")
            
            responses = item.get("responses", [])
            extracted_answers = item.get("extracted_answers", [])
            metrics = item.get("response_metrics", [])
            
            # 1. Identify final voted answer (Check if teacher-refined version exists)
            was_reprompted = item.get("was_reprompted", False)
            if was_reprompted:
                voted_answer = strip_string(item.get("teacher_maj_answer"))
                sc_score = item.get("teacher_sc_score", 0.0)
            else:
                # Standard majority voting from original rollouts
                valid_extracted = []
                for i, ans in enumerate(extracted_answers):
                    norm_ans = strip_string(ans)
                    if norm_ans and norm_ans != "[NO_ANSWER]":
                        valid_extracted.append(norm_ans)
                
                if not valid_extracted:
                    # Fallback to sc_answer if available (from rollouts.py direct calculation)
                    fallback_ans = item.get("sc_answer")
                    if fallback_ans:
                        voted_answer = strip_string(fallback_ans)
                        sc_score = item.get("sc_score", 0.0)
                    else:
                        continue
                else:
                    counter = Counter(valid_extracted)
                    voted_answer, count = counter.most_common(1)[0]
                    sc_score = count / len(valid_extracted)
            
            # Accuracy analysis (using final voted result)
            is_voted_correct = False
            if ground_truth:
                has_gt_count += 1
                gt_norm = strip_string(ground_truth)
                if voted_answer == gt_norm:
                    is_voted_correct = True
                    correct_voted_count += 1
                
                # Pass@N check (from original responses)
                for ans in extracted_answers:
                    if strip_string(ans) == gt_norm:
                        any_correct_count += 1
                        break

            # 2. Optional: Filter by ground truth
            if args.filter_gt and ground_truth:
                if not is_voted_correct:
                    continue
            
            # 3. Identify matching metadata (logprob) from original responses if possible
            best_idx = 0
            if not was_reprompted:
                matching_indices = [i for i in range(len(extracted_answers)) if strip_string(extracted_answers[i]) == voted_answer]
                if matching_indices:
                    if args.strategy == "max_logprob":
                        max_lp = -float("inf")
                        for idx in matching_indices:
                            if metrics and idx < len(metrics):
                                lp = metrics[idx].get("mean_logprob")
                                if lp is not None and lp > max_lp:
                                    max_lp = lp
                                    best_idx = idx
                    else:
                        best_idx = matching_indices[0]
            
            # Format for SFT
            training_sample = {
                "instruction": problem,
                "response": f"\\boxed{{{voted_answer}}}",
                "ground_truth": ground_truth,
                "voted_answer": voted_answer,
                "sc_score": sc_score,
                "mean_logprob": metrics[best_idx].get("mean_logprob") if (not was_reprompted and metrics and best_idx < len(metrics)) else None
            }
            
            f_out.write(json.dumps(training_sample, ensure_ascii=False) + "\n")
            selected_count += 1

    print("\n" + "="*50)
    print("PSEUDO-LABEL ANALYSIS SUMMARY")
    print("="*50)
    print(f"Total problems processed    : {len(lines)}")
    print(f"Problems with voted answer  : {selected_count}")
    if has_gt_count > 0:
        print(f"Problems with Ground Truth  : {has_gt_count}")
        print(f"Voted Answer Accuracy (SC)  : {correct_voted_count/has_gt_count:.2%} ({correct_voted_count}/{has_gt_count})")
        print(f"Pass@N Accuracy             : {any_correct_count/has_gt_count:.2%} ({any_correct_count}/{has_gt_count})")
    print(f"Output saved to             : {args.output_file}")
    print("="*50)

if __name__ == "__main__":
    main()
