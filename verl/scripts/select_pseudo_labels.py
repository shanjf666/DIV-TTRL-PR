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

def main():
    parser = argparse.ArgumentParser(description="Select pseudo-labels from rollouts for offline training.")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSONL file from rollouts.py")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL file for training")
    parser.add_argument("--strategy", type=str, choices=["max_logprob", "random", "first"], default="max_logprob",
                        help="Strategy to select the best response among correct ones.")
    args = parser.parse_args()

    results = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Processing {len(lines)} problems from {args.input_file}...")
    
    selected_count = 0
    total_count = 0

    with open(args.output_file, "w", encoding="utf-8") as f_out:
        for line in tqdm(lines):
            item = json.loads(line)
            problem = item["problem"]
            ground_truth = item.get("answer") or item.get("solution")
            if not ground_truth:
                continue
                
            gt_norm = strip_string(ground_truth)
            
            responses = item.get("responses", [])
            extracted_answers = item.get("extracted_answers", [])
            metrics = item.get("response_metrics", [])
            
            correct_indices = []
            for i, extracted in enumerate(extracted_answers):
                if strip_string(extracted) == gt_norm:
                    correct_indices.append(i)
            
            if not correct_indices:
                continue
            
            # Selection logic
            if args.strategy == "max_logprob":
                # Select the one with the highest mean_logprob
                best_idx = -1
                max_lp = -float("inf")
                for idx in correct_indices:
                    lp = metrics[idx].get("mean_logprob")
                    if lp is not None and lp > max_lp:
                        max_lp = lp
                        best_idx = idx
                
                # Fallback to first if no logprob available
                if best_idx == -1:
                    best_idx = correct_indices[0]
            elif args.strategy == "random":
                import random
                best_idx = random.choice(correct_indices)
            else: # first
                best_idx = correct_indices[0]
            
            best_response = responses[best_idx]
            
            # Format for SFT
            training_sample = {
                "instruction": problem,
                "response": best_response,
                "ground_truth": ground_truth,
                "mean_logprob": metrics[best_idx].get("mean_logprob") if metrics else None
            }
            
            f_out.write(json.dumps(training_sample, ensure_ascii=False) + "\n")
            selected_count += 1
            total_count += 1

    print(f"Selected {selected_count} pseudo-labels out of {len(lines)} problems.")
    print(f"Output saved to {args.output_file}")

if __name__ == "__main__":
    main()
