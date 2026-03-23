"""
Analyze Exploration Results vs Original setup and Plot Frequency Comparison
"""
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

def strip_string(string):
    if string is None:
        return ""
    string = str(string)
    # Handle ground truth boxed format if present
    import re
    if "\\boxed{" in string:
        m = re.search(r"\\boxed{([^}]*)}", string)
        if m:
            string = m.group(1)
    string = string.replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "").replace(" ", "")
    if string == "0.5":
        string = "\\frac{1}{2}"
    return string

def load_jsonl(filepath):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def main(args):
    orig_data = load_jsonl(args.original_file)
    expl_data = load_jsonl(args.explore_file)
    
    # Map original data by problem text to ensure strict 1-to-1 matching
    orig_map = {item["problem"]: item for item in orig_data}
    
    # We only care about problems that were actively explored
    explored_items = [d for d in expl_data if d.get("responses", [])]
    total_explored = len(explored_items)
    
    print(f"Total problems in explore file: {len(expl_data)}")
    print(f"Total problems actively explored: {total_explored}")
    
    if total_explored == 0:
        print("No exploration data found! Please check if threshold filters out all samples.")
        return

    orig_sc_ratios = []
    new_freq_of_old_maj = []
    valid_indices = []
    
    old_correct_count = 0
    new_correct_count = 0
    same_maj_count = 0
    
    missing_in_orig = 0

    for i, item in enumerate(explored_items):
        prob = item["problem"]
        if prob not in orig_map:
            missing_in_orig += 1
            continue
            
        orig_item = orig_map[prob]
        
        # 1. Original Majority Info
        old_maj = strip_string(orig_item.get("sc_answer", ""))
        orig_ratio = orig_item.get("sc_score", 0.0)
        
        # Calculate frequency of OLD majority in NEW answers (excluding no-answers)
        valid_new = [a for a in new_extracted if a not in ["[NO_ANSWER]", ""]]
        if len(valid_new) > 0:
            count_old_in_new = sum(1 for a in valid_new if a == old_maj)
            freq_old_in_new = count_old_in_new / len(valid_new)
        else:
            freq_old_in_new = 0.0
            
        # Stats
        if old_maj == new_maj and old_maj != "":
            same_maj_count += 1
        if old_maj == gt_ans:
            old_correct_count += 1
        if new_maj == gt_ans:
            new_correct_count += 1
            
        # Data for plotting
        orig_sc_ratios.append(orig_ratio)
        new_freq_of_old_maj.append(freq_old_in_new)
        valid_indices.append(i)

    if not valid_indices:
        print("No matched problems found between files for analysis.")
        if missing_in_orig > 0:
            print(f"Note: {missing_in_orig} problems in explore file were missing from original file.")
        return

    print(f"Matched {len(valid_indices)} problems for analysis.")

    # Sort everything by Original SC Ratio for a cleaner visualization trend
    p_data = sorted(zip(orig_sc_ratios, new_freq_of_old_maj), key=lambda x: x[0])
    sorted_orig_ratios = [x[0] for x in p_data]
    sorted_new_freqs = [x[1] for x in p_data]
    sorted_ids = list(range(len(valid_indices)))

    # Print statistics
    n = len(valid_indices)
    print("-" * 65)
    print("Exploration Statistics vs Original")
    print("-" * 65)
    print(f"Identical Majority Ratio (New == Old): {same_maj_count/n:>7.2%} ({same_maj_count}/{n})")
    print(f"Avg Original SC Ratio:                  {np.mean(orig_sc_ratios):>7.2%}")
    print(f"Avg Freq of Old Maj in New Samples:     {np.mean(new_freq_of_old_maj):>7.2%}")
    print("-" * 65)
    print(f"Original Majority Accuracy:              {old_correct_count/n:>7.2%} ({old_correct_count}/{n})")
    print(f"New Majority Accuracy:                   {new_correct_count/n:>7.2%} ({new_correct_count}/{n})")
    print("-" * 65)

    # Plotting
    plt.figure(figsize=(14, 7))
    
    # Scatter points
    plt.scatter(sorted_ids, sorted_orig_ratios, label='Original SC Ratio', color='#3498db', alpha=0.5, s=20)
    plt.scatter(sorted_ids, sorted_new_freqs, label='Freq of Old Maj (in New)', color='#e74c3c', alpha=0.5, s=20)
    
    # Optional: Add a rolling average to see trends
    window = max(1, len(sorted_ids) // 10)
    if window > 1:
        rolling_avg = np.convolve(sorted_new_freqs, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(sorted_ids)), rolling_avg, color='#c0392b', linewidth=2, label=f'Trend (Rolling Avg w={window})')

    plt.xlabel('Prompt Index (Sorted by Original SC Ratio)')
    plt.ylabel('Ratio / Frequency')
    plt.title('Comparison: How Original Majority Confidence relates to its reappearance in New Exploration')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.ylim(-0.05, 1.05) # Fixed Y range for better comparison

    output_plot = args.output_plot
    plt.savefig(output_plot, dpi=150)
    print(f"Enhanced Plot saved to {output_plot}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze and Plot Exploration Results")
    parser.add_argument("--original_file", type=str, required=True, help="Path to qwen64.jsonl")
    parser.add_argument("--explore_file", type=str, required=True, help="Path to explore_results.jsonl")
    parser.add_argument("--output_plot", type=str, default="consistency_comparison.png", help="Path to save the plot")
    args = parser.parse_args()
    main(args)
