import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import Counter, defaultdict
import os
import math

def compute_delta_pass_k(N: int, c: int, k: int) -> float:
    """
    Compute ΔPass@k for a specific cluster size c out of N samples.
    Using combination formula explicitly: C(N-c, k-1) / C(N, k)
    """
    if c == 0 or N == 0:
        return 0.0
    if c == N:
        return 0.0
    if N - c < k - 1:
        return 0.0
        
    numerator = math.comb(N - c, k - 1)
    denominator = math.comb(N, k)
    return numerator / denominator

def analyze_cluster_vs_length(jsonl_path, output_png):
    """
    Reads qwen4b.jsonl and extracts cluster sizes vs average reasoning lengths.
    Structure:
      extracted_answers: [ans1, ans2, ...]
      response_metrics: [{"response_length": int, ...}, ...]
    """
    print(f"Reading data from {jsonl_path}...")
    
    # Store tuples of (cluster_size, response_length)
    cluster_size_to_lengths = defaultdict(list)
    
    total_problems = 0
    total_responses = 0
    
    if not os.path.exists(jsonl_path):
        print(f"Error: {jsonl_path} does not exist.")
        return

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            data = json.loads(line)
            total_problems += 1
            
            extracted_answers = data.get("extracted_answers", [])
            response_metrics = data.get("response_metrics", [])
            
            if not extracted_answers or not response_metrics:
                continue
                
            # Filter out [NO_ANSWER]
            valid_answers = [ans for ans in extracted_answers if ans != "[NO_ANSWER]" and ans is not None]
            
            # Count cluster sizes for this problem
            counter = Counter(valid_answers)
            
            # Map each response to its cluster size and length
            for i, ans in enumerate(extracted_answers):
                if i >= len(response_metrics):
                    break
                    
                if ans == "[NO_ANSWER]" or ans is None:
                    continue # Skip invalid answers
                    
                cluster_size = counter[ans]
                res_length = response_metrics[i].get("response_length", 0)
                
                if res_length > 0:
                    cluster_size_to_lengths[cluster_size].append(res_length)
                    total_responses += 1

    print(f"Processed {total_problems} problems, {total_responses} valid responses.")
    
    if not cluster_size_to_lengths:
        print("No valid data found to plot!")
        return

    # Aggregate data for plotting
    unique_sizes = sorted(cluster_size_to_lengths.keys())
    avg_lengths = []
    std_lengths = []
    counts = []
    
    for size in unique_sizes:
        lengths = cluster_size_to_lengths[size]
        avg_lengths.append(np.mean(lengths))
        std_lengths.append(np.std(lengths))
        counts.append(len(lengths))

    # ===== Compute g(c) =====
    k_val = 4  # Assuming k=4 as used in training
    g_c_values = []
    
    # We can calculate N per problem, but since num_return_sequences varies (due to invalid ones),
    # we'll approximate N as the max observed cluster sum across problems, or explicitly 32-64.
    # From your command line, it seems you generate 64. Let's dynamically infer N per group.
    
    print("\nCluster Size | Avg Length (tokens) | Std Dev | Sample Count | Env g(c) (approx)")
    print("-" * 75)
    for s, m, std, c in zip(unique_sizes, avg_lengths, std_lengths, counts):
        # We approximate g(c) here just for printing, assuming N=64 for the general formula showcase
        N_approx = 64
        delta = compute_delta_pass_k(N_approx, s, k_val)
        gc = (s / N_approx) * delta
        print(f"{s:^12} | {m:^19.1f} | {std:^7.1f} | {c:^12} | {gc:^15.6f}")

    # To plot actual g(c) distribution, we need to compute it per problem instance:
    all_gc_samples = []
    
    # We also want per-problem statistics: max, min, mean, median, mode
    problem_gc_stats = {
        'idx': [], 'max': [], 'min': [], 'mean': [], 'median': [], 'mode': []
    }
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if not line.strip(): continue
            data = json.loads(line)
            extracted = data.get("extracted_answers", [])
            valid_ans = [a for a in extracted if a != "[NO_ANSWER]" and a is not None]
            N_valid = len(valid_ans)
            if N_valid < k_val: continue
            
            c_counts = Counter(valid_ans)
            problem_gcs = []
            for ans, c in c_counts.items():
                delta = compute_delta_pass_k(N_valid, c, k_val)
                gc = (c / N_valid) * delta
                # Append g(c) for EACH sample in that cluster to represent the reward distribution
                all_gc_samples.extend([gc] * c)
                problem_gcs.extend([gc] * c)
            
            if problem_gcs:
                problem_gc_stats['idx'].append(idx)
                problem_gc_stats['max'].append(np.max(problem_gcs))
                problem_gc_stats['min'].append(np.min(problem_gcs))
                problem_gc_stats['mean'].append(np.mean(problem_gcs))
                problem_gc_stats['median'].append(np.median(problem_gcs))
                # For mode of continuous values, we can just take the most frequent using Counter
                # Since g(c) takes discrete values based on c, Counter works perfectly
                mode_val = Counter(problem_gcs).most_common(1)[0][0]
                problem_gc_stats['mode'].append(mode_val)

    # ===== Generating Plot =====
    print(f"\nGenerating plot: {output_png}...")
    
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 20), gridspec_kw={'height_ratios': [2.5, 1, 1.5, 2]})
    
    # Top Plot: Scatter + Line for Avg Reasoning Length
    ax1.plot(unique_sizes, avg_lengths, marker='o', linestyle='-', color='b', linewidth=2.5, markersize=8)
    ax1.fill_between(
        unique_sizes, 
        [m - min(std, m) for m, std in zip(avg_lengths, std_lengths)], # don't go below 0
        [m + std for m, std in zip(avg_lengths, std_lengths)], 
        alpha=0.2, color='b'
    )
    
    ax1.set_title("Cluster Size vs. Average Reasoning Length (Tokens)", fontsize=16, fontweight='bold', pad=15)
    ax1.set_ylabel("Avg Reasoning Length (Tokens)", fontsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Bottom Plot 1: Bar chart for sample counts
    ax2.bar(unique_sizes, counts, color='gray', alpha=0.6)
    ax2.set_xlabel("Cluster Size ($c$)", fontsize=14)
    ax2.set_ylabel("Sample Count", fontsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Bottom Plot 2: g(c) Reward Distribution
    sns.histplot(all_gc_samples, bins=50, color='purple', ax=ax3, kde=True)
    ax3.set_title("Distribution of Sampled Delta Pass@k Rewards $g(c)$", fontsize=14, fontweight='bold')
    ax3.set_xlabel("Reward $g(c)$ Value", fontsize=14)
    ax3.set_ylabel("Frequency", fontsize=14)
    ax3.tick_params(axis='both', which='major', labelsize=12)
    
    mean_gc = np.mean(all_gc_samples) if all_gc_samples else 0
    ax3.axvline(mean_gc, color='red', linestyle='dashed', linewidth=2, label=f'Mean = {mean_gc:.6f}')
    ax3.legend()

    # Bottom Plot 3: Per-problem Statistics (Max, Min, Mean, Median, Mode)
    if problem_gc_stats['idx']:
        # Sort by idx just in case
        sort_order = np.argsort(problem_gc_stats['idx'])
        x_idx = np.array(problem_gc_stats['idx'])[sort_order]
        y_max = np.array(problem_gc_stats['max'])[sort_order]
        y_min = np.array(problem_gc_stats['min'])[sort_order]
        y_mean = np.array(problem_gc_stats['mean'])[sort_order]
        y_median = np.array(problem_gc_stats['median'])[sort_order]
        y_mode = np.array(problem_gc_stats['mode'])[sort_order]
        
        # Plotting styling
        ax4.plot(x_idx, y_max, marker='^', linestyle='', color='red', markersize=5, alpha=0.7, label='Max')
        ax4.plot(x_idx, y_min, marker='v', linestyle='', color='blue', markersize=5, alpha=0.7, label='Min')
        ax4.plot(x_idx, y_mean, marker='o', linestyle='-', color='green', linewidth=1, markersize=4, alpha=0.9, label='Mean')
        ax4.plot(x_idx, y_median, marker='s', linestyle='', color='orange', markersize=4, alpha=0.6, label='Median')
        ax4.plot(x_idx, y_mode, marker='x', linestyle='', color='purple', markersize=6, alpha=0.8, label='Mode')
        
        ax4.set_title("Per-Problem $g(c)$ Statistics Across Dataset Samples", fontsize=14, fontweight='bold')
        ax4.set_xlabel("Problem Sample Index", fontsize=14)
        ax4.set_ylabel("$g(c)$ Value", fontsize=14)
        ax4.tick_params(axis='both', which='major', labelsize=12)
        ax4.grid(True, linestyle='--', alpha=0.7)
        ax4.legend(loc='upper right', framealpha=0.9)

    plt.tight_layout()
    
    try:
        os.makedirs(os.path.dirname(output_png), exist_ok=True)
        plt.savefig(output_png, dpi=300, bbox_inches='tight')
        print(f"Plot successfully saved to {output_png}")
    except Exception as e:
        print(f"Error saving plot: {e}")

if __name__ == "__main__":
    # Assuming qwen4b.jsonl is in the current working directory, or adjust path as needed
    jsonl_file = "base.jsonl"
    output_img = "verl/plots/cluster_size_vs_length_base.png"
    
    # Check current directory and verl directory
    if not os.path.exists(jsonl_file):
        test_paths = ["verl/qwen4b.jsonl", "../qwen4b.jsonl", "data/qwen4b.jsonl", "/root/autodl-tmp/DIV-TTRL/verl/qwen4b.jsonl"]
        for p in test_paths:
            if os.path.exists(p):
                jsonl_file = p
                print(f"Found JSONL file at {jsonl_file}")
                break
                
    analyze_cluster_vs_length(jsonl_file, output_img)
