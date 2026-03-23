#!/usr/bin/env python3
"""
Compare rollout JSONL outputs from different training stages.

Produces:
  1. Per-file summary table (accuracy, length, entropy, diversity, exploration metrics)
  2. Side-by-side bar charts for all input files
  3. Per-ID paired comparison (scatter + delta histograms) when exactly 2 files given
  4. Per-problem comparison CSV (2-file mode)

Usage:
    # Multi-file summary
    python scripts/compare_rollouts.py --inputs base.jsonl aftertrain.jsonl lendiv.jsonl --output_dir compare_plots

    # Two-file paired analysis
    python scripts/compare_rollouts.py --inputs base.jsonl aftertrain.jsonl --output_dir compare_plots
"""
import argparse
import csv
import json
import math
import os
import sys
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ═══════════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════════

def compute_ngram_repetition(text, n=5):
    """
    Computes the fraction of duplicate n-grams (by whitespace tokenization).
    Returns 0.0 if no repetition, up to ~1.0 if highly repetitive.
    """
    if not text:
        return 0.0
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    
    ngrams = [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    total_ngrams = len(ngrams)
    unique_ngrams = len(set(ngrams))
    
    # Repetition ratio: proportion of n-grams that are duplicates
    return 1.0 - (unique_ngrams / total_ngrams)


# ═══════════════════════════════════════════════════════════════════════════
# Per-file metric computation
# ═══════════════════════════════════════════════════════════════════════════

def _strip_answer(s):
    """Minimal normalization for answer comparison."""
    if s is None:
        return ""
    return str(s).strip()


def pass_at_k(n, c, k):
    """
    Computes the unbiased estimator for Pass@k.
    n: total samples
    c: correct samples
    k: k in pass@k
    """
    if c == 0:
        return 0.0
    if n < k:
        return None
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)

def compute_file_metrics(records):
    """
    Compute aggregate metrics for one JSONL file.

    Returns:
        summary: dict of scalar metrics
        per_problem: list[dict] with per-problem details (keyed by index)
    """
    n_problems = len(records)
    total_responses = 0
    correct_sc = 0
    coverage_count = 0  # problems with at least 1 correct (= Pass@N)

    all_lengths = []
    all_entropies = []
    all_clipped = []
    all_repetition = []

    no_answer_count = 0
    total_answer_slots = 0

    per_problem = []

    for idx, rec in enumerate(records):
        answer_gt = _strip_answer(rec.get("answer", rec.get("solution", "")))
        sc_answer = _strip_answer(rec.get("sc_answer"))
        sc_score = rec.get("sc_score", 0.0) or 0.0
        extracted = rec.get("extracted_answers", [])
        metrics = rec.get("response_metrics", [])
        responses = rec.get("responses", [])
        n_resp = len(metrics)
        total_responses += n_resp

        # --- SC accuracy ---
        if answer_gt and sc_answer and sc_answer == answer_gt:
            correct_sc += 1

        # --- Per-response stats ---
        lengths_this = []
        correct_lengths = []
        entropies_this = []
        rep5_this = []
        correct_count = 0
        no_ans_this = 0

        for i, m in enumerate(metrics):
            rl = m.get("response_length")
            ent = m.get("token_entropy_approx")
            clipped = m.get("is_clipped", False)
            resp_text = responses[i] if i < len(responses) else ""
            
            rep5 = compute_ngram_repetition(resp_text, n=5)
            rep5_this.append(rep5)
            all_repetition.append(rep5)

            if rl is not None:
                lengths_this.append(rl)
                all_lengths.append(rl)
            if ent is not None:
                entropies_this.append(ent)
                all_entropies.append(ent)
            all_clipped.append(clipped)

            ea = extracted[i] if i < len(extracted) else "[NO_ANSWER]"
            total_answer_slots += 1
            if ea == "[NO_ANSWER]":
                no_ans_this += 1
                no_answer_count += 1
            elif answer_gt and _strip_answer(ea) == answer_gt:
                correct_count += 1
                if rl is not None:
                    correct_lengths.append(rl)

        # --- Coverage (Pass@N): at least 1 correct among N samples ---
        if correct_count > 0:
            coverage_count += 1

        # --- True Pass@1: unbiased estimator = c/n per problem ---
        pass1_this = pass_at_k(n_resp, correct_count, 1) if n_resp >= 1 else 0.0
        pass4_this = pass_at_k(n_resp, correct_count, 4)
        pass16_this = pass_at_k(n_resp, correct_count, 16)

        # --- Diversity ---
        valid_answers = [a for a in extracted if a != "[NO_ANSWER]"]
        n_valid = len(valid_answers)
        unique_valid = len(set(valid_answers))
        diversity_ratio = unique_valid / n_valid if n_valid > 0 else 0.0

        # --- Answer entropy (Shannon) ---
        if n_valid > 0:
            counter = Counter(valid_answers)
            probs = np.array(list(counter.values()), dtype=float) / n_valid
            answer_entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
        else:
            answer_entropy = 0.0

        # --- Length variance ---
        mean_len = float(np.mean(lengths_this)) if lengths_this else 0.0
        length_var = float(np.var(lengths_this)) if lengths_this else 0.0
        
        below_mean_count = sum(1 for l in lengths_this if l < mean_len)
        above_mean_count = sum(1 for l in lengths_this if l > mean_len)

        # --- Length Reward (r_div) for SC > 0.5 ---
        mean_below_reward_div = 0.0
        mean_above_reward_div = 0.0
        sc_ratio = correct_count / n_resp if n_resp > 0 else 0.0
        if sc_ratio > 0.5 and len(correct_lengths) > 0:
            mu_l = float(np.mean(correct_lengths))
            sigma_l = float(np.std(correct_lengths))
            below_rewards = []
            above_rewards = []
            lam_div = 0.2
            c_max = 2.0
            for l_i in correct_lengths:
                div_val = abs(l_i - mu_l) / (sigma_l + 1e-5)
                reward_div = lam_div * min(div_val, c_max)
                if l_i < mu_l:
                    below_rewards.append(reward_div)
                elif l_i > mu_l:
                    above_rewards.append(reward_div)
            if below_rewards:
                mean_below_reward_div = float(np.mean(below_rewards))
            if above_rewards:
                mean_above_reward_div = float(np.mean(above_rewards))

        per_problem.append({
            "idx": idx,
            "problem": rec.get("problem", "")[:80],
            "answer_gt": answer_gt,
            "sc_answer": sc_answer,
            "sc_correct": (answer_gt and sc_answer == answer_gt),
            "sc_score": sc_score,
            "coverage": correct_count > 0,
            "pass_at_1": pass1_this,
            "pass_at_4": pass4_this,
            "pass_at_16": pass16_this,
            "correct_count": correct_count,
            "n_resp": n_resp,
            "mean_length": mean_len,
            "median_length": float(np.median(lengths_this)) if lengths_this else 0.0,
            "count_below_mean_length": below_mean_count,
            "count_above_mean_length": above_mean_count,
            "mean_below_reward_div": mean_below_reward_div,
            "mean_above_reward_div": mean_above_reward_div,
            "mean_entropy": float(np.mean(entropies_this)) if entropies_this else None,
            "median_entropy": float(np.median(entropies_this)) if entropies_this else None,
            "diversity_ratio": diversity_ratio,
            "answer_entropy": answer_entropy,
            "length_variance": length_var,
            "mean_repetition_5": float(np.mean(rep5_this)) if rep5_this else 0.0,
            "no_answer_count": no_ans_this,
        })

    # --- Aggregate ---
    acc_sc = correct_sc / n_problems if n_problems else 0.0
    coverage = coverage_count / n_problems if n_problems else 0.0
    
    pass1_list = [p["pass_at_1"] for p in per_problem if p["pass_at_1"] is not None]
    pass4_list = [p["pass_at_4"] for p in per_problem if p["pass_at_4"] is not None]
    pass16_list = [p["pass_at_16"] for p in per_problem if p["pass_at_16"] is not None]

    pass1 = float(np.mean(pass1_list)) if pass1_list else 0.0
    pass4 = float(np.mean(pass4_list)) if pass4_list else None
    pass16 = float(np.mean(pass16_list)) if pass16_list else None
    
    exploration_gain = coverage - acc_sc

    # Wrong-but-diverse: among SC-wrong problems, fraction with diversity > median diversity
    sc_wrong = [p for p in per_problem if not p["sc_correct"]]
    all_div = [p["diversity_ratio"] for p in per_problem]
    median_div = float(np.median(all_div)) if all_div else 0.0
    if sc_wrong:
        wrong_but_diverse = sum(1 for p in sc_wrong if p["diversity_ratio"] > median_div) / len(sc_wrong)
    else:
        wrong_but_diverse = 0.0

    summary = {
        "n_problems": n_problems,
        "total_responses": total_responses,
        "accuracy_sc": acc_sc,
        "pass_at_1": pass1,
        "pass_at_4": pass4,
        "pass_at_16": pass16,
        "coverage": coverage,
        "mean_length": float(np.mean(all_lengths)) if all_lengths else 0.0,
        "median_length": float(np.median(all_lengths)) if all_lengths else 0.0,
        "mean_count_below_mean_length": float(np.mean([p["count_below_mean_length"] for p in per_problem])) if per_problem else 0.0,
        "mean_count_above_mean_length": float(np.mean([p["count_above_mean_length"] for p in per_problem])) if per_problem else 0.0,
        "mean_below_reward_div_global": float(np.mean([p["mean_below_reward_div"] for p in per_problem])) if per_problem else 0.0,
        "mean_above_reward_div_global": float(np.mean([p["mean_above_reward_div"] for p in per_problem])) if per_problem else 0.0,
        "mean_entropy": float(np.mean(all_entropies)) if all_entropies else 0.0,
        "median_entropy": float(np.median(all_entropies)) if all_entropies else 0.0,
        "mean_diversity_ratio": float(np.mean([p["diversity_ratio"] for p in per_problem])),
        "mean_answer_entropy": float(np.mean([p["answer_entropy"] for p in per_problem])),
        "no_answer_rate": no_answer_count / total_answer_slots if total_answer_slots else 0.0,
        "clipped_rate": sum(all_clipped) / len(all_clipped) if all_clipped else 0.0,
        "mean_sc_score": float(np.mean([p["sc_score"] for p in per_problem])),
        "exploration_gain": exploration_gain,
        "mean_length_variance": float(np.mean([p["length_variance"] for p in per_problem])),
        "wrong_but_diverse_rate": wrong_but_diverse,
        "mean_repetition_5": float(np.mean(all_repetition)) if all_repetition else 0.0,
    }
    return summary, per_problem


# ═══════════════════════════════════════════════════════════════════════════
# Console output
# ═══════════════════════════════════════════════════════════════════════════

METRIC_DISPLAY = [
    ("n_problems",            "Problems",              "d",   ""),
    ("total_responses",       "Total Responses",       "d",   ""),
    ("accuracy_sc",           "Accuracy (SC)",         ".2%", ""),
    ("pass_at_1",             "Pass@1",                ".2%", "E[c/n] per problem"),
    ("pass_at_4",             "Pass@4",                ".2%", "1 - C(n-c, 4)/C(n,4)"),
    ("pass_at_16",            "Pass@16",               ".2%", "1 - C(n-c, 16)/C(n,16)"),
    ("coverage",              "Coverage (Pass@All)",   ".2%", "≥1 correct among all"),
    ("exploration_gain",      "Exploration Gain",      ".2%", "Coverage − SC Acc"),
    ("mean_length",           "Mean Length (tok)",      ".1f", ""),
    ("median_length",         "Median Length (tok)",    ".1f", ""),
    ("mean_count_below_mean_length", "Resp < Mean Len", ".2f", "avg count per problem"),
    ("mean_count_above_mean_length", "Resp > Mean Len", ".2f", "avg count per problem"),
    ("mean_below_reward_div_global", "R_div (< Mean)",  ".4f", "avg length reward div (correct, sc>0.5)"),
    ("mean_above_reward_div_global", "R_div (> Mean)",  ".4f", "avg length reward div (correct, sc>0.5)"),
    ("mean_entropy",          "Mean Entropy",           ".4f", "≈ −mean logprob"),
    ("median_entropy",        "Median Entropy",         ".4f", ""),
    ("mean_diversity_ratio",  "Diversity Ratio",        ".4f", "unique/total per problem"),
    ("mean_answer_entropy",   "Answer Entropy (bits)",  ".3f", "Shannon over answer dist"),
    ("mean_length_variance",  "Length Variance",        ".1f", "strategy diversity proxy"),
    ("mean_repetition_5",     "5-Gram Repetition",      ".3f", "higher = more stuck/looping"),
    ("no_answer_rate",        "No-Answer Rate",        ".2%", ""),
    ("clipped_rate",          "Clipped Rate",          ".2%", ""),
    ("mean_sc_score",         "Mean SC Score",          ".4f", ""),
    ("wrong_but_diverse_rate","Wrong-but-Diverse",     ".2%", "exploration among failures"),
]


def print_comparison_table(names, summaries):
    """Pretty-print a comparison table."""
    col_w = max(14, max(len(n) for n in names) + 2)
    label_w = 24
    note_w = 30

    header = f"{'Metric':<{label_w}}" + "".join(f"{n:>{col_w}}" for n in names) + f"  {'Note':<{note_w}}"
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    for key, label, fmt, note in METRIC_DISPLAY:
        row = f"{label:<{label_w}}"
        for s in summaries:
            val = s.get(key, 0)
            if val is None:
                row += f"{'N/A':>{col_w}}"
            else:
                try:
                    formatted_val = f"{val:{fmt}}"
                except ValueError:
                    formatted_val = str(val)
                row += f"{formatted_val:>{col_w}}"
            
        if note:
            row += f"  {note}"
        print(row)

    print("=" * len(header))


# ═══════════════════════════════════════════════════════════════════════════
# Visualization: Per-problem line series across all files
# ═══════════════════════════════════════════════════════════════════════════

def plot_metrics_per_problem(names, all_per_problem, output_dir):
    """
    Plots a line/scatter chart showing mean_length and mean_entropy per problem index
    for all input files. Sorts problems by the first file's length to make trends visible.
    """
    n_problems = min(len(pp) for pp in all_per_problem) if all_per_problem else 0
    if n_problems == 0:
        return

    # To avoid a chaotic hairball, let's sort the problem indices by the mean length
    # of the *first* file. This will make the first file a steadily increasing baseline,
    # and deviations in other files will be clearly visible.
    sorted_indices = sorted(range(n_problems), key=lambda i: all_per_problem[0][i]["mean_length"])
    
    x = np.arange(n_problems)
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(names))))
    window_size = max(1, n_problems // 20)  # ~5% window size for smoothing

    fig, axes = plt.subplots(3, 1, figsize=(18, 16), sharex=True)
    fig.subplots_adjust(hspace=0.15)

    # Helper for moving average
    def moving_average(a, n=window_size):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        valid_means = ret[n - 1:] / n
        return np.pad(valid_means, (n//2, n - 1 - n//2), mode='edge')

    # 1. Mean Length
    ax = axes[0]
    for fi, (name, pp) in enumerate(zip(names, all_per_problem)):
        lengths = [pp[i]["mean_length"] for i in sorted_indices]
        ax.scatter(x, lengths, color=colors[fi], alpha=0.15, s=6, zorder=fi)
        smoothed = moving_average(lengths, window_size)
        ax.plot(x, smoothed, label=f"{name} (Trend)", color=colors[fi], alpha=0.9, linewidth=2.5, zorder=10 + fi)
    ax.set_title("Per-Problem Mean Length (Sorted by Stage 1 Length)", fontsize=14, pad=15)
    ax.set_ylabel("Mean Length (tokens)", fontsize=12)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.2)

    # 2. Mean Entropy
    ax = axes[1]
    for fi, (name, pp) in enumerate(zip(names, all_per_problem)):
        entropies = [pp[i]["mean_entropy"] if pp[i]["mean_entropy"] is not None else 0.0 for i in sorted_indices]
        ax.scatter(x, entropies, color=colors[fi], alpha=0.15, s=6, zorder=fi)
        smoothed = moving_average(entropies, window_size)
        ax.plot(x, smoothed, label=f"{name} (Trend)", color=colors[fi], alpha=0.9, linewidth=2.5, zorder=10 + fi)
    ax.set_title("Per-Problem Mean Entropy (Using Same Ordering)", fontsize=14, pad=15)
    ax.set_ylabel("Mean Entropy", fontsize=12)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.2)

    # 3. Length Variance
    ax = axes[2]
    for fi, (name, pp) in enumerate(zip(names, all_per_problem)):
        variances = [pp[i]["length_variance"] for i in sorted_indices]
        ax.scatter(x, variances, color=colors[fi], alpha=0.15, s=6, zorder=fi)
        smoothed = moving_average(variances, window_size)
        ax.plot(x, smoothed, label=f"{name} (Trend)", color=colors[fi], alpha=0.9, linewidth=2.5, zorder=10 + fi)
    ax.set_title("Per-Problem Length Variance (Using Same Ordering)", fontsize=14, pad=15)
    ax.set_ylabel("Length Variance", fontsize=12)
    ax.set_xlabel("Problem Index (Sorted)", fontsize=12)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.2)

    path = os.path.join(output_dir, "per_problem_series.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {path}")

# ═══════════════════════════════════════════════════════════════════════════
# Visualization: multi-file bar charts
# ═══════════════════════════════════════════════════════════════════════════

def plot_bar_comparison(names, summaries, output_dir):
    """Side-by-side bar charts for key metrics across all files."""
    groups = [
        ("Accuracy & Coverage", [
            ("accuracy_sc", "SC Accuracy"),
            ("pass_at_1", "Pass@1"),
            ("pass_at_4", "Pass@4"),
            ("coverage", "Coverage"),
            ("exploration_gain", "Exp Gain"),
        ]),
        ("Exploration & Quality", [
            ("mean_diversity_ratio", "Diversity Ratio"),
            ("mean_answer_entropy", "Ans Entropy (bits)"),
            ("wrong_but_diverse_rate", "Wrong-but-Diverse"),
            ("mean_repetition_5", "5-Gram Repetition"),
        ]),
        ("Generation Status", [
            ("mean_entropy", "Mean Token Entropy"),
            ("no_answer_rate", "No-Answer Rate"),
            ("clipped_rate", "Clipped Rate"),
        ]),
        ("Length Counts", [
            ("mean_count_below_mean_length", "Count < Mean Len"),
            ("mean_count_above_mean_length", "Count > Mean Len"),
        ]),
        ("Length Rewards", [
            ("mean_below_reward_div_global", "R_div (< Mean)"),
            ("mean_above_reward_div_global", "R_div (> Mean)"),
        ]),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(32, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))

    for ax_idx, (group_title, metric_list) in enumerate(groups):
        ax = axes[ax_idx]
        x = np.arange(len(metric_list))
        width = 0.8 / len(names)

        for fi, (name, summary) in enumerate(zip(names, summaries)):
            vals = [summary.get(k, 0) or 0 for k, _ in metric_list]
            ax.bar(x + fi * width, vals, width, label=name, color=colors[fi], alpha=0.85)

        ax.set_xticks(x + width * (len(names) - 1) / 2)
        ax.set_xticklabels([label for _, label in metric_list], fontsize=9, rotation=15, ha="right")
        ax.set_title(group_title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    # Separate plot for length (different scale)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    x2 = np.arange(3)
    width2 = 0.8 / len(names)
    for fi, (name, summary) in enumerate(zip(names, summaries)):
        # Plot mean length, median length, and std dev of length (sqrt of variance)
        # Using std dev instead of variance so it's on the same scale (tokens)
        std_len = math.sqrt(summary["mean_length_variance"]) if summary["mean_length_variance"] > 0 else 0
        vals = [summary["mean_length"], summary["median_length"], std_len]
        ax2.bar(x2 + fi * width2, vals, width2, label=name, color=colors[fi], alpha=0.85)
    ax2.set_xticks(x2 + width2 * (len(names) - 1) / 2)
    ax2.set_xticklabels(["Mean Length", "Median Length", "Length StdDev"], fontsize=10)
    ax2.set_title("Response Length Comparison (Tokens)", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Rollout Comparison: Key Metrics", fontsize=14, y=1.02)
    fig.tight_layout()
    path1 = os.path.join(output_dir, "comparison_metrics.png")
    fig.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {path1}")

    fig2.tight_layout()
    path2 = os.path.join(output_dir, "comparison_length.png")
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"[Saved] {path2}")


# ═══════════════════════════════════════════════════════════════════════════
# Visualization: two-file paired comparison
# ═══════════════════════════════════════════════════════════════════════════

def paired_comparison(name_a, name_b, pp_a, pp_b, output_dir):
    """
    When exactly 2 files are given, produce per-problem paired analysis.
    Assumes problems are aligned by index.
    """
    n = min(len(pp_a), len(pp_b))
    if n == 0:
        print("[Skip] No problems to compare.")
        return

    lengths_a = np.array([pp_a[i]["mean_length"] for i in range(n)])
    lengths_b = np.array([pp_b[i]["mean_length"] for i in range(n)])

    ent_a = []
    ent_b = []
    ent_idx = []
    for i in range(n):
        ea = pp_a[i]["mean_entropy"]
        eb = pp_b[i]["mean_entropy"]
        if ea is not None and eb is not None:
            ent_a.append(ea)
            ent_b.append(eb)
            ent_idx.append(i)
    ent_a = np.array(ent_a)
    ent_b = np.array(ent_b)

    # ── Scatter: Length ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.scatter(lengths_a, lengths_b, alpha=0.4, s=18, c="#4C72B0", edgecolors="none")
    lo = min(lengths_a.min(), lengths_b.min()) * 0.9
    hi = max(lengths_a.max(), lengths_b.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="y = x")
    ax.set_xlabel(f"Mean Length — {name_a}", fontsize=11)
    ax.set_ylabel(f"Mean Length — {name_b}", fontsize=11)
    ax.set_title("Per-Problem Mean Length", fontsize=13)
    ax.legend()
    ax.grid(alpha=0.25)

    # ── Scatter: Entropy ──
    ax = axes[1]
    if len(ent_a) > 0:
        ax.scatter(ent_a, ent_b, alpha=0.4, s=18, c="#C44E52", edgecolors="none")
        lo = min(ent_a.min(), ent_b.min()) * 0.9
        hi = max(ent_a.max(), ent_b.max()) * 1.05
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="y = x")
    ax.set_xlabel(f"Mean Entropy — {name_a}", fontsize=11)
    ax.set_ylabel(f"Mean Entropy — {name_b}", fontsize=11)
    ax.set_title("Per-Problem Mean Entropy", fontsize=13)
    ax.legend()
    ax.grid(alpha=0.25)

    fig.suptitle(f"Paired Comparison: {name_a} vs {name_b}", fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(output_dir, "paired_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {path}")

    # ── Delta histograms ──
    delta_len = lengths_b - lengths_a
    delta_ent = ent_b - ent_a if len(ent_a) > 0 else np.array([])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(delta_len, bins=50, color="#55A868", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="black", linestyle="--", alpha=0.5)
    ax.axvline(np.mean(delta_len), color="red", linestyle="--", label=f"mean={np.mean(delta_len):.1f}")
    ax.set_xlabel(f"Δ Length ({name_b} − {name_a})", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Per-Problem Length Change", fontsize=13)
    ax.legend()

    ax = axes[1]
    if len(delta_ent) > 0:
        ax.hist(delta_ent, bins=50, color="#8172B3", edgecolor="white", alpha=0.85)
        ax.axvline(0, color="black", linestyle="--", alpha=0.5)
        ax.axvline(np.mean(delta_ent), color="red", linestyle="--", label=f"mean={np.mean(delta_ent):.4f}")
        ax.legend()
    ax.set_xlabel(f"Δ Entropy ({name_b} − {name_a})", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Per-Problem Entropy Change", fontsize=13)

    fig.suptitle(f"Delta Distribution: {name_b} − {name_a}", fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(output_dir, "paired_delta_hist.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {path}")

    # ── Exploration comparison: diversity & answer entropy scatter ──
    div_a = np.array([pp_a[i]["diversity_ratio"] for i in range(n)])
    div_b = np.array([pp_b[i]["diversity_ratio"] for i in range(n)])
    ae_a = np.array([pp_a[i]["answer_entropy"] for i in range(n)])
    ae_b = np.array([pp_b[i]["answer_entropy"] for i in range(n)])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.scatter(div_a, div_b, alpha=0.4, s=18, c="#DD8452", edgecolors="none")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="y = x")
    ax.set_xlabel(f"Diversity Ratio — {name_a}", fontsize=11)
    ax.set_ylabel(f"Diversity Ratio — {name_b}", fontsize=11)
    ax.set_title("Per-Problem Diversity Ratio", fontsize=13)
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.scatter(ae_a, ae_b, alpha=0.4, s=18, c="#4C72B0", edgecolors="none")
    hi = max(ae_a.max(), ae_b.max()) * 1.05 if len(ae_a) > 0 else 1
    ax.plot([0, hi], [0, hi], "k--", alpha=0.5, label="y = x")
    ax.set_xlabel(f"Answer Entropy — {name_a}", fontsize=11)
    ax.set_ylabel(f"Answer Entropy — {name_b}", fontsize=11)
    ax.set_title("Per-Problem Answer Entropy (bits)", fontsize=13)
    ax.legend()
    ax.grid(alpha=0.25)

    fig.suptitle(f"Exploration Comparison: {name_a} vs {name_b}", fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(output_dir, "paired_exploration.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {path}")

    # ── Scatter: Length counts ──
    below_len_a = np.array([pp_a[i]["count_below_mean_length"] for i in range(n)])
    below_len_b = np.array([pp_b[i]["count_below_mean_length"] for i in range(n)])
    above_len_a = np.array([pp_a[i]["count_above_mean_length"] for i in range(n)])
    above_len_b = np.array([pp_b[i]["count_above_mean_length"] for i in range(n)])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.scatter(below_len_a, below_len_b, alpha=0.4, s=18, c="#2CA02C", edgecolors="none")
    lo = min(below_len_a.min(), below_len_b.min()) * 0.9 if len(below_len_a) > 0 else 0
    hi = max(below_len_a.max(), below_len_b.max()) * 1.05 if len(below_len_a) > 0 else 1
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="y = x")
    ax.set_xlabel(f"Count < Mean Length — {name_a}", fontsize=11)
    ax.set_ylabel(f"Count < Mean Length — {name_b}", fontsize=11)
    ax.set_title("Per-Problem Count < Mean Length", fontsize=13)
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.scatter(above_len_a, above_len_b, alpha=0.4, s=18, c="#D62728", edgecolors="none")
    lo = min(above_len_a.min(), above_len_b.min()) * 0.9 if len(above_len_a) > 0 else 0
    hi = max(above_len_a.max(), above_len_b.max()) * 1.05 if len(above_len_a) > 0 else 1
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="y = x")
    ax.set_xlabel(f"Count > Mean Length — {name_a}", fontsize=11)
    ax.set_ylabel(f"Count > Mean Length — {name_b}", fontsize=11)
    ax.set_title("Per-Problem Count > Mean Length", fontsize=13)
    ax.legend()
    ax.grid(alpha=0.25)

    fig.suptitle(f"Length Distribution Counts Comparison: {name_a} vs {name_b}", fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(output_dir, "paired_length_counts.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {path}")

    # ── Scatter: Reward Divs ──
    below_rdiv_a = np.array([pp_a[i]["mean_below_reward_div"] for i in range(n)])
    below_rdiv_b = np.array([pp_b[i]["mean_below_reward_div"] for i in range(n)])
    above_rdiv_a = np.array([pp_a[i]["mean_above_reward_div"] for i in range(n)])
    above_rdiv_b = np.array([pp_b[i]["mean_above_reward_div"] for i in range(n)])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.scatter(below_rdiv_a, below_rdiv_b, alpha=0.4, s=18, c="#8C564B", edgecolors="none")
    lo = min(below_rdiv_a.min(), below_rdiv_b.min()) * 0.9 if len(below_rdiv_a) > 0 else 0
    hi = max(below_rdiv_a.max(), below_rdiv_b.max()) * 1.05 if len(below_rdiv_a) > 0 else 1
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="y = x")
    ax.set_xlabel(f"R_div (< Mean) — {name_a}", fontsize=11)
    ax.set_ylabel(f"R_div (< Mean) — {name_b}", fontsize=11)
    ax.set_title("Per-Problem R_div (< Mean)", fontsize=13)
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.scatter(above_rdiv_a, above_rdiv_b, alpha=0.4, s=18, c="#E377C2", edgecolors="none")
    lo = min(above_rdiv_a.min(), above_rdiv_b.min()) * 0.9 if len(above_rdiv_a) > 0 else 0
    hi = max(above_rdiv_a.max(), above_rdiv_b.max()) * 1.05 if len(above_rdiv_a) > 0 else 1
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="y = x")
    ax.set_xlabel(f"R_div (> Mean) — {name_a}", fontsize=11)
    ax.set_ylabel(f"R_div (> Mean) — {name_b}", fontsize=11)
    ax.set_title("Per-Problem R_div (> Mean)", fontsize=13)
    ax.legend()
    ax.grid(alpha=0.25)

    fig.suptitle(f"Length Reward Divs Comparison: {name_a} vs {name_b}", fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(output_dir, "paired_rdiv.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {path}")

    # ── Correctness transition analysis ──
    # Categorize each problem: both_correct, gained, lost, both_wrong
    gained = 0
    lost = 0
    both_correct = 0
    both_wrong = 0
    for i in range(n):
        ca = pp_a[i]["sc_correct"]
        cb = pp_b[i]["sc_correct"]
        if ca and cb:
            both_correct += 1
        elif not ca and cb:
            gained += 1
        elif ca and not cb:
            lost += 1
        else:
            both_wrong += 1

    print(f"\n  Correctness Transition ({name_a} → {name_b}):")
    print(f"    Both correct : {both_correct:>4d}  ({both_correct/n:.1%})")
    print(f"    Gained       : {gained:>4d}  ({gained/n:.1%})  (wrong→right)")
    print(f"    Lost         : {lost:>4d}  ({lost/n:.1%})  (right→wrong)")
    print(f"    Both wrong   : {both_wrong:>4d}  ({both_wrong/n:.1%})")

    # ── Save per-problem CSV ──
    csv_path = os.path.join(output_dir, "per_problem_comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.writer(csvf)
        writer.writerow([
            "idx", "problem_snippet",
            f"sc_correct_{name_a}", f"sc_correct_{name_b}", "transition",
            f"mean_length_{name_a}", f"mean_length_{name_b}", "delta_length",
            f"mean_entropy_{name_a}", f"mean_entropy_{name_b}", "delta_entropy",
            f"diversity_{name_a}", f"diversity_{name_b}", "delta_diversity",
            f"answer_entropy_{name_a}", f"answer_entropy_{name_b}", "delta_answer_entropy",
            f"count_below_len_{name_a}", f"count_below_len_{name_b}",
            f"count_above_len_{name_a}", f"count_above_len_{name_b}",
            f"below_rdiv_{name_a}", f"below_rdiv_{name_b}",
            f"above_rdiv_{name_a}", f"above_rdiv_{name_b}",
        ])
        for i in range(n):
            ca = pp_a[i]["sc_correct"]
            cb = pp_b[i]["sc_correct"]
            if ca and cb:
                trans = "both_correct"
            elif not ca and cb:
                trans = "gained"
            elif ca and not cb:
                trans = "lost"
            else:
                trans = "both_wrong"

            ea = pp_a[i]["mean_entropy"]
            eb = pp_b[i]["mean_entropy"]
            de = (eb - ea) if ea is not None and eb is not None else ""

            writer.writerow([
                i, pp_a[i]["problem"][:60],
                ca, cb, trans,
                f"{pp_a[i]['mean_length']:.1f}", f"{pp_b[i]['mean_length']:.1f}",
                f"{pp_b[i]['mean_length'] - pp_a[i]['mean_length']:.1f}",
                f"{ea:.4f}" if ea is not None else "", f"{eb:.4f}" if eb is not None else "",
                f"{de:.4f}" if isinstance(de, float) else "",
                f"{pp_a[i]['diversity_ratio']:.4f}", f"{pp_b[i]['diversity_ratio']:.4f}",
                f"{pp_b[i]['diversity_ratio'] - pp_a[i]['diversity_ratio']:.4f}",
                f"{pp_a[i]['answer_entropy']:.4f}", f"{pp_b[i]['answer_entropy']:.4f}",
                f"{pp_b[i]['answer_entropy'] - pp_a[i]['answer_entropy']:.4f}",
                pp_a[i]["count_below_mean_length"], pp_b[i]["count_below_mean_length"],
                pp_a[i]["count_above_mean_length"], pp_b[i]["count_above_mean_length"],
                f"{pp_a[i]['mean_below_reward_div']:.4f}", f"{pp_b[i]['mean_below_reward_div']:.4f}",
                f"{pp_a[i]['mean_above_reward_div']:.4f}", f"{pp_b[i]['mean_above_reward_div']:.4f}",
            ])
    print(f"[Saved] {csv_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Compare rollout JSONL outputs from different training stages."
    )
    parser.add_argument(
        "--inputs", nargs="+", required=True,
        help="Paths to JSONL files to compare (produced by rollouts.py)"
    )
    parser.add_argument(
        "--labels", nargs="+", default=None,
        help="Optional display labels for each file (same order as --inputs). "
             "Defaults to filenames without extension."
    )
    parser.add_argument(
        "--output_dir", type=str, default="compare_plots",
        help="Directory to save plots and CSV (default: compare_plots)"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Derive labels
    if args.labels:
        if len(args.labels) != len(args.inputs):
            print(f"Error: --labels count ({len(args.labels)}) must match --inputs count ({len(args.inputs)})")
            sys.exit(1)
        names = args.labels
    else:
        names = [os.path.splitext(os.path.basename(p))[0] for p in args.inputs]

    # Load & compute
    all_summaries = []
    all_per_problem = []
    for path, name in zip(args.inputs, names):
        print(f"\nLoading {path} ({name}) ...")
        records = load_jsonl(path)
        print(f"  Loaded {len(records)} problems.")
        summary, per_problem = compute_file_metrics(records)
        all_summaries.append(summary)
        all_per_problem.append(per_problem)

    # Print comparison table
    print("\n")
    print_comparison_table(names, all_summaries)

    # Bar charts (always)
    plot_bar_comparison(names, all_summaries, args.output_dir)

    # Per-problem series plot
    plot_metrics_per_problem(names, all_per_problem, args.output_dir)

    # Paired analysis (2-file mode)
    if len(args.inputs) == 2:
        print(f"\n── Paired Comparison: {names[0]} vs {names[1]} ──")
        paired_comparison(names[0], names[1], all_per_problem[0], all_per_problem[1], args.output_dir)

    print(f"\nAll outputs saved to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
