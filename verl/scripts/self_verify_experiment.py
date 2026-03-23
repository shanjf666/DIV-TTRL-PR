"""
Explore Alternative Answers Offline Experiment

Usage:
    python scripts/self_verify_experiment.py \
        --model_path Qwen/Qwen2.5-7B \
        --input_file qwen64.jsonl \
        --output_file explore_results.jsonl
"""

import json
import random
import argparse
import numpy as np
from collections import Counter

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# =====================================================
# Utility functions
# =====================================================

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    return None if right_brace_idx is None else string[idx : right_brace_idx + 1]


def remove_boxed(s):
    if s is None:
        return None
    if "\\boxed " in s:
        return s.replace("\\boxed ", "")
    if s.startswith("\\boxed{") and s.endswith("}"):
        return s[len("\\boxed{") : -1]
    return s


def strip_string(string):
    if string is None:
        return ""
    string = str(string)
    # Handle ground truth boxed format if present
    if "\\boxed{" in string:
        extracted = extract_answer(string)
        if extracted is not None:
            string = extracted
    string = string.replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "").replace(" ", "")
    if string == "0.5":
        string = "\\frac{1}{2}"
    return string


def extract_answer(text):
    boxed = last_boxed_only_string(text)
    if boxed is None:
        return None
    return remove_boxed(boxed)


# =====================================================
# Prompt
# =====================================================

SYSTEM_PROMPT = """You are an expert mathematical problem solver. Your task is to explore alternative solutions to a given problem."""

USER_TEMPLATE = """[Problem]
{problem}

[Previous Answer]
A previous attempt resulted in the following answer:
{majority_answer}

[Task]
Please solve the problem step by step.
IMPORTANT: You must NOT generate the previous answer ({majority_answer}). Please explore a completely different reasoning path and provide an alternative answer.
Conclude your final answer strictly enclosed in \\boxed{{}}."""

def build_explore_prompt(problem, majority_answer, tokenizer):
    content = USER_TEMPLATE.format(problem=problem, majority_answer=majority_answer)
    if tokenizer:
        try:
            prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                tokenize=False,
                add_generation_prompt=True
            )
            return prompt
        except Exception:
            pass
    return f"{SYSTEM_PROMPT}\n\n{content}"

def parse_explore_response(response_text):
    raw = extract_answer(response_text)
    if raw is None:
        return None
    return strip_string(raw)

# =====================================================
# Candidate Extraction
# =====================================================

def get_majority_and_consistency(answers):
    valid = [a for a in answers if a not in ["[NO_ANSWER]", ""]]
    if not valid:
        return None, 0.0
    freq = Counter(valid)
    N = len(valid)
    majority = freq.most_common(1)[0][0]
    consistency = freq[majority] / N
    return majority, consistency

# =====================================================
# Single Experiment Runner
# =====================================================

def run_experiment(data, llm, tokenizer, sampling_params, threshold=0.3):
    print(f"\n{'='*80}")
    print(f"Experiment: Explore Alternative Answers")
    print(f"{'='*80}")

    results = []
    explore_indices = []
    explore_prompts = []

    for idx, item in enumerate(data):
        answers = item.get("extracted_answers", [])
        majority, consistency = get_majority_and_consistency(answers)
        gt_norm = strip_string(str(item.get("answer", "")))
        maj_answer = strip_string(str(item.get("sc_answer", majority)))

        r = {
            "consistency": consistency,
            "gt_norm": gt_norm,
            "maj_answer": maj_answer,
            "maj_correct": (maj_answer == gt_norm)
        }
        results.append(r)

        # Only explore for low-consistency problems (e.g., <= threshold)
        if consistency <= threshold:
            prompt = build_explore_prompt(item["problem"], maj_answer, tokenizer)
            explore_indices.append(idx)
            explore_prompts.append(prompt)

    print(f"  Low-consistency problems to explore: {len(explore_indices)}")

    if explore_prompts:
        outputs = llm.generate(explore_prompts, sampling_params)

        for i, output in enumerate(outputs):
            idx = explore_indices[i]
            responses = [out.text for out in output.outputs]
            
            extracted_answers = []
            for resp in responses:
                raw_ans = parse_explore_response(resp)
                extracted_answers.append(raw_ans if raw_ans is not None else "[NO_ANSWER]")
            
            valid_answers = [a for a in extracted_answers if a not in ["[NO_ANSWER]", ""]]
            if valid_answers:
                counter = Counter(valid_answers)
                most_common = counter.most_common(1)[0]
                sc_answer = most_common[0]
                sc_score = most_common[1] / len(valid_answers)
            else:
                sc_answer = None
                sc_score = 0.0

            results[idx]["explore_prompt"] = explore_prompts[i]
            results[idx]["explore_responses"] = responses
            results[idx]["explored_answers"] = extracted_answers
            results[idx]["explored_sc_answer"] = sc_answer
            results[idx]["explored_sc_score"] = sc_score

    # For non-explored problems, fallback to MAJ
    for r in results:
        if "explore_responses" not in r:
            r["explore_responses"] = []
            r["explored_answers"] = []
            r["explored_sc_answer"] = r["maj_answer"]
            r["explored_sc_score"] = 0.0

    return results

# =====================================================
# Analysis
# =====================================================

def print_analysis(results):
    print(f"\n--- Exploration Analysis ---")

    buckets = [
        ("Low(<=0.3)", lambda r: r["consistency"] <= 0.3),
        ("Mid(0.3-0.7)", lambda r: 0.3 < r["consistency"] <= 0.7),
        ("High(>0.7)", lambda r: r["consistency"] > 0.7),
        ("All", lambda r: True),
    ]

    print(f"  {'Bucket':<16} {'N':<6} {'MAJ Acc':<10} {'Exp Acc':<10} {'Gain':<8}")
    print("  " + "-" * 54)

    for bname, bfn in buckets:
        bucket = [r for r in results if bfn(r)]
        if not bucket:
            continue
        n = len(bucket)
        maj_c = sum(1 for r in bucket if r["maj_correct"])
        exp_c = sum(1 for r in bucket if r.get("explored_sc_answer") == r["gt_norm"])

        maj_acc = maj_c / n
        exp_acc = exp_c / n
        gain = exp_acc - maj_acc

        print(f"  {bname:<16} {n:<6} {maj_acc:<10.1%} {exp_acc:<10.1%} {gain:<+8.1%}")

# =====================================================
# Main
# =====================================================

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    print(f"Loading {args.input_file}")
    data = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"Loaded {len(data)} problems")

    # Load tokenizer and model
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    print("Initializing vLLM...")
    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        tensor_parallel_size=torch.cuda.device_count() or 1,
        trust_remote_code=True,
        dtype="auto",
    )

    sampling_params = SamplingParams(
        n=args.num_return_sequences,
        temperature=args.temperature,
        top_p=args.top_p if hasattr(args, 'top_p') else 1.0,
        max_tokens=args.max_verify_tokens,
        stop=["<|eot_id|>", "</s>", "<|im_end|>"],
    )

    results = run_experiment(data, llm, tokenizer, sampling_params, threshold=args.threshold)
    print_analysis(results)

    # Save results
    if args.output_file:
        print(f"\nSaving results to {args.output_file}")
        with open(args.output_file, "w", encoding="utf-8") as f:
            for i, item in enumerate(data):
                r = results[i]
                record = {
                    "problem": item["problem"],
                    "answer": item.get("answer", item.get("solution")),
                    "responses": r.get("explore_responses", []),
                    "extracted_answers": r.get("explored_answers", []),
                    "sc_answer": r.get("explored_sc_answer", r["maj_answer"]),
                    "sc_score": r.get("explored_sc_score", 0.0),
                    
                    # Store original values for reference
                    "maj_answer": r["maj_answer"],
                    "consistency": r["consistency"],
                    "explore_prompt": r.get("explore_prompt", "")
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explore Alternative Answers")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="explore_results.jsonl")
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--max_verify_tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
