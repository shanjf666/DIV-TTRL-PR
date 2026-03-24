"""
Teacher-Prompt Generation and Voting Experiment (Parallel 32-Sample Version)

For each problem with SC < 0.3, discards the original rollouts and extracts 
the Top-K most frequent (but likely incorrect) answers.
Constructs a single new prompt for the problem using either:
  1. 'trap': Warns the model to avoid the Top 3 common mistakes.
  2. 'mcq': Presents the Top 5 answers as A-E options for reference.
Uses vLLM to sample N=32 completely new paths in parallel.
Extracts the \boxed{} answers and performs Majority Voting.

Usage Example:
CUDA_VISIBLE_DEVICES=0 python scripts/teacher_voting_experiment.py \
    --model_path /data/home/jianfeng/data/models/modelscope_cache/models/Qwen/Qwen3-4B-Base \
    --input_file scripts/base.jsonl \
    --strategy trap \
    --temperature 0.6 \
    --n_samples 32
    --output_file out_trap_06.jsonl
CUDA_VISIBLE_DEVICES=0 python scripts/teacher_voting_experiment.py \
    --model_path /data/home/jianfeng/data/models/modelscope_cache/models/Qwen/Qwen3-4B-Base \
    --input_file scripts/base.jsonl \
    --strategy trap \
    --temperature 1 \
    --n_samples 32
    --output_file out_trap_10.jsonl
CUDA_VISIBLE_DEVICES=0 python scripts/teacher_voting_experiment.py \
    --model_path /data/home/jianfeng/data/models/modelscope_cache/models/Qwen/Qwen3-4B-Base \
    --input_file scripts/base.jsonl \
    --strategy mcq \
    --temperature 0.6 \
    --n_samples 32
    --output_file out_mcq_06.jsonl
CUDA_VISIBLE_DEVICES=0 python scripts/teacher_voting_experiment.py \
    --model_path /data/home/jianfeng/data/models/modelscope_cache/models/Qwen/Qwen3-4B-Base \
    --input_file scripts/base.jsonl \
    --strategy mcq \
    --temperature 1 \
    --n_samples 32
    --output_file out_mcq_10.jsonl
"""

import json
import argparse
from collections import Counter
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# =====================================================
# 1. Utility Functions
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

def get_top_k_answers(extracted_list, k=3):
    """Get the top K most common valid answers from the list."""
    valid_ans = [strip_string(a) for a in extracted_list if a and a != "[NO_ANSWER]"]
    if not valid_ans:
        return []
    counter = Counter(valid_ans)
    most_common = counter.most_common(k)
    return [ans for ans, count in most_common]


# =====================================================
# 2. Base Model Specific Prompts
# =====================================================

TRAP_PROMPT_TEMPLATE = """Problem:
{problem}

[Note]
This is a highly challenging problem. In previous attempts, many students fell into logical traps or calculation errors and incorrectly answered: {trap_answers}.

Please think carefully, avoid these common mistakes, and write down a rigorous step-by-step solution.
Conclude your final derived result strictly inside a \\boxed{{}} environment.

Solution:
"""

MCQ_PROMPT_TEMPLATE = """Problem:
{problem}

Here are 5 possible answers derived from different methods. The correct answer might be one of them, or it might be none of them:
{mcq_options}

Solve the problem step-by-step through independent derivation. Do not guess. If your final calculated result matches one of the options above, output that result. If your result is different from all options, output your own result.
Strictly put your final answer inside \\boxed{{}}.

Step-by-step derivation:
"""


def build_new_prompt(problem, extracted_list, strategy, tokenizer, model_path):
    """Build the prompt based on the chosen strategy (trap or mcq)."""
    if strategy == "trap":
        top_3 = get_top_k_answers(extracted_list, k=3)
        if not top_3:
            trap_str = "some incorrect values"
        else:
            trap_str = ", ".join(top_3)
        content = TRAP_PROMPT_TEMPLATE.format(problem=problem, trap_answers=trap_str)
        
    elif strategy == "mcq":
        top_5 = get_top_k_answers(extracted_list, k=5)
        labels = ["A)", "B)", "C)", "D)", "E)"]
        options_str = ""
        for i in range(5):
            ans = top_5[i] if i < len(top_5) else "None of the above"
            options_str += f"{labels[i]} {ans}\n"
        content = MCQ_PROMPT_TEMPLATE.format(problem=problem, mcq_options=options_str.strip())
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    name = model_path.lower()
    if "instruct" in name or "llama" in name:
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
            
    return content


# =====================================================
# 3. Analysis & Reporting
# =====================================================

def print_full_comparison(results, strategy, temp):
    buckets = [
        ("Low (SC<0.3)",  lambda r: r["sc_score"] < 0.3),
        ("Mid (0.3-0.7)", lambda r: 0.3 <= r["sc_score"] < 0.7),
        ("High (≥0.7)",   lambda r: r["sc_score"] >= 0.7),
        ("ALL",           lambda r: True),
    ]

    print("\n" + "=" * 105)
    print(f"TEACHER-PROMPT RESULTS | Strategy: {strategy.upper()} | Temp: {temp}")
    print("=" * 105)

    header = (
        f"  {'Bucket':<16} {'N':>5}  "
        f"{'Orig MAJ Acc':>14}  {'Teacher MAJ Acc':>17}  {'Δ(Accuracy)':>13}  "
        f"{'Rescued':>7}  {'Harmed':>6}"
    )
    print(header)
    print("  " + "-" * 101)

    for bname, bfn in buckets:
        bucket = [r for r in results if bfn(r)]
        if not bucket:
            continue
        n = len(bucket)

        maj_correct = sum(1 for r in bucket if r["maj_correct"])
        teacher_correct = sum(1 for r in bucket if r["teacher_maj_correct"])

        rescued = sum(1 for r in bucket if not r["maj_correct"] and r["teacher_maj_correct"])
        harmed = sum(1 for r in bucket if r["maj_correct"] and not r["teacher_maj_correct"])

        maj_acc = maj_correct / n
        teacher_acc = teacher_correct / n
        delta = teacher_acc - maj_acc

        print(
            f"  {bname:<16} {n:>5}  "
            f"{maj_acc:>13.1%}    {teacher_acc:>15.1%}  {delta:>12.1%}  "
            f"{rescued:>7}  {harmed:>6}"
        )

    low = [r for r in results if r["sc_score"] < 0.3]
    if not low:
        return

    print(f"\n  Transition Matrix (Orig MAJ → Teacher MAJ) [Low SC]:")
    both_correct = sum(1 for r in low if r["maj_correct"] and r["teacher_maj_correct"])
    rescued = sum(1 for r in low if not r["maj_correct"] and r["teacher_maj_correct"])
    harmed = sum(1 for r in low if r["maj_correct"] and not r["teacher_maj_correct"])
    both_wrong = sum(1 for r in low if not r["maj_correct"] and not r["teacher_maj_correct"])

    print(f"    Orig MAJ ✓ → Teacher MAJ ✓ (kept):      {both_correct}")
    print(f"    Orig MAJ ✗ → Teacher MAJ ✓ (rescued):   {rescued}")
    print(f"    Orig MAJ ✓ → Teacher MAJ ✗ (harmed):    {harmed}")
    print(f"    Orig MAJ ✗ → Teacher MAJ ✗ (still ✗):   {both_wrong}")
    print(f"    Net gain: {rescued - harmed:+d}")


# =====================================================
# 4. Main
# =====================================================

def main(args):
    print(f"Loading {args.input_file}")
    data = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"Loaded {len(data)} problems")

    low_sc_count = sum(1 for d in data if d.get("sc_score", 1.0) < 0.3)
    print(f"Low-consistency (SC < 0.3) problems: {low_sc_count}")
    if low_sc_count == 0:
        return

    print(f"Loading tokenizer & vLLM engine for {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        trust_remote_code=True,
        dtype="auto",
    )
    
    # We use temperature > 0 and n_samples for parallel exploration
    sampling_params = SamplingParams(
        n=args.n_samples,
        temperature=args.temperature,
        top_p=0.95 if args.temperature > 0 else 1.0,
        max_tokens=args.max_tokens,
        stop=["<|eot_id|>", "</s>", "<|im_end|>", "Q:"],
    )

    all_prompts = []
    metadata_map = [] # To map prompt index back to problem_idx
    all_results = []

    for idx, item in enumerate(data):
        sc_score = item.get("sc_score", 1.0)
        gt_norm = strip_string(str(item.get("answer", "")))
        maj_answer = strip_string(str(item.get("sc_answer", "")))
        
        result = {
            "idx": idx,
            "problem": item["problem"],
            "gt": item.get("answer", ""),
            "gt_norm": gt_norm,
            "sc_score": sc_score,
            "maj_answer": maj_answer,
            "maj_correct": (maj_answer == gt_norm),
            "was_reprompted": False,
            "teacher_maj_answer": maj_answer,
            "teacher_maj_correct": (maj_answer == gt_norm),
            "teacher_sc_score": 0.0,
            "teacher_derived_answers": []
        }
        all_results.append(result)

        if sc_score < 0.3:
            extracted = item.get("extracted_answers", [])
            prompt = build_new_prompt(item["problem"], extracted, args.strategy, tokenizer, args.model_path)
            
            all_prompts.append(prompt)
            metadata_map.append(idx)

    print(f"Total Unique Teacher Prompts (1 per low-SC problem): {len(all_prompts)}")
    print(f"Parallel samples per prompt: {args.n_samples} | Temperature: {args.temperature}")
    
    print("Generating Teacher Verification responses...")
    outputs = llm.generate(all_prompts, sampling_params)

    # Process exactly 32 outputs per problem
    for i, output in enumerate(outputs):
        p_idx = metadata_map[i]
        
        answers_for_this_prob = []
        for gen_out in output.outputs:
            raw_ans = extract_answer(gen_out.text)
            norm_ans = strip_string(raw_ans) if raw_ans else "[NO_ANSWER]"
            answers_for_this_prob.append(norm_ans)
            
        all_results[p_idx]["was_reprompted"] = True
        all_results[p_idx]["teacher_derived_answers"] = answers_for_this_prob
        
        valid_answers = [a for a in answers_for_this_prob if a != "[NO_ANSWER]"]
        counter = Counter(valid_answers)
        most_common = counter.most_common(1)
        
        if most_common:
            best_ans, count = most_common[0]
            teacher_sc_score = count / len(valid_answers)
        else:
            best_ans = "[NO_ANSWER]"
            teacher_sc_score = 0.0
            
        all_results[p_idx]["teacher_maj_answer"] = best_ans
        all_results[p_idx]["teacher_maj_correct"] = (best_ans == all_results[p_idx]["gt_norm"])
        all_results[p_idx]["teacher_sc_score"] = teacher_sc_score

    print_full_comparison(all_results, args.strategy, args.temperature)
    
    if args.output_file:
        print(f"\nSaving detailed results to {args.output_file}")
        with open(args.output_file, "w", encoding="utf-8") as f:
            for r in all_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Teacher-Prompt Generation and Voting Experiment")
    parser.add_argument("--model_path", type=str, required=True, help="HuggingFace model path")
    parser.add_argument("--input_file", type=str, default="scripts/base.jsonl", help="Input JSONL")
    parser.add_argument("--output_file", type=str, default="scripts/teacher_voting_results.jsonl", help="Output JSONL")
    
    # NEW ARGUMENTS
    parser.add_argument("--strategy", type=str, choices=["trap", "mcq"], default="trap", 
                        help="'trap' explicitly warns against top 3 errors. 'mcq' lists top 5 as A-E options.")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--n_samples", type=int, default=32, help="Number of parallel samples per prompt")
    
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="vLLM tensor parallel size")
    parser.add_argument("--max_model_len", type=int, default=4096, help="Max context length to restrict KV cache size")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8, help="Fraction of GPU memory for vLLM")
    parser.add_argument("--enforce_eager", action="store_true", help="Disable CUDA graphs (saves some memory)")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Max tokens for Teacher generation")
    args = parser.parse_args()
    main(args)