import argparse
import json
import re
import os
import multiprocessing as mp
import random
from collections import Counter

import numpy as np
import torch
from transformers import AutoTokenizer

def remove_boxed(s):
    if "\\boxed{" not in s:
        return s
    left = s.find("\\boxed{") + 7
    level = 1
    right = left
    while right < len(s) and level > 0:
        if s[right] == '{': level += 1
        elif s[right] == '}': level -= 1
        right += 1
    return s[left:right-1] if level == 0 else s[left:]

def extract_answer(text):
    if not isinstance(text, str): return None
    # Use a more robust approach to find the last \boxed
    idx = text.rfind("\\boxed")
    if idx < 0: return None
    
    # Check for \boxed{...}
    if text[idx:].startswith("\\boxed{"):
        return remove_boxed(text[idx:])
    
    # Check for \boxed ... (without braces)
    match = re.search(r"\\boxed\s+([^\s$]+)", text[idx:])
    if match:
        return match.group(1).strip()
    
    return None

def is_valid_answer(s):
    if s is None:
        return False
    s_lower = str(s).strip().lower()
    invalid_patterns = ["[no_answer]", "[no answer]", "none", "n/a", "", "unknown"]
    return s_lower not in invalid_patterns

def strip_string(s):
    if not isinstance(s, str): return ""
    s = s.strip()
    # If the string itself contains \boxed, extract it first
    if "\\boxed" in s:
        ext = extract_answer(s)
        if ext: s = ext
        
    # Standard normalization: remove spaces, lowercase, and common LaTeX artifacts
    s = s.replace(" ", "").lower()
    s = s.replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
    s = s.replace("tfrac", "frac").replace("dfrac", "frac")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")
    s = s.replace("\\$", "")
    
    # Basic fraction/decimal normalization
    if s == "0.5": s = "1/2"
    if s == "1.0": s = "1"
    
    return s

SYSTEM_PROMPT = """You are a rigorous mathematical reviewer."""

USER_TEMPLATE = """Problem:
{problem}

[Hypothesis to Test]
A previous attempt at this problem resulted in the following answer:
{candidate_answer}

[Task]
Act as a rigorous mathematical reviewer. 
1. Reverse Verification Stage: Treat the previous answer ({candidate_answer}) as a given hypothesis. Plug this answer BACK into the original problem conditions. Perform a rigorous backward-substitution to check if it satisfies all constraints or if it leads to a mathematical contradiction. You MUST conclude this stage by explicitly stating either "Verification Result: True" (if the hypothesis perfectly satisfies all conditions) or "Verification Result: False" (if it leads to any contradiction).
2. Solution Stage: Based on the insights gained from your reverse verification, explore a robust reasoning path to independently solve the problem from scratch.

You MUST strictly use the following XML format for your response:
<reverse_verification>
(Your step-by-step backward substitution checking if {candidate_answer} contradicts the problem conditions)
Verification Result: [True/False]
</reverse_verification>
<final_solution>
(Your complete, alternative step-by-step mathematical derivation)
Therefore, the final answer is \\boxed{{...}}
</final_solution>"""

def parse_verification_response(response_text):
    verif_res = None
    lower_text = response_text.lower()
    if "verification result: true" in lower_text or "verification result:true" in lower_text:
        verif_res = True
    elif "verification result: false" in lower_text or "verification result:false" in lower_text:
        verif_res = False
        
    final_ans = None
    raw = extract_answer(response_text)
    if raw is not None:
        final_ans = strip_string(raw)
        
    return verif_res, final_ans

def _worker_process(rank_idx, physical_gpu_id, args, sampling_kwargs, prompts_chunk, return_dict):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)
    from vllm import LLM, SamplingParams
    
    print(f"[Worker GPU {physical_gpu_id}] Initializing vLLM for {len(prompts_chunk)} prompts...")
    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        tensor_parallel_size=1, 
        trust_remote_code=True,
        dtype="auto",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
    )
    
    sampling_params = SamplingParams(**sampling_kwargs)
    
    print(f"[Worker GPU {physical_gpu_id}] Generating...")
    outputs = llm.generate(prompts_chunk, sampling_params)
    
    parsed_results = []
    for out in outputs:
        verifs = []
        finals = []
        for o in out.outputs:
            v, f = parse_verification_response(o.text)
            verifs.append(v)
            finals.append(f)
            
        valid_verifs = [v for v in verifs if v is not None]
        final_verif = Counter(valid_verifs).most_common(1)[0][0] if valid_verifs else None
        
        valid_finals = [f for f in finals if f is not None]
        final_derived = Counter(valid_finals).most_common(1)[0][0] if valid_finals else None
        
        parsed_results.append({
            "verif_res": final_verif,
            "final_derived": final_derived,
            "individual_votes": [{"v": v, "f": f} for v, f in zip(verifs, finals)]
        })
        
    print(f"[Worker GPU {physical_gpu_id}] Finished chunk.")
    return_dict[rank_idx] = parsed_results

def print_analysis(data):
    total = len(data)
    if total == 0:
        print("No data to analyze.")
        return

    # --- Pass@K 召回率相关指标 ---
    total_raw_candidates = 0
    raw_pass_probs = 0       # 原始候选集包含GT的题目数 (Recall Upper Bound)
    filtered_pass_probs = 0  # 验证+修正后包含GT的题目数
    gt_in_candidates = 0    
    
    # --- Maj@1 端到端系统准确率 ---
    raw_maj1_acc = 0         # 基于原始候选集直接多数投票的准确率
    filtered_maj1_acc = 0    # 经过验证和修正后多数投票的准确率

    # --- 验证器底层微观指标 (Micro) ---
    TP, FN, FP, TN = 0, 0, 0, 0
    parse_failures = 0       # 格式解析失败计数
    contra_t_but_diff, contra_f_but_same, total_verifications = 0, 0, 0

    for item in data:
        gt = strip_string(str(item.get("answer", "")))
        original_answers = item.get("extracted_answers", [])
        
        # 1. 获取并清洗原始候选集
        raw_candidates_all = [strip_string(str(a)) for a in original_answers]
        raw_candidates = [p for p in raw_candidates_all if is_valid_answer(p)]
        total_raw_candidates += len(set(raw_candidates))
        
        # 2. 计算 Baseline: 原始候选集表现
        if raw_candidates:
            # 计算原始的多数投票结果
            most_common_raw = Counter(raw_candidates).most_common(1)[0][0]
            if most_common_raw == gt:
                raw_maj1_acc += 1
            
            # 计算原始的 Pass@K
            if any(c == gt for c in set(raw_candidates)):
                raw_pass_probs += 1
                gt_in_candidates += 1

        # 3. 处理验证结果
        verifs = item.get("candidate_verifications", {}) 
        true_set = []                # 被判定为 True 的候选答案
        final_extracted_answers = [] # 模型最后独立求解给出的有效答案
        
        for cand, res in verifs.items():
            total_verifications += 1
            v_res, f_ans = res["verif_res"], res["final_derived"]
            
            norm_cand = strip_string(cand)
            
            # 追踪解析失败
            if v_res is None:
                parse_failures += 1
            else:
                # 构建混淆矩阵
                if norm_cand == gt:
                    if v_res is True: TP += 1
                    elif v_res is False: FN += 1
                else:
                    if v_res is True: FP += 1
                    elif v_res is False: TN += 1
            
            # 无论是哪个验证结果，作为兜底投票，收集模型的独立解答
            if is_valid_answer(f_ans):
                final_extracted_answers.append(f_ans)
                
            # 将多数投票判为True的候选答案放入 true_set
            if v_res is True:
                true_set.append(norm_cand)
                
            # 逻辑一致性分析 (按要求：对每1个采样的结果独立进行统计)
            for vote in res.get("individual_votes", [{"v": v_res, "f": f_ans}]):
                indiv_v = vote["v"]
                indiv_f = vote["f"]
                
                if indiv_v is True:
                    if is_valid_answer(indiv_f) and norm_cand != indiv_f: 
                        contra_t_but_diff += 1
                elif indiv_v is False:
                    if is_valid_answer(indiv_f) and norm_cand == indiv_f: 
                        contra_f_but_same += 1
        
        # 4. 计算 Filtered Pass@K: 集合中是否还保留了GT
        if gt in true_set or any(ans == gt for ans in final_extracted_answers):
            filtered_pass_probs += 1

        # 5. 计算经过验证Pipeline后的最终系统输出 (Maj@1)
        # 策略：
        # 1. 优先从 true_set（验证通过的候选答案）中选择。
        # 2. 由于对候选集进行了去重，单纯在 true_set 上做 Counter 无法体现原始分布。
        # 3. 因此，我们权衡原始频率：在 true_set 中选择原始出现频率最高的答案。
        # 4. 如果 true_set 为空，则对 final_extracted_answers 进行普通多数投票兜底。
        
        final_decision = None
        if true_set:
            # 在验证通过的集合中，选取原始原始频率最高的一个
            # raw_candidates_all 包含了未去重的原始分布
            true_set_counts = Counter([p for p in raw_candidates_all if p in true_set])
            if true_set_counts:
                final_decision = true_set_counts.most_common(1)[0][0]
        
        if not final_decision and final_extracted_answers:
            # 兜底：如果没一个验证通过，则看模型独立生成的答案分布
            final_decision = Counter(final_extracted_answers).most_common(1)[0][0]

        if final_decision == gt:
            filtered_maj1_acc += 1

    # --- 最终指标计算与打印 ---
    total_eval_strict = max(1, total_verifications) 
    micro_acc = (TP + TN) / total_eval_strict
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2*(precision*recall)/(precision+recall) if (precision+recall)>0 else 0

    print("\n" + "="*90)
    print(" " * 25 + "🚀 稳健版：深度验证与系统效能评估报告 🚀")
    print("="*90)
    
    print(f"\n[1] 系统端到端准确率 (Pipeline Real Accuracy - Majority Voting@1):")
    print(f"    - Baseline Acc (原始直接投票): {raw_maj1_acc/total:.2%}")
    print(f"    - Post-Verify Acc (验证纠错后投票): {filtered_maj1_acc/total:.2%}")
    print(f"    - 🏆 净准确率提升 (Net Gain): {(filtered_maj1_acc - raw_maj1_acc)/total:+.2%}")
    
    print(f"\n[2] 答案召回率折损 (Oracle Pass@K):")
    print(f"    - Initial Pass@K: {raw_pass_probs/total:.2%} (原始候选包含正确答案的比例)")
    print(f"    - Filtered Pass@K: {filtered_pass_probs/total:.2%} (经过验证清洗后，依然保留正确答案的比例)")
    print(f"    - Recall Loss: {(filtered_pass_probs - raw_pass_probs)/total:+.2%}")
    
    print(f"\n[3] 验证器客观表现 (Micro Metrics):")
    print(f"    - Strict Micro Accuracy: {micro_acc:.2%} (含格式解析失败惩罚)")
    print(f"    - Precision: {precision:.2%}, Recall: {recall:.2%}, F1: {f1:.4f}")
    print(f"    - Details: TP={TP}, TN={TN}, FP={FP}, FN={FN}, Parse Failures={parse_failures}")
    
    print(f"\n[4] 逻辑一致性分析 (Logical Consistency):")
    print(f"    - Type A (T-Diff): {contra_t_but_diff} (判定为True，但独立求解给出不同答案)")
    print(f"    - Type B (F-Same): {contra_f_but_same} (判定为False，但独立求解依然给出相同答案)")
    print("="*90 + "\n")

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading {args.input_file}")
    data = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): data.append(json.loads(line))
    print(f"Loaded {len(data)} problems")

    print("Extracting Candidates and Building Prompts...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    job_map = [] 
    prompts = []
    
    for idx, item in enumerate(data):
        answers = item.get("extracted_answers", [])
        norm_candidates_all = [strip_string(str(a)) for a in answers]
        norm_candidates = list(set([p for p in norm_candidates_all if is_valid_answer(p)]))
        
        # Consistent baseline maj calculation
        if norm_candidates:
            item["original_maj_answer"] = Counter([p for p in norm_candidates_all if is_valid_answer(p)]).most_common(1)[0][0]
        else:
            item["original_maj_answer"] = None
            
        for norm_cand in norm_candidates:
            content = USER_TEMPLATE.format(problem=item["problem"], candidate_answer=norm_cand)
            try:
                prompt_text = tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": content},
                    ],
                    tokenize=False,
                    add_generation_prompt=True
                )
                prompt_text += "<reverse_verification>\n"
            except:
                prompt_text = f"{SYSTEM_PROMPT}\n\n{content}\n\n<reverse_verification>\n"
            prompts.append(prompt_text)
            job_map.append((idx, norm_cand))
            
    print(f"  Total unique candidates flattened: {len(prompts)}")
    if not prompts:
        print("No prompts to verify.")
        return

    # DP settings
    env_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env_cvd:
        physical_devices = env_cvd.split(",")
    else:
        num_gpus = torch.cuda.device_count() or 1
        physical_devices = [str(i) for i in range(num_gpus)]
    
    num_workers = len(physical_devices)
    chunk_size = (len(prompts) + num_workers - 1) // num_workers
    chunks = [prompts[i:i+chunk_size] for i in range(0, len(prompts), chunk_size)]
    
    sampling_kwargs = dict(
        n=args.num_return_sequences,
        temperature=args.temperature,
        top_p=args.top_p if hasattr(args, 'top_p') else 1.0,
        max_tokens=args.max_verify_tokens,
        stop=["<|eot_id|>", "</s>", "<|im_end|>"],
    )

    print(f"Manually spawning {len(chunks)} non-daemonic processes for DP...")
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []
    
    for rank_idx, chunk in enumerate(chunks):
        p_gpu = physical_devices[rank_idx]
        p = mp.Process(target=_worker_process, args=(rank_idx, p_gpu, args, sampling_kwargs, chunk, return_dict))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
        
    print("All processes finished. Aggregating results...")
    all_final_results = []
    for i in range(len(chunks)):
        all_final_results.extend(return_dict[i])
        
    for item in data:
        item["candidate_verifications"] = {}
        item["verified_true_set"] = []
        
    for i, res in enumerate(all_final_results):
        idx, cand = job_map[i]
        data[idx]["candidate_verifications"][cand] = res
        if res["verif_res"] is True:
            data[idx]["verified_true_set"].append(cand)
            
    print_analysis(data)

    if args.output_file:
        print(f"\nSaving results to {args.output_file}")
        with open(args.output_file, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser(description="Evaluate Self-Verification as an Answer Filter")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="filter_results.jsonl")
    parser.add_argument("--num_return_sequences", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_verify_tokens", type=int, default=8192)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--enforce_eager", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
