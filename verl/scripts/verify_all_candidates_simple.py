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

def extract_answer(text):
    if not isinstance(text, str): return None
    idx = text.rfind("\\boxed")
    if idx < 0: return None
    if text[idx:].startswith("\\boxed{"):
        return remove_boxed(text[idx:])
    match = re.search(r"\\boxed\s+([^\s$]+)", text[idx:])
    if match: return match.group(1).strip()
    return None

def is_valid_answer(s):
    if s is None: return False
    s_lower = str(s).strip().lower()
    invalid_patterns = ["[no_answer]", "[no answer]", "none", "n/a", "", "unknown"]
    return s_lower not in invalid_patterns

def strip_string(s):
    if not isinstance(s, str): return ""
    s = s.strip()
    if "\\boxed" in s:
        ext = extract_answer(s)
        if ext: s = ext
    s = s.replace(" ", "").lower()
    s = s.replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
    s = s.replace("tfrac", "frac").replace("dfrac", "frac")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")
    s = s.replace("\\$", "")
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
Treat the previous answer ({candidate_answer}) as a given hypothesis. Plug this answer BACK into the original problem conditions. Perform a rigorous backward-substitution to check if it satisfies all constraints or if it leads to a mathematical contradiction. 

You MUST strictly use the following XML format for your response:
<reverse_verification>
(Your step-by-step backward substitution checking if {candidate_answer} contradicts the problem conditions)
Verification Result: [True/False]
</reverse_verification>"""

def parse_verification_response(response_text):
    lower_text = response_text.lower()
    if "verification result: true" in lower_text or "verification result:true" in lower_text:
        return True
    elif "verification result: false" in lower_text or "verification result:false" in lower_text:
        return False
    return None

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
        for o in out.outputs:
            verifs.append(parse_verification_response(o.text))
            
        valid_verifs = [v for v in verifs if v is not None]
        final_verif = Counter(valid_verifs).most_common(1)[0][0] if valid_verifs else None
        
        parsed_results.append({
            "verif_res": final_verif,
            "individual_votes": verifs 
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
    total_verifications = 0 

    for item in data:
        gt = strip_string(str(item.get("answer", "")))
        original_answers = item.get("extracted_answers", [])
        
        raw_candidates_all = [strip_string(str(a)) for a in original_answers]
        raw_candidates = [p for p in raw_candidates_all if is_valid_answer(p)]
        raw_freq = Counter(raw_candidates)  # 第一阶段原始频率分布
        total_raw_candidates += len(set(raw_candidates))
        
        most_common_raw = None
        if raw_candidates:
            most_common_raw = raw_freq.most_common(1)[0][0]
            if most_common_raw == gt:
                raw_maj1_acc += 1
            
            if any(c == gt for c in set(raw_candidates)):
                raw_pass_probs += 1
                gt_in_candidates += 1

        verifs = item.get("candidate_verifications", {}) 
        true_set = []                # 验证通过的候选答案
        cand_true_freq = {}          # 每个候选答案的 True 出现频率 (权重)
        
        for cand, res in verifs.items():
            total_verifications += 1
            norm_cand = strip_string(cand)
            
            # 从 individual_votes 计算 True/False 频率
            individual = res.get("individual_votes", [])
            true_count = sum(1 for v in individual if v is True)
            false_count = sum(1 for v in individual if v is False)
            total_votes = true_count + false_count
            true_freq = true_count / total_votes if total_votes > 0 else 0.0
            
            # 使用多数投票结果(verif_res)作为 TP/TN/FP/FN 的判定依据
            v_res = res["verif_res"]
            if v_res is None:
                parse_failures += 1
            else:
                if norm_cand == gt:
                    if v_res is True: TP += 1
                    elif v_res is False: FN += 1
                else:
                    if v_res is True: FP += 1
                    elif v_res is False: TN += 1
            
            # 对于所有 top k 候选答案记录 True 频率权重
            cand_true_freq[norm_cand] = true_freq
            
            if true_count > false_count:
                true_set.append(norm_cand)
                
        # 4. 计算 Filtered Pass@K: 集合中是否还保留了GT
        if gt in true_set:
            filtered_pass_probs += 1

        # 5. 伪标签选择：在所有被验证的 top k 回答中，选 True 频率最高的答案
        #    如果 True 频率相同，则按第一阶段原始出现频率来选
        final_decision = None
        if cand_true_freq:
            # 按 (true_freq 降序, 原始频率 降序) 排序
            candidates_sorted = sorted(
                cand_true_freq.keys(),
                key=lambda c: (cand_true_freq.get(c, 0), raw_freq.get(c, 0)),
                reverse=True
            )
            final_decision = candidates_sorted[0]
        
        # 兜底：如果没任何验证数据，则退回原始的多数投票
        if not final_decision and most_common_raw:
            final_decision = most_common_raw

        # 将这些核心答案信息存入数据结构，以便最后保存到 JSONL 里
        item["filtered_pseudo_label"] = final_decision
        item["pseudo_label_true_freq"] = cand_true_freq.get(final_decision, 0.0) if cand_true_freq else 0.0
        item["baseline_correct"] = bool(most_common_raw == gt)
        item["filtered_correct"] = bool(final_decision == gt)

        if final_decision == gt:
            filtered_maj1_acc += 1

    total_eval_strict = max(1, total_verifications) 
    micro_acc = (TP + TN) / total_eval_strict
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2*(precision*recall)/(precision+recall) if (precision+recall)>0 else 0

    print("\n" + "="*90)
    print(" " * 25 + "🚀 纯粹验证版：基础验证能力评估 🚀")
    print("="*90)
    
    print(f"\n[1] 系统端到端准确率 (Pipeline Real Accuracy - Majority Voting@1):")
    print(f"    - Baseline Acc (原始直接投票): {raw_maj1_acc/total:.2%}")
    print(f"    - Post-Verify Acc (验证集内投票): {filtered_maj1_acc/total:.2%}")
    print(f"    - 🏆 净准确率提升 (Net Gain): {(filtered_maj1_acc - raw_maj1_acc)/total:+.2%}")
    
    print(f"\n[2] 答案召回率 (Oracle Pass@K):")
    print(f"    - Initial Pass@K: {raw_pass_probs/total:.2%} (原始候选包含正确答案比例)")
    print(f"    - Filtered Pass@K: {filtered_pass_probs/total:.2%} (经过验证清洗保留比例)")
    print(f"    - Recall Loss: {(filtered_pass_probs - raw_pass_probs)/total:+.2%}")
    
    print(f"\n[3] 验证器客观表现 (Micro Metrics):")
    print(f"    - Strict Micro Accuracy: {micro_acc:.2%} (含格式解析失败惩罚)")
    print(f"    - Precision: {precision:.2%}, Recall: {recall:.2%}, F1: {f1:.4f}")
    print(f"    - Details: TP={TP}, TN={TN}, FP={FP}, FN={FN}, Parse Failures={parse_failures}")
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
        valid_candidates = [p for p in norm_candidates_all if is_valid_answer(p)]
        
        if valid_candidates:
            item["original_maj_answer"] = Counter(valid_candidates).most_common(1)[0][0]
        else:
            item["original_maj_answer"] = None
        
        # 选取出现频率 Top-K 的候选答案进行验证 (如果 top_k <= 0, 则使用全部答案)
        freq = Counter(valid_candidates)
        if args.top_k > 0:
            top_k_candidates = [ans for ans, _ in freq.most_common(args.top_k)]
        else:
            top_k_candidates = [ans for ans, _ in freq.most_common()]
            
        for norm_cand in top_k_candidates:
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
        stop=["</reverse_verification>", "<|eot_id|>", "</s>", "<|im_end|>"],
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
    parser = argparse.ArgumentParser(description="Evaluate Simple Self-Verification as an Answer Filter")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="filter_results_simple.jsonl")
    parser.add_argument("--num_return_sequences", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_verify_tokens", type=int, default=4096)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--enforce_eager", type=bool, default=True)
    parser.add_argument("--top_k", type=int, default=-1, help="Number of top-frequency candidates per problem to verify. Set to 0 or -1 to verify ALL unique candidates.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
