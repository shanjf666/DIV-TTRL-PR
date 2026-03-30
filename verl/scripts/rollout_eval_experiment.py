"""
Rollout-by-Rollout Math Evaluation Experiment

目标：
1. 对输入文件中的每一条 rollout 逐条打分（0-10）。
2. 先应用硬规则零分判定：
   - 无有效输出/乱码
   - 无法提取最终答案（缺少明确答案格式，例如 \boxed{}）
   - 大量重复/疑似 babbling
3. 对通过硬规则的 rollout，调用评审模型给分。
4. 增加不合理分数检查与剔除机制（例如超范围、无法解析、与硬规则冲突）。
5. 每道题最终答案集合由“最高有效分”的答案组成。

Usage:
    python scripts/rollout_eval_experiment.py \
        --model_path Qwen/Qwen2.5-7B \
        --input_file scripts/qwen64.jsonl \
        --output_file scripts/rollout_eval_results.jsonl
"""

import argparse
import json
import math
import random
import re
from collections import Counter

import numpy as np
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# =====================================================
# Utility
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


def extract_answer(text):
    if text is None:
        return None
    boxed = last_boxed_only_string(text)
    if boxed is None:
        return None
    return remove_boxed(boxed)


def strip_string(string):
    if string is None:
        return ""
    string = str(string)
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


def normalize_for_repeat(text):
    if not text:
        return ""
    x = text.lower()
    x = re.sub(r"\s+", " ", x)
    return x.strip()


def repetition_ratio_by_ngrams(text, n=4):
    if not text:
        return 0.0
    if len(text) < n:
        return 0.0
    ngrams = [text[i : i + n] for i in range(len(text) - n + 1)]
    if not ngrams:
        return 0.0
    cnt = Counter(ngrams)
    repeated = sum(v for v in cnt.values() if v > 1)
    return repeated / len(ngrams)


def line_repetition_ratio(text):
    if not text:
        return 0.0
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0.0
    cnt = Counter(lines)
    repeated = sum(v for v in cnt.values() if v > 1)
    return repeated / len(lines)


def is_gibberish_or_invalid_output(text):
    if text is None:
        return True
    s = str(text).strip()
    if not s:
        return True
    if len(s) < 8:
        return True
    # 大部分字符都不是常见可读字符时，视为无效输出
    printable = sum(ch.isprintable() for ch in s)
    if printable / max(1, len(s)) < 0.8:
        return True
    return False


def triggers_babbling(text, ngram_repeat_threshold=0.38, line_repeat_threshold=0.30):
    x = normalize_for_repeat(text)
    if not x:
        return False

    # 短文本不按 babbling 判定
    if len(x) < 200:
        return False

    ngram_ratio = repetition_ratio_by_ngrams(x, n=4)
    line_ratio = line_repetition_ratio(x)

    # 连续块重复检测
    repeated_block = re.search(r"(.{40,200}?)\1{2,}", x) is not None

    return (
        ngram_ratio >= ngram_repeat_threshold
        or line_ratio >= line_repeat_threshold
        or repeated_block
    )


# =====================================================
# Judge Prompt
# =====================================================

SYSTEM_PROMPT = """# Role: 资深数学阅卷专家 (Expert Mathematics Evaluator)
你现在的任务是对 AI 模型解答数学问题的过程和结果进行客观、严格的评分。"""

USER_TEMPLATE = """## 核心原则 (Core Principles)
数学解答的评估必须以逻辑严密性和结果准确性为核心。你需要甄别模型是真正理解了问题，还是在通过“幻觉”或“胡言乱语”凑答案。

## 零分否决项 (Zero-Point Triggers - Hard Fails)
如果【模型解答】触犯以下任何一条，请立刻停止评估，并给出 0 分：
1. 无有效输出或乱码：输出完全无关的内容，或者未完成解答。
2. 答案无法提取：虽然有正常的解题格式，但最终没有给出明确的结论，或者没有按照规范格式（如未包含在 \\boxed{{}} 中，或缺乏明显的“答：”标识）给出最终答案。
3. 陷入死循环/大量重复：解题过程中出现大段毫无意义的废话重复、车轱辘话，或者陷入逻辑死循环（“Babbling”）。

## 评分维度与阶梯 (Scoring Rubric: 1 - 10)
如果未触犯零分否决项，请基于以下数学特性进行打分：

* 【1-3分】思路完全错误：最终答案错误。推理方向南辕北辙，使用了完全不适用的定理或公式，存在根本性的逻辑断层。
* 【4-6分】方向正确但存在重大漏洞：使用了正确的解题思路或公式，但计算过程中出现严重错误，或者漏掉了题目中关键的限制条件（如忽略了定义域、正负号取舍等），导致最终结果错误。
* 【7-9分】逻辑自洽但存在微小瑕疵：推理过程清晰，最终答案基本正确。但可能存在符号表达不规范、分数未化简、或者某些中间步骤略显跳跃和繁琐。
* 【10分】完美解答：逻辑链条无懈可击，公式定理应用严谨，计算完全正确，最终答案清晰准确，且解题过程精炼高效。

## 评估步骤 (Evaluation Process)
在给出分数前，请先按照以下步骤进行思考：
1. 格式检查：检查是否触碰了【零分否决项】。
2. 逻辑梳理：提炼【模型解答】的核心推理路径，与标准数学逻辑进行对比。
3. 验算核对：核对解答中的关键计算步骤是否成立。
4. 综合定级：结合【评分维度与阶梯】给出最终分数。

## 输入数据 (Input Data)
[用户问题]: {question}
[模型解答]: {model_response}

## 输出格式 (Output Format)
1. 先给出一句简短评语。
2. 再严格把最终分数放到 \\boxed{{}} 里，例如 \\boxed{{8.5}}。
"""


def build_eval_prompt(question, model_response, tokenizer):
    content = USER_TEMPLATE.format(question=question, model_response=model_response)
    if tokenizer:
        try:
            prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            return prompt
        except Exception:
            pass
    return f"{SYSTEM_PROMPT}\n\n{content}"


def parse_score_from_text(text):
    if not text:
        return None

    # 优先解析 boxed 分数
    boxed = last_boxed_only_string(text)
    if boxed:
        inner = remove_boxed(boxed)
        if inner is not None:
            inner = inner.strip()
            m = re.search(r"-?\d+(?:\.\d+)?", inner)
            if m:
                try:
                    return float(m.group(0))
                except ValueError:
                    pass

    # 兼容 \box{...}
    m_box = re.search(r"\\box\{\s*(-?\d+(?:\.\d+)?)\s*\}", text)
    if m_box:
        try:
            return float(m_box.group(1))
        except ValueError:
            pass

    # 回退：全文最后一个 0-10 数字
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if nums:
        try:
            return float(nums[-1])
        except ValueError:
            return None
    return None


def is_reasonable_score(score, hard_fail):
    if score is None:
        return False
    if not isinstance(score, (int, float)):
        return False
    if not math.isfinite(float(score)):
        return False

    # 硬规则触发时分数必须是 0
    if hard_fail:
        return abs(float(score) - 0.0) < 1e-8

    # 非硬规则时，按 rubric 应在 [1, 10]
    return 1.0 <= float(score) <= 10.0


# =====================================================
# Runner
# =====================================================

def collect_responses(item):
    for key in ["responses", "rollouts", "outputs"]:
        val = item.get(key)
        if isinstance(val, list):
            return [str(x) if x is not None else "" for x in val]
    return []


def collect_extracted_answers(item, responses):
    ext = item.get("extracted_answers")
    if isinstance(ext, list) and len(ext) == len(responses):
        out = []
        for x in ext:
            if x is None:
                out.append("[NO_ANSWER]")
            else:
                sx = str(x).strip()
                out.append(sx if sx else "[NO_ANSWER]")
        return out

    out = []
    for r in responses:
        ans = extract_answer(r)
        out.append(strip_string(ans) if ans is not None else "[NO_ANSWER]")
    return out


def get_majority_and_consistency(answers):
    valid = [a for a in answers if a not in ["[NO_ANSWER]", "", None]]
    if not valid:
        return None, 0.0
    freq = Counter(valid)
    n = len(valid)
    majority = freq.most_common(1)[0][0]
    consistency = freq[majority] / n
    return majority, consistency


def run_experiment(data, llm, tokenizer, sampling_params, consistency_threshold=0.3, judge_all_samples=False):
    print(f"\n{'=' * 80}")
    print("Experiment: Rollout-by-Rollout Evaluation")
    print(f"{'=' * 80}")

    # 先做硬规则判定，再批量调用评审模型
    prompts = []
    prompt_refs = []  # (sample_idx, rollout_idx)
    records = []
    hard_fail_reason_counter = Counter()
    low_consistency_samples = 0

    for i, item in enumerate(data):
        question = item.get("problem", "")
        responses = collect_responses(item)
        extracted = collect_extracted_answers(item, responses)
        _, consistency = get_majority_and_consistency(extracted)
        is_low_consistency = consistency <= consistency_threshold
        if is_low_consistency:
            low_consistency_samples += 1
        should_judge_this_sample = judge_all_samples or is_low_consistency

        row = {
            "question": question,
            "answer": item.get("answer", item.get("solution", "")),
            "responses": responses,
            "extracted_answers": extracted,
            "consistency": consistency,
            "is_low_consistency": is_low_consistency,
            "skipped_by_consistency": not should_judge_this_sample,
            "rollout_evals": [],
        }

        for j, resp in enumerate(responses):
            no_valid_output = is_gibberish_or_invalid_output(resp)
            no_extractable_answer = extracted[j] in ["[NO_ANSWER]", "", None]
            babbling = triggers_babbling(resp)

            hard_fail = no_valid_output or no_extractable_answer or babbling
            hard_fail_reasons = []
            if no_valid_output:
                hard_fail_reasons.append("no_valid_output")
            if no_extractable_answer:
                hard_fail_reasons.append("no_extractable_answer")
            if babbling:
                hard_fail_reasons.append("babbling_or_repetition")
            for reason in hard_fail_reasons:
                hard_fail_reason_counter[reason] += 1

            eval_item = {
                "rollout_index": j,
                "response": resp,
                "extracted_answer": extracted[j],
                "hard_fail": hard_fail,
                "hard_fail_reasons": hard_fail_reasons,
                "judge_prompt": "",
                "judge_output": "",
                "raw_score": 0.0 if hard_fail else None,
                "score_valid": True if hard_fail else False,
                "final_score": 0.0 if hard_fail else None,
            }

            if (not hard_fail) and should_judge_this_sample:
                prompt = build_eval_prompt(question, resp, tokenizer)
                prompts.append(prompt)
                prompt_refs.append((i, j))
                eval_item["judge_prompt"] = prompt
            elif (not hard_fail) and (not should_judge_this_sample):
                eval_item["score_valid"] = False
                eval_item["final_score"] = None

            row["rollout_evals"].append(eval_item)

        records.append(row)

    print(f"Total samples: {len(records)}")
    print(f"Low-consistency samples (<= {consistency_threshold}): {low_consistency_samples}")
    print(f"Need model judging rollouts: {len(prompts)}")
    if hard_fail_reason_counter:
        print("Hard-fail reason counts:")
        for k, v in hard_fail_reason_counter.items():
            print(f"  - {k}: {v}")

    if prompts:
        outputs = llm.generate(prompts, sampling_params)

        for out_idx, out in enumerate(outputs):
            i, j = prompt_refs[out_idx]
            text = out.outputs[0].text if out.outputs else ""
            score = parse_score_from_text(text)

            eval_item = records[i]["rollout_evals"][j]
            eval_item["judge_output"] = text
            eval_item["raw_score"] = score

            valid = is_reasonable_score(score, hard_fail=False)
            eval_item["score_valid"] = valid
            eval_item["final_score"] = float(score) if valid else None

    # 二次检查与剔除：过滤不合理分数后再做最终答案集合
    for row in records:
        evals = row["rollout_evals"]

        valid_scored = [e for e in evals if e["score_valid"] and e["final_score"] is not None]
        invalid_scored = [e for e in evals if not e["score_valid"]]
        fallback_by_consistency = False

        if valid_scored:
            best_score = max(e["final_score"] for e in valid_scored)
            best_items = [e for e in valid_scored if abs(e["final_score"] - best_score) < 1e-8]
        else:
            best_score = None
            best_items = []

        # 对于被一致性门控跳过的样本，不应丢失结果：回退到多数答案聚合
        if (not best_items) and row.get("skipped_by_consistency", False):
            fallback_by_consistency = True
            majority, _ = get_majority_and_consistency(row.get("extracted_answers", []))
            if majority not in [None, "", "[NO_ANSWER]"]:
                best_items = [
                    e for e in evals if strip_string(e.get("extracted_answer")) == strip_string(majority)
                ]
                # 跳过模型打分时，用一致性映射一个可解释的代理分数
                best_score = row.get("consistency", 0.0) * 10.0

        # 最高分答案集合（去重，保序）
        best_answer_set = []
        seen = set()
        for e in best_items:
            ans = e.get("extracted_answer")
            if ans in [None, "", "[NO_ANSWER]"]:
                continue
            key = strip_string(ans)
            if key and key not in seen:
                seen.add(key)
                best_answer_set.append(ans)

        row["best_score"] = best_score
        row["best_rollout_indices"] = [e["rollout_index"] for e in best_items]
        row["best_answer_set"] = best_answer_set
        row["filtered_invalid_score_count"] = len(invalid_scored)
        row["fallback_by_consistency"] = fallback_by_consistency

        # 兼容旧字段：sc_answer/sc_score
        row["sc_answer"] = best_answer_set[0] if best_answer_set else None
        row["sc_score"] = (best_score / 10.0) if best_score is not None else 0.0

    return records


def print_analysis(records):
    total_samples = len(records)
    total_rollouts = sum(len(r["rollout_evals"]) for r in records)
    hard_fails = sum(1 for r in records for e in r["rollout_evals"] if e["hard_fail"])
    valid_scores = sum(1 for r in records for e in r["rollout_evals"] if e["score_valid"])
    invalid_scores = total_rollouts - valid_scores
    skipped_samples = sum(1 for r in records if r.get("skipped_by_consistency", False))
    fallback_samples = sum(1 for r in records if r.get("fallback_by_consistency", False))

    # 正确率统计
    def norm(x):
        return strip_string(x) if x is not None else ""
    correct_count = 0
    for r in records:
        gt = norm(r.get("answer", ""))
        best_set = [norm(ans) for ans in r.get("best_answer_set", [])]
        if gt and any(ans == gt for ans in best_set):
            correct_count += 1

    print("\n--- Rollout Eval Analysis ---")
    print(f"Samples: {total_samples}")
    print(f"Rollouts: {total_rollouts}")
    print(f"Hard fails (forced 0): {hard_fails}")
    print(f"Valid scores: {valid_scores}")
    print(f"Invalid/filtered scores: {invalid_scores}")
    print(f"Skipped samples by consistency gate: {skipped_samples}/{total_samples}")
    print(f"Fallback aggregated by consistency: {fallback_samples}/{total_samples}")
    with_best = sum(1 for r in records if r["best_score"] is not None)
    print(f"Samples with at least one valid scored rollout: {with_best}/{total_samples}")
    print(f"Accuracy (best_answer_set hit gt): {correct_count}/{total_samples} = {correct_count/total_samples:.2%}")


# =====================================================
# Main
# =====================================================

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading {args.input_file}")
    data = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"Loaded {len(data)} samples")

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
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_eval_tokens,
        stop=["<|eot_id|>", "</s>", "<|im_end|>"],
    )

    records = run_experiment(
        data,
        llm,
        tokenizer,
        sampling_params,
        consistency_threshold=args.consistency_threshold,
        judge_all_samples=args.judge_all_samples,
    )
    print_analysis(records)

    if args.output_file:
        print(f"\nSaving results to {args.output_file}")
        # 统计正确率
        def norm(x):
            return strip_string(x) if x is not None else ""
        correct_count = 0
        for row in records:
            gt = norm(row.get("answer", ""))
            best_set = [norm(ans) for ans in row.get("best_answer_set", [])]
            if gt and any(ans == gt for ans in best_set):
                correct_count += 1
        accuracy = correct_count / len(records) if records else 0.0

        with open(args.output_file, "w", encoding="utf-8") as f:
            for row in records:
                out_record = {
                    "problem": row["question"],
                    "answer": row.get("answer", ""),
                    "responses": row["responses"],
                    "extracted_answers": row["extracted_answers"],
                    "consistency": row.get("consistency", 0.0),
                    "is_low_consistency": row.get("is_low_consistency", False),
                    "skipped_by_consistency": row.get("skipped_by_consistency", False),
                    "fallback_by_consistency": row.get("fallback_by_consistency", False),
                    "rollout_evals": row["rollout_evals"],
                    "best_score": row["best_score"],
                    "best_rollout_indices": row["best_rollout_indices"],
                    "best_answer_set": row["best_answer_set"],
                    "filtered_invalid_score_count": row["filtered_invalid_score_count"],
                    "sc_answer": row["sc_answer"],
                    "sc_score": row["sc_score"],
                }
                f.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            # 在文件末尾追加整体正确率统计
            f.write(json.dumps({"accuracy": accuracy, "correct_count": correct_count, "total": len(records)}) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rollout-by-rollout math evaluation")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="rollout_eval_results.jsonl")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--consistency_threshold", type=float, default=0.3)
    parser.add_argument("--judge_all_samples", action="store_true")
    parser.add_argument("--max_eval_tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
