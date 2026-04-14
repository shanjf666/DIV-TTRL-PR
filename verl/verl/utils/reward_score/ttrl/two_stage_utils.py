# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Two-Stage Self-Verification Pipeline Utilities.

This module implements the core logic for the second-stage verification pass,
where the training model acts as its own verifier by performing reverse-verification
on candidate answers extracted from Pass 1 rollouts.

Supports two verification modes:
  - Greedy: temperature=0, n=1 per candidate. Select True + highest frequency.
  - Sampling: temperature=0.6, n=N per candidate. Select by majority True votes.

Supports two fallback strategies when all candidates verify as False:
  - "majority": fallback to the highest-frequency candidate from Pass 1.
  - "penalize": return None, signaling the trainer to apply a -1 advantage penalty.
"""

import re
import logging
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto

logger = logging.getLogger(__name__)

# ===========================================================================
# Verification Prompt Templates
# ===========================================================================

VERIFICATION_SYSTEM_PROMPT = """You are a rigorous mathematical reviewer."""

VERIFICATION_USER_TEMPLATE = """Problem:
{problem}

[Hypothesis to Test]
A previous attempt at this problem resulted in the following answer:
{candidate_answer}

[Task]
Act as a rigorous mathematical reviewer. 
Treat the previous answer ({candidate_answer}) as a given hypothesis. Plug this answer BACK into the original problem conditions. Perform a rigorous backward-substitution to check if it satisfies all constraints or if it leads to a mathematical contradiction. 

You MUST conclude your verification by explicitly stating either "Verification Result: True" (if the hypothesis perfectly satisfies all conditions) or "Verification Result: False" (if it leads to any contradiction).

You MUST strictly use the following XML format for your response:
<reverse_verification>
(Your step-by-step backward substitution checking if {candidate_answer} contradicts the problem conditions)
Verification Result: [True/False]
"""


# ===========================================================================
# Answer Extraction from Pass 1
# ===========================================================================

def extract_candidate_answers(
    pass1_data: DataProto,
    tokenizer,
    n_votes_per_prompt: int,
    task: str = "math",
    max_candidates: int = 10,
) -> List[Dict]:
    """Extract unique candidate answers from Pass 1 rollout data, grouped by prompt.

    For each prompt group (n_votes_per_prompt samples), extracts and deduplicates
    answers, returning them sorted by frequency (descending).

    Args:
        pass1_data: DataProto from Pass 1 generation (already union'd with batch).
        tokenizer: HuggingFace tokenizer for decoding.
        n_votes_per_prompt: Number of rollout samples per prompt.
        task: Task name for answer extraction (e.g., "math", "gpqa").
        max_candidates: Maximum number of candidate answers to keep per prompt.

    Returns:
        List of dicts, one per prompt group. Each dict contains:
            - "problem_text": str, the decoded problem prompt
            - "candidates": List[Tuple[str, int]], (answer, frequency) sorted desc
            - "all_answers": List[str], all extracted answers (including duplicates)
            - "prompt_group_idx": int, the index of this prompt group
    """
    from verl.utils.reward_score.ttrl.auto_extract import auto_extract

    assert len(pass1_data) % n_votes_per_prompt == 0, (
        f"Data length {len(pass1_data)} must be divisible by n_votes_per_prompt {n_votes_per_prompt}"
    )
    num_prompts = len(pass1_data) // n_votes_per_prompt

    prompt_groups = []

    for prompt_i in range(num_prompts):
        group_responses = []
        group_extra_info = []
        problem_text = None

        for j in range(n_votes_per_prompt):
            idx = prompt_i * n_votes_per_prompt + j
            data_item = pass1_data[idx]

            # Decode prompt (only need once per group)
            if problem_text is None:
                prompt_ids = data_item.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]
                valid_prompt_length = int(data_item.batch["attention_mask"][:prompt_length].sum().item())
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]
                problem_text = tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)

            # Decode response
            response_ids = data_item.batch["responses"]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_response_length = int(data_item.batch["attention_mask"][prompt_length:].sum().item())
            valid_response_ids = response_ids[:valid_response_length]
            response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=False)

            group_responses.append(response_str)

            extra_info = data_item.non_tensor_batch.get("extra_info", None)
            group_extra_info.append(extra_info)

        # Extract answers using the standard extractor
        model_answers = auto_extract(task, group_responses, extra_info=group_extra_info)

        # Count frequencies
        counter = Counter(model_answers)
        # Sort by frequency descending, then limit to max_candidates if specified
        if max_candidates is not None and max_candidates > 0:
            candidates = counter.most_common(max_candidates)
        else:
            candidates = counter.most_common()

        prompt_groups.append({
            "problem_text": problem_text,
            "candidates": candidates,  # List[(answer, count)]
            "all_answers": model_answers,
            "prompt_group_idx": prompt_i,
            "majority_rate": candidates[0][1] / n_votes_per_prompt if candidates else 0.0,
            "majority_answer": candidates[0][0] if candidates else None,
        })

    return prompt_groups


# ===========================================================================
# Construct Verification DataProto
# ===========================================================================

def construct_verification_dataproto(
    prompt_groups: List[Dict],
    tokenizer,
    verification_mode: str = "greedy",
    verification_n: int = 1,
    max_prompt_length: int = 2048,
    max_answer_length: int = 200,
    system_prompt: str = None,
    user_template: str = None,
) -> Tuple[Optional[DataProto], List[Dict]]:
    """Construct a DataProto for the verification pass from extracted candidate answers.

    For each prompt group and each candidate answer, creates a verification prompt
    using the template, tokenizes it, and packs it into a DataProto.

    Args:
        prompt_groups: Output of extract_candidate_answers().
        tokenizer: HuggingFace tokenizer.
        verification_mode: "greedy" or "sampling".
        verification_n: Number of verification samples per candidate (for sampling mode).
        max_prompt_length: Maximum token length for the verification prompt.
        max_answer_length: Maximum character length for candidate answer strings.
        system_prompt: Custom system prompt (defaults to VERIFICATION_SYSTEM_PROMPT).
        user_template: Custom user template (defaults to VERIFICATION_USER_TEMPLATE).

    Returns:
        Tuple of:
            - DataProto for verification generation (or None if no candidates)
            - List[Dict] mapping each row in the DataProto back to:
                {"prompt_group_idx": int, "candidate_answer": str, "frequency": int}
    """
    if system_prompt is None:
        system_prompt = VERIFICATION_SYSTEM_PROMPT
    if user_template is None:
        user_template = VERIFICATION_USER_TEMPLATE

    all_token_ids = []
    all_attention_masks = []
    verification_mapping = []

    for group in prompt_groups:
        problem_text = group["problem_text"]
        prompt_group_idx = group["prompt_group_idx"]

        for candidate_answer, frequency in group["candidates"]:
            if candidate_answer is None or str(candidate_answer).strip() == "":
                continue

            # Truncate long candidate answers
            answer_str = str(candidate_answer)[:max_answer_length]

            # Build the verification prompt text
            user_content = user_template.format(
                problem=problem_text,
                candidate_answer=answer_str,
            )

            # Apply chat template if available
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]

            try:
                prompt_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                prompt_text += "<reverse_verification>\n"
            except Exception:
                # Fallback: manual ChatML formatting
                prompt_text = (
                    f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                    f"<|im_start|>user\n{user_content}<|im_end|>\n"
                    f"<|im_start|>assistant\n<reverse_verification>\n"
                )

            # Tokenize
            encoded = tokenizer(
                prompt_text,
                truncation=True,
                max_length=max_prompt_length,
                padding=False,
                return_tensors=None,
            )

            token_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

            # In sampling mode, repeat for verification_n samples.
            # If verification_n is None or < 0, dynamically use candidate frequency.
            if verification_mode == "sampling":
                if verification_n is None or verification_n < 0:
                    repeat_count = frequency
                else:
                    repeat_count = verification_n
            else:
                repeat_count = 1
                
            for _ in range(repeat_count):
                all_token_ids.append(token_ids)
                all_attention_masks.append(attention_mask)
                verification_mapping.append({
                    "prompt_group_idx": prompt_group_idx,
                    "candidate_answer": candidate_answer,
                    "frequency": frequency,
                })

    if not all_token_ids:
        logger.warning("No valid candidates found for verification. Skipping Pass 2.")
        return None, []

    # Pad all sequences to the same length (left-padded, matching verl convention)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    max_len = max(len(ids) for ids in all_token_ids)

    padded_input_ids = []
    padded_attention_masks = []
    padded_position_ids = []

    for ids, mask in zip(all_token_ids, all_attention_masks):
        pad_len = max_len - len(ids)
        # Left padding
        padded_ids = [pad_token_id] * pad_len + ids
        padded_mask = [0] * pad_len + mask
        # Position IDs: 0 for padding, then 0, 1, 2, ...
        pos_ids = [0] * pad_len + list(range(len(ids)))

        padded_input_ids.append(padded_ids)
        padded_attention_masks.append(padded_mask)
        padded_position_ids.append(pos_ids)

    # Convert to tensors
    input_ids_tensor = torch.tensor(padded_input_ids, dtype=torch.long)
    attention_mask_tensor = torch.tensor(padded_attention_masks, dtype=torch.long)
    position_ids_tensor = torch.tensor(padded_position_ids, dtype=torch.long)

    batch_size = input_ids_tensor.shape[0]

    # Build DataProto
    batch = TensorDict(
        {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
            "position_ids": position_ids_tensor,
        },
        batch_size=batch_size,
    )

    # Store raw_prompt_ids for vLLM compatibility
    non_tensor_batch = {
        "raw_prompt_ids": np.array([ids for ids in all_token_ids], dtype=object),
    }

    # Meta info: configure for verification generation
    meta_info = {
        "verification_mode": verification_mode,
        "do_sample": verification_mode != "greedy",
        "validate": False,
    }

    verification_batch = DataProto(
        batch=batch,
        non_tensor_batch=non_tensor_batch,
        meta_info=meta_info,
    )

    logger.info(
        f"Constructed verification DataProto: {batch_size} samples "
        f"(from {len(prompt_groups)} prompt groups), max_len={max_len}"
    )

    return verification_batch, verification_mapping


# ===========================================================================
# Parse Verification Results
# ===========================================================================

def parse_verification_result(text: str) -> Optional[bool]:
    """Extract 'Verification Result: True/False' from the verifier's output.

    Uses robust keyword matching syntax since we compel standard formatting.
    """
    if not text:
        return None

    lower_text = text.lower()
    if "verification result: true" in lower_text or "verification result:true" in lower_text:
        return True
    elif "verification result: false" in lower_text or "verification result:false" in lower_text:
        return False
        
    return None

def resolve_filtered_pseudo_labels(
    verification_outputs: List[str],
    verification_mapping: List[Dict],
    prompt_groups: List[Dict],
) -> Tuple[List[Optional[str]], List[float]]:
    """Match verification results back to prompt groups and resolve the best pseudo-label.
    
    Filters candidates based on majority vote (True_votes > False_votes).
    If no candidate passes, falls back to the original majority answer from Pass 1.

    Args:
        verification_outputs: List of decoded verification response strings.
        verification_mapping: The mapping list from construct_verification_dataproto().
        prompt_groups: Output of extract_candidate_answers(), contains fallback info.

    Returns:
        Tuple of:
            - List of length len(prompt_groups), each containing the selected pseudo-label answer.
            - List of length len(prompt_groups), each containing the consistency score of the selected pseudo-label.
    """
    assert len(verification_outputs) == len(verification_mapping), (
        f"Mismatch: {len(verification_outputs)} outputs vs {len(verification_mapping)} mappings"
    )

    num_prompt_groups = len(prompt_groups)
    
    # Group results by prompt_group_idx
    group_results: Dict[int, List[Dict]] = {}

    for output_text, mapping in zip(verification_outputs, verification_mapping):
        group_idx = mapping["prompt_group_idx"]
        if group_idx not in group_results:
            group_results[group_idx] = []
        is_true = parse_verification_result(output_text)

        group_results[group_idx].append({
            "candidate_answer": mapping["candidate_answer"],
            "frequency": mapping["frequency"],
            "is_true": is_true,
        })

    pseudo_labels = [None] * num_prompt_groups
    consistencies = [0.0] * num_prompt_groups

    for i, group in enumerate(prompt_groups):
        group_idx = group["prompt_group_idx"]
        results = group_results.get(group_idx, [])
        original_majority_ans = group.get("majority_answer")
        
        if not results:
            pseudo_labels[i] = original_majority_ans
            continue
            
        candidate_scores: Dict[str, Dict] = {}
        for r in results:
            ans = r["candidate_answer"]
            if ans not in candidate_scores:
                candidate_scores[ans] = {
                    "frequency": r["frequency"],
                    "true_count": 0,
                    "false_count": 0
                }
            if r["is_true"] is True:
                candidate_scores[ans]["true_count"] += 1
            elif r["is_true"] is False:
                candidate_scores[ans]["false_count"] += 1
                
        # All candidates sorted by true_count (desc), then frequency (desc)
        all_candidates = [(ans, info["true_count"], info["frequency"]) for ans, info in candidate_scores.items()]
        all_candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
                
        if all_candidates and all_candidates[0][1] > 0:
            # Pick the one with the highest true_count
            pseudo_labels[i] = all_candidates[0][0]
            # Consistency is frequency / total votes
            # wait, n_votes_per_prompt is not passed directly, but majority_rate is frequency / n_votes.
            # freq = all_candidates[0][2]
            # Since majority_rate = cand[0][1] / N, we can calculate N = cand[0][1] / majority_rate
            # But the simplest is to find the overall candidate freq within the group
            freq = all_candidates[0][2]
            top_freq = group["candidates"][0][1] if group["candidates"] else 1
            majority_rate = group.get("majority_rate", 0.0)
            n_votes = top_freq / majority_rate if majority_rate > 0 else 1
            consistencies[i] = freq / n_votes
        else:
            # Fallback: direct majority from Pass 1 if all true_counts are 0
            pseudo_labels[i] = original_majority_ans
            consistencies[i] = group.get("majority_rate", 0.0)

    return pseudo_labels, consistencies

def compute_proxy_cm_reward(
    verification_outputs: List[str],
    verification_mapping: List[Dict],
    final_pseudo_labels: Dict[int, str],
    consistency_scores: Dict[int, float],
) -> Tuple[List[float], Dict[str, float]]:
    """Compute surrogate CM rewards for each verification sample based on standard rules.
    
    Args:
        verification_outputs: List of decoded verification response strings.
        verification_mapping: Output from construct_verification_dataproto().
        final_pseudo_labels: Dict mapping prompt_group_idx to the chosen pseudo label string.
        consistency_scores: Dict mapping prompt_group_idx to the consistency float.
        
    Returns:
        rewards: List of float rewards
        metrics: Dict with tp/tn/fp/fn and format error rates.
    """
    rewards = []
    
    tp_count = 0
    tn_count = 0
    fp_count = 0
    fn_count = 0
    format_error_count = 0
    total = len(verification_outputs)
    
    for output_text, mapping in zip(verification_outputs, verification_mapping):
        group_idx = mapping["prompt_group_idx"]
        candidate = mapping["candidate_answer"]
        
        pl = final_pseudo_labels.get(group_idx)
        consistency = consistency_scores.get(group_idx, 1.0)
        
        # Check format
        has_tag = False
        has_result = False
        if output_text:
            text_lower = output_text.lower()
            has_tag = "<reverse_verification>" in text_lower or "reverse_verification" in text_lower
            has_result = "verification result" in text_lower
            
        if not (has_tag and has_result):
            rewards.append(-1.0)
            format_error_count += 1
            continue
            
        parsed_result = parse_verification_result(output_text)
        is_pl = (candidate == pl)
        
        if parsed_result is None:
            # Format exists but result can't be parsed properly (e.g. "Verification Result: maybe")
            rewards.append(-1.0)
            format_error_count += 1
        elif is_pl and parsed_result is True:   # TP
            rewards.append(1.0 * consistency)
            tp_count += 1
        elif not is_pl and parsed_result is False: # TN
            rewards.append(1.0 * consistency)
            tn_count += 1
        elif not is_pl and parsed_result is True:  # FP
            rewards.append(-1.0 * consistency)
            fp_count += 1
        elif is_pl and parsed_result is False:   # FN
            rewards.append(-0.5 * consistency)
            fn_count += 1
            
    metrics = {
        "tp_rate": tp_count / total if total > 0 else 0.0,
        "tn_rate": tn_count / total if total > 0 else 0.0,
        "fp_rate": fp_count / total if total > 0 else 0.0,
        "fn_rate": fn_count / total if total > 0 else 0.0,
        "format_error_rate": format_error_count / total if total > 0 else 0.0,
        "reward_mean": sum(rewards) / total if total > 0 else 0.0,
    }
    
    return rewards, metrics


# ===========================================================================
# Decode Verification Outputs
# ===========================================================================

def decode_verification_outputs(
    verification_gen_output: DataProto,
    tokenizer,
) -> List[str]:
    """Decode the generated verification responses back to strings.

    Args:
        verification_gen_output: DataProto output from generate_sequences().
        tokenizer: HuggingFace tokenizer.

    Returns:
        List of decoded response strings.
    """
    decoded_texts = []
    batch_size = len(verification_gen_output)

    for i in range(batch_size):
        data_item = verification_gen_output[i]
        response_ids = data_item.batch["responses"]
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        attention_mask = data_item.batch["attention_mask"]

        # Calculate valid response length
        valid_response_length = int(attention_mask[prompt_length:].sum().item())
        valid_response_ids = response_ids[:valid_response_length]

        text = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        decoded_texts.append(text)

    return decoded_texts
