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
Diversity-based TTRL Reward Manager

This manager mirrors the structure of ``SemanticTTRLRewardManager`` but removes
semantic embedding calls. It adjusts negative rewards using a count-based
diversity score computed within each prompt group.

For each prompt group (rollout):
    diversity_term_i = (unique_answers - 1) / (rollout - majority_num) * (1 / c_i)
    final_reward_i = -1 + 0.5 * diversity_term_i   (for negative samples)
where:
    - ``unique_answers`` is the number of distinct decoded answers in the group.
    - ``majority_num`` is the maximum frequency of any single answer.
    - ``rollout`` is the total number of samples in the group.
    - ``c_i`` is the frequency of answer i inside the group.
Positive samples keep their base reward from ``test_time_train_metrics`` or
``auto_verify``.
"""

from collections import Counter, defaultdict
from functools import partial
import numpy as np
import torch
from math import cos, pi
from verl import DataProto
from verl.utils.reward_score.ttrl.auto_verify import auto_verify
from verl.utils.reward_score.ttrl.ttt_metrics import (
    post_test_time_train_metrics, test_time_train_metrics)
from verl.utils.reward_score.ttrl.latex_clean import normalize_latex
from verl.utils.reward_score.ttrl.qwen.qwen_math_parser import extract_answer


class DiversityTTRLRewardManager:
    """TTRL Reward Manager with count-based diversity adjustment."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        reward_fn_key: str = "data_source",
        compute_score=None,
        n_votes_per_prompt: int = 1,
        n_samples_per_prompt: int = 1,
        mode: str = "eval",
        eval_n_samples: int = 1,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.n_votes_per_prompt = n_votes_per_prompt
        self.n_samples_per_prompt = n_samples_per_prompt
        self.mode = mode
        self.eval_n_samples = eval_n_samples
        self.debug_mode = num_examine > 0

        assert n_votes_per_prompt >= n_samples_per_prompt, (
            f"TTRL requirement: n_votes_per_prompt({n_votes_per_prompt}) >= "
            f"n_samples_per_prompt({n_samples_per_prompt})"
        )

        print(
            "DiversityTTRLRewardManager initialized with "
            f"n_votes_per_prompt {n_votes_per_prompt}, "
            f"n_samples_per_prompt {n_samples_per_prompt}, "
            f"eval_n_samples {eval_n_samples}"
        )

    # === Helpers ===
    def _data_source_to_task(self, data_source):
        ds = str(data_source)
        if ds in ["MATH-TTT", "AIME-TTT", "AMC-TTT", "AIME25"]:
            return "math"
        if ds in ["GPQA-TTT"]:
            return "gpqa"
        if ds in ["BBEH", "bbeh", "BigBench-Extra-Hard"]:
            return "bbeh"

        dsl = ds.lower()
        if any(key in dsl for key in ["gpqa"]):
            return "gpqa"
        if any(key in dsl for key in ["aime", "math", "amc", "aime25"]):
            return "math"
        if "bbeh" in dsl or "bigbench" in dsl:
            return "bbeh"

        raise NotImplementedError(
            f"Data source {data_source} is not supported for DiversityTTRLRewardManager"
        )

    def _extract_after_think(self, text: str) -> str:
        """Extract content after </think> tag for semantic similarity calculation"""
        think_end_tag = "</think>"
        think_end = text.find(think_end_tag)
        if think_end != -1:
            return text[think_end + len(think_end_tag):].strip()
        return text.strip()

    def _extract_final_answers(self, task: str, outputs: list[str]) -> list[str]:
        """
        Extract final answers from outputs for diversity calculation.
        Uses task-specific answer extraction logic.
        """
        extract_fn = partial(extract_answer, data_name=task)
        normalized_outputs = [normalize_latex(x) for x in outputs]
        final_answers = [extract_fn(text) or "<empty>" for text in normalized_outputs]
        return final_answers

    def _decode_data_item(self, data_item):
        prompt_idx = data_item.batch["prompts"]
        prompt_length = prompt_idx.shape[-1]
        valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
        valid_prompt_idx = prompt_idx[-valid_prompt_length:]

        response_idx = data_item.batch["responses"]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        valid_response_idx = response_idx[:valid_response_length]

        prompt_str = self.tokenizer.decode(valid_prompt_idx, skip_special_tokens=False)
        response_str = self.tokenizer.decode(valid_response_idx, skip_special_tokens=False)

        return prompt_str, response_str, valid_response_length

    def _compute_strategy_entropy(self, data_items):
        if not data_items:
            return 0.0
        
        log_probs_list = []
        response_lengths = []
        
        # 第一遍：过滤和提取（无异常捕获）
        for data_item in data_items:
            if not hasattr(data_item, "batch") or "old_log_probs" not in data_item.batch:
                continue
            
            batch = data_item.batch
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is None or len(attention_mask) <= batch["prompts"].shape[-1]:
                continue
            
            prompt_length = batch["prompts"].shape[-1]
            response_length_val = attention_mask[prompt_length:].sum().item()
            if response_length_val <= 0:
                continue
            
            old_log_probs = batch["old_log_probs"]
            if not isinstance(old_log_probs, torch.Tensor) or old_log_probs.numel() == 0:
                continue
            
            # 裁切逻辑（同上）
            log_probs_length = old_log_probs.shape[-1]
            if log_probs_length == response_length_val:
                response_log_probs = old_log_probs
            elif log_probs_length == prompt_length + response_length_val:
                response_log_probs = old_log_probs[prompt_length:prompt_length + response_length_val]
            elif log_probs_length > response_length_val:
                response_log_probs = old_log_probs[-response_length_val:]
            else:
                continue
            
            log_probs_list.append(response_log_probs.sum().item())
            response_lengths.append(response_length_val)
        
        if not log_probs_list:
            return 0.0
        
        # 向量计算（一次性）
        log_probs_array = np.array(log_probs_list)
        response_lengths_array = np.array(response_lengths)
        neg_log_likelihoods = -log_probs_array / response_lengths_array
        
        return float(np.mean(neg_log_likelihoods))
        
    def _apply_diversity_adjustment(
        self,
        pred_outputs: list[str],
        base_rewards: list[float],
        task: str,
    ) -> tuple[list[float], float]:
        """
        Apply count-based diversity adjustment to negative rewards within a prompt group.
        """
        n_answers = len(pred_outputs)
        final_answers = self._extract_final_answers(task, pred_outputs)
        freq = Counter(final_answers)

        unique_answers = len(freq)
        majority_num = max(freq.values()) if freq else 0
        diversity_ratio = unique_answers / n_answers if n_answers > 0 else 0.0
        denom = n_answers - majority_num

        final_rewards: list[float] = []
        for idx, base_reward in enumerate(base_rewards):
            if base_reward > 0:
                diversity_reward = (unique_answers)/n_answers
                diversity_reward = max(0.0, min(diversity_reward, 1.0))
                adjusted_reward = 0.5 + 0.5 * diversity_reward
                adjusted_reward = max(0.5, min(adjusted_reward, 1.0))
                final_rewards.append(float(adjusted_reward))
                # final_rewards.append(float(base_reward))
                continue

            ci = freq.get(final_answers[idx], 1)
            diversity_term = 0.0
            if unique_answers > 1 and denom > 0:
                # use factional scaling
                diversity_term = ((unique_answers - 1) / denom) * (1.0 / ci)
                # use non-linear scaling
                # diversity_term = ((unique_answers - 1) / denom) * cos(ci/ (self.n_votes_per_prompt / 2 ) * (pi / 2))
                # use linear scaling
                # diversity_term = ((unique_answers - 1) / denom) * (1- (ci) / (self.n_votes_per_prompt/2 ))
                diversity_term = max(0.0, min(diversity_term, 1.0))

            adjusted_reward = -1.0 + diversity_term
            adjusted_reward = max(-1.0, min(adjusted_reward, -0.5))
            final_rewards.append(float(adjusted_reward))

        # 返回最终奖励以及本组的多样性比率，调用方负责把该指标写入 ttrl_metrics
        return base_rewards, float(diversity_ratio)

    # === Metrics ===
    def compute_post_ttrl_metrics(self, data: DataProto):
        """
        Compute post-TTRL training evaluation metrics
        
        This method is used to evaluate the effectiveness of TTRL training by analyzing
        the quality and consistency of model-generated answers.
        
        Args:
            data (DataProto): Data containing prompts, responses and labels
            
        Returns:
            dict: Dictionary containing various evaluation metrics
        """
        assert len(data) % self.n_samples_per_prompt == 0, (
            f"Length of data {len(data)} should be divisible by n_samples_per_prompt {self.n_samples_per_prompt}"
        )
        prompt_num = len(data) // self.n_samples_per_prompt
        print(f"Computing post-TTRL metrics, {prompt_num} prompts in total...")

        post_ttrl_info = {}
        post_ttrl_metrics_list = defaultdict(list)

        for prompt_i in range(prompt_num):
            group_vote_rewards = []
            group_pred_outputs = []
            group_labels = []
            group_extra_info = []
            task = None

            for i in range(self.n_samples_per_prompt):
                data_item = data[prompt_i * self.n_samples_per_prompt + i]
                prompt_idx = data_item.batch["prompts"]
                prompt_length = prompt_idx.shape[-1]
                valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                valid_prompt_idx = prompt_idx[-valid_prompt_length:]
                response_idx = data_item.batch["responses"]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_idx = response_idx[:valid_response_length]
                prompt_str = self.tokenizer.decode(valid_prompt_idx, skip_special_tokens=False)
                response_str = self.tokenizer.decode(valid_response_idx, skip_special_tokens=False)
                ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
                data_source = data_item.non_tensor_batch[self.reward_fn_key]
                vote_reward = data_item.batch["acc"]
                extra_info = data_item.non_tensor_batch["extra_info"]

                if task is None:
                    task = self._data_source_to_task(data_source)
                else:
                    if task != self._data_source_to_task(data_source):
                        raise NotImplementedError(
                            f"Non consistent task {task} and {self._data_source_to_task(data_source)} "
                            "for DiversityTTRLRewardManager"
                        )

                group_labels.append(ground_truth)
                group_pred_outputs.append(response_str)
                group_vote_rewards.append(vote_reward)
                group_extra_info.append(extra_info)

            post_ttrl_metrics = post_test_time_train_metrics(
                group_pred_outputs, group_labels, group_vote_rewards, task=task, extra_info=group_extra_info
            )
            for k, v in post_ttrl_metrics.items():
                post_ttrl_metrics_list[k].append(v)

        # Calculate average metrics and output
        for k, v in post_ttrl_metrics_list.items():
            if isinstance(v, list):
                v = np.mean(v)
                print(f"[{k}]", v)
                post_ttrl_info[k] = v
        return post_ttrl_info

    # === Reward computation ===
    def _compute_ttrl_reward(self, data: DataProto):
        print("Starting TTRL reward calculation with diversity adjustment...")

        reward_extra_info = defaultdict(list)
        ttrl_info = {}

        # Check if data is already down-sampled or if it's at original size
        # Data should be divisible by n_votes_per_prompt for original size
        # If not, it may already be down-sampled, so return default values
        if len(data) % self.n_votes_per_prompt != 0:
            # Data has been down-sampled already; cannot compute diversity metrics
            # Return zero rewards and default answer types/consistency rates
            print(f"WARNING: Data size {len(data)} is not divisible by n_votes_per_prompt {self.n_votes_per_prompt}")
            print(f"Data may already be down-sampled. Returning default values.")
            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
            ttrl_info["_answer_types"] = np.arange(len(data), dtype=np.int64)  # Default IDs
            ttrl_info["_consistency_rate"] = np.ones(len(data), dtype=np.float32) * 0.5  # Default consistency
            return reward_tensor, reward_extra_info, ttrl_info

        prompt_num = len(data) // self.n_votes_per_prompt
        reward_tensor = torch.zeros_like(
            data.batch["responses"][: prompt_num * self.n_samples_per_prompt], dtype=torch.float32
        )

        already_print_data_sources = {}
        all_ttrl_metrics = defaultdict(list)
        scores = [0.0 for _ in range(len(data))]
        
        # === NEW: Collect answer_types, consistency_rate, and accuracy_rate for diversity density advantage ===
        # answer_types: per-sample answer type id (hash of extracted answer)
        # consistency_rate: per-sample self-consistency rate (majority_ratio from this prompt group)
        # accuracy_rate: per-sample accuracy rate (correct_count / total from this prompt group)
        all_answer_types = []
        all_oracle_answer_types = []  # Oracle (true-label based) answer types for advantage bias diagnostics
        all_consistency_rates = []
        all_accuracy_rates = []
        all_label_accuracies = []  # Binary indicator of majority vote correctness

        for prompt_i in range(prompt_num):
            group_pred_outputs = []
            group_labels = []
            group_extra_info = []
            group_resp_lengths = []
            group_prompts = []
            group_indices = []
            task = None

            for i in range(self.n_votes_per_prompt):
                data_item = data[prompt_i * self.n_votes_per_prompt + i]
                prompt_str, response_str, valid_response_length = self._decode_data_item(data_item)
                ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
                data_source = data_item.non_tensor_batch[self.reward_fn_key]
                extra_info = data_item.non_tensor_batch["extra_info"]

                if task is None:
                    task = self._data_source_to_task(data_source)
                else:
                    if task != self._data_source_to_task(data_source):
                        raise NotImplementedError(
                            f"Non consistent task {task} and {self._data_source_to_task(data_source)} "
                            "for DiversityTTRLRewardManager"
                        )

                group_labels.append(ground_truth)
                group_pred_outputs.append(response_str)
                group_extra_info.append(extra_info)
                group_resp_lengths.append(int(valid_response_length))
                group_prompts.append(prompt_str)

            base_rewards, ttrl_metrics = test_time_train_metrics(
                group_pred_outputs, group_labels, task=task, extra_info=group_extra_info
            )

            current_group_data = data[prompt_i * self.n_votes_per_prompt : (prompt_i + 1) * self.n_votes_per_prompt]
            strategy_entropy = self._compute_strategy_entropy(current_group_data)
            ttrl_metrics["neg_log_likelihood"] = strategy_entropy
            if self.debug_mode and strategy_entropy > 0:
                print(f"    Strategy entropy: H_ttrl={strategy_entropy:.3f} (normalized negative log-likelihood)")

            final_rewards, diversity_ratio = self._apply_diversity_adjustment(group_pred_outputs, base_rewards, task)
            # expose diversity metric for this group
            ttrl_metrics["diversity_ratio"] = diversity_ratio
            
            # === Extract answer types, consistency rate, and diagnostic metrics ===
            final_answers = self._extract_final_answers(task, group_pred_outputs)
            freq = Counter(final_answers)
            majority_num = max(freq.values()) if freq else 0
            consistency_rate = majority_num / self.n_votes_per_prompt if self.n_votes_per_prompt > 0 else 0.0
            accuracy_rate = ttrl_metrics.get("ground_truth_ratio", 0.0)
            label_accuracy = ttrl_metrics.get("label_accuracy", 0.0)
            
            # === Compute true_rewards for diagnostic metrics ===
            # base_rewards: matches majority (pseudo-label), true_rewards: matches ground truth
            ground_truth = group_labels[0]
            true_rewards, _ = auto_verify(
                task, group_pred_outputs, [ground_truth] * len(group_pred_outputs),
                extra_info=group_extra_info
            )
            
            # false_positive_rate: fraction of pseudo-positive samples that are actually wrong
            n_pseudo_pos = sum(1 for b in base_rewards if b > 0)
            n_false_pos = sum(1 for b, t in zip(base_rewards, true_rewards) if b > 0 and t == 0)
            fp_rate = n_false_pos / n_pseudo_pos if n_pseudo_pos > 0 else 0.0
            
            # false_negative_rate: fraction of pseudo-negative samples that are actually correct
            n_pseudo_neg = sum(1 for b in base_rewards if b == 0)
            n_false_neg = sum(1 for b, t in zip(base_rewards, true_rewards) if b == 0 and t > 0)
            fn_rate = n_false_neg / n_pseudo_neg if n_pseudo_neg > 0 else 0.0
            
            ttrl_metrics["false_positive_rate"] = fp_rate
            ttrl_metrics["false_negative_rate"] = fn_rate
            
            # Create answer type mapping (hash of answer string -> integer id)
            answer_to_id = {ans: hash(ans) for ans in set(final_answers)}
            
            for i in range(self.n_votes_per_prompt):
                # --- TTA answer_types (based on pseudo-label / majority voting) ---
                is_correct = base_rewards[i] > 0
                if is_correct:
                    ans_type = 0
                else:
                    ans_type = answer_to_id[final_answers[i]]
                    if ans_type == 0:
                        ans_type = 1
                all_answer_types.append(ans_type)
                
                # --- Oracle answer_types (based on ground truth) ---
                if true_rewards[i] > 0:
                    oracle_type = 0
                else:
                    oracle_type = answer_to_id[final_answers[i]]
                    if oracle_type == 0:
                        oracle_type = 1
                all_oracle_answer_types.append(oracle_type)
                
                all_consistency_rates.append(consistency_rate)
                all_accuracy_rates.append(accuracy_rate)
                all_label_accuracies.append(label_accuracy)

            for k, v in ttrl_metrics.items():
                all_ttrl_metrics[k].append(v)

            for i in range(self.n_votes_per_prompt):
                current_reward = final_rewards[i]
                vlen = group_resp_lengths[i]

                if i < self.n_samples_per_prompt and vlen > 0:
                    reward_tensor[prompt_i * self.n_samples_per_prompt + i, vlen - 1] = current_reward

                scores[prompt_i * self.n_votes_per_prompt + i] = current_reward

                data_item = data[prompt_i * self.n_votes_per_prompt + i]
                data_source = data_item.non_tensor_batch[self.reward_fn_key]
                if data_source not in already_print_data_sources:
                    already_print_data_sources[data_source] = 0
                if already_print_data_sources[data_source] < self.num_examine:
                    already_print_data_sources[data_source] += 1
                    print("\n    === Sample Debug Output ===")
                    print("    [prompt]", group_prompts[i])
                    print("    [response]", group_pred_outputs[i])
                    print(f"    [final_score] {current_reward:.4f}")
                    print(f"    [base_reward] {base_rewards[i]:.4f}")

        data.batch["acc"] = torch.tensor(scores, dtype=torch.float32, device=data.batch["prompts"].device)
        
        # Store per-sample arrays for downstream advantage computation (only training samples)
        training_answer_types = []
        training_oracle_answer_types = []
        training_consistency_rates = []
        training_accuracy_rates = []
        training_label_accuracies = []
        for prompt_i in range(prompt_num):
            for i in range(self.n_samples_per_prompt):
                global_idx = prompt_i * self.n_votes_per_prompt + i
                training_answer_types.append(all_answer_types[global_idx])
                training_oracle_answer_types.append(all_oracle_answer_types[global_idx])
                training_consistency_rates.append(all_consistency_rates[global_idx])
                training_accuracy_rates.append(all_accuracy_rates[global_idx])
                training_label_accuracies.append(all_label_accuracies[global_idx])
        
        ttrl_info["_answer_types"] = np.array(training_answer_types)
        ttrl_info["_oracle_answer_types"] = np.array(training_oracle_answer_types)
        ttrl_info["_consistency_rate"] = np.array(training_consistency_rates)
        ttrl_info["_accuracy_rate"] = np.array(training_accuracy_rates)
        ttrl_info["_label_accuracy"] = np.array(training_label_accuracies)

        print("\n=== TTRL Training Metrics Summary ===")
        for k, v in all_ttrl_metrics.items():
            if isinstance(v, list):
                avg_v = np.mean(v)
                print(f"[{k}] {avg_v:.4f}")
                ttrl_info[k] = avg_v

        return reward_tensor, reward_extra_info, ttrl_info

    def _compute_eval_reward(self, data: DataProto):
        print("Starting evaluation reward calculation with diversity adjustment...")

        reward_extra_info = defaultdict(list)
        ttrl_info = {}
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        already_print_data_sources = {}

        assert len(data) % self.eval_n_samples == 0, (
            f"Evaluation data length ({len(data)}) must be divisible by eval_n_samples ({self.eval_n_samples})"
        )

        prompt_num = len(data) // self.eval_n_samples
        print(f"  Processing {prompt_num} prompts, each with {self.eval_n_samples} samples")

        group_pred_outputs = []
        group_labels = []
        group_extra_info = []
        sample_valid_resp_len: dict[int, int] = {}
        task_groups = {}

        for i in range(len(data)):
            data_item = data[i]
            prompt_str, response_str, valid_response_length = self._decode_data_item(data_item)
            sample_valid_resp_len[i] = int(valid_response_length)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch["extra_info"]

            group_labels.append(ground_truth)
            group_pred_outputs.append(response_str)
            group_extra_info.append(extra_info)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)

            task_key = self._data_source_to_task(data_source)
            if task_key not in task_groups:
                task_groups[task_key] = {"indices": [], "outputs": [], "labels": [], "extra": []}
            task_groups[task_key]["indices"].append(i)
            task_groups[task_key]["outputs"].append(response_str)
            task_groups[task_key]["labels"].append(ground_truth)
            task_groups[task_key]["extra"].append(extra_info)

        base_rewards = [0.0] * len(data)
        for task_key, group in task_groups.items():
            rewards, verify_extra_info = auto_verify(task_key, group["outputs"], group["labels"], extra_info=group["extra"])
            for k, v in verify_extra_info.items():
                if isinstance(v, list):
                    reward_extra_info[k] += v
            for idx_in_group, sample_idx in enumerate(group["indices"]):
                base_rewards[sample_idx] = float(rewards[idx_in_group])

        final_rewards = [0.0] * len(base_rewards)
        # store per-prompt diversity ratios computed from outputs
        prompt_diversity_ratios = [0.0] * prompt_num
        for prompt_i in range(prompt_num):
            start_idx = prompt_i * self.eval_n_samples
            end_idx = start_idx + self.eval_n_samples
            prompt_outputs = group_pred_outputs[start_idx:end_idx]
            prompt_base_rewards = base_rewards[start_idx:end_idx]

            first_idx = prompt_i * self.eval_n_samples
            first_ds = data[first_idx].non_tensor_batch[self.reward_fn_key]
            prompt_task = self._data_source_to_task(first_ds)
            prompt_final_rewards, diversity_ratio = self._apply_diversity_adjustment(
                prompt_outputs, prompt_base_rewards, prompt_task
            )
            prompt_diversity_ratios[prompt_i] = diversity_ratio
            for j, sample_idx in enumerate(range(start_idx, end_idx)):
                final_rewards[sample_idx] = prompt_final_rewards[j]

        for i in range(len(data)):
            vlen = sample_valid_resp_len.get(i, 0)
            if vlen > 0:
                reward_tensor[i, vlen - 1] = final_rewards[i]

        print("\n=== Calculating TTRL Evaluation Metrics ===")
        all_ttrl_metrics = defaultdict(list)

        for prompt_i in range(prompt_num):
            group_pred_outputs_ttrl = []
            group_labels_ttrl = []
            group_extra_info_ttrl = []

            for i in range(self.eval_n_samples):
                idx = prompt_i * self.eval_n_samples + i
                data_item = data[idx]
                _, response_str, _ = self._decode_data_item(data_item)
                ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
                extra_info = data_item.non_tensor_batch["extra_info"]

                group_pred_outputs_ttrl.append(response_str)
                group_labels_ttrl.append(ground_truth)
                group_extra_info_ttrl.append(extra_info)

            first_ds = data[prompt_i * self.eval_n_samples].non_tensor_batch[self.reward_fn_key]
            group_task = self._data_source_to_task(first_ds)

            _, ttrl_metrics = test_time_train_metrics(
                group_pred_outputs_ttrl, group_labels_ttrl, task=group_task, extra_info=group_extra_info_ttrl
            )

            current_group_data = data[prompt_i * self.eval_n_samples : (prompt_i + 1) * self.eval_n_samples]
            strategy_entropy = self._compute_strategy_entropy(current_group_data)
            ttrl_metrics["neg_log_likelihood"] = strategy_entropy
            # attach diversity metric computed earlier for this prompt
            try:
                ttrl_metrics["diversity_ratio"] = prompt_diversity_ratios[prompt_i]
            except Exception:
                ttrl_metrics["diversity_ratio"] = 0.0

            for k, v in ttrl_metrics.items():
                all_ttrl_metrics[k].append(v)

        for k, v in all_ttrl_metrics.items():
            if isinstance(v, list):
                avg_v = np.mean(v)
                print(f"[{k}] {avg_v:.4f}")
                ttrl_info[k] = avg_v

        return reward_tensor, reward_extra_info, ttrl_info

    def __call__(self, data: DataProto, return_dict: bool = False):
        print("\n" + "=" * 50)
        print("DiversityTTRLRewardManager execution started")
        print(f"Mode: {self.mode}")
        print(f"Data size: {len(data)}")
        print("=" * 50)

        if self.mode == "train":
            reward_tensor, reward_extra_info, ttrl_info = self._compute_ttrl_reward(data)
        elif self.mode == "eval":
            reward_tensor, reward_extra_info, ttrl_info = self._compute_eval_reward(data)
        else:
            raise NotImplementedError(f"Mode {self.mode} is not supported for DiversityTTRLRewardManager")

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
                "ttrl_info": ttrl_info,
            }
        return reward_tensor
