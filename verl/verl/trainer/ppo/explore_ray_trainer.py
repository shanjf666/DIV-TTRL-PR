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
Two-stage Exploration PPO Trainer with Ray-based single controller.
Extends RayPPOTrainer with a two-stage rollout: Round 1 generates, evaluates
consistency, and Round 2 re-generates low-consistency prompts with self-verify
prompts. All original trainer metrics and advantage estimators are preserved.
"""

import gc
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from codetiming import Timer
from omegaconf import OmegaConf
from tensordict import TensorDict
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    _timer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)


class RayExplorePPOTrainer(RayPPOTrainer):
    """
    Two-stage Exploration PPO Trainer.

    Extends RayPPOTrainer by adding an optional two-stage rollout:
      - Round 1: Standard rollout + consistency evaluation
      - Round 2: Re-generate low-consistency prompts with a self-verify prompt
      - Merge both rounds and proceed with the standard PPO pipeline

    All original trainer capabilities (advantage estimators, metrics, etc.)
    are preserved as a strict superset.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Explore configuration (read from config.algorithm)
        self.use_explore_rollout = getattr(self.config.algorithm, "use_explore_rollout", False)
        self.explore_threshold = getattr(self.config.algorithm, "explore_threshold", 0.5)

        # Default system and user templates for exploration.
        default_system = "You are an expert mathematical checker and solver. Carefully inspect suspicious solutions and re-solve problems correctly step by step."
        default_template = (
            "[Problem]\n"
            "{problem}\n\n"
            "[Previous Answer]\n"
            "A previous attempt resulted in the following answer:\n"
            "{majority_answer}\n\n"
            "[Task]\n"
            "Please solve the problem step by step.\n"
            "IMPORTANT: The previous answer is likely wrong due to low consistency. Please explore a completely different reasoning path and provide an alternative answer.\n"
            "Conclude your final answer strictly enclosed in \\boxed{{}}."
        )
        self.explore_system_prompt = getattr(
            self.config.algorithm, "explore_system_prompt", default_system
        )
        self.explore_prompt_template = getattr(
            self.config.algorithm, "explore_prompt_template", default_template
        )

        if self.use_explore_rollout:
            print(
                f"[RayExplorePPOTrainer] Exploration mode enabled: "
                f"threshold={self.explore_threshold}"
            )

    # ------------------------------------------------------------------ #
    #  Helper: Detect chat template format from tokenizer or prompt
    # ------------------------------------------------------------------ #
    def _detect_chat_format(self, prompt_text: str) -> dict:
        """
        Detect the chat template format used in the prompt.
        Returns a dict with markers for system, user, assistant, and end tokens.
        
        Supports: ChatML (Qwen), Llama-3, Gemma, and fallback to plain text.
        """
        # ChatML format (Qwen, etc.)
        if "<|im_start|>" in prompt_text:
            return {
                "format": "chatml",
                "system_start": "<|im_start|>system\n",
                "user_start": "<|im_start|>user\n", 
                "assistant_start": "<|im_start|>assistant\n",
                "end": "<|im_end|>\n",
            }
        # Llama-3 format
        elif "<|start_header_id|>" in prompt_text:
            return {
                "format": "llama3",
                "system_start": "<|start_header_id|>system<|end_header_id|>\n\n",
                "user_start": "<|start_header_id|>user<|end_header_id|>\n\n",
                "assistant_start": "<|start_header_id|>assistant<|end_header_id|>\n\n",
                "end": "<|eot_id|>",
            }
        # Gemma format
        elif "<start_of_turn>" in prompt_text:
            return {
                "format": "gemma",
                "system_start": "",  # Gemma doesn't have system role
                "user_start": "<start_of_turn>user\n",
                "assistant_start": "<start_of_turn>model\n",
                "end": "<end_of_turn>\n",
            }
        # Plain text fallback
        else:
            return {
                "format": "plain",
                "system_start": "System: ",
                "user_start": "\n\nUser: ",
                "assistant_start": "\n\nAssistant: ",
                "end": "\n",
            }

    # ------------------------------------------------------------------ #
    #  Helper: Smart truncation that avoids breaking LaTeX/code
    # ------------------------------------------------------------------ #
    def _smart_truncate(self, text: str, max_chars: int) -> str:
        """
        Truncate text intelligently, avoiding breaks in LaTeX formulas or code blocks.
        Tries to find a safe truncation point (sentence end, paragraph, etc.).
        """
        if len(text) <= max_chars:
            return text
        
        # Try to find a safe truncation point
        truncated = text[:max_chars]
        
        # Look for safe break points (in order of preference)
        safe_breaks = [
            "\n\n",      # Paragraph break
            ".\n",       # Sentence end with newline  
            ". ",        # Sentence end
            "}\n",       # End of LaTeX block
            "$$\n",      # End of display math
            "$\n",       # End of inline math
            "\n",        # Line break
        ]
        
        best_break = -1
        for break_str in safe_breaks:
            pos = truncated.rfind(break_str)
            if pos > max_chars * 0.5:  # Only accept if we keep at least 50%
                best_break = pos + len(break_str)
                break
        
        if best_break > 0:
            return truncated[:best_break].rstrip() + "\n... (truncated)"
        else:
            # Fallback: just truncate but add indicator
            return truncated.rstrip() + "... (truncated)"

    # ------------------------------------------------------------------ #
    #  Helper: Build self-verify generation batch for low-consistency prompts
    # ------------------------------------------------------------------ #
    def _build_explore_gen_batch(self, gen_batch, low_indices, consistency_results):
        """
        Build a new generation batch for low-consistency prompts with
        self-verify prompts injected.
        
        CRITICAL: The generated verify prompts MUST NOT exceed the original R1
        prompt length to prevent PPO gradient errors. If the verify prompt would
        be longer, we truncate it here (before generation) rather than after,
        ensuring generation and log_prob computation see the same prompt.

        Args:
            gen_batch: Original generation batch (batch_size prompts).
            low_indices: List of prompt indices that need round 2.
            consistency_results: List of dicts with majority_answer / consistency_rate.

        Returns:
            DataProto: New batch ready for ``generate_sequences``.
        """
        # Get the target prompt length from R1 (this is the maximum allowed)
        r1_prompt_len = gen_batch.batch["input_ids"].shape[1]
        
        new_input_ids_list = []

        for prompt_idx in low_indices:
            # Decode original prompt from input_ids (left-padded)
            original_ids = gen_batch.batch["input_ids"][prompt_idx]
            original_mask = gen_batch.batch["attention_mask"][prompt_idx]
            valid_length = int(original_mask.sum().item())
            valid_ids = original_ids[-valid_length:]

            original_prompt = self.tokenizer.decode(valid_ids, skip_special_tokens=False)
            majority_answer = consistency_results[prompt_idx]["majority_answer"]

            # Detect chat format dynamically (supports ChatML, Llama-3, Gemma, plain)
            chat_format = self._detect_chat_format(original_prompt)
            
            # Extract original problem from the prompt using detected format
            problem_text = original_prompt
            user_marker = chat_format["user_start"]
            end_marker = chat_format["end"]
            
            if user_marker and user_marker in original_prompt:
                start_idx = original_prompt.find(user_marker) + len(user_marker)
                end_idx = original_prompt.find(end_marker, start_idx)
                if end_idx != -1:
                    problem_text = original_prompt[start_idx:end_idx].strip()
                else:
                    problem_text = original_prompt[start_idx:].strip()
            
            # Smart truncation of majority_answer to prevent OOM
            # Use smart truncation that avoids breaking LaTeX/code
            max_ans_chars = 1500  # Reduced to leave room for verify template
            ans_str = str(majority_answer) if majority_answer is not None else "unknown"
            ans_str = self._smart_truncate(ans_str, max_ans_chars)
                
            # Format the target template
            verify_text = self.explore_prompt_template.format(
                problem=problem_text,
                majority_answer=ans_str
            )

            # Reconstruct the prompt using detected format
            if chat_format["format"] == "chatml":
                modified_prompt = (
                    f"{chat_format['system_start']}{self.explore_system_prompt}{chat_format['end']}"
                    f"{chat_format['user_start']}{verify_text}{chat_format['end']}"
                    f"{chat_format['assistant_start']}"
                )
            elif chat_format["format"] == "llama3":
                modified_prompt = (
                    f"<|begin_of_text|>"
                    f"{chat_format['system_start']}{self.explore_system_prompt}{chat_format['end']}"
                    f"{chat_format['user_start']}{verify_text}{chat_format['end']}"
                    f"{chat_format['assistant_start']}"
                )
            elif chat_format["format"] == "gemma":
                # Gemma doesn't have system role, include it in user message
                modified_prompt = (
                    f"<bos>"
                    f"{chat_format['user_start']}{self.explore_system_prompt}\n\n{verify_text}{chat_format['end']}"
                    f"{chat_format['assistant_start']}"
                )
            else:
                # Plain text fallback
                modified_prompt = (
                    f"System: {self.explore_system_prompt}\n\n"
                    f"User: {verify_text}\n\n"
                    f"Assistant: "
                )

            # Re-tokenize
            new_ids = self.tokenizer.encode(modified_prompt, add_special_tokens=False)
            
            # CRITICAL: Ensure verify prompt doesn't exceed R1 prompt length
            # This prevents the PPO gradient error where generation uses a different
            # prompt than log_prob computation
            if len(new_ids) > r1_prompt_len:
                print(
                    f"[Explore] WARNING: Verify prompt ({len(new_ids)} tokens) exceeds "
                    f"R1 prompt length ({r1_prompt_len}). Truncating to prevent PPO gradient error."
                )
                # Truncate from the middle of the problem_text to preserve structure
                # We need to reduce by (len(new_ids) - r1_prompt_len) tokens
                excess_tokens = len(new_ids) - r1_prompt_len + 50  # +50 buffer
                
                # Estimate chars to remove (rough: 1 token ≈ 4 chars)
                chars_to_remove = excess_tokens * 4
                
                # Truncate problem_text (the longest part)
                if len(problem_text) > chars_to_remove + 100:
                    problem_text = self._smart_truncate(
                        problem_text, 
                        len(problem_text) - chars_to_remove
                    )
                    
                    # Rebuild and re-tokenize
                    verify_text = self.explore_prompt_template.format(
                        problem=problem_text,
                        majority_answer=ans_str
                    )
                    
                    if chat_format["format"] == "chatml":
                        modified_prompt = (
                            f"{chat_format['system_start']}{self.explore_system_prompt}{chat_format['end']}"
                            f"{chat_format['user_start']}{verify_text}{chat_format['end']}"
                            f"{chat_format['assistant_start']}"
                        )
                    elif chat_format["format"] == "llama3":
                        modified_prompt = (
                            f"<|begin_of_text|>"
                            f"{chat_format['system_start']}{self.explore_system_prompt}{chat_format['end']}"
                            f"{chat_format['user_start']}{verify_text}{chat_format['end']}"
                            f"{chat_format['assistant_start']}"
                        )
                    elif chat_format["format"] == "gemma":
                        modified_prompt = (
                            f"<bos>"
                            f"{chat_format['user_start']}{self.explore_system_prompt}\n\n{verify_text}{chat_format['end']}"
                            f"{chat_format['assistant_start']}"
                        )
                    else:
                        modified_prompt = (
                            f"System: {self.explore_system_prompt}\n\n"
                            f"User: {verify_text}\n\n"
                            f"Assistant: "
                        )
                    
                    new_ids = self.tokenizer.encode(modified_prompt, add_special_tokens=False)
                
                # Final safety: hard truncate if still too long
                if len(new_ids) > r1_prompt_len:
                    new_ids = new_ids[-r1_prompt_len:]  # Keep the end (assistant marker)
            
            new_input_ids_list.append(torch.tensor(new_ids, dtype=torch.long))

            # Print first modified prompt for debugging
            if prompt_idx == low_indices[0]:
                print(f"[Explore] Modified prompt (first sample, idx={prompt_idx}):")
                print(f"  Original length: {valid_length}, New length: {len(new_ids)}, Max allowed: {r1_prompt_len}")
                print(f"  Chat format: {chat_format['format']}")
                # Print the last 200 chars to verify injection
                print(f"  ...{modified_prompt[-200:]}")

        # Pad to same length (left-pad with pad_token_id)
        # Use R1 prompt length as target to ensure consistency
        target_len = r1_prompt_len
        max_len = max(len(ids) for ids in new_input_ids_list)
        # Ensure we pad to at least target_len for proper merge later
        max_len = max(max_len, target_len)
        
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        padded_input_ids = []
        attention_masks = []
        position_ids_list = []

        for ids in new_input_ids_list:
            pad_len = max_len - len(ids)
            padded = torch.cat([
                torch.full((pad_len,), pad_id, dtype=torch.long),
                ids,
            ])
            mask = torch.cat([
                torch.zeros(pad_len, dtype=torch.long),
                torch.ones(len(ids), dtype=torch.long),
            ])
            # position_ids MUST start at 0 for the first valid token
            pos_ids = torch.cat([
                torch.zeros(pad_len, dtype=torch.long),
                torch.arange(len(ids), dtype=torch.long),
            ])

            padded_input_ids.append(padded)
            attention_masks.append(mask)
            position_ids_list.append(pos_ids)

        device = gen_batch.batch["input_ids"].device

        # Build new DataProto
        tensors = {
            "input_ids": torch.stack(padded_input_ids).to(device),
            "attention_mask": torch.stack(attention_masks).to(device),
            "position_ids": torch.stack(position_ids_list).to(device),
        }

        # Carry over raw_prompt_ids for the selected indices
        non_tensors = {}
        if "raw_prompt_ids" in gen_batch.non_tensor_batch:
            non_tensors["raw_prompt_ids"] = gen_batch.non_tensor_batch["raw_prompt_ids"][
                np.array(low_indices)
            ]
        
        # CRITICAL: Carry over extra_info for the selected indices
        # This ensures explore_output has correct index information after generation
        if "extra_info" in gen_batch.non_tensor_batch:
            non_tensors["extra_info"] = gen_batch.non_tensor_batch["extra_info"][
                np.array(low_indices)
            ]

        explore_batch = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)
        explore_batch.meta_info = dict(gen_batch.meta_info)

        return explore_batch

    # ------------------------------------------------------------------ #
    #  Helper: Merge high-consistency R1 outputs + low-consistency R2 outputs
    # ------------------------------------------------------------------ #
    def _merge_explore_outputs(
        self,
        gen_batch_output,
        explore_output,
        high_indices,
        low_indices,
        n_votes_per_prompt,
    ):
        """
        Reconstruct the full gen_batch_output by:
          - Keeping round-1 responses for high-consistency prompts
          - Replacing with round-2 responses for low-consistency prompts

        Args:
            gen_batch_output: Original round-1 output (all prompts * n_votes).
            explore_output: Round-2 output (low-consistency prompts * n_votes).
            high_indices: Prompt indices that keep round-1 responses.
            low_indices: Prompt indices that use round-2 responses.
            n_votes_per_prompt: Number of responses per prompt.

        Returns:
            DataProto: Merged output with same structure as gen_batch_output.
        """
        # Build a mapping: low_idx -> position in explore_output
        low_idx_to_explore = {idx: i for i, idx in enumerate(low_indices)}

        # Collect output DataProtos in original prompt order
        all_protos = []
        total_prompts = len(high_indices) + len(low_indices)

        for prompt_idx in range(total_prompts):
            start = prompt_idx * n_votes_per_prompt
            end = start + n_votes_per_prompt

            if prompt_idx in low_idx_to_explore:
                # Use round-2 output
                explore_pos = low_idx_to_explore[prompt_idx]
                e_start = explore_pos * n_votes_per_prompt
                e_end = e_start + n_votes_per_prompt
                all_protos.append(explore_output[e_start:e_end])
            else:
                # Use round-1 output
                all_protos.append(gen_batch_output[start:end])

        # Find the maximum dimensions for all 2D tensors to pad them safely
        # keys usually are: 'responses', 'prompts', etc.
        tensor_keys = list(all_protos[0].batch.keys())
        max_shapes = {key: 0 for key in tensor_keys if all_protos[0].batch[key].dim() == 2}

        # Calculate max sequence length for each 2D tensor
        for proto in all_protos:
            for key in max_shapes:
                if proto.batch[key].shape[1] > max_shapes[key]:
                    max_shapes[key] = proto.batch[key].shape[1]

        # Pad all 2D tensors to match the max dimensions
        for proto in all_protos:
            for key in max_shapes:
                tensor = proto.batch[key]
                pad_len = max_shapes[key] - tensor.shape[1]
                if pad_len > 0:
                    # Pad on the right side with 0
                    import torch.nn.functional as F
                    padded_tensor = F.pad(tensor, (0, pad_len), value=self.tokenizer.pad_token_id if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None else 0)
                    proto.batch[key] = padded_tensor

        merged = DataProto.concat(all_protos)
        return merged

    # ------------------------------------------------------------------ #
    #  Main training loop (overrides RayPPOTrainer.fit)
    # ------------------------------------------------------------------ #
    def fit(self):
        """
        The training loop of PPO with optional two-stage exploration.

        This is a full override of RayPPOTrainer.fit() that adds the
        explore-rollout logic between generation and reward computation.
        All original metrics, advantage estimators, and logging are preserved.
        """
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(
            total=self.total_training_steps, initial=self.global_steps, desc="Training Progress"
        )

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                batch.meta_info["do_vote"] = False
                if self.use_ttrl:
                    self.config.actor_rollout_ref.rollout.n = self.n_votes_per_prompt
                    batch.meta_info["do_vote"] = True

                # pop those keys for generation
                if "multi_modal_inputs" in batch.non_tensor_batch.keys():
                    gen_batch = batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "multi_modal_inputs"],
                        meta_info_keys=["do_vote"],
                    )
                else:
                    gen_batch = batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                        meta_info_keys=["do_vote"],
                    )
                
                # Copy extra_info to gen_batch without removing it from batch
                # gen_batch needs it for _build_explore_gen_batch, batch needs it for sorting
                if "extra_info" in batch.non_tensor_batch:
                    gen_batch.non_tensor_batch["extra_info"] = batch.non_tensor_batch["extra_info"]

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # ============================================================
                    # Round 1: Generate initial rollouts
                    # ============================================================
                    with _timer("gen", timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        if self.use_ttrl:
                            assert len(gen_batch_output) == len(batch) * self.n_votes_per_prompt

                    # ============================================================
                    # Two-Stage Exploration (if enabled)
                    # ============================================================
                    if self.use_explore_rollout and self.use_ttrl:
                        with _timer("explore", timing_raw):
                            # --- Consistency Evaluation ---
                            # Build a temporary merged batch for consistency computation
                            tmp_batch = batch.repeat(
                                repeat_times=self.n_votes_per_prompt, interleave=True
                            )
                            tmp_batch = tmp_batch.union(gen_batch_output)

                            # Sort by index for correct per-prompt grouping
                            # This sorted order will be used for ALL subsequent operations
                            sorted_indices = sorted(
                                range(len(tmp_batch)),
                                key=lambda i: tmp_batch[i].non_tensor_batch["extra_info"]["index"],
                            )
                            tmp_batch = tmp_batch[sorted_indices]
                            
                            # CRITICAL: Use the SAME sorted_indices to reorder gen_batch_output
                            # gen_batch_output itself doesn't have extra_info, but tmp_batch.union()
                            # gave it the same ordering as batch.repeat(). So sorted_indices applies.
                            gen_batch_output = gen_batch_output[sorted_indices]
                            
                            # Also sort gen_batch and batch to align indices for _build_explore_gen_batch
                            # gen_batch has extra_info (it's a subset of batch)
                            gen_batch_sorted_indices = sorted(
                                range(len(gen_batch)),
                                key=lambda i: gen_batch[i].non_tensor_batch["extra_info"]["index"],
                            )
                            gen_batch = gen_batch[gen_batch_sorted_indices]
                            
                            # Also sort `batch` to match for later batch.repeat().union()
                            batch_sorted_indices = sorted(
                                range(len(batch)),
                                key=lambda i: batch[i].non_tensor_batch["extra_info"]["index"],
                            )
                            batch = batch[batch_sorted_indices]

                            # Compute consistency per prompt
                            consistency_results = self.reward_fn.compute_majority_and_consistency(
                                tmp_batch
                            )

                            # --- Split into high / low consistency ---
                            # Now indices refer to sorted order (gen_batch, gen_batch_output, batch are ALL sorted)
                            high_indices = []
                            low_indices = []
                            prompt_num = len(consistency_results)
                            for i, result in enumerate(consistency_results):
                                if result["consistency_rate"] > self.explore_threshold:
                                    high_indices.append(i)
                                else:
                                    low_indices.append(i)

                            round2_count = len(low_indices)
                            round2_ratio = round2_count / prompt_num if prompt_num > 0 else 0.0

                            # Log explore metrics
                            metrics["explore/round2_sample_quantity"] = round2_count
                            metrics["explore/round2_sample_ratio"] = round2_ratio

                            print(
                                f"[Explore] Round 2 triggered for {round2_count}/{prompt_num} "
                                f"prompts ({round2_ratio:.2%}), threshold={self.explore_threshold}"
                            )

                            # --- Round 2 Rollout (if needed) ---
                            if low_indices:
                                # Clean up temporary batch before generating Round 2 to save memory
                                del tmp_batch
                                gc.collect()
                                torch.cuda.empty_cache()

                                # Build self-verify prompts for low-consistency samples
                                explore_gen_batch = self._build_explore_gen_batch(
                                    gen_batch, low_indices, consistency_results
                                )

                                # Pad for DP
                                explore_gen_batch_padded, explore_pad_size = (
                                    pad_dataproto_to_divisor(
                                        explore_gen_batch, self.actor_rollout_wg.world_size
                                    )
                                )

                                # Generate round-2 responses
                                explore_output_padded = (
                                    self.actor_rollout_wg.generate_sequences(explore_gen_batch_padded)
                                )

                                # Unpad (must unpad padding prompts, taking N_votes into account!)
                                explore_output = unpad_dataproto(
                                    explore_output_padded, pad_size=explore_pad_size * self.n_votes_per_prompt
                                )

                                # Merge: keep high-consistency R1, replace low-consistency with R2
                                gen_batch_output = self._merge_explore_outputs(
                                    gen_batch_output,
                                    explore_output,
                                    high_indices,
                                    low_indices,
                                    self.n_votes_per_prompt,
                                )

                                del explore_gen_batch, explore_gen_batch_padded
                                del explore_output_padded, explore_output
                                gc.collect()
                                torch.cuda.empty_cache()
                            else:
                                del tmp_batch
                                gc.collect()
                                torch.cuda.empty_cache()

                    # ============================================================
                    # REMAX baseline (unchanged from original)
                    # ============================================================
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(
                                gen_baseline_batch
                            )

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    # ============================================================
                    # Standard PPO pipeline (unchanged from original)
                    # ============================================================
                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                    )
                    batch = batch.union(gen_batch_output)

                    batch.batch["response_mask"] = compute_response_mask(batch)
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(
                        batch.batch["attention_mask"], dim=-1
                    ).tolist()

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(
                            loss_mat=entropys,
                            loss_mask=response_masks,
                            loss_agg_mode=loss_agg_mode,
                        )
                        old_log_prob_metrics = {
                            "actor/entropy_loss": entropy_loss.detach().item(),
                            "train/entropy": entropy_loss.detach().item(),
                        }
                        old_log_prob_metrics["actor/entropy_mean_eval"] = (
                            entropys.mean().detach().item()
                        )
                        metrics.update(old_log_prob_metrics)
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        with _timer("ref", timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # compute scores
                        if self.use_ttrl:
                            # Re-sort batch by index to ensure correct per-prompt grouping
                            # This is critical after explore merge which may change order
                            sorted_indices = sorted(
                                range(len(batch)),
                                key=lambda i: batch[i].non_tensor_batch["extra_info"]["index"],
                            )
                            batch = batch[sorted_indices]
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        reward_extra_infos_dict: dict[str, list]
                        try:
                            reward_result = self.reward_fn(batch, return_dict=True)
                            reward_tensor = reward_result["reward_tensor"]
                            reward_extra_infos_dict = reward_result["reward_extra_info"]
                            if self.use_ttrl:
                                ttrl_metrics = reward_result["ttrl_info"]
                                for k, v in ttrl_metrics.items():
                                    if not k.startswith("_"):
                                        metrics.update({f"train/{k}": v})

                                # Down Sampling
                                batch = self._select_top_k_per_prompt(
                                    batch, self.n_votes_per_prompt, self.n_samples_per_prompt
                                )
                                self.config.actor_rollout_ref.rollout.n = self.n_samples_per_prompt

                                # Recompute ttrl metrics
                                post_reward_result = self.reward_fn.compute_post_ttrl_metrics(batch)
                                for k, v in post_reward_result.items():
                                    metrics.update({f"train/{k}": v})

                                # Recompute Entropy
                                post_entropy_loss = agg_loss(
                                    loss_mat=batch.batch["entropys"],
                                    loss_mask=batch.batch["response_mask"],
                                    loss_agg_mode=loss_agg_mode,
                                )
                                metrics.update(
                                    {"train/post_entropy": post_entropy_loss.detach().item()}
                                )

                                if "_answer_types" in ttrl_metrics:
                                    batch.non_tensor_batch["answer_types"] = ttrl_metrics[
                                        "_answer_types"
                                    ]
                                if "_oracle_answer_types" in ttrl_metrics:
                                    batch.non_tensor_batch["oracle_answer_types"] = ttrl_metrics[
                                        "_oracle_answer_types"
                                    ]
                                if "_consistency_rate" in ttrl_metrics:
                                    batch.non_tensor_batch["consistency_rate"] = ttrl_metrics[
                                        "_consistency_rate"
                                    ]
                                if "_accuracy_rate" in ttrl_metrics:
                                    batch.non_tensor_batch["accuracy_rate"] = ttrl_metrics[
                                        "_accuracy_rate"
                                    ]
                                if "_label_accuracy" in ttrl_metrics:
                                    batch.non_tensor_batch["label_accuracy"] = ttrl_metrics[
                                        "_label_accuracy"
                                    ]
                                if "_zero_advantage_mask" in ttrl_metrics:
                                    batch.non_tensor_batch["zero_advantage_mask"] = ttrl_metrics[
                                        "_zero_advantage_mask"
                                    ]

                        except Exception as e:
                            print(f"Error in reward_fn: {e}")
                            reward_tensor = self.reward_fn(batch)
                            reward_extra_infos_dict = {}

                        batch.batch["token_level_scores"] = reward_tensor

                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch,
                                kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages
                        diversity_density_config = None
                        if self.config.algorithm.adv_estimator in [
                            AdvantageEstimator.PASS_GRPO,
                            AdvantageEstimator.PASS_GRPO_PENALIZED,
                        ]:
                            diversity_density_config = {
                                "k": getattr(self.config.algorithm, "diversity_density_k", 4),
                                "fallback_estimator": getattr(
                                    self.config.algorithm, "diversity_density_fallback", "grpo"
                                ),
                                "use_metric": getattr(
                                    self.config.algorithm,
                                    "diversity_density_use_metric",
                                    "consistency_rate",
                                ),
                                "consistency_threshold": getattr(
                                    self.config.algorithm, "consistency_threshold", 0.0
                                ),
                                "lam_div": getattr(self.config.algorithm, "lam_div", 0.05),
                                "c_max": getattr(self.config.algorithm, "c_max", 2.0),
                                "tau_rep": getattr(self.config.algorithm, "tau_rep", 0.2),
                                "gamma": getattr(self.config.algorithm, "gamma", 1.0),
                                "p_max": getattr(self.config.algorithm, "p_max", 0.15),
                                "n_gram_size": getattr(self.config.algorithm, "n_gram_size", 3),
                                "use_rep_penalty": getattr(
                                    self.config.algorithm, "use_rep_penalty", False
                                ),
                                "div_sc_threshold": getattr(
                                    self.config.algorithm, "div_sc_threshold", 0.5
                                ),
                            }

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            diversity_density_config=diversity_density_config,
                        )

                        # Apply zero_advantage_mask if present
                        if "zero_advantage_mask" in batch.non_tensor_batch:
                            zero_mask = torch.tensor(
                                batch.non_tensor_batch["zero_advantage_mask"],
                                dtype=torch.float32,
                                device=batch.batch["advantages"].device,
                            ).unsqueeze(-1)
                            batch.batch["advantages"] = batch.batch["advantages"] * (1.0 - zero_mask)
                            n_zeroed = int(zero_mask.sum().item())
                            print(
                                f"[test_minority] Applied zero_advantage_mask: "
                                f"zeroed {n_zeroed}/{len(zero_mask)} samples"
                            )
                            metrics["train/test_minority_zeroed_ratio"] = float(
                                zero_mask.mean().item()
                            )

                        # Log diversity density usage statistics if available
                        if "diversity_density_ratio" in batch.meta_info:
                            metrics["train/diversity_density_ratio"] = float(
                                batch.meta_info["diversity_density_ratio"]
                            )
                        if "fallback_ratio" in batch.meta_info:
                            metrics["train/fallback_ratio"] = float(
                                batch.meta_info["fallback_ratio"]
                            )

                        # === Advantage Bias Diagnostics ===
                        if (
                            "oracle_answer_types" in batch.non_tensor_batch
                            and self.config.algorithm.adv_estimator == AdvantageEstimator.PASS_GRPO
                            and diversity_density_config is not None
                        ):
                            try:
                                oracle_adv, _ = core_algos.compute_pass_grpo_advantage(
                                    token_level_rewards=batch.batch["token_level_rewards"],
                                    response_mask=batch.batch["response_mask"],
                                    index=batch.non_tensor_batch["uid"],
                                    answer_types=batch.non_tensor_batch["oracle_answer_types"],
                                    k=diversity_density_config["k"],
                                )
                                tta_adv = batch.batch["advantages"]

                                tta_scalar = tta_adv.sum(-1)
                                oracle_scalar = oracle_adv.sum(-1)
                                valid = batch.batch["response_mask"].sum(-1) > 0

                                if valid.any():
                                    sign_match = (
                                        (tta_scalar > 0) == (oracle_scalar > 0)
                                    ).float()
                                    metrics["diag/adv_sign_match_rate"] = (
                                        sign_match[valid].mean().item()
                                    )
                                    metrics["diag/adv_mse"] = (
                                        ((tta_scalar - oracle_scalar) ** 2)[valid].mean().item()
                                    )
                                    metrics["diag/adv_mean_bias"] = (
                                        (tta_scalar - oracle_scalar)[valid].mean().item()
                                    )
                            except Exception as e:
                                print(f"Warning: Advantage bias diagnostics failed: {e}")

                        # Log pass_grpo diagnostic metrics if available
                        if "pass_grpo/correct_ratio" in batch.meta_info:
                            metrics["train/pass_grpo_correct_ratio"] = float(
                                batch.meta_info["pass_grpo/correct_ratio"]
                            )
                        if "pass_grpo/avg_correct_advantage" in batch.meta_info:
                            metrics["train/pass_grpo_avg_correct_adv"] = float(
                                batch.meta_info["pass_grpo/avg_correct_advantage"]
                            )
                        if "pass_grpo/avg_incorrect_advantage" in batch.meta_info:
                            metrics["train/pass_grpo_avg_incorrect_adv"] = float(
                                batch.meta_info["pass_grpo/avg_incorrect_advantage"]
                            )
                        if "pass_grpo/avg_total_advantage" in batch.meta_info:
                            metrics["train/pass_grpo_avg_total_adv"] = float(
                                batch.meta_info["pass_grpo/avg_total_advantage"]
                            )

                        # Log diversity density advantage metrics
                        if "diversity/avg_advantage" in batch.meta_info:
                            metrics["train/diversity_avg_adv"] = float(
                                batch.meta_info["diversity/avg_advantage"]
                            )

                        # Log bootstrap_passk metrics if available
                        for bp_key in [
                            "bootstrap_passk/num_low_prompts",
                            "bootstrap_passk/num_high_prompts",
                            "bootstrap_passk/low_ratio",
                            "bootstrap_passk/avg_low_advantage",
                            "bootstrap_passk/avg_high_advantage",
                            "bootstrap_passk/avg_total_advantage",
                        ]:
                            if bp_key in batch.meta_info:
                                metrics[f"train/{bp_key.replace('/', '_')}"] = float(
                                    batch.meta_info[bp_key]
                                )

                        # Log pass_grpo_penalized metrics if available
                        for pp_key in [
                            "pass_grpo_penalized/avg_r_div",
                            "pass_grpo_penalized/r_div_triggered_ratio",
                            "pass_grpo_penalized/avg_raw_a_passk",
                            "pass_grpo_penalized/avg_adv_raw",
                            "pass_grpo_penalized/avg_total_advantage",
                        ]:
                            if pp_key in batch.meta_info:
                                metrics[f"train/{pp_key.replace('/', '_')}"] = float(
                                    batch.meta_info[pp_key]
                                )

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(
                    compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus)
                )

                logger.log(data=metrics, step=self.global_steps)

                # Explicitly free batch and metrics
                del batch, batch_dict, metrics, timing_raw
                if "gen_batch" in locals():
                    del gen_batch
                if "gen_batch_output" in locals():
                    del gen_batch_output
                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
