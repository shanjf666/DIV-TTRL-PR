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

        # Default self-verify prompt template. {majority_answer} will be replaced.
        default_template = (
            "\n\nA previous attempt suggests the answer might be: {majority_answer}\n"
            "Please verify this answer by solving the problem again carefully. "
            "If you agree, confirm the answer. If not, provide the correct answer."
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
    #  Helper: Build self-verify generation batch for low-consistency prompts
    # ------------------------------------------------------------------ #
    def _build_explore_gen_batch(self, gen_batch, low_indices, consistency_results):
        """
        Build a new generation batch for low-consistency prompts with
        self-verify prompts injected.

        Args:
            gen_batch: Original generation batch (batch_size prompts).
            low_indices: List of prompt indices that need round 2.
            consistency_results: List of dicts with majority_answer / consistency_rate.

        Returns:
            DataProto: New batch ready for ``generate_sequences``.
        """
        new_input_ids_list = []

        for prompt_idx in low_indices:
            # Get the majority answer to build the verification prompt
            result = consistency_results[prompt_idx]
            majority_answer = result.get("majority_answer", "None")
            verify_text = f"The previous majority answer is {majority_answer}. Please verify if it is correct and provide your own derivation."

            # Use raw_prompt (messages list) if available for cleaner templating
            messages = None
            if "raw_prompt" in gen_batch.non_tensor_batch:
                messages = list(gen_batch.non_tensor_batch["raw_prompt"][prompt_idx])

            if messages is not None:
                # Add/Embed the verify text into the message list
                new_messages = deepcopy(messages)
                if new_messages and new_messages[-1]["role"] == "user":
                    new_messages[-1]["content"] += "\n\n" + verify_text
                else:
                    new_messages.append({"role": "user", "content": verify_text})

                # Apply chat template with enable_thinking=False
                modified_prompt = self.tokenizer.apply_chat_template(
                    new_messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
                )
            else:
                # Original fallback: Manual string injection for Base models without full message history
                original_ids = gen_batch.batch["input_ids"][prompt_idx]
                original_mask = gen_batch.batch["attention_mask"][prompt_idx]
                valid_length = int(original_mask.sum().item())
                valid_ids = original_ids[-valid_length:]
                original_prompt = self.tokenizer.decode(valid_ids, skip_special_tokens=False)

                assistant_marker = "<|im_start|>assistant"
                if assistant_marker in original_prompt:
                    insert_pos = original_prompt.rfind(assistant_marker)
                    end_marker = "<|im_end|>"
                    end_pos = original_prompt.rfind(end_marker, 0, insert_pos)
                    if end_pos >= 0:
                        modified_prompt = (
                            original_prompt[:end_pos] + verify_text + original_prompt[end_pos:]
                        )
                    else:
                        modified_prompt = (
                            original_prompt[:insert_pos] + verify_text + "\n" + original_prompt[insert_pos:]
                        )
                else:
                    modified_prompt = original_prompt + verify_text

            # Re-tokenize
            new_ids = self.tokenizer.encode(modified_prompt, add_special_tokens=False)
            new_input_ids_list.append(torch.tensor(new_ids, dtype=torch.long))

            # Print first modified prompt for debugging
            if prompt_idx == low_indices[0]:
                print(f"[Explore] Modified prompt (first sample, idx={prompt_idx}):")
                print(f"  Original length: {valid_length}, New length: {len(new_ids)}")
                # Print the last 200 chars to verify injection
                print(f"  ...{modified_prompt[-200:]}")

        # Pad to same length (left-pad with pad_token_id)
        max_len = max(len(ids) for ids in new_input_ids_list)
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
            pos_ids = torch.arange(max_len, dtype=torch.long)

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
        if "raw_prompt" in gen_batch.non_tensor_batch:
            non_tensors["raw_prompt"] = gen_batch.non_tensor_batch["raw_prompt"][
                np.array(low_indices)
            ]

        explore_batch = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)
        explore_batch.meta_info = dict(gen_batch.meta_info)

        return explore_batch

    # ------------------------------------------------------------------ #
    #  Helper: In-place replace low-consistency R1 slices with R2 outputs
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
        In-place replacement: directly overwrite the low-consistency prompt
        slices in gen_batch_output with the Round-2 explore_output.

        This avoids DataProto.concat entirely, eliminating padding/length
        mismatch bugs. If R2 has a different sequence length than R1, the
        shorter side is padded to match.
        """
        if not low_indices or explore_output is None:
            return gen_batch_output

        # Get sequence lengths from both sides
        r1_seq_len = gen_batch_output.batch["input_ids"].shape[1]
        r2_seq_len = explore_output.batch["input_ids"].shape[1]
        target_len = max(r1_seq_len, r2_seq_len)

        # Helper: pad a single tensor along dim=1 to target_len
        def _pad_to(tensor, target, pad_val=0):
            if tensor.shape[1] >= target:
                return tensor
            pad_shape = (tensor.shape[0], target - tensor.shape[1]) + tensor.shape[2:]
            padding = torch.full(pad_shape, pad_val, dtype=tensor.dtype, device=tensor.device)
            return torch.cat([tensor, padding], dim=1)

        # Step 1: If R1 is shorter, expand gen_batch_output to target_len
        if r1_seq_len < target_len:
            new_tensors = {}
            for key, tensor in gen_batch_output.batch.items():
                if tensor.dim() >= 2:
                    pad_val = self.tokenizer.pad_token_id if "input_ids" in key else 0
                    new_tensors[key] = _pad_to(tensor, target_len, pad_val)
                else:
                    new_tensors[key] = tensor
            gen_batch_output.batch = TensorDict(
                source=new_tensors, batch_size=gen_batch_output.batch.batch_size
            )

        # Step 2: For each low-consistency prompt, overwrite its slice
        for explore_pos, prompt_idx in enumerate(low_indices):
            r1_start = prompt_idx * n_votes_per_prompt
            r1_end = r1_start + n_votes_per_prompt
            e_start = explore_pos * n_votes_per_prompt
            e_end = e_start + n_votes_per_prompt

            # Overwrite tensor batch entries
            for key, tensor in gen_batch_output.batch.items():
                r2_tensor = explore_output.batch[key][e_start:e_end]
                if tensor.dim() >= 2 and r2_tensor.shape[1] < target_len:
                    pad_val = self.tokenizer.pad_token_id if "input_ids" in key else 0
                    r2_tensor = _pad_to(r2_tensor, target_len, pad_val)
                gen_batch_output.batch[key][r1_start:r1_end] = r2_tensor

            # Overwrite non-tensor batch entries
            for key in gen_batch_output.non_tensor_batch:
                if key in explore_output.non_tensor_batch:
                    gen_batch_output.non_tensor_batch[key][r1_start:r1_end] = (
                        explore_output.non_tensor_batch[key][e_start:e_end]
                    )

        return gen_batch_output


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
                        non_tensor_batch_keys=["raw_prompt_ids", "raw_prompt"],
                        meta_info_keys=["do_vote"],
                    )

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
                            sorted_indices = sorted(
                                range(len(tmp_batch)),
                                key=lambda i: tmp_batch[i].non_tensor_batch["extra_info"]["index"],
                            )
                            tmp_batch = tmp_batch[sorted_indices]

                            # Compute consistency per prompt
                            consistency_results = self.reward_fn.compute_majority_and_consistency(
                                tmp_batch
                            )

                            # --- Split into high / low consistency ---
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

                                # Unpad
                                explore_output = unpad_dataproto(
                                    explore_output_padded, pad_size=explore_pad_size
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
