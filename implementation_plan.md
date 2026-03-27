# Implementation Plan: Two-Stage Exploration Training Mode

## Goal Description
Implement a new two-stage "exploration" training mode. The flow generates an initial set of rollouts and evaluates their consistency. For high-consistency samples, the initial rollouts are kept and used for training. For low-consistency samples, rollouts are re-generated using a "self-verify" prompt containing the previous majority answer, and only this second round of samples is used for training (their first round is discarded). Finally, GRPO updates are performed on the combined batch of high-consistency round 1 samples and low-consistency round 2 samples.

## User Review Required
> [!IMPORTANT]
> **Prompt Injection**: Constructing the new prompt requires extracting the original problem text. Since `gen_batch` contains tokenized inputs, we need to carefully decode, string-manipulate, and re-tokenize without breaking the chat template (e.g., `<|im_start|>assistant`).

---

## Proposed Changes

### Configuration Layer
- Add new arguments to the trainer configuration (e.g., `algorithm.use_explore_rollout: bool`, `algorithm.explore_threshold: float`, `algorithm.explore_prompt_template: str`).

### `verl/verl/workers/reward_manager/ttrl.py`
#### [MODIFY] `ttrl.py`(file:///d:/Repository/GitHub/DIV-TTRL/verl/verl/workers/reward_manager/ttrl.py)
Summary of changes:
- Add a new lightweight method `compute_majority_and_consistency(self, data: DataProto)` to do a quick pass of `auto_extract` over a generated batch to yield the `majority_answer` and `consistency_rate` for each prompt without doing the heavy full-reward computation or entropy tracking.

### `verl/verl/trainer/ppo/explore_ray_trainer.py` [NEW]
Summary of changes:
- Duplicate `ray_trainer.py` to create `explore_ray_trainer.py`. We create a new file because `ray_trainer.py` is very large (1400 lines) and injecting complex batch-reconstruction logic natively could introduce bugs to standard training.
- Check `if self.config.algorithm.use_explore_rollout` in the `fit` method.
- **Round 1 Rollout**: Call `gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)`.
- **Consistency Evaluation**: Merge `gen_batch` and `gen_batch_output` into a temporary batch and call `ttrl_manager.compute_majority_and_consistency(tmp_batch)`.
- **Batch Splitting & Prompt Modification**: 
  - Identify prompts with `consistency_rate <= threshold`.
  - For those low-consistency prompts, parse out the original prompt/problem text, inject the `self_verify_experiment` template, and re-tokenize. Create a new data batch `low_consistency_batch` containing only these rewritten prompts.
  - The high-consistency prompts (and their round 1 rollouts) are mapped aside.
- **Round 2 Rollout**: If there are low-consistency samples, call `explore_batch_output = self.actor_rollout_wg.generate_sequences(low_consistency_batch)`.
- **Batch Reconstruction**: Merge the high-consistency round 1 outputs with the newly generated `explore_batch_output`.
- **Proceed to Standard Training**: Use the newly merged batch as the data for the rest of the PPO pipeline (recomputing rewards, advantages, and updating actor).
- **Retain Original Metrics & Capabilities**: Ensure `explore_ray_trainer.py` preserves all optional advantage estimators and calculation metrics that exist in the original `ray_trainer.py`. It should act as a strict superset of the original capabilities.
- **Log Exploration Metrics to Wandb**: Add explicit tracking for the second round of generation. Log the following to Wandb at each iteration:
  - `explore/round2_sample_quantity`: The absolute number of prompts that triggered a round 2 rollout.
  - `explore/round2_sample_ratio`: The proportion of these prompts relative to the entire batch size.

### `verl/verl/trainer/main_explore_ppo.py` [NEW]
- Duplicate `main_ppo.py` to `main_explore_ppo.py` which instantiates our new `RayExploreTrainer` instead of `RayPPOTrainer`.

---

## Verification Plan
### Automated Tests
- Run `tests/e2e/run_ray_trainer.sh` locally with the newly configured explore parameters to ensure the training loop does not crash and gradients flow.
- Ensure that the new Wandb metrics (`explore/round2_sample_quantity`, `explore/round2_sample_ratio`) are actively logging and that the original `ray_trainer` metrics (rewards, KL, advantage distributions) are intact and updating correctly.

### Manual Verification
- Output log inspection: Add print statements for the first modified prompt per node to visually confirm that the explore prompt was successfully injected before `<|im_start|>assistant` without breaking tokenization syntax.
