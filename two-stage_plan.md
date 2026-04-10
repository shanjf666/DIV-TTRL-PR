# Shared-Weight Two-Stage Generator-Verifier Pipeline Plan

This document outlines the detailed implementation plan for a two-stage math problem pseudo-labeling pipeline in `verl`. The training model acts as its own verifier during the rollout phase.

## 1. Overview and Architecture
The verification inference is performed by the `ActorRolloutWorker` (Ray workers holding the model weights). The PPO/GRPO training loop in `ray_trainer.py` becomes a **two-pass rollout process**.

A critical challenge is that `ActorRolloutWorker` operates on tensor-based `DataProto` objects (containing `input_ids`, `attention_mask`), not plain strings. Furthermore, Pass 2 involves significantly longer prompts and generations, presenting a high risk of Out-Of-Memory (OOM) errors.

## 2. OOM Mitigation & DataTransformation Strategies

### 2.1 DataTransformation (`DataProto` ↔ Text)
1. **Decoding:** Extract the prompt and responses from Pass 1 `DataProto` tensors using the `Tokenizer`.
2. **Template Filling:** Construct the new Verification Prompts (System + Problem + Hypothesis + Instructions) as plain text.
3. **Encoding:** Tokenize the new prompts back into `input_ids` and `attention_mask`, and pack them into a brand-new `DataProto` for Pass 2.

### 2.2 OOM Mitigation
Because Pass 2 prompts prepend a large hypothesis and enforce a verbose `<reverse_verification>` and `<final_solution>` reasoning chain, memory consumption will spike. We address this via:
- **Micro-Batching (Chunking):** Instead of passing the entire Pass 2 `DataProto` to `generate_sequences` at once, we split it into smaller micro-batches (e.g., `pass1_batch_size // 4`).
- **Strict Length Clamping:** 
  - Cap the length of the extracted `majority_answer` before inserting it into the prompt.
  - Dynamically adjust (or cap) `max_new_tokens` for Pass 2 to prevent runaway XML generation.
- **VRAM Cleanup:** Explicitly call `torch.cuda.empty_cache()` between Passes and between micro-batches if necessary.
- **Candidate Pruning:** Only generate Pass 2 Verification for the *top K* (e.g., K=1 or 2) unique answers to minimize the workload.

## 3. Verification Prompts

**Global System Prompt:**
```text
You are a rigorous mathematical reviewer.
```

**User Prompt Template:**
```text
Problem:
{problem}

[Hypothesis to Test]
A previous attempt at this problem resulted in the following high-frequency answer:
{majority_answer}

[Task]
Act as a rigorous mathematical reviewer.
1. Reverse Verification Stage: Treat the previous answer ({majority_answer}) as a given hypothesis. Plug this answer BACK into the original problem conditions. Perform a rigorous backward-substitution to check if it satisfies all constraints or if it leads to a mathematical contradiction. You MUST conclude this stage by explicitly stating either "Verification Result: True" (if the hypothesis perfectly satisfies all conditions) or "Verification Result: False" (if it leads to any contradiction).
2. Solution Stage: Based on the insights gained from your reverse verification, explore a robust reasoning path to independently solve the problem from scratch.

You MUST strictly use the following XML format for your response:
<reverse_verification>
(Your step-by-step backward substitution checking if {majority_answer} contradicts the problem conditions)
Verification Result: [True/False]
</reverse_verification>
<final_solution>
(Your complete, alternative step-by-step mathematical derivation)
Therefore, the final answer is \boxed{{...}}
</final_solution>
```

## 4. Concrete Integration Plan into `verl`

### 4.1. Refactoring the PPO Loop `verl/trainer/ppo/ray_trainer.py`

#### Pseudo-Code for `ray_trainer.py` Loop:
```python
import torch
from verl import DataProto
from verl.utils.reward_score.ttrl.two_stage_utils import construct_verification_dataproto, resolve_pseudo_labels

# --- Pass 1: Generate initial solutions ---
rollout_data_pass_1 = self.actor_rollout_wg.generate_sequences(batch)

# Free VRAM immediately after Pass 1
torch.cuda.empty_cache()

# --- Intermediary: Data transformation & DataProto construction ---
# 1. Decode Pass 1 tensors to strings using self.tokenizer
# 2. Extract answers, count frequencies, build string prompts
# 3. Encode back to a new DataProto
verification_batch, verification_mapping = construct_verification_dataproto(
    rollout_data_pass_1, 
    tokenizer=self.tokenizer,
    system_prompt=SYSTEM_PROMPT, 
    user_template=USER_TEMPLATE,
    verification_mode=self.config.verification_mode,
    max_candidates=2 # Limit to top 2 to save OOM
)

if verification_batch is not None and len(verification_batch) > 0:
    # Adjust GenConfig for Pass 2
    temp_gen_config = self.gen_config.copy()
    temp_gen_config['temperature'] = 0.0 if self.config.verification_mode == 'greedy' else 0.6
    # Optional: override max_new_tokens if it's too large for the available sequence length
    
    # --- OOM Mitigation: Micro-Batching ---
    micro_batch_size = max(1, self.config.train_batch_size // 4)
    verification_outputs_decoded = []
    
    for i in range(0, len(verification_batch), micro_batch_size):
        chunk = verification_batch[i:i+micro_batch_size]
        
        # Pass 2: Generate verification reasoning
        chunk_rollout = self.actor_rollout_wg.generate_sequences(chunk, generation_config=temp_gen_config)
        
        # Decode chunk tensors back to strings for regex parsing
        for i in range(len(chunk_rollout)):
            resp_ids = chunk_rollout.batch['responses'][i]
            # only decode valid tokens using attention_mask
            valid_len = chunk_rollout.batch['attention_mask'][i].sum()
            text = self.tokenizer.decode(resp_ids[:valid_len], skip_special_tokens=True)
            verification_outputs_decoded.append(text)
            
        torch.cuda.empty_cache() # cleanup per chunk
else:
    verification_outputs_decoded = []

# --- Intermediary: Parse Verification Results and assign Pseudo-Labels ---
pseudo_labels = resolve_pseudo_labels(
    verification_outputs=verification_outputs_decoded, 
    mapping=verification_mapping,
    mode=self.config.verification_mode
)

# Inject the verified pseudo_labels into the Pass 1 data for reward computation
# We write this into non_tensor_batch so TTRLRewardManager can read it
for i in range(len(rollout_data_pass_1)):
    rollout_data_pass_1[i].non_tensor_batch["verified_pseudo_label"] = pseudo_labels[i]

# --- Reward Assignment ---
reward_tensor = self.reward_model_wg.compute_reward(rollout_data_pass_1)
```

### 4.2. Utility Module (`verl/utils/reward_score/ttrl/two_stage_utils.py`)
Handles parsing and generation of `DataProto`.

```python
import re
import torch
from typing import List, Dict, Tuple
from verl import DataProto

def construct_verification_dataproto(pass_1_data: DataProto, tokenizer, system_prompt, user_template, mode, max_candidates=2) -> Tuple[DataProto, Dict]:
    """
    Decodes Pass 1 outputs, extracts majority answers, formats the templates,
    and tokenizes them into a new DataProto ready for Pass 2.
    Returns: newly created VERIFICATION_DATAPROTO and a MAPPING_DICT to track 
    which candidate corresponds to which original prompt.
    """
    # 1. Decode Loop ...
    # 2. Extract Answers & Limit to top `max_candidates` ...
    # 3. Format Strings ...
    # 4. Tokenize ...
    # return DataProto.from_dict({'prompts': ... }), mapping
    pass

def parse_verification_result(text: str) -> bool:
    """Extracts 'Verification Result: True/False' from the verifier's completion."""
    match = re.search(r"Verification Result:\s*(True|False)", text, re.IGNORECASE)
    return match.group(1).lower() == 'true' if match else False

def resolve_pseudo_labels(verification_outputs: List[str], mapping: Dict, mode: str) -> List[str]:
    """
    Matches the flat parsed boolean results back to the original Pass 1 sample indices
    using the mapping. Resolves ties using frequency.
    """
    pass
```

### 4.3. Update `TTRLRewardManager` (`ttrl.py`)
Modify the reward function to utilize the dynamic `verified_pseudo_label` we injected during `ray_trainer.py`.

```python
# In `TTRLRewardManager._compute_ttrl_reward`

verified_label = data_item.non_tensor_batch.get("verified_pseudo_label", None)
if verified_label is None and getattr(self, "enable_hybrid", False) and online_consistency_rate < 0.3:
    verified_label = offline_voted_answer # Fallback if verification fails universally

# Compute rewards using the dynamically verified_label 
rewards, ttrl_metrics = test_time_train_metrics(
    group_pred_outputs, group_labels, task=task, 
    extra_info=group_extra_info, verified_label=verified_label
)
```