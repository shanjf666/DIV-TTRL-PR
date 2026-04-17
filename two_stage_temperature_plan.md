# 动态温度采样实现计划 (Stage 2 Verification)

## 目标
根据第一阶段输出的一致性动态调整第二阶段验证的采样温度：
- **高一致性样本 (`majority_rate >= 0.5`)**：使用 $T=1.0$ 进行验证采样，增加探索度，挖掘更有效的验证梯度。
- **低一致性样本 (`majority_rate < 0.5`)**：使用 $T=0.6$ 进行验证采样，保持判断稳健性。

## 原计划问题检查

### 问题 1：DataProto 拆分的复杂性被低估

> [!WARNING]
> 原计划说"根据 `majority_rate` 对 `verification_batch` (DataProto) 进行索引过滤，生成两个子 DataProto"。
> 但实际代码中，`verification_batch` 的行是按 **候选答案（candidate）** 粒度排列的，而 `majority_rate` 是按 **Prompt Group** 粒度存在的。
> 一个 Prompt Group 可能包含多个 candidates，它们在 `verification_batch` 中对应多行。
> 因此，拆分不是简单的"按行过滤"，而需要通过 `verification_mapping` 反查每行属于哪个 Prompt Group，再读取该 Group 的 `majority_rate`。

### 问题 2：温度是通过 `meta_info` 传递的，不是 per-sample 的

> [!IMPORTANT]
> 当前的温度传递路径是：
> 1. `ray_trainer.py` 将温度写入 `verification_batch.meta_info["verification_temperature"]`
> 2. `fsdp_workers.py` 从 `prompts.meta_info` 中读取，传给 `update_sampling_params`
> 3. `vllm_rollout.py` 的 `update_sampling_params` 用它修改全局 `SamplingParams`
>
> **这意味着温度是 Batch 级别的，不是 Sample 级别的。** 要实现不同温度，**必须**拆成两个独立的 Batch 分别调用 `generate_sequences`。原计划的方向是正确的，但需要明确这一点。

### 问题 3：拆分后 `n_samples` 和 `reorder_indices` 的对齐

> [!WARNING]
> 当前的推理循环中使用了复杂的 `outputs_by_orig_idx` 和 `reorder_indices` 机制来处理 micro-batch 乱序后的重排。
> 如果我们拆成 HC 和 LC 两个子批次分别推理，每个子批次内部的 `orig_idx` 都是**相对于自身子批次**的（从 0 开始），而不是相对于原始 `verification_batch` 的。
> 合并时必须将子批次的局部索引映射回全局索引。

### 问题 4：`batch_second` 的合并顺序

> [!IMPORTANT]
> `batch_second` 是用于 Stage 2 训练的 DataProto。拆分推理后，HC 和 LC 的 `chunk_output` 需要按**原始全局顺序**合并成一个完整的 `batch_second`，否则后续的 `verification_mapping` 索引、`should_update_flags` 过滤、以及 `two_stage_high_consistency_mask` 注入都会错位。

### 问题 5：边界情况——某一类为空

> [!NOTE]
> 如果某个 Step 中所有 Prompt Groups 都是高一致性（或都是低一致性），那么另一个子批次为空。需要确保空子批次不会导致空 `DataProto.concat([])` 报错。

---

## 修正后的实施方案

### 总体策略

**不修改** `construct_verification_dataproto` 和底层 rollout worker，只在 `ray_trainer.py` 的 `_run_two_stage_verification` 方法中进行修改。核心思路：

1. 在 `construct_verification_dataproto` 返回后，利用 `verification_mapping` 中的 `prompt_group_idx` 反查 `groups_to_verify` 的 `majority_rate`，为每一行打上 HC/LC 标签。
2. 将 `verification_batch` 按标签拆成两个子 DataProto（`hc_batch` 和 `lc_batch`），并记录每行的全局原始索引。
3. 对 `hc_batch` 设置 `meta_info["verification_temperature"] = hc_temp`，对 `lc_batch` 设置 `meta_info["verification_temperature"] = lc_temp`。
4. 分别对两个子批次执行现有的 micro-batched 推理循环。
5. 将两次推理的输出按全局索引合并回正确顺序。

### 新增超参数

在 `ray_trainer.py` 的 `__init__` 中新增（可通过 Hydra 配置）：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `two_stage_hc_temperature` | `1.0` | 高一致性样本的验证温度 |
| `two_stage_lc_temperature` | `0.6` | 低一致性样本的验证温度 |
| `two_stage_consistency_threshold` | `0.5` | 高/低一致性的分界阈值 |

---

### 具体修改

#### [MODIFY] [ray_trainer.py](file:///d:/学习/科研/DIV-TTRL-PR/verl/verl/trainer/ppo/ray_trainer.py)

**A. `__init__` 方法 (约 L410-L423)：** 新增 3 个配置项读取。

```python
self.two_stage_hc_temperature = getattr(self.config, 'two_stage_hc_temperature', 1.0)
self.two_stage_lc_temperature = getattr(self.config, 'two_stage_lc_temperature', 0.6)
self.two_stage_consistency_threshold = getattr(self.config, 'two_stage_consistency_threshold', 0.5)
```

**B. `_run_two_stage_verification` 方法 (约 L1074-L1207)：** 重构推理循环。

将现有的"Step 3 → Step 4"替换为以下逻辑：

```
Step 3: 按一致性拆分
  ├── 遍历 verification_mapping，通过 prompt_group_idx 查 majority_rate
  ├── 生成 hc_indices（全局行索引列表）和 lc_indices
  ├── hc_batch = verification_batch[hc_indices]  （DataProto 支持索引切片）
  └── lc_batch = verification_batch[lc_indices]

Step 4a: HC 子批次推理 (T=hc_temp)
  ├── 设置 hc_batch.meta_info["verification_temperature"] = hc_temp
  ├── 复用现有的 micro-batch + generate_sequences 循环
  └── 收集 hc_outputs_by_local_idx

Step 4b: LC 子批次推理 (T=lc_temp)
  ├── 设置 lc_batch.meta_info["verification_temperature"] = lc_temp
  ├── 复用相同的推理循环
  └── 收集 lc_outputs_by_local_idx

Step 5: 合并
  ├── 建立 global_outputs_by_orig_idx：
  │   hc_indices[local_i] → hc_outputs_by_local_idx[local_i]
  │   lc_indices[local_i] → lc_outputs_by_local_idx[local_i]
  ├── 按 orig_idx 从 0 到 N-1 顺序拼出 all_verification_outputs
  └── 按 orig_idx 顺序拼出 batch_second（DataProto.concat）
```

**C. 推理循环提取为辅助函数：** 将 L1097-L1198 的 micro-batch 推理循环提取为一个私有方法，例如：

```python
def _run_verification_inference(self, sub_batch, verification_mapping_subset, n_samples):
    """Run micro-batched verification inference on a sub-batch.
    
    Returns:
        outputs_by_local_idx: dict mapping local row index -> list of decoded strings
        chunk_outputs: list of DataProto chunks (on CPU)
    """
    ...
```

这样 HC 和 LC 两次推理只需各调用一次该函数，避免代码重复。

**D. `batch_second` 合并逻辑：** 需要将两次推理的 `chunk_outputs` 重排为全局顺序后再 concat：

```python
# 将 hc 和 lc 的 chunk_outputs 按全局索引交织合并
global_reorder = [0] * total_verification_size * n_samples
# ... 填充 global_reorder，使得 hc_chunk_outputs 和 lc_chunk_outputs
#     中的每一行映射到最终 batch_second 中的正确位置
batch_second = DataProto.concat(all_chunks)[global_reorder]
```

---

### 不需要修改的文件

| 文件 | 原因 |
|---|---|
| `two_stage_utils.py` | `verification_mapping` 已包含 `prompt_group_idx`，可以反查 `majority_rate`。无需额外字段。 |
| `fsdp_workers.py` | 已正确从 `meta_info` 读取温度，无需改动。 |
| `vllm_rollout.py` | 温度通过 `update_sampling_params` 上下文管理器设定，无需改动。 |
| `dp_actor.py` | 训练侧逻辑不受影响。 |

---

## 新增 Metric

| 指标名称 | 含义 |
|---|---|
| `train/two_stage_hc_verify_count` | 高一致性子批次的验证样本数 |
| `train/two_stage_lc_verify_count` | 低一致性子批次的验证样本数 |
| `train/two_stage_hc_temperature` | 本轮使用的 HC 温度 |
| `train/two_stage_lc_temperature` | 本轮使用的 LC 温度 |

---

## 验证计划

### 自动化验证
1. **维度校验**：合并后 `len(all_verification_outputs)` 必须等于 `len(verification_mapping)`。增加 assert 语句。
2. **日志校验**：打印 HC/LC 拆分数量，确认阈值生效。
3. **空批次测试**：若某一侧为空（全部高一致或全部低一致），确保不会抛异常。

### 手动验证
1. 在 WandB 中观察 `train/two_stage_hc_verify_count` 和 `lc_verify_count` 的比例变化。
2. 对比调整前后的 `actor/high_consistency_stage2_pg_loss_token_mean` 是否有显著提升（预期 HC 组由于更高温度会产生更多样化的验证，从而带来更大的梯度）。

## Open Questions

> [!IMPORTANT]
> **温度数值确认**：当前计划使用的是 HC $T=1.0$、LC $T=0.6$。请确认：
> 1. $T=1.0$ 对于验证任务是否过高？验证 prompt 通常期望较精确的"True/False"输出，$T=1.0$ 可能导致 parse_fail_rate 升高。
> 2. 是否需要在首轮实验后根据 `two_stage_parse_fail_rate` 的变化来动态微调？

> [!NOTE]
> **`top_p` 是否也需要差异化？** 当前 HC 和 LC 使用相同的 `top_p`。如果 HC 温度调到 1.0，可能需要搭配一个略低的 `top_p`（如 0.9）来防止输出过于发散。
