# Second-Stage Speed-Up Implementation Summary

按照 `speed_up.md` 中的两个优化思路已完成实现。本文档说明改动的完整性和核心逻辑。

## 改动概览

### 思路一：引擎原生批量采样 ✅

**目标**：把"固定重复采样"从 Python 层 DataProto 复制移到 rollout 引擎内部。

**涉及文件**：
1. [two_stage_utils.py](verl/verl/utils/reward_score/ttrl/two_stage_utils.py) - 采样策略调整
2. [fsdp_workers.py](verl/verl/workers/fsdp_workers.py) - 引擎侧 n-sampling 支持

**核心改动**：

#### 1.1 `construct_verification_dataproto()` 中的采样逻辑

- **Prior Logic**：在 sampling 模式下，无论 `verification_n` 是多少，都会在 Python 层复制样本
  ```python
  repeat_count = frequency if verification_n is None or verification_n < 0 else verification_n
  ```

- **New Logic**：区分"原生引擎采样"和"频率基动态采样"
  ```python
  if verification_mode == "sampling" and (verification_n is None or verification_n < 0):
      # 动态频率基采样（兼容模式）
      repeat_count = frequency
      native_n_sampling = 1
  else:
      # 原生引擎采样（优化模式）
      repeat_count = 1
      native_n_sampling = verification_n
  ```

- **效果**：
  - 当 `verification_n` 有效时，BatchProto 中每个候选只占一行
  - 引擎一次生成 `verification_n` 个响应
  - 减少 Python 层对象数，降低 host-device 传输成本

#### 1.2 `meta_info` 中的 `verification_n` 传递

```python
meta_info = {
    ...
    "verification_n": max(1, verification_n) if verification_mode == "sampling" and verification_n > 0 else 1,
}
```

- 信号引擎进行原生 n-sampling

#### 1.3 `fsdp_workers.py` 中的 rollout 调用

```python
if verification_mode == "sampling":
    verification_n = prompts.meta_info.get("verification_n", 1)
    verification_kwargs["n"] = verification_n  # 传递给 vLLM
```

- 把 `verification_n` 直接传给 vLLM 的 n-sampling 接口

**Mapping 兼容性**：
- `verification_mapping` 数量 = 候选总数（不再乘以 `verification_n`）
- 每条 mapping 记录 `native_n_sampling` 以标记该候选需要的采样数
- 在 `ray_trainer.py` 中处理响应时，通过 `n_samples` 参数来正确分组

---

### 思路二：Token-aware Micro-batch 分桶 ✅

**目标**：不按"样本数"切分，而按"有效 token 数"切分，减少 padding 浪费。

**涉及文件**：
- [ray_trainer.py](verl/verl/trainer/ppo/ray_trainer.py) - Step 4 的 micro-batch 构造

**核心改动**：

#### 2.1 使用 `rearrange_micro_batches()` 替代固定切片

- **Prior Logic**：固定步长遍历
  ```python
  for chunk_start in range(0, total_verification_size, micro_bs):
      chunk = verification_batch[chunk_start:chunk_end]
  ```

- **New Logic**：Token-aware 平衡分桶
  ```python
  from verl.utils.seqlen_balancing import rearrange_micro_batches
  
  target_chunk_tokens = max_token_len * micro_bs
  micro_batches, micro_bsz_idx = rearrange_micro_batches(
      batch=verification_batch.batch,
      max_token_len=target_chunk_tokens,
      dp_group=None
  )
  ```

- **效果**：
  - 每个 chunk 的 token 总数更均匀
  - 长样本和短样本混跑时，没有一个长样本导致整块 padding
  - GPU 利用率更稳定，不会出现频繁的"锯齿"波动

#### 2.2 处理 Out-of-Order 输出重排

因为 `rearrange_micro_batches` 会改变样本顺序（重排以优化 padding），需要记录重排索引：

```python
outputs_by_orig_idx = {}
reorder_indices = [0] * (total_verification_size * n_samples)

for chunk_idx, (micro_batch_tensors, indices) in enumerate(zip(micro_batches, micro_bsz_idx)):
    # 处理 chunk 并收集响应
    for i, orig_idx in enumerate(indices):
        # 映射回原始位置
        for j in range(n_samples):
            reorder_indices[orig_idx * n_samples + j] = current_unordered_offset + i * n_samples + j
```

#### 2.3 Batch 重排和 UID 生成

```python
if not has_error:
    batch_second = batch_second_unordered[reorder_indices]
    # 现在 batch_second 中的样本顺序与 verification_mapping 一致
    # UID 生成可以直接按 mapping 顺序进行
    for m in verification_mapping:
        uid = f"verify_group_{m['prompt_group_idx']}_{m['candidate_answer']}"
```

**Cache 清理策略调整**：
- 之前：每个 chunk 都执行 `gc.collect()` 和 `torch.cuda.empty_cache()`
- 现在：只在所有 chunk 完成后执行一次
- 原因：频繁清缓存会导致 GPU 调度中断，阻挡连续性

---

## 集成逻辑验证

### 数据流完整性

```
Pass1 Rollout
    ↓
extract_candidate_answers() → prompt_groups
    ↓
construct_verification_dataproto(verification_n=N)
    ├─ Python 层：每个候选 1 行（native n-sampling）
    ├─ mapping：候选总数行
    ├─ meta_info["verification_n"] = N
    └─ verification_batch (size M)
    ↓
ray_trainer.py::_run_two_stage_verification()
    ├─ rearrange_micro_batches() → chunks
    ├─ For each chunk:
    │   ├─ generate_sequences() → M*N 行响应
    │   ├─ decode_verification_outputs() → M*N 个文本
    │   ├─ chunk_decoded[i*N : (i+1)*N] 分配给 outputs_by_orig_idx[orig_idx]
    │   └─ 记录 reorder_indices
    ├─ 重排 batch_second
    └─ 生成对应的 UID 序列
    ↓
select_final_pseudo_labels()
    ├─ 遍历 verification_outputs (M*N 个)
    ├─ 遍历 verification_mapping (M 个，每个对应 N 个响应)
    └─ 聚合统计，输出最终 pseudo_labels (M 个)
```

### 关键匹配点

| 阶段 | 数量应为 | 说明 |
|-----|--------|------|
| prompt_groups | P | P 个不同 prompt |
| candidates (total) | C | P 个 group 中的总候选数 |
| verification_mapping | C | 每个候选 1 个 mapping 条目 |
| verification_batch | C | C 行 |
| all_verification_outputs | C*N | N-sampling 后的总响应数 |
| pseudo_labels | C | 最终标签仍为 C 个 |
| batch_second (after filter) | C*N' | 有效参与 Stage2 的响应 |

其中 N = `two_stage_n`（verification_n）。

---

## 性能预期

### 思路一（原生采样）

- **优势**：
  - 减少 Python 层对象复制（从 C*N 个 sample 减到 C 个）
  - 减少重复 tokenization 和 padding
  - vLLM 可优化多采样推理（batch 内的 beam search 等）

- **预期收益**：
  - Verification generation 的吞吐提升 10~30%（取决于 N 和库实现）
  - 峰值显存降低（样本复制成本消除）

### 思路二（Token-aware 分桶）

- **优势**：
  - 减少 padding 浪费
  - 更均匀的 chunk 大小 → 更稳定的 GPU 利用率

- **预期收益**：
  - 显存利用率更稳定（减少 "锯齿" 波动）
  - 可能允许更大的 chunk 大小（相同显存下）

---

## 向后兼容性

### 保持兼容的地方

1. **接口不变**：`two_stage_verify`, `two_stage_mode`, `two_stage_n` 等外部参数含义不变
2. **Fallback 模式**：设置 `verification_n=None` 或 `< 0` 时自动回退到频率基采样（旧逻辑）
3. **Pseudo-label 结果**：最终 `pseudo_labels` 数量和内容逻辑不变，仅计算路径优化

### 需要验证的地方

1. **vLLM n-sampling 实现**：确保 `n > 1` 时能正确返回 batch_size*n 个响应
2. **Reorder 索引正确性**：在 out-of-order 场景下，验证 batch_second 和 mapping 仍然对齐
3. **映射完整性**：原始顺序的 pseudo_labels 和 final_consistencies 仍与 prompt_groups 对齐

---

## 下一步验证步骤

1. **单元测试**：
   - 小 batch 对比新旧 verification_outputs 的一致性
   - 检验 batch_second 中样本和 mapping 的对齐

2. **集成测试**：
   - 完整 training loop 运行，观察 two_stage metrics
   - 确认 Loss 曲线的稳定性

3. **性能基准**：
   - 对比 verification generation 的耗时（应下降 10~30%）
   - 对比 GPU Memory 波动（应变平稳）
   - 对比 End-to-end training 时间

---

## 故障排查清单

| 问题 | 诊断 | 解决 |
|-----|-----|------|
| ValueError: len(outputs) ≠ len(mapping) | mapping 生成错误 | 检查 construct_verification_dataproto 中 repeat_count 逻辑 |
| batch_second UID 不对齐 | 重排索引错误 | 打印 outputs_by_orig_idx 和 reorder_indices |
| vLLM 返回数量与预期不符 | n-sampling 实现问题 | 确认 vLLM 版本是否支持 n > 1 |
| OOM 反而增加 | token-aware 分桶失效 | 检查 rearrange_micro_batches 的返回格式 |

