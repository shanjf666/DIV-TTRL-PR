# 两阶段逻辑 Debug 修复计划

## Bug 状态总览

| # | Bug 描述 | 状态 |
|---|---------|------|
| 1 | 解包崩溃：`select_final_pseudo_labels` 返回 3 个值，调用方期望 4 个 | ✅ 已修复 |
| 2 | `fallback_mode` 逻辑反转：`no_update_second` 实际上在更新第二阶段 | ✅ 已修复 |
| 3 | `no_update_both` 功能不全：仅屏蔽了 Stage 2，Stage 1 仍使用劣质标签更新 | ❌ **待修复** |

---

## 已修复的 Bug（用户手动修复）

### Bug 1：解包崩溃 ✅
- 新增 `routes` 返回值（`"A"` / `"B1"` / `"B2"`），函数现在返回 4 个值。

### Bug 2：`fallback_mode` 逻辑反转 ✅
- 修改前：`should_update_second[i] = (fallback_mode != "no_update_both")`
- 修改后：`should_update_second[i] = fallback_mode not in {"no_update_second", "no_update_both"}`
- 现在两种模式在 Route B2 下都会正确设置 `should_update_second = False`。

---

## 待修复的 Bug

### Bug 3：`no_update_both` 缺少对第一阶段的屏蔽 ❌

#### 问题描述
当 `fallback_mode = "no_update_both"` 且样本走了 Route B2（低一致性 + 验证器无法选出候选）时：
- **Stage 2**：✅ 已正确跳过（`should_update_second = False` → 从 `batch_second` 中过滤掉）。
- **Stage 1**：❌ **仍然使用回退的 Majority 伪标签参与正常的 PPO 更新**。没有任何屏蔽。

这意味着在 `no_update_both` 模式下，第一阶段仍然在用一个连验证器都不认可的劣质Majority标签训练生成器。

#### 修改方案

**文件**：[ray_trainer.py](file:///d:/学习/科研/DIV-TTRL-PR/verl/verl/trainer/ppo/ray_trainer.py)

**Step 1**：在 `_run_two_stage_verification` 返回之前,构造一个标记哪些 prompt group 需要跳过 Stage 1 的列表：

```python
# 在 return 之前：
skip_stage1_groups = []
fallback_mode = "no_update_second"  # 从配置中读取
if fallback_mode == "no_update_both":
    for i, route in enumerate(verified_routes):
        if route == "B2":
            skip_stage1_groups.append(i)
```

**Step 2**：在 `fit()` 循环中，在 compute_advantage 之后，对被标记的 prompt group 应用 zero advantage mask：

```python
# 在 compute_advantage(...) 之后
if skip_stage1_groups:
    # 每个 prompt group 在 batch 中对应 n_votes_per_prompt 个连续样本
    zero_mask = torch.zeros(len(batch), dtype=torch.float32)
    for group_idx in skip_stage1_groups:
        start = group_idx * self.n_votes_per_prompt
        end = start + self.n_votes_per_prompt
        zero_mask[start:end] = 1.0
    
    zero_mask = zero_mask.to(batch.batch["advantages"].device).unsqueeze(-1)
    batch.batch["advantages"] = batch.batch["advantages"] * (1.0 - zero_mask)
    
    n_zeroed = int(zero_mask.sum().item())
    print(f"[no_update_both] Zeroed Stage1 advantages for {n_zeroed} samples ({len(skip_stage1_groups)} groups)")
    metrics["train/no_update_both_zeroed_ratio"] = float(zero_mask.mean().item())
```

> [!WARNING]
> **注意**：在 `fit()` 循环中，batch 会经过 `seqlen_balancing` 重新排列（第 1547-1550 行）。这会打乱样本的原始顺序。因此，zero mask 需要在重排之前注入到 `batch.non_tensor_batch` 中（类似已有的 `uid` 字段），然后在 `compute_advantage` 之后按 `non_tensor_batch` 中的标记来应用。
>
> 具体来说，应在 `_run_two_stage_verification` 返回后立即将 `skip_stage1_groups` 展开为 per-sample 的 mask 并注入 `batch.non_tensor_batch["zero_advantage_mask"]`。

#### `no_update_second` vs `no_update_both` 行为对比（修复后）

| 模式 | Route B2 时 Stage 1 | Route B2 时 Stage 2 |
|:---|:---:|:---:|
| `no_update_second` | ✅ 正常更新（用 Majority 标签） | ❌ 不更新 |
| `no_update_both` | ❌ 不更新（Advantage = 0） | ❌ 不更新 |

---

## 验证计划

### 功能验证
1. 设置 `fallback_mode = "no_update_both"`，观察日志中 `[no_update_both] Zeroed Stage1 advantages` 的输出。
2. 确认 `train/no_update_both_zeroed_ratio` 指标不为 0。
3. 对比 `no_update_second` 和 `no_update_both` 两种模式下的训练曲线差异。

### 边界情况
- 所有样本都走 Route B2 时，Stage 1 的所有 advantage 都为 0 → 不应崩溃，只是一轮空更新。
- 所有样本都走 Route A 时，`skip_stage1_groups` 为空 → 不触发任何 mask 逻辑。
