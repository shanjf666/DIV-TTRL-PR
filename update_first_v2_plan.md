# Implementation Plan - Two-Stage Verification v2

该计划旨在优化 TTRL 两阶段验证的采样精度与伪标签选取逻辑。

---

## 核心设计变更

> [!IMPORTANT]
> **v2 三大变更点：**
> 1. **动态采样次数**：每个候选答案在第二阶段的验证次数 = 它在第一阶段出现的频次（`frequency`）。
> 2. **采样参数调整**：验证阶段使用 `temperature=0.2`、`top_p=0.85`。
> 3. **选取逻辑重构**：最终伪标签由验证结果中 **`True` 计数最高** 的候选答案决定（平局时选频次最高者）。

---

## 拟议变更

### 1. `two_stage_utils.py` — 核心采样与选取逻辑

#### [MODIFY] [two_stage_utils.py](file:///d:/学习/科研/DIV-TTRL-PR/verl/verl/utils/reward_score/ttrl/two_stage_utils.py)

##### (a) `construct_verification_dataproto` — 动态采样次数

**当前逻辑 (L249-L258)**：
```python
repeat_count = verification_n if verification_mode == "sampling" else 1
for _ in range(repeat_count):
    ...  # 每个候选答案重复固定 N 次
```

**修改为**：当 `verification_n` 传入 `None`（或 `-1`）时，使用 `frequency` 作为该候选的采样次数：
```python
if verification_n is None or verification_n < 0:
    repeat_count = frequency  # 动态：用第一轮的出现频次
else:
    repeat_count = verification_n if verification_mode == "sampling" else 1
```

##### (b) `resolve_filtered_pseudo_labels` — 按 True 总数选取

**当前逻辑 (L411-L422)**：
```python
# 先筛选出 true_count > false_count 的候选
# 然后在 valid 中按 frequency 排序选最高频
```

**修改为**：
```python
# 所有候选按 true_count 降序排列，平局时按 frequency 降序
all_candidates = [(ans, info["true_count"], info["frequency"]) for ans, info in candidate_scores.items()]
all_candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)

if all_candidates and all_candidates[0][1] > 0:
    # 选 true_count 最高的
    pseudo_labels[i] = all_candidates[0][0]
else:
    # 全部 true_count == 0，Fallback 到第一阶段 Majority
    pseudo_labels[i] = original_majority_ans
```

---

### 2. `ray_trainer.py` — 温度与 top_p 传递

#### [MODIFY] [ray_trainer.py](file:///d:/学习/科研/DIV-TTRL-PR/verl/verl/trainer/ppo/ray_trainer.py)

##### (a) 修改采样参数 (L1082-L1084)

**当前**：
```python
verification_batch.meta_info["verification_temperature"] = 0.6
```

**修改为**：
```python
verification_batch.meta_info["verification_temperature"] = 0.2
verification_batch.meta_info["verification_top_p"] = 0.85
```

##### (b) 传入动态采样标记 (L1062)

**当前**：
```python
verification_n=self.two_stage_n if self.two_stage_mode == "sampling" else 1,
```

**修改为**：
```python
verification_n=None,  # v2: 使用动态频率作为采样次数
```

---

### 3. `fsdp_workers.py` — 底层 top_p 传递

#### [MODIFY] [fsdp_workers.py](file:///d:/学习/科研/DIV-TTRL-PR/verl/verl/workers/fsdp_workers.py)

**当前 (L584-L586)**：仅传递 temperature。

**修改为**：同时传递 `top_p`：
```python
verification_temp = prompts.meta_info.get("verification_temperature", 0.6)
verification_top_p = prompts.meta_info.get("verification_top_p", 1.0)
verification_kwargs["temperature"] = verification_temp
verification_kwargs["top_p"] = verification_top_p
```

> [!WARNING]
> **这是原计划遗漏的关键步骤**。如果不修改 `fsdp_workers.py`，`top_p=0.85` 将不会生效，
> vLLM 会继续使用其默认的 `top_p=1.0`。

---

### 4. `update_first.sh` — 清理过时参数

#### [MODIFY] [update_first.sh](file:///d:/学习/科研/DIV-TTRL-PR/verl/examples/labelfree/update_first.sh)

- 删除 `+two_stage_n=8`（因为现在是动态采样，该参数不再生效）。
- 删除 `+two_stage_mode=sampling`（动态模式下始终为 sampling）。

---

## 原计划的问题总结

| # | 问题 | 影响 | 修正方案 |
|---|------|------|----------|
| 1 | **`top_p` 传递链断裂**：原计划只在 `ray_trainer.py` 设置了 `top_p`，但没有修改 `fsdp_workers.py` 读取并下发给 vLLM | `top_p=0.85` 实际不会生效，vLLM 仍用默认 `top_p=1.0` | 在 `fsdp_workers.py` 中增加 `verification_top_p` 的读取和传递 |
| 2 | **`verification_n=None` 的传递遗漏**：原计划没说明 `ray_trainer.py` 应如何修改调用 `construct_verification_dataproto` 的参数 | 仍会使用旧的固定 `two_stage_n` | 在 `ray_trainer.py` 中将 `verification_n` 改为 `None` |
| 3 | **脚本清理**：原计划没提出应删除 `+two_stage_n` 和 `+two_stage_mode` 这两个现在多余的启动参数 | 不影响正确性，但会造成混淆 | 从 `update_first.sh` 中删除 |

---

## 平局处理（已确认）

> 多个候选答案的 `True` 总数相同时，选择在第一阶段出现频次更高的答案。

---

## 验证计划

### 自动化测试
- 运行 `update_first.sh`，通过终端日志观察：
    - `[TwoStage] Verification batch size` 是否随候选人频次动态变化（不再是固定的 `candidates * 8`）。
    - `[TwoStage] Results` 中 `ParseFail` 在更低温度下是否有所降低。
    - `Accuracy Comparison` 指标检查 v2 策略是否提升了标签准确率。
