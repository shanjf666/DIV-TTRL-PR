# 两阶段验证逻辑更新执行计划 v3.1

## 核心需求明确

### 三个关键决策
1. **全样本验证**：所有样本（无论高低一致性）都送进第二阶段验证推理
2. **伪标签选择规则**：
   - 高一致性 (majority_rate ≥ high_consistency_threshold, 默认 0.5)：保持 majority_answer
   - 低一致性 (majority_rate < high_consistency_threshold)：
     - 在满足 true > false 的候选集合中，根据 `low_consistency_strategy` 参数选择：
       - "true": 选择 true_count 最高的候选
       - "majority": 选择在第一阶段出现频率最高的候选
     - 若选不出（无 true > false 的候选）则取决于 fallback_mode
3. **奖励函数**：格式错误 -1，TP/TN +consistency，FN -consistency，FP -0.5×consistency

---

## 修改方案（三大块）

### 修改块 1：两阶段处理参数与设置

#### 参数名称和含义
```python
high_consistency_threshold: float = 0.5  # 区分高低一致性的阈值
low_consistency_strategy: str = "true"  # "true" 或 "majority"
fallback_mode: str = "no_update_second"  # 低一致性选不出伪标签时的处理
    - "no_update_second": 只用 majority_answer，仅参与第一阶段更新，不更新第二阶段
    - "no_update_both": 当前实现下同样跳过该样本的第二阶段更新
```

#### 伪标签类型定义
- 类型A（高一致性）：majority_rate ≥ high_consistency_threshold → `pseudo_label = majority_answer`
- 类型B（低一致性+有效）：majority_rate < high_consistency_threshold 且存在满足 true > false 的候选：
  - 若 strategy == "true": 选 true_count 最大者
  - 若 strategy == "majority": 选 Stage 1 频率最高者
- 类型C（低一致性+无效）：选不出 true > false 的候选 → `pseudo_label = majority_answer`

只有类型A和B的样本参与第二阶段训练；类型C 按 fallback_mode 决定。

---

### 修改块 2：[two_stage_utils.py] 新增/修改函数

#### 函数 A：`select_final_pseudo_labels()`

**输入参数**：
```python
def select_final_pseudo_labels(
    verification_outputs: List[str],            # Stage 2 验证结果
    verification_mapping: List[Dict],           # 样本映射 {prompt_group_idx, candidate_answer, frequency}
    prompt_groups: List[Dict],                  # Stage 1 提取的候选组
    n_votes_per_prompt: int = 8,                # Stage 1 采样次数
    high_consistency_threshold: float = 0.5,    # 高一致性分界
    low_consistency_strategy: str = "true",   # 低一致性筛选策略
    fallback_mode: str = "no_update_second",  # 低一致性无效时的处理方式
) -> Tuple[List[str], List[float], List[bool]]:
```

**处理逻辑**（按 prompt_group 分类处理）：

```python
for each prompt_group i in prompt_groups:
    majority_rate = group.majority_rate
    
    if majority_rate >= high_consistency_threshold:
        # ===== 类型A：高一致性 =====
        # 始终相信 Stage 1 的 Majority Result
        pseudo_label[i] = group.majority_answer
        consistency[i] = majority_rate
        should_update_second[i] = True
        
    else:
        # ===== 类型B/C：低一致性 =====
        # 1. 统计该 group 中候选答案在验证结果中的 true_count / false_count
        true_set_candidates = []  # [(ans, true_count, freq), ...]
        candidate_stats = {}
        for output_text, mapping in zip(verification_outputs, verification_mapping):
            if mapping["prompt_group_idx"] != i:
                continue
            ans = mapping["candidate_answer"]
            if ans not in candidate_stats:
                candidate_stats[ans] = {"true_count": 0, "false_count": 0, "frequency": mapping["frequency"]}
            parsed = parse_verification_result(output_text)
            if parsed is True:
                candidate_stats[ans]["true_count"] += 1
            elif parsed is False:
                candidate_stats[ans]["false_count"] += 1
        for ans, info in candidate_stats.items():
            if info["true_count"] > info["false_count"]:
                true_set_candidates.append((ans, info["true_count"], info["frequency"]))
        
        if true_set_candidates:
            # ===== 类型B：低一致性+有效 =====
            if low_consistency_strategy == "true":
                # 按 true_count 降序，然后按 freq 降序
                true_set_candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
            else: # "majority"
                # 按 freq 降序，然后按 true_count 降序
                true_set_candidates.sort(key=lambda x: (x[2], x[1]), reverse=True)
            
            best_ans, _, best_freq = true_set_candidates[0]
            pseudo_label[i] = best_ans
            consistency[i] = best_freq / n_votes_per_prompt
            should_update_second[i] = True
        else:
            # ===== 类型C：低一致性+无效 =====
            # 选不出符合条件的，退回到 majority_answer
            pseudo_label[i] = group.majority_answer
            consistency[i] = majority_rate
            should_update_second[i] = fallback_mode not in {"no_update_second", "no_update_both"}
```

#### 函数 B：修改 `compute_proxy_cm_reward()`

**改动点**：奖励计算规则（原位置约第537~548行）

```python
# 新奖励公式
if parsed_result is None:
    reward = -1.0  # 格式错误
elif parsed_result is True and is_pl:        # TP
    reward = +1.0 * consistency
elif parsed_result is False and not is_pl:   # TN
    reward = +1.0 * consistency
elif parsed_result is False and is_pl:       # FN（最重的惩罚）
    reward = -1.0 * consistency               # ← 改为 -consistency
elif parsed_result is True and not is_pl:    # FP
    reward = -0.5 * consistency               # ← 保持 -0.5×consistency
```

| 验证结果 | 是/否鞭标 | 名称 | 奖励 |
|---------|---------|------|------|
| 格式错 | - | - | `-1.0` |
| True | 伪标 | TP | `+consistency` |
| True | 非伪 | FP | `-0.5×consistency` |
| False | 伪标 | FN | `-consistency` |
| False | 非伪 | TN | `+consistency` |

---

### 修改块 3：[ray_trainer.py] - `_run_two_stage_verification()` 方法

#### Step 3.1：移除高一致性过滤（第 1048-1051 行）

**原代码**：
```python
groups_to_verify = []
for g in prompt_groups:
    if g.get("majority_rate", 1.0) < 0.3:  # ← 只选低一致性
        groups_to_verify.append(g)
```

**新代码**：
```python
# ===== 全量验证 =====
groups_to_verify = prompt_groups  # 不过滤，所有样本都验证
```

#### Step 3.2：更新伪标签解析（第 1175~1197 行）

**原代码**（合并逻辑）的问题：
- 依赖 `majority_rate < 0.3` 判断是否被验证
- 高一致性跳过，造成索引错位

**新代码**：
```python
# ===== 调用新的伪标签选择函数 =====
verified_labels, verified_consistencies, should_update_flags = select_final_pseudo_labels(
    verification_outputs=all_verification_outputs,
    verification_mapping=verify_mapping,
    prompt_groups=groups_to_verify,
    n_votes_per_prompt=self.n_votes_per_prompt,
    high_consistency_threshold=0.5,
    low_consistency_strategy="true",
    fallback_mode="no_update_second",
)

# ===== 直接赋值（无需合并，因为已全量覆盖）=====
final_pseudo_labels = verified_labels  # [str] × N
final_consistencies = verified_consistencies  # [float] × N
```

#### Step 3.3：根据 should_update_flags 过滤 batch_second

**新增代码**（在原返回前）：

```python
# ===== 按 should_update_flags 过滤 batch_second =====
if batch_second is not None and len(batch_second) > 0:
    keep_indices = []
    for i, m in enumerate(verify_mapping):
        g_idx = m["prompt_group_idx"]
        if should_update_flags[g_idx]:  # only Type A & B samples
            keep_indices.append(i)
    
    if keep_indices:
        # ===== 同步过滤三个数组 =====
        batch_second = batch_second[keep_indices]
        verify_mapping = [verify_mapping[i] for i in keep_indices]
        all_verification_outputs = [all_verification_outputs[i] for i in keep_indices]
        
        # 日志信息
        filtered_count = len(verify_mapping) - len(keep_indices)
        if filtered_count > 0:
            print(f"[TwoStage] Filtered {filtered_count} Type-C samples, keeping {len(keep_indices)} for Stage2")
    else:
        batch_second = None  # ← Type C 样本过多，无Stage2训练
        print(f"[TwoStage] All samples are Type-C (no valid Stage2 candidates), skipping Stage2 training")
```

#### Step 3.4：返回值修改

**原返回**：
```python
return final_pseudo_labels, batch_second, all_verification_outputs, verify_mapping, groups_to_verify, final_consistencies
```

**新返回**：
```python
# 返回值顺序和含义保持一致
return final_pseudo_labels, batch_second, all_verification_outputs, verify_mapping, groups_to_verify, final_consistencies
# 注：should_update_flags 已在过滤中被消费，无需额外返回
# （或可选择返回用于监控/调试）
```

---

## 样本类型与处理流程

### 按样本类型分类

| 类型 | 条件 | 伪标签来源 | Stage1更新 | Stage2参与 | Consistency权重 |
|------|------|----------|----------|----------|---------------|
| **A** | majority_rate ≥ T | majority_answer | ✓ | ✓ | majority_rate |
| **B** | majority_rate < T + 选出有效（T>F） | 该有效伪标签 | ✓ | ✓ | freq/n_votes |
| **C** | majority_rate < T + 无有效伪标签 | majority_answer | ✓ | ✗ | majority_rate |

### 数据流示意

```
Stage1 采样 (N次)
  ↓
extract_candidate_answers() → prompt_groups
[A类样本] [B类样本] [C类样本]  (识别不出来，统一发送验证)
  │
  ↓
全量发送Stage2验证 (所有样本都验证)
  │
  ├─── 类型A：majority_rate ≥ T
  │    └─ 直接使用 majority_answer
  │       consistency = majority_rate → Stage2更新
  │
  ├─── 类型B：majority_rate < T + valid_candidate found
  │    └─ 使用选中的伪标签
  │       consistency = freq/n_votes → Stage2更新
  │
  └─── 类型C：majority_rate < T + no valid_candidate
       └─ 使用 majority_answer
          consistency = majority_rate → Stage2不更新
           
         [过滤 batch_second] ← 移除类型C
           ↓
     compute_proxy_cm_reward (新公式)
           ↓
     compute_grpo_outcome_advantage
           ↓
     update_actor(batch, batch_second)
```

---

## 关键实现细节

### 1. 一致性权重的三种计算

| 情形 | 公式 | 说明 |
|------|------|------|
| 类型A伪标签权重 | `majority_rate` | 该伪标签在Stage1中出现的频率 |
| 类型B伪标签权重 | `freq / n_votes_per_prompt` | 该新伪标签在Stage1中出现的相对频率 |
| 类型C伪标签权重 | `majority_rate` | fallback到majority_rate |

### 2. 索引对应关系保证

- `verify_mapping[i]` → `batch_second[i]` → `all_verification_outputs[i]` → `rewards[i]`
- 过滤时**必须同步更新这四者**，保持长度一致
- Token级别奖励注入：`token_level_rewards[i, valid_resp_len-1] = rewards[i]`

### 3. Type C 样本处理

- **不参与Stage2训练**但仍参与Stage1
- 如果过滤后 `batch_second` 为空，走 `dummy_second` 路径（已有的容错机制）
- 生成日志统计Type C比例用于监控

### 4. 伪标签选择的唯一性

- 对同一个prompt_group中的多个候选，选 `true_count` 最高的
- 若多个候选 `true_count` 相同，则选 `frequency` 最高的（来自Stage1）
- 这确保选择的伪标签一定存在于Stage1的Top 5候选中

---

## 验证清单

### 代码层面检查
- [ ] `select_final_pseudo_labels` 返回三元组长度都等于 `len(prompt_groups)`
- [ ] `should_update_flags` 中 True 的数量 ≤ 原 `verify_mapping` 长度
- [ ] 过滤后 `batch_second`, `verify_mapping`, `all_verification_outputs` 三者长度一致
- [ ] Type C 样本的 `pseudo_label` 正确设为 `majority_answer`
- [ ] Type B 样本的 `consistency = freq / n_votes_per_prompt` 计算正确

### 指标验证（WandB）
- [ ] `train/two_stage_trigger_rate` = 1.0 常数（全量触发）
- [ ] `train/reward_mean` 在 `[-1, +1]` 范围内
- [ ] `train/fn_rate > 0` 和 `train/fp_rate > 0`（两类错误都在统计）
- [ ] `train/second_stage_loss` 应非零（Type A样本参与了Stage2）
- [ ] 新增指标：Type C 比例（可选日志）

### 功能验证
- [ ] Type A样本伪标签不变 vs 原来的 early return 行为
- [ ] Type B样本伪标签改变的样本数（应 > 0 才说明算法生效）
- [ ] Type C比例（应 < 10%，否则阈值可能设错）
- [ ] dummy_second 路径在 `batch_second=None` 时仍可用（边界案例）

---

## 新旧代码对比总结

| 维度 | 旧版本 | 新版本 |
|------|--------|--------|
| **验证范围** | 只验证 majority_rate < 0.3 的样本 | 全量验证所有样本 |
| **伪标签选择** | 低一致性按验证，高一致性直接用majority | Type A/B都参与更新，Type C不参与Stage2 |
| **奖励-FN** | -0.5×C | -1.0×C |
| **奖励-FP** | -1.0×C | -0.5×C |
| **Consistency** | 对低一致性是投票率，对高一致性是majority_rate（不统一） | 对A类是majority_rate，对B类是freq/n_votes，对C类是majority_rate（明确定义） |
| **Stage2过滤** | 无，所有低一致性样本都参与 | 只保留Type A/B，过滤掉Type C |

