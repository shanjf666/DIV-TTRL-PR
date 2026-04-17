# 动态温度采样实现回顾

已成功在 `RayTrainer` 中实现了第二阶段验证的动态采样策略。该功能通过差异化的温度设置，在模型“自信”的情况下增加探索度，从而提供更高质量的梯度信号。

## 主要变更

### 1. 配置增强 [RayTrainer](file:///d:/学习/科研/DIV-TTRL-PR/verl/verl/trainer/ppo/ray_trainer.py)
在 `__init__` 中新增了以下超参数，支持动态控温：
- `two_stage_hc_temperature`: 默认为 `1.0`（高一致性采样温度）。
- `two_stage_lc_temperature`: 默认为 `0.6`（低一致性采样温度）。
- `two_stage_consistency_threshold`: 默认为 `0.5`（分类阈值）。

### 2. 推理逻辑重构
- **抽取辅助方法**：新增 `_run_verification_inference`，将复杂的 micro-batch 调度、生成和解码逻辑封装。
- **分支推理**：
    - 在 `_run_two_stage_verification` 中将全量批次拆分为 **HC 组** 和 **LC 组**。
    - 先后启动两组推理，每组应用其对应的 `meta_info["verification_temperature"]`。
- **索引重映射**：实现了跨 Batch 的全局索引重排 logic，确保合并后的 `batch_second` 与原始 `verification_mapping` 完美对齐。

### 3. 数据完整性保护
- **UID 生成**：保留并适配了 GRPO 所需的 UID 生成逻辑，确保合并后的 batch 能正确分组。
- **鲁棒合并**：增加了对“掉块（Dropped Chunks）”的检测，防止由于显存溢出等异常导致的 DataProto 维度不匹配。

## 监控与统计

已在 WandB 中添加以下指标，用于观察动态采样效果：
- `train/two_stage_hc_verify_count`: 高一致性样本统计。
- `train/two_stage_lc_verify_count`: 低一致性样本统计。
- `train/two_stage_hc_temperature`: 记录 HC 实际温度。
- `train/two_stage_lc_temperature`: 记录 LC 实际温度。

---

## 验证结果

- **逻辑验证**：通过代码审查确认，两次推理产生的 `DataProto` 片段已按照原始全局 `orig_idx` 正确回填。
- **容错性**：代码已处理某一类样本（如 LC 或 HC）数量为 0 的边界情况。

> [!TIP]
> **后续建议**：
> 如果发现 `train/two_stage_parse_fail_rate`（解析失败率）由于 $T=1.0$ 而显著升高，可以考虑将 `two_stage_hc_temperature` 稍微下调至 `0.9` 或尝试减小 `two_stage_top_p`。
