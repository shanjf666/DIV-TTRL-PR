# Two-Stage Verification Implementation Summary

## 修改内容

### 1. `verl/verl/trainer/ppo/ray_trainer.py`
- 将 `groups_to_verify` 改为 `prompt_groups`，实现 **全样本第二阶段验证**。
- 新增 `select_final_pseudo_labels()` 调用，替换原来的 `resolve_filtered_pseudo_labels()`。
- 加入 `should_update_second` 过滤逻辑：只有类型 A/B 样本保留到 `batch_second`，类型 C 样本会跳过第二阶段训练。
- 保留 `final_pseudo_labels` 对所有 prompt groups 的覆盖。
- 计算 `train/two_stage_trigger_rate` 为 `1.0`，反映全量验证。

### 2. `verl/verl/utils/reward_score/ttrl/two_stage_utils.py`
- 新增 `select_final_pseudo_labels()`：
  - 高一致性样本直接使用 `majority_answer`。
  - 低一致性样本根据 `true_count > false_count` 集合选择伪标签。
  - 支持 `low_consistency_strategy="true"` 和 `low_consistency_strategy="majority"`。
  - `fallback_mode="no_update_second"` 和 `fallback_mode="no_update_both"` 控制失败样本是否参与第二阶段。
- 修正 `compute_proxy_cm_reward()`：
  - TP/TN: `+consistency`
  - FN: `-consistency`
  - FP: `-0.5 * consistency`
  - 格式错误：`-1.0`

## 实现效果
- 所有样本都参与第二阶段验证推理。
- 低一致性样本只有在验证结果支持 `true > false` 时才会进入第二阶段训练；否则可选择跳过 Stage2。
- 奖励规则已按需求修正，FN 惩罚大于 FP。

## 验证情况
- 对修改后的文件执行了语法检查：`py_compile` 通过，无语法错误。

## 变更文件
- `verl/verl/trainer/ppo/ray_trainer.py`
- `verl/verl/utils/reward_score/ttrl/two_stage_utils.py`

## 建议测试
- 运行一个小规模两阶段训练，观察：
  - `train/two_stage_trigger_rate == 1.0`
  - `train/reward_mean` 在合理范围内
  - `train/fn_rate` 与 `train/fp_rate` 都有非零值
  - `batch_second` 在 Type C 样本情况下正确置为 `None`

## Two-stage 相关超参数说明与建议
- `two_stage_n`:
  - Stage1 验证采样次数，默认 `4`。
  - 建议先用 `4` 或 `6`，保证每个 prompt group 有足够候选生成结果。
- `two_stage_max_candidates`:
  - Stage1 验证时最多保留的候选答案数，默认 `10`。
  - 建议保持 `10`，避免过多候选导致验证计算量暴增。
- `two_stage_max_new_tokens`:
  - 第二阶段验证最多生成 token 数，默认 `2048`。
  - 对长问题/答案场景可保留 `2048`，对短问题可适当降低到 `512` 或 `1024`。
- `two_stage_micro_batch_size`:
  - Stage2 推理的微批次大小，默认 `0`（自动计算）。
  - 如果显存紧张可手动设置为 `1` 或 `2`；如果 GPU 充足，可保持 `0`。
- `two_stage_temperature`:
  - Stage2 验证生成温度，默认 `0.2`。
  - 建议使用较低温度以提高验证稳定性，`0.1-0.3` 为合理范围。
- `two_stage_top_p`:
  - Stage2 验证采样 top-p，默认 `0.85`。
  - 建议保留 `0.85` 或略调到 `0.8`，增强验证结果一致性。
- `n_votes_per_prompt` / `n_samples_per_prompt`:
  - `n_votes_per_prompt` 控制 Stage1 的候选答案数量，`n_samples_per_prompt` 控制最终保留样本数。
  - 例如 `n_votes_per_prompt=8`、`n_samples_per_prompt=4` 是常见组合；`n_votes_per_prompt` 应大于等于 `n_samples_per_prompt`。
- `algorithm.lambda_second`:
  - Stage2 advantage 缩放系数，默认 `0.5`。
  - 建议从 `0.5` 开始调节，若希望更强的 Stage2 学习信号可提高到 `0.8-1.0`。

## 新增参数的意义与推荐配置
- `high_consistency_threshold`:
  - 默认 `0.5`。
  - 该阈值用于判断是否直接使用 Stage1 的 `majority_answer`。
  - 推荐保持 `0.5`：只有大于半数候选一致时才认为高一致性，避免过早决策。
  - 如果想更保守，可提高到 `0.6` 或 `0.7`；如果希望更多样本直接进入高一致性分支，可降低到 `0.4`。
- `low_consistency_strategy`:
  - 默认 `"true"`。
  - `"true"`：优先选择 `true_count > false_count` 且 `true_count` 最大的候选答案。
  - `"majority"`：在满足 `true_count > false_count` 的候选中优先选择原始频率最大的答案。
  - 推荐 `"true"`，因为该策略更注重验证结果对候选的明确支持；`"majority"` 可用在验证结果本身不够稳定时。
- `fallback_mode`:
  - 默认 `"no_update_second"`。
  - `"no_update_second"`：当低一致性组没有候选满足 `true_count > false_count` 时，保留当前伪标签但不让该组进入 Stage2 训练。
  - `"no_update_both"`：当低一致性失败时，避免该组进入 Stage2 训练，同时不将该组作为“可靠”二阶段更新样本。
  - 推荐先使用默认 `"no_update_second"`，这既保留伪标签决策，又避免不稳定样本伤害 Stage2 学习。

## 运行建议
- 先做小规模验证：`n_votes_per_prompt` 保持 `8` 或 `16`，`two_stage_n=4`。
- 观察 `train/two_stage_trigger_rate`、`train/fn_rate`、`train/fp_rate` 和 `train/reward_mean`。
- 若第二阶段训练效果不稳，可先提高 `high_consistency_threshold` 或改用 `low_consistency_strategy="majority"`。
- 若希望更强的 Stage2 监督信号，可逐步增加 `algorithm.lambda_second`。