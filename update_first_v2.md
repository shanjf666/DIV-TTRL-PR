# Two-Stage Verification v2 修改文档

本文档总结了根据 `update_first_v2_plan.md` 计划对 TTRL 两阶段验证框架执行的 V2 版本代码修改。此次修改主要将验证参数缩紧以提高稳定性，并将采样次数和伪标签选择机制改为“动态绝对多数”策略。

---

## 核心变更内容

### 1. 动态采样次数支持 (`two_stage_utils.py`)
在 `construct_verification_dataproto` 函数中，我们对抽样逻辑进行了如下重构：
- 当识别到 `verification_n` 参数为 `None` (或 `< 0`) 时，自动启用动态采样。
- **机制**：当前候选答案在验证阶段的采样次数（`repeat_count`）将直接等于其在第一阶段 Rollouts 中的出现频次（`frequency`）。这使得模型高频率生成的先验答案能在第二阶段获得更高权重的算力去验证。

### 2. 测试集最优选取策略 (`two_stage_utils.py`)
在 `resolve_filtered_pseudo_labels` 函数中，我们彻底改变了对最终伪标签的裁定方式：
- **旧逻辑**：只要 `True_votes > False_votes`，所有的候选都被当做合法答案（`valid_candidates`），然后再根据原始阶段的频率选取一个作为最终答案。
- **新逻辑**：计算出所有答案的绝对 `True` 指针票数 (`true_count`)。不看比例，直接**选取获得 `true_count` 最高**的答案。
- **平局处理**：若有多个答案的 `true_count` 相同，则依照原始阶段的 `frequency` 降序打破平局。
- **降级回退 (Fallback)**：如果所有候选答案的 `true_count` 都为 0，程序将稳定回退至第一阶段的最原始最高频答案（Majority）。

### 3. 温度与采样限制传递 (`ray_trainer.py` & `fsdp_workers.py`)
为了进一步规范 V2 中的验证回复以削减 ParseFail，启用了低温稳态推理：
- **参数写入 (`ray_trainer.py`)**：在 `verification_batch.meta_info` 中下发了：
  - `verification_temperature = 0.2`
  - `verification_top_p = 0.85`
- **默认值提升**：将 `two_stage_max_new_tokens` 的代码默认值从 `512` 提升至 **`2048`**，确保模型有足够的字数空隙进行复杂数学推导。
- **参数承接 (`fsdp_workers.py`)**：修复了原系统中只消费 `temperature` 的问题。现在 `update_sampling_params` 会同时捕获并把 `verification_top_p` 精准下发给 vLLM 推理引擎。

### 4. 灵活参数化配置 (`update_first.sh`)
系统不再强行硬编码动态逻辑，而是通过配置触发：
- **触发动态验证**：在脚本中显式设置 **`+two_stage_n=-1`**。当检测到负值时，`ray_trainer.py` 将指示后端启用“频次 = 采样数”逻辑。
- **候选答案数量可调**：增加了 **`+two_stage_max_candidates`** 参数。
    - 设置为 **正整数（如 5）**：仅验证第一阶段出现频率最高的前 K 个答案。
    - 设置为 **`-1` 或 `0`**：验证第一阶段产生的**全部**不同回答（不设上限）。
- **并发估算逻辑**：当 `+two_stage_micro_batch_size=0` 时，系统将直接采用与验证前训练阶段一致的 **`train_batch_size`** 作为推理并发量。这能更充分地利用 GPU 算力，同时保证显存占用在安全范围内。

---

## 预期效果与后续建议

- **更严格的推理**：采样参数 (`temp=0.2, top_p=0.85`) 会促使验证文字输出更加模板化，`ParseFail` 数据指标有望进一步降低。
- **算力自适应**：算力将向具有高先验频率的答案倾斜，不再是无条件地将次优候选人同等对待，这有助于优化 GPU 吞吐。
- **调试建议**：如果显存（VRAM）充足，建议尝试手动调大 `micro_batch_size`（如等于 `train_batch_size`）以加快跑测试的速度。
