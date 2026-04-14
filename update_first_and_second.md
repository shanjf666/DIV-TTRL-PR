# 两阶段联合更新功能变更日志 (update_first_and_second)

本文档记录了为 `DIV-TTRL-PR` 增加第二阶段联合训练与混淆矩阵奖励的全部修改内容。这一更新实现了 Generator 与 Verifier 的双重自强化学习更新。架构采用双 DataProto 分离传递机制，从而有效避免了由于拼接引起的一系列问题（如 batch size 未对齐、不同序列长度 pad 以及梯度的混入等）。

---

## 1. 核心流程修改 (`verl/trainer/ppo/ray_trainer.py`)

### 1.1 获取验证 DataProto
在 `_run_two_stage_verification` 中，原先解码文本后便将 `chunk_output` 回收（释放显存）。现在，在回收前将每次预测结果保存回 CPU，用于构建第二阶段的整体 `DataProto`。

- 新增：收集每个验证小批次的张量结果：`all_chunk_outputs.append(chunk_output.to("cpu"))`
- 新增：将其拼成第二阶段的训练 Batch `batch_second = DataProto.concat(all_chunk_outputs)`。
- **分组标识注入**：为了确保随后的 GRPO 归一化能让同个 `(prompt, candidate)` 的验证结果成为一组，给 `batch_second` 内元素分配 `uid` 字段（例如 `verify_group_${prompt_group_idx}_${candidate_answer}`）。
- 返回值增加：将 `batch_second`、验证结果原始文本串 (`verify_outputs`)、验证映射字典 (`verify_mapping`) 及触发验证的问题组 (`groups_to_verify`) 全局带出。

### 1.2 主训练循环中注入两阶段强化学习 (`fit` 函数)
在计算第一阶段 `data` 时获取完第一阶段生成的旧概率（`old_log_probs`）之后，执行了第二阶段数据的预处理与 Advantage 计算流程。

- **奖励计算**：调用了混淆矩阵计算工具 `compute_proxy_cm_reward` 计算得到奖励值 `rewards` 及指标信息。
- **Token Level 奖励注入**：寻找 `batch_second` 中每一个回答的结尾 token（对应 `response_mask` 最后为 1 的位置），并将标量 reward 填入张量对应的该位置 `token_level_rewards[i, valid_resp_len - 1] = rewards[i]`。
- **旧概率与 Advantage 计算**：调用模型侧 `actor_rollout_wg.compute_log_prob(batch_second)` 并将得到的结果使用 GRPO advantage 对 `batch_second` 组内求归一化结果，得到了 `advantages`。
- **加权机制**：对得到的 `advantages` 增加 $\alpha = \lambda_{second}$（默认 0.5）作直接衰减缩放。
- 将 `batch_second` 对象加入 `update_actor` 入参。如果不存在第二阶段数据（验证没被触发），使用第一阶段的一个 batch 作为占位 `dummy_second` 进行传入计算。

## 2. 工具和奖励函数 (`verl/utils/reward_score/ttrl/two_stage_utils.py`)

### 2.1 伪标签的验证一致性分析
- 在 `resolve_filtered_pseudo_labels` 中，计算伪标签被采纳时它的出现概率/一致率 `consistency`。并将原先只返回单一预测列表改为了返回 `(pseudo_labels, consistencies)` 元组结构。

### 2.2 定义 Proxy CM
- 新增 `compute_proxy_cm_reward` 函数，用来定义真正的混淆矩阵强化学习 Reward：
    - `reward = +1 * consistency` 当输出验证标签时且选对了（TP/TN）。
    - `reward = -1 * consistency` 当为选错且标签被过度信任（FP）。
    - `reward = -0.5 * consistency` 当选错且本应对的标签（FN）。
    - 缺失 `<reverse_verification>` 结构时提供扣除 `-1.0` 的重罚项，同时将相关的 `tp_rate` / `tn_rate` / `format_error_rate` 加以分类统计记录并整合进 Metric。

## 3. 并行框架调整 (`verl/workers/fsdp_workers.py`)

- 修改 `update_actor(self, data: DataProto)` 的签名变为 `update_actor(self, data: DataProto, data_second: DataProto = None)` 以支持额外的 DataProto 导入。包含以下逻辑：
  - 加载 `has_second_stage = data_second.meta_info.get("has_second_stage", False)` 以判断占位 DataProto 是否真实存在第二阶段数据，如无则置空。
  - 在分布式通信层面为 `data_second` 也加入 `ulysses_sharding_manager` 通信管理器进行相应的 Tensor 掩码操作，最后转入底层模型 `update_policy` 进行更新和梯度下降。

## 4. 底层模型计算逻辑 (`verl/workers/actor/dp_actor.py`)

### 4.1 分组策略与内存管理
在 `update_policy` 处理中接收 `data_second` 后，增加了一套独立的逻辑在循环中只为第二阶段的 loss 做贡献。
- 首先将 `data_second` 的全样本按 `uid` `(prompt, candidate)` 重组出 `verification_groups` 组。
- 将这些组再分配到第一阶段 `num_mini_batches` 对应等比例的 `verification_mini_batches` 当中。确保这几组分别通过微批次 (micro batching) 计算减少显存开销。

### 4.2 计算与累计 Loss
- 每个 epoch 的正常的第一阶段 `mini_batch` 处理并触发 `loss.backward()` （Loss 会在单步自动累计）后。
- 执行 `Stage 2 Backward Pass`，调用相同的模型前向计算提取 `pg_loss_s`。
- 将返回结果除以验证组群与组群内部采样样本乘积 $ \text{loss} = \frac{\text{pg\_loss}}{\text{num\_groups} \times \text{num\_mbs\_per\_group}} $ 后执行后向平摊 `loss_s.backward()`（注意由于在上一层已做 Alpha $\lambda_{second}$ 衰减处理，在梯度下降阶段该模型可直接相加梯度执行步进计算）。
- 当当前微批次中全部前反向传播累积结束，执行总体 `self.actor_optimizer.step()`。

## 5. 配置参数设置 (`verl/examples/labelfree/update_first.sh`)

修改执行脚本对应第二阶段策略配置，包含：
- 修改每次请求执行从最高票中选定的 candidate 验证数配置：`+two_stage_n=8`（原 -1）。
- 修改需要验证的 candidate 回答样本候选数配置：`+two_stage_max_candidates=5`（原 -1）。
- 新增增加学习率系数缩放：`+algorithm.lambda_second=0.5`。

## 6. 已知问题与修复记录

在实现过程中发现并修复了以下两个关键 Bug，以确保在大规模分布式训练下系统的稳定性：

### 6.1 Numpy 命名空间错误 (NameError)
- **现象**：在 `dp_actor.py` 中执行 UID 分组逻辑时报错 `name 'np' is not defined`。
- **修复**：在 `verl/workers/actor/dp_actor.py` 顶层添加了 `import numpy as np`。

### 6.2 训练关键词缺失 (KeyError: 'ref_log_prob')
- **现象**：当第一阶段开启 KL 散度约束时，底层 `DataProto.select` 会强制要求所有数据批次（包含第二阶段数据）都必须拥有 `ref_log_prob` 字段。由于 Verifier 的 Rollouts 并没有计算该字段，导致训练中断。
- **修复**：在 `dp_actor.py` 的 `update_policy` 函数中引入了**动态键值过滤机制**。它会自动检测当前 `DataProto` 实际拥有的 Key，确保护理第二阶段数据时不再强行索取不存在的 KL 相关字段，从而支持 Generator 有 KL 约束而 Verifier 无 KL 约束的异构训练模式。

