# DIV-TTRL-PR 第二阶段共同进化与联合更新计划

本计划参考 `self-harmony` 的多分支数据联合优化方式，在 `DIV-TTRL-PR` 上实现第一阶段（Generator）与第二阶段（Verifier）的共同进化。

## 核心设计理念：完全解耦采样频率与Loss计算，并保持二阶段尺度可控
**解耦声明**：伪标签的选取策略（如：对同一个初步答案进行验证时，可能是多次取样，也可能是少次取样）**完全不影响后端的更新逻辑**。
无论第二阶段针对某一个答案生成了 $m$ 个验证样本还是 $k$ 个验证样本，后端都统一为：**第一阶段保持原有 Loss 计算，第二阶段先在同一答案组内对每个验证样本分别计算 policy loss，再对组内 Loss 做归一化聚合，最后与第一阶段 Loss 相加并一次性反传（Backward）。**

推荐的数学形式为：
$$
\mathcal{L}_{second}^{(i)} = \frac{1}{n_i} \sum_{j=1}^{n_i} \ell_{second}^{(i,j)},
\qquad
\mathcal{L}_{second} = \frac{1}{G} \sum_{i=1}^{G} \mathcal{L}_{second}^{(i)}
$$
$$
\mathcal{L}_{total} = \mathcal{L}_{first} + \lambda \mathcal{L}_{second}
$$
其中 $G$ 是一个 batch 内的答案组数，$n_i$ 是第 $i$ 个答案组下实际发生的第二阶段采样数。这个写法比直接对所有第二阶段样本做 raw sum 更接近第一阶段的数值尺度，也更方便统一设计 $\lambda$。

---

## 详细修改步骤

### 1. 结构与格式奖励层（Reward 机制）
**目标**：仅依赖最终提供的 `pseudo_label` 和该标签是否被接受的标识，通过混淆矩阵评价验证结果，不用管伪标签是怎么取样出来的。

**修改位置**: `verl/utils/reward_score/ttrl/two_stage_utils.py` (或对应的奖励函数文件)
**逻辑实现**:
- **格式约束（Format Constraint）**：若 `<reverse_verification>` 或 `Verification Result` 未出现，直接判定格式失败，`Reward = -1`。
- **代理混淆矩阵（Proxy CM）奖励**：
  给定当前验证出来的结果 `verifier_pred`，以及外部传入的对比基准 `pseudo_label`、是否接收标记 `is_accepted` 和当前筛选的一致性分值 `consistency`：
  - **Proxy TP**: 预测一致且标签被接受 $\rightarrow$ `Reward = 1 * consistency`
  - **Proxy TN**: 预测不一致且标签未被接受 $\rightarrow$ `Reward = 1 * consistency`
  - **Proxy FP**: 预测一致但标签未被接受 $\rightarrow$ `Reward = -1 * consistency`
  - **Proxy FN**: 预测不一致但标签被接受 $\rightarrow$ `Reward = -0.5 * consistency`

### 2. 数据对齐与群组优势计算（Group GRPO in Stage 2）
**目标**：在同一个初步答案下，可能会生成多个验证回复，需要对这组验证回复进行 GRPO 优势归一化。

**修改位置**: `verl/trainer/ppo/ray_trainer.py`
**逻辑实现**:
1. **保留 DataProto**：不再仅提取文本，而是将第二阶段生成的 `verification_dataproto` 保留，聚合成 `batch_second`。
2. **计算 Second Stage Log Prob**：
   利用当前 Actor 求解：`batch_second = self.actor_rollout_wg.compute_log_prob(batch_second)`
3. **注入 Reward**：将计算好的 Proxy CM Reward 写入 `batch_second.batch["token_level_rewards"]`。
4. **组内计算 Advantage**：
   调用 `compute_advantage(batch_second, adv_estimator="grpo", norm_adv_by_std_in_grpo=True, num_repeat=m)`，其中 `m` 动态适配当前的验证采样次数。

### 2.1 二阶段 Loss 标定原则
为了让第二阶段 Loss 的量级和第一阶段尽量接近，建议不要把所有验证样本的 loss 直接 raw sum，而是采用“组内均值、组间均值”的聚合方式。

更具体地说：
- 对于同一个答案组，先对该组内的每个验证样本分别计算 GRPO 后的 policy loss。
- 组内先做平均，得到一个答案组级别的标量 loss。
- 再对 batch 内所有答案组做平均，得到最终的第二阶段总 loss。

这样做的好处是：
- 第二阶段损失不会随着单个答案的验证采样次数线性放大。
- $\lambda$ 只需要平衡“第一阶段 vs 第二阶段”两个任务，不需要再额外补偿采样次数。
- 伪标签筛选策略可以继续自由变化，因为更新端只关心 group-level 的归一化 loss。

### 3. 联合优化层：单次反向 + 二阶段组均值
**目标**：参考 `self-harmony` 的更新方式，但二阶段不再按固定双 batch 处理，而是按“答案组”的可变长度验证样本集合处理。最终效果仍然是一次 `zero_grad()`、一次 `backward()`、一次 `step()`。

**修改位置 3.1: Trainer 顶层调度** (`verl/trainer/ppo/ray_trainer.py`)
```python
# 传递第一阶段 batch 和二阶段验证组数据一同更新
actor_output = self.actor_rollout_wg.update_actor(batch_first, verification_groups)
```

**修改位置 3.2: FSDP Workers 派发接口** (`verl/workers/fsdp_workers.py`)
```python
@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
def update_actor(self, data_first: DataProto, verification_groups):
    metrics = self.actor.update_policy(data_first=data_first, verification_groups=verification_groups)
    return metrics
```

**修改位置 3.3: 核心策略更新逻辑** (`verl/workers/actor/dp_actor.py`)
主要逻辑是在一个 `optimizer.zero_grad()` 周期内：第一阶段正常计算 Loss；第二阶段对每个答案组先做组内平均，再对所有组做平均，最后把两部分相加后统一反传。
```python
def update_policy(self, data_first: DataProto, verification_groups):
    dataloader_first = data_first.batch.split(self.config.ppo_mini_batch_size)

    for epoch in range(self.config.ppo_epochs):
        for batch_f in dataloader_first:
            self.actor_optimizer.zero_grad()
            
            micro_batches_f = batch_f.split(self.config.ppo_micro_batch_size_per_gpu)
            
            first_stage_loss = 0.0
            for mb_f in micro_batches_f:
                # 1. First Stage Forward
                entropy_f, log_prob_f, *_ = self._forward_micro_batch(micro_batch=mb_f)
                pg_loss_f, _, _, _ = compute_policy_loss(..., log_prob=log_prob_f)
                first_stage_loss = first_stage_loss + (pg_loss_f - entropy_loss_f * entropy_coeff)

            second_stage_group_losses = []
            for group in verification_groups:
                group_micro_batches = group.split(self.config.ppo_micro_batch_size_per_gpu)
                group_loss = 0.0
                for mb_s in group_micro_batches:
                    entropy_s, log_prob_s, *_ = self._forward_micro_batch(micro_batch=mb_s)
                    pg_loss_s, _, _, _ = compute_policy_loss(..., log_prob=log_prob_s)
                    group_loss = group_loss + (pg_loss_s - entropy_s * entropy_coeff)
                second_stage_group_losses.append(group_loss / max(1, len(group_micro_batches)))

            second_stage_loss = sum(second_stage_group_losses) / max(1, len(second_stage_group_losses))
            lambda_factor = self.config.get("lambda_second", 0.5)
            loss = first_stage_loss + lambda_factor * second_stage_loss
            loss.backward()
            
            self._optimizer_step()
```

## 方案优势
此方案下，前端的**验证频率、采样逻辑**和后端的**联合训练机制**实现了真正的隔离。前端只需组装好带有合法 Rewards 和 Advantages 的验证组数据，后端就能统一进行单次反传式的协同更新，而且第二阶段的 loss 尺度也会更接近第一阶段，便于稳定设计 $\lambda$。