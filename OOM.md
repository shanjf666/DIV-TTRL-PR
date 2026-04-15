# Two-Stage Verify OOM

在 DIV-TTRL-PR 的 `two_stage_verify` 路径中，`update_actor` 的反向传播 OOM 主要不是单点 backward 算子本身，而是第二阶段数据与计算图在进入反传前已经被明显放大。

## 现象

- 训练在 `two_stage_verify=True` 时更容易触发 CUDA OOM。
- `response` 变长、`batch_second` 变大时更容易爆显存。
- OOM 往往发生在二阶段样本准备完毕、进入 actor update / backward 之前或过程中。

## 根因

1. 二阶段数据先被完整拼成 `batch_second`，再统一送入 GPU。相关位置见 [ray_trainer.py](verl/verl/trainer/ppo/ray_trainer.py#L1158) 和 [ray_trainer.py](verl/verl/trainer/ppo/ray_trainer.py#L1224)。
2. `batch_second` 上会额外写入 `token_level_rewards`、重新计算 `old_log_probs`，再计算 `advantages`，使峰值显存继续抬高。见 [ray_trainer.py](verl/verl/trainer/ppo/ray_trainer.py#L1420)、[ray_trainer.py](verl/verl/trainer/ppo/ray_trainer.py#L1432)、[ray_trainer.py](verl/verl/trainer/ppo/ray_trainer.py#L1444)。
3. worker 端 `update_actor` 会把 `data` 和 `data_second` 直接搬到 GPU；如果 `data_second` 也很大，显存峰值会进一步上升。见 [fsdp_workers.py](verl/verl/workers/fsdp_workers.py#L500)、[fsdp_workers.py](verl/verl/workers/fsdp_workers.py#L508)。
4. 二阶段 actor 更新里还存在一个高风险问题：`update_policy` 看起来被连续调用了两次。见 [fsdp_workers.py](verl/verl/workers/fsdp_workers.py#L528) 和 [fsdp_workers.py](verl/verl/workers/fsdp_workers.py#L529)。这会让显存与计算量翻倍，是首要排查点。
5. 二阶段微批反传是按 verification groups 逐个 micro-batch 做 `loss_s.backward()`，样本多、序列长时会继续堆高激活显存。见 [dp_actor.py](verl/verl/workers/actor/dp_actor.py#L302)、[dp_actor.py](verl/verl/workers/actor/dp_actor.py#L446)、[dp_actor.py](verl/verl/workers/actor/dp_actor.py#L499)、[dp_actor.py](verl/verl/workers/actor/dp_actor.py#L502)。

## 解决方法

### 优先级 1: 先修重复调用

- 确认并移除 `fsdp_workers.py` 里重复的 `update_policy` 调用。
- 同时检查 `preprocess_data` 是否被对 `data_second` 重复执行。

### 优先级 2: 降低二阶段峰值

- 减小 `n_votes_per_prompt`、`n_samples_per_prompt`、`two_stage_max_candidates`、`two_stage_max_new_tokens`。
- 必要时降低 `data.train_batch_size`、`ppo_mini_batch_size`、`ppo_micro_batch_size_per_gpu`。
- 让 `data_second` 尽量晚一点再搬到 GPU，不要在进入 worker 时就完整上卡。

### 优先级 3: 减少反传激活占用

- 开启或确认使用 bf16 / fp16。
- 保持 gradient checkpointing 开启。
- 如仍然 OOM，再考虑把二阶段反传拆成更小的 chunk 或先只保留验证统计、不做二阶段训练。

### 优先级 4: 只做必要的内存清理

- 在关键大张量用完后显式 `del`。
- `torch.cuda.empty_cache()` 只能缓解碎片，不是根治。
- 不要把它当成替代缩小 batch 或去掉重复调用的手段。

## 建议的排查顺序

1. 先检查 [fsdp_workers.py](verl/verl/workers/fsdp_workers.py#L500-L529) 是否存在重复 `update_policy`。
2. 再观察 `batch_second` 的大小和 `response_length` 是否过大。
3. 然后把二阶段样本规模减半，确认 OOM 是否消失。
4. 如果仍然 OOM，再继续压低主 batch 和二阶段 micro-batch。

## 一句话结论

这个 OOM 的本质是二阶段验证数据和二阶段反传同时把显存峰值抬高了，最值得优先修的是 worker 层重复调用，其次才是缩小 `batch_second` 和降低序列/批量规模。