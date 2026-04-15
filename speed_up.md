# Second-Stage Speed Up Notes

这份文档只聚焦 second-stage verification 的输出效率，不改你的核心语义，也不动你当前高一致性 / 低一致性分支的判断逻辑。目标是在参数基本不变的前提下，让第二阶段更接近第一阶段那种持续高利用率的 rollout 形态。

## 1. 现状判断

当前 second-stage 的主路径在 [verl/verl/trainer/ppo/ray_trainer.py](verl/verl/trainer/ppo/ray_trainer.py#L997) 到 [verl/verl/trainer/ppo/ray_trainer.py](verl/verl/trainer/ppo/ray_trainer.py#L1147) 之间。

它的执行方式有三个明显特征：

- 先把 Pass 1 的候选答案抽取出来，再把所有需要验证的候选展开成独立 verification 样本。
- 然后按固定 `micro_bs` 一块一块串行跑，每个 chunk 都会先做一次 padding，再丢给 rollout 引擎。
- 每个 chunk 结束后都会执行一轮显式清理，包括 `gc.collect()` 和 `torch.cuda.empty_cache()`。

这就是你图里 second-stage GPU utilization 和显存上下波动的直接来源。第一阶段之所以更稳，是因为它更像一次连续的大批量 rollout；第二阶段现在更像“很多小请求串行排队”。

另外，当前 verification 的构造逻辑在 [verl/verl/utils/reward_score/ttrl/two_stage_utils.py](verl/verl/utils/reward_score/ttrl/two_stage_utils.py#L223) 里已经把 sampling / frequency 复制做掉了，也就是说，第二阶段的样本数不是自然长出来的，而是先在 Python 层展开，再交给 rollout。

## 2. 实现思路一：把固定重复采样改成引擎原生批量采样

### 核心目标

不改变你现有的 `two_stage_mode`、`two_stage_n`、`two_stage_max_candidates` 这些外部参数含义，但把“一个候选要生成多条 verification 输出”这件事，从 Python 层的 DataProto 复制，迁移到 rollout 引擎内部一次性完成。

### 为什么这一步有效

当前 `construct_verification_dataproto()` 会在 [verl/verl/utils/reward_score/ttrl/two_stage_utils.py](verl/verl/utils/reward_score/ttrl/two_stage_utils.py#L250) 附近按 candidate 频次或固定 `verification_n` 复制样本。这样做的问题不是语义错，而是执行代价高：

- 同一份 prompt 会被重复 tokenization、重复 padding、重复打包。
- DataProto 复制会放大 Python 侧对象数和 host-device 传输成本。
- 生成端看到的是很多“彼此几乎一样”的小样本，调度粒度碎，吞吐不容易跑满。

相比之下，第一阶段之所以更饱满，是因为 rollout 本身就是按 `n_vote` 或 `n` 的方式批量跑的，而不是靠外层复制样本去模拟多采样。

### 具体改法

#### 2.1 在 verification batch 里显式传递采样数

现在 [verl/verl/workers/fsdp_workers.py](verl/verl/workers/fsdp_workers.py#L586) 的 verification 分支只识别 `verification_mode`，并读取 `verification_max_new_tokens`、`verification_temperature`、`verification_top_p`，但没有把“本次 verification 要生成多少条 completion”作为一个独立信号下发。

建议做法是：

- 在 `ray_trainer.py` 构造 verification batch 时，额外写入 `verification_n`。
- 对固定采样模式，`verification_n` 就是当前 `two_stage_n`。
- 对动态频次模式，保留现有复制逻辑或者显式把 `verification_n` 设成频次派生值。

这样 worker 侧可以根据 `verification_n` 判断该走“原生多采样”还是“动态频次复制”。

#### 2.2 让 worker 侧直接做 n-sampling

在 [verl/verl/workers/fsdp_workers.py](verl/verl/workers/fsdp_workers.py#L586) 的 verification branch 里，当前做法是：

- 设置 verification max tokens。
- 如果是 sampling，就设置 temperature / top_p。
- 但生成调用最终仍然把 `n` 当成 1。

建议改成：

- 当 `verification_mode == sampling` 且 `verification_n > 1` 时，让 rollout 引擎一次性生成 `n` 条 completion。
- 只有在动态频次模式下，才保留“按频次展开样本”的兼容路径。

这一步的关键是把“重复生成”移到引擎内部，让 vLLM 直接吃多采样请求，而不是外面复制 N 份同样的 prompt。

#### 2.3 mapping 的处理方式不变，只改展开位置

你后面的 pseudo label 选择、True/False 解析、proxy reward 计算都依赖 `verification_mapping`。

建议保留映射语义不变，但把映射的扩展点从“构造 batch 时展开”前移到“引擎返回 completion 后展开”。这样能做到：

- 输出格式不变。
- 聚合逻辑不变。
- 只是把重复样本的产生点从 Python 层挪到 rollout 层。

### 推荐伪代码

```python
# ray_trainer.py
verification_batch.meta_info["verification_n"] = self.two_stage_n
verification_batch.meta_info["verification_mode"] = self.two_stage_mode

# fsdp_workers.py
verification_n = prompts.meta_info.get("verification_n", 1)
if verification_mode == "sampling" and verification_n > 1:
    verification_kwargs["n"] = verification_n
else:
    verification_kwargs["n"] = 1
```

### 预期收益

- 少做一轮 DataProto 复制。
- 少做一轮重复 tokenization / padding。
- 减少 worker 和 rollout 之间的对象搬运。
- 提高第二阶段的连续调度性，让 GPU 更容易维持较高占用。

### 风险点

- 要确保 sample 展开的顺序和 `verification_mapping` 一一对应。
- 要验证 `verification_n` 在 sampling 模式下不会改变你现有的 pseudo label 聚合结果。
- 如果你保留动态频次路径，要确认它不会和原生 n-sampling 发生混用。

### 验证方式

先用小 batch 对比以下四件事：

- 总输出条数是否一致。
- 每个 prompt group 的 candidate 计数是否一致。
- `parse_verification_result()` 的 True / False 比例是否一致。
- 最终 `select_final_pseudo_labels()` 的结果是否一致。

---

## 3. 实现思路二：第二阶段 micro-batch 改成 token-aware 分桶

### 核心目标

不按“样本个数”切 verification chunk，而按“有效 token 数”切 verification chunk。这样可以减少 padding 浪费，缓解短块和长块混跑造成的 GPU 空转。

### 为什么这一步很重要

当前 second-stage 的 chunk 逻辑在 [verl/verl/trainer/ppo/ray_trainer.py](verl/verl/trainer/ppo/ray_trainer.py#L1098) 到 [verl/verl/trainer/ppo/ray_trainer.py](verl/verl/trainer/ppo/ray_trainer.py#L1147) 是固定 `micro_bs` 切片：

- 样本数相同，不代表 token 数相同。
- verification prompt 的长度会因为问题长度、candidate answer 长度、模板拼接长度而明显波动。
- chunk 内只要混入一个长样本，整块 padding 成本就会抬高。

这也是你看到显存和利用率上下抖动的第二个核心原因。

### 现成工具可以直接复用

仓库里已经有按长度平衡分桶的工具：

- [verl/verl/utils/seqlen_balancing.py](verl/verl/utils/seqlen_balancing.py#L223) 的 `rearrange_micro_batches()`。
- 这个工具在 actor / critic 路径已经在用，例如 [verl/verl/workers/actor/dp_actor.py](verl/verl/workers/actor/dp_actor.py#L246) 和 [verl/verl/workers/critic/dp_critic.py](verl/verl/workers/critic/dp_critic.py#L151)。

这说明它不是“新的实验性逻辑”，而是仓库里已经验证过的分桶方式。second-stage 直接沿用它最稳。

### 具体改法

#### 3.1 先算 verification prompt 的有效长度

在构造好 `verification_batch` 后，不要立刻按样本数切片。先统计每条 verification 样本的有效 token 长度，依据 `attention_mask` 的非零长度做分桶。

这个长度信息可以直接从 `verification_batch.batch["attention_mask"]` 读，不需要额外 tokenization。

#### 3.2 用 `rearrange_micro_batches()` 替代固定切片

把当前的：

- `for chunk_start in range(0, total_verification_size, micro_bs)`
- `chunk = verification_batch[chunk_start:chunk_end]`

替换成 token-aware 的 micro batch 列表。也就是：

- 按总 token 目标把 verification_batch 分成若干个平衡 chunk。
- 让每个 chunk 的 token 总量尽量接近。
- 再对每个 chunk 做 `pad_dataproto_to_divisor()`，保证 dp 分片可用。

这样每个 chunk 的“真实工作量”会更均匀，GPU 不会一会儿满载、一会儿掉下去。

#### 3.3 把清缓存从“每块一次”改成“整轮一次”

你现在在每个 chunk 完成后都做：

- `del chunk_output_padded`
- `del chunk_output`
- `del chunk_padded`
- `gc.collect()`
- `torch.cuda.empty_cache()`

这对防 OOM 有帮助，但会把连续性打碎。

建议改成：

- 每个 chunk 只释放局部变量。
- `gc.collect()` 和 `torch.cuda.empty_cache()` 改成在整轮 second-stage 结束后调用。
- 如果你担心碎片，可以每处理若干个 chunk 再清一次，而不是每块都清。

### 推荐伪代码

```python
from verl.utils.seqlen_balancing import rearrange_micro_batches

micro_batches, _ = rearrange_micro_batches(
    batch=verification_batch.batch,
    max_token_len=self.config.data.max_prompt_length + self.two_stage_max_new_tokens,
)

for micro_batch in micro_batches:
    chunk = DataProto(batch=micro_batch, ...)
    chunk_padded, pad_size = pad_dataproto_to_divisor(chunk, self.actor_rollout_wg.world_size)
    chunk_output = self.actor_rollout_wg.generate_sequences(chunk_padded)
```

### 预期收益

- 减少 padding 浪费。
- 降低 chunk 之间的负载波动。
- 让 GPU utilization 更稳定。
- second-stage 的 memory 曲线会更平滑，不会像现在这样频繁掉到低位。

### 风险点

- 必须保留 `verification_mapping` 和输出顺序的可恢复性。
- 如果后续还要按 prompt group 聚合，最好保留一个 reverse index。
- 分桶后 chunk 之间的样本顺序可能变化，但只要映射能回填，语义就不会变。

### 验证方式

建议重点看三项：

- 每个 chunk 的 token 总量是否更均匀。
- GPU utilization 是否从明显锯齿变成相对平滑的平台。
- second-stage 总耗时是否下降，或者至少在相同参数下更稳定。

---

## 4. 推荐落地顺序

如果你想先做最小改动，我建议按这个顺序：

1. 先做思路一，减少固定重复采样带来的 Python 层开销。
2. 再做思路二，把 chunk 切分改成 token-aware。
3. 最后再考虑是否进一步收紧 cache 清理频率。

这样改的好处是：外部参数可以先不动，主要优化执行路径；同时如果结果不理想，也容易回滚单个点。

---

## 5. 不建议现在立刻改的内容

为了保持语义稳定，我不建议你一开始就改这些：

- 不要先改 `two_stage_max_candidates`、`two_stage_n` 这类行为参数。
- 不要先把 verification 的判断逻辑和 pseudo label 选择逻辑重写掉。
- 不要先动 reward 计算和 GRPO advantage 的公式。

当前问题是 second-stage 的执行形态不够高效，不是它的语义有问题。

如果你要，我下一步可以直接根据这份文档把代码改出来，先做“引擎原生 sampling + token-aware 分桶”这两个点。