# 两阶段自验证伪标签流水线 - 最终总结

我们已经成功按照之前的计划，完成了两阶段自验证 (Two-Stage Self-Verification) 训练流水线的重构和落地。以下是本次修改的详细梳理：

## 1. 核心架构与工具类 (New)

#### `verl/utils/reward_score/ttrl/two_stage_utils.py`
创建了管理两阶段验证核心逻辑的工具包：
- **`extract_candidate_answers`**: 能够解析第一阶段生成的 Rollout，通过标准的 `auto_extract` 解析数学答案，并按照频次选出最有可能的 Top-K（可通过 `two_stage_max_candidates` 控制）。
- **`construct_verification_dataproto`**: 将第一阶段提炼出的有效候选答案，与原题目整合成 `DataProto`。引入了专门设计的 System Prompt 和 User Template 要求模型执行逆向验证（Reverse Verification）并最终按照固定 `XML` 输出 `Verification Result: [True/False]`。
- **`parse_verification_result`**: 基于正则智能解析 `XML` 中的格式和 Fallback 到结尾独立的 `True/False` 字段。
- **`resolve_pseudo_labels`**: 提供了 **Greedy** 和 **Sampling** 两种模式的聚合，并支持了两种 Fallback 策略：
  - `majority`: 当所有人均被判断为 False 时，仍然采用频率最高的原本候选人。
  - `penalize`: 返回 `None`，代表模型认为全军覆没并给后续发放 -1 惩罚信号。

## 2. 训练主循环改造

#### `verl/trainer/ppo/ray_trainer.py`
通过拦截 `gen_batch_output` 并在其生成后插入了一个完整且相互独立的验证推理流程：
- **无缝集成 `DataProto` 机制**: 利用现成的 `pad_dataproto_to_divisor` 将验证专用的 batch 传给 `actor_rollout_wg.generate_sequences` 实现了安全分布式推理。
- **OOM 缓解与显存清理**: 结合你之前的 OOM 处理经验，加入 `two_stage_micro_batch_size` 切片。在每一个 Micro-batch 完毕后严格使用 `del` 消除废弃变量，并运行 `gc.collect()` + `torch.cuda.empty_cache()` 最大程度释放闲置显存。
- **伪标签流转**: 解析完毕的强化伪标签列表会被平铺对齐到每个子样本并赋值进 `batch.non_tensor_batch["verified_pseudo_label"]`，供 Reward Manager 后续抓取。

## 3. 推理引擎传参渗透

#### `verl/workers/fsdp_workers.py`
底层 `generate_sequences` 方法被适度改造以识别验证任务：
- 在 `prompts.meta_info` 识别探针变量 `verification_mode`。
- 如果探测出验证任务，则会使用 `verification_max_new_tokens` (默认 512，而非动辄上千的原始解题长度)，并在 `sampling` 模式下接收特定的温度 (如 `0.6`)。
- 通过 `update_sampling_params` 在不破坏底层架构的情况下完成了临时参数的安全覆盖。

## 4. TTRL 奖励控制适配

#### `verl/workers/reward_manager/ttrl.py`
重构了 `_compute_ttrl_reward` 内的 Label 选择树，引入了最新的三重优先级原则：
1. **最高级 `two_stage`**: 如果有 `verified_pseudo_label` 结果直接采信验证出来的答案；如果是 `penalize` 失败标志，则替换成魔术字符串 `__TWO_STAGE_PENALIZE_ALL__`。
2. **次高级 `offline_hybrid`**: 若第一层失效，走之前的 Offline 离线强一致性匹配。
3. **兜底层 `online_majority`**: 最后 fallback 为 Online 频次本身。
- **全面惩罚**: 如果第一层认定全军覆没（Penalize），重写全体奖励直接结算为 `-1.0` 来告诉 PPO 模型这个 prompt 下所有的 rollout 均是错误方向。
- **修复**: 修改了内部因为字符聚合导致 `np.mean` 崩溃的旧代码，将 `label_source` 细分转化为连续字典值如 `label_source_two_stage`, `label_source_penalize` 共计聚合观测。

## 5. 配置与启动脚本更新

#### `examples/labelfree/aime25_train.sh`
支持了一系列环境变量传递（目前通过 `train.sh` 末尾传参，方便你在未来的超参搜索中自由调控）：
```bash
  +two_stage_verify=False \
  +two_stage_mode=greedy \         # 方案选择: greedy / sampling
  +two_stage_n=4 \                 # sampling 下的票数
  +two_stage_max_candidates=10 \   # Candidate上限
  +two_stage_max_new_tokens=512 \  # 验证专有长度限制
  +two_stage_fallback=majority \   # 失败回退: majority / penalize
  +two_stage_micro_batch_size=0 \  # OOM调优
```

---
> [!NOTE] 
> 所有的修改都已经集成，并且通过了基本的语法树解析测试。该架构对 OOM 进行了深度调优以适配后续的 4B 模型。如果你后续在运行期间发觉显存出现瓶颈，请考虑调整 `two_stage_micro_batch_size` 并且降低 `two_stage_max_candidates` 来动态止盈。
