# DIV-TTRL-PR Metric Audit

这份文档梳理当前 DIV-TTRL-PR 里会写入 wandb 的主要指标，并重点回答 two-stage-verify 场景下两个问题：哪些指标的名字和真实含义已经不一致，哪些指标因为实现冲突或只做了占位初始化，导致训练时长期显示为 0。

审计范围主要是以下四个文件：

- [verl/verl/trainer/ppo/ray_trainer.py](verl/verl/trainer/ppo/ray_trainer.py)
- [verl/verl/workers/reward_manager/ttrl.py](verl/verl/workers/reward_manager/ttrl.py)
- [verl/verl/utils/reward_score/ttrl/ttt_metrics.py](verl/verl/utils/reward_score/ttrl/ttt_metrics.py)
- [verl/verl/utils/reward_score/ttrl/two_stage_utils.py](verl/verl/utils/reward_score/ttrl/two_stage_utils.py)

## 先给结论

1. two-stage-verify 开启后，最容易误读的指标是 raw 的 label_accuracy 和 majority_voting_reward。它们已经不再只表示“多数票”的结果，而是表示“最终用于奖励的标签”对应的结果。
2. train/tp_rate、train/tn_rate、train/fp_rate、train/fn_rate、train/gt_tp_rate、train/gt_tn_rate、train/gt_fp_rate、train/gt_fn_rate 这一组指标，在当前代码里是高风险项。它们在 [ray_trainer.py](verl/verl/trainer/ppo/ray_trainer.py#L1349) 先被初始化为 0.0，而 [two_stage_utils.py](verl/verl/utils/reward_score/ttrl/two_stage_utils.py#L572) 又提供了一个同名但不同语义的 compute_proxy_cm_reward 实现，导致 trainer 的调用链和返回值 schema 不一致。
3. 如果当前 step 没有任何 prompt group 触发验证，two-stage 相关的覆盖率指标显示为 0 是正常现象，不是 bug。真正需要重点排查的是“本该有验证却一直为 0”的场景，以及“指标名和计算对象不一致”的场景。

## Wandb 指标总览

### 1. TTRL 基础指标

这些指标的原始定义来自 [ttt_metrics.py](verl/verl/utils/reward_score/ttrl/ttt_metrics.py#L44)；训练时由 [ttrl.py](verl/verl/workers/reward_manager/ttrl.py#L345) 统一加上 train/ 前缀写入 wandb。

| 指标 | 计算方式 | 含义 | two-stage 下是否仍然成立 |
|---|---|---|---|
| label_accuracy | 在 test_time_train_metrics 内，对 selected label 做 auto_verify 后得到的正确率 | 最终用于奖励的标签是否和 ground truth 一致 | 成立，但这里的 selected label 可能是 verified_label，不一定是多数票 |
| reward_accuracy | selected label 产生的逐样本奖励，与 ground truth 奖励逐项对齐后的相等比例 | 这是“奖励模式是否和真值奖励一致”，不是分类准确率 | 成立，但语义已经不是单纯的多数票评估 |
| majority_ratio | 多数答案频率 / 样本数 | 多数票强度 | 成立，仍然是多数票统计 |
| ground_truth_ratio | ground-truth 奖励的平均值 | 更接近 oracle 正确率，而不是“ground truth 的比例” | 成立，但名字容易误导 |
| majority_voting_reward | selected label 下的平均奖励 | 其实是“当前标签策略的平均奖励” | 成立，但 two-stage 时名字不再准确 |
| pass@k | 至少有一个样本在 ground truth 下为正 | 组内存在正确候选答案 | 成立 |
| neg_log_likelihood | 当前实现里实际由策略熵或类似 entropy 指标填充 | 训练策略的分布熵/不确定性 | 成立，但名字和物理量不完全一致 |

补充说明：post_reward_accuracy、post_ground_truth_ratio、post_pass@k 这类字段只出现在 helper 函数 [ttt_metrics.py](verl/verl/utils/reward_score/ttrl/ttt_metrics.py#L79) 中，当前主训练链路并没有把它们作为 wandb 主指标常规写入。

### 2. two-stage 下的奖励管理指标

这些指标主要由 [ttrl.py](verl/verl/workers/reward_manager/ttrl.py#L291) 计算并在 wandb 里写成 train/ 前缀。

| 指标 | 计算位置 | 含义 | two-stage 下的解释 |
|---|---|---|---|
| label_accuracy_majority | [ttrl.py](verl/verl/workers/reward_manager/ttrl.py#L345) | 在线多数答案对 ground truth 的准确率 | 这是 two-stage 前的基线，应当保留 |
| label_accuracy_two_stage | [ttrl.py](verl/verl/workers/reward_manager/ttrl.py#L346) | 直接复用 ttrl_metrics[label_accuracy] | 在 two-stage 开启时，它表示“最终用于奖励的标签”的准确率 |
| off_policy_ratio | [ttrl.py](verl/verl/workers/reward_manager/ttrl.py#L348) | 是否启用了 offline_hybrid fallback | 仅用于看 fallback 发生比例 |
| label_source_two_stage | [ttrl.py](verl/verl/workers/reward_manager/ttrl.py#L350) | 本组标签是否来自 two-stage verified label | 0/1 标志 |
| label_source_penalize | [ttrl.py](verl/verl/workers/reward_manager/ttrl.py#L351) | 是否触发 two-stage penalize fallback | 0/1 标志 |
| label_source_offline | [ttrl.py](verl/verl/workers/reward_manager/ttrl.py#L352) | 是否触发 offline_hybrid fallback | 0/1 标志 |
| two_stage_penalized | [ttrl.py](verl/verl/workers/reward_manager/ttrl.py#L355) | 是否把该组整体打成惩罚模式 | 0/1 标志 |
| false_positive_rate | [ttrl.py](verl/verl/workers/reward_manager/ttrl.py#L375) | 选中的伪标签与真值之间的 FP 率 | 这是“伪标签视角”的错误率 |
| false_negative_rate | [ttrl.py](verl/verl/workers/reward_manager/ttrl.py#L376) | 选中的伪标签与真值之间的 FN 率 | 同上 |
| diversity_ratio | [ttrl.py](verl/verl/workers/reward_manager/ttrl.py#L387) | unique answers / total answers | 仍然是多样性指标 |

对这组指标最关键的解释是：raw 的 label_accuracy 已经不是“纯多数票准确率”了。要看多数票基线，请看 label_accuracy_majority；要看 two-stage 最终生效标签，请看 label_accuracy_two_stage。

### 3. two-stage 验证流程指标

这组指标由 [ray_trainer.py](verl/verl/trainer/ppo/ray_trainer.py#L997) 的 _run_two_stage_verification 直接写入 metrics，再统一进入 wandb。

| 指标 | 计算方式 | 含义 | 是否建议保留 |
|---|---|---|---|
| two_stage_trigger_rate | 触发验证的 prompt group 数 / 全部 prompt group 数 | 验证覆盖率 | 建议保留 |
| two_stage_majority_rate_mean | 所有 prompt group 的平均 majority_rate | 当前批次多数票强度均值 | 建议保留 |
| two_stage_total_candidates | 触发验证的组里候选答案总数 | 二阶段实际要验证多少候选 | 建议保留 |
| two_stage_filtered_ratio | 成功筛出“不同于多数票”的新标签比例 | 两阶段纠偏能力 | 建议保留 |
| two_stage_true_rate | 解析出 True 的比例 | verifier 输出中的 True 比例 | 建议保留 |
| two_stage_false_rate | 解析出 False 的比例 | verifier 输出中的 False 比例 | 建议保留 |
| two_stage_parse_fail_rate | 无法解析出 True/False 的比例 | verifier 格式稳定性 | 建议保留 |
| two_stage_verified_ratio | 非 None 的 verified label 比例 | 进入最终标签路径的比例 | 建议保留 |

### 4. 代理混淆矩阵指标

这里是最容易出问题的部分。当前代码里，[ray_trainer.py](verl/verl/trainer/ppo/ray_trainer.py#L1413) 会把二阶段样本送进 compute_proxy_cm_reward，再把返回值写入 wandb。

但是 [two_stage_utils.py](verl/verl/utils/reward_score/ttrl/two_stage_utils.py#L458) 和 [two_stage_utils.py](verl/verl/utils/reward_score/ttrl/two_stage_utils.py#L572) 里存在两个同名的 compute_proxy_cm_reward，两个版本的参数和返回 schema 不一致。前一个版本返回 tp_rate / gt_tp_rate 这类字段，后一个版本返回 proxy_cm/tp_rate 这类字段，并且会覆盖前一个定义。

这意味着当前两阶段混淆矩阵指标存在 schema 冲突，不能直接把它当成稳定可靠的统计结果。

## two-stage 下哪些指标需要更换名字

以下是“名字已经不再准确”的指标，建议在看板或后续代码里换成更直接的语义：

| 当前指标 | 为什么名字不准确 | 建议替换名 |
|---|---|---|
| label_accuracy | 在 two-stage 里它表示最终生效标签的准确率，不再只是多数票准确率 | effective_label_accuracy 或 selected_label_accuracy |
| reward_accuracy | 依赖 selected label 的奖励一致率，不是通用 reward accuracy | selected_label_reward_accuracy |
| majority_voting_reward | two-stage 时它并不一定来自多数票 | selected_label_reward_mean |
| ground_truth_ratio | 实际上是 oracle reward mean / 正确率 | oracle_correct_rate 或 gt_reward_mean |
| neg_log_likelihood | 实际更接近策略熵 | strategy_entropy |
| tp_rate / tn_rate / fp_rate / fn_rate | 如果用于 two-stage verifier，它是代理混淆矩阵，不是普通分类 TP/TN | pseudo_cm_tp_rate 等更明确的名字 |

其中最需要优先改的，是 majority_voting_reward 和 ground_truth_ratio。前者在 two-stage 里已经不再是“多数票奖励”，后者也不是普通意义上的 ground truth ratio，而是 oracle 正确率。

## 哪些指标会一直是 0，或者看起来像 0

### 1. 现在最像“真 bug”的零值指标

这些字段在 [ray_trainer.py](verl/verl/trainer/ppo/ray_trainer.py#L1349) 里先被写成 0.0，但当前 trainer 代码并没有稳定地用同一套 schema 去覆盖它们：

- train/tp_rate
- train/tn_rate
- train/fp_rate
- train/fn_rate
- train/gt_tp_rate
- train/gt_tn_rate
- train/gt_fp_rate
- train/gt_fn_rate

原因不是“这些数一定真的为 0”，而是当前实现里这组 key 的初始化和实际返回值 schema 没有对齐。

更具体地说：

1. [ray_trainer.py](verl/verl/trainer/ppo/ray_trainer.py#L1349) 先写了这一组占位 0。
2. [two_stage_utils.py](verl/verl/utils/reward_score/ttrl/two_stage_utils.py#L572) 的后一个 compute_proxy_cm_reward 返回的是 proxy_cm/* 这一套 key，而 trainer 的占位 key 仍然是 tp_rate / gt_*。
3. 如果你只看 wandb 里 train/tp_rate 这一组字段，就会看到它们长期为 0，或者根本不随真实验证变化。

### 2. 另一类“看起来像 0，但其实是正常”的指标

这些指标为 0 并不一定是 bug：

- two_stage_trigger_rate 为 0：说明当前 batch 没有 prompt group 满足 majority_rate < 0.3，属于正常情况。
- two_stage_filtered_ratio 为 0：说明触发验证的组里没有筛出新标签，可能是 verifier 没有改写结果，也可能是样本本身就很稳。
- two_stage_parse_fail_rate 为 0：说明 verifier 格式很稳定，解析没有失败。

## 目前最应该关注的修复点

1. 统一 compute_proxy_cm_reward 的版本，只保留一套 schema。当前 [two_stage_utils.py](verl/verl/utils/reward_score/ttrl/two_stage_utils.py) 里有两个同名实现，这是第一优先级问题。
2. 让 ray_trainer 的初始化 key 和实际写入 key 完全一致。现在的 train/tp_rate 和 train/gt_* 这组 key，和后面的返回结果不在同一语义层上。
3. 如果你要继续沿用当前看板，建议把 raw label_accuracy 当成“effective label accuracy”看，不要再把它直接理解成多数票准确率。

## 推荐的看板解读方式

如果你在 wandb 里做日常观察，优先看下面这几个组合：

- 覆盖率：two_stage_trigger_rate
- 验证稳定性：two_stage_parse_fail_rate
- 两阶段纠偏能力：two_stage_filtered_ratio
- 最终标签质量：label_accuracy_majority 对比 label_accuracy_two_stage
- 代理混淆矩阵：在修复 compute_proxy_cm_reward schema 之后再看 tp_rate / gt_* 这一组；修复前不要把它们当成可信结论

如果你希望，我下一步可以直接把这份审计里指出的两个代码问题也修掉：

1. 删掉或重命名 two_stage_utils.py 里重复的 compute_proxy_cm_reward。
2. 把 ray_trainer.py 的混淆矩阵 key 对齐到唯一的 schema，避免 train/tp_rate 这组字段继续长期为 0。