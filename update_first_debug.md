# 最近两次提交（Commit）修改内容文档

本文档总结了在 TTRL 两阶段验证框架下进行的最近两次核心代码提交，主要涉及**鲁棒性修复**、**采样逻辑优化**以及**监控指标增强**。

---

## 提交1：优化采样逻辑与 Bug 修复
**Commit ID**: `797294994082a261bd1224bab3f7ebef9984af0a`
**主要变更文件**: `two_stage_utils.py`, `update_first.sh`

### 1. 技术修复 (two_stage_utils.py)
*   **修复索引越界与 KeyError**: 
    - 针对 `prompt_groups` 在过滤后索引不连续的问题，将 `group_results` 由固定长度列表改为动态字典寻址 `group_results.get(group_idx, [])`。
    - 修正了 `pseudo_labels` 的赋值逻辑，确保使用当前 Batch 内的遍历索引 `i` 而非原始全量索引 `group_idx`。
*   **鲁棒性对比**: 增强了在只有部分题目触发验证时的处理能力。

### 2. 参数与环境适配 (update_first.sh)
*   **多样本验证**: 启用了采样模式（Sampling Mode），并将每个候选人的验证次数提升至 8 次 (`+two_stage_n=8`)，以通过多数投票提高验证可信度。
*   **路径迁移**: 将默认的模型和输出路径适配至 AutoDL 容器路径 (`/root/autodl-tmp/`)。

---

## 提交2：监控指标增强与准确率对比
**Commit ID**: `99989b5f46bf6c8d91be5f05293d054599f995bc`
**主要变更文件**: `ray_trainer.py`, `ttrl.py`

### 1. 指标计算 (ttrl.py)
*   **新增准确率对比**: 
    - 显式引入了 `label_accuracy_majority`：记录如果不进行验证，原始多数投票选出的标签准确率。
    - 显式引入了 `label_accuracy_two_stage`：记录经过验证/过滤/回退流程后，最终被训练采纳的标签准确率。
*   **地基准备**: 在 Reward Manager 中利用 Ground Truth 完成上述对比计算，为 RL 训练提供了直观的效果闭环反馈。

### 2. 交互与可视化 (ray_trainer.py)
*   **终端实时打印**: 在训练循环中增加了对以上两个指标的实时对比输出，格式为：
    `[TwoStage] Accuracy Comparison: Majority=0.xxxx, Two-Stage=0.xxxx`
*   **Wandb 同步**: 确保上述对比指标自动同步至远程看板，方便回溯验证逻辑对伪标签质量的真实贡献率。

---

## 总结
这两次修改标志着两阶段验证算法从“可用”转变为“稳定且可量化”。通过 **提交1** 解决了系统崩溃问题并增强了验证深度，通过 **提交2** 建立了完整的准确率反馈回路。

---

## 当前代码状态与最近结果（Latest Results）的对比

根据您之前提供的终端输出（例如 `ParseFail=52`, `filtered in 0/3`），当前代码在以下方面进行了针对性优化：

### 1. 监控透明度 (Visibility)
*   **之前**: 只能看到最终的 `label_accuracy`，无法判断验证逻辑是否真的带来了正向收益。
*   **现在**: 增加了 `Majority vs Two-Stage` 的实时对比。您可以直观看到验证后的准确率是否高于原始多数票（例如：`Majority=0.6250 -> Two-Stage=0.6875`）。

### 2. 解析鲁棒性 (Parsing)
*   **之前**: 存在较高的 `ParseFail` (约 43%)。这通常是因为模型回复格式不规范。
*   **现在**: 
    - 引入了 `assistant` 回复引导（Prefill: `<reverse_verification>\n`），强制模型从正确格式开始输出。
    - 重构了解析器逻辑，采用更稳健的关键词包含匹配，不再依赖严苛的正规则表达式，预计将显著降低 `ParseFail`。

### 3. 验证深度与置信度 (Confidence)
*   **之前**: 可能存在单次验证偶然性大或者筛选逻辑未充分触发。
*   **现在**: 
    - 统一采用 **采样验证 (N=8)** 配合 **多数票表决 (True > False)**。这能有效对抗模型偶尔产生的错误验证回复。
    - 修复了 `prompt_group_idx` 的映射 Bug，确保每一个成功验证的伪标签都能精准“归位”到对应的 Prompt。

### 4. 训练性能 (Performance)
*   **优化建议**: 针对 GPU 利用率不连续的问题，建议在脚本中尝试调大 `+two_stage_micro_batch_size=64`。当前代码已支持该参数的动态注入。

