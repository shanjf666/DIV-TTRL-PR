# TTRL (两阶段无监督验证强化学习) 整体方法框架

## 一、 核心思想 (Core Idea)

传统强化学习在数学或逻辑推理任务中面临稀疏奖励（Sparse Reward）和探索盲区的问题。本方法的**核心思想在于“生成-验证”闭环的统一自进化**：
1. 使用单一共享权重的 LLM 同时扮演“解题生成器 (Generator)”和“裁判验证器 (Verifier)”。
2. 基于**“自适应信心路由 (Adaptive Confidence Routing)”**和**“非对称验证惩罚矩阵 (Asymmetric Penalty Matrix)”**，在无监督环境下自发挖掘高质量伪标签（Pseudo-labels）。
3. 采用物理隔离的相对优势估计与联合更新法则，消除两种任务分布差异带来的学习冲突，实现“表达能力”与“批判能力”的双向攀升。

---

## 二、 自适应路由策略 (Adaptive Routing Strategy)

在第一阶段采样结束后，所有样本全量送入第二阶段进行推理评估。评估完成后，根据第一阶段样本投票分布的多数票一致性 $\mathcal{C}_{maj}$ (Majority Rate) 设定分流阈值 $\tau$ (high_consistency_threshold, 默认 0.5)，执行智能双路由决策树：

### 路线 A: 高一致性舒适区 ($\mathcal{C}_{maj} \ge \tau$)
* **判定特征**：模型对当前解法表现出高度自信（共识度高）。
* **行动**：不强行干预，直接采信第一阶段的多数派答案 (`majority_answer`) 作为事实伪标签（Pseudo-label）。
* **信心得分 (Consistency)**：直接继承该答案的多数派频次比例。

### 路线 B: 低一致性探索区 ($\mathcal{C}_{maj} < \tau$)
* **判定特征**：样本回答发散，存在探索迷失或困难模式。
* **行动**：在第二阶段的自我验证结果中，统计所有候选答案的验证分布，筛选出满足核心逻辑检验 **`True 判决数 > False 判决数`** 的候选答案池。
* **进一步划分为两个子路径：**
  * **路线 B1: 破局成功（存在合法候选者）** 
    从合格池中，根据指定的探索策略 (`low_consistency_strategy`) 选择最终答案作为伪标签：
    - `"true" 策略`：绝对信任验证模型，挑选 `True 判决票数` 最多的答案。
    - `"majority" 策略`：权衡生成偏好，挑选在此前“第一阶段中生成频率”最高的答案。
  * **路线 B2: 退回防守（无任何候选者满足 True>False）**
    说明该维度属于模型当前能力之外的绝对盲区。此时触发后退机制 (`fallback_mode`)，将伪标签重置为 `majority_answer`，并可以选择通过丢弃该样本第二阶段的梯度，保护模型不被随机噪声破坏。

---

## 三、 奖励设计 (Reward Design)

框架基于生成的伪标签，执行双轨奖励分配。

### 第一阶段奖励 (Stage 1 Reward: 驱动多态生成)
* 依靠 PASS-GRPO 体系获取基础答案间的比较优势，用于维持生成结果在符合格式约束条件下的广度探索。第一阶段产生的回复会与环境本身的规则校验对齐（如符合特定 XML 标签闭环）。

### 第二阶段奖励 (Stage 2 Reward: 塑造批判能力)
第二阶段通过构建**代理混淆矩阵（Proxy Confusion Matrix）** 实施奖励。引入了一套与“信心得分（Consistency）”动态绑定的非对称矩阵：

以 $\mathcal{C}$ 代表最终确定的伪标签信心得分（即最终被采信的频率比）：

| 验证结论 | 样本是否为伪标 | 类别 | 强化奖励 (Reward) | 演化逻辑及惩罚力度 |
|:---:|:---:|:---:|:---:| :--- |
| **解析失败** | - | - | **`-1.0`** | 强烈惩罚拒绝执行判定格式的模型动作。 |
| **True** | 是伪标 | **TP** | **`+1.0 * C`** | 正确确认“自认为正确”的答案，予以完整表彰。 |
| **False** | 非伪标 | **TN** | **`+1.0 * C`** | 成功驳回明显偏离共识的离群答案，同样表彰排错能力。 |
| **True** | 非伪标 | **FP** | **`-0.5 * C`** | **轻度惩罚**。宽容误判，允许模型在探索期适度“盲目自信”，防范在困难题下完全收缩。 |
| **False** | 是伪标 | **FN** | **`-1.0 * C`** | **严重惩罚**。严厉打击自我怀疑和幻觉验证，保证模型对核心共识具备坚定持有的定力。 |

---

## 四、 隔离优势计算与联合更新法则

由于生成与判别任务分布的方差迥异，直接混合 PPO 会导致梯度坍塌。框架使用了分离计算手段：

1. **第一阶段优势 (Generator Advantage)**：
   基于同一个 Prompt 内的不同候选答案进行 GRPO 优势估计（白化中心化）。如果在“路线 B2”下命中 `no_update_both` 模式，会激活 **Zero Advantage Mask (优势清零掩码)**，对该组样本实施截断处理（Advantage 乘 0），以此隔离极端噪声。
   
2. **第二阶段优势 (Verifier Advantage)**：
  仅限参与了第二阶段判定评估的样本组。这里采用 `Prompt Index` 作为分组键，把同一 prompt 下的所有候选验证样本放到同一个 **UID** 中做归一化。模型基于同一个 prompt 的全部验证奖励（如 N 采样）进行大组级别的优势局域中心化（Grouped Advantage Normalization）。

3. **联合梯度叠加**：
   在最终的 PPO Actor-Critic 图计算时，对两类 Advantage 产生的 Loss 标量施加全局调权：
   $$ \mathcal{L}_{total} = \mathcal{L}_{first} + \lambda_{second} \cdot \mathcal{L}_{second} $$

---

## 五、 指标监控 (Metrics)

模型在训练过程中提取了一系列独占的诊断指标，直接反馈判别能力与纠偏能力：

* **自验证干预规模**
  * `train/two_stage_trigger_rate`: 触发二阶段重选机制的数据切片比率。
  * `train/two_stage_filtered_ratio`: 因为跌入 B2 盲区被 Fallback 截断的样本失效比。
* **代理矩阵置信分布**
  * `train/tp_rate`, `train/tn_rate`, `train/fp_rate`, `train/fn_rate`: 统计验证混淆概率。
  * `train/parse_fail_rate`: 反映模型对验证任务的理解服从度。
* **高阶纠正式增益**
  * `train/two_stage_flip_rate`: 表明二阶段推翻原始一阶段多数派，引发“觉醒纠正（Flip）”的比例，反映验证器的干预强度。
  * `train/two_stage_net_correction_gain`: $( \text{Correct Flips} - \text{Incorrect Flips} )$，是衡量这套 TTRL 是否带来额外“净增益”的核心指标。
