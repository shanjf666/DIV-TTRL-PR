# 简化计划: 清理 RayTrainer、CoreAlgos 及配置文件中的冗余实验逻辑

为了保持代码库的精简，我们将移除不再使用的实验性 Advantage Estimator 逻辑及其在 Ray 工作流、核心算法和配置中的所有引用。

## 一、 清理目标 (移除符号)

我们将移除以下四个不再维护的实验性符号及其完整实现链路：
*   `DIVERSITY_DENSITY`：基于答案密度的优势估计。
*   `DIVERSITY_DENSITY_HYBRID`：混合型密度估计。
*   `SELECTIVE_PASSK`：选择性 Pass@k 更新策略。
*   `ADAPTIVE_PASSK`：自适应采样策略。

## 二、 详细简化步骤

### 1. 修改 `verl/trainer/ppo/ray_trainer.py`
这是逻辑最集中的地方，需要进行以下清理：
*   **AdvantageEstimator 类**：删除这四个枚举成员。
*   **compute_advantages 函数**：
    *   移除 `diversity_density_config: dict = None` 参数定义及其相关的 `epsilon` 特权逻辑。
    *   彻底删除 `DIVERSITY_DENSITY`、`DIVERSITY_DENSITY_HYBRID`、`SELECTIVE_PASSK` 和 `ADAPTIVE_PASSK` 的所有 `elif` 分支。
*   **_update_policy 函数**：
    *   移除内部 `diversity_density_config` 的构建逻辑（包含从 `self.config.algorithm` 读取 `diversity_density_k` 等操作）。
    *   移除 `train/diversity_density_ratio` 等指标的 TensorBoard/WandB 记录逻辑。
*   **Validator/Safety Checks**：移除所有针对这些 estimator 的依赖项检查（如对 `answer_types` 或 `consistency_rate` 的强制要求逻辑）。

### 2. 修改 `verl/trainer/ppo/core_algos.py`
*   **函数移除**：彻底删除 `compute_diversity_density_advantage_from_prompts` 函数及其内部所有的密度估计和混合概率计算代码。
*   **辅助逻辑清理**：移除为上述函数专门引入的、未被其他成熟 Estimator（如 GRP0、GAE）引用的数学辅助工具或临时状态变量。

### 3. 修改 `verl/trainer/config/ppo_trainer.yaml`
*   **参数配置**：
    *   移除 `diversity_density_fallback`、`diversity_density_k`、`diversity_density_use_metric` 等配置键值对。
    *   移除 `selective_passk_threshold` 的设定。
*   **注释更新**：清理 `adv_estimator` 列表中的过时说明，防止用户在配置时选择已失效的选项。

### 4. 检查其他潜在关联文件
*   **main_ppo.py**：经检查，该文件主要负责流程调度和 Ray 环境初始化，不直接包含上述 symbols，但需确保其通过 Hydra 加载的 Config 在经过 `ppo_trainer.yaml` 修改后依然能正确 Resolve。
*   **__init__.py**：如果这四个 estimator 被作为模块成员导出，需同步清理。

## 三、 修改原则

*   **平滑迁移**：参考 POS 版本中已验证的简化范围，但在操作时严格确保不破坏 `RayPPOTrainer` 的核心状态机和主流算法（GRP0, RLOO, GAE）。
*   **代码隔离**：不建议对底层 DataProto 协议做破坏性改动，仅移除逻辑层的消费端。

## 四、 总结
本次清理旨在移除约 250+ 行“僵尸代码”，消除 RayTrainer 运行循环中不必要的条件分支判断和指标记录，使整个 PPO 训练链条更加透明、高效。