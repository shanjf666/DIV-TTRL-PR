# 🎯 DIV-TTRL: Diversity-Enhanced Test-Time Reinforcement Learning

## 🧭 Overview

DIV-TTRL 是一个基于多样性增强的测试时强化学习框架，用于在无标签数据上提升大语言模型的数学推理能力。

### 🔑 核心创新

- **多样性密度优势 (Diversity Density Advantage)**: 基于超几何分布的优势估计，保护少数派答案
- **Pass@k 重加权 GRPO**: 基于 pass@k 概率的优势重加权机制
- **混合优势估计 (Hybrid Advantage)**: 根据自洽性动态选择优势计算方法
  - 高自洽性 → Pass@k GRPO 或 Pass@1 GRPO (信任多数投票)
  - 低自洽性 → Diversity Density (鼓励探索)

## 📁 项目结构

```
DIV-TTRL/
└── verl/
    ├── examples/labelfree/      # 训练脚本
    │   ├── math.sh              # 主训练脚本
    │   └── ttrl_baseline.sh     # TTRL baseline
    ├── verl/
    │   ├── trainer/ppo/
    │   │   ├── ray_trainer.py   # 训练器 (含优势估计逻辑)
    │   │   └── core_algos.py    # 核心算法 (GRPO, Diversity Density)
    │   └── workers/reward_manager/
    │       └── diversity_reward.py  # 多样性奖励管理器
    └── data/                    # 数据集
```

## 🚀 快速开始

### 1. 安装

```bash
cd verl
pip install -e .
pip install antlr4-python3-runtime==4.9.3
pip install numpy==1.26.4
```

### 2. 准备数据

```bash
cd data
python preprocess_simplerl.py
```

### 3. 训练

```bash
# DIV-TTRL 训练 (使用多样性密度混合优势)
bash examples/labelfree/math.sh \
    --task math_train \
    --backbone /path/to/Qwen3-4B-Base \
    --clip-high \
    --ent 0.003

# TTRL Baseline
bash examples/labelfree/ttrl_baseline.sh --task math_train
```

## ⚙️ 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--task` | `AMC` | 训练任务 (math_train, AIME, AMC) |
| `--backbone` | `Qwen3-4B-Base` | 基座模型路径 |
| `--clip-high` | 0.28 | 高 PPO clip ratio |
| `--ent` | 0.000 | 熵正则化系数 |
| `--temp` | 1.0 | 生成温度 |

## 📊 奖励系统设计

### 三阶段流程

```
1. Base Reward (多数投票)
   ↓
2. Diversity Adjustment (多样性调整) 
   ↓  
3. Advantage Estimation (优势估计)
   - Diversity Density: 基于答案类型分布
   - Pass@k GRPO: 基于多数投票奖励
```

### 混合优势公式

```
p = 自洽性比率 (majority_count / total_samples)

if random() > p:
    advantage = diversity_density_advantage  # 低自洽性
else:
    advantage = pass_grpo_advantage          # 高自洽性
```

## 📈 支持的数据集

- **Math Training**: MATH 训练集
- **AIME-TTT**: AIME 2024 竞赛题
- **AMC-TTT**: AMC 竞赛题
- **MATH-TTT**: MATH-500 测试集
- **AIME25-TTT**: 训练集是 AIME 2022-2024 竞赛题，测试集是AIME 2025 竞赛题

## 🤖 支持的模型

- Qwen3-4B-Base
- Qwen3-8B-Base

## 📝 License

Apache License 2.0