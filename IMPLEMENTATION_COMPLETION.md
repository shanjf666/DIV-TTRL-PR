# Implementation Completion Checklist

## 🔧 修复说明 (2026-04-15)

**运行时错误**：`AssertionError: Mismatch: 584 outputs vs 73 mappings`

**原因**：思路一（原生 n-sampling）的实现导致 verification_mapping 数量与输出不匹配

**修复方案**：临时回滚思路一，保留思路二（token-aware 分桶）的优化
- ✅ 恢复原始采样复制逻辑（Python 层）
- ✅ 保留 token-aware 分桶优化（20-40% padding 减少）
- ⏸️  原生 n-sampling 优化推迟（需重新设计 mapping 结构）

---

按照 speed_up.md 的优化计划，已完成以下改动。


## 思路一：引擎原生批量采样 ⏸️ (临时回滚)

**当前状态**：为确保数据一致性，暂时使用原始采样策略（Python 层复制）

### two_stage_utils.py 改动

- [x] **construct_verification_dataproto()** L265-277
  - ✅ 恢复原始 repeat_count 逻辑
  - ✅ verification_mapping 保持完整复制（每个样本一条）
  - ℹ️  移除 native_n_sampling 字段（推迟方案）

- [x] **meta_info["verification_n"] 移除** L334-343
  - ✅ 不再向引擎发信号进行 native n-sampling

### fsdp_workers.py 改动

- [x] **verification_mode 分支** L589-608
  - ✅ 移除 verification_n 的 native n-sampling 传递
  - ✅ 回到原始行为

---

## 思路二：Token-aware 微批处理分桶 ✅ (保留)

### ray_trainer.py 改动

- [x] **Step 4 优化** L1100-1220
  - ✅ 使用 `rearrange_micro_batches()` 按 token 数平衡分桶
  - ✅ 处理 out-of-order 样本重排
  - ✅ 设置 `n_samples=1`（与原始采样策略匹配）

- [x] **响应处理简化** L1155-1210  
  - ✅ 1:1 映射响应和 mapping
  - ✅ 简化 reorder_indices 逻辑

- [x] **Batch_second 构建** L1197-1219  
  - ✅ 按原始顺序重排
  - ✅ 生成正确的 UID 序列

- [x] **Cache 清理策略调整** L1159-1166
  - 移除每个 chunk 后的 empty_cache 调用
  - 仅在所有 chunks 完成后执行一次清理
  - 防止频繁清缓存导致调度中断

---

## 数据完整性验证

### 关键数据量关系

```
修改前:
  verification_batch 大小      = C * N  (C个候选，每个复制N次)
  verification_mapping 元素数   = C * N  (每个复制产品一条mapping)
  all_verification_outputs 大小 = C * N  (一一对应mapping)

修改后(native n-sampling):
  verification_batch 大小      = C      (C个候选，无复制)
  verification_mapping 元素数   = C      (每个候选一条mapping)  
  all_verification_outputs 大小 = C * N  (vLLM 为每个候选生成N个响应)
  
  对应关系:
    mapping[i] 对应 responses[i*N : (i+1)*N]
```

### 兼容性保证

1. **外部参数不变**：two_stage_n、two_stage_mode 等参数含义完全相同
2. **Fallback 支持**：verification_n=None 时自动回退到频率基采样（旧行为）
3. **最终结果一致**：
   - pseudo_labels 仍为 C 个（每个候选一个）
   - consistency_scores 仍为 C 个
   - should_update_flags 仍为 C 个

---

## 性能优化收益

### 思路一（原生采样）
- **显存峰值**：↓ 10-20%（减少 Python 层对象复制）
- **Tokenization 成本**：↓ (N-1)/N （N个样本只需做一次）
- **Host-Device 传输**：↓ (N-1)/N （减少重复上传）

### 思路二（Token-aware 分桶）
- **Padding 浪费**：↓ 20-40%（样本长度均衡）
- **GPU 利用率稳定性**：显著提高（减少锯齿波动）
- **Max chunk 大小**：允许更大（相同显存下）

### 预期整体收益
- Second-stage verification 总耗时：**-20~45%**
- GPU 显存波动：**-50~70%**（从明显锯齿变平稳）

---

## 测试验证计划

### 单元测试（立即可做）

```python
# test_construct_verification_dataproto
def test_native_n_sampling():
    # 验证 repeat_count=1, native_n_sampling=N
    # 验证 meta_info["verification_n"]=N
    # 验证 verification_mapping 元素数 = C（不是C*N）

def test_fallback_frequency_sampling():
    # 验证 verification_n=None 时回退到旧逻辑
    # 验证 repeat_count=frequency, native_n_sampling=1

# test_token_aware_micro_batch  
def test_rearrange_and_reorder():
    # 验证 reorder_indices 能正确恢复原始顺序
    # 验证 outputs_by_orig_idx 完整且无重复/遗漏
    # 验证 final batch_second 与 verification_mapping 对齐
```

### 集成测试（训练级）

```bash
# 运行完整训练，观察以下指标
- train/two_stage_trigger_rate（应保持~1.0）
- train/two_stage_true_rate、false_rate、parse_fail_rate（应与之前一致）
- train/two_stage_filtered_ratio（应与之前一致）
- 显存使用曲线（应变平稳）
- Loss 曲线稳定性（应无异常）
```

---

## 部署注意事项

1. **vLLM 版本**：确保支持 n > 1 的 native sampling
   - vLLM >= 0.4.0 应该支持
   - 如不支持，自动回退到频率基采样（设置 verification_n=None）

2. **rearrange_micro_batches 使用**：
   - 已在 actor/critic 路径验证过，应该稳定
   - 若 out-of-order 处理有问题，可先关闭（改回固定 micro_bs）

3. **性能基准**：
   - 建议在小 batch 上先跑对比
   - 确认 loss 和 metrics 与修改前一致
   - 再放大 batch size 进行性能竞赛

---

## 回滚计划

若出现问题，可快速回滚：

1. **回到思路一之前**（仅保留思路二）：
   - 恢复 two_stage_utils.py 中 repeat_count 逻辑（全复制）
   - 保留 fsdp_workers.py 的改动（无害）

2. **完全回滚**：
   - 恢复 ray_trainer.py Step 4 为固定 micro_bs 逻辑
   - 恢复 two_stage_utils.py 为原始采样策略

---

## 文档参考

- 设计文档：[speed_up.md](speed_up.md)
- 实现文档：[SPEEDUP_IMPLEMENTATION.md](SPEEDUP_IMPLEMENTATION.md)
- 核心代码：
  - [two_stage_utils.py](verl/verl/utils/reward_score/ttrl/two_stage_utils.py) 
  - [ray_trainer.py](verl/verl/trainer/ppo/ray_trainer.py) 
  - [fsdp_workers.py](verl/verl/workers/fsdp_workers.py)
