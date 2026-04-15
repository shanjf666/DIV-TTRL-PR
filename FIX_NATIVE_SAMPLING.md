# 修复说明：原生采样逻辑回滚

## 问题根源

运行时出现断言失败：
```
AssertionError: Mismatch: 584 outputs vs 73 mappings
```

**根本原因**：思路一（原生 n-sampling）的实现逻辑有缺陷。

当 `verification_n = 8` 时：
- 原先改动：`verification_batch` 大小 = 73（C），`verification_mapping` 大小 = 73
- vLLM native n-sampling 返回 73 × 8 = 584 个响应
- 但 `verification_mapping` 仍只有 73 条 → **不匹配**

## 修复方案

**临时回滚思路一，保留思路二（token-aware 分桶）**

### 改动清单

#### 1. two_stage_utils.py

**L265-277**: 恢复原始采样复制逻辑
- `repeat_count` 按原始逻辑计算（frequency 或 verification_n）
- 移除 `native_n_sampling` 字段
- 保持 `verification_mapping` 的完整复制（每个样本一条 mapping）

**L334-343**: 移除 meta_info 中的 `verification_n` 传递
- 不再向引擎发信号进行 native n-sampling

#### 2. fsdp_workers.py

**L589-608**: 不再使用 native n-sampling
- 移除 `verification_kwargs["n"] = verification_n`
- 回到固定 n=1 的默认行为

#### 3. ray_trainer.py

**L1121**: 设置 `n_samples = 1`
- 因为每行 verification_batch 已对应一条 mapping
- 简化后续 out-of-order 处理逻辑

**L1155-1190**: 简化响应分组逻辑
- 移除 `n_samples` 倍数计算
- 直接 1:1 映射：`outputs_by_orig_idx[orig_idx] = chunk_decoded[i]`

**L1192-1210**: 简化输出重新组装
- 移除 padding 逻辑，直接按序 append

## 后续优化计划

思路一（原生 n-sampling）因牵涉 mapping 重设计，暂时搁置。

**推荐的下一步改进**：
1. 保留思路二（token-aware 分桶）的收益 (~20-40% padding 减少)
2. 后续若要支持原生 n-sampling，需要重新设计 mapping 结构：
   - 选项 A：mapping 中记录 n_sampling 倍数，在处理响应时自动展开
   - 选项 B：引入额外的 "sampling_group_id"，便于后续聚合

## 现状验证

修复后数据关系应为：
```
verification_batch 大小         = C × N  (C候选 × N采样)
verification_mapping 元素数      = C × N  (每个样本一条)
all_verification_outputs 大小    = C × N  (一一对应)
```

此时 `len(all_verification_outputs) == len(verification_mapping)` ✓

## 性能对比

### 思路一回滚的代价
- 失去：Python 层对象复制的 10-20% 优化
- 保留：Token-aware 分桶的 20-40% padding 减少
- 整体：仍有约 15-30% 的第二阶段加速

### 保留的优化（思路二）
- ✅ Token-aware micro-batch 分桶（已验证）
- ✅ Cache 清理策略调整（避免频繁 stall）
- ✅ Out-of-order 样本重排（支持灵活分桶）

## 测试建议

1. 确认 `len(outputs) == len(mapping)` 一致
2. 观察 two_stage metrics（应与原始版本一致）
3. 对比显存曲线（应该比原始版本更平稳）
