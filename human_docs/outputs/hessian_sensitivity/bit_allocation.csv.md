# 文件：`outputs/hessian_sensitivity/bit_allocation.csv`

## 作用
- 保存最终 layer-wise 位宽分配表。
- 作为后续 mixed-precision 复现实验的直接配置来源。

## 关键字段
- `layer_name`
- `assigned_bits`
- `params`
- `sensitivity_score`

## 如何使用
- 按 `layer_name -> assigned_bits` 构建量化策略。
- 可与 `uniform` 结果做同预算对比。
