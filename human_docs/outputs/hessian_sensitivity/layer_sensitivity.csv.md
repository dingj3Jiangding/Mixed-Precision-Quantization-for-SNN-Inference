# 文件：`outputs/hessian_sensitivity/layer_sensitivity.csv`

## 作用
- 保存每个可量化层的 Hessian 近似敏感度结果。
- 用于层排序、热力图绘制和位宽分配决策。

## 关键字段
- `rank`
- `layer_name`
- `params`
- `avg_grad2`
- `sensitivity_score`
- `score_density`
- `assigned_bits`

## 如何使用
- 按 `sensitivity_score` 从高到低查看高敏层。
- `assigned_bits` 可直接作为 mixed-precision 配置输入。
