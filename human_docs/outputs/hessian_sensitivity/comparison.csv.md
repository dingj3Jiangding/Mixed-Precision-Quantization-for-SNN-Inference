# 文件：`outputs/hessian_sensitivity/comparison.csv`

## 作用
- 汇总 `FP32`、`UniformWk`、`HessianMixed` 三个设置的同口径指标。
- 用于生成第一版核心对比图和表格。

## 关键字段
- `setting`
- `test_acc`
- `spike_rate`
- `avg_batch_infer_ms`
- `sop_proxy`

## 如何使用
- 直接用于绘制 accuracy-resource 对比图。
- 作为 checkpoint 汇报中的核心结果表。
