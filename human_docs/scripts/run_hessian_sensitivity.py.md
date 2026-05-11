# 文件：`scripts/run_hessian_sensitivity.py`

## 作用
- Hessian 敏感度分析与混合精度分配的命令行入口。
- 基于 FP32 checkpoint 运行层级敏感度估计、位宽分配和对比评估。

## 如何使用
- 在仓库根目录执行：
  - `python scripts/run_hessian_sensitivity.py --checkpoint-path outputs/baseline/fp32_last.pt --bits 8,4,2 --target-avg-bits 4`

## 常用参数
- `--checkpoint-path`：FP32 模型权重路径
- `--bits`：候选位宽列表，逗号分隔
- `--target-avg-bits`：目标平均位宽
- `--max-hessian-batches`：敏感度估计时限制 batch 数（调试加速）
- `--max-test-batches`：评估时限制 batch 数（调试加速）

## 输出位置
- 默认输出到 `outputs/hessian_sensitivity/`
- 关键文件：
  - `layer_sensitivity.csv`
  - `bit_allocation.csv`
  - `comparison.csv`
  - `summary.json`
