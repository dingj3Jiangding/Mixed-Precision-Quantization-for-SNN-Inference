# 文件：`scripts/run_hessian_sensitivity.py`

## 作用
- Hessian trace 敏感度分析、混合精度分配和 post-quantization fine-tuning 的命令行入口。
- 基于 FP32 checkpoint 运行 Hutchinson Hessian trace 估计、ranking-based 位宽分配、mixed-precision fine-tuning 和对比评估。

## 如何使用
- 在仓库根目录执行：
  - `python scripts/run_hessian_sensitivity.py --checkpoint-path outputs/baseline/fp32_last.pt --bits 8,4,2 --trace-probes 1 --quant-epochs 1`

## 常用参数
- `--checkpoint-path`：FP32 模型权重路径
- `--bits`：候选位宽列表，逗号分隔
- `--trace-probes`：每层 Hessian trace 的 Hutchinson probe 数量
- `--max-hessian-batches`：Hessian trace 估计时限制 batch 数（调试加速）
- `--allocation-policy`：位宽分配策略，支持 `rank-map` 和 `tiered`
- `--quant-epochs`：mixed-precision fine-tuning 轮数
- `--quant-lr`：fine-tuning 学习率
- `--quant-weight-decay`：fine-tuning weight decay
- `--max-train-batches`：fine-tuning 时限制训练 batch 数（调试加速）
- `--max-test-batches`：评估时限制 batch 数（调试加速）
- `--target-avg-bits`：兼容旧脚本的参数，现在只用于选择 uniform reference bit

## 输出位置
- 默认输出到 `outputs/hessian_sensitivity/`
- 关键文件：
  - `layer_sensitivity.csv`
  - `bit_allocation.csv`
  - `comparison.csv`
  - `quant_finetune_epoch_metrics.csv`
  - `summary.json`
