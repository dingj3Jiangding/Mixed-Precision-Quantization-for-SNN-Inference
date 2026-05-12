# 文件：`baseline/hessian.py`

## 作用
- 实现更接近 Lui & Neftci 2021 方法逻辑的 Hessian-aware mixed-precision 主流程。
- 使用 Hutchinson probe 和 Hessian-vector product 估计每个 Conv/Linear 层的 Hessian trace。
- 根据 Hessian trace 排名做保守的 layer-wise bit 分配，而不是旧的平均 bit 预算贪心分配。
- 对 mixed-precision 模型做量化感知 fine-tuning，再对比 `FP32`、`Uniform`、`HessianMixed` 三种设置。

## 如何使用
- 主函数：
  - `run_hessian_sensitivity_analysis(cfg, checkpoint_path, bits_list, output_dir, trace_probes, quant_epochs, allocation_policy)`
- 必要输入：
  - `checkpoint_path`：已训练 FP32 权重（例如 `outputs/baseline/fp32_last.pt`）
  - `bits_list`：可选位宽（例如 `[8, 4, 2]`）
- 常用控制：
  - `trace_probes`：每层 Hessian trace 的 Hutchinson probe 数量
  - `max_hessian_batches`：trace 估计使用的 batch 数
  - `allocation_policy`：`rank-map` 或 `tiered`
  - `quant_epochs` / `quant_lr` / `quant_weight_decay`：mixed-precision fine-tuning 设置
  - `target_avg_bits`：仅保留为兼容参数，用来选择 uniform reference bit，不再驱动 mixed allocation

## 输出文件
- `layer_sensitivity.csv`：每层 Hessian trace、trace density、排名、分配位宽
- `bit_allocation.csv`：最终 layer-wise 位宽分配表
- `comparison.csv`：`FP32 / Uniform / HessianMixed` 指标对比，其中 `HessianMixed` 是 fine-tuning 后结果
- `quant_finetune_epoch_metrics.csv`：mixed-precision fine-tuning 每轮训练/测试指标
- `summary.json`：本次运行摘要
- `sensitivity_ranking.png`：Hessian trace 排名图（若 matplotlib 可用）

## 注意
- 当前实现只做 weight-side Hessian-aware quantization，没有实现论文中的 state-variable quantization。
- 当前实验栈仍是本仓库的 CIFAR-10 + SpikingJelly direct encoding，不是 Lui & Neftci 原始的 N-MNIST / DECOLLE 设置。
