# `allocation_aware_kd_nmnist/runner.py`

## 作用

- 实现 N-MNIST 数据集上的 **allocation-aware KD** 实验。
- 保持与原始 `baseline_nmnist/distill_quant.py` 分离。
- 复用现有 N-MNIST 数据加载、模型和 mixed-precision allocation 工具。

## 方法结构

损失函数形式与 CIFAR-10 版本一致：

```text
L = (1 - alpha) * CE + alpha * KD + beta * FeatureKD_alloc
```

其中：

- `KD` 是 logits distillation
- `FeatureKD_alloc` 是根据 layer-wise allocation severity 加权的中间特征蒸馏

## 输入

- `checkpoint_path`
- `bits_list`
- `weight_allocation_csv`
- `distill_alpha`
- `distill_temperature`
- `feature_distill_beta`
- `severity_power`

## 输出

默认输出目录：

```text
outputs/allocation_aware_kd_nmnist/
```

主要产物：

- `layer_sensitivity.csv`
- `bit_allocation.csv`
- `layer_severity.csv`
- `comparison.csv`
- `allocation_aware_kd_epoch_metrics.csv`
- `summary.json`

## 说明

N-MNIST 版本用于 supporting experiment。
如果该方法在 CIFAR-10 主线有效，而 N-MNIST 也给出一致趋势，就能增强其泛化性说服力。
