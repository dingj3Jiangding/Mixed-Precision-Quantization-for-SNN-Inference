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

# 推荐命令

python scripts/run_allocation_aware_kd_nmnist.py \
  --checkpoint-path outputs/baseline_nmnist/fp32_best.pt \
  --data-root baseline_nmnist/data \
  --output-dir outputs/allocation_aware_kd_nmnist \
  --bits 8,4 \
  --weight-allocation-csv outputs/baseline_nmnist_hessian_sensitivity_formal/bit_allocation.csv \
  --t-steps 16 \
  --batch-size-train 32 \
  --batch-size-test 64 \
  --max-test-batches 0 \
  --quant-epochs 5 \
  --quant-lr 1e-4 \
  --quant-weight-decay 5e-4 \
  --distill-alpha 0.5 \
  --distill-temperature 2.0 \
  --feature-distill-beta 0.1 \
  --severity-power 1.0 \
  --seed 42 \
  --device cuda
