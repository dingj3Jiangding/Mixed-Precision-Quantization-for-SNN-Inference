# `allocation_aware_kd_vgg16/runner.py`

## 作用

- 实现 CIFAR-10 `baseline_vgg16` 上的 **allocation-aware KD** 实验。
- 保持与原始 `baseline_vgg16/distill_quant.py` 分离，不混入旧方法实现。
- 在 logits KD 的基础上，新增一个由 mixed-precision allocation 驱动的特征蒸馏项。

## 方法要点

当前实现包含三部分损失：

```text
L = (1 - alpha) * CE + alpha * KD + beta * FeatureKD_alloc
```

其中：

- `CE`：student 对真实标签的分类损失
- `KD`：teacher-student logits distillation
- `FeatureKD_alloc`：对选定层的中间特征做蒸馏
- `beta`：控制 allocation-aware feature distillation 的强度

## allocation-aware 的具体做法

不是对所有层一视同仁，而是先根据已有 mixed-precision allocation 计算每一层的 `severity`：

- bit 越低，severity 越高
- 对 low-bit 层内部，优先按 `trace_density`（不可用时退化到 `hessian_trace`）比较敏感度
- 为避免被全网最敏感早期层“压扁”，severity 只在实际 low-bit 层集合内部归一化
- 同时引入层内 sensitivity rank，避免多个 low-bit 层的权重几乎相同
- 最终只对有量化压力的层施加更强的 feature matching

输出里会额外保存：

- `layer_severity.csv`

## 主要输入

- `checkpoint_path`
- `bits_list`
- `weight_allocation_csv`（可选）
- `distill_alpha`
- `distill_temperature`
- `feature_distill_beta`
- `severity_power`

## 主要输出

输出目录默认：

```text
outputs/allocation_aware_kd_vgg16/
```

生成内容包括：

- `layer_sensitivity.csv`
- `bit_allocation.csv`
- `layer_severity.csv`
- `comparison.csv`
- `allocation_aware_kd_epoch_metrics.csv`
- `summary.json`

## 结果含义

`comparison.csv` 中会至少包含：

- `FP32`
- `UniformW*`
- `AllocAwareKD_*`

用于和原始 HessianMixed / plain KD 结果做对照。

# 推荐命令

python scripts/run_allocation_aware_kd_vgg16.py \
  --checkpoint-path outputs/baseline_vgg16/fp32_last.pt \
  --data-root baseline/data \
  --output-dir outputs/allocation_aware_kd_vgg16 \
  --bits 8,4 \
  --weight-allocation-csv outputs/baseline_vgg16_hessian_sensitivity_formal/bit_allocation.csv \
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
