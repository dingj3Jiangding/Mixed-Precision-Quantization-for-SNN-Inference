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
- Hessian trace 越大，severity 越高
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
