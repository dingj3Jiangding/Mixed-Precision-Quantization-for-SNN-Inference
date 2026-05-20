# `difficulty_adaptive_state_precision_vgg16/runner.py`

## 作用

- 实现 CIFAR-10 `baseline_vgg16` 上的 **Difficulty-Adaptive State Precision** 原型实验。
- 保持与旧的 `baseline_vgg16` 方法目录分离。
- 使用与 `weight_state_mixed` 一致的 proxy 口径：量化每个 LIF 节点的输入，而不是直接改神经元内部膜电位更新。

## 方法结构

比较 4 组状态精度设定：

- `FP32State`
- `FixedStateHighB*`
- `FixedStateLowB*`
- `AdaptiveStateB*to*`

其中 adaptive 策略是：

1. 前 `warmup_steps` 步全部使用高 state bits
2. 根据前若干步的累计输出估计样本难度
3. easy samples 在后续步切换到低 bits
4. hard samples 保持高 bits

## 难度判断

支持两种 difficulty signal：

- `confidence`
- `margin`

默认使用：

- `confidence`
- `easy_threshold = 0.7`

## 主要输出

默认输出目录：

```text
outputs/difficulty_adaptive_state_precision_vgg16/
```

主要文件：

- `state_cost_proxy.csv`
- `comparison.csv`
- `fixed_low_state_epoch_metrics.csv`
- `adaptive_state_epoch_metrics.csv`
- `summary.json`

## 输出指标

除准确率外，还会记录：

- `avg_state_bits_used`
- `easy_fraction_post_warmup`
- `estimated_state_bytes_per_sample`

这些指标用于体现 state-side resource tradeoff。
