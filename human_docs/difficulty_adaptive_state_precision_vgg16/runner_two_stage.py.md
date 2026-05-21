# `difficulty_adaptive_state_precision_vgg16/runner_two_stage.py`

## 作用

- 保留 CIFAR-10 `baseline_vgg16` 上当前默认的 **two-stage** 实现。
- 内容与 `runner.py` 一致，便于和 `runner_samplewise.py` 明确区分。
- 使用与 `weight_state_mixed` 一致的 proxy 口径：量化每个 LIF 节点的输入，而不是直接改神经元内部膜电位更新。

## 方法结构

比较 4 组状态精度设定：

- `FP32State`
- `FixedStateHighB*`
- `FixedStateLowB*`
- `AdaptiveStateB*to*`

其中 adaptive 策略采用 **batch-wise two-stage mode**：

1. 前 `warmup_steps` 步全部使用高 state bits
2. 根据前若干步的累计输出估计样本难度
3. 将 batch 内样本划分为 easy / hard 两组
4. 后续步分别以低 bits / 高 bits 对两组样本执行

当前实现中，two-stage adaptive fine-tuning 会临时关闭时序 dropout，并冻结 batch normalization 的 training mode。原因是子 batch 执行会破坏原始 batch 级别的 dropout mask 与 batchnorm running statistics。

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
