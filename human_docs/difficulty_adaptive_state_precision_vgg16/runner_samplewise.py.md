# `difficulty_adaptive_state_precision_vgg16/runner_samplewise.py`

## 作用

- 保留 CIFAR-10 VGG16 上最初的 **sample-wise adaptive** 实现。
- 这份文件是归档版，用于复现旧结果，不是当前默认 runner。

## 方法结构

比较 4 组状态精度设定：

- `FP32State`
- `FixedStateHighB*`
- `FixedStateLowB*`
- `AdaptiveStateB*to*`

其中 adaptive 路径采用 **sample-wise timestep routing**：

1. 前 `warmup_steps` 步统一使用高 state bits
2. 根据 early logits 的 confidence / margin 判断每个样本难度
3. 同一 batch 内 easy 样本切到低 bits
4. hard 样本继续保留高 bits

## 与当前默认版本的区别

- `runner.py` / `runner_two_stage.py`：当前版本，采用 **batch-wise two-stage mode**
- `runner_samplewise.py`：旧版，采用 **sample-wise** 路由

这两套结果应分开报告。

## 输出指标

主要输出仍包括：

- `avg_state_bits_used`
- `easy_fraction_post_warmup`
- `estimated_state_bytes_per_sample`
- `summary.json`
