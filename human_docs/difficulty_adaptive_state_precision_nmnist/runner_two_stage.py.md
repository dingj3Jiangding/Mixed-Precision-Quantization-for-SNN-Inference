# `difficulty_adaptive_state_precision_nmnist/runner_two_stage.py`

## 作用

- 保留 N-MNIST 上当前默认的 **two-stage** 实现。
- 内容与 `runner.py` 一致，便于显式区分 two-stage 与 sample-wise 两套版本。

## 方法口径

adaptive 路径采用 **batch-wise two-stage mode**：

- 前若干 timestep 用高 state bits 做 warmup
- 依据 early confidence / margin 判断样本难度
- 将 batch 内样本分成 easy / hard 两组
- easy group 用低 bits
- hard group 用高 bits

这是当前建议使用的版本。它仍然是一个 state-side proxy experiment，不直接改神经元内部状态更新方程。

## 输出

默认输出目录：

```text
outputs/difficulty_adaptive_state_precision_nmnist/
```

主要结果文件：

- `state_cost_proxy.csv`
- `comparison.csv`
- `fixed_low_state_epoch_metrics.csv`
- `adaptive_state_epoch_metrics.csv`
- `summary.json`
