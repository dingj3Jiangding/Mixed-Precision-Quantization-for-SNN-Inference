# `difficulty_adaptive_state_precision_nmnist/runner.py`

## 作用

- 实现 N-MNIST 上的 **Difficulty-Adaptive State Precision** 原型实验。
- 复用 N-MNIST 的现有数据加载和模型定义，但不混入旧方法目录。

## 方法口径

与 CIFAR-10 版本一致：

- 前若干 timestep 用高 state bits 做 warmup
- 依据 early confidence / margin 判断样本难度
- easy samples 用低 bits
- hard samples 继续用高 bits

这是一个 state-side proxy experiment，不直接改神经元内部状态更新方程。

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
