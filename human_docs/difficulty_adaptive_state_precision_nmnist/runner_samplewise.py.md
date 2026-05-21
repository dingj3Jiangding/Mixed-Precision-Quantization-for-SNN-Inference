# `difficulty_adaptive_state_precision_nmnist/runner_samplewise.py`

## 作用

- 保留 N-MNIST 上最初的 **sample-wise adaptive** 实现。
- 这份文件是归档版，用来复现实验早期结果，不作为当前默认入口。

## 方法口径

adaptive 路径采用 **sample-wise timestep routing**：

- 前若干 timestep 用高 state bits 做 warmup
- 根据 early confidence / margin 为每个样本单独判定 easy / hard
- 后续 timestep 在同一个 batch 内按样本分别使用低 bits / 高 bits

这是一种 state-side proxy experiment，不直接改神经元内部状态更新方程。

## 与当前默认版本的区别

- `runner.py` / `runner_two_stage.py`：当前版本，采用 **batch-wise two-stage mode**
- `runner_samplewise.py`：旧版，采用 **sample-wise** 路由

因此：

- 旧结果和 two-stage 新结果不要直接混写
- 如果做表格，建议明确标成 `SampleWise` 与 `TwoStage`

## 输出

输出文件结构与当前版本一致，主要包括：

- `state_cost_proxy.csv`
- `comparison.csv`
- `fixed_low_state_epoch_metrics.csv`
- `adaptive_state_epoch_metrics.csv`
- `summary.json`
