# `state_write_aware_quantization_nmnist/runner.py`

## 作用

- 实现 N-MNIST 上的 **State-Write-Aware Quantization** 原型实验。
- 不直接修改神经元内部 membrane update，而是近似为：
  - 输入量化
  - 条件复用上一时刻量化值
  - 显式统计写回比例

## 输出

- `state_cost_proxy.csv`
- `layer_write_ratio.csv`
- `comparison.csv`
- `fixed_state_epoch_metrics.csv`
- `state_write_aware_epoch_metrics.csv`
- `summary.json`
