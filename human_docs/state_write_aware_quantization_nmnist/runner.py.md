# `state_write_aware_quantization_nmnist/runner.py`

## 作用

- 实现 N-MNIST 上的 **State-Write-Aware Quantization** 原型实验。
- 不直接修改神经元内部 membrane update，而是近似为：
  - `module.v` 状态 proxy 量化
  - 条件复用上一时刻缓存 state
  - 显式统计写回比例

## 输出

- `state_cost_proxy.csv`
- `layer_write_ratio.csv`
- `comparison.csv`
- `fixed_state_epoch_metrics.csv`
- `state_write_aware_epoch_metrics.csv`
- `summary.json`
