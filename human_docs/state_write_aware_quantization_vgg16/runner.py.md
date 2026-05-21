# `state_write_aware_quantization_vgg16/runner.py`

## 作用

- 实现 CIFAR-10 `baseline_vgg16` 上的 **State-Write-Aware Quantization** 原型实验。
- 保持与旧方法目录分离。
- 使用 state-side proxy 口径：量化并条件复用 `LIF` 节点的内部 state proxy（`module.v`），而不是只比较节点输入。

## 方法口径

比较 3 组：

- `FP32State`
- `FixedStateB*`
- `StateWriteAwareB*`

write-aware 机制：

1. 前 `write_warmup_steps` 步强制写入
2. 后续若相邻量化输入的相对变化小于 `relative_delta_threshold`
3. 则复用上一时刻缓存的量化 state proxy，并记为跳过写回

## 主要输出

- `state_cost_proxy.csv`
- `layer_write_ratio.csv`
- `comparison.csv`
- `fixed_state_epoch_metrics.csv`
- `state_write_aware_epoch_metrics.csv`
- `summary.json`

## 关键指标

- `write_ratio`
- `estimated_state_write_bytes_per_sample`
- `test_acc`
