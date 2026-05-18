# 文件：`baseline_vgg16/metrics.py`

## 作用
- 复用旧 baseline 的指标工具。
- 保持 `test_acc`、`spike_rate`、`avg_batch_infer_ms`、`sop_proxy` 等指标计算口径一致。

## 如何使用
- `from baseline_vgg16.metrics import parameter_count, sop_proxy`
- 通常由 `baseline_vgg16/unquant_runner.py`、`baseline_vgg16/uniform_runner.py` 和 `baseline_vgg16/hessian.py` 间接调用。

## 输出
- 参数量、模型大小、synapse proxy、SOP proxy 和 spike rate 统计。
