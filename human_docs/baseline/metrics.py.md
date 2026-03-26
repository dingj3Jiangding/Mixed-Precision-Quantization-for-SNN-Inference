# 文件：`baseline/metrics.py`

## 作用
- 提供 baseline 的指标工具：
  - `SpikeRateTracker`：统计全模型脉冲发放率
  - `parameter_count`：参数量
  - `model_size_mb`：参数占用大小（MB）
  - `synapse_count_proxy`：突触数量代理指标（conv/linear 权重数）
  - `sop_proxy`：SOP 代理指标（`spike_rate * synapse_count * T`）

## 如何使用
- 评估前创建跟踪器：
  - `tracker = SpikeRateTracker(model)`
- 批处理完成后读取：
  - `rate = tracker.rate()`
  - `tracker.close()`
- 其余函数按需直接调用：
  - `params = parameter_count(model)`
  - `size = model_size_mb(model)`
