# 文件：`outputs/baseline/epoch_metrics.csv`

## 作用
- 记录每个 epoch 的核心指标。
- 用于后续画图、做表格和比较实验版本。

## 如何使用
- 关键列：
  - `train_acc`, `test_acc`
  - `spike_rate`
  - `model_size_mb`
  - `sop_proxy`
- 典型用途：
  - 用 pandas 读取后绘制精度-资源曲线
  - 作为后续 uniform quantization / mixed-precision 对比基线
