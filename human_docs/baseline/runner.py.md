# 文件：`baseline/runner.py`

## 作用
- 提供 baseline 端到端执行主流程：
  - 设定随机种子
  - 训练一个或多个 epoch
  - 测试并统计精度/脉冲率/时延代理指标
  - 输出 `epoch_metrics.csv` 与 `summary.json`

## 如何使用
- 直接调用：
  - `summary = run_baseline(cfg)`
- 关键函数：
  - `set_global_seed`：可复现控制
  - `train_one_epoch`：单轮训练
  - `evaluate`：单轮评估并统计 spike rate
- 输出文件位置：
  - 默认为 `outputs/baseline/epoch_metrics.csv`
  - 默认为 `outputs/baseline/summary.json`
