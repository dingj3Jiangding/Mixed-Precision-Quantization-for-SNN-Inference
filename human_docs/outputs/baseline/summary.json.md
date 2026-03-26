# 文件：`outputs/baseline/summary.json`

## 作用
- 保存一次运行的摘要信息。
- 包含配置快照、设备、最佳精度、最后一轮指标与输出文件路径。

## 如何使用
- 用于复现实验：
  - 根据 `config` 字段还原命令参数。
- 用于汇总报告：
  - 直接读取 `best_test_acc` 和 `final_epoch`。
- 典型读取方式：
  - Python: `json.load(open('outputs/baseline/summary.json'))`
