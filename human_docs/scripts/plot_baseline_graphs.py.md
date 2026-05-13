# 文件：`scripts/plot_baseline_graphs.py`

## 作用
- 从指定的 `epoch_metrics.csv` 读取训练结果。
- 自动生成 baseline 常用可视化图。
- 默认读取旧 baseline 输出，也支持通过参数给 `baseline_full` 画图。

## 输入与输出
- 默认输入文件：
  - `outputs/baseline/epoch_metrics.csv`
- 默认输出目录：
  - `outputs/baseline/figures`
- 输出图片：
  - `accuracy_vs_epoch.png`
  - `loss_vs_epoch.png`
  - `spike_rate_vs_epoch.png`
  - `infer_time_vs_epoch.png`
  - `accuracy_vs_sop_proxy.png`（仅当 csv 含 `sop_proxy` 列时）

## 如何使用
- 给旧 baseline 画图：
  - `python scripts/plot_baseline_graphs.py`
- 给新 `baseline_full` 画图：
  - `python scripts/plot_baseline_graphs.py --metrics-csv outputs/baseline_full/epoch_metrics.csv --output-dir outputs/baseline_full/figures`
- 成功后终端会打印：
  - `Done. Figures saved to: <output-dir>`

## 依赖
- `pandas`
- `matplotlib`

## 注意事项
- 运行前先确保 baseline 已完成训练并产出 `epoch_metrics.csv`。
- 若缺少 `sop_proxy` 列，脚本会跳过资源-精度散点图，仅生成前四张图。
