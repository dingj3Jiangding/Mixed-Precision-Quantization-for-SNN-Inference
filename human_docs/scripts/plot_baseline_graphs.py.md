# 文件：`scripts/plot_baseline_graphs.py`

## 作用
- 从 `outputs/baseline/epoch_metrics.csv` 读取训练结果。
- 自动生成 baseline 常用可视化图，输出到 `outputs/baseline/figures/`。

## 输入与输出
- 输入文件：
  - `outputs/baseline/epoch_metrics.csv`
- 输出目录：
  - `outputs/baseline/figures`
- 输出图片：
  - `accuracy_vs_epoch.png`
  - `loss_vs_epoch.png`
  - `spike_rate_vs_epoch.png`
  - `infer_time_vs_epoch.png`
  - `accuracy_vs_sop_proxy.png`（仅当 csv 含 `sop_proxy` 列时）

## 如何使用
- 在仓库根目录执行：
  - `python scripts/plot_baseline_graphs.py`
- 成功后终端会打印：
  - `Done. Figures saved to: outputs/baseline/figures`

## 依赖
- `pandas`
- `matplotlib`

## 注意事项
- 运行前先确保 baseline 已完成训练并产出 `epoch_metrics.csv`。
- 若缺少 `sop_proxy` 列，脚本会跳过资源-精度散点图，仅生成前四张图。
