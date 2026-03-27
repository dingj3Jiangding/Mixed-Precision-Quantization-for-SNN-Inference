# 文件：`baseline/unquant_runner.py`

## 作用
- 提供未量化（FP32）baseline 的端到端训练与评估主流程。
- 负责随机种子设置、训练循环、测试评估、指标汇总与结果落盘。

## 如何使用
- 代码调用：
  - `summary = run_baseline(cfg)`
- 常用入口：
  - `python scripts/run_baseline.py --epochs 10 --device cuda`

## 主要输出
- `outputs/baseline/epoch_metrics.csv`
- `outputs/baseline/summary.json`
- `outputs/baseline/fp32_last.pt`（供 uniform quant 脚本加载）

## 关键函数
- `set_global_seed`：控制可复现性
- `train_one_epoch`：单轮训练
- `evaluate`：单轮评估（含 `spike_rate` 与时延统计）
- `run_baseline`：完整训练评估并导出结果
