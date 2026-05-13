# 文件：`baseline_full/config.py`

## 作用
- 定义 `BaselineFullConfig`，保存 paper-like baseline 的训练、评估、数据和设备参数。
- 默认输出目录为 `outputs/baseline_full`，避免覆盖旧 `baseline/` 实验结果。

## 如何使用
- 通常由 `scripts/run_baseline_full.py`、`scripts/run_uniform_quant_full.py` 和 `scripts/run_hessian_sensitivity_full.py` 自动创建。
- 也可以在 Python 中直接构造：
  - `cfg = BaselineFullConfig(epochs=10, device="cuda")`

## 关键字段
- `data_root`：CIFAR-10 数据目录。
- `output_dir`：输出目录。
- `epochs`、`t_steps`、`lr`、`weight_decay`：训练配置。
- `max_train_batches`、`max_test_batches`：smoke test 或截断实验用。
