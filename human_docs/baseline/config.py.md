# 文件：`baseline/config.py`

## 作用
- 定义 `BaselineConfig` 配置结构（dataclass）。
- 管理 baseline 训练/评估所需的核心超参数与路径。
- 提供 `resolve_device()` 自动设备选择逻辑（`cuda -> mps -> cpu`）。

## 如何使用
- 代码中创建配置对象：
  - `cfg = BaselineConfig(epochs=10, t_steps=16, seed=42)`
- 传给主流程：
  - `summary = run_baseline(cfg)`
- 常用参数：
  - `data_root`：数据目录
  - `output_dir`：输出指标目录
  - `max_train_batches/max_test_batches`：快速调试时限制 batch 数
