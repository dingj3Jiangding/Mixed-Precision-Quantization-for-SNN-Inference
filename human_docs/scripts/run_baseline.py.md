# 文件：`scripts/run_baseline.py`

## 作用
- baseline 的命令行入口脚本。
- 负责解析参数，构建 `BaselineConfig`，并调用 `run_baseline`。

## 如何使用
- 在仓库根目录运行：
  - `python scripts/run_baseline.py`
- 常用参数示例：
  - `python scripts/run_baseline.py --epochs 20 --t-steps 16 --seed 42`
  - `python scripts/run_baseline.py --max-train-batches 20 --max-test-batches 10`（快速调试）
  - `python scripts/run_baseline.py --device cuda`
- 运行后会打印 summary JSON，并在输出目录写入 csv/json 指标文件。
