# 文件：`scripts/run_uniform_quant.py`

## 作用
- 作为 Uniform Quantization 对比实验的命令行入口。
- 基于 baseline checkpoint 运行 `FP32/W8/W4/W2` 评估并导出结果。

## 如何使用
- 在仓库根目录执行：
  - `python scripts/run_uniform_quant.py --checkpoint-path outputs/baseline/fp32_last.pt --bits 8,4,2`

## 常用参数
- `--checkpoint-path`：baseline 模型权重路径
- `--bits`：量化位宽列表，逗号分隔，如 `8,4,2`
- `--device`：`auto/cuda/cpu/mps`
- `--max-test-batches`：调试时限制测试 batch 数

## 输出
- 结果目录默认 `outputs/uniform_quant/`
- 主要文件：
  - `uniform_comparison.csv`
  - `summary.json`
