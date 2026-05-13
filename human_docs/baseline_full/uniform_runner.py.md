# 文件：`baseline_full/uniform_runner.py`

## 作用
- 对 `baseline_full` 的 FP32 checkpoint 做 uniform weight quantization 对比。
- 默认建议比较 `W8/W4`，避免当前阶段 `W2` 过于激进导致主结果不可用。

## 如何运行
```bash
python scripts/run_uniform_quant_full.py \
  --checkpoint-path outputs/baseline_full/fp32_last.pt \
  --bits 8,4 \
  --device cuda
```

## 输出文件
- `outputs/baseline_full_uniform_quant/uniform_comparison.csv`
- `outputs/baseline_full_uniform_quant/summary.json`
