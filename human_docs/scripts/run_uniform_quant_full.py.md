# 文件：`scripts/run_uniform_quant_full.py`

## 作用
- 对 `baseline_full` checkpoint 运行 uniform quantization 对比。
- 默认 checkpoint 为 `outputs/baseline_full/fp32_last.pt`。

## 如何运行
```bash
python scripts/run_uniform_quant_full.py \
  --checkpoint-path outputs/baseline_full/fp32_last.pt \
  --bits 8,4 \
  --device cuda
```

## 输出
- `outputs/baseline_full_uniform_quant/uniform_comparison.csv`
- `outputs/baseline_full_uniform_quant/summary.json`
