# 文件：`scripts/run_uniform_quant_vgg16.py`

## 作用
- 对 `baseline_vgg16` checkpoint 运行 uniform quantization 对比。
- 默认 checkpoint 为 `outputs/baseline_vgg16/fp32_last.pt`。

## 如何运行
```bash
python scripts/run_uniform_quant_vgg16.py \
  --checkpoint-path outputs/baseline_vgg16/fp32_last.pt \
  --bits 8,4 \
  --device cuda
```

## 输出
- `outputs/baseline_vgg16_uniform_quant/uniform_comparison.csv`
- `outputs/baseline_vgg16_uniform_quant/summary.json`
