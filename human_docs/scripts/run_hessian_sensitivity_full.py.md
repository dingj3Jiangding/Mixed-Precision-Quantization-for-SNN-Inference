# 文件：`scripts/run_hessian_sensitivity_full.py`

## 作用
- 对 `baseline_full` checkpoint 运行 Hessian sensitivity 和 mixed-precision 验证。
- 默认使用 `bits=8,4` 和输出目录 `outputs/baseline_full_hessian_sensitivity`。

## 如何运行
```bash
python scripts/run_hessian_sensitivity_full.py \
  --checkpoint-path outputs/baseline_full/fp32_last.pt \
  --bits 8,4 \
  --device cuda
```

## 输出
- sensitivity ranking
- bit allocation
- FP32 / Uniform / HessianMixed comparison
- quantization fine-tuning epoch metrics
