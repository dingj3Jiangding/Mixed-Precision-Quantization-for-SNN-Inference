# 文件：`scripts/run_hessian_sensitivity_vgg16.py`

## 作用
- 对 `baseline_vgg16` checkpoint 运行 Hessian sensitivity 和 mixed-precision 验证。
- 默认使用 `bits=8,4` 和输出目录 `outputs/baseline_vgg16_hessian_sensitivity`。
- 当前版本只评估无 state-aware 的 Hessian-only mixed precision，用于先建立 `baseline_vgg16` 上的基础结果。

## 如何运行
```bash
python scripts/run_hessian_sensitivity_vgg16.py \
  --checkpoint-path outputs/baseline_vgg16/fp32_last.pt \
  --bits 8,4 \
  --device cuda \
  --batch-size-train 8 \
  --batch-size-test 16 \
  --max-hessian-batches 1 \
  --max-test-batches 5 \
  --trace-probes 1 \
  --quant-epochs 1
```

## 关键参数
- `--max-hessian-batches`：Hessian trace 使用的训练 batch 数。
- `--quant-epochs`：mixed-precision fine-tuning 轮数。

## 输出
- sensitivity ranking
- Hessian-only bit allocation
- FP32 / Uniform / HessianMixed comparison
- quantization fine-tuning epoch metrics
