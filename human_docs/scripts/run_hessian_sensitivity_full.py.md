# 文件：`scripts/run_hessian_sensitivity_full.py`

## 作用
- 对 `baseline_full` checkpoint 运行 Hessian sensitivity 和 mixed-precision 验证。
- 默认使用 `bits=8,4` 和输出目录 `outputs/baseline_full_hessian_sensitivity`。
- 同时评估 Hessian-only mixed precision 和 state-aware Hessian mixed precision。

## 如何运行
```bash
python scripts/run_hessian_sensitivity_full.py \
  --checkpoint-path outputs/baseline_full/fp32_last.pt \
  --bits 8,4 \
  --device cuda \
  --state-aware-alpha 0.75
```

## 关键参数
- `--state-aware-alpha`：state-aware score 中 Hessian sensitivity 的权重。
  - `1.0` 等价 Hessian-only。
  - `0.75` 是默认推荐值。
  - 更小的值会提高 state-cost proxy 的影响。
- `--max-hessian-batches`：Hessian trace 和 state-cost proxy 使用的训练 batch 数。
- `--quant-epochs`：mixed-precision fine-tuning 轮数。

## 输出
- sensitivity ranking
- Hessian-only bit allocation
- State-aware bit allocation
- FP32 / Uniform / HessianMixed / StateAwareHessianMixed comparison
- Hessian-only 与 state-aware quantization fine-tuning epoch metrics
