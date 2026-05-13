# 文件：`baseline_full/hessian.py`

## 作用
- 对 `baseline_full` 模型做 Hutchinson Hessian trace 层敏感度估计。
- 根据敏感度排名分配 layer-wise bit-width。
- 对 mixed-precision 模型做简单 fine-tuning，并输出 FP32、Uniform、HessianMixed 对比。
- 当前脚本先用于验证无 state-aware 的 `baseline_full` Hessian 结果；state-aware 方法暂不接入该主流程。

## 如何运行
```bash
python scripts/run_hessian_sensitivity_full.py \
  --checkpoint-path outputs/baseline_full/fp32_last.pt \
  --bits 8,4 \
  --device cuda \
  --max-hessian-batches 10 \
  --trace-probes 1 \
  --quant-epochs 1
```

## 输出文件
- `outputs/baseline_full_hessian_sensitivity/layer_sensitivity.csv`
- `outputs/baseline_full_hessian_sensitivity/bit_allocation.csv`
- `outputs/baseline_full_hessian_sensitivity/comparison.csv`
- `outputs/baseline_full_hessian_sensitivity/quant_finetune_epoch_metrics.csv`
- `outputs/baseline_full_hessian_sensitivity/summary.json`
- `outputs/baseline_full_hessian_sensitivity/sensitivity_ranking.png`
