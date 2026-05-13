# 文件：`baseline_full/hessian.py`

## 作用
- 对 `baseline_full` 模型做 Hutchinson Hessian trace 层敏感度估计。
- 根据敏感度排名分配 layer-wise bit-width。
- 估计每个 Conv/Linear 层的 state-cost proxy，并实现 state-aware Hessian mixed precision。
- 对 Hessian-only 和 State-aware 两种 mixed-precision 模型分别做简单 fine-tuning，并输出 FP32、Uniform、HessianMixed、StateAwareHessianMixed 对比。

## State-aware 方法
- `state_cost_proxy` 使用量化层输出张量规模估计：
  - 约等于每个样本在该层产生的 `[T, ...]` state/activation volume。
- 综合排序分数：
  - `state_aware_score = alpha * hessian_norm + (1 - alpha) * state_cost_norm`
- `alpha=1.0` 等价于 Hessian-only 排序。
- 默认 `alpha=0.75`，表示仍以 Hessian sensitivity 为主，同时引入 SNN state/resource 视角。

## 如何运行
```bash
python scripts/run_hessian_sensitivity_full.py \
  --checkpoint-path outputs/baseline_full/fp32_last.pt \
  --bits 8,4 \
  --device cuda \
  --max-hessian-batches 10 \
  --trace-probes 1 \
  --quant-epochs 1 \
  --state-aware-alpha 0.75
```

## 输出文件
- `outputs/baseline_full_hessian_sensitivity/layer_sensitivity.csv`
- `outputs/baseline_full_hessian_sensitivity/bit_allocation.csv`
- `outputs/baseline_full_hessian_sensitivity/state_aware_bit_allocation.csv`
- `outputs/baseline_full_hessian_sensitivity/comparison.csv`
- `outputs/baseline_full_hessian_sensitivity/hessian_quant_finetune_epoch_metrics.csv`
- `outputs/baseline_full_hessian_sensitivity/state_aware_quant_finetune_epoch_metrics.csv`
- `outputs/baseline_full_hessian_sensitivity/summary.json`
- `outputs/baseline_full_hessian_sensitivity/sensitivity_ranking.png`
