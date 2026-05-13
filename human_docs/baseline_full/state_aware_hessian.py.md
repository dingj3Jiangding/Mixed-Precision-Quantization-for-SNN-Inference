# 文件：`baseline_full/state_aware_hessian.py`

## 作用
- 在不影响 `baseline_full/hessian.py` 主验证链的前提下，单独实现 state-aware Hessian mixed precision。
- 先估计 Hessian layer sensitivity，再估计每个量化层的 `state_cost_proxy`。
- 使用加权分数：
  - `state_aware_score = alpha * hessian_norm + (1 - alpha) * state_cost_norm`
- 输出 `FP32 / Uniform / StateAwareHessianMixed` 比较结果。

## 如何运行
```bash
python scripts/run_state_aware_hessian_full.py \
  --checkpoint-path outputs/baseline_full/fp32_last.pt \
  --bits 8,4 \
  --device cuda \
  --state-aware-alpha 0.75
```

## 输出文件
- `outputs/baseline_full_state_aware_hessian/layer_sensitivity.csv`
- `outputs/baseline_full_state_aware_hessian/state_aware_bit_allocation.csv`
- `outputs/baseline_full_state_aware_hessian/comparison.csv`
- `outputs/baseline_full_state_aware_hessian/state_aware_quant_finetune_epoch_metrics.csv`
- `outputs/baseline_full_state_aware_hessian/summary.json`
- `outputs/baseline_full_state_aware_hessian/sensitivity_ranking.png`
