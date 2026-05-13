# 文件：`scripts/run_state_aware_hessian_full.py`

## 作用
- 单独运行 `baseline_full` 的 state-aware Hessian mixed-precision 验证。
- 不改动 Hessian-only 的 `scripts/run_hessian_sensitivity_full.py` 行为。

## 如何运行
```bash
python scripts/run_state_aware_hessian_full.py \
  --checkpoint-path outputs/baseline_full/fp32_last.pt \
  --bits 8,4 \
  --device cuda \
  --state-aware-alpha 0.75
```

## 关键参数
- `--state-aware-alpha`：state-aware score 中 Hessian 项的权重。
- `--max-hessian-batches`：Hessian trace 和 state-cost proxy 使用的训练 batch 数。
- `--quant-epochs`：mixed-precision fine-tuning 轮数。

## 输出
- `layer_sensitivity.csv`
- `state_aware_bit_allocation.csv`
- `comparison.csv`
- `state_aware_quant_finetune_epoch_metrics.csv`
- `summary.json`
