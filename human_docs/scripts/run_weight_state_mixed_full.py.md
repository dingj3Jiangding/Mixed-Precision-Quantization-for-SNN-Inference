# 文件：`scripts/run_weight_state_mixed_full.py`

## 作用
- 单独运行 `Weight + State Mixed Precision` 的简化版验证脚本。
- 不影响 `run_hessian_sensitivity_full.py` 和 `run_state_aware_hessian_full.py`。

## 如何运行
```bash
python scripts/run_weight_state_mixed_full.py \
  --checkpoint-path outputs/baseline_full/fp32_last.pt \
  --weight-bits 8,4 \
  --state-bits 8,4 \
  --device cuda
```

如果已有 `outputs/baseline_full_hessian_sensitivity/bit_allocation.csv`，推荐：
```bash
python scripts/run_weight_state_mixed_full.py \
  --checkpoint-path outputs/baseline_full/fp32_last.pt \
  --weight-allocation-csv outputs/baseline_full_hessian_sensitivity/bit_allocation.csv \
  --state-bits 8,4 \
  --device cuda
```

## 关键参数
- `--weight-bits`：候选权重位宽集合。
- `--state-bits`：候选 state 位宽集合。
- `--weight-allocation-csv`：复用已有 Hessian-only 权重分配结果，避免重新跑最耗显存的二阶梯度。
- `--max-hessian-batches`：Hessian ranking 使用的训练 batch 数。
- `--quant-epochs`：joint mixed-precision fine-tuning 轮数。

## 注意
- 当前 state 量化是对 `LIFNode` 连续输入电流的近似量化，不是直接改内部膜电位更新公式。
- 如果显存不够，优先降低 `--batch-size-train`、`--max-hessian-batches` 和 `--t-steps`。
