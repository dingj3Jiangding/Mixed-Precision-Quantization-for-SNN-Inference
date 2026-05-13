# 文件：`baseline_full/weight_state_mixed.py`

## 作用
- 运行一个简化版的 `Weight + State Mixed Precision` 实验。
- 权重侧继续使用 Hessian-guided mixed precision。
- state 侧不直接修改 SpikingJelly 神经元内部膜电位更新，而是通过量化每个 `LIFNode` 的连续输入电流，近似低精度 state update。

## 方法说明
- `assigned_weight_bits`：
  - 按 Hessian trace 排名分配高/低 bit。
- `assigned_state_bits`：
  - 按 `state_cost_proxy` 排名分配 state bit。
  - 高 state cost 层优先使用更低的 state bit，以模拟减少 temporal state storage / traffic 的策略。
- 这是一个简化版 joint experiment，不应表述为“已完成严格的内部膜电位量化实现”。

## 如何运行
```bash
python scripts/run_weight_state_mixed_full.py \
  --checkpoint-path outputs/baseline_full/fp32_last.pt \
  --weight-bits 8,4 \
  --state-bits 8,4 \
  --device cuda
```

如果你已经跑过 `baseline_full` 的 Hessian-only 结果，推荐直接复用权重位宽分配，跳过最耗显存的 Hessian 二阶梯度：
```bash
python scripts/run_weight_state_mixed_full.py \
  --checkpoint-path outputs/baseline_full/fp32_last.pt \
  --weight-allocation-csv outputs/baseline_full_hessian_sensitivity/bit_allocation.csv \
  --state-bits 8,4 \
  --device cuda
```

## 输出文件
- `outputs/baseline_full_weight_state_mixed/weight_bit_allocation.csv`
- `outputs/baseline_full_weight_state_mixed/state_bit_allocation.csv`
- `outputs/baseline_full_weight_state_mixed/comparison.csv`
- `outputs/baseline_full_weight_state_mixed/weight_state_quant_finetune_epoch_metrics.csv`
- `outputs/baseline_full_weight_state_mixed/summary.json`
- `outputs/baseline_full_weight_state_mixed/weight_sensitivity_ranking.png`
