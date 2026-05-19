# 文件：`scripts/run_weight_state_mixed_nmnist.py`

## 作用
- 单独运行 `Weight + State Mixed Precision` 的简化版验证脚本。
- 不影响 `run_hessian_sensitivity_nmnist.py` 和 `run_state_aware_hessian_nmnist.py`。

## 如何运行
```bash
python scripts/run_weight_state_mixed_nmnist.py \
  --checkpoint-path outputs/baseline_nmnist/fp32_last.pt \
  --weight-bits 8,4 \
  --state-bits 8,4 \
  --device cuda
```

如果已有 `outputs/baseline_nmnist_hessian_sensitivity/bit_allocation.csv`，推荐：
```bash
python scripts/run_weight_state_mixed_nmnist.py \
  --checkpoint-path outputs/baseline_nmnist/fp32_last.pt \
  --weight-allocation-csv outputs/baseline_nmnist_hessian_sensitivity/bit_allocation.csv \
  --state-bits 8,4 \
  --device cuda
```

## 推荐高精度命令
下面这条更适合作为 N-MNIST 上 `Weight + State Mixed Precision` 的正式实验配置。它是**偏向高精度和稳定性**的推荐命令，不等于已经验证过的全局最优：

```bash
python scripts/run_weight_state_mixed_nmnist.py \
  --checkpoint-path outputs/baseline_nmnist/fp32_best.pt \
  --data-root baseline_nmnist/data \
  --output-dir outputs/baseline_nmnist_weight_state_mixed \
  --weight-bits 8,4 \
  --state-bits 8,6 \
  --weight-allocation-csv outputs/baseline_nmnist_hessian_sensitivity/bit_allocation.csv \
  --t-steps 16 \
  --batch-size-train 32 \
  --batch-size-test 64 \
  --max-hessian-batches 5 \
  --max-train-batches 0 \
  --max-test-batches 0 \
  --trace-probes 1 \
  --quant-epochs 5 \
  --quant-lr 1e-4 \
  --quant-weight-decay 5e-4 \
  --seed 42 \
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
