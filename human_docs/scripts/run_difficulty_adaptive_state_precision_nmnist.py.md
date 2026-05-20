# `scripts/run_difficulty_adaptive_state_precision_nmnist.py`

## 作用

- 提供 N-MNIST 上 difficulty-adaptive state precision 的命令行入口。

## 常用命令

```bash
python scripts/run_difficulty_adaptive_state_precision_nmnist.py \
  --checkpoint-path outputs/baseline_nmnist/fp32_best.pt \
  --data-root baseline_nmnist/data \
  --output-dir outputs/difficulty_adaptive_state_precision_nmnist \
  --low-state-bits 4 \
  --high-state-bits 8 \
  --warmup-steps 4 \
  --difficulty-metric confidence \
  --easy-threshold 0.7 \
  --t-steps 16 \
  --batch-size-train 32 \
  --batch-size-test 64 \
  --max-test-batches 0 \
  --quant-epochs 5 \
  --quant-lr 1e-4 \
  --quant-weight-decay 5e-4 \
  --seed 42 \
  --device cuda
```

## 说明

- 如果本地没有 `fp32_best.pt`，需要改成你实际存在的 checkpoint 路径。
- 该脚本输出 fixed low-state 与 adaptive-state 两组 finetune 结果，便于直接比较精度和 state 资源指标。
