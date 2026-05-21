# `scripts/run_state_write_aware_quantization_nmnist.py`

## 作用

- 提供 N-MNIST 上 state-write-aware quantization 的命令行入口。

## 常用命令

```bash
python scripts/run_state_write_aware_quantization_nmnist.py \
  --checkpoint-path outputs/baseline_nmnist/fp32_best.pt \
  --data-root baseline_nmnist/data \
  --output-dir outputs/state_write_aware_quantization_nmnist \
  --state-bits 8 \
  --relative-delta-threshold 0.05 \
  --write-warmup-steps 4 \
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
