# `scripts/run_difficulty_adaptive_state_precision_vgg16.py`

## 作用

- 提供 CIFAR-10 `baseline_vgg16` 上 difficulty-adaptive state precision 的命令行入口。

## 常用命令

```bash
python scripts/run_difficulty_adaptive_state_precision_vgg16.py \
  --checkpoint-path outputs/baseline_vgg16/fp32_last.pt \
  --data-root baseline/data \
  --output-dir outputs/difficulty_adaptive_state_precision_vgg16 \
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

## 关键参数

- `--low-state-bits`
  - easy samples 在后续步使用的低比特
- `--high-state-bits`
  - warmup 阶段和 hard samples 使用的高比特
- `--warmup-steps`
  - 先用多少个 timestep 估计样本难度
- `--difficulty-metric`
  - `confidence` 或 `margin`
- `--easy-threshold`
  - easy sample 判定阈值
