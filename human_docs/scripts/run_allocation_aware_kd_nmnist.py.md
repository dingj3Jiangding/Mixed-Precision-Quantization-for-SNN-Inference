# `scripts/run_allocation_aware_kd_nmnist.py`

## 作用

- 提供 N-MNIST 上 allocation-aware KD 的命令行入口。

## 常用命令

```bash
python scripts/run_allocation_aware_kd_nmnist.py \
  --checkpoint-path outputs/baseline_nmnist/fp32_best.pt \
  --data-root baseline_nmnist/data \
  --output-dir outputs/allocation_aware_kd_nmnist \
  --bits 8,4 \
  --weight-allocation-csv outputs/baseline_nmnist_hessian_sensitivity_formal/bit_allocation.csv \
  --t-steps 16 \
  --batch-size-train 32 \
  --batch-size-test 64 \
  --max-test-batches 0 \
  --quant-epochs 5 \
  --quant-lr 1e-4 \
  --quant-weight-decay 5e-4 \
  --distill-alpha 0.5 \
  --distill-temperature 2.0 \
  --feature-distill-beta 0.1 \
  --severity-power 1.0 \
  --seed 42 \
  --device cuda
```

## 关键参数

- `--weight-allocation-csv`
- `--feature-distill-beta`
- `--severity-power`

## 输出

- `outputs/allocation_aware_kd_nmnist/comparison.csv`
- `outputs/allocation_aware_kd_nmnist/layer_severity.csv`
- `outputs/allocation_aware_kd_nmnist/summary.json`
