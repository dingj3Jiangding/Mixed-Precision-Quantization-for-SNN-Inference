# `scripts/run_allocation_aware_kd_vgg16.py`

## 作用

- 提供 CIFAR-10 `baseline_vgg16` 上 allocation-aware KD 的命令行入口。

## 常用命令

```bash
python scripts/run_allocation_aware_kd_vgg16.py \
  --checkpoint-path outputs/baseline_vgg16/fp32_last.pt \
  --data-root baseline/data \
  --output-dir outputs/allocation_aware_kd_vgg16 \
  --bits 8,4 \
  --weight-allocation-csv outputs/baseline_vgg16_hessian_sensitivity_formal/bit_allocation.csv \
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
  - 指向已有的 Hessian mixed-precision allocation
- `--feature-distill-beta`
  - allocation-aware 特征蒸馏项权重
- `--severity-power`
  - 控制 layer severity 放大的力度

## 输出

- `outputs/allocation_aware_kd_vgg16/comparison.csv`
- `outputs/allocation_aware_kd_vgg16/layer_severity.csv`
- `outputs/allocation_aware_kd_vgg16/summary.json`
