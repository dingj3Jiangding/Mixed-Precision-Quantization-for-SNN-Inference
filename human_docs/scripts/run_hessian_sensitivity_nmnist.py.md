# 文件：`scripts/run_hessian_sensitivity_nmnist.py`

## 作用
- 对 `baseline_nmnist` checkpoint 运行 Hessian sensitivity 和 mixed-precision 验证。
- 默认使用 `bits=8,4` 和输出目录 `outputs/baseline_nmnist_hessian_sensitivity`。
- 当前版本只评估无 state-aware 的 Hessian-only mixed precision，用于先建立 `baseline_nmnist` 上的基础结果。

## 如何运行
```bash
python scripts/run_hessian_sensitivity_nmnist.py \
  --checkpoint-path outputs/baseline_nmnist/fp32_last.pt \
  --bits 8,4 \
  --device cuda \
  --batch-size-train 8 \
  --batch-size-test 16 \
  --max-hessian-batches 1 \
  --max-test-batches 5 \
  --trace-probes 1 \
  --quant-epochs 1
```

## 推荐高精度命令
下面这条更适合作为 N-MNIST 上 Hessian mixed precision 的正式实验配置。它是**偏向高精度**的推荐命令，不等于已经验证过的全局最优：

```bash
python scripts/run_hessian_sensitivity_nmnist.py \
  --checkpoint-path outputs/baseline_nmnist/fp32_best.pt \
  --data-root baseline_nmnist/data \
  --output-dir outputs/baseline_nmnist_hessian_sensitivity \
  --bits 8,4 \
  --t-steps 16 \
  --batch-size-train 32 \
  --batch-size-test 64 \
  --max-hessian-batches 10 \
  --max-test-batches 0 \
  --trace-probes 1 \
  --quant-epochs 5 \
  --quant-lr 1e-4 \
  --quant-weight-decay 5e-4 \
  --allocation-policy rank-map \
  --seed 42 \
  --device cuda
```

## 关键参数
- `--max-hessian-batches`：Hessian trace 使用的训练 batch 数。
- `--quant-epochs`：mixed-precision fine-tuning 轮数。

## 输出
- sensitivity ranking
- Hessian-only bit allocation
- FP32 / Uniform / HessianMixed comparison
- quantization fine-tuning epoch metrics
