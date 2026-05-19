# 文件：`scripts/run_state_aware_hessian_nmnist.py`

## 作用
- 单独运行 `baseline_nmnist` 的 state-aware Hessian mixed-precision 验证。
- 不改动 Hessian-only 的 `scripts/run_hessian_sensitivity_nmnist.py` 行为。

## 如何运行
```bash
python scripts/run_state_aware_hessian_nmnist.py \
  --checkpoint-path outputs/baseline_nmnist/fp32_last.pt \
  --bits 8,4 \
  --device cuda \
  --state-aware-alpha 0.75
```

## 推荐高精度命令
下面这条更适合作为 N-MNIST 上 state-aware Hessian 的正式实验配置。它是**偏向高精度**的推荐命令，不等于已经验证过的全局最优：

```bash
python scripts/run_state_aware_hessian_nmnist.py \
  --checkpoint-path outputs/baseline_nmnist/fp32_best.pt \
  --data-root baseline_nmnist/data \
  --output-dir outputs/baseline_nmnist_state_aware_hessian \
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
  --state-aware-alpha 0.75 \
  --seed 42 \
  --device cuda
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
