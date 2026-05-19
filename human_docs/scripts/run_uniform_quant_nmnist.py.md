# 文件：`scripts/run_uniform_quant_nmnist.py`

## 作用
- 对 `baseline_nmnist` checkpoint 运行 uniform quantization 对比。
- 默认 checkpoint 为 `outputs/baseline_nmnist/fp32_last.pt`。

## 如何运行
```bash
python scripts/run_uniform_quant_nmnist.py \
  --checkpoint-path outputs/baseline_nmnist/fp32_last.pt \
  --bits 8,4 \
  --device cuda
```

## 推荐高精度命令
下面这条更适合作为 N-MNIST 上 uniform quant 的正式对比配置。它是**偏向高精度与稳定比较**的推荐命令，不等于已经验证过的全局最优：

```bash
python scripts/run_uniform_quant_nmnist.py \
  --checkpoint-path outputs/baseline_nmnist/fp32_best.pt \
  --data-root baseline_nmnist/data \
  --output-dir outputs/baseline_nmnist_uniform_quant \
  --bits 8,4 \
  --t-steps 16 \
  --batch-size-test 128 \
  --seed 42 \
  --device cuda
```

## 输出
- `outputs/baseline_nmnist_uniform_quant/uniform_comparison.csv`
- `outputs/baseline_nmnist_uniform_quant/summary.json`
