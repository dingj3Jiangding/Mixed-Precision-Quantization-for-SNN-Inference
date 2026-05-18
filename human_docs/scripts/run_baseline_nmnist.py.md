# 文件：`scripts/run_baseline_nmnist.py`

## 作用
- 命令行启动标准 DECOLLE-like N-MNIST SNN baseline 的 FP32 训练与评估。
- 输出目录默认使用 `outputs/baseline_nmnist`，不覆盖旧 `baseline` 和 `baseline_full`。

## 如何运行
```bash
python scripts/run_baseline_nmnist.py --epochs 10 --device cuda
```

## 推荐正式结果命令
下面这条更适合作为 N-MNIST 主 baseline 的正式训练配置。它不是当前仓库里已经验证过的“最优结果”，而是按当前 DECOLLE-like 结构给出的**推荐正式配置**：

```bash
python scripts/run_baseline_nmnist.py \
  --data-root baseline_nmnist/data \
  --output-dir outputs/baseline_nmnist \
  --epochs 20 \
  --t-steps 16 \
  --batch-size-train 64 \
  --batch-size-test 128 \
  --lr 1e-3 \
  --weight-decay 5e-4 \
  --seed 42 \
  --device cuda
```

## 调试用命令
如果只是先检查数据和模型链路是否跑通，建议先用更轻的配置：

```bash
python scripts/run_baseline_nmnist.py \
  --data-root baseline_nmnist/data \
  --epochs 3 \
  --batch-size-train 16 \
  --batch-size-test 32 \
  --max-train-batches 20 \
  --max-test-batches 5 \
  --device cuda
```

## 常用参数
- `--epochs`
- `--t-steps`
- `--batch-size-train`
- `--batch-size-test`
- `--lr`
- `--weight-decay`
- `--device`
- `--max-train-batches`
- `--max-test-batches`
- `--output-dir`
