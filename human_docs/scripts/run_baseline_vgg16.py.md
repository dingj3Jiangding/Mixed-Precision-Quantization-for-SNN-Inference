# 文件：`scripts/run_baseline_vgg16.py`

## 作用
- 命令行启动标准 VGG-16-like CIFAR-10 SNN baseline 的 FP32 训练与评估。
- 输出目录默认使用 `outputs/baseline_vgg16`，不覆盖旧 `baseline` 和 `baseline_full`。

## 如何运行
```bash
python scripts/run_baseline_vgg16.py --epochs 10 --device cuda
```

## 常用参数
- `--epochs`
- `--t-steps`
- `--device`
- `--max-train-batches`
- `--max-test-batches`
- `--output-dir`
