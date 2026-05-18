# 文件：`baseline_vgg16/unquant_runner.py`

## 作用
- 训练和评估 `baseline_vgg16` 的 FP32 VGG-16-like CIFAR-10 SNN。
- 复用旧 baseline 的 direct encoding、spike rate 统计、SOP proxy 和 csv/json 输出格式。

## 如何运行
```bash
python scripts/run_baseline_vgg16.py --epochs 10 --device cuda
```

## 输出文件
- `outputs/baseline_vgg16/epoch_metrics.csv`
- `outputs/baseline_vgg16/summary.json`
- `outputs/baseline_vgg16/fp32_last.pt`
