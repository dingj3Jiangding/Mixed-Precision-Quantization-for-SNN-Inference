# 文件：`baseline_nmnist/unquant_runner.py`

## 作用
- 负责 `baseline_nmnist` 的 FP32 训练、评估、日志记录和 checkpoint 保存。

## 与旧 baseline 的区别
- 输入不再是静态图像复制到多个时间步，而是直接使用 N-MNIST 分帧后的 `[B, T, C, H, W]` 序列。
- `direct_encode` 在这里的作用是把 batch 维和时间维换位，变成 `[T, B, C, H, W]`。
- 每个 epoch 结束都会落盘：
  - `epoch_metrics.csv`
  - `fp32_last.pt`
  - `fp32_best.pt`

## 输出文件
- `outputs/baseline_nmnist/epoch_metrics.csv`
- `outputs/baseline_nmnist/fp32_last.pt`
- `outputs/baseline_nmnist/fp32_best.pt`
- `outputs/baseline_nmnist/summary.json`
