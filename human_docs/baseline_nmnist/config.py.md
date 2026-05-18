# 文件：`baseline_nmnist/config.py`

## 作用
- 定义 `baseline_nmnist` 的实验配置结构。
- 集中管理 N-MNIST 数据路径、训练 epoch、时间步、batch size 和设备设置。

## 特点
- 默认输出目录为 `outputs/baseline_nmnist`
- 默认数据目录为 `baseline_nmnist/data`
- 默认 `epochs=20`，更接近 N-MNIST / DECOLLE-like 小模型的基线训练长度
