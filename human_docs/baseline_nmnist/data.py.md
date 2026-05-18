# 文件：`baseline_nmnist/data.py`

## 作用
- 构建 N-MNIST 的训练和测试 DataLoader。
- 使用 `spikingjelly.datasets.n_mnist.NMNIST` 读取事件数据，并将其转换为固定时间步的 frame 序列。

## 数据表示
- 单个样本应为 `[T, C, H, W]`
- batch 后为 `[B, T, C, H, W]`
- 后续在 runner 中再转为多步 SNN 所需的 `[T, B, C, H, W]`

## 说明
- 当前实现采用固定 `frames_number = t_steps` 和 `split_by="number"` 的离散时间分帧方案。
- 这是为了和现有 mixed-precision / Hessian 代码结构保持兼容，而不是实现原生异步事件驱动推理。
