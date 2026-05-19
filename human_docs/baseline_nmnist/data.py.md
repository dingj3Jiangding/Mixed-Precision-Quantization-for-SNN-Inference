# 文件：`baseline_nmnist/data.py`

## 作用
- 构建 N-MNIST 的训练和测试 DataLoader。
- 使用 `spikingjelly.datasets.n_mnist.NMNIST` 读取事件数据，并将其在线转换为固定时间步的 frame 序列。

## 数据表示
- 单个样本应为 `[T, C, H, W]`
- batch 后为 `[B, T, C, H, W]`
- 后续在 runner 中再转为多步 SNN 所需的 `[T, B, C, H, W]`

## 说明
- 当前实现先以 `data_type="event"` 读取 N-MNIST，再对每个样本在线调用 `integrate_events_by_fixed_frames_number(...)` 生成固定帧数。
- 这样做是为了避免依赖 SpikingJelly 的全量 frame 预生成缓存目录；当该缓存目录被中断后留下半成品时，容易出现部分类别目录为空的问题。
- 分帧策略仍然采用固定 `frames_number = t_steps` 和 `split_by="number"`，目的是与现有 mixed-precision / Hessian 代码结构保持兼容，而不是实现原生异步事件驱动推理。
