# 文件：`baseline_full/model.py`

## 作用
- 定义更接近 CIFAR-10 SNN 论文常用 backbone 的 VGG-like 多步 SNN。
- 使用 `Conv2d -> BatchNorm2d -> LIFNode` 重复卷积块，替代旧 baseline 中由巨大 `fc1` 主导参数量的结构。
- 保持输入契约为 `[T, B, C, H, W]`，并使用 `functional.set_step_mode(..., "m")` 兼容现有训练、量化和 Hessian 流程。

## 模型结构
- stage 1：两层 `3x3` 卷积，通道数 64，然后 `AvgPool2d(2)`。
- stage 2：两层 `3x3` 卷积，通道数 128，然后 `AvgPool2d(2)`。
- stage 3：两层 `3x3` 卷积，通道数 256，然后 `AdaptiveAvgPool2d((1, 1))`。
- classifier：轻量 `Linear(256, 10)`。

## 如何使用
- 构建模型：
  - `model = build_model(num_classes=10)`
- 推荐通过脚本训练：
  - `python scripts/run_baseline_full.py --epochs 10 --device cuda`

## 输出
- forward 返回形状为 `[T, B, num_classes]` 的 logits 序列。
