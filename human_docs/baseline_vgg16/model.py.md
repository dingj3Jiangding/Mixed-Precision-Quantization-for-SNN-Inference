# 文件：`baseline_vgg16/model.py`

## 作用
- 定义更接近标准 VGG-16 的 CIFAR-10 多步 SNN。
- 使用 13 个卷积层的 `Conv2d -> BatchNorm2d -> LIFNode` 结构，更适合与论文中的 VGG-16 类 CIFAR-10 SNN baseline 比较。
- 保持输入契约为 `[T, B, C, H, W]`，并使用 `functional.set_step_mode(..., "m")` 兼容现有训练、量化和 Hessian 流程。

## 模型结构
- block 1：`64, 64`
- block 2：`128, 128`
- block 3：`256, 256, 256`
- block 4：`512, 512, 512`
- block 5：`512, 512, 512`
- 每个 block 使用 `AvgPool2d` 下采样，最后接 `AdaptiveAvgPool2d((1, 1)) + Linear(512, 10)`。

## 如何使用
- 构建模型：
  - `model = build_model(num_classes=10)`
- 推荐通过脚本训练：
  - `python scripts/run_baseline_vgg16.py --epochs 10 --device cuda`

## 输出
- forward 返回形状为 `[T, B, num_classes]` 的 logits 序列。
