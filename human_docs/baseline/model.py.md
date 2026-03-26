# 文件：`baseline/model.py`

## 作用
- 定义 `CifarBaselineSNN`：用于 CIFAR-10 的基线 SNN 模型。
- 网络结构为两层卷积 + 两层全连接，中间使用 `LIFNode`。
- 使用 SpikingJelly 的多步模式（`step_mode='m'`），输入形状为 `[T,B,C,H,W]`。

## 如何使用
- 构建模型：
  - `model = build_model(num_classes=10)`
- 前向输入：
  - `x_seq` 形状必须是 `[T,B,C,H,W]`
- 前向输出：
  - 返回 `logits_seq`，形状 `[T,B,num_classes]`
- 训练时常见做法：
  - `logits = logits_seq.mean(dim=0)` 后再做交叉熵
