# 文件：`baseline_vgg16/data.py`

## 作用
- 复用旧 baseline 的 CIFAR-10 dataloader 构建函数。
- 保持数据增强、标准化、seed 和 `DataLoader` 行为与旧 baseline 一致，方便做 old-vs-new baseline 对比。

## 如何使用
- `from baseline_vgg16.data import build_cifar10_loaders`
- `train_loader, test_loader = build_cifar10_loaders(cfg, device)`

## 输出
- 返回 CIFAR-10 训练和测试 `DataLoader`。
