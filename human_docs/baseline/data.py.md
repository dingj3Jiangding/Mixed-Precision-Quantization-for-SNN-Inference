# 文件：`baseline/data.py`

## 作用
- 构建 CIFAR-10 训练与测试 `DataLoader`。
- 统一图像预处理（训练增强 + 标准化）。
- 通过 `seed_worker` 与 `generator` 提升数据加载可复现性。

## 如何使用
- 主函数：
  - `train_loader, test_loader = build_cifar10_loaders(cfg, device)`
- 输入：
  - `cfg`：`BaselineConfig` 实例
  - `device`：`cuda/cpu/mps`，用于决定 `pin_memory`
- 输出：
  - 返回 `(train_loader, test_loader)`，供训练与评估流程调用
