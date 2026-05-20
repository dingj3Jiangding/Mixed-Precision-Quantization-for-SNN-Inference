# Allocation-Aware KD

## 作用

- 记录当前新增方法的设计动机与实现边界。
- 方便后续写入论文方法章节或 related work boundary。

## 方法定位

这不是“首次把 KD 用到 SNN”或“首次把 KD 用到量化 SNN”。

更准确的定位是：

- 在现有 Hessian-guided mixed-precision SNN quantization pipeline 上
- 新增一个 **allocation-aware distillation** 变体

## 与普通 logits KD 的区别

普通 logits KD：

- 所有层共用一个全局蒸馏损失
- 不区分 mixed-precision allocation 的层间异质性

allocation-aware KD：

- 先读取或估计 layer-wise bit allocation
- 再根据 bit-width 和 Hessian trace 构造每层的量化压力 `severity`
- 对更“难”的层施加更强的 feature matching

## 当前实现形式

当前实现使用：

- 全局 logits KD
- allocation-aware feature distillation

损失可写为：

```text
L = (1 - alpha) * CE + alpha * KD + beta * FeatureKD_alloc
```

其中：

- `alpha`：logits KD 权重
- `beta`：allocation-aware feature distillation 权重

## 当前实现边界

- 仍然是 teacher-student offline training
- 不改变 deployment-time student 结构
- 不改变 layer-wise bit allocation 本身
- 不是完整的 membrane trajectory distillation

## 实验建议

最核心的比较对象是：

1. HessianMixed
2. HessianMixed + plain KD
3. HessianMixed + allocation-aware KD

如果第 3 项优于第 2 项，就能说明：

- 不是单纯“有 KD 就行”
- 而是“让 KD respect the mixed-precision allocation”有额外价值
