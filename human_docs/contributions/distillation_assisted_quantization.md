# Distillation-Assisted Quantization

## 作用
- 解释蒸馏辅助量化的基本原理。
- 说明它为什么适合接在当前项目的 mixed-precision quantization 流程后面。
- 给论文写作提供一份可直接改写的原理说明。

## 核心思想

蒸馏辅助量化的目标不是改变 bit allocation 本身，而是：

- 在量化后模型表达能力下降的情况下，
- 让量化后的 student 模型不仅学习真实标签，
- 还学习 FP32 teacher 模型的输出行为，
- 从而尽量减少量化带来的精度损失。

在当前项目中：

- teacher：FP32 baseline 模型
- student：量化后的 mixed-precision 模型

## 基本训练方式

普通量化 fine-tuning 一般只优化分类损失：

```text
L = CE(student_logits, y)
```

其中：

- `student_logits` 是量化模型输出
- `y` 是真实标签

蒸馏辅助量化会在此基础上增加一个 teacher-student 对齐项：

```text
L = (1 - alpha) * CE + alpha * KD
```

其中：

- `CE`：student 对真实标签的交叉熵损失
- `KD`：student 和 teacher 输出分布之间的蒸馏损失
- `alpha`：控制分类损失和蒸馏损失的权重

## 为什么 teacher 有用

真实标签只提供硬监督，例如：

- 这张图属于某一类

但 FP32 teacher 的输出分布提供了更多软信息，例如：

- 正确类别概率最高
- 其他类别中哪些更相似
- 类别之间的相对置信关系

这些信息能够帮助量化后的 student 保持更接近 FP32 模型的决策边界和类别结构，因此在量化后通常比只用硬标签更稳。

## Temperature 的作用

蒸馏通常不会直接使用普通 softmax，而是使用带温度的 softmax：

```text
softmax(logits / T)
```

其中：

- `T = 1` 时是普通 softmax
- `T > 1` 时输出分布更平滑

更平滑的 teacher 分布可以让 student 更容易学习类别之间的相对关系，而不仅仅是学习“最大类别是谁”。

当前项目里的 logits distillation 本质上是：

- 对 teacher logits 做温度缩放
- 对 student logits 做相同温度缩放
- 用 KL divergence 衡量两者分布差异

## 为什么它对量化有效

量化会引入权重离散化误差，常见影响包括：

- 决策边界偏移
- 特征表达退化
- 深层网络输出分布变形

蒸馏的作用是：

- 用 teacher 约束 student 不要偏离 FP32 行为太远
- 在低比特约束下保留更多原始模型的判别能力

因此，蒸馏辅助量化通常不是“提高模型理论上限”，而是：

- 缓解量化损伤
- 提高量化模型稳定性
- 改善 accuracy-efficiency trade-off

## 在本项目中的使用方式

当前项目新增的蒸馏量化路径是：

- 先使用 Hessian-guided mixed precision 生成 layer-wise bit allocation
- 再用 FP32 teacher 对量化 student 做蒸馏 fine-tuning

也就是说，蒸馏在这里的角色是：

- 不替代 Hessian allocation
- 而是在已有 bit allocation 基础上进一步降低精度损失

## 当前实现范围

当前版本实现的是最简单、最稳的一种蒸馏形式：

- **logits distillation**

即只约束最终分类输出。

尚未实现的更强扩展包括：

- spike activity distillation
- feature distillation
- membrane/state trajectory distillation

这些扩展更贴近 SNN 的内部动态，但实现复杂度和显存成本也更高。

## 适合论文中的表述

可以将该方法描述为：

> 在 Hessian-guided mixed-precision quantization 的基础上，引入 FP32 teacher 对低比特 student 的蒸馏约束，使量化模型在保留压缩收益的同时，更好地逼近全精度模型的输出分布，从而减轻量化误差带来的性能退化。

更简洁的表述也可以是：

> Distillation-assisted quantization uses a full-precision teacher to regularize the quantized student, so that the student preserves the prediction behavior of the original model under low-bit constraints.

## 写作注意点

- 不要把蒸馏写成新的 bit allocation 方法。
- 更准确的说法是：蒸馏是一种 **quantization fine-tuning enhancement**。
- 论文里应强调它的价值在于：
  - 缓解量化误差
  - 提高低比特模型稳定性
  - 改善精度与效率之间的折中

## 一句话总结

蒸馏辅助量化的本质是：

- 用 FP32 teacher 给量化 student 提供额外监督，
- 让 student 在低比特条件下尽量保持与原模型一致的输出行为，
- 从而减少量化导致的精度下降。
