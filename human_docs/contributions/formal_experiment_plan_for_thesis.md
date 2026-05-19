# Formal Experiment Plan For Thesis

## 作用
- 给当前项目提供一份适合严肃论文写作的正式实验方案。
- 明确哪些实验必须做、哪些实验是扩展或探索性实验。
- 统一量化、蒸馏和 state-aware 方法的比较协议。

## 一、实验总目标

本文实验需要回答以下问题：

1. Uniform low-bit quantization 会在多大程度上损伤 SNN 精度？
2. Hessian-guided mixed precision 是否能在相近低比特预算下优于 uniform quantization？
3. 将状态相关代价纳入量化分配后，是否能进一步改善 SNN 的精度-资源折中？
4. Distillation-assisted quantization 是否能在不增加部署期推理结构成本的前提下，恢复低比特 mixed-precision student 的精度？
5. Joint weight-state compression 在当前实现中能提供哪些额外观察，即使它还不足以成为主方法？

## 二、实验主线与优先级

### A. 必做主线

1. FP32 baseline
2. Uniform quantization
3. Hessian-guided mixed precision
4. Hessian-guided mixed precision + KD

### B. 重要扩展

5. State-aware Hessian mixed precision

### C. 探索性实验

6. Weight + State mixed precision
7. N-MNIST supporting experiments

## 三、数据集与模型角色分工

### 1. CIFAR-10 + `baseline_vgg16`

作为论文最主要的结果主线：

- 更适合与标准图像分类量化工作比较
- 更适合承载 Hessian / KD 主结果

### 2. N-MNIST + `baseline_nmnist`

作为事件驱动数据集的支持性实验主线：

- 用于证明方法并非只适用于静态图像 SNN
- 也用于强化 SNN-specific motivation

### 3. `baseline_full`

主要作为项目开发过程中的方法验证与早期实验参考。

在论文中可以出现，但不建议作为最主要最终主表的 backbone。

## 四、正式实验时必须统一的协议

如果是进入论文最终主表的结果，必须统一：

- 同一个模型结构
- 同一个 teacher / initialization checkpoint
- 同一个 `t_steps`
- 同一个 test set
- 同一个 test batch size
- 同一个 device
- 同一个 full evaluation 协议（如 `max_test_batches = 0`）

如果不统一，结果只能视为调试、快速验证或支持性观察，不能直接进入最终主表。

## 五、各方法的正式比较对象

### 1. Uniform quantization

目的：

- 建立最基本的低比特基线

推荐比较：

- FP32
- W8
- W4

### 2. Hessian-guided mixed precision

目的：

- 证明 layer-wise Hessian sensitivity 可比 uniform quantization 更合理地分配 bit-width

推荐比较：

- FP32
- UniformW4
- HessianMixed

### 3. State-aware Hessian

目的：

- 检查显式考虑 SNN state-related cost 后，是否能进一步改善分配策略

推荐比较：

- FP32
- UniformW4
- HessianMixed
- StateAwareHessian

### 4. Distillation-assisted quantization

目的：

- 检查在**相同 mixed-precision deployment budget** 下，KD 是否能恢复 student 精度

推荐比较：

- HessianMixed
- HessianMixed + KD

关键点：

- bit allocation 必须相同
- 只改变 KD on/off

### 5. Weight + State mixed precision

目的：

- 探索联合压缩权重与状态的潜力

注意：

- 当前实现是 proxy experiment
- 不建议作为论文主结果的唯一主打方法

## 六、正式推荐参数（主结果向）

以下参数不是声称“全局最优”，而是当前项目中适合正式比较、偏向高精度和稳定性的推荐起点。

### A. `baseline_vgg16` FP32

- `epochs = 100`
- `t_steps = 16`
- `batch_size_train = 128`
- `batch_size_test = 256`

### B. `baseline_vgg16` Hessian mixed precision

- `bits = 8,4`
- `t_steps = 16`
- `batch_size_train = 32`
- `batch_size_test = 64`
- `max_hessian_batches = 10`
- `trace_probes = 1`
- `quant_epochs = 5`
- `quant_lr = 1e-4`
- `quant_weight_decay = 5e-4`
- `allocation_policy = rank-map`

### C. `baseline_vgg16` State-aware Hessian

- same as Hessian mixed
- `state_aware_alpha = 0.75`

### D. `baseline_vgg16` Distillation-assisted quantization

- teacher checkpoint: `fp32_best.pt`
- same `bit_allocation.csv` as Hessian mixed baseline
- `quant_epochs = 5`
- `distill_alpha = 0.5`
- `distill_temperature = 2.0`

### E. `baseline_vgg16` Weight + State mixed precision

- `weight_bits = 8,4`
- `state_bits = 8,6`
- `quant_epochs = 5`
- reuse Hessian weight allocation if possible

### F. `baseline_nmnist` supporting experiments

Use the corresponding N-MNIST script docs as the operational command source, but keep the same comparative logic:

- FP32
- UniformW4
- HessianMixed
- optionally KD and state-aware variants

## 七、蒸馏主实验的严格设计

这是蒸馏进入论文主结果最关键的一部分。

### 必须固定

- 同一个 `fp32_best.pt`
- 同一个 `bit_allocation.csv`
- 同一个 `bits = 8,4`
- 同一个 `quant_epochs`
- 同一个 `t_steps`
- 同一个 full evaluation setting

### 只改变

- whether KD is used

### 推荐主对照

1. `HessianMixed`
2. `HessianMixed + KD`

### 推荐 KD 初始参数

- `alpha = 0.5`
- `T = 2.0`

### 必要的最小 KD 消融

- `alpha ∈ {0.3, 0.5, 0.7}`
- `T ∈ {1, 2, 4}`

目的是证明蒸馏结果不是偶然超参数命中。

## 八、状态相关方法的实验边界

### State-aware Hessian

可以作为重要扩展，但不应强行宣称为全新方向。论文中应强调：

- 本文研究的是在 Hessian-guided mixed precision 基础上加入状态相关资源代价的扩展分配策略

### Weight + State mixed precision

当前实现应明确写成：

- proxy implementation
- exploratory direction

不应写成已经完成了成熟的内部膜电位联合量化方案。

## 九、需要汇报的指标

### A. 精度

- `test_acc`
- `accuracy_drop_vs_fp32`

### B. 存储

- `avg_weight_bits`
- `model_size_mb`
- `compression_ratio`

### C. 推理效率

- `avg_batch_infer_ms`
- `spike_rate`
- `sop_proxy`
- `bit_weighted_sop`

### D. SNN 状态相关

- `avg_state_bits`
- `state_memory_proxy`

### E. 训练成本（特别是 KD）

- fine-tuning time per epoch
- total fine-tuning time
- peak training memory if available

## 十、结果呈现方式

### 主结果表

建议至少包含：

- FP32
- UniformW4
- HessianMixed
- StateAwareHessian
- HessianMixed + KD

列：

- Accuracy
- Avg Weight Bits
- Model Size
- Compression Ratio
- Infer Time
- SOP Proxy
- Bit-weighted SOP

### KD 专门对照表

比较：

- HessianMixed
- HessianMixed + KD

要求：

- 明确说明 allocation 相同
- 明确说明 deployment cost class 相同

### 训练成本表

比较：

- HessianMixed fine-tuning
- HessianMixed + KD fine-tuning

列：

- Epochs
- Time per epoch
- Total fine-tuning time
- Peak memory

## 十一、各方法该如何在论文中定性

### HessianMixed

- 主 baseline quantization method

### StateAwareHessian

- SNN-oriented extension

### Hessian + KD

- main added training enhancement
- most suitable method to be highlighted as a practical contribution

### Weight + State mixed

- exploratory / future-work method

## 十二、实验章节结论应如何写

实验章节的结论不应写成：

- 量化提高了模型性能

而应写成：

- 混合精度量化在保持可接受精度的前提下显著降低了模型存储和推理期权重带宽需求；
- Hessian-guided mixed precision 在相近低比特预算下优于 uniform quantization；
- Distillation-assisted quantization 在不增加部署期 student 结构成本的前提下，进一步恢复了低比特 mixed-precision SNN 的精度；
- state-aware 资源分析为 SNN-specific quantization 提供了有价值的扩展视角；
- joint weight-state compression 仍值得继续研究，但当前实现更适合作为探索性工作。
