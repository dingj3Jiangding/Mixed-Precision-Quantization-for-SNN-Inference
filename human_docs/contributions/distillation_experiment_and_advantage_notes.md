# Distillation Experiment And Advantage Notes

## 作用
- 汇总蒸馏辅助量化在当前项目中的实验设计原则。
- 说明为什么该方法适合写入论文主结果。
- 整理蒸馏方法在精度、部署效率、训练成本和系统指标上的优势口径。

## 一、蒸馏方法在当前项目中的正确定位

蒸馏辅助量化不是新的 bit allocation 方法，而是：

- 一种 **quantization fine-tuning enhancement**
- 一种训练阶段使用 teacher 的精度恢复策略
- 用于在不改变 student 推理结构的前提下，减轻量化误差带来的性能退化

因此在论文中更准确的表述是：

- Hessian-guided mixed precision 是主量化方法
- Distillation-assisted quantization 是在该主方法上的训练增强

## 二、蒸馏主结果应该证明什么

蒸馏方法进入主结果，至少要证明以下命题：

> 在相同量化配置、相同 student 结构、相同推理成本下，引入蒸馏后量化模型精度更高或更稳定。

因此蒸馏实验最重要的不是“绝对精度最高”，而是：

- same bit budget
- same deployment cost
- better accuracy

## 三、蒸馏实验必须固定的控制变量

要比较：

1. `HessianMixed`
2. `HessianMixed + KD`

必须固定：

- 同一个 `fp32_best.pt`
- 同一个 `bit_allocation.csv`
- 同一个 `bits` 配置
- 同一个 `t_steps`
- 同一个 `batch-size-test`
- 同一个 full test set
- 同一个 `quant_epochs`
- 同一个 `quant_lr`
- 同一个 `quant_weight_decay`

只允许改变：

- 是否使用 KD

如果这些变量不固定，就不能严格说明性能提升来自蒸馏本身。

## 四、当前项目中推荐的蒸馏主实验设置

对于 `baseline_vgg16` 主线，当前推荐的蒸馏主实验设置是：

- teacher checkpoint：`outputs/baseline_vgg16/fp32_best.pt`
- bit allocation：复用已生成的 Hessian `bit_allocation.csv`
- candidate bits：`8,4`
- `quant_epochs = 5`
- `distill_alpha = 0.5`
- `distill_temperature = 2.0`

这些参数不是理论最优，只是当前最适合做主结果的稳健起点。

## 五、为什么这组参数适合体现 KD 优势

### 1. `8,4` mixed precision 难度适中

- `W8` 通常太容易，KD 增益不明显
- `W4` uniform 太难，student 容易掉太多
- `8,4` 的 Hessian mixed precision 处于“有压缩压力、但仍有恢复空间”的区间

这正是 teacher 最容易发挥作用的场景。

### 2. `distill_alpha = 0.5`

- 保持真实标签监督和 teacher 监督平衡
- 避免只靠 teacher，也避免 teacher 信号过弱

### 3. `distill_temperature = 2.0`

- 能提供更平滑的 soft target
- 在不过度软化类别分布的前提下，增强类别间相似性信息

### 4. `quant_epochs = 5`

- 比 `1 epoch` 更能让量化 student 吸收 teacher 信息
- 训练成本仍可控，适合做正式 mixed-precision fine-tuning

## 六、蒸馏还缺哪些实验才算完整

如果要把蒸馏写进论文主结果，建议至少补足以下实验。

### A. 严格主对照实验

固定：

- 同一个 `fp32_best.pt`
- 同一个 `bit_allocation.csv`
- 同一个 full test evaluation
- 同一个 `quant_epochs`

比较：

1. `HessianMixed`
2. `HessianMixed + KD`

这是蒸馏主结果最关键的实验。

### B. 小范围 KD 超参数消融

推荐最小网格：

- `alpha ∈ {0.3, 0.5, 0.7}`
- `T ∈ {1, 2, 4}`

目的不是大规模搜索，而是说明：

- 当前选用的 `alpha / temperature` 不是任意拍脑袋决定
- 蒸馏方法对超参数有可解释的趋势

### C. 统一评估协议

当前不同结果中 FP32 精度可能不同，通常来自：

- `max_test_batches` 不同
- quick validation 和 full evaluation 混用
- checkpoint 选取不一致

正式论文主表必须统一：

- 同一个 `fp32_best.pt`
- `max_test_batches = 0`
- 同一个 full test set
- 同一个 batch size
- 同一个 device

## 七、蒸馏方法在精度之外的优势

蒸馏方法不能只讲“精度更高”，还应强调它在部署成本上的特殊属性。

### 1. 相同推理成本下精度更高

蒸馏不改变：

- student 结构
- layer-wise bit allocation
- average weight bits
- model size
- SOP proxy

因此，如果 `HessianMixed + KD` 精度高于 `HessianMixed`，其优势可以表述为：

> 在相同部署预算下，蒸馏提升了低比特量化 student 的精度。

这比“更准”本身更有论文价值。

### 2. 推理期零额外结构成本

teacher 只在训练阶段存在，部署时被移除。

因此推理阶段：

- 不需要 teacher
- 不需要额外分支
- 不增加参数量
- 不增加平均权重位宽
- 不增加权重存储
- 不增加带宽需求

理论上，只要 student 架构和量化配置不变，推理复杂度就不变。

### 3. 更好的 accuracy-efficiency trade-off

如果两个方法：

- `avg_weight_bits` 一样
- `model size` 一样
- `inference time` 基本一样
- `SOP proxy` 基本一样

但 KD 版本精度更高，那么就可以写：

> KD improves the accuracy-efficiency trade-off without increasing deployment-time cost.

## 八、蒸馏增加了什么成本

蒸馏不是没有代价，它增加的是训练阶段的成本。

### 增加的成本

- 训练时间更长
- 训练显存更高
- 训练阶段需要额外执行 teacher forward
- 训练阶段需要计算蒸馏损失

### 不增加的成本

- student 参数量
- student 推理结构复杂度
- student 部署期权重带宽
- student 平均量化 bit

因此，蒸馏更准确的定义是：

> 用额外的离线训练成本，换取更好的部署期精度-效率折中。

## 九、哪些实验指标最能体现蒸馏优势

建议至少统一汇报：

- `test_acc`
- `accuracy_drop_vs_fp32`
- `avg_weight_bits`
- `model_size_mb`
- `compression_ratio`
- `avg_batch_infer_ms`
- `sop_proxy`

如果能补充：

- `bit_weighted_sop`
- `training_time_per_epoch`
- `total_finetune_time`
- `peak_train_memory`

则更能完整说明 KD 的训练代价和部署收益。

## 十、最适合论文主表的结果形状

蒸馏方法最理想的结果不是：

- 大幅增加推理资源后提高精度

而是：

| Method | Acc | Avg bits | Model size | Infer time | SOP proxy |
|---|---:|---:|---:|---:|---:|
| HessianMixed | lower | same | same | same | same |
| HessianMixed + KD | higher | same | same | same | same |

这种结果结构最能支撑蒸馏的论文价值。

## 十一、当前项目的结论口径

在当前项目里，蒸馏方法最适合被写成：

- Hessian mixed precision 的增强项
- 一种恢复低比特量化 student 精度的训练方法
- 一种不增加部署期结构开销的离线训练增强策略

不应写成：

- 新的 bit allocation 方法
- 新的推理结构
- 新的硬件加速结构

## 十二、一句话总结

蒸馏辅助量化最值得写进论文主结果的原因是：

- 它增加的是训练阶段成本，
- 但不增加部署阶段成本，
- 并且有潜力在相同 mixed-precision budget 下提高量化 student 的精度，
- 因而改善整体的 accuracy-efficiency trade-off。
