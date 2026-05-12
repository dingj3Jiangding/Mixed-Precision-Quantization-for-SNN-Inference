# 创新点方案：面向 SNN 推理的状态感知 Hessian 引导混合精度

## 文档目的
本文档定义了一个适合作为毕业论文创新点的研究方向，该方向建立在当前仓库中已经开展的 Hessian mixed precision 复现工作之上。

当前 Hessian pipeline 主要回答的问题是：
- 哪些权重层对量化扰动更敏感？

然而，对于脉冲神经网络（SNN）而言，推理效率并不只由权重精度决定，还会显著受到以下因素影响：
- 时间维度上的 `T` 步展开
- 神经元状态 / 膜电位存储
- spike 活动模式
- 与状态相关的内存访问开销

因此，一个合理的毕业论文创新点，是把 Hessian-guided mixed precision 进一步扩展为一种**面向 SNN 推理的状态感知分配策略（state-aware allocation strategy）**。

---

## 提议的创新点
### 名称
**State-Aware Hessian-Guided Mixed Precision for SNN Inference**

### 核心思想
保留基于 Hessian 的层敏感度作为“精度风险”的主要指标，但在进行精度分配时，额外引入 SNN 特有的状态/资源视角。

也就是说，不再仅仅根据 Hessian 敏感度来决定每一层的 bit-width，而是同时考虑一个**与状态相关的推理代价 proxy**。

这样做会使精度分配策略比单纯的 ANN 风格 Hessian allocation 更适合 SNN，因为 SNN 的效率不仅取决于权重量化敏感度，还取决于 temporal/state overhead。

---

## 动机（Motivation）
### 当前复现版 Hessian baseline 的局限性
一个 Lui 风格的 Hessian baseline 能告诉我们：
- 哪些层对量化更敏感
- 哪些层应保留更高精度，以减少精度损失

但这个 baseline **并不能完整刻画 SNN 特有的代价结构**，原因包括：
1. 状态变量会在多个时间步中被反复更新
2. 状态存储/内存开销可能主导一部分推理代价
3. spike 稀疏性会改变实际有效计算模式
4. 一个好的 SNN precision policy 应该同时平衡：
   - **accuracy sensitivity**
   - **state-related deployment cost**

### 论文动机表述
如果一个 mixed-precision policy 只遵循“权重侧 Hessian 敏感度”，那么它对 SNN inference 来说可能不是最优的，因为它忽略了 temporal dynamics 带来的状态相关资源开销。因此，本文提出一种 state-aware extension，在 Hessian-guided precision allocation 中引入 state/resource proxy 信息。

---

## 最小可实现版本（Minimal Realizable Version）
为了保证在毕业论文周期内可落地，第一版实现**不要求真正完成完整的 state-variable quantization**。

相反，第一版创新方案建议做以下事情：
1. 复现一个 Lui 风格的 Hessian-trace mixed-precision baseline（面向权重层）
2. 为每一层定义一个 state-related cost proxy
3. 在 layer-wise bit-width 分配时，将 Hessian sensitivity 和 state-related cost 结合起来
4. 对比三类方法：
   - Uniform quantization
   - Hessian-only mixed precision
   - State-aware Hessian mixed precision

如果方法动机清楚、实验比较完整，那么这已经足以构成一个成立的毕业论文创新点。

---

## 可选的状态相关 Proxy（Candidate State-Related Proxies）
实现时不需要精确建模真实硬件内存，只要定义一个一致的 proxy 用于实验比较即可。

### 方案 A：activation/state volume proxy
对每一层，用一个与以下因素成比例的 proxy 来近似状态相关代价：
- 输出特征图大小
- 通道数 / 神经元数
- 时间步数 `T`

直觉是：
- 输出状态张量越大的层，通常 temporal storage/update cost 越高

### 方案 B：spike-activity-weighted state proxy
使用测得的 spike activity 来修正这个 proxy：
- state cost proxy ∝ state volume × spike rate

这样可以体现：
- 实际 spike 活动模式会影响真实 temporal cost

### 方案 C：synapse/state hybrid proxy
使用组合 proxy，例如：
- state volume 项
- synapse count 项
- 可选 spike-rate 加权

这样既能体现权重侧压力，也能体现状态侧压力。

---

## 可选的分配公式（Candidate Allocation Formulations）
第一版实现应保持**简单且可解释**。

### 公式 1：加权排序分数（weighted ranking score）
对每一层定义：
- `H_i`：Hessian trace（或归一化 Hessian trace）
- `C_i`：归一化 state-cost proxy

然后定义综合分数：

`Score_i = alpha * norm(H_i) + (1 - alpha) * norm(C_i)`

解释：
- `H_i` 越大，表示 accuracy sensitivity 越高
- `C_i` 越大，表示状态/资源重要性越高
- 综合分数越大的层，应保留更高精度

这是最推荐的第一版实现，因为它最简单、最好解释。

### 公式 2：乘法式优先级（multiplicative prioritization）
`Score_i = norm(H_i) * (1 + beta * norm(C_i))`

解释：
- state cost 作为对 sensitivity importance 的放大因子
- 适合“仍以 Hessian 为主，state-aware 作为修正项”的表达方式

### 公式 3：两阶段排序（two-stage ranking）
1. 先用 Hessian trace 把层分成 sensitive / non-sensitive 两组
2. 再在组内用 state-cost proxy 排序并决定 bit assignment

解释：
- 保留论文中 Hessian-first 的精神
- 同时加入 SNN-specific refinement

---

## 推荐的第一版方法
第一版建议优先采用：
**公式 1：weighted ranking score**

原因：
- 最容易实现
- 最容易调参
- 最容易在论文中解释
- 最容易与 Hessian-only baseline 比较

推荐的初始实验设置：
- 测试 `alpha ∈ {1.0, 0.75, 0.5}`
- `alpha = 1.0` 对应 Hessian-only baseline
- `alpha < 1.0` 引入 state-awareness

---

## 实验比较方案（Experimental Comparison Plan）
该创新点至少应比较以下三种方法：

1. **Uniform Quantization**
   - 例如 W8 / W4 / W2

2. **Hessian-Only Mixed Precision**
   - Lui 风格复现 baseline

3. **State-Aware Hessian Mixed Precision**
   - 本文提出的方法

### 建议使用的评价指标
尽量保留仓库现有指标：
- `test_acc`
- `spike_rate`
- `model_size_mb` 或 `model_size_mb_proxy`
- `sop_proxy`
- `avg_batch_infer_ms`

如果可行，建议增加一个显式 state-related proxy 列，例如：
- `state_cost_proxy`

---

## 预期的论文贡献表述（Expected Thesis Contribution Statement）
可以使用如下表述：

> 本文在 Hessian-guided mixed-precision quantization for SNN 的基础上，引入状态相关的 temporal resource proxy 来指导 layer-wise precision allocation。与仅依赖 Hessian sensitivity 的分配方式相比，所提出的 state-aware 策略能够更好地反映 SNN inference 的特性，因为 SNN 的部署代价不仅取决于权重量化敏感度，还取决于 temporal state overhead。

---

## 风险控制（Risk Control）
### 为什么这是一个好的毕业论文创新点
- 它紧贴当前代码基础，不需要改变整个研究方向。
- 它强调的是一个真正的 SNN-specific 问题，而不是单纯复现 ANN-style mixed precision。
- 它可以分阶段实现。
- 即使提升幅度有限，只要动机清楚、方法合理、实验完整，依然是一个可以 defend 的贡献点。

### 需要严格控制的范围
**不要立刻**把这个方向扩展成：
- 完整 state quantization
- hardware mapping
- RL/ILP search
- 多数据集/多模型大规模扩展

第一目标应该是：
**先做出一条稳定、可解释、可复现的比较链。**

---

## 建议的后续步骤（Suggested Next Steps）
1. 完成 Lui 风格 Hessian mixed-precision baseline 的验证。
2. 定义具体的 per-layer state-cost proxy。
3. 实现一种 combined ranking formula。
4. 运行三组对比实验：
   - Uniform
   - Hessian-only
   - State-aware Hessian
5. 形成一个比较表和一个 trade-off 图。
6. 再决定是否将 precision-time-step allocation 作为 future work 继续扩展。
