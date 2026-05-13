# 更激进的备选创新方向（BCD）

## 文档目的
本文档记录了几个比当前 **State-Aware Hessian-Guided Mixed Precision for SNN Inference** 更激进的毕业论文创新方向。

这些方向在“新颖性”和“研究感”上更强，但同时也带来了更高的实现风险与实验风险。它们应被视为毕业论文定位时的候选方向，而不是现在立刻必须投入实现的承诺。

---

## 方向 B：Weight + State Mixed Precision

### 核心思想
不再停留在只做权重 mixed precision，而是显式考虑 **SNN 中权重与神经元状态的联合混合精度（joint mixed precision for weights and neuron states）**。

也就是说，不只是给每层分配一个 weight bit-width，而是同时分配：
- 每层的权重精度
- 每层的 state / membrane 精度

这样更能体现 ANN 与 SNN 量化问题的本质差异，因为 SNN inference 不仅受到权重量化扰动影响，还受到以下因素影响：
- membrane/state storage
- 时间维上的重复 state update
- 与状态相关的 memory traffic

### 为什么这个方向更强
- 它比普通的 Hessian mixed precision 更有 SNN 特性。
- 它更接近真实 neuromorphic deployment 的考虑。
- 它可以被定位为对 ANN-style mixed-precision quantization 更强的一种扩展。

### 研究价值
一个合适的论文表述可以是：

> 现有 Hessian-aware mixed-precision 方法主要关注权重量化敏感度，而本方向尝试通过联合考虑 weight precision 和 state precision，构建一个更完整的 SNN quantization 设定，从而更好地匹配 SNN inference 的真实资源瓶颈。

### 风险
- 工程复杂度远高于 weight-only quantization。
- 当前仓库尚未具备直接支持 state-variable quantization 的基础设施。
- 这可能要求修改 neuron state update 逻辑或 quantization 注入点。
- 这一方向很容易超出毕业设计可承受的工作范围。

### 实践建议
这是一个 **high-risk, high-reward** 的方向。
如果选择这个方向，建议分阶段推进：
1. 先定义 state-precision abstraction 和 state-cost proxy
2. 再加入一个简化版 state quantization experiment
3. 最后才尝试完整 joint weight+state mixed precision

### 当前仓库中的最小实现版本
当前可以在本仓库中落地的版本，是一个**简化版 Weight + State Mixed Precision experiment**：
- weight 侧：
  - 仍按 Hessian trace 做 layer-wise mixed precision。
- state 侧：
  - 不直接改 SpikingJelly 神经元内部膜电位更新逻辑。
  - 改为量化每个 `LIFNode` 的连续输入电流，作为低精度 state update 的近似实现。

这样做的优点：
- 能跑通独立实验链。
- 有明确的 weight bits / state bits 输出。
- 不需要侵入式重写 neuron 内部实现。

这样做的边界：
- 严格来说，它是 **state quantization proxy experiment**，不是完整的内部膜电位量化实现。
- 论文里应明确表述为“简化版 joint weight-state quantization experiment”。

---

## 方向 C：SNN-Aware Hessian Allocation Metric

### 核心思想
不再使用纯粹的 Hessian-based layer ranking，而是提出一个新的 **SNN-aware allocation score**，将曲率信息与 SNN 特有的推理因素结合起来。

这个方法仍保留 Hessian 作为主要 sensitivity signal，但会用额外的 SNN-related proxy 来修正层的重要性，例如：
- state cost proxy
- spike activity
- temporal burden
- state volume 或 output-state size

### 可能的公式形式
一个通用的 score 可以写成：

#### 加法形式
`Score_i = alpha * norm(H_i) + beta * norm(C_i) + gamma * norm(S_i)`

其中：
- `H_i` = Hessian trace 或 Hessian-based sensitivity
- `C_i` = state-related cost proxy
- `S_i` = spike-related activity proxy

#### 乘法形式
`Score_i = norm(H_i) * (1 + beta * norm(C_i))`

#### 两阶段形式
1. 先用 Hessian trace 决定 sensitive set
2. 再在 sensitive set 内用 state/spike proxy 细化排序

### 为什么这个方向强
- 它比简单说“加入 state awareness”更像一个正式方法。
- 它天然能够产出公式、ablation 和对比实验。
- 它比完整的 weight+state quantization 更容易实现。
- 它与当前代码仓库的契合度较高，可以直接叠加在 Lui-style Hessian baseline 之上。

### 研究价值
一个合适的论文表述可以是：

> 本方向提出一种 SNN-aware Hessian allocation criterion，在 layer-wise Hessian sensitivity 的基础上，引入 state-related 和 spike-related inference proxy，从而形成一个更符合 SNN deployment 特性的 precision assignment rule。

### 风险
- 如果新 score 太 heuristic，贡献可能显得偏弱。
- 系数 `alpha`、`beta`、`gamma` 需要有充分动机与 ablation 支撑。
- 如果评估不充分，这个方向可能会被看作“只是又改了一个排序指标”。

### 实践建议
这是一个最好的 **中间强度方向（middle-ground）**：
- 比纯复现更强
- 比单纯 state-aware narrative 更正式
- 同时仍然适合毕业论文周期内完成

如果你想要一个“**比当前方案更激进、但仍然可控**”的方向，这是最推荐的选择。

---

## 方向 D：Pareto-Aware Mixed Precision for SNN Inference

### 核心思想
把 precision allocation 问题明确地转化成一个 **多目标 SNN inference optimization problem**。

也就是说，不再只围绕 accuracy 或 sensitivity 单独做精度分配，而是显式平衡多个 deployment metric，例如：
- accuracy
- model size
- spike rate
- SOP proxy
- state-related cost proxy
- latency proxy

核心目标是：
构造或搜索出一组 mixed-precision setting，使其在 SNN inference 的多个指标之间形成更优的 trade-off frontier。

### 为什么这个方向强
- 它非常有“研究论文”的味道。
- 它和毕业论文里“面向推理的量化”这个主题非常契合。
- 它天然适合生成 Pareto-style 图表。
- 它能把问题从“只追求 accuracy”转向“accuracy-resource trade-off”，这更符合 SNN deployment 的真实需求。

### 研究价值
一个合适的论文表述可以是：

> 本方向将 SNN mixed-precision allocation 重新表述为一个 Pareto-aware inference optimization problem，在精度分配时同时平衡 accuracy、memory、temporal-state cost 与 compute-related proxy，而不是只依赖单一 sensitivity metric。

### 可能的实现方式
这不一定要求做重量级全局搜索，可以先从轻量版开始，例如：
- 生成若干候选 mixed-precision policy
- 在多个指标下进行比较
- 识别 non-dominated trade-off points
- 基于 Pareto preference 设计一个简单 selection rule

### 风险
- 如果做成 full search，复杂度可能膨胀得太快。
- 如果做得太轻，可能看起来只是“画了个图”。
- 需要认真设计 metric，才能让“Pareto-aware”这个说法真正站得住。

### 实践建议
这是一个非常好的 **论文定位与结果展示方向**，尤其适合最终答辩和论文 framing。

它最适合和另一个方法创新点结合，例如：
- 先用方向 C 构造一个 SNN-aware allocation score
- 再在 Pareto-aware comparison framework 下评估这个 score

---

## 总体建议
在这三个更激进的方向中：

- **方向 B（Weight + State Mixed Precision）** 在原始创新性上最强，但实现风险也最高。
- **方向 C（SNN-Aware Hessian Allocation Metric）** 在创新性、清晰度和可行性之间平衡最好。
- **方向 D（Pareto-Aware Mixed Precision for SNN Inference）** 最适合作为论文 framing 和结果展示层面的增强方向。

### 建议优先级
如果只能选一个更激进的扩展方向：
1. **方向 C** 作为主要方法贡献
2. **方向 D** 作为评估框架 / 结果呈现方式
3. **方向 B** 仅在时间充裕且代码基础允许 state quantization experiment 时再考虑
