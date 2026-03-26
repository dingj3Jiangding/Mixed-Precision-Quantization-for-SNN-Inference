# Research Status and Catch-Up Plan

## Document Purpose
This document is intended to be directly usable by a coding-oriented model or collaborator. It summarizes:
- current research status
- current level of understanding of the project
- the main gap versus the original schedule
- the highest-priority next actions to produce deliverable results as quickly as possible

---

## 1. Current Research Status

### Overall assessment
The project has already moved beyond the stage of vague topic exploration. The research direction is now substantially narrowed to:

**Mixed-precision quantization for SNN inference**

However, the project has **not yet fully transitioned into a stable experimental production phase**. In terms of actual progress, the project is currently best described as:

- **problem definition mostly completed**
- **literature landscape largely established**
- **research hypotheses partially formed**
- **baseline and experimental pipeline not yet fully closed**
- **first strong deliverable results not yet produced**

### Practical stage estimate
Relative to the original schedule, the project is closer to:
- **late Phase 1 (probing / sensitivity analysis design)**
- **early Phase 2 (mixed-precision design preparation)**

rather than being truly ready for Phase 3/4 execution.

### What has already been achieved
1. The research topic has been narrowed from general SNN efficiency / quantization to a clearer problem:
   - mixed-precision quantization for SNN inference
   - with emphasis on the coupling between weight precision, state precision, and time steps

2. A structured literature understanding has been built. Existing work can already be grouped into several main lines:
   - system-level SNN quantization
   - mixed-precision / bit-width allocation
   - time-step / precision tradeoff
   - post-training or one-shot compression

3. The key difficulty of the problem is understood:
   - SNN efficiency is not guaranteed automatically by sparsity
   - state storage and memory access can become major bottlenecks
   - quantization can change spike patterns and therefore alter actual inference cost
   - this is a system problem, not only an algorithm problem

### What is still missing
The main missing piece is **a short, stable, reproducible experimental path that produces visible results quickly**.

The project currently has good conceptual preparation, but not yet a completed result chain like:

**baseline -> uniform quantization -> mixed-precision result -> comparison figure/table**

---

## 2. Current Level of Understanding

At this point, the project understanding is already beyond simple paper reading.

### I already understand the following well

#### 2.1 Why SNN quantization is different from ANN quantization
SNN quantization is not only about weights and activations. It also involves:
- membrane/state variables
- thresholds / leak-related quantities
- temporal unrolling across time steps
- training dynamics under surrogate gradient / BPTT

So the real optimization variables are not just weight bit-width, but a coupled space involving:
- **W = weight precision**
- **U = state / membrane precision**
- **T = number of time steps**

#### 2.2 Why this is a system-level optimization problem
The final efficiency of SNN inference depends on the combined effect of:
- arithmetic cost
- state storage cost
- memory access cost
- spike activity pattern
- hardware constraints

Therefore, a good solution cannot be judged only by accuracy. It must also be evaluated by multiple metrics such as:
- accuracy
- model size
- state/storage cost
- spike rate
- SOPs / operation proxy
- energy or memory proxy metrics

#### 2.3 What kind of result would count as meaningful research progress
A meaningful result should not just say "lower precision works." It should show:
- a reproducible baseline
- a controlled comparison under fixed experimental settings
- a quantitative tradeoff curve or Pareto comparison
- ideally an interpretable allocation strategy, not just brute-force search

### What I still do not fully have yet
I still have not fully locked the project into **one single primary research line**.

Currently there are multiple plausible directions:
1. Hessian-guided layer-wise mixed precision
2. state-priority mixed precision
3. joint precision-time-step allocation

These are all valid, but trying to advance all of them at once will slow down result production.

---

## 3. Gap Versus the Original Schedule

The biggest gap is **not lack of reading**, but **lack of finished deliverable experiments**.

### The real delays are:

#### 3.1 Baseline closure is not complete enough
The original Phase 0 required stable setup and benchmark reproduction. But right now, the most important missing output is still:
- a reproducible baseline table
- stable scripts
- fixed dataset / model / evaluation settings

#### 3.2 Sensitivity analysis is still more of a research design than a finished output
The original Phase 1 planned probing and sensitivity heatmaps. At present, the idea is clear, but the actual deliverables are not yet available:
- no finalized sensitivity heatmap
- no stable layer ranking result
- no complete bit-allocation output yet

#### 3.3 The mixed-precision design path is still too broad
The original Phase 2 assumed that the allocation problem would already be operationalized. In reality, the project still needs one clearly chosen path before complex search / QAT / hardware evaluation makes sense.

#### 3.4 QAT and hardware mapping should not be immediate top priority
Given the current stage, directly pushing QAT refinement or hardware mapping would likely spread effort too thin. The first priority must be to create a minimal but convincing set of results.

---

## 4. Top Priority Principle From Now On

### Core principle
**Deliverable result first. Completeness of the grand plan second.**

This means the project should now prioritize the shortest path that can produce:
- a figure
- a table
- a comparison result
- a reproducible script pipeline

before expanding to more ambitious extensions.

### The best short path
The most practical minimal research pipeline is:

**Full-precision baseline -> uniform quantization -> Hessian-guided mixed precision -> Pareto comparison**

This path should be prioritized because it is:
- interpretable
- feasible within limited time
- aligned with current understanding
- likely to generate the first strong deliverables quickly

---

## 5. Recommended Main Research Line

### Recommended primary line
Use the following as the main line for the next stage:

**Hessian-guided layer-wise mixed-precision quantization for SNN inference**

### Why this should be the main line
1. It is easier to implement than RL-based or fully differentiable search.
2. It is more interpretable than black-box search.
3. It naturally produces useful deliverables:
   - sensitivity heatmap
   - layer ranking
   - bit allocation table
   - comparison with uniform quantization
4. It can later be extended toward:
   - state-priority mixed precision
   - time-step / precision co-optimization

### What should not be done immediately
Until the first strong results are obtained, avoid expanding effort into too many branches such as:
- full RL search
- complicated ILP pipeline
- complete QAT refinement for many variants
- detailed hardware mapping for multiple accelerators
- too many datasets/models at once

---

## 6. Immediate Action Plan (Highest Priority First)

## Priority 1: Close the baseline pipeline
### Goal
Produce a fully reproducible baseline training/evaluation pipeline.

### Required outputs
- fixed framework and environment
- fixed dataset(s)
- fixed model(s)
- fixed logging / metric scripts
- one reproducible baseline result table

### Concrete actions
- choose a minimal benchmark set
  - 1 static dataset
  - 1 event-based dataset if time allows
- choose 1-2 baseline architectures only
- standardize training and evaluation config
- save results in a unified format

### Definition of done
A new run can reproduce baseline numbers with acceptable variance using one command or one documented script chain.

---

## Priority 2: Produce uniform quantization results
### Goal
Create the first real comparison table.

### Required outputs
- full precision result
- W8 / W4 / W2 comparison
- if possible, one small W/U grid result

### Concrete actions
- implement or reuse a simple uniform quantization path
- run controlled experiments under fixed settings
- collect:
  - accuracy
  - spike rate
  - model size
  - SOP proxy / compute proxy

### Definition of done
At least one table exists that clearly shows the tradeoff between full precision and uniform low precision.

---

## Priority 3: Implement Hessian-based sensitivity analysis
### Goal
Turn the research idea into a visible algorithmic result.

### Required outputs
- per-layer sensitivity scores
- sensitivity heatmap or ranking chart
- one bit-allocation strategy derived from the sensitivity result

### Concrete actions
- decide the exact sensitivity estimator
- run layer-wise analysis on the chosen baseline model
- visualize layer importance
- map sensitivity ranking to a bit allocation policy under a fixed budget

### Definition of done
There is at least one heatmap / ranking figure and one derived mixed-precision configuration.

---

## Priority 4: Compare mixed precision against uniform quantization
### Goal
Produce the first convincing research figure.

### Required outputs
- comparison table: uniform vs Hessian-guided mixed precision
- at least one Pareto-style plot

### Concrete actions
- keep total bit budget comparable
- compare under the same model / dataset / time-step setting
- report both accuracy and resource-related metrics

### Definition of done
A result figure/table clearly shows whether Hessian-guided MPQ gives a better tradeoff than uniform quantization.

---

## Priority 5: Only then consider the second innovation point
### Candidate extension choices
After the first result chain is stable, choose only one of the following:

#### Option A: state-priority mixed precision
Focus on heterogeneous precision for membrane/state variables.

#### Option B: joint time-step / precision allocation
Study whether temporal budget can substitute part of precision budget.

### Recommendation
If the goal is fastest progress toward a strong deliverable, choose:
- **state-priority mixed precision first**

because it is more tightly connected to the actual bottleneck understanding already formed in this project.

---

## 7. Suggested 4-6 Week Catch-Up Schedule

## Week 1
### Objective
Finish baseline closure.

### Deliverables
- final selected dataset/model list
- working training/evaluation pipeline
- first baseline table

## Week 2
### Objective
Finish uniform quantization experiments.

### Deliverables
- W8/W4/W2 result table
- basic tradeoff observations

## Week 3
### Objective
Finish Hessian sensitivity analysis.

### Deliverables
- layer sensitivity ranking
- heatmap / visualization
- first mixed-precision allocation table

## Week 4
### Objective
Finish main comparison and reporting package.

### Deliverables
- uniform vs mixed-precision comparison table
- Pareto figure
- short internal report or checkpoint slides

## Week 5-6
### Objective
Add one extension only if the first pipeline is stable.

### Deliverables
- either state-priority extension
- or time-step / precision extension
- but not both at once

---

## 8. Deliverables to Target Immediately

### Deliverable 1 (highest priority)
**Reproducible baseline repo + run script**

### Deliverable 2
**Uniform quantization comparison table**

### Deliverable 3
**Hessian sensitivity heatmap + layer-wise allocation result**

### Deliverable 4
**Pareto plot: accuracy vs resource cost**

### Deliverable 5
**Short technical summary for mentor/checkpoint use**

---

## 9. Final Summary

The project is currently in a strong conceptual state but a weak experimental-delivery state.

That means:
- the topic is no longer vague
- the literature understanding is already structured
- the core technical difficulty is already understood
- the missing piece is not more reading, but a short path to visible, reproducible results

Therefore, the correct next move is not to expand the scope, but to compress it.

### Final recommendation
For the next stage, the project should be executed in this order:

1. baseline closure
2. uniform quantization
3. Hessian-guided mixed precision
4. Pareto comparison
5. one extension only after the above is stable

### One-sentence project status
The project has already completed direction narrowing and problem understanding, but now urgently needs to convert that understanding into a minimal reproducible result pipeline with deliverable figures and tables.

