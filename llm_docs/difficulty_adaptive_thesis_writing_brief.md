# Difficulty-Adaptive Thesis Writing Brief

This document is for the LLM/agent responsible for revising the thesis draft.

It reflects the **current final paper direction** and should override older draft assumptions centered on Hessian+KD or other abandoned exploratory branches.

---

## 1. Current final paper direction

The thesis should now be organized around:

- **Difficulty-Adaptive State Precision for SNN Inference**

The final implementation to describe is:

- **batch-wise two-stage mode**

Do **not** present the paper as mainly about:

- Hessian-guided mixed precision
- allocation-aware KD
- state-write-aware quantization
- old sample-wise adaptive routing

Those can only appear as old exploration, discarded branch, or future work if absolutely needed.

---

## 2. Core claim the paper should make

The safe claim is:

- We propose a **difficulty-adaptive state precision policy** for SNN inference.
- Instead of applying a fixed low state precision to all samples and all timesteps, the method uses a short high-precision warmup, estimates sample difficulty from early predictions, and then assigns lower/higher state precision to easy/hard samples for the remaining timesteps.
- The method aims to **recover accuracy relative to fixed low state precision**, while retaining a **modest reduction** in average state precision cost.

The paper should **not** claim:

- large resource savings with almost no accuracy cost
- universal gains across all settings
- exact hardware-level state-kernel implementation

This is a **proxy state-precision study**, not a production hardware kernel paper.

---

## 3. How methodology should be written

The methodology section should be rebuilt around four parts.

### 3.1 Problem Setup

Explain:

- SNN inference cost is affected not only by weight precision, but also by the precision used for dynamic state-related computation.
- A fixed low state precision applied uniformly across all samples and timesteps can degrade accuracy.
- Sample difficulty observed in early timesteps suggests that later state precision does not need to be uniform.

Key framing:

- This is **state-side adaptive precision**
- This is **not** weight mixed precision
- This is **not** KD

### 3.2 Two-Stage Difficulty-Adaptive Policy

Describe the final policy precisely:

- Use high state precision for the first `T_w` timesteps.
- Accumulate early logits.
- Compute a difficulty score using prediction confidence.
- If confidence exceeds threshold `tau`, mark the sample as easy.
- For the remaining timesteps:
  - easy samples use low state bits `b_l`
  - hard samples use high state bits `b_h`

Suggested symbols:

- `T`: total timesteps
- `T_w`: warmup timesteps
- `b_h`: high state bits
- `b_l`: low state bits
- `s_i`: sample difficulty/confidence score
- `m_i = 1[s_i >= tau]`: easy mask

Include either:

- one short algorithm block
- or one compact pseudo-code figure

### 3.3 Batch-Wise Two-Stage Grouped Execution

This must be explicit.

The final implementation is **not** free-form per-sample routing at every step.

It is:

- warmup at high precision
- estimate easy/hard split
- grouped easy/hard execution for remaining timesteps

Explain that this grouped execution is the final practical implementation used in experiments.

If old drafts mention sample-wise routing as the main implementation, update them.

### 3.4 State Precision Proxy Formulation

This section is essential for correctness.

The final implementation does **not** rewrite the neuron internal membrane update kernel directly.

Instead, the current experiment approximates state precision effects by:

- quantizing the continuous inputs to each LIF node

Recommended wording:

- "We study a proxy formulation of adaptive state precision by quantizing continuous inputs to each LIF node, rather than modifying the neuron’s internal membrane update kernel directly."

Do not hide this limitation.

---

## 4. How experiments should be organized

The experiments section should have a clear hierarchy.

### 4.1 CIFAR-10 / VGG16 = main result dataset

This is the most important dataset.

The paper should treat CIFAR-10 as the dataset that determines whether the method stands.

The main table should include:

- `FP32State`
- `FixedStateHighB8`
- `FixedStateLowB7`
- `AdaptiveStateB7to8`

Suggested metrics:

- test accuracy
- accuracy drop vs FP32State
- average state bits used
- estimated state bytes per sample
- easy fraction post warmup

Current key result:

- `FixedStateLowB7 = 0.8033`
- `AdaptiveStateB7to8 = 0.8411`
- gain over fixed low = `+3.78 percentage points`
- `adaptive_avg_state_bits_used = 7.7669`
- `adaptive_easy_fraction_post_warmup = 0.4663`

Interpretation:

- the method substantially recovers accuracy over fixed low state precision
- while retaining a modest reduction in average state precision cost

Important:

- describe the resource gain as **modest**
- do not oversell it as large compression

### 4.2 N-MNIST = supporting dataset

N-MNIST should be clearly secondary.

Use it to show:

- the same policy also works positively on a simpler SNN benchmark

Suggested table structure can mirror CIFAR.

Current key result:

- `FixedStateLowB7 = 0.9498`
- `AdaptiveStateB7to8 = 0.9561`
- gain over fixed low = `+0.63 percentage points`
- `adaptive_avg_state_bits_used = 7.7691`

Interpretation:

- a modest positive trade-off
- supporting evidence, not the main proof of the method

---

## 5. What should be removed or downgraded

### 5.1 Related work

Related work should no longer be centered around:

- Hessian mixed precision
- KD-enhanced mixed precision
- write-aware proxy branches

Instead, related work should be reorganized around:

- SNN quantization
- dynamic / adaptive precision
- conditional inference / early confidence-based routing
- temporal adaptation in SNN inference if relevant

If older text still frames the thesis as Hessian+KD centered, rewrite it.

### 5.2 Old method figures

Remove or replace any figure that visually centers:

- Hessian sensitivity ranking
- allocation-aware KD pipeline
- write-aware state skipping

The main method figure should now be:

- warmup phase
- difficulty estimation
- easy/hard split
- low/high state precision allocation

### 5.3 Old results narrative

Downgrade or delete text that suggests:

- the main novelty is Hessian-guided weight allocation
- the main gain comes from KD
- write-aware state suppression is the final solution

Those are no longer aligned with the current thesis direction.

---

## 6. What the paper can safely claim now

Safe claims:

- The method improves over fixed low state precision on CIFAR-10.
- The method also shows positive supporting behavior on N-MNIST.
- The proposed policy preserves a modest reduction in average state precision cost.
- Early prediction confidence can serve as a useful signal for state precision allocation in SNN inference.

Unsafe claims:

- the method delivers large hardware savings
- the method is universally optimal
- the method directly implements low-level adaptive neuron-state kernels
- the method is already stronger than all weight-side quantization approaches

---

## 7. Recommended writing tone for the results section

Use language like:

- "accuracy recovery over fixed low precision"
- "modest reduction in average state precision cost"
- "supporting evidence on N-MNIST"
- "proxy formulation of adaptive state precision"

Avoid language like:

- "dramatic resource reduction"
- "negligible accuracy cost"
- "strong universal generalization"

---

## 8. Self-checklist for the thesis-writing agent

Before finalizing the draft, verify all of the following.

### 8.1 Title / abstract / introduction

- Does the title clearly center **Difficulty-Adaptive State Precision**?
- Does the abstract avoid framing the thesis as Hessian+KD?
- Does the introduction state the real target: improving over fixed low state precision with modest state-cost savings?

### 8.2 Methodology

- Is the final implementation described as **batch-wise two-stage mode**?
- Is the warmup + confidence + easy/hard split process explained clearly?
- Is the proxy nature of state precision explicitly acknowledged?
- Are obsolete branches excluded from the main method?

### 8.3 Experiments

- Is CIFAR-10 clearly the main dataset?
- Is N-MNIST clearly supporting?
- Are the final CIFAR numbers updated to:
  - `0.8033` for `FixedStateLowB7`
  - `0.8411` for `AdaptiveStateB7to8`
  - `7.7669` average state bits
  - `0.4663` easy fraction
- Are the N-MNIST supporting numbers updated to:
  - `0.9498` for `FixedStateLowB7`
  - `0.9561` for `AdaptiveStateB7to8`
  - `7.7691` average state bits

### 8.4 Figures and tables

- Does the main method figure match the two-stage adaptive policy?
- Do all tables use the updated final method names?
- Are old Hessian/KD/write-aware figures removed from the main paper if they no longer match the final story?

### 8.5 Related work

- Has related work been refocused on adaptive precision / conditional inference / SNN quantization?
- Are old literature groupings from the Hessian+KD storyline removed or reduced?

---

## 9. Bottom line for the writing agent

The thesis is now about:

- **difficulty-conditioned adaptive state precision**

The paper should present:

- CIFAR-10 as the main positive result
- N-MNIST as supporting positive evidence
- modest but real state-cost reduction
- accuracy recovery over fixed low state precision

The paper should not present:

- Hessian+KD as the main story
- failed exploratory branches as central evidence
- exaggerated resource-saving claims
