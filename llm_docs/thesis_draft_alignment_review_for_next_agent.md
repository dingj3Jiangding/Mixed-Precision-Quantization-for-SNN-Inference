# Thesis Draft Alignment Review For Next Agent

This document is for a future agent who will revise the current thesis draft.

Reviewed artifact:

- `/Users/fudijie/Downloads/ST4001_paper__1_.pdf`

Reference project strategy docs:

- `paper_docs/project_thesis_outline_zh.md`
- `paper_docs/paper_organization_strategy.md`
- `paper_docs/thesis_agent_briefing.md`
- `human_docs/contributions/formal_experiment_plan_for_thesis.md`

The goal of this review is **not** to judge whether the current numerical results are final.  
The main question is:

> Aside from placeholder / non-final experiment numbers, does the draft match the current thesis design and contribution positioning?

Short answer:

- **Overall, yes.**
- The draft is broadly aligned with the current thesis strategy.
- However, several **structural and emphasis-level mismatches** still need correction before the final writing pass.

---

## 1. What Is Already Aligned

The following major thesis-level decisions are already consistent with the current design:

### 1.1 Title positioning is correct

The draft title:

- `Distillation-Assisted Mixed-Precision Quantization for Spiking Neural Network Inference`

This is aligned with the current strategy, which now treats:

- mixed precision as the core framework
- distillation as the strongest added contribution

### 1.2 Novelty claims are already restrained

The draft correctly avoids claiming that:

- Hessian-guided SNN quantization is newly invented here
- state-aware SNN quantization is newly invented here

This is important and should be preserved.

### 1.3 Method hierarchy is mostly correct

The draft already reflects the intended method roles:

- `Hessian-guided mixed precision` = main quantization baseline line
- `State-aware Hessian` = extension / analysis line
- `Distillation-assisted quantization` = main training enhancement
- `Weight-state mixed precision` = exploratory proxy

### 1.4 Baseline family separation is correct

The draft properly distinguishes:

- `baseline/` as early prototype
- `baseline_full/` as compact development line
- `baseline_vgg16/` as main CIFAR-10 paper backbone
- `baseline_nmnist/` as supporting event-driven line

This is consistent with repo reality and current thesis strategy.

### 1.5 Proxy / limitation wording is mostly disciplined

The draft already correctly states that:

- current efficiency metrics are proxies rather than hardware measurements
- current weight-state method is a proxy rather than full membrane-state quantization

This is aligned with the required caution level.

---

## 2. Main Remaining Mismatches

These are the key issues that still need revision.  
They are **not primarily about missing final numbers**, but about thesis organization and argument shape.

## 2.1 `State-aware` is still slightly too prominent relative to `KD`

Although the draft no longer overclaims novelty, the reader still feels that:

- `State-aware Hessian`
- `Distillation-assisted quantization`

are presented as almost equally central.

This is not the preferred final positioning.

Current thesis strategy should feel more like:

1. `HessianMixed` is the main baseline quantization path
2. `StateAwareHessian` is an important SNN-specific extension
3. `HessianMixed + KD` is the strongest added enhancement and one of the main headline results

### Required adjustment

In the revised draft:

- reduce the “core-method” feeling of `State-aware`
- increase the “main added contribution” feeling of `KD`

Places where this should be reflected:

- abstract
- contribution list
- method chapter emphasis
- result chapter ordering / table design
- discussion section

---

## 2.2 The draft is missing a dedicated “Efficiency Analysis Beyond Accuracy” result section

This is the clearest structural gap.

The current thesis strategy explicitly requires that the paper not rely on accuracy alone.  
It should separately discuss:

- average weight bits
- model size
- compression ratio
- inference latency
- SOP proxy
- bit-weighted SOP
- state-related proxy where applicable

But in the current draft, Chapter 5 does not contain a dedicated result section for this.

### Current issue

Efficiency discussion is present in fragments, but not consolidated into one explicit result section.

This weakens the central thesis claim:

> the work studies the accuracy-efficiency trade-off, not just accuracy.

### Required adjustment

Add a distinct section such as:

- `5.x Efficiency Analysis Beyond Accuracy`

This section should summarize the deployment-oriented metrics across methods rather than scattering them only inside method-specific subsections.

Recommended placement:

- after method-specific result sections
- before or near final discussion

---

## 2.3 The KD result should be presented with a stricter controlled comparison

The draft correctly says in words that the right KD comparison is:

- same teacher checkpoint
- same student architecture
- same bit allocation
- same evaluation setting
- only KD on/off differs

However, the actual result presentation still leans too much on comparing:

- FP32
- UniformW4
- HessianMixed + KD

rather than the most important comparison:

- `HessianMixed`
- `HessianMixed + KD`

### Why this matters

The thesis claim for KD is not:

- “KD gives the best absolute result in a loose table”

The correct claim is:

> Under the same deployment budget and same student structure, KD improves the low-bit mixed-precision student.

### Required adjustment

The revised draft should include a dedicated controlled-comparison table for KD:

- `HessianMixed`
- `HessianMixed + KD`

with explicit note that the student allocation and deployment class are unchanged.

This should likely become one of the main tables in Chapter 5.

---

## 2.4 N-MNIST is slightly too visible for its intended role

The current strategy treats N-MNIST as:

- supporting evidence
- not the main thesis backbone

The draft already says this in text, which is good.

However, its placement still gives it slightly more weight than ideal, especially if the final formal CIFAR-10 line remains the main evidence base.

### Required adjustment

N-MNIST should stay in the thesis, but its role should remain clearly secondary:

- shorter subsection
- less headline emphasis
- more explicit phrasing that it supports generality rather than carrying the main thesis claim

If needed, some supporting details can move to appendix.

---

## 2.5 Related-work subsection title for state-aware quantization could be more conservative

Current draft subsection title:

- `2.6 State-Aware SNN Quantization`

This is not wrong, but it still sounds slightly too tailored to the thesis method label.

### Safer alternatives

Consider renaming to one of:

- `State-Related Considerations in SNN Quantization`
- `Stateful and Membrane-Aware Quantization in SNNs`

This makes the background chapter sound more like related work and less like a direct pre-labeling of the thesis contribution.

---

## 2.6 Abstract wording is still a little too “project report” in tone

The draft abstract contains expressions such as:

- `The project builds ...`
- `Existing experiment records show ...`

These are acceptable for a draft, but not ideal for the final serious thesis tone.

### Required adjustment

Prefer more thesis-style phrasing such as:

- `This thesis develops ...`
- `Experimental results indicate ...`
- `Recorded preliminary results suggest ...`

This is a writing-quality issue rather than a research-direction issue, but it should be cleaned up.

---

## 2.7 “Formal result” vs “recorded validation result” separation is conceptually correct, but still visually weak

The draft already explains that:

- some results are formal full-test results
- some results are recorded validation / limited-batch results

This is good.

But the presentation still risks letting a reader visually interpret all tables as equally formal.

### Required adjustment

In the revised draft:

- only final unified-protocol results should appear as main formal tables
- limited-batch / recorded validation results should be clearly marked as:
  - preliminary
  - validation-only
  - supporting
  - or moved to appendix / smaller tables if appropriate

The visual hierarchy should match the methodological discipline.

---

## 3. Recommended Structural Revisions

The next agent should consider the following revision direction for Chapter 5:

### Current rough structure

- 5.1 CIFAR-10 FP32 and Uniform Quantization
- 5.2 Hessian-Guided Mixed Precision
- 5.3 State-Aware Extension
- 5.4 Distillation-Assisted Quantization
- 5.5 N-MNIST Supporting Results
- 5.6 Weight-State Exploratory Study
- 5.7 Discussion

### Recommended revised structure

- 5.1 CIFAR-10 FP32 and Uniform Quantization
- 5.2 Hessian-Guided Mixed Precision
- 5.3 State-Aware Extension
- 5.4 Distillation-Assisted Quantization
- 5.5 Efficiency Analysis Beyond Accuracy
- 5.6 N-MNIST Supporting Results
- 5.7 Weight-State Exploratory Study
- 5.8 Discussion

This revised structure better matches the thesis claim that deployment trade-off is central.

---

## 4. Recommended Table Priorities

The next agent should make sure the final result presentation emphasizes the following table logic.

### Table priority A: Main CIFAR-10 accuracy-efficiency comparison

Recommended methods:

- FP32
- UniformW8
- UniformW4
- HessianMixed
- HessianMixed + KD
- StateAwareHessian

Recommended columns:

- accuracy
- accuracy drop vs FP32
- avg weight bits
- model size
- compression ratio
- inference time
- SOP proxy
- bit-weighted SOP

### Table priority B: KD-specific controlled comparison

This is currently under-emphasized and should be made explicit.

Compare only:

- HessianMixed
- HessianMixed + KD

Purpose:

- same student
- same allocation
- same deployment cost class
- only KD changes

### Table priority C: State-aware analysis table

This should focus less on inflated headline accuracy and more on:

- Hessian score
- state proxy
- combined score
- resulting bit allocation logic

This helps keep state-aware framing honest and analytical.

---

## 5. Do Not Misread The Review

This review does **not** say the draft is fundamentally wrong.

The correct interpretation is:

- the draft is already largely consistent with the current thesis direction
- the remaining work is mainly about tightening emphasis, hierarchy, and result presentation logic

In other words:

- **no major conceptual rewrite is required**
- but **several important structural refinements are still needed**

---

## 6. Immediate Revision Priorities For The Next Agent

If time is limited, prioritize the following in order:

1. Add a dedicated `Efficiency Analysis Beyond Accuracy` section.
2. Strengthen KD as the main added contribution and reduce the relative prominence of `State-aware`.
3. Add or redesign the KD controlled-comparison table: `HessianMixed` vs `HessianMixed + KD`.
4. Reduce the visual / rhetorical prominence of N-MNIST to a supporting role.
5. Make the distinction between formal results and recorded validation results visually stricter.
6. Clean up abstract and related-work wording for more formal thesis tone.

---

## 7. Bottom-Line Assessment

If another agent is asked:

> “Does the current draft basically match the latest thesis design?”

The correct answer is:

> Yes, mostly. The draft is directionally aligned. The main remaining issues are not topic mismatch, but emphasis mismatch, result-structure mismatch, and incomplete separation between formal and preliminary evidence.

