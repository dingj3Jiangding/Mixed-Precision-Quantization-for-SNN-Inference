# Coding Task Brief: Add a More Paper-Like CIFAR-10 SNN Baseline

You are adding a stronger, more literature-like SNN baseline to this repository so that later quantization experiments are built on a backbone that is closer to common CIFAR-10 SNN papers than the current minimal prototype.

## Goal
Add a new baseline architecture that is more similar to common paper baselines while preserving compatibility with the repository’s existing experiment pipeline:

- FP32 baseline training/evaluation
- uniform weight quantization comparison
- Hessian-guided mixed-precision analysis

This task is **not** a full research rewrite. The purpose is to improve the baseline architecture while keeping the rest of the repo stable and reusable.

---

## Why this change is needed
The current baseline in `baseline/model.py` is intentionally simple:
- `conv1`
- `conv2`
- `fc1`
- `fc2`

This is useful as a prototype, but it has an important limitation:
- `fc1` dominates total parameter count
- layer-wise mixed-precision allocation becomes unstable because one large FC layer controls most of the average bit-width
- the model is less similar to the multi-conv baselines commonly used in CIFAR-10 SNN literature

The new baseline should reduce dependence on a single huge FC hidden layer and provide a more balanced set of quantizable layers.

---

## Recommended architecture direction
Implement a **conservative VGG-like convolutional SNN baseline for CIFAR-10**.

Target properties:
- more than 2 convolutional layers
- repeated `Conv -> BatchNorm -> LIF` blocks
- periodic pooling between conv stages
- lightweight classifier head
- avoid one giant hidden FC layer dominating total parameters
- keep the model compatible with multi-step SNN execution

Recommended style:
- stay inside `spikingjelly.activation_based`
- continue using `layer.Conv2d`, `layer.BatchNorm2d`, `layer.Linear`, `neuron.LIFNode`
- keep `functional.set_step_mode(self, step_mode="m")`

### Good fit examples
A small paper-like CIFAR-10 SNN could look conceptually like:
- conv block 1: 2 conv layers
- pool
- conv block 2: 2 conv layers
- pool
- conv block 3: 1-2 conv layers
- small classifier head

The exact channel sizes can be chosen conservatively to keep training manageable.

### Avoid in this task
Do **not** turn this into:
- a ResNet rewrite
- ANN-to-SNN conversion workflow
- transformer-based baseline
- major dataset expansion
- a large framework refactor

This change should remain a **baseline upgrade**, not a new research branch.

---

## Preferred implementation strategy
Safer default:

### Add a new stronger baseline option without deleting the old one
This is preferable to immediately replacing the old model because it:
- preserves backward compatibility
- allows old-vs-new baseline comparison
- avoids breaking existing checkpoints and analysis unexpectedly

Possible pattern:
- keep the current simple baseline as one option
- add a second, stronger baseline model
- introduce a small model-selection mechanism only if needed

Only replace the old default completely if the user explicitly decides to fully migrate.

---

## Files likely to change
Primary likely targets:
- `baseline/model.py`
- `baseline/__init__.py` (only if exports change)
- `scripts/run_baseline.py` (only if model selection is added)
- `baseline/unquant_runner.py` (only if small compatibility adjustments are needed)
- `baseline/uniform_runner.py` (verify checkpoint compatibility)
- `baseline/hessian.py` (verify deeper layer ranking still works cleanly)

Mandatory human-doc updates for every modified code file:
- `human_docs/baseline/model.py.md`
- plus matching `human_docs/...` docs for any other modified files

Do not skip the human-doc requirement.

---

## Existing code that should be reused
Reuse the current repo infrastructure as much as possible.

### Config and data
- `baseline/config.py` -> `BaselineConfig`
- `baseline/data.py` -> `build_cifar10_loaders`

### Training / evaluation pipeline
From `baseline/unquant_runner.py`, reuse where possible:
- `set_global_seed`
- `direct_encode`
- `train_one_epoch`
- `evaluate`
- `write_epoch_metrics_csv`

### Metrics and reporting
- `baseline/metrics.py`
  - parameter counting
  - model size helpers
  - SOP proxy helpers

### Quantization helpers
- `baseline/quantization.py`
- existing uniform quantization path in `baseline/uniform_runner.py`
- existing Hessian analysis flow in `baseline/hessian.py`

The new baseline should fit into these existing flows instead of creating parallel pipelines.

---

## Hard compatibility constraints
These constraints are important.

1. Keep **CIFAR-10** as the baseline dataset.
2. Preserve the model input contract:
   - input shape must remain `[T, B, C, H, W]`
3. Preserve the current **direct encoding** path unless there is a strong reason not to.
4. Continue using SpikingJelly activation-based modules.
5. Keep reset behavior compatible with existing training/evaluation code.
6. Do not break output locations and schemas already used by:
   - `outputs/baseline/`
   - `outputs/uniform_quant/`
   - `outputs/hessian_sensitivity/`
7. Keep metric names stable where possible:
   - `test_acc`
   - `spike_rate`
   - `avg_batch_infer_ms`
   - `sop_proxy`
   - related csv/json fields already used downstream
8. Preserve Python 3.9 compatibility.

---

## Practical architecture guidance
The main design objective is **better parameter distribution**, not maximum novelty.

Desired outcomes:
- more quantizable layers than the current 4-layer weight set
- less concentration of parameters in one FC layer
- better suitability for layer-wise mixed precision experiments
- baseline accuracy that is more representative of paper-style CIFAR-10 SNNs

### Strong recommendation
Prefer a classifier head that is much lighter than the current `128 * 8 * 8 -> 256 -> 10` style hidden FC bottleneck.

If flattening is still used, keep the FC head small.
If possible, prefer a more convolution-heavy design before the final classifier.

---

## Verification requirements
After implementation, verify the new baseline through the existing pipeline.

### 1) Baseline training
Run:
```bash
python scripts/run_baseline.py --epochs 10 --device cuda
```
Confirm:
- training runs successfully
- checkpoint is created
- `outputs/baseline/epoch_metrics.csv` is written
- `outputs/baseline/summary.json` is written

### 2) Uniform quantization compatibility
Run:
```bash
python scripts/run_uniform_quant.py --checkpoint-path outputs/baseline/fp32_last.pt --bits 8,4 --device cuda
```
Confirm:
- the checkpoint loads correctly
- `outputs/uniform_quant/uniform_comparison.csv` is produced
- quantized evaluation still works on the new architecture

### 3) Hessian compatibility
Run:
```bash
python scripts/run_hessian_sensitivity.py --checkpoint-path outputs/baseline/fp32_last.pt --bits 8,4 --device cuda
```
Confirm:
- deeper layer collection works
- sensitivity ranking is produced
- bit allocation csv is generated
- mixed-precision evaluation still completes

### 4) Structural sanity checks
Compare old vs new baseline on:
- total parameter count
- number of quantizable Conv/Linear layers
- whether parameter distribution is more balanced
- whether mixed-precision assignment is less dominated by one layer

### 5) Reproducibility sanity check
Run the baseline more than once if practical and confirm results are reasonably stable.

---

## Suggested success criteria
This task should be considered successful if:
- a stronger baseline model is added cleanly
- the repository’s existing baseline/uniform/Hessian scripts still work
- the model has a more paper-like convolutional structure
- parameter distribution is less pathological than the current huge-`fc1` design
- documentation in `human_docs/` is updated for every modified code file

---

## Important non-goals
This task should **not** attempt to solve all research goals at once.

Do not bundle in:
- state quantization
- time-step search
- new datasets
- hardware mapping
- QAT redesign beyond what is needed for compatibility
- large-scale architecture search

Keep the scope focused: **upgrade the baseline backbone while preserving the existing experimental pipeline**.
