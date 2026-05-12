# Coding Task Brief: Rewrite Hessian Analysis to Follow Lui & Neftci 2021

You are rewriting the Hessian analysis pipeline in this repository so that it follows **Lui & Neftci 2021, "Hessian Aware Quantization of Spiking Neural Networks"** much more closely than the current prototype implementation.

## Goal
Replace the current Hessian workflow, which is based on:
- a `grad^2 * weight^2` sensitivity proxy
- a custom greedy bit-budget allocator
- immediate evaluation after quantization

with a new workflow centered on:
- **layer-wise Hessian trace estimation**
- **Hutchinson probes + Hessian-vector products**
- **conservative, ranking-based layer-wise bit assignment**
- **post-quantization fine-tuning of the mixed-precision model**
- final comparison/reporting after fine-tuning

This is not a small patch. Treat it as a **methodological rewrite of the Hessian pipeline**, while still reusing as much of the repository’s existing baseline/data/training/evaluation infrastructure as possible.

---

## Files to modify
Primary targets:
- `baseline/hessian.py`
- `scripts/run_hessian_sensitivity.py`

Modify only if needed:
- `baseline/__init__.py`

Mandatory doc updates:
- `human_docs/baseline/hessian.py.md`
- `human_docs/scripts/run_hessian_sensitivity.py.md`

---

## Existing code that should be reused
Prefer reusing these functions/utilities instead of duplicating logic.

### From `baseline/unquant_runner.py`
- `set_global_seed`
- `direct_encode`
- `_iter_with_limit`
- `evaluate`
- `write_epoch_metrics_csv`

### From `baseline/data.py`
- `build_cifar10_loaders`

### From `baseline/model.py`
- `build_model`

### From `baseline/metrics.py`
- `parameter_count`
- `synapse_count_proxy`
- `sop_proxy`

### From `baseline/quantization.py`
- `parse_bits_list`
- basic symmetric quantization helpers, if useful

---

## Current logic that should no longer be the core method
The following parts of the current Hessian implementation should be removed, replaced, or demoted from the main path:
- the `grad^2 * weight^2` proxy in `estimate_layer_sensitivity()`
- the greedy budget allocator in `allocate_bits_by_budget()`
- the old `_quantize_model_by_layer_bits()` path if it only supports “quantize once then evaluate immediately”
- the current `run_hessian_sensitivity_analysis()` structure that directly compares `FP32 / Uniform / HessianMixed` without post-quantization fine-tuning

You may reuse small helper structure where useful, but do **not** preserve these as the core algorithm.

---

## Target workflow to implement
Rework `run_hessian_sensitivity_analysis(...)` into this sequence:

1. Set seed and resolve device.
2. Build train/test loaders.
3. Build the model and load the FP32 checkpoint.
4. Estimate **layer-wise Hessian trace** on the FP32 model using training data.
5. Export a trace ranking table.
6. Derive a **conservative, interpretable** layer-wise bit assignment from the ranking.
7. Evaluate an FP32 reference.
8. Optionally evaluate a uniform reference (recommended to preserve repo comparison flow).
9. Construct a mixed-precision quantized model initialized from the FP32 checkpoint.
10. Fine-tune the mixed-precision model for a configurable number of epochs.
11. Re-evaluate the fine-tuned mixed model.
12. Export artifacts: trace table, bit allocation table, fine-tuning epoch metrics, comparison table, summary JSON, optional ranking figure.

---

## 1) Hessian trace estimation requirements
Do **not** use the current squared-gradient proxy.

Implement a real Hutchinson-style layer-wise trace estimator:

### Mathematical target
For each quantizable layer:

`Tr(H_i) ≈ E[v^T H_i v]`

### Required implementation behavior
- Collect quantizable layers (`Conv` / `Linear`) similarly to the current `_collect_weight_layers()` helper.
- For each layer, draw one or more random probe vectors `v` with the same shape as the weight tensor.
- Normalize `v`.
- Compute gradients:
  - `g_i = dL / dW_i`
- Compute Hessian-vector products using autograd:
  - `H_i v = d(g_i^T v) / dW_i`
- Accumulate `v^T H_i v` across batches (and across probes).
- Average these values to obtain a per-layer Hessian trace estimate.

### Repository-specific constraints
- Keep using `direct_encode(x, t_steps)` for SNN input construction.
- Continue resetting network state with `functional.reset_net(model)` between batches.
- Expose runtime controls such as:
  - `--trace-probes`
  - `--max-hessian-batches`

### Suggested per-layer output fields
- `layer_name`
- `params`
- `hessian_trace`
- `trace_density` (optional helper field)
- `rank`
- `assigned_bits`

---

## 2) Replace the current greedy budget allocator
Do not keep `allocate_bits_by_budget()` as the default mixed-precision method.

The paper does **not** justify the current greedy budget-filling behavior, and this allocator has already shown pathological behavior in the repo (e.g. under-spending the intended budget and starving large layers).

### Replacement strategy
Use a **simple, conservative, paper-aligned ranking-based assignment rule**.

Recommended supported policies:

#### `rank-map`
- Sort layers by Hessian trace.
- Assign the highest precision to the most sensitive layers.
- Assign the lowest precision to the least sensitive layers.
- Map the intermediate layers accordingly.

#### `tiered`
- Split layers into coarse sensitivity bands (e.g. low / medium / high).
- Assign bit-widths by band.

### Important behavior
- The default policy should be conservative.
- Highly sensitive layers should retain the highest bit-width.
- Avoid extreme assignments caused purely by parameter-count effects.
- Do **not** claim this is an exact optimal allocator from the paper; it is a repo-level implementation choice inspired by the paper’s manual/guided layer-wise precision allocation.

---

## 3) Add post-quantization fine-tuning
This is the most important missing component in the current implementation.

The new pipeline must **not** stop at “quantize then evaluate.”

Instead:
- initialize the mixed-precision model from the FP32 checkpoint
- apply layer-wise quantization during training
- fine-tune for a configurable number of epochs
- evaluate after fine-tuning

### Recommended minimal implementation strategy
- Keep the optimizer/loss structure close to `run_baseline`.
- Preserve trainable parameters as FP32 tensors.
- Re-apply layer-wise weight quantization under `torch.no_grad()` before each forward pass or at the start of each batch.
- Use the existing CIFAR-10 SNN training/evaluation structure where possible.
- Continue resetting SNN state after each batch.

### Important non-goal for this rewrite
Do **not** attempt to fully implement **state-variable quantization** in this rewrite unless it can be done cleanly with minimal complexity. The Lui paper discusses state quantization, but this repository currently does not have the neuron/model infrastructure to support a faithful version without much larger architectural changes.

For this rewrite, focus on:
- **weight-side Hessian-aware mixed precision**
- **Hessian-trace-guided layer ranking**
- **quantization-aware fine-tuning**

---

## 4) CLI changes in `scripts/run_hessian_sensitivity.py`
Update the CLI so it matches the rewritten pipeline.

### Keep these arguments
- `--checkpoint-path`
- `--data-root`
- `--output-dir`
- `--bits`
- `--t-steps`
- `--batch-size-train`
- `--batch-size-test`
- `--num-workers`
- `--seed`
- `--device`
- `--deterministic`
- `--download`
- `--max-hessian-batches`
- `--max-test-batches`

### Add these arguments
- `--trace-probes`
- `--quant-epochs`
- `--quant-lr`
- `--quant-weight-decay`
- `--allocation-policy`

Optional if useful:
- `--save-epoch-metrics`

### Remove or de-emphasize
- `--target-avg-bits`

Unless you keep it only for backward compatibility, it should not remain the main control parameter.

### CLI help text
Update help/description strings so they clearly state that the script now performs:
- Hessian trace estimation
- layer-wise mixed-precision assignment
- post-quantization fine-tuning

---

## 5) Output files to preserve or add
Keep the output directory structure under:
- `outputs/hessian_sensitivity/`

Recommended output artifacts:
- `layer_sensitivity.csv`
- `bit_allocation.csv`
- `comparison.csv`
- `summary.json`
- `quant_finetune_epoch_metrics.csv`
- `sensitivity_ranking.png`

### `summary.json` should include
- method name
- allocation policy
- `trace_probes`
- `max_hessian_batches`
- `quant_epochs`
- quant optimizer hyperparameters
- assigned bits per layer
- output file paths
- best/final mixed-precision accuracy

---

## 6) Important constraints
1. **Trace estimation is expensive**
   - Hutchinson trace will be significantly slower than the current proxy.
   - Make sure probe count and batch count are configurable.

2. **`evaluate()` is evaluation-only**
   - It is decorated with `@torch.no_grad()`.
   - Do not try to reuse it for Hessian trace estimation or quantization-aware training.

3. **This repo is not reproducing the exact original Lui experiment stack**
   - Current repo: CIFAR-10 + SpikingJelly + repeated-frame direct encoding
   - Lui paper: N-MNIST + DECOLLE + simplified neuron model
   - The goal here is to adopt the **Hessian method logic**, not to fully reproduce the paper’s full benchmark setup.

4. **Human docs are mandatory**
   - Every modified file must have its mirrored documentation updated in `human_docs/`.

---

## 7) Validation requirements
At minimum, validate the rewrite with the following:

### CLI sanity check
```bash
python scripts/run_hessian_sensitivity.py --help
```

### Smoke test
Run a small test with limited settings, e.g.:
- small `--max-hessian-batches`
- small `--trace-probes`
- small `--max-test-batches`
- `--quant-epochs 1`

### Check expected outputs exist
Confirm generation of at least:
- `layer_sensitivity.csv`
- `bit_allocation.csv`
- `comparison.csv`
- `summary.json`
- `quant_finetune_epoch_metrics.csv`

### Check ranking/allocation sanity
Verify that:
- higher-trace layers receive higher precision
- the new path no longer behaves like the old pathological allocator
- there is no hidden “budget under-spent but algorithm stopped” behavior from the old logic

### Check that mixed results now come from fine-tuning
Ensure the new mixed-precision result is produced by:
- quantization-aware fine-tuning

and not by:
- immediate post-quantization evaluation

---

## 8) Final deliverables
Please provide:
1. updated code
2. updated CLI
3. updated `human_docs/`
4. a short implementation summary explaining:
   - which old logic was removed/replaced
   - in what ways the new pipeline is closer to Lui & Neftci 2021
   - which parts are still not a full reproduction (e.g. state quantization / DECOLLE / N-MNIST)

---

## One-sentence summary
The core task is **not** to “fix the old allocator.” The core task is to replace the current Hessian analysis with a **Lui-style workflow: Hessian trace estimation + conservative ranking-based bit assignment + post-quantization fine-tuning**.
