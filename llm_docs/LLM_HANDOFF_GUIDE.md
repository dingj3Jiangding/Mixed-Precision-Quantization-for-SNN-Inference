# LLM Handoff Guide (Repo Contract)

This file is for future coding agents working in this repository.
Goal: keep execution stable, keep structure clear, and keep docs synchronized.

## 1) Scope and priority

- Primary research line: `FP32 baseline -> uniform quantization (W8/W4/W2) -> Hessian-guided mixed precision`.
- Current engineering priority: reproducible experiment pipeline and deliverable tables/figures.
- Do not expand scope to unrelated branches before baseline + uniform comparison is stable.

## 2) Required directory contract

Top-level directories currently used:

- `baseline/`: core training/eval/quant code.
- `scripts/`: runnable CLI entry scripts.
- `outputs/`: generated experiment artifacts (csv/json/figures/checkpoints).
- `human_docs/`: human-facing docs for every created/modified file.
- `llm_docs/`: project context docs from user.
- `llm_doc/`: agent handoff/spec docs (this file lives here).

Keep `llm_doc/` and `human_docs/` intact. Do not remove or repurpose them.

## 3) Baseline code map (must stay consistent)

- `baseline/config.py`: `BaselineConfig`, runtime parameters.
- `baseline/data.py`: CIFAR-10 dataloaders and transforms.
- `baseline/model.py`: baseline SNN model definition.
- `baseline/unquant_runner.py`: FP32 train/eval pipeline, exports checkpoint.
- `baseline/quantization.py`: uniform weight quantization utilities.
- `baseline/uniform_runner.py`: FP32 vs W8/W4/W2 comparison pipeline.
- `baseline/__init__.py`: package exports.

CLI scripts:

- `scripts/run_baseline.py`
- `scripts/run_uniform_quant.py`
- `scripts/plot_baseline_graphs.py`

## 4) Code style and implementation rules

- Language: Python.
- Keep changes minimal and local; avoid unrelated refactors.
- Prefer explicit function boundaries over inline script logic.
- Preserve reproducibility controls:
  - fixed seeds
  - deterministic settings where possible
  - fixed dataset/model/T when comparing experiments
- Keep metric names stable across csv/json outputs.
- Do not silently change output schema once downstream plotting depends on it.
- Maintain compatibility with user environment (currently Python 3.9 in conda env).
  - Avoid 3.10+ only syntax (`int | None`, dataclass `slots=True`) unless environment is upgraded.

## 5) Human docs requirement (mandatory)

User rule: for **every file you create or modify**, add/update a doc under `human_docs/`.

Practical convention used in this repo:

- Mirror source path in `human_docs/`.
- For code files: `<path>.md` (example: `human_docs/scripts/run_uniform_quant.py.md`).
- Each doc must include at least:
  - what the file does
  - how to run/use it
  - key inputs/outputs

When outputs are newly generated and become part of workflow, add docs too (as done for baseline csv/json).

## 6) Runbook (expected commands)

Run FP32 baseline (produces checkpoint):

```bash
python scripts/run_baseline.py --epochs 10 --device cuda
```

Expected artifact:

- `outputs/baseline/fp32_last.pt`

Run uniform quant comparison:

```bash
python scripts/run_uniform_quant.py --checkpoint-path outputs/baseline/fp32_last.pt --bits 8,4,2 --device cuda
```

Expected artifact:

- `outputs/uniform_quant/uniform_comparison.csv`

Generate baseline training graphs:

```bash
python scripts/plot_baseline_graphs.py
```

## 7) If execution fails (fallback protocol)

1. Check interpreter/env first:
   - verify `python -V`
   - prefer user conda env in `spikingjelly_path.txt` context
2. Check core dependencies:
   - `torch`, `torchvision`, `spikingjelly`, `pandas`, `matplotlib`
3. Validate imports without full training:
   - run `--help` for each script
4. Use smoke runs before long runs:
   - `--max-train-batches`, `--max-test-batches`, small batch sizes
5. Confirm GPU usage explicitly:
   - `torch.cuda.is_available()`
   - pass `--device cuda` if available
6. If interrupted/aborted, re-check artifact existence before next stage.

## 8) Output consistency requirements

For experiment tables, keep these core columns available for comparison:

- `test_acc`
- `spike_rate`
- `model_size_mb` or `model_size_mb_proxy`
- `sop_proxy`
- `avg_batch_infer_ms`

Do not rename these casually; plotting/reporting depends on them.

## 9) Non-goals right now

- Do not start RL/ILP/full QAT/hardware mapping before first strong comparison package is ready.
- Do not add many datasets/models at once; keep 1 main benchmark path stable first.
