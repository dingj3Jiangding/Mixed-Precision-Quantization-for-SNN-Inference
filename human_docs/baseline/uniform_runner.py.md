# 文件：`baseline/uniform_runner.py`

## 作用
- 执行 `FP32` 与 `W8/W4/W2` 的统一量化对比流程。
- 读取 baseline 训练得到的 checkpoint，逐个 bit-width 做评估。
- 导出对比表和摘要，供后续画 trade-off 图与汇报。

## 如何使用
- 主函数：
  - `run_uniform_quant_comparison(cfg, checkpoint_path, bits_list, output_dir)`
- 输入要求：
  - `checkpoint_path` 必须存在（例如 `outputs/baseline/fp32_last.pt`）
  - `cfg` 中 `data_root / t_steps / batch_size_test` 与 baseline 保持一致
- 输出文件：
  - `outputs/uniform_quant/uniform_comparison.csv`
  - `outputs/uniform_quant/summary.json`

## 输出指标
- 每个设置（`FP32`、`W8`、`W4`、`W2`）包含：
  - `test_acc`
  - `spike_rate`
  - `avg_batch_infer_ms`
  - `model_size_mb_proxy`
  - `sop_proxy`
