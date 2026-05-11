# 文件：`baseline/__init__.py`

## 作用
- 作为 `baseline` 包的统一入口。
- 对外暴露四个常用对象：`BaselineConfig`、`run_baseline`、`run_uniform_quant_comparison`、`run_hessian_sensitivity_analysis`。

## 如何使用
- 在其他脚本中直接导入：
  - `from baseline import BaselineConfig, run_baseline, run_uniform_quant_comparison, run_hessian_sensitivity_analysis`
- 推荐配合 `scripts/run_baseline.py` 使用，无需手动拼装底层模块。
