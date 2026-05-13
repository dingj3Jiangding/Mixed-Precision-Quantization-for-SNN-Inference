# 文件：`baseline_full/__init__.py`

## 作用
- 暴露 `baseline_full` package 的主要入口。
- 让脚本可以通过 `from baseline_full import ...` 访问配置、FP32 runner、uniform runner 和 Hessian runner。

## 导出对象
- `BaselineFullConfig`
- `run_baseline`
- `run_uniform_quant_comparison`
- `run_hessian_sensitivity_analysis`
