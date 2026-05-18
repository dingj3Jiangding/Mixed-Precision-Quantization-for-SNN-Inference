# 文件：`baseline_nmnist/__init__.py`

## 作用
- 暴露 `baseline_nmnist` package 的主要入口。
- 让脚本可以通过 `from baseline_nmnist import ...` 访问配置、FP32 runner、uniform runner 和 Hessian runner。

## 导出对象
- `BaselineNMNISTConfig`
- `run_baseline`
- `run_uniform_quant_comparison`
- `run_hessian_sensitivity_analysis`
- `run_distillation_quantization_analysis`
