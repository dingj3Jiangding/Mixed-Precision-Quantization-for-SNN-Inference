# 文件：`baseline_vgg16/__init__.py`

## 作用
- 暴露 `baseline_vgg16` package 的主要入口。
- 让脚本可以通过 `from baseline_vgg16 import ...` 访问配置、FP32 runner、uniform runner 和 Hessian runner。

## 导出对象
- `BaselineVGG16Config`
- `run_baseline`
- `run_uniform_quant_comparison`
- `run_hessian_sensitivity_analysis`
- `run_distillation_quantization_analysis`
