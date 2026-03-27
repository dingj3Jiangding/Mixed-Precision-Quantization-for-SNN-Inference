# 文件：`baseline/quantization.py`

## 作用
- 提供统一权重量化（symmetric per-tensor）的核心函数。
- 支持把模型克隆后量化，避免覆盖原 FP32 权重。
- 提供量化后模型大小代理估计（MB）。

## 如何使用
- 克隆并量化：
  - `quant_model, quant_count = clone_and_quantize_model_weights(model, bits=4)`
- 估计模型大小：
  - `size_mb = estimate_quantized_model_size_mb(quant_model, bits=4, quantized_weight_count=quant_count)`
- 解析比特位列表：
  - `bits = parse_bits_list([8, 4, 2])`

## 量化范围
- 默认量化 `Conv/Linear` 的 `weight` 参数。
- 非浮点参数或不含 `weight` 的模块不会被量化。
