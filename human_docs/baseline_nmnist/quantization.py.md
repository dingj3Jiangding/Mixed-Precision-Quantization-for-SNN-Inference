# 文件：`baseline_nmnist/quantization.py`

## 作用
- 复用旧 baseline 的 weight-side symmetric per-tensor quantization 工具。
- 保持 uniform quantization 和 Hessian mixed precision 的量化行为一致。

## 如何使用
- `clone_and_quantize_model_weights(model, bits=8)`
- `parse_bits_list([8, 4])`

## 输出
- 返回量化后的模型副本、量化参数数量、模型大小 proxy 等。
