from __future__ import annotations

import copy
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn


def _quantize_tensor_symmetric_per_tensor(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    if bits < 2:
        raise ValueError("bits must be >= 2 for symmetric quantization.")
    if not torch.is_floating_point(tensor):
        return tensor

    max_abs = tensor.detach().abs().max()
    if max_abs == 0:
        return tensor.clone()

    qmax = (2 ** (bits - 1)) - 1
    scale = max_abs / qmax
    quantized = torch.round(tensor / scale).clamp(-qmax, qmax)
    dequantized = quantized * scale
    return dequantized


def is_weight_module(module: nn.Module) -> bool:
    return isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear))


def quantize_module_weights_inplace(module: nn.Module, bits: int) -> int:
    quantized_param_count = 0
    for submodule in module.modules():
        if not is_weight_module(submodule):
            continue
        weight = getattr(submodule, "weight", None)
        if weight is None:
            continue
        with torch.no_grad():
            q_weight = _quantize_tensor_symmetric_per_tensor(weight.data, bits)
            weight.data.copy_(q_weight)
            quantized_param_count += int(weight.numel())
    return quantized_param_count


def clone_and_quantize_model_weights(model: nn.Module, bits: int) -> Tuple[nn.Module, int]:
    quantized_model = copy.deepcopy(model)
    quantized_count = quantize_module_weights_inplace(quantized_model, bits=bits)
    return quantized_model, quantized_count


def estimate_quantized_model_size_mb(
    model: nn.Module, bits: int, quantized_weight_count: Optional[int] = None
) -> float:
    if quantized_weight_count is None:
        quantized_weight_count = sum(
            int(getattr(module, "weight").numel())
            for module in model.modules()
            if is_weight_module(module) and getattr(module, "weight", None) is not None
        )
    total_param_count = sum(int(param.numel()) for param in model.parameters())
    non_quantized_count = max(total_param_count - int(quantized_weight_count), 0)

    weight_bytes = (quantized_weight_count * bits) / 8.0
    non_quantized_bytes = non_quantized_count * 4.0
    return float((weight_bytes + non_quantized_bytes) / 1024 / 1024)


def parse_bits_list(bits_list: Iterable[int]) -> list[int]:
    bits = sorted(set(int(item) for item in bits_list), reverse=True)
    for bit in bits:
        if bit < 2:
            raise ValueError("All bit-width values must be >= 2.")
    return bits
