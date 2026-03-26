from __future__ import annotations

import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron


class SpikeRateTracker:
    def __init__(self, model: nn.Module) -> None:
        self._spike_sum = 0.0
        self._spike_count = 0.0
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        for module in model.modules():
            if isinstance(module, neuron.BaseNode):
                self._handles.append(module.register_forward_hook(self._hook))

    def _hook(self, _module: nn.Module, _inputs, output) -> None:
        out = output[0] if isinstance(output, (tuple, list)) else output
        if not torch.is_tensor(out):
            return
        spikes = (out > 0).to(torch.float32)
        self._spike_sum += float(spikes.sum().item())
        self._spike_count += float(spikes.numel())

    def reset(self) -> None:
        self._spike_sum = 0.0
        self._spike_count = 0.0

    def rate(self) -> float:
        if self._spike_count == 0:
            return 0.0
        return self._spike_sum / self._spike_count

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def parameter_count(model: nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters()))


def model_size_mb(model: nn.Module) -> float:
    total_bytes = sum(param.numel() * param.element_size() for param in model.parameters())
    return float(total_bytes / 1024 / 1024)


def synapse_count_proxy(model: nn.Module) -> int:
    total = 0
    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            if module.weight is not None:
                total += int(module.weight.numel())
    return total


def sop_proxy(spike_rate: float, synapse_count: int, t_steps: int) -> float:
    return float(spike_rate * synapse_count * t_steps)
