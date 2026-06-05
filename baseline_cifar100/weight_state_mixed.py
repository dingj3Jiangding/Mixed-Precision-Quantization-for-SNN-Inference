from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from spikingjelly.activation_based import functional, neuron

from .hessian import _iter_with_limit
from .unquant_runner import direct_encode


def _collect_state_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, neuron.BaseNode):
            layers.append((name, module))
    return layers


@torch.no_grad()
def estimate_state_cost_proxy(
    model: nn.Module,
    loader,
    device: str,
    t_steps: int,
    max_batches: Optional[int],
) -> List[dict]:
    state_layers = _collect_state_layers(model)
    if not state_layers:
        raise RuntimeError("No state layers were found.")

    state_acc: Dict[str, float] = {name: 0.0 for name, _ in state_layers}
    state_count: Dict[str, int] = {name: 0 for name, _ in state_layers}
    handles: List[torch.utils.hooks.RemovableHandle] = []

    def make_hook(layer_name: str):
        def hook(_module: nn.Module, _inputs, output) -> None:
            out = output[0] if isinstance(output, (tuple, list)) else output
            if not torch.is_tensor(out) or out.ndim < 2:
                return
            batch_size = int(out.shape[1]) if out.ndim >= 3 else int(out.shape[0])
            if batch_size <= 0:
                return
            state_acc[layer_name] += float(out.numel()) / float(batch_size)
            state_count[layer_name] += 1

        return hook

    for name, module in state_layers:
        handles.append(module.register_forward_hook(make_hook(name)))

    try:
        model.eval()
        batch_count = 0
        for x, _y in _iter_with_limit(loader, max_batches):
            x = x.to(device)
            x_seq = direct_encode(x, t_steps)
            model(x_seq)
            batch_count += 1
            functional.reset_net(model)
        if batch_count == 0:
            raise RuntimeError("No batches were processed for state-cost estimation.")
    finally:
        for handle in handles:
            handle.remove()

    rows: List[dict] = []
    for name, _module in state_layers:
        proxy = state_acc[name] / float(max(state_count[name], 1))
        rows.append(
            {
                "state_layer_name": name,
                "state_cost_proxy": proxy,
            }
        )
    rows.sort(key=lambda item: item["state_cost_proxy"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["state_rank"] = rank
    return rows
