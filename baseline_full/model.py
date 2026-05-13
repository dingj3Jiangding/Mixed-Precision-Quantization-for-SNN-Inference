from __future__ import annotations

import torch
import torch.nn as nn
from spikingjelly.activation_based import functional, layer, neuron, surrogate


class VGGStyleCifarSNN(nn.Module):
    """A conservative VGG-like multi-step SNN for CIFAR-10."""

    def __init__(
        self,
        num_classes: int = 10,
        channels: tuple[int, int, int] = (64, 128, 256),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        c1, c2, c3 = channels

        self.features = nn.Sequential(
            layer.Conv2d(3, c1, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(c1),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan()),
            layer.Conv2d(c1, c1, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(c1),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan()),
            layer.AvgPool2d(2),
            layer.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(c2),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan()),
            layer.Conv2d(c2, c2, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(c2),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan()),
            layer.AvgPool2d(2),
            layer.Conv2d(c2, c3, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(c3),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan()),
            layer.Conv2d(c3, c3, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(c3),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan()),
            layer.AdaptiveAvgPool2d((1, 1)),
        )
        self.dropout = layer.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.classifier = layer.Linear(c3, num_classes, bias=True)

        functional.set_step_mode(self, step_mode="m")

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        if x_seq.ndim != 5:
            raise ValueError("Expected input shape [T, B, C, H, W].")

        x_seq = self.features(x_seq)
        x_seq = x_seq.flatten(2)
        x_seq = self.dropout(x_seq)
        logits_seq = self.classifier(x_seq)
        return logits_seq


def build_model(num_classes: int = 10) -> VGGStyleCifarSNN:
    return VGGStyleCifarSNN(num_classes=num_classes)
