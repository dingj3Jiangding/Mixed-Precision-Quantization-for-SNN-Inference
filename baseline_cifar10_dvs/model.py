from __future__ import annotations

import torch
import torch.nn as nn
from spikingjelly.activation_based import functional, layer, neuron, surrogate


class CIFAR10DVSSNN(nn.Module):
    """A compact VGG-like multi-step SNN for framed CIFAR10-DVS inputs."""

    def __init__(self, num_classes: int = 10, dropout: float = 0.2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            layer.Conv2d(2, 64, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(64),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan()),
            layer.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(64),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan()),
            layer.AvgPool2d(2),
            layer.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(128),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan()),
            layer.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(128),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan()),
            layer.AvgPool2d(2),
            layer.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(256),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan()),
            layer.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(256),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan()),
            layer.AvgPool2d(2),
            layer.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(512),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan()),
            layer.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(512),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan()),
            layer.AvgPool2d(2),
            layer.AdaptiveAvgPool2d((1, 1)),
        )
        self.dropout = layer.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.classifier = layer.Linear(512, num_classes, bias=True)
        functional.set_step_mode(self, step_mode="m")

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        if x_seq.ndim != 5:
            raise ValueError("Expected input shape [T, B, C, H, W].")
        x_seq = self.features(x_seq)
        x_seq = x_seq.flatten(2)
        x_seq = self.dropout(x_seq)
        return self.classifier(x_seq)


def build_model(num_classes: int = 10) -> CIFAR10DVSSNN:
    return CIFAR10DVSSNN(num_classes=num_classes)
