from __future__ import annotations

import torch
import torch.nn as nn
from spikingjelly.activation_based import functional, layer, neuron, surrogate


class CifarBaselineSNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.conv1 = layer.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = layer.BatchNorm2d(64)
        self.sn1 = neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan())
        self.pool1 = layer.AvgPool2d(2)

        self.conv2 = layer.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn2 = layer.BatchNorm2d(128)
        self.sn2 = neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan())
        self.pool2 = layer.AvgPool2d(2)

        self.fc1 = layer.Linear(128 * 8 * 8, 256, bias=False)
        self.sn3 = neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan())
        self.fc2 = layer.Linear(256, num_classes, bias=True)

        functional.set_step_mode(self, step_mode="m")

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        if x_seq.ndim != 5:
            raise ValueError("Expected input shape [T, B, C, H, W].")

        x_seq = self.pool1(self.sn1(self.bn1(self.conv1(x_seq))))
        x_seq = self.pool2(self.sn2(self.bn2(self.conv2(x_seq))))
        x_seq = x_seq.flatten(2)
        x_seq = self.sn3(self.fc1(x_seq))
        logits_seq = self.fc2(x_seq)
        return logits_seq


def build_model(num_classes: int = 10) -> CifarBaselineSNN:
    return CifarBaselineSNN(num_classes=num_classes)
