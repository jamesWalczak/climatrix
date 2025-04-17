# Adapted from https://github.com/vsitzmann/siren?tab=MIT-1-ov-file#readme
import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import _calculate_correct_fan


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, "weight"):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(
                -np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30
            )


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, "weight"):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement
            # Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)


class SineActivation(nn.Module):
    scale: float

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.scale * x)


class SiNET(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        mlp: list[int],
        scale: float = 1.0,
        scale_first_layer: float = 30.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if len(mlp) == 0:
            mlp = [64, 64]
        layers = [
            nn.Linear(in_features, mlp[0], bias=bias),
            SineActivation(scale=scale_first_layer),
        ]
        for i in range(len(mlp) - 1):
            layers.append(nn.Linear(mlp[i], mlp[i + 1], bias=bias))
            layers.append(SineActivation(scale=scale))

        layers.append(nn.Linear(mlp[-1], out_features, bias=bias))

        self.net = nn.Sequential(*layers)

        self.net.apply(sine_init)
        self.net[0].apply(first_layer_sine_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
