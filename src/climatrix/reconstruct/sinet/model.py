import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierFeatures(nn.Module):
    def __init__(
        self,
        input_dim: int,
        mapping_size: int,
        scale: float,
        trainable: bool = False,
    ):
        super().__init__()
        if trainable:
            self.log_scale = nn.Parameter(torch.log(torch.tensor(scale)))
            B = torch.randn((input_dim, mapping_size))
            self.B = nn.Parameter(B)
        else:
            self.log_scale = torch.log(torch.tensor(scale))
            B = torch.randn((input_dim, mapping_size))
            self.register_buffer("B", B)

    def forward(self, x) -> torch.Tensor:
        x_proj = 2 * np.pi * x @ (self.B * torch.exp(self.log_scale))
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Sine(nn.Module):

    def __init__(self):
        super().__init__()
        self.amp = nn.Parameter(torch.tensor(1.0))
        self.freq = nn.Parameter(torch.tensor(2 * np.pi))
        self.offset = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the sine activation function to the input tensor.

        The sine function is defined as:
            f(x) = amp * sin(freq * x + offset)
        """
        return self.amp * torch.sin(self.freq * x + self.offset)


class GausSine(nn.Module):
    """
    Gaussian Sine activation function.
    This function applies a Gaussian function followed by a sine function.
    """

    def __init__(self, traiable: bool = False):
        super().__init__()
        if traiable:
            self.beta = nn.Parameter(torch.tensor(1.0))
            self.omega = nn.Parameter(torch.tensor(2 * np.pi / 1.5))
        else:
            self.register_buffer("beta", torch.tensor(1.0 / (2 * 1.5**2)))
            self.register_buffer("omega", torch.tensor(2 * np.pi / 1.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.exp(-self.beta * x**2) * torch.sin(self.omega * x)
        return torch.sin(weight * x)


class PositionalEncoding(nn.Module):
    def __init__(self, num_frequencies: int = 6, include_input: bool = True):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        self.freq_bands = 2.0 ** torch.linspace(
            0, num_frequencies - 1, num_frequencies
        )

    def forward(self, x):
        out = [x] if self.include_input else []
        for freq in self.freq_bands.to(x.device):
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))
        return torch.cat(out, dim=-1)


class GroupSortActivation(nn.Module):
    def __init__(self, group_size):
        super().__init__()
        self.group_size = group_size

    def forward(self, x):
        batch_size, dim = x.shape
        if dim % self.group_size != 0:
            raise ValueError(
                f"Input dimension {dim} must be divisible by group size {self.group_size}."
            )
        x_reshaped = x.view(batch_size, -1, self.group_size)
        x_sorted, _ = x_reshaped.sort(dim=-1)
        return x_sorted.view(batch_size, dim)


class SiNET(nn.Module):
    FOURIER_FEATURES: int = 64

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mlp: list[int],
        bias: bool = True,
    ) -> None:
        super().__init__()
        if len(mlp) == 0 or mlp is None:
            mlp = [64, 64]
        self.fourier_features = FourierFeatures(
            input_dim=in_features,
            mapping_size=self.FOURIER_FEATURES,
            scale=1.5,
            trainable=False,
        )
        layers = []
        in_dim = self.FOURIER_FEATURES * 2
        for i, layer_size in enumerate(mlp):
            layers.append(nn.Linear(in_dim, layer_size, bias=bias))
            layers.append(GroupSortActivation(16))
            in_dim = layer_size
        layers.append(nn.Linear(in_dim, out_features, bias=bias))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fourier_features = self.fourier_features(x)
        scores = self.net(fourier_features)
        return scores
