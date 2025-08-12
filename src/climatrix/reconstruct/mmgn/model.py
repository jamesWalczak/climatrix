"""MMGN reconstruction module.

This module is an adaptation of the MMGN method presented elaborated
in [1]. The code is heavily inspired by the original implementation
from the authors, which can be found at:
https://github.com/Xihaier/Continuous-Field-Reconstruction-MMGN


References
----------
[1] LUO, Xihaier, et al. Continuous field reconstruction from sparse
    observations with implicit neural networks.
    arXiv preprint arXiv:2401.11611, 2024.
"""

from collections import namedtuple
from enum import Enum, StrEnum
from typing import Self

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class _FilterType(StrEnum):
    GABOR = "gabor"
    FOURIER = "fourier"
    LINEAR = "linear"

    @classmethod
    def get(cls, name: str | Self) -> Self:
        if isinstance(name, _FilterType):
            return name
        try:
            return cls[name.upper()]
        except KeyError:
            return cls(name.upper())

    @classmethod
    def choices(cls) -> list[str]:
        return [member.value for member in cls]


_InitType = namedtuple("InitType", ["name", "callable"])


class _LatentInitType(Enum):
    UNIFORM = _InitType("uniform", nn.init.uniform_)
    NORMAL = _InitType("normal", nn.init.normal_)
    ZEROS = _InitType("zeros", nn.init.zeros_)
    ONES = _InitType("ones", nn.init.ones_)
    XAVIER_UNIFORM = _InitType("xavier_uniform", nn.init.xavier_uniform_)
    XAVIER_NORMAL = _InitType("xavier_normal", nn.init.xavier_normal_)
    KAIMING_UNIFORM = _InitType("kaiming_uniform", nn.init.kaiming_uniform_)
    KAIMING_NORMAL = _InitType("kaiming_normal", nn.init.kaiming_normal_)
    TRUNCATED_NORMAL = _InitType("truncated_normal", nn.init.trunc_normal_)
    ORTHOGONAL = _InitType("orthogonal", nn.init.orthogonal_)

    @classmethod
    def get(cls, name: str | Self) -> Self:
        if isinstance(name, _LatentInitType):
            return name
        if isinstance(name, str):
            name = name.upper()
        if isinstance(name, _InitType):
            name = name.name.upper()
        try:
            return cls[name]
        except KeyError:
            return cls(name)

    @classmethod
    def choices(cls) -> list[str]:
        return [member.value.name for member in cls]


class FuseLinear(nn.Module):
    b: int
    A: nn.Parameter
    B: nn.Parameter
    bias: nn.Parameter | None

    def __init__(
        self, coord_dim: int, latent_dim: int, out_dim: int, bias: bool = True
    ) -> None:
        super().__init__()
        self.b = coord_dim
        self.A = nn.Parameter(torch.empty(out_dim, coord_dim))
        self.B = nn.Parameter(torch.empty(out_dim, latent_dim))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_dim))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        bounds = 1.0 / np.sqrt(self.b)
        nn.init.kaiming_uniform_(self.A, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.B, a=np.sqrt(5))
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bounds, bounds)

    def forward(self, x: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        linear_coord = torch.einsum("pi,oi->po", x, self.A)
        linear_latent = torch.einsum("j,oj->o", latent, self.B)

        out = linear_coord + linear_latent
        if self.bias is not None:
            out += self.bias

        return out


class FourierLayer(nn.Module):
    weight: nn.Parameter
    weight_scale: float

    def __init__(
        self, input_dim: int, out_dim: int, weight_scale: float
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, input_dim))
        self.weight_scale = weight_scale
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                torch.sin(F.linear(x, self.weight * self.weight_scale)),
                torch.cos(F.linear(x, self.weight * self.weight_scale)),
            ],
            dim=-1,
        )


class GaborLayer(nn.Module):
    linear: nn.Linear
    mu: nn.Parameter
    gamma: nn.Parameter

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        weight_scale: float,
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.mu = nn.Parameter(2 * torch.rand(out_dim, in_dim) - 1)
        self.gamma = nn.Parameter(
            torch.distributions.gamma.Gamma(alpha, beta).sample((out_dim,))
        )
        self.linear.weight.data *= weight_scale * torch.sqrt(
            self.gamma[:, None]
        )
        self.linear.bias.data.uniform_(-np.pi, np.pi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        D = (
            (x**2).sum(dim=-1, keepdim=True)
            + (self.mu**2).sum(dim=-1)[None, :]
            - 2 * x @ self.mu.T
        )
        return torch.sin(self.linear(x)) * torch.exp(
            -0.5 * D * self.gamma[None, :]
        )


class MMGNet(nn.Module):
    input_dim: int
    hidden_dim: int
    latent_dim: int
    out_dim: int
    n_layers: int
    input_scale: int
    alpha: float
    filter_type: _FilterType
    latent_init: _LatentInitType

    bilinear: nn.ModuleList
    filters: nn.ModuleList
    output_layer: nn.Module
    latents: nn.Parameter

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        out_dim: int,
        n_layers: int,
        input_scale: int,
        alpha: float,
        filter_type: str | _FilterType = "gabor",
        latent_init: str | _LatentInitType = "zeros",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.input_scale = input_scale
        self.alpha = alpha
        self.filter_type = _FilterType.get(filter_type)
        self.latent_init = _LatentInitType.get(latent_init)

        self._init_network()

    def _init_network(self):
        self.bilinear = nn.ModuleList(
            [FuseLinear(self.input_dim, self.latent_dim, self.hidden_dim)]
            + [
                FuseLinear(self.hidden_dim, self.latent_dim, self.hidden_dim)
                for _ in range(self.n_layers - 1)
            ]
        )

        match self.filter_type:
            case _FilterType.GABOR:
                self.filters = nn.ModuleList(
                    [
                        GaborLayer(
                            self.input_dim,
                            self.hidden_dim,
                            self.input_scale / np.sqrt(self.n_layers),
                        )
                        for _ in range(self.n_layers)
                    ]
                )
            case _FilterType.FOURIER:
                self.filters = nn.ModuleList(
                    [
                        FourierLayer(
                            self.input_dim,
                            self.hidden_dim // 2,
                            self.input_scale / np.sqrt(self.n_layers),
                        )
                        for _ in range(self.n_layers)
                    ]
                )
            case _FilterType.LINEAR:
                self.filters = nn.ModuleList(
                    [
                        nn.Linear(self.input_dim, self.hidden_dim)
                        for _ in range(self.n_layers)
                    ]
                )
            case _:
                raise ValueError(f"Unknown filter type: {self.filter}")

        self.output_layer = nn.Linear(self.hidden_dim, self.out_dim)
        self.latents = nn.Parameter(torch.FloatTensor(self.latent_dim))

        with torch.no_grad():
            self.latent_init.value.callable(self.latents)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.latents
        zi = self.filters[0](x) * self.bilinear[0](torch.zeros_like(x), latent)
        for i in range(1, self.n_layers):
            zi = self.filters[i](x) * self.bilinear[i](zi, latent)
        output = self.output_layer(zi)
        if self.out_dim == 1:
            return output.squeeze(-1)  # Only squeeze last dimension
        else:
            return output
