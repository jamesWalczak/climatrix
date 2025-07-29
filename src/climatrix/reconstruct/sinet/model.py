import numpy as np
import torch
import torch.nn as nn


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
        *,
        layers: int = 2,
        hidden_dim: int = 64,
        sorting_group_size: int = 16,
        scale: float = 1.5,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("Hidden dimension must be a positive integer.")
        if layers <= 0:
            raise ValueError("Number of layers must be a positive integer.")
        if sorting_group_size <= 0:
            raise ValueError("Sorting group size must be a positive integer.")
        if hidden_dim % sorting_group_size != 0:
            raise ValueError(
                f"Hidden dimension {hidden_dim} must be divisible by sorting group size {sorting_group_size}."
            )
        self.fourier_features = FourierFeatures(
            input_dim=in_features,
            mapping_size=self.FOURIER_FEATURES,
            scale=scale,
            trainable=False,
        )
        mlps = []
        in_dim = self.FOURIER_FEATURES * 2
        for i in range(layers):
            mlps.append(nn.Linear(in_dim, hidden_dim, bias=bias))
            mlps.append(GroupSortActivation(sorting_group_size))
            in_dim = hidden_dim
        mlps.append(nn.Linear(in_dim, out_features, bias=bias))
        self.net = nn.Sequential(*mlps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fourier_features = self.fourier_features(x)
        scores = self.net(fourier_features)
        return scores
