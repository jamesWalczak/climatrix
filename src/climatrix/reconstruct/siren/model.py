import numpy as np
import torch
import torch.nn as nn


class SineLayer(nn.Module):
    """
    Sine activation layer with custom frequency and initialization.
    Based on https://github.com/vsitzmann/siren
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        is_first: bool = False,
        omega_0: float = 30,
    ) -> None:
        """
        Initialize sine activation layer with frequency omega_0.

        Args:
            in_features: Number of input features
            out_features: Number of output features
            bias: Whether to use bias in the linear layer
            is_first: Whether this is the first layer
            omega_0: Frequency of the sine activation
        """
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(
                    -1 / self.in_features, 1 / self.in_features
                )
            else:
                limit = np.sqrt(6 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-limit, limit)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(nn.Module):
    """
    SIREN (Sinusoidal Representation Networks) implementation.
    """

    def __init__(
        self,
        in_features: int = 3,
        out_features: int = 1,
        hidden_features: int = 256,
        hidden_layers: int = 3,
        outermost_linear: bool = True,
        omega_0: float = 30,
        omega_hidden: float = 30,
    ) -> None:
        """
        Initialize SIREN network with specified architecture.

        Args:
            in_features: Number of input features
            out_features: Number of output features
            hidden_features: Width of hidden layers
            hidden_layers: Number of hidden layers
            outermost_linear: Whether to use linear activation in final layer
            omega_0: Frequency for first sine layer
            omega_hidden: Frequency for subsequent sine layers
        """
        super().__init__()

        self.net = []
        self.net.append(
            SineLayer(
                in_features, hidden_features, is_first=True, omega_0=omega_0
            )
        )

        for i in range(hidden_layers - 1):
            self.net.append(
                SineLayer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega_0=omega_hidden,
                )
            )

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                limit = np.sqrt(6 / hidden_features) / omega_hidden
                final_linear.weight.uniform_(-limit, limit)
            self.net.append(final_linear)
        else:
            self.net.append(
                SineLayer(
                    hidden_features,
                    out_features,
                    is_first=False,
                    omega_0=omega_hidden,
                )
            )

        self.net = nn.Sequential(*self.net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the SIREN network.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)
        """
        return self.net(x)
