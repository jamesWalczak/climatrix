import numpy as np
import torch
import torch.nn as nn


class SineLayer(nn.Module):
    def __init__(
        self, in_features, out_features, bias=True, is_first=False, omega_0=30
    ):
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
    def __init__(
        self,
        in_features=3,
        out_features=1,
        hidden_features=256,
        hidden_layers=3,
        outermost_linear=True,
        omega_0=30,
        omega_hidden=30,
    ):
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

    def forward(self, x):
        return self.net(x)
