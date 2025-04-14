# Adapted from https://github.com/vsitzmann/siren?tab=MIT-1-ov-file#readme
import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import _calculate_correct_fan


class BatchLinear(nn.Linear):
    """A linear meta-layer that can deal with batched weight
    matrices and biases, as for instance output by a
    hypernetwork."""

    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        bias = params.get("bias", None)
        weight = params["weight"]

        output = input.matmul(
            weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2)
        )
        output += bias.unsqueeze(-2)
        return output


class Sine(nn.Module):

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement
        # Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)


class FCBlock(nn.Module):
    """A fully connected neural network that also allows swapping out
      the weights when used with a hypernetwork.
    Can be used just as a normal neural network though, as well.
    """

    def __init__(
        self,
        in_features,
        out_features,
        num_hidden_layers,
        hidden_features,
        outermost_linear=False,
        nonlinearity="relu",
        weight_init=None,
    ):
        super().__init__()

        self.first_layer_init = None

        # Dictionary that maps nonlinearity name to the respective
        # function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {
            "sine": (Sine(), sine_init, first_layer_sine_init),
            "relu": (nn.ReLU(inplace=True), init_weights_normal, None),
            "sigmoid": (nn.Sigmoid(), init_weights_xavier, None),
            "tanh": (nn.Tanh(), init_weights_xavier, None),
            "selu": (nn.SELU(inplace=True), init_weights_selu, None),
            "softplus": (nn.Softplus(), init_weights_normal, None),
            "elu": (nn.ELU(inplace=True), init_weights_elu, None),
        }

        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        self.net = []
        self.net.append(
            nn.Sequential(BatchLinear(in_features, hidden_features), nl)
        )

        for i in range(num_hidden_layers):
            self.net.append(
                nn.Sequential(
                    BatchLinear(hidden_features, hidden_features), nl
                )
            )

        if outermost_linear:
            self.net.append(
                nn.Sequential(BatchLinear(hidden_features, out_features))
            )
        else:
            self.net.append(
                nn.Sequential(BatchLinear(hidden_features, out_features), nl)
            )

        self.net = nn.Sequential(*self.net)
        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        if first_layer_init is not None:
            self.net[0].apply(first_layer_init)

    def forward(self, coords, params=None, **kwargs):
        if params is None:
            params = OrderedDict(self.named_parameters())

        output = self.net(coords)
        return output


class SingleBVPNet(nn.Module):
    """A canonical representation network for a BVP."""

    def __init__(
        self,
        out_features=1,
        type="sine",
        in_features=2,
        mode="mlp",
        hidden_features=256,
        num_hidden_layers=3,
        **kwargs,
    ):
        super().__init__()
        self.mode = mode

        self.image_downsampling = ImageDownsampling(
            sidelength=kwargs.get("sidelength", None),
            downsample=kwargs.get("downsample", False),
        )
        self.net = FCBlock(
            in_features=in_features,
            out_features=out_features,
            num_hidden_layers=num_hidden_layers,
            hidden_features=hidden_features,
            outermost_linear=True,
            nonlinearity=type,
        )

    def forward(self, model_input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        # Enables us to compute gradients w.r.t. coordinates
        coords_org = model_input
        # coords_org = model_input.clone().detach().requires_grad_(True)
        coords = coords_org

        # various input processing methods for different applications
        if self.image_downsampling.downsample:
            coords = self.image_downsampling(coords)
        if self.mode == "rbf":
            coords = self.rbf_layer(coords)
        elif self.mode == "nerf":
            coords = self.positional_encoding(coords)

        output = self.net(coords)
        return output


class ImageDownsampling(nn.Module):
    """Generate samples in upper,v plane according to downsampling
    blur kernel"""

    def __init__(self, sidelength, downsample=False):
        super().__init__()
        if isinstance(sidelength, int):
            self.sidelength = (sidelength, sidelength)
        else:
            self.sidelength = sidelength

        if self.sidelength is not None:
            self.sidelength = torch.Tensor(self.sidelength).cuda().float()
        else:
            assert downsample is False
        self.downsample = downsample

    def forward(self, coords):
        if self.downsample:
            return coords + self.forward_bilinear(coords)
        else:
            return coords

    def forward_box(self, coords):
        return 2 * (torch.rand_like(coords) - 0.5) / self.sidelength

    def forward_bilinear(self, coords):
        Y = torch.sqrt(torch.rand_like(coords)) - 1
        Z = 1 - torch.sqrt(torch.rand_like(coords))
        b = torch.rand_like(coords) < 0.5

        Q = (b * Y + ~b * Z) / self.sidelength
        return Q


# Encoder modules
class SetEncoder(nn.Module):

    def __init__(
        self,
        in_features,
        out_features,
        num_hidden_layers,
        hidden_features,
        nonlinearity="relu",
    ):
        super().__init__()

        assert nonlinearity in ["relu", "sine"], "Unknown nonlinearity type"

        if nonlinearity == "relu":
            nl = nn.ReLU(inplace=True)
            weight_init = init_weights_normal
        elif nonlinearity == "sine":
            nl = Sine()
            weight_init = sine_init

        self.net = [nn.Linear(in_features, hidden_features), nl]
        self.net.extend(
            [
                nn.Sequential(nn.Linear(hidden_features, hidden_features), nl)
                for _ in range(num_hidden_layers)
            ]
        )
        self.net.extend([nn.Linear(hidden_features, out_features), nl])
        self.net = nn.Sequential(*self.net)

        self.net.apply(weight_init)

    def forward(self, context_x, context_y, ctxt_mask=None, **kwargs):
        input = torch.cat((context_x, context_y), dim=-1)
        embeddings = self.net(input)

        if ctxt_mask is not None:
            embeddings = embeddings * ctxt_mask
            embedding = embeddings.mean(dim=-2) * (
                embeddings.shape[-2] / torch.sum(ctxt_mask, dim=-2)
            )
            return embedding
        return embeddings.mean(dim=-2)


########################
# Initialization methods
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # For PINNet, Raissi et al. 2019
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/
    # truncated_normal.pdf
    # grab from upstream pytorch branch and paste here for now
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [lower, upper], then
        #  translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * lower - 1, 2 * upper - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def init_weights_trunc_normal(m):
    # For PINNet, Raissi et al. 2019
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentati
    # ons/truncated_normal.pdf
    if isinstance(m, (BatchLinear, nn.Linear)):
        if hasattr(m, "weight"):
            fan_in = m.weight.size(1)
            fan_out = m.weight.size(0)
            std = math.sqrt(2.0 / float(fan_in + fan_out))
            mean = 0.0
            # initialize with the same behavior as tf.truncated_normal
            # "The generated values follow a normal distribution with
            # specified mean and
            # standard deviation, except that values whose magnitude is
            # more than 2
            # standard deviations from the mean are dropped and re-picked."
            _no_grad_trunc_normal_(m.weight, mean, std, -2 * std, 2 * std)


def init_weights_normal(m):
    if isinstance(m, (BatchLinear, nn.Linear)):
        if hasattr(m, "weight"):
            nn.init.kaiming_normal_(
                m.weight, a=0.0, nonlinearity="relu", mode="fan_in"
            )


def init_weights_selu(m):
    if isinstance(m, (BatchLinear, nn.Linear)):
        if hasattr(m, "weight"):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))


def init_weights_elu(m):
    if isinstance(m, (BatchLinear, nn.Linear)):
        if hasattr(m, "weight"):
            num_input = m.weight.size(-1)
            nn.init.normal_(
                m.weight,
                std=math.sqrt(1.5505188080679277) / math.sqrt(num_input),
            )


def init_weights_xavier(m):
    if isinstance(m, (BatchLinear, nn.Linear)):
        if hasattr(m, "weight"):
            nn.init.xavier_normal_(m.weight)


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


###################
# Complex operators
def compl_conj(x):
    y = x.clone()
    y[..., 1::2] = -1 * y[..., 1::2]
    return y


def compl_div(x, y):
    """x / y"""
    a = x[..., ::2]
    b = x[..., 1::2]
    c = y[..., ::2]
    d = y[..., 1::2]

    outr = (a * c + b * d) / (c**2 + d**2)
    outi = (b * c - a * d) / (c**2 + d**2)
    out = torch.zeros_like(x)
    out[..., ::2] = outr
    out[..., 1::2] = outi
    return out


def compl_mul(x, y):
    """x * y"""
    a = x[..., ::2]
    b = x[..., 1::2]
    c = y[..., ::2]
    d = y[..., 1::2]

    outr = a * c - b * d
    outi = (a + b) * (c + d) - a * c - b * d
    out = torch.zeros_like(x)
    out[..., ::2] = outr
    out[..., 1::2] = outi
    return out


def siren_uniform_(tensor: torch.Tensor, mode: str = "fan_in", c: float = 6):
    fan = _calculate_correct_fan(tensor, mode)
    std = 1.0 / math.sqrt(fan)
    bound = math.sqrt(c) * std
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


class SineActivation(nn.Module):
    scale: float

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.scale * x)


class SIREN(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        mlp: list[int],
        scale: float = 1.0,
        scale_first_layer: float = 30.0,
        bias: bool = True,
        c: float = 6,
    ):
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

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                siren_uniform_(m.weight, mode="fan_in", c=c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
