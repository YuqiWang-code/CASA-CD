# Source: Hypformer (KDD 2024), MIT License; see ../LICENSE.
import math
import torch
import torch.nn as nn
import torch.nn.init as init


class HypLayerNorm(nn.Module):
    """
    Hyperbolic Layer Normalization Layer

    Parameters:
        manifold (Manifold): The manifold to use for normalization.
        in_features (int): The number of input features.
        manifold_out (Manifold, optional): The output manifold. Default is None.
    """

    def __init__(self, manifold, in_features, manifold_out=None):
        super(HypLayerNorm, self).__init__()
        self.in_features = in_features
        self.manifold = manifold
        self.manifold_out = manifold_out
        self.layer = nn.LayerNorm(self.in_features)
        self.reset_parameters()

    def reset_parameters(self):
        """Reset layer parameters."""
        self.layer.reset_parameters()

    def forward(self, x):
        """Forward pass for hyperbolic layer normalization."""
        x_space = x[..., 1:]
        x_space = self.layer(x_space)
        x_time = ((x_space ** 2).sum(dim=-1, keepdims=True) + self.manifold.k).sqrt()
        x = torch.cat([x_time, x_space], dim=-1)

        if self.manifold_out is not None:
            x = x * (self.manifold_out.k / self.manifold.k).sqrt()
        return x


class HypLinear(nn.Module):
    """
    Hyperbolic Linear Layer

    Parameters:
        manifold (Manifold): The manifold to use for the linear transformation.
        in_features (int): The size of each input sample.
        out_features (int): The size of each output sample.
        bias (bool, optional): If set to False, the layer will not learn an additive bias. Default is True.
        dropout (float, optional): The dropout probability. Default is 0.0.
        manifold_out (Manifold, optional): The output manifold. Default is None.
    """

    def __init__(self, manifold, in_features, out_features, bias=True, dropout=0.0, manifold_out=None):
        super().__init__()
        self.in_features = in_features + 1  # +1 for time dimension
        self.out_features = out_features
        self.bias = bias
        self.manifold = manifold
        self.manifold_out = manifold_out

        self.linear = nn.Linear(self.in_features, self.out_features, bias=bias)
        self.dropout_rate = dropout
        self.reset_parameters()

    def reset_parameters(self):
        """Reset layer parameters."""
        init.xavier_uniform_(self.linear.weight, gain=math.sqrt(2))
        if self.bias:
            init.constant_(self.linear.bias, 0)

    def forward(self, x, x_manifold='hyp'):
        """Forward pass for hyperbolic linear layer."""
        if x_manifold != 'hyp':
            x = torch.cat([torch.ones_like(x)[..., 0:1], x], dim=-1)
            x = self.manifold.expmap0(x)
        x_space = self.linear(x)

        x_time = ((x_space ** 2).sum(dim=-1, keepdims=True) + self.manifold.k).sqrt()
        x = torch.cat([x_time, x_space], dim=-1)
        if self.manifold_out is not None:
            x = x * (self.manifold_out.k / self.manifold.k).sqrt()
        return x
