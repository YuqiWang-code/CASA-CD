# Source: Hypformer (KDD 2024), MIT License; see ../LICENSE.
from typing import Optional
import torch
from geoopt import Lorentz as LorentzBase


class Lorentz(LorentzBase):
    def cinner(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the cross-inner product of two points.

        Parameters:
            x (torch.Tensor): First point.
            y (torch.Tensor): Second point.

        Returns:
            torch.Tensor: Cross-inner product.
        """
        x = x.clone()
        x.narrow(-1, 0, 1).mul_(-1)
        return x @ y.transpose(-1, -2)

    def mid_point(self, x: torch.Tensor, w: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the midpoint of points on the manifold.

        Parameters:
            x (torch.Tensor): Points on the manifold.
            w (torch.Tensor, optional): Weights for each point. Default is None.

        Returns:
            torch.Tensor: Midpoint.
        """
        if w is not None:
            ave = w.matmul(x)
        else:
            ave = x.mean(dim=-2)
        denom = (-self.inner(ave, ave, ave, keepdim=True)).abs().clamp_min(1e-8).sqrt()
        return self.k.sqrt() * ave / denom
