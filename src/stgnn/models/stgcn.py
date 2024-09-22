"""STGCN (Yu et al., 2018): two ST-Conv blocks + linear output.

Each ST-Conv block is: temporal gated conv -> Chebyshev spatial conv ->
temporal gated conv. Temporal convs use causal padding so the time-axis
length is preserved; normalisation uses GroupNorm so the model is robust
to batch size = 1.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TemporalGatedConv(nn.Module):
    """Causal gated temporal conv that preserves the time-axis length."""

    def __init__(self, in_channels: int, out_channels: int, kernel: int = 3):
        super().__init__()
        self.kernel = kernel
        self.conv = nn.Conv2d(in_channels, 2 * out_channels, (1, kernel),
                              padding=(0, kernel - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, N, T) — right-pad and trim to keep T constant
        out = self.conv(x)
        if self.kernel > 1:
            out = out[..., : -(self.kernel - 1)]
        a, b = out.chunk(2, dim=1)
        return a * torch.sigmoid(b)


class ChebyshevConv(nn.Module):
    """Truncated Chebyshev spectral conv (Defferrard 2016).

    Falls back to identity propagation when the supplied polynomials contain
    NaN/Inf values (e.g. on degenerate adjacency matrices)."""

    def __init__(self, in_channels: int, out_channels: int, k: int = 3):
        super().__init__()
        self.k = k
        self.weight = nn.Parameter(torch.randn(k, in_channels, out_channels) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor, polys: list[torch.Tensor]) -> torch.Tensor:
        B, C, N, T = x.shape
        x = x.permute(0, 3, 2, 1).reshape(B * T, N, C)
        # Drop polynomials that contain non-finite values; T_0 = I is always
        # safe so the conv degenerates to a per-node MLP in the worst case.
        out = torch.matmul(x, self.weight[0])  # T_0 = I always
        for k, P in enumerate(polys[1: self.k], start=1):
            if not torch.isfinite(P).all():
                continue
            out = out + P @ x @ self.weight[k]
        out = out + self.bias
        out = out.reshape(B, T, N, -1).permute(0, 3, 2, 1)
        return out


def _group_norm(channels: int) -> nn.GroupNorm:
    """Pick a group count that divides ``channels`` and stays close to 8."""
    for g in (8, 4, 2, 1):
        if channels % g == 0:
            return nn.GroupNorm(g, channels)
    raise ValueError(f"cannot pick a group count for {channels} channels")


class STConvBlock(nn.Module):
    def __init__(self, in_channels: int, spatial_channels: int,
                 out_channels: int, k: int = 3):
        super().__init__()
        self.t1 = TemporalGatedConv(in_channels, spatial_channels)
        self.s = ChebyshevConv(spatial_channels, spatial_channels, k=k)
        self.t2 = TemporalGatedConv(spatial_channels, out_channels)
        self.norm = _group_norm(out_channels)

    def forward(self, x: torch.Tensor, polys: list[torch.Tensor]) -> torch.Tensor:
        h = self.t1(x)
        h = torch.relu(self.s(h, polys))
        h = self.t2(h)
        return self.norm(h)


class STGCN(nn.Module):
    def __init__(self, n_nodes: int, in_channels: int = 1,
                 hidden_channels: int = 32, out_steps: int = 12,
                 cheb_k: int = 3):
        super().__init__()
        self.n_nodes = n_nodes
        self.out_steps = out_steps
        self.block1 = STConvBlock(in_channels, hidden_channels, hidden_channels, cheb_k)
        self.block2 = STConvBlock(hidden_channels, hidden_channels, hidden_channels, cheb_k)
        self.head = nn.Conv2d(hidden_channels, out_steps, (1, 1))

    def forward(self, x: torch.Tensor, polys: list[torch.Tensor]) -> torch.Tensor:
        # x: (B, T, N, F) -> (B, F, N, T)
        x = x.permute(0, 3, 2, 1)
        h = self.block1(x, polys)
        h = self.block2(h, polys)
        h_mean = h.mean(dim=-1, keepdim=True)  # (B, C, N, 1)
        out = self.head(h_mean).squeeze(-1)    # (B, out_steps, N)
        return out
