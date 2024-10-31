"""DCRNN (Li et al., 2018): diffusion-convolutional GRU encoder-decoder.

Each step uses a graph-diffusion convolution as the linear operation inside
the GRU gates. We implement the bidirectional random-walk diffusion.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _diffusion_pow(adj_rw: torch.Tensor, x: torch.Tensor, k: int) -> torch.Tensor:
    """Compute [I @ x, A @ x, A^2 @ x, ..., A^{k-1} @ x] -> (..., N, k*C)."""
    out = [x]
    last = x
    for _ in range(1, k):
        last = adj_rw @ last
        out.append(last)
    return torch.cat(out, dim=-1)


class DCGRUCell(nn.Module):
    def __init__(self, n_nodes: int, in_features: int, hidden: int, k: int = 2):
        super().__init__()
        self.n_nodes = n_nodes
        self.hidden = hidden
        self.k = k
        d = in_features + hidden
        self.wr = nn.Linear(2 * k * d, hidden)
        self.wu = nn.Linear(2 * k * d, hidden)
        self.wc = nn.Linear(2 * k * d, hidden)

    def forward(self, x: torch.Tensor, h: torch.Tensor,
                adj_fwd: torch.Tensor, adj_bwd: torch.Tensor) -> torch.Tensor:
        # x: (B, N, in_features), h: (B, N, hidden)
        xh = torch.cat([x, h], dim=-1)
        fwd = _diffusion_pow(adj_fwd, xh, self.k)
        bwd = _diffusion_pow(adj_bwd, xh, self.k)
        inp = torch.cat([fwd, bwd], dim=-1)
        r = torch.sigmoid(self.wr(inp))
        u = torch.sigmoid(self.wu(inp))
        xrh = torch.cat([x, r * h], dim=-1)
        fwd_c = _diffusion_pow(adj_fwd, xrh, self.k)
        bwd_c = _diffusion_pow(adj_bwd, xrh, self.k)
        inp_c = torch.cat([fwd_c, bwd_c], dim=-1)
        c = torch.tanh(self.wc(inp_c))
        return u * h + (1 - u) * c


class DCRNN(nn.Module):
    def __init__(self, n_nodes: int, in_features: int = 1, hidden: int = 32,
                 k: int = 2, encoder_steps: int = 12, decoder_steps: int = 12):
        super().__init__()
        self.n_nodes = n_nodes
        self.hidden = hidden
        self.encoder = DCGRUCell(n_nodes, in_features, hidden, k)
        self.decoder = DCGRUCell(n_nodes, 1, hidden, k)
        self.head = nn.Linear(hidden, 1)
        self.encoder_steps = encoder_steps
        self.decoder_steps = decoder_steps

    def forward(self, x: torch.Tensor,
                adj_fwd: torch.Tensor, adj_bwd: torch.Tensor) -> torch.Tensor:
        # x: (B, T_enc, N, F)
        B, T, N, _ = x.shape
        h = torch.zeros(B, N, self.hidden, device=x.device, dtype=x.dtype)
        for t in range(T):
            h = self.encoder(x[:, t], h, adj_fwd, adj_bwd)

        # decoder: feed previous prediction (start with zeros)
        prev = torch.zeros(B, N, 1, device=x.device, dtype=x.dtype)
        outs = []
        for _ in range(self.decoder_steps):
            h = self.decoder(prev, h, adj_fwd, adj_bwd)
            y = self.head(h)  # (B, N, 1)
            outs.append(y.squeeze(-1))
            prev = y
        return torch.stack(outs, dim=1)  # (B, T_dec, N)
