"""MTGNN (Wu et al., 2020): TCN + mix-hop graph convolution with learnable adj."""
from __future__ import annotations

import torch
import torch.nn as nn


class GraphConstructor(nn.Module):
    """Learnable adjacency via node embeddings (top-k sparsification)."""

    def __init__(self, n_nodes: int, emb_dim: int = 16, k: int = 8):
        super().__init__()
        self.e1 = nn.Embedding(n_nodes, emb_dim)
        self.e2 = nn.Embedding(n_nodes, emb_dim)
        self.k = k
        self.alpha = 3.0

    def forward(self) -> torch.Tensor:
        idx = torch.arange(self.e1.num_embeddings, device=self.e1.weight.device)
        m1 = torch.tanh(self.alpha * self.e1(idx))
        m2 = torch.tanh(self.alpha * self.e2(idx))
        a = torch.relu(torch.tanh(self.alpha * (m1 @ m2.T - m2 @ m1.T)))
        topk_val, topk_idx = a.topk(self.k, dim=1)
        mask = torch.zeros_like(a).scatter_(1, topk_idx, 1.0)
        return a * mask


class MixHopConv(nn.Module):
    def __init__(self, in_c: int, out_c: int, depth: int = 2):
        super().__init__()
        self.depth = depth
        self.proj = nn.Linear(in_c * (depth + 1), out_c)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (B, C, N, T)
        B, C, N, T = x.shape
        x_perm = x.permute(0, 3, 2, 1).reshape(B * T, N, C)
        norm = adj / (adj.sum(dim=1, keepdim=True) + 1e-6)
        outs = [x_perm]
        cur = x_perm
        for _ in range(self.depth):
            cur = norm @ cur
            outs.append(cur)
        cat = torch.cat(outs, dim=-1)
        y = self.proj(cat)
        return y.reshape(B, T, N, -1).permute(0, 3, 2, 1)


class MTGNNBlock(nn.Module):
    def __init__(self, channels: int, kernel: int, dilation: int, mix_depth: int):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.filter = nn.Conv2d(channels, channels, (1, kernel),
                                dilation=(1, dilation), padding=(0, pad))
        self.gate = nn.Conv2d(channels, channels, (1, kernel),
                              dilation=(1, dilation), padding=(0, pad))
        self.gc = MixHopConv(channels, channels, depth=mix_depth)
        self.norm = nn.BatchNorm2d(channels)
        self.kernel = kernel
        self.dilation = dilation

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        pad = (self.kernel - 1) * self.dilation
        f = self.filter(x)
        g = self.gate(x)
        if pad > 0:
            f = f[..., :-pad]
            g = g[..., :-pad]
        t = torch.tanh(f) * torch.sigmoid(g)
        s = self.gc(t, adj)
        # residual; align time dim
        s = s + x[..., -s.size(-1):]
        return self.norm(s)


class MTGNN(nn.Module):
    def __init__(self, n_nodes: int, in_channels: int = 1,
                 hidden_channels: int = 32, n_blocks: int = 3,
                 kernel: int = 2, mix_depth: int = 2,
                 out_steps: int = 12, learn_graph: bool = True):
        super().__init__()
        self.n_nodes = n_nodes
        self.start = nn.Conv2d(in_channels, hidden_channels, (1, 1))
        self.blocks = nn.ModuleList([
            MTGNNBlock(hidden_channels, kernel, 2 ** i, mix_depth)
            for i in range(n_blocks)
        ])
        self.graph = GraphConstructor(n_nodes) if learn_graph else None
        self.end1 = nn.Conv2d(hidden_channels, hidden_channels, (1, 1))
        self.end2 = nn.Conv2d(hidden_channels, out_steps, (1, 1))

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None = None
                ) -> torch.Tensor:
        x = x.permute(0, 3, 2, 1)
        adj = adj if adj is not None else self.graph()
        h = self.start(x)
        for block in self.blocks:
            h = block(h, adj)
        out = torch.relu(self.end1(h))
        out = self.end2(out).mean(dim=-1)  # (B, out_steps, N)
        return out
