"""GraphWaveNet (Wu et al., 2019): stacked dilated TCN + self-adaptive adjacency."""
from __future__ import annotations

import torch
import torch.nn as nn


class GraphConv(nn.Module):
    def __init__(self, in_c: int, out_c: int, n_powers: int = 2):
        super().__init__()
        self.weight = nn.Linear(in_c * (n_powers + 1), out_c)
        self.n_powers = n_powers

    def forward(self, x: torch.Tensor, supports: list[torch.Tensor]
                ) -> torch.Tensor:
        # x: (B, C, N, T) - apply graph conv along N dimension for each (B, T, C)
        B, C, N, T = x.shape
        x_perm = x.permute(0, 3, 2, 1).reshape(B * T, N, C)
        out = [x_perm]
        for A in supports[: self.n_powers]:
            out.append(A @ x_perm)
        cat = torch.cat(out, dim=-1)
        y = self.weight(cat)  # (B*T, N, out_c)
        return y.reshape(B, T, N, -1).permute(0, 3, 2, 1)


class GatedTCN(nn.Module):
    def __init__(self, channels: int, kernel: int, dilation: int):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.tanh = nn.Conv2d(channels, channels, (1, kernel),
                              dilation=(1, dilation), padding=(0, pad))
        self.sigmoid = nn.Conv2d(channels, channels, (1, kernel),
                                 dilation=(1, dilation), padding=(0, pad))
        self.kernel = kernel
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_total = (self.kernel - 1) * self.dilation
        a = self.tanh(x)[..., :-pad_total] if pad_total > 0 else self.tanh(x)
        b = self.sigmoid(x)[..., :-pad_total] if pad_total > 0 else self.sigmoid(x)
        return torch.tanh(a) * torch.sigmoid(b)


class GraphWaveNet(nn.Module):
    def __init__(self, n_nodes: int, in_channels: int = 1,
                 hidden_channels: int = 32, n_blocks: int = 2,
                 n_layers_per_block: int = 2, kernel: int = 2,
                 out_steps: int = 12, adaptive_dim: int = 8):
        super().__init__()
        self.n_nodes = n_nodes
        self.out_steps = out_steps
        self.start = nn.Conv2d(in_channels, hidden_channels, (1, 1))
        self.adaptive_e1 = nn.Parameter(torch.randn(n_nodes, adaptive_dim) * 0.1)
        self.adaptive_e2 = nn.Parameter(torch.randn(adaptive_dim, n_nodes) * 0.1)

        self.tcns = nn.ModuleList()
        self.gconvs = nn.ModuleList()
        self.residual = nn.ModuleList()
        self.skip = nn.ModuleList()
        for b in range(n_blocks):
            for li in range(n_layers_per_block):
                d = 2 ** li
                self.tcns.append(GatedTCN(hidden_channels, kernel, d))
                self.gconvs.append(GraphConv(hidden_channels, hidden_channels, n_powers=2))
                self.residual.append(nn.Conv2d(hidden_channels, hidden_channels, (1, 1)))
                self.skip.append(nn.Conv2d(hidden_channels, hidden_channels, (1, 1)))

        self.end_conv1 = nn.Conv2d(hidden_channels, hidden_channels, (1, 1))
        self.end_conv2 = nn.Conv2d(hidden_channels, out_steps, (1, 1))

    def _supports(self, fwd: torch.Tensor, bwd: torch.Tensor) -> list[torch.Tensor]:
        adp = torch.softmax(torch.relu(self.adaptive_e1 @ self.adaptive_e2), dim=1)
        return [fwd, bwd, adp]

    def forward(self, x: torch.Tensor, fwd: torch.Tensor, bwd: torch.Tensor
                ) -> torch.Tensor:
        # x: (B, T, N, F) -> (B, F, N, T)
        x = x.permute(0, 3, 2, 1)
        h = self.start(x)
        skip = 0
        supports = self._supports(fwd, bwd)
        for tcn, gc, res, sk in zip(self.tcns, self.gconvs, self.residual,
                                    self.skip, strict=True):
            t_out = tcn(h)
            s = sk(t_out)
            skip = s + (skip if isinstance(skip, torch.Tensor) else 0)
            g_out = gc(t_out, supports)
            h = res(g_out) + h[..., -t_out.size(-1):]
        out = torch.relu(skip)
        out = torch.relu(self.end_conv1(out))
        out = self.end_conv2(out).mean(dim=-1, keepdim=False)
        # out: (B, out_steps, N)
        return out
