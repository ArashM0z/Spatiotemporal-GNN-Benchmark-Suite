"""Graph utilities: adjacency normalisation, Chebyshev polynomials, masking."""
from __future__ import annotations

import numpy as np
import torch


def normalize_adj_symmetric(adj: torch.Tensor, add_self_loops: bool = True
                            ) -> torch.Tensor:
    """Symmetric normalisation: A_hat = D^-1/2 (A + I) D^-1/2."""
    a = adj.clone().float()
    if add_self_loops:
        a = a + torch.eye(a.size(0), device=a.device)
    deg = a.sum(dim=1)
    d_inv_sqrt = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
    d_mat = torch.diag(d_inv_sqrt)
    return d_mat @ a @ d_mat


def normalize_adj_random_walk(adj: torch.Tensor, add_self_loops: bool = False
                              ) -> torch.Tensor:
    """Random-walk normalisation: D^-1 A."""
    a = adj.clone().float()
    if add_self_loops:
        a = a + torch.eye(a.size(0), device=a.device)
    deg = a.sum(dim=1)
    d_inv = torch.where(deg > 0, deg.pow(-1.0), torch.zeros_like(deg))
    return torch.diag(d_inv) @ a


def chebyshev_polynomials(adj: torch.Tensor, k: int) -> list[torch.Tensor]:
    """T_0..T_{k-1} of the scaled Laplacian (used by STGCN, DCRNN)."""
    n = adj.size(0)
    eye = torch.eye(n, device=adj.device, dtype=adj.dtype)
    a_norm = normalize_adj_symmetric(adj, add_self_loops=False)
    L = eye - a_norm
    # symmetrise to guard against accumulated FP error; eigvalsh requires it
    L = 0.5 * (L + L.T)
    try:
        ev = torch.linalg.eigvalsh(L)
        l_max = float(ev.max().item())
    except Exception:
        l_max = 2.0
    if not (l_max > 1e-6 and l_max != float("inf")):
        l_max = 2.0
    L_tilde = 2 * L / l_max - eye
    polys = [eye, L_tilde]
    for _ in range(2, k):
        polys.append(2 * L_tilde @ polys[-1] - polys[-2])
    return polys[:k]


def gaussian_kernel_adj(distances: np.ndarray, sigma: float | None = None,
                        threshold: float = 0.1) -> torch.Tensor:
    n = distances.shape[0]
    iu = np.triu_indices(n, k=1)
    sigma = sigma or float(distances[iu].std() + 1e-8)
    w = np.exp(-(distances ** 2) / (sigma ** 2))
    w[w < threshold] = 0.0
    return torch.from_numpy(w).float()
