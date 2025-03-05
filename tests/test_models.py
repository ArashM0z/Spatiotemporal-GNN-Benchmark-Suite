import torch

from stgnn.data import synthesize
from stgnn.graph import (
    chebyshev_polynomials,
    gaussian_kernel_adj,
    normalize_adj_random_walk,
)
from stgnn.models.dcrnn import DCRNN
from stgnn.models.graph_wavenet import GraphWaveNet
from stgnn.models.mtgnn import MTGNN
from stgnn.models.stgcn import STGCN


def _inputs(n_nodes=8, T=12):
    v, d = synthesize(n_steps=80, n_nodes=n_nodes, seed=0)
    adj = gaussian_kernel_adj(d)
    x = torch.from_numpy(v[:T]).unsqueeze(0)  # (1, T, N, F)
    return x, adj


def test_stgcn_forward_shape():
    x, adj = _inputs(n_nodes=8, T=12)
    polys = chebyshev_polynomials(adj, k=3)
    m = STGCN(n_nodes=8, in_channels=1, hidden_channels=16,
              out_steps=12, cheb_k=3)
    out = m(x, polys)
    assert out.shape[-1] == 8


def test_dcrnn_forward_shape():
    x, adj = _inputs(n_nodes=8, T=12)
    fwd = normalize_adj_random_walk(adj, add_self_loops=True)
    bwd = normalize_adj_random_walk(adj.T, add_self_loops=True)
    m = DCRNN(n_nodes=8, in_features=1, hidden=8, decoder_steps=6)
    out = m(x, fwd, bwd)
    assert out.shape == (1, 6, 8)


def test_graph_wavenet_forward_shape():
    x, adj = _inputs(n_nodes=8, T=12)
    fwd = normalize_adj_random_walk(adj, add_self_loops=True)
    bwd = normalize_adj_random_walk(adj.T, add_self_loops=True)
    m = GraphWaveNet(n_nodes=8, in_channels=1, hidden_channels=16,
                     out_steps=12)
    out = m(x, fwd, bwd)
    assert out.shape == (1, 12, 8)


def test_mtgnn_forward_shape():
    x, adj = _inputs(n_nodes=8, T=12)
    m = MTGNN(n_nodes=8, in_channels=1, hidden_channels=16,
              out_steps=12)
    out = m(x, adj)
    assert out.shape == (1, 12, 8)


def test_gradients_flow_in_stgcn():
    x, adj = _inputs(n_nodes=8, T=12)
    # use batched input so GroupNorm receives finite per-channel statistics
    x = x.repeat(2, 1, 1, 1)
    polys = chebyshev_polynomials(adj, k=3)
    m = STGCN(n_nodes=8, in_channels=1, hidden_channels=16,
              out_steps=12, cheb_k=3)
    out = m(x, polys)
    assert torch.isfinite(out).all()
    out.pow(2).mean().backward()
    n_grad_params = sum(int(p.grad is not None and p.grad.abs().sum() > 0)
                        for p in m.parameters())
    assert n_grad_params > 0
