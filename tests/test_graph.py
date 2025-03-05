import torch

from stgnn.graph import (
    chebyshev_polynomials,
    gaussian_kernel_adj,
    normalize_adj_random_walk,
    normalize_adj_symmetric,
)


def test_symmetric_normalisation_is_symmetric_and_bounded():
    a = torch.tensor([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]])
    n = normalize_adj_symmetric(a)
    assert torch.isfinite(n).all()
    assert torch.allclose(n, n.T, atol=1e-6)
    # spectral radius of D^-1/2 (A+I) D^-1/2 is bounded by 1
    eig = torch.linalg.eigvalsh(n)
    assert eig.abs().max().item() <= 1.0 + 1e-5


def test_random_walk_rows_sum_to_one_with_self_loops():
    a = torch.tensor([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]])
    n = normalize_adj_random_walk(a, add_self_loops=True)
    s = n.sum(dim=1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-5)


def test_chebyshev_polynomials_have_correct_count_and_shape():
    a = torch.eye(4)
    polys = chebyshev_polynomials(a, k=3)
    assert len(polys) == 3
    for p in polys:
        assert p.shape == (4, 4)


def test_gaussian_kernel_adj_is_symmetric():
    import numpy as np
    d = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.5], [2.0, 1.5, 0.0]])
    adj = gaussian_kernel_adj(d, threshold=0.0)
    assert torch.allclose(adj, adj.T)
