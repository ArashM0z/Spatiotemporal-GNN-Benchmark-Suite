# Spatio-Temporal GNN Benchmarks

A single unified training+evaluation harness for four reference
spatio-temporal GNN architectures:

| Model | Paper | Key idea |
| --- | --- | --- |
| **STGCN** | [Yu, Yin, Zhu (2018)](https://arxiv.org/abs/1709.04875) | gated temporal conv + Chebyshev spatial conv |
| **DCRNN** | [Li, Yu, Shahabi, Liu (2018)](https://arxiv.org/abs/1707.01926) | diffusion-convolutional GRU (bi-random-walk) |
| **GraphWaveNet** | [Wu et al. (2019)](https://arxiv.org/abs/1906.00121) | stacked dilated TCN + self-adaptive adjacency |
| **MTGNN** | [Wu et al. (2020)](https://arxiv.org/abs/2005.11650) | learned graph constructor + mix-hop conv |

All four are implemented from scratch (no `torch_geometric` dependency) so
the code is small, self-contained, and easy to read.

## What's in the box

- Uniform `(B, T, N, F)` input convention.
- Shared `SlidingWindowDataset` and Gaussian-kernel adjacency builder.
- Random-walk + symmetric normalisation, Chebyshev polynomials.
- MLflow + Weights & Biases dual logging per benchmark run.
- A single CLI: `stgnn-bench --model all` produces a JSON leaderboard.

## Quickstart

```bash
pip install -e ".[dev]"

# 60-second CPU smoke benchmark of all four models on synthetic data
WANDB_MODE=disabled stgnn-bench --config configs/smoke.yaml \
                                --demo --model all

# Full benchmark
stgnn-bench --config configs/default.yaml \
            --values data/metrla_values.npz \
            --distances data/metrla_dist.npz \
            --model all
```

Leaderboard output (`bench_results.json`):

```json
[
  {"model": "stgcn",        "test_mae": 0.42, "test_rmse": 0.61},
  {"model": "dcrnn",        "test_mae": 0.38, "test_rmse": 0.55},
  {"model": "graphwavenet", "test_mae": 0.36, "test_rmse": 0.53},
  {"model": "mtgnn",        "test_mae": 0.35, "test_rmse": 0.52}
]
```

(Numbers depend on dataset, model size, and training budget.)

## Layout

```
src/stgnn/
├── graph.py              # adjacency normalisations + Chebyshev polynomials
├── data.py               # sliding-window dataset + synthesiser
├── models/
│   ├── stgcn.py          # STGCN (Yu 2018)
│   ├── dcrnn.py          # DCRNN (Li 2018)
│   ├── graph_wavenet.py  # GraphWaveNet (Wu 2019)
│   └── mtgnn.py          # MTGNN (Wu 2020)
├── runner.py             # train+eval harness with MLflow + W&B
└── cli.py                # stgnn-bench entrypoint
configs/                  # default + smoke
tests/                    # graph utils, per-model shapes, end-to-end smoke
```
