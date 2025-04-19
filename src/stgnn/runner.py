"""Unified train+eval runner used by the benchmark CLI."""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from stgnn.data import SlidingWindowDataset
from stgnn.graph import (
    chebyshev_polynomials,
    gaussian_kernel_adj,
    normalize_adj_random_walk,
)
from stgnn.models.dcrnn import DCRNN
from stgnn.models.graph_wavenet import GraphWaveNet
from stgnn.models.mtgnn import MTGNN
from stgnn.models.stgcn import STGCN

log = logging.getLogger(__name__)

try:
    import wandb
    _WANDB = True
except ImportError:  # pragma: no cover
    _WANDB = False


@dataclass
class BenchConfig:
    encoder_steps: int = 12
    decoder_steps: int = 12
    hidden: int = 32
    epochs: int = 10
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-5
    seed: int = 20260514
    device: str = "cpu"
    val_fraction: float = 0.2
    test_fraction: float = 0.2
    cheb_k: int = 3
    mlflow_tracking_uri: str = "file:./mlruns"
    mlflow_experiment: str = "stgnn-bench"
    wandb_project: str = "stgnn-bench"


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _split(values: np.ndarray, val_frac: float, test_frac: float):
    n = values.shape[0]
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    n_train = n - n_val - n_test
    return values[:n_train], values[n_train:n_train + n_val], values[n_train + n_val:]


def _build_model(name: str, n_nodes: int, in_features: int,
                 cfg: BenchConfig) -> nn.Module:
    if name == "stgcn":
        return STGCN(n_nodes, in_channels=in_features, hidden_channels=cfg.hidden,
                     out_steps=cfg.decoder_steps, cheb_k=cfg.cheb_k)
    if name == "dcrnn":
        return DCRNN(n_nodes, in_features=in_features, hidden=cfg.hidden,
                     encoder_steps=cfg.encoder_steps, decoder_steps=cfg.decoder_steps)
    if name == "graphwavenet":
        return GraphWaveNet(n_nodes, in_channels=in_features,
                            hidden_channels=cfg.hidden, out_steps=cfg.decoder_steps)
    if name == "mtgnn":
        return MTGNN(n_nodes, in_channels=in_features, hidden_channels=cfg.hidden,
                     out_steps=cfg.decoder_steps)
    raise ValueError(f"unknown model: {name}")


def _model_forward(model: nn.Module, name: str, x: torch.Tensor,
                   adj: torch.Tensor) -> torch.Tensor:
    if name == "stgcn":
        polys = chebyshev_polynomials(adj, k=3)
        return model(x, polys)
    if name == "dcrnn":
        adj_fwd = normalize_adj_random_walk(adj, add_self_loops=True)
        adj_bwd = normalize_adj_random_walk(adj.T, add_self_loops=True)
        return model(x, adj_fwd, adj_bwd)
    if name == "graphwavenet":
        adj_fwd = normalize_adj_random_walk(adj, add_self_loops=True)
        adj_bwd = normalize_adj_random_walk(adj.T, add_self_loops=True)
        return model(x, adj_fwd, adj_bwd)
    if name == "mtgnn":
        return model(x, adj)
    raise ValueError(f"unknown model: {name}")


def _init_wandb(cfg: BenchConfig, name: str):
    if not _WANDB or os.environ.get("WANDB_MODE") == "disabled":
        return None
    return wandb.init(project=cfg.wandb_project, name=name,
                      config=cfg.__dict__, reinit=True)


def run_benchmark(values: np.ndarray, distances: np.ndarray,
                  model_name: str, cfg: BenchConfig) -> dict[str, float]:
    _seed_all(cfg.seed)
    device = torch.device(cfg.device)

    adj = gaussian_kernel_adj(distances).to(device)
    tr, va, te = _split(values, cfg.val_fraction, cfg.test_fraction)
    train_ds = SlidingWindowDataset(tr, cfg.encoder_steps, cfg.decoder_steps)
    val_ds = SlidingWindowDataset(va, cfg.encoder_steps, cfg.decoder_steps)
    test_ds = SlidingWindowDataset(te, cfg.encoder_steps, cfg.decoder_steps)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    model = _build_model(model_name, values.shape[1], values.shape[-1], cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment)
    run_name = f"{model_name}-{int(time.time())}"
    wb = _init_wandb(cfg, run_name)

    def step(loader: DataLoader, train_mode: bool) -> tuple[float, float]:
        model.train(train_mode)
        tot_mae = 0.0
        tot_rmse = 0.0
        n = 0
        for enc, target in loader:
            enc = enc.to(device)
            target = target.to(device)
            with torch.set_grad_enabled(train_mode):
                pred = _model_forward(model, model_name, enc, adj)
                mae = (pred - target).abs().mean()
                rmse = ((pred - target) ** 2).mean().sqrt()
                if train_mode:
                    opt.zero_grad()
                    mae.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
            tot_mae += mae.item() * enc.size(0)
            tot_rmse += rmse.item() * enc.size(0)
            n += enc.size(0)
        return tot_mae / max(n, 1), tot_rmse / max(n, 1)

    best_val = float("inf")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({**cfg.__dict__, "model": model_name})
        for ep in range(cfg.epochs):
            tr_mae, tr_rmse = step(train_loader, True)
            v_mae, v_rmse = step(val_loader, False)
            row = {"train_mae": tr_mae, "train_rmse": tr_rmse,
                   "val_mae": v_mae, "val_rmse": v_rmse}
            mlflow.log_metrics(row, step=ep)
            if wb is not None:
                wb.log(row, step=ep)
            log.info("model=%s ep=%03d val_mae=%.4f", model_name, ep, v_mae)
            if v_mae < best_val:
                best_val = v_mae
        t_mae, t_rmse = step(test_loader, False)
        mlflow.log_metrics({"test_mae": t_mae, "test_rmse": t_rmse,
                            "best_val_mae": best_val})
        mlflow.pytorch.log_model(model, artifact_path="model")

    if wb is not None:
        wb.finish()
    return {"model": model_name, "best_val_mae": float(best_val),
            "test_mae": float(t_mae), "test_rmse": float(t_rmse)}


MODEL_NAMES = ("stgcn", "dcrnn", "graphwavenet", "mtgnn")
