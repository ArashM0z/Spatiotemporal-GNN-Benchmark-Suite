"""Sliding-window dataset for traffic speed (T, N, F) tensors."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class SlidingWindowDataset(Dataset):
    def __init__(self, values: np.ndarray, encoder_steps: int,
                 decoder_steps: int):
        if values.ndim != 3:
            raise ValueError("values must be (T, N, F)")
        if values.shape[0] < encoder_steps + decoder_steps:
            raise ValueError("series too short for requested window")
        self.values = values.astype(np.float32)
        self.encoder_steps = encoder_steps
        self.decoder_steps = decoder_steps
        mu = self.values.mean(axis=(0, 1), keepdims=True)
        sigma = self.values.std(axis=(0, 1), keepdims=True) + 1e-6
        self.mean = mu
        self.std = sigma
        self.values_norm = (self.values - mu) / sigma

    def __len__(self) -> int:
        return self.values.shape[0] - self.encoder_steps - self.decoder_steps + 1

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        enc = self.values_norm[i: i + self.encoder_steps]
        dec = self.values_norm[
            i + self.encoder_steps:
            i + self.encoder_steps + self.decoder_steps,
            :, 0,  # speed channel
        ]
        return torch.from_numpy(enc), torch.from_numpy(dec)


def synthesize(n_steps: int = 600, n_nodes: int = 12, n_features: int = 1,
               seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0, 2 * np.pi, n_nodes)
    t = np.arange(n_steps)
    speed = 50 + 20 * np.sin(2 * np.pi * t[:, None] / 96 + phase[None, :])
    speed = speed + rng.normal(0, 3, size=speed.shape)
    vals = speed.astype(np.float32)[..., None]
    if n_features > 1:
        extras = rng.normal(0, 1, size=(n_steps, n_nodes, n_features - 1)).astype(np.float32)
        vals = np.concatenate([vals, extras], axis=-1)
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    pts = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    distances = np.linalg.norm(pts[:, None] - pts[None, :], axis=-1)
    return vals, distances
