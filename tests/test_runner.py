import os
from pathlib import Path

import pytest

os.environ["WANDB_MODE"] = "disabled"

from stgnn.data import synthesize
from stgnn.runner import MODEL_NAMES, BenchConfig, run_benchmark


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_runner_smoke_per_model(model_name: str, tmp_path: Path):
    cfg = BenchConfig(
        encoder_steps=6, decoder_steps=3, hidden=8, epochs=1, batch_size=2,
        device="cpu", val_fraction=0.3, test_fraction=0.3,
        mlflow_tracking_uri=f"file:{tmp_path / 'mlruns'}",
    )
    values, distances = synthesize(n_steps=40, n_nodes=4, seed=0)
    out = run_benchmark(values, distances, model_name, cfg)
    assert out["model"] == model_name
    assert out["test_mae"] == out["test_mae"]  # not NaN
