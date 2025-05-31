"""CLI: stgnn-bench --model stgcn|dcrnn|graphwavenet|mtgnn."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import yaml

from stgnn.data import synthesize
from stgnn.runner import MODEL_NAMES, BenchConfig, run_benchmark


def bench_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--model", choices=MODEL_NAMES + ("all",), default="all")
    p.add_argument("--values", type=Path)
    p.add_argument("--distances", type=Path)
    p.add_argument("--demo", action="store_true")
    p.add_argument("--out", type=Path, default=Path("bench_results.json"))
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raw = yaml.safe_load(args.config.read_text()) or {}
    cfg = BenchConfig(**raw)

    if args.demo or args.values is None:
        values, distances = synthesize(n_steps=400, n_nodes=8)
    else:
        values = np.load(args.values)["data"]
        distances = np.load(args.distances)["data"] if args.distances else np.eye(values.shape[1])

    targets = MODEL_NAMES if args.model == "all" else (args.model,)
    results = [run_benchmark(values, distances, m, cfg) for m in targets]
    args.out.write_text(json.dumps(results, indent=2))
    for r in results:
        print(f"{r['model']:14s} test_mae={r['test_mae']:.4f} "
              f"test_rmse={r['test_rmse']:.4f}")
    return 0
