"""Train and evaluate several configs over several seeds, resumably.

``python -m src.run_seeds --configs configs/lstm_ph30.yaml configs/lstm_ins_ph30.yaml --seeds 0 1 2 3 4``

Each run is named ``<config>_seed<N>``; its checkpoint and per-patient test table are skipped if
they already exist. Runs only write their own files, so several workers can run in parallel;
rebuild the shared tables afterwards with ``python -m src.evaluate --config configs/base.yaml``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluate import evaluate_checkpoint
from src.train import train
from src.utils import get_logger, load_config

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train + evaluate configs over seeds.")
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    args = parser.parse_args()
    for seed in args.seeds:  # seeds outermost: an interrupted sweep still has every variant for early seeds
        for config in args.configs:
            cfg, name = load_config(config), f"{Path(config).stem}_seed{seed}"
            cfg["seed"] = seed
            results = Path(cfg["paths"]["results_dir"]) / f"model_{name}_per_patient.csv"
            ckpt = Path(cfg["paths"]["checkpoint_dir"]) / f"{name}.pt"
            if results.exists() and ckpt.exists():
                log.info("%s already done, skipping", name)
                continue
            evaluate_checkpoint(cfg, train(cfg, name), name)
            log.info("%s done", name)


if __name__ == "__main__":
    main()
