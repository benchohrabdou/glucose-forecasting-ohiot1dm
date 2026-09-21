"""Train one model for one horizon.

``python -m src.train --config configs/lstm_ph30.yaml``

MSE loss on scaled glucose, Adam, gradient clipping, early stopping on VALIDATION RMSE in mg/dL
(patience epochs) with the best weights restored. Test data is never touched here. Saves the
checkpoint together with the training-fit scaler, and persists the seed, the resolved config and
the per-epoch log to results/.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import build_datasets, collect_arrays, feature_columns
from src.evaluate import predict
from src.models import build_model
from src.utils import get_logger, load_config, rmse, set_seed

log = get_logger(__name__)


def train(cfg: dict, name: str) -> Path:
    """Fit on the training windows, early-stop on validation, save the best checkpoint."""
    seed, tcfg = cfg["seed"], cfg["train"]
    set_seed(seed)
    train_ds, val_ds, _, scaler = build_datasets(cfg)  # scaler fit on training rows only
    n_features = len(feature_columns(cfg))
    xva, yva, _, _ = collect_arrays(val_ds)
    yva_mgdl = scaler.unscale_glucose(yva.numpy()).ravel()

    model = build_model(cfg, n_features)
    optimiser = torch.optim.Adam(model.parameters(), lr=tcfg["lr"])
    loss_fn = nn.MSELoss()
    loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(seed))

    results_dir, ckpt_dir = Path(cfg["paths"]["results_dir"]), Path(cfg["paths"]["checkpoint_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{name}_resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    log.info("%s: seed=%d, %d train / %d val windows, %d features", name, seed, len(train_ds), len(val_ds), n_features)

    best, best_state, best_epoch, history = float("inf"), None, 0, []
    for epoch in range(1, tcfg["max_epochs"] + 1):
        model.train()
        losses = []
        for x, y, _ in loader:
            optimiser.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            optimiser.step()
            losses.append(loss.item())
        val_rmse = rmse(scaler.unscale_glucose(predict(model, xva).numpy()).ravel(), yva_mgdl)
        history.append({"epoch": epoch, "train_mse_scaled": float(np.mean(losses)), "val_rmse_mgdl": val_rmse})
        log.info("epoch %3d  train_mse=%.4f  val_rmse=%.2f mg/dL", epoch, history[-1]["train_mse_scaled"], val_rmse)
        if val_rmse < best:
            best, best_epoch, best_state = val_rmse, epoch, copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= tcfg["patience"]:
            log.info("early stop: no validation improvement for %d epochs", tcfg["patience"])
            break

    pd.DataFrame(history).to_csv(results_dir / f"{name}_training_log.csv", index=False)
    path = ckpt_dir / f"{name}.pt"
    torch.save({"model_state": best_state, "cfg": cfg, "scaler": {"mean": scaler.mean, "std": scaler.std},
                "columns": feature_columns(cfg), "seed": seed, "best_epoch": best_epoch, "best_val_rmse": best}, path)
    scaler.save(ckpt_dir / f"{name}.scaler.json")
    log.info("best epoch %d, val RMSE %.2f mg/dL -> %s", best_epoch, best, path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a forecasting model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, help="override the config seed; run is named <config>_seed<N>")
    args = parser.parse_args()
    cfg, name = load_config(args.config), Path(args.config).stem
    if args.seed is not None:
        cfg["seed"], name = args.seed, f"{name}_seed{args.seed}"
    train(cfg, name)


if __name__ == "__main__":
    main()
