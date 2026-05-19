"""
train.py
RO-2026-002 — ABPET Threshold
Frontier Translational Research Lab (Independent)
PI: Julian Borges, MD

Training loop for a single model. Called by ablation.py for each of A-F.

Usage:
    python src/train.py --model_name E --train_csv data/processed/adni_train.csv \
        --val_csv data/processed/adni_val.csv --epochs 100 --output_dir results/model_E/
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model import get_model
from src.losses import get_loss
from src.dataset import PETDataset
from src.augmentation import get_train_transform, get_val_transform
from src.evaluate import global_mae, pearson_r

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_medicalnet_weights(model: nn.Module, weights_path: str) -> nn.Module:
    """
    Load MedicalNet resnet_18.pth encoder weights for Model F.
    Head is randomly initialized (regression task differs from segmentation).
    If weights produce worse TSA_24 than Model E, this is a publishable null result.
    Document outcome in DEC-009.
    """
    state = torch.load(weights_path, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]

    model_state = model.state_dict()
    matched = {k: v for k, v in state.items()
               if k in model_state and v.shape == model_state[k].shape}
    model_state.update(matched)
    model.load_state_dict(model_state)
    log.info(f"MedicalNet: loaded {len(matched)}/{len(model_state)} layers")
    return model


def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0
    for imgs, targets, tracer_idx, _ in loader:
        imgs = imgs.to(device)
        targets = targets.to(device)
        tracer_idx = tracer_idx.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            preds = model(imgs, tracer_idx)
            loss = criterion(preds, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    for imgs, targets, tracer_idx, _ in loader:
        imgs = imgs.to(device)
        tracer_idx = tracer_idx.to(device)
        preds = model(imgs, tracer_idx)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.numpy())
    y_pred = np.array(all_preds)
    y_true = np.array(all_targets)
    return y_pred, y_true


def train_model(model_name: str, train_csv: str, val_csv: str,
                epochs: int, output_dir: str,
                batch_size: int = 4, lr: float = 1e-4,
                medicalnet_weights: str = None) -> dict:

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    log.info(f"Training Model {model_name} on {device}")

    train_ds = PETDataset(train_csv, transform=get_train_transform(model_name))
    val_ds = PETDataset(val_csv, transform=get_val_transform(model_name))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    model = get_model(model_name).to(device)

    if model_name.upper() == "F" and medicalnet_weights:
        model = load_medicalnet_weights(model, medicalnet_weights)

    criterion = get_loss(model_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # LR scheduler: cosine for B-F, none for A (hackathon baseline)
    if model_name.upper() == "A":
        scheduler = None
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6)

    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    best_mae = float("inf")
    best_epoch = 0
    patience = 15
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        y_pred, y_true = validate(model, val_loader, device)
        val_mae = global_mae(y_true, y_pred)
        val_r = pearson_r(y_true, y_pred)

        if scheduler:
            scheduler.step()

        history.append({
            "epoch": epoch, "train_loss": round(train_loss, 4),
            "val_mae": round(val_mae, 4), "val_r": round(val_r, 4)
        })
        log.info(f"Epoch {epoch:3d} | loss={train_loss:.4f} | val_mae={val_mae:.4f} | r={val_r:.4f}")

        if val_mae < best_mae:
            best_mae = val_mae
            best_epoch = epoch
            torch.save(model.state_dict(), output_dir / f"model_{model_name}_best.pt")

        if epoch - best_epoch >= patience:
            log.info(f"Early stopping at epoch {epoch} (patience={patience})")
            break

    with open(output_dir / f"history_{model_name}.json", "w") as f:
        json.dump(history, f, indent=2)

    log.info(f"Model {model_name} done. Best val MAE={best_mae:.4f} at epoch {best_epoch}")
    return {"model": model_name, "best_val_mae": best_mae, "best_epoch": best_epoch}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True, choices=list("ABCDEF"))
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--medicalnet_weights", default=None,
                        help="Path to MedicalNet resnet_18.pth (Model F only)")
    args = parser.parse_args()
    train_model(**vars(args))
