"""Config-driven training: YAML in, run directory out.

Usage (via scripts/train.py):  python scripts/train.py configs/resnet1d.yaml
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import keras
import yaml

from gwml.data.loader import build_subset_masks, load_arrays, make_dataset
from gwml.data.transforms import TargetTransforms
from gwml.models import build_model
from gwml.training.callbacks import (
    DiagnosticSubsetsCallback,
    LiveScatterCallback,
    WarmupLR,
)
from gwml.training.losses import MultiHeadTrainer


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _run_base_dir(cfg: dict) -> Path:
    return Path(cfg.get("run_dir", f"runs/{cfg['name']}"))


def _timestamp(now: datetime | None = None) -> str:
    return (now or datetime.now()).strftime("%Y%m%d_%H%M%S")


def create_run_dir(cfg: dict, now: datetime | None = None) -> Path:
    """Create a timestamped run directory under the configured run base."""
    base_dir = _run_base_dir(cfg)
    stamp = _timestamp(now)

    for suffix in [""] + [f"_{i:02d}" for i in range(1, 100)]:
        run_dir = base_dir / f"{stamp}{suffix}"
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return run_dir

    raise RuntimeError(f"could not create a unique run directory under {base_dir}")


def latest_run_dir(cfg: dict) -> Path:
    """Return the newest timestamped run directory, falling back to legacy paths."""
    base_dir = _run_base_dir(cfg)
    legacy_weights = base_dir / "best.weights.h5"
    if legacy_weights.exists():
        return base_dir

    children = [p for p in base_dir.iterdir() if p.is_dir()] if base_dir.exists() else []
    if not children:
        return base_dir

    return max(children, key=lambda p: (p.stat().st_mtime, p.name))


def build_trainer(cfg: dict) -> MultiHeadTrainer:
    heads = cfg["model"].get("heads")  # None -> DEFAULT_HEADS
    base = build_model(
        cfg["model"]["trunk"],
        cfg["model"].get("trunk_cfg", {}),
        cfg["model"].get("head_cfg", {}),
        heads=heads,
    )
    trainer = MultiHeadTrainer(base, cfg.get("loss", {}), heads=heads)
    trainer.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=cfg.get("optim", {}).get("lr", 1e-3)
        )
    )
    return trainer


def _build_callbacks(cfg, run_dir, val_strain, val_params, transforms):
    tcfg = cfg.get("train", {})
    diag_n = tcfg.get("diagnostics_every_n", 5)
    n_scatter = tcfg.get("scatter_subset", 1000)
    callbacks = [
        LiveScatterCallback(
            val_strain[:n_scatter], val_params[:n_scatter], transforms,
            run_dir / "scatter", every_n=diag_n,
        ),
        DiagnosticSubsetsCallback(
            val_strain, val_params, transforms,
            run_dir / "diagnostics.csv", every_n=diag_n,
        ),
        keras.callbacks.ModelCheckpoint(
            str(run_dir / "best.weights.h5"),
            monitor="val_loss", save_best_only=True, save_weights_only=True,
        ),
        keras.callbacks.CSVLogger(str(run_dir / "history.csv")),
        keras.callbacks.TensorBoard(log_dir=str(run_dir / "tb")),
    ]

    ocfg = cfg.get("optim", {})
    base_lr = ocfg.get("lr", 1e-3)
    warmup = ocfg.get("warmup_epochs", 0)
    lr_cfg = ocfg.get("schedule", {"type": "plateau"})
    if lr_cfg.get("type") == "plateau":
        if warmup:
            callbacks.append(WarmupLR(base_lr, warmup))
        callbacks.append(keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=lr_cfg.get("factor", 0.5),
            patience=lr_cfg.get("patience", 5),
            min_lr=lr_cfg.get("min_lr", 1e-6),
        ))
    elif lr_cfg.get("type") == "step":
        step, gamma = lr_cfg.get("step_epochs", 20), lr_cfg.get("gamma", 0.5)

        def step_lr(epoch):
            if epoch < warmup:
                return base_lr * (epoch + 1) / warmup
            return base_lr * gamma ** ((epoch - warmup) // step)

        callbacks.append(keras.callbacks.LearningRateScheduler(step_lr))

    if tcfg.get("early_stopping", False):
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=tcfg.get("early_stopping_patience", 15),
            restore_best_weights=True,
        ))
    return callbacks


def run_experiment(config_path: str | Path) -> MultiHeadTrainer:
    cfg = load_config(config_path)
    tcfg = cfg.get("train", {})

    seed = tcfg.get("seed", 42)
    keras.utils.set_random_seed(seed)
    if tcfg.get("deterministic_ops", False):
        import tensorflow as tf

        tf.config.experimental.enable_op_determinism()

    run_dir = create_run_dir(cfg)
    print(f"run directory: {run_dir}")

    dcfg = cfg["data"]
    max_n = dcfg.get("max_samples")
    train_strain, train_params = load_arrays(dcfg["path"], "training", max_n)
    val_strain, val_params = load_arrays(dcfg["path"], "validation", max_n)

    # --- targeted oversampling (Phase 3.1) ---
    aug_cfg = dcfg.get("augmentation", {}).get("oversample", {})
    if aug_cfg:
        masks = build_subset_masks(train_params)
        for subset_name, factor in aug_cfg.items():
            mask = masks.get(subset_name)
            if mask is not None and mask.any() and factor > 1:
                n_dup = int(mask.sum())
                dup_strain = train_strain[mask]
                dup_params = train_params[mask]
                for _ in range(int(factor) - 1):
                    train_strain = np.concatenate([train_strain, dup_strain], axis=0)
                    train_params = np.concatenate([train_params, dup_params], axis=0)
                print(f"oversample: {subset_name} x{factor} ({n_dup} rows → "
                      f"{n_dup * factor} rows)")
    # --- end oversampling ---

    transforms = TargetTransforms(heads=cfg["model"].get("heads")).fit(train_params)
    transforms.to_json(run_dir / "transforms.json")

    alpha = cfg.get("loss", {}).get("snr_weight_alpha")  # None = off
    batch = dcfg.get("batch_size", 128)
    train_ds = make_dataset(train_strain, train_params, transforms, batch,
                            shuffle=True, seed=seed, snr_weight_alpha=alpha)
    val_ds = make_dataset(val_strain, val_params, transforms, batch)

    trainer = build_trainer(cfg)
    trainer.fit(
        train_ds,
        validation_data=val_ds,
        epochs=tcfg.get("epochs", 50),
        callbacks=_build_callbacks(cfg, run_dir, val_strain, val_params, transforms),
        verbose=tcfg.get("verbose", 2),
    )
    trainer.save_weights(run_dir / "final.weights.h5")
    return trainer
