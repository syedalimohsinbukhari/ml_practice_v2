"""Custom callbacks: LiveScatterCallback and DiagnosticSubsetsCallback.

Both work on a fixed validation subset in *physical* units, so their output is
directly interpretable while training runs.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import keras

from gwml.data.loader import build_subset_masks
from gwml.data.transforms import PARAM_COLUMNS, TargetTransforms, signed_error
from gwml.evaluation.plots import scatter_grid


def _predict_physical(model, strain, transforms, batch_size):
    pred = model.predict(strain, batch_size=batch_size, verbose=0)
    return transforms.inverse(pred)


class WarmupLR(keras.callbacks.Callback):
    """Linear LR warmup over the first ``warmup_epochs`` epochs.

    Sets lr = base_lr * (epoch + 1) / warmup_epochs during warmup, then stops
    touching the optimizer so ReduceLROnPlateau (or nothing) takes over.
    """

    def __init__(self, base_lr: float, warmup_epochs: int):
        super().__init__()
        self.base_lr = float(base_lr)
        self.warmup_epochs = int(warmup_epochs)

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
            self.model.optimizer.learning_rate.assign(lr)


class PeriodicCheckpoint(keras.callbacks.Callback):
    """Save weights every N epochs, in addition to best/final.

    Historically only ``best.weights.h5`` (best val_loss) and
    ``final.weights.h5`` (end of training) were ever saved — instrument
    calibration against early training was blocked once because no
    intermediate checkpoint existed (see the φc/ψ degeneracy redo's
    perturbation-trace work, and this repo's CLAUDE.md "Training runs:
    periodic checkpoints" rule). This callback closes that gap.
    """

    def __init__(self, out_dir: str | Path, every_n: int = 10):
        super().__init__()
        self.out_dir = Path(out_dir)
        self.every_n = every_n

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every_n:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(
            str(self.out_dir / f"epoch_{epoch + 1:03d}.weights.h5")
        )


class LiveScatterCallback(keras.callbacks.Callback):
    """Every N epochs, save a 2x2 pred-vs-true scatter grid to PNG.

    Lets you watch each head converge — or collapse to predicting the mean —
    without waiting for training to finish.
    """

    def __init__(
        self,
        strain: np.ndarray,
        params: np.ndarray,
        transforms: TargetTransforms,
        out_dir: str | Path,
        every_n: int = 5,
        batch_size: int = 256,
    ):
        super().__init__()
        self.strain = strain
        self.true_physical = transforms.physical_targets(params)
        self.transforms = transforms
        self.out_dir = Path(out_dir)
        self.every_n = every_n
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every_n:
            return
        pred = _predict_physical(
            self.model, self.strain, self.transforms, self.batch_size
        )
        scatter_grid(
            self.true_physical,
            pred,
            self.transforms.heads,
            self.out_dir / f"epoch_{epoch + 1:04d}.png",
            title=f"epoch {epoch + 1}",
        )


class DiagnosticSubsetsCallback(keras.callbacks.Callback):
    """Per-head MAE on fixed validation subsets every N epochs, appended to CSV.

    Subsets: SNR terciles (errors should order cleanly by SNR), mchirp
    low/high halves (mass-range bias), merger-time early/late halves
    (time-localization bias), q terciles and their cross with mchirp
    low/high (near-equal-mass and mchirp-confound bias — see
    q_head_action_plan.md).
    """

    def __init__(
        self,
        strain: np.ndarray,
        params: np.ndarray,
        transforms: TargetTransforms,
        csv_path: str | Path,
        every_n: int = 5,
        batch_size: int = 256,
    ):
        super().__init__()
        self.strain = strain
        self.params = params
        self.true_physical = transforms.physical_targets(params)
        self.transforms = transforms
        self.csv_path = Path(csv_path)
        self.every_n = every_n
        self.batch_size = batch_size
        self.subsets = self._build_subsets(params)

    @staticmethod
    def _build_subsets(params: np.ndarray) -> dict[str, np.ndarray]:
        return build_subset_masks(params)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every_n:
            return
        pred = _predict_physical(
            self.model, self.strain, self.transforms, self.batch_size
        )
        heads = self.transforms.heads
        write_header = not self.csv_path.exists()
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "epoch", "subset", "n",
                    *[f"mae_{h}" for h in heads],
                    *[f"r2_{h}" for h in heads],
                    *[f"std_ratio_{h}" for h in heads],
                ])
            for name, mask in self.subsets.items():
                maes, r2s, ratios = [], [], []
                for h in heads:
                    t = np.ravel(self.true_physical[h][mask])
                    p = np.ravel(pred[h][mask])
                    res = signed_error(h, t, p)
                    maes.append(float(np.mean(np.abs(res))))
                    ss_tot = float(np.sum((t - t.mean()) ** 2))
                    ss_res = float(np.sum(res**2))
                    r2s.append(1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan)
                    ratios.append(float(p.std() / (t.std() + 1e-12)))
                writer.writerow([
                    epoch + 1, name, int(mask.sum()),
                    *[f"{v:.6g}" for v in (*maes, *r2s, *ratios)],
                ])
