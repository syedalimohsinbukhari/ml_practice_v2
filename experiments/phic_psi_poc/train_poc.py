#!/usr/bin/env python
"""Train Run A or Run B for the φc/ψ degeneracy PoC.

Replicates the flow from ``src/gwml/training/train.py`` but uses
``SumDiffTrainer`` instead of ``MultiHeadTrainer``, and automatically
ensures the required heads (``coa_phase``, ``polarization_angle``,
``inclination``) are present in the head list.

Usage::

    # Run B (PoC — sum/diff heads + curriculum weighting)
    python experiments/phic_psi_poc/train_poc.py \\
        experiments/phic_psi_poc/config_poc.yaml

    # Run A (baseline — independent φc/ψ, same circular loss)
    python experiments/phic_psi_poc/train_poc.py \\
        experiments/phic_psi_poc/config_baseline.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src/ is on the path (same pattern as scripts/train.py:11)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import keras
import numpy as np
import yaml

from gwml.data.loader import build_subset_masks, load_arrays, make_dataset
from gwml.data.transforms import TargetTransforms
from gwml.models import build_model
from gwml.training.callbacks import (
    DiagnosticSubsetsCallback,
    LiveScatterCallback,
    WarmupLR,
)
from gwml.training.train import (
    _build_callbacks,
    _timestamp,
    create_run_dir,
    load_config,
)

from experiments.phic_psi_poc.trainer import SumDiffTrainer


# Heads that MUST be present for the PoC to work
_REQUIRED_HEADS = ("coa_phase", "polarization_angle", "inclination")


def _ensure_required_heads(heads: list[str] | None) -> list[str]:
    """Add coa_phase, polarization_angle, and inclination if missing.

    If ``heads`` is None, starts from ``DEFAULT_HEADS``.  Inserts new
    heads at the end of the list (order doesn't affect the model — each
    head is an independent branch).
    """
    from gwml.heads_spec import DEFAULT_HEADS

    result = list(heads) if heads is not None else list(DEFAULT_HEADS)
    for h in _REQUIRED_HEADS:
        if h not in result:
            result.append(h)
    return result


def build_sumdiff_trainer(cfg: dict) -> SumDiffTrainer:
    """Build the base model + SumDiffTrainer, compiled and ready to fit.

    Reads mode, well_constrained_combo, and combo_log_var_clamp from
    ``cfg["loss"]`` and passes them to the trainer constructor.
    """
    loss_cfg = cfg.get("loss", {})
    mode = loss_cfg.get("mode", "poc")

    heads = _ensure_required_heads(cfg["model"].get("heads"))

    base = build_model(
        cfg["model"]["trunk"],
        cfg["model"].get("trunk_cfg", {}),
        cfg["model"].get("head_cfg", {}),
        heads=heads,
    )

    trainer_kwargs = {}
    if mode == "poc":
        trainer_kwargs["well_constrained_combo"] = loss_cfg.get(
            "well_constrained_combo", "combo_A"
        )

    trainer = SumDiffTrainer(
        base,
        loss_cfg,
        heads=heads,
        mode=mode,
        **trainer_kwargs,
    )

    lr = cfg.get("optim", {}).get("lr", 1e-3)
    trainer.compile(optimizer=keras.optimizers.Adam(learning_rate=lr))
    return trainer


def run_poc_experiment(config_path: str | Path) -> SumDiffTrainer:
    """Main entry point — analogous to ``gwml.training.train.run_experiment()``.

    Returns the trained ``SumDiffTrainer`` instance.
    """
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

    # --- Targeted oversampling (Phase 3.1, copied from train.py) ---
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
                    train_strain = np.concatenate(
                        [train_strain, dup_strain], axis=0
                    )
                    train_params = np.concatenate(
                        [train_params, dup_params], axis=0
                    )
                print(
                    f"oversample: {subset_name} x{factor} "
                    f"({n_dup} rows → {n_dup * factor} rows)"
                )

    # Build transforms with the augmented head list
    heads = _ensure_required_heads(cfg["model"].get("heads"))
    transforms = TargetTransforms(heads=heads).fit(train_params)
    transforms.to_json(run_dir / "transforms.json")

    alpha = cfg.get("loss", {}).get("snr_weight_alpha")
    batch = dcfg.get("batch_size", 128)
    train_ds = make_dataset(
        train_strain, train_params, transforms, batch,
        shuffle=True, seed=seed, snr_weight_alpha=alpha,
    )
    val_ds = make_dataset(val_strain, val_params, transforms, batch)

    trainer = build_sumdiff_trainer(cfg)
    trainer.fit(
        train_ds,
        validation_data=val_ds,
        epochs=tcfg.get("epochs", 50),
        callbacks=_build_callbacks(
            cfg, run_dir, val_strain, val_params, transforms
        ),
        verbose=tcfg.get("verbose", 2),
    )
    trainer.save_weights(run_dir / "final.weights.h5")
    return trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config", help="Path to YAML experiment config"
    )
    args = parser.parse_args()
    run_poc_experiment(args.config)


if __name__ == "__main__":
    main()
