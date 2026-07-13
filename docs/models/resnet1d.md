# resnet1d

## What it is

A 1D residual network — the expected workhorse of the zoo:

```
(4096, 2) strain
  → BatchNormalization
  → stem: Conv1D(64, k=15, stride 2) → BN → ReLU → MaxPool(2)     # T = 1024
  → stage 0: 2 residual blocks, 64 filters,  first block stride 2  # T = 512
  → stage 1: 2 residual blocks, 128 filters, stride 2, dilation 2  # T = 256
  → stage 2: 2 residual blocks, 256 filters, stride 2, dilation 4  # T = 128
  → GAP ⊕ GMP concat → (512,) features
```

Each block: Conv → BN → ReLU → dilated Conv → BN, added to a (projected)
shortcut, then ReLU.

## Why it's here

Residual connections let the trunk go deep enough to build hierarchical
features (cycles → chirp segments → whole-chirp descriptors) without the
vanishing-gradient pathologies of a plain deep CNN. Strided downsampling plus
dilation in the later stages grows the receptive field to cover the full 2 s
window, which matters because mchirp is encoded in the *global* frequency
evolution, not any local patch. The GAP ⊕ GMP concat keeps both an average
summary and a "sharpest event" summary — the latter helps merger-time.

## Provenance

ResNet (He et al. 2016) translated to 1D; deep dilated CNNs on whitened strain
follow George & Huerta (2018) and the later dilated variants used across the
GW ML literature (e.g. Gebhard et al. 2019's convolutional detection nets).

## Knobs (trunk_cfg)

| key | default | meaning |
|-----|---------|---------|
| `stem_filters` / `stem_kernel` | 64 / 15 | stem conv |
| `stage_filters` | `[64, 128, 256]` | filters per stage |
| `blocks_per_stage` | 2 | residual blocks per stage |
| `kernel_size` | 7 | block conv kernel |
| `stage_dilations` | `[1, 2, 4]` | dilation of each stage's second conv |
| `window_len` | 4096 | input length |
