# cnn_baseline

## What it is

A deliberately small plain convolutional network, ~100k parameters:

```
(4096, 2) strain
  → BatchNormalization                  # in-model input normalization
  → 4 × [Conv1D(k=16) → BN → ReLU → MaxPool(4)]
        filters 32 → 64 → 128 → 128
        length   1024 → 256 → 64 → 16
  → GlobalAveragePooling1D → (128,) features
```

## Why it's here

It exists to **prove the pipeline**, not to win. If cnn_baseline can't beat
predicting the training mean, the bug is in data loading, transforms, losses,
or callbacks — not the architecture. It's also the overfit-one-batch and smoke-
config workhorse: fast enough to run anywhere, including CPU.

The aggressive 4× pooling gives each GAP feature a receptive field covering the
whole window cheaply, which is enough to pick up the chirp's gross morphology
(duration, bandwidth, amplitude) that determines mchirp and SNR to first order.

## Provenance

The generic conv-pool stack every GW deep-learning paper starts from; closest
to the plain CNNs of George & Huerta (2018) and Gabbard et al. (2018), scaled
down.

## Knobs (trunk_cfg)

| key | default | meaning |
|-----|---------|---------|
| `filters` | `[32, 64, 128, 128]` | one entry per conv block |
| `kernel_size` | 16 | conv kernel length |
| `pool_size` | 4 | max-pool factor per block |
| `window_len` | 4096 | input length |
