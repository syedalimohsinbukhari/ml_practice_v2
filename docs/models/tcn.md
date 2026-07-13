# tcn

## What it is

A Temporal Convolutional Network — a stack of causal, exponentially dilated
residual blocks:

```
(4096, 2) strain
  → BatchNormalization
  → stem: Conv1D(64, k=7, stride 2)                        # T = 2048
  → 10 × TCN block, dilation 1, 2, 4, …, 512:
        [causal dilated Conv1D(64, k=3) → BN → ReLU → SpatialDropout] × 2
        + residual shortcut
  → GAP ⊕ GMP concat → (128,) features
```

Receptive field ≈ 2 · (k−1) · Σ dilations ≈ 4092 samples — effectively the
entire window — reached with only ~10 thin layers.

## Why it's here

TCNs get a *full-window* receptive field at a fraction of the parameters and
depth a plain CNN would need — the cheapest way in the zoo to let every output
feature condition on the whole 2 s of strain. That is exactly the right bias
for mchirp (global frequency evolution) and a useful contrast with resnet1d,
which reaches similar coverage via downsampling instead of dilation: comparing
the two tells us whether preserving time resolution (TCN keeps T=2048
throughout) matters for merger-time accuracy.

Causality is inherited from the original design; it is not required for our
offline windows but is harmless, and keeps the door open for streaming-style
inference later.

## Provenance

Bai, Kolter & Koltun (2018), "An Empirical Evaluation of Generic Convolutional
and Recurrent Networks for Sequence Modeling"; WaveNet-style dilation stacks
(van den Oord et al. 2016).

## Knobs (trunk_cfg)

| key | default | meaning |
|-----|---------|---------|
| `filters` | 64 | channels in every block |
| `kernel_size` | 3 | conv kernel |
| `dilations` | `[1 … 512]` (powers of 2) | one block per entry |
| `dropout` | 0.1 | SpatialDropout1D rate |
| `stem_stride` | 2 | initial decimation (1 disables the stem) |
| `window_len` | 4096 | input length |
