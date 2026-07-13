# inception_time

## What it is

An InceptionTime-style multi-scale trunk:

```
(4096, 2) strain
  → BatchNormalization
  → stem: Conv1D(32, k=7, stride 4)                       # T = 1024
  → 6 × inception module:
        bottleneck Conv1D(32, k=1)
        ∥ Conv1D(32, k=9) ∥ Conv1D(32, k=19) ∥ Conv1D(32, k=39)
        ∥ MaxPool(3, stride 1) → Conv1D(32, k=1)
        → concat (128 ch) → BN → ReLU
     with a residual shortcut every 3 modules
  → GlobalAveragePooling1D → (128,) features
```

## Why it's here

A chirp is a signal whose characteristic timescale *changes through the
window*: long, slow cycles early in the inspiral, millisecond structure near
merger. A single conv kernel size must compromise; parallel kernels (9/19/39
samples ≈ 4–19 ms at 2048 Hz) let every block see short, medium, and long
structure simultaneously. This is the strongest published general-purpose
time-series classifier family, and its inductive bias matches the
time-varying-frequency nature of GW signals unusually well.

The stride-4 stem is our adaptation: at T=4096 the original full-resolution
InceptionTime is needlessly expensive, and the first decimation loses little
for a signal band-limited well below Nyquist.

## Provenance

InceptionTime (Ismail Fawaz et al. 2020), itself descended from Inception-v4;
bottleneck + parallel-kernel design as in the original, residual every 3
blocks as in the paper.

## Knobs (trunk_cfg)

| key | default | meaning |
|-----|---------|---------|
| `depth` | 6 | number of inception modules |
| `filters` | 32 | filters per branch (4 branches → 128 ch) |
| `kernel_sizes` | `[9, 19, 39]` | parallel conv branch kernels |
| `bottleneck` | 32 | 1×1 bottleneck width |
| `stem_stride` | 4 | initial decimation (1 disables the stem) |
| `window_len` | 4096 | input length |
