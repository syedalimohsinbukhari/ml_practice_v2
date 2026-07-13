# cnn_attention

## What it is

A convolutional front-end that compresses the strain into a short token
sequence, followed by transformer encoder blocks and attention pooling:

```
(4096, 2) strain
  → BatchNormalization
  → 5 × [Conv1D(k=7, stride 2) → BN → ReLU]   filters 32→64→128→128→128
                                              # T: 4096 → 128 tokens
  → Dense projection to model_dim = 128
  → learned positional embedding (additive)
  → 2 × transformer encoder (MHA 4 heads + FFN 256, post-LN, dropout 0.1)
  → attention pooling (learned scoring → softmax-weighted sum) → (128,) features
```

## Why it's here

Convolutions are translation-equivariant: good for finding chirp structure
anywhere, but weak at *relating* distant parts of the signal to each other.
Self-attention over the 128-token sequence lets the model directly compare
early-inspiral tokens with near-merger tokens (the relation that encodes the
chirp rate, hence mchirp) and compare H1 features against L1 features at
different lags — information relevant even for our non-sky heads via effective
amplitude/SNR. Attention pooling replaces GAP with a learned "where to look"
summary, which should sharpen merger-time.

This is the heaviest trunk and the most data-hungry; it is GPU-targeted and
included as the "is attention worth it at 25k samples?" experiment. The conv
front-end (rather than raw patching) is what makes it tractable: attention is
O(T²), so we attend over 128 tokens, not 4096 samples.

## Provenance

Transformer encoder (Vaswani et al. 2017); conv-frontend + attention hybrids
as used in speech (Gulati et al. 2020, Conformer) and in GW work such as
Zhao et al. 2023 and other transformer-based GW detection papers.

## Knobs (trunk_cfg)

| key | default | meaning |
|-----|---------|---------|
| `conv_filters` | `[32, 64, 128, 128, 128]` | one stride-2 conv per entry |
| `kernel_size` | 7 | front-end conv kernel |
| `model_dim` | 128 | token width |
| `num_blocks` | 2 | encoder blocks |
| `num_heads` | 4 | attention heads |
| `ff_dim` | 256 | FFN hidden width |
| `dropout` | 0.1 | attention + FFN dropout |
| `window_len` | 4096 | input length |
