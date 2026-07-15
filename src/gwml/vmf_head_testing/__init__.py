"""Prototype vMF sky-head — integrated into the main pipeline 2026-07-15.

The core functions have been moved into the main modules:
  - sky_transform.py   → gwml.data.sky_transform
  - vmf_nll_loss       → gwml.training.losses
  - build_vmf_head     → gwml.models.heads (inline in attach_heads)
  - sky_position spec  → gwml.heads_spec

The files here are kept as a standalone reference / sanity-check playground.
"""
