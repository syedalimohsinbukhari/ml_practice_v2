"""Trunk registry: name -> builder.

Every trunk module registers a builder with signature

    build(cfg: dict) -> (inputs: keras.Input, features: KerasTensor)

where ``inputs`` has shape (window_len, 2) and ``features`` is the pooled
feature vector (B, F). Heads, losses, and evaluation never change across
trunks — swapping architectures is a one-line config change.
"""

from __future__ import annotations

from collections.abc import Callable

_TRUNKS: dict[str, Callable] = {}


def register(name: str):
    def decorator(fn):
        if name in _TRUNKS:
            raise ValueError(f"trunk {name!r} registered twice")
        _TRUNKS[name] = fn
        return fn

    return decorator


def build_trunk(name: str, cfg: dict | None = None):
    if name not in _TRUNKS:
        raise KeyError(f"unknown trunk {name!r}; available: {available()}")
    return _TRUNKS[name](cfg or {})


def available() -> list[str]:
    return sorted(_TRUNKS)
