import gwml.models.trunks  # noqa: F401  (side effect: registers all trunks)
from gwml.models.heads import attach_heads
from gwml.models.registry import available, build_trunk, register


def build_model(trunk_name: str, trunk_cfg: dict | None = None,
                head_cfg: dict | None = None, heads=None):
    """Trunk name + configs + active heads -> functional multi-head keras.Model."""
    inputs, features = build_trunk(trunk_name, trunk_cfg)
    return attach_heads(inputs, features, heads, head_cfg)


__all__ = ["attach_heads", "available", "build_model", "build_trunk", "register"]
