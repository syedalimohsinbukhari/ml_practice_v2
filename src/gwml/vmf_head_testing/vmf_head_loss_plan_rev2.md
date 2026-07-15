That's a solid fix to the dataclass — defaulting `column` to `None` and validating in `__post_init__` avoids the "meaningless placeholder" problem I flagged, and it keeps dataclass field ordering valid since `column` picked up a default too.

Two things worth nailing down before you write `columns=(5, 6)`:

**1. Order convention — document it explicitly, don't leave it implicit.** `PARAM_COLUMNS` has `declination: 5, ra: 6`, so `columns=(5, 6)` reads as `(dec, ra)` by position — but that's easy to get backwards later (someone skimming the tuple six months from now will reasonably assume `(ra, dec)`, since that's the more common way people say the pair out loud). Worth either:
- naming it explicitly as `columns=(PARAM_COLUMNS["declination"], PARAM_COLUMNS["ra"])` so it's self-documenting at the call site, or
- adding a comment right next to the `radec_to_unit_vector(ra, dec)` call in `transforms.py` confirming which tuple slot maps to which argument.

This is exactly the kind of silent-wrong-not-crashing bug I flagged for the z-score issue earlier — if dec and ra get swapped, training would proceed, the loss would decrease, and you'd get a plausible-looking but geometrically wrong sky position, likely not caught until someone eyeballs actual predicted vs. true coordinates.

**2. Ripple effect worth adding to the task list now: existing YAML configs.** Once `ra`/`declination` are removed from `HEAD_SPECS`, any config with `model.heads: [..., "ra", "declination", ...]` will hit `resolve_heads()`'s unknown-head `ValueError` — which is good (fails loud, not silent), but means you'll need to grep `configs/*.yaml` and swap those two entries for `"sky_position"` before those configs run again. Worth doing that sweep as part of this same change rather than discovering it config-by-config later.

Everything else — `__post_init__` validation, the two-pass sequencing, keeping `heads.py`/`losses.py`/eval-script changes bundled into Pass 1 as discussed — looks right. Go ahead with `heads_spec.py`.
