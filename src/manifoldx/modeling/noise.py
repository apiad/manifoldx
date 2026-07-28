"""Back-compat shim. Noise sources now live in manifoldx.modeling.fields."""

from manifoldx.modeling.fields import perlin, fbm, _resolve_rng  # noqa: F401

__all__ = ["perlin", "fbm"]
