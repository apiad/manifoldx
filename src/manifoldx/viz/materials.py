"""Sci-viz materials.

ColormapMaterial: maps a per-instance scalar value through a 1D LUT.
Camera-facing point sprites with sphere-imposter normal reconstruction.
"""
import numpy as np

from manifoldx.viz import colormaps


class ColormapMaterial:
    """Point-sprite material that colormaps a per-instance scalar.

    Per-batch uniform (4 floats):
        vmin, vmax, lit_flag, _pad

    Per-instance storage buffers (read in shader by instance_index):
        transforms (mat4x4)         — existing
        scalar_values (float)       — new, sourced from ScalarValue
        radii (float)               — new, sourced from Radius

    Per-batch texture: 256x1 RGBA8 1D LUT, sampled in fragment shader.
    """

    def __init__(self, cmap: str, vmin: float, vmax: float, lit: bool = False):
        # Validate cmap exists
        colormaps.get_colormap(cmap)
        self.cmap = cmap
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.lit = bool(lit)

    @property
    def pipeline_subtype(self) -> str:
        """Used by RenderPipeline cache to share pipelines across cmaps."""
        return self.cmap

    def get_data(self, n: int, registry=None) -> np.ndarray:
        """Per-batch material uniform data, broadcast to n rows.

        The renderer reads row 0 as the uniform; n rows are produced for
        compatibility with the existing material registry's per-instance
        data convention.
        """
        row = np.array(
            [self.vmin, self.vmax, 1.0 if self.lit else 0.0, 0.0],
            dtype=np.float32,
        )
        return np.broadcast_to(row, (n, 4)).copy()

    def get_lut(self) -> np.ndarray:
        """Return the (256, 4) uint8 LUT for this material's colormap."""
        return colormaps.get_colormap(self.cmap)
