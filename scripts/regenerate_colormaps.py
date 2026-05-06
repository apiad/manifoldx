"""Regenerate the viz colormap hex literals from matplotlib.

Usage:
    uv pip install matplotlib
    uv run python scripts/regenerate_colormaps.py

Paste the output into src/manifoldx/viz/colormaps.py.
"""
import matplotlib.colormaps as mplcm
import numpy as np

NAMES = ["viridis", "magma", "plasma", "inferno", "turbo", "gray"]

for name in NAMES:
    lut = (mplcm[name].resampled(256)(np.linspace(0, 1, 256)) * 255).astype(np.uint8)
    var = f"_{name.upper()}_HEX"
    print(f'{var} = (\n    "{lut.tobytes().hex()}"\n)\n')
