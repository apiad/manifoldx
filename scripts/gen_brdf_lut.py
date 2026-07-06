#!/usr/bin/env python3
"""Offline Smith-GGX BRDF LUT generator (split-sum, 512x512, 1024 samples per texel).

Usage: uv run python scripts/gen_brdf_lut.py
Output: src/manifoldx/assets/ibl/brdf_lut.npy  — (512, 512, 2) float32
"""
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent.parent / "src/manifoldx/assets/ibl/brdf_lut.npy"
SIZE = 512
SAMPLES = 1024

rng = np.random.default_rng(0)


def geometry_schlick_ggx(ndotv, roughness):
    k = (roughness * roughness) / 2.0
    return ndotv / (ndotv * (1 - k) + k)


def geometry_smith(ndotv, ndotl, roughness):
    return geometry_schlick_ggx(ndotv, roughness) * geometry_schlick_ggx(ndotl, roughness)


lut = np.zeros((SIZE, SIZE, 2), dtype=np.float32)

for j in range(SIZE):
    roughness = (j + 0.5) / SIZE
    a = roughness * roughness
    xi1 = rng.random(SAMPLES).astype(np.float32)
    xi2 = rng.random(SAMPLES).astype(np.float32)
    cos_theta = np.sqrt((1 - xi2) / np.maximum(1 - (1 - a**2) * xi2, 1e-8))
    sin_theta = np.sqrt(np.maximum(1 - cos_theta**2, 0))
    phi = 2 * np.pi * xi1
    H = np.stack([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta], axis=-1)

    for i in range(SIZE):
        NdotV = (i + 0.5) / SIZE
        V = np.array([np.sqrt(1 - NdotV**2), 0.0, NdotV], dtype=np.float32)
        VdotH = np.clip((V[None, :] * H).sum(axis=-1), 0, 1)
        L = 2 * VdotH[:, None] * H - V[None, :]
        NdotL = np.clip(L[:, 2], 0, 1)
        NdotH = np.clip(H[:, 2], 0, 1)
        mask = NdotL > 0
        if mask.sum() == 0:
            continue
        G = geometry_smith(NdotV, NdotL[mask], roughness)
        G_vis = G * VdotH[mask] / np.maximum(NdotH[mask] * NdotV, 1e-8)
        Fc = (1 - VdotH[mask]) ** 5
        lut[i, j, 0] += (G_vis * (1 - Fc)).sum() / SAMPLES
        lut[i, j, 1] += (G_vis * Fc).sum() / SAMPLES

lut = np.clip(lut, 0, 1)
OUT.parent.mkdir(parents=True, exist_ok=True)
np.save(OUT, lut)
print(f"Saved BRDF LUT to {OUT}  shape={lut.shape}")
