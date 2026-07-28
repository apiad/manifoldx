"""Animated volumetric smoke — Perlin FBM advected through a sliding window.

The smoke field comes from a 4-octave 3D Perlin FBM precomputed once on a
96³ source lattice. Each frame we extract a 64³ trilinear-interpolated
window starting at a smoothly-drifting fractional offset (rising +
drifting horizontally, like smoke caught in a light breeze) and upload
it to the GPU via `engine.update_volume`. Camera orbits and dollies
gently, so you see the wisps from changing angles. Frame work is one
FFT-free numpy slice extract + a handful of `lerp`s — comfortably
realtime at 64³.

Run interactive:
    uv run python examples/smoke_demo.py

Render to MP4:
    uv run python examples/smoke_demo.py --render --duration 6 --fps 30 \\
        --output /tmp/smoke.mp4
"""

import math

import numpy as np

import manifoldx as mx
from manifoldx.components import Material, Transform
from manifoldx.systems import Query
from manifoldx.viz import Volume, VolumeMaterial


# ── Perlin FBM (vectorized numpy) ────────────────────────────────────────────
# 12 standard Perlin gradient directions (Improved Noise, Perlin 2002).
_GRAD3 = np.array(
    [
        (1, 1, 0), (-1, 1, 0), (1, -1, 0), (-1, -1, 0),
        (1, 0, 1), (-1, 0, 1), (1, 0, -1), (-1, 0, -1),
        (0, 1, 1), (0, -1, 1), (0, 1, -1), (0, -1, -1),
    ],
    dtype=np.float32,
)


def _make_perm_table(seed: int) -> np.ndarray:
    """Doubled-up 256-entry permutation table for cheap modular indexing."""
    rng = np.random.default_rng(seed)
    perm = np.arange(256, dtype=np.int32)
    rng.shuffle(perm)
    return np.concatenate([perm, perm])  # length 512


def _fade(t: np.ndarray) -> np.ndarray:
    # Perlin's quintic fade — C² continuous; smoother than the cubic Hermite.
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def perlin3_tileable(
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
    perm: np.ndarray, period: int,
) -> np.ndarray:
    """3D Perlin gradient noise that tiles seamlessly with period `period`
    on the lattice. Lattice cell coordinates are reduced modulo `period`
    before they index `perm`, so the noise field is identical at lattice
    positions `i` and `i + period`.
    """
    xi = np.floor(x).astype(np.int32) % period
    yi = np.floor(y).astype(np.int32) % period
    zi = np.floor(z).astype(np.int32) % period
    xi1 = (xi + 1) % period
    yi1 = (yi + 1) % period
    zi1 = (zi + 1) % period
    xf = x - np.floor(x)
    yf = y - np.floor(y)
    zf = z - np.floor(z)
    u = _fade(xf)
    v = _fade(yf)
    w = _fade(zf)

    a   = perm[xi  & 255]
    a1  = perm[xi1 & 255]
    aa  = perm[(a  + (yi  & 255)) & 255]
    aa1 = perm[(a  + (yi1 & 255)) & 255]
    a1a = perm[(a1 + (yi  & 255)) & 255]
    a1a1 = perm[(a1 + (yi1 & 255)) & 255]

    h000 = perm[(aa   + (zi  & 255)) & 255] % 12
    h001 = perm[(aa   + (zi1 & 255)) & 255] % 12
    h010 = perm[(aa1  + (zi  & 255)) & 255] % 12
    h011 = perm[(aa1  + (zi1 & 255)) & 255] % 12
    h100 = perm[(a1a  + (zi  & 255)) & 255] % 12
    h101 = perm[(a1a  + (zi1 & 255)) & 255] % 12
    h110 = perm[(a1a1 + (zi  & 255)) & 255] % 12
    h111 = perm[(a1a1 + (zi1 & 255)) & 255] % 12

    def dot(h: np.ndarray, dx: np.ndarray, dy: np.ndarray, dz: np.ndarray) -> np.ndarray:
        g = _GRAD3[h]
        return g[..., 0] * dx + g[..., 1] * dy + g[..., 2] * dz

    g000 = dot(h000, xf,       yf,       zf)
    g001 = dot(h001, xf,       yf,       zf - 1)
    g010 = dot(h010, xf,       yf - 1,   zf)
    g011 = dot(h011, xf,       yf - 1,   zf - 1)
    g100 = dot(h100, xf - 1,   yf,       zf)
    g101 = dot(h101, xf - 1,   yf,       zf - 1)
    g110 = dot(h110, xf - 1,   yf - 1,   zf)
    g111 = dot(h111, xf - 1,   yf - 1,   zf - 1)

    x00 = g000 + u * (g100 - g000)
    x01 = g001 + u * (g101 - g001)
    x10 = g010 + u * (g110 - g010)
    x11 = g011 + u * (g111 - g011)
    y0 = x00 + v * (x10 - x00)
    y1 = x01 + v * (x11 - x01)
    return (y0 + w * (y1 - y0)).astype(np.float32)


def perlin_fbm3_tileable(
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
    perm: np.ndarray, octaves: int = 4, base_period: int = 3,
    persistence: float = 0.5, lacunarity: int = 2,
) -> np.ndarray:
    """Tileable FBM of 3D Perlin noise.

    Each octave's lattice has period `base_period * lacunarity**k`, so the
    sum is intrinsically periodic with period `base_period * lacunarity**(octaves-1)`
    in lattice coordinates. Caller is responsible for choosing source
    coordinates that step exactly `base_period` units across the source
    volume so the volume tiles cleanly.
    """
    total = np.zeros_like(x)
    amplitude = 1.0
    frequency = 1
    norm = 0.0
    for _ in range(octaves):
        period = base_period * frequency
        total += amplitude * perlin3_tileable(
            x * frequency, y * frequency, z * frequency, perm, period
        )
        norm += amplitude
        amplitude *= persistence
        frequency *= lacunarity
    return (total / norm).astype(np.float32)


# ── Pre-compute the smoke field ──────────────────────────────────────────────
N_WINDOW = 64        # uploaded volume size
N_SOURCE = 96        # pre-computed source size; drift window slides through it
OCTAVES = 4
BASE_PERIOD = 3      # tile period (in lattice cells) at octave 0
# At octave k, the lattice has BASE_PERIOD * 2**k cells per tile. The largest
# octave is BASE_PERIOD * 2**(OCTAVES-1) = 24 cells; N_SOURCE=96 must be an
# integer multiple of that for clean tiling. 96 / 24 = 4. ✓

print(f"Pre-computing {N_SOURCE}³ tileable Perlin FBM ({OCTAVES} octaves)…")
# `endpoint=False` is critical: voxel N_SOURCE would be at lattice position
# BASE_PERIOD = 3.0, which equals lattice 0.0 in the periodic noise. So
# voxels [0, N_SOURCE-1] cover one full tile without duplicating endpoints.
xs = np.linspace(0.0, float(BASE_PERIOD), N_SOURCE, endpoint=False, dtype=np.float32)
SX, SY, SZ = np.meshgrid(xs, xs, xs, indexing="ij")
PERM = _make_perm_table(seed=42)
DENSITY_SOURCE = perlin_fbm3_tileable(
    SX, SY, SZ, PERM, octaves=OCTAVES, base_period=BASE_PERIOD, persistence=0.55,
)
# Normalize to [0, 1]
DENSITY_SOURCE = (DENSITY_SOURCE - DENSITY_SOURCE.min()) / (
    np.ptp(DENSITY_SOURCE) + 1e-9
)
DENSITY_SOURCE = DENSITY_SOURCE.astype(np.float32)
print(f"  source range: [{DENSITY_SOURCE.min():.3f}, {DENSITY_SOURCE.max():.3f}]")

# Pad with `mode='wrap'` by N_WINDOW+1 cells so plain integer slicing into
# DENSITY_PADDED works for any window starting at offset [0, N_SOURCE-1] —
# crossing the periodic boundary is now a continuous slice, not a hard cut.
PAD = N_WINDOW + 1
DENSITY_PADDED = np.pad(
    DENSITY_SOURCE, ((0, PAD), (0, PAD), (0, PAD)), mode="wrap"
).astype(np.float32)

# Soft envelope so the smoke fades at the box edges instead of clipping flat.
# Anisotropic: tighter falloff in y so the column rises and dissipates,
# wider in xz so the body reads as a billowing volume rather than a sphere.
xs_local = np.linspace(-1.0, 1.0, N_WINDOW, dtype=np.float32)
LX, LY, LZ = np.meshgrid(xs_local, xs_local, xs_local, indexing="ij")
ENVELOPE = np.exp(-(LX * LX * 0.7 + LY * LY * 0.4 + LZ * LZ * 0.7)).astype(np.float32)


def extract_window(fx: float, fy: float, fz: float) -> np.ndarray:
    """Trilinear-interpolated 64³ slice of the smoke field at (fx, fy, fz).

    Reads from `DENSITY_PADDED` (= `DENSITY_SOURCE` periodically extended
    by N_WINDOW+1 voxels), so the integer base offset can be anywhere in
    [0, N_SOURCE) and the slice never bumps into the array bound — the
    periodic wrap is continuous. Cost: 8 numpy slices + 7 fused lerps.
    """
    fx_w = fx - N_SOURCE * np.floor(fx / N_SOURCE)
    fy_w = fy - N_SOURCE * np.floor(fy / N_SOURCE)
    fz_w = fz - N_SOURCE * np.floor(fz / N_SOURCE)
    ix = int(fx_w); iy = int(fy_w); iz = int(fz_w)
    ax = np.float32(fx_w - ix); ay = np.float32(fy_w - iy); az = np.float32(fz_w - iz)

    s = DENSITY_PADDED
    n = N_WINDOW
    v000 = s[ix     : ix + n,     iy     : iy + n,     iz     : iz + n]
    v001 = s[ix     : ix + n,     iy     : iy + n,     iz + 1 : iz + 1 + n]
    v010 = s[ix     : ix + n,     iy + 1 : iy + 1 + n, iz     : iz + n]
    v011 = s[ix     : ix + n,     iy + 1 : iy + 1 + n, iz + 1 : iz + 1 + n]
    v100 = s[ix + 1 : ix + 1 + n, iy     : iy + n,     iz     : iz + n]
    v101 = s[ix + 1 : ix + 1 + n, iy     : iy + n,     iz + 1 : iz + 1 + n]
    v110 = s[ix + 1 : ix + 1 + n, iy + 1 : iy + 1 + n, iz     : iz + n]
    v111 = s[ix + 1 : ix + 1 + n, iy + 1 : iy + 1 + n, iz + 1 : iz + 1 + n]

    x00 = v000 + ax * (v100 - v000)
    x01 = v001 + ax * (v101 - v001)
    x10 = v010 + ax * (v110 - v010)
    x11 = v011 + ax * (v111 - v011)
    y0 = x00 + ay * (x10 - x00)
    y1 = x01 + ay * (x11 - x01)
    return (y0 + az * (y1 - y0)).astype(np.float32)


# ── Engine setup ─────────────────────────────────────────────────────────────
engine = mx.Engine("Volumetric smoke — Perlin FBM, drifting + camera orbit")
engine.camera.fit(1.8, azimuth=35, elevation=12)

# Initial frame; @engine.system below replaces it every frame.
initial = (extract_window(0.0, 0.0, 0.0) * ENVELOPE).astype(np.float32)
vol_id = engine.register_volume(initial, name="smoke")
engine.spawn(
    Volume(volume_id=vol_id),
    Material(
        VolumeMaterial(
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            # Smoke transfer function: zero opacity below the wisp threshold,
            # rising through the mid-band, dense in the core. Tweaked by eye
            # against the FBM histogram (which clusters around 0.4–0.7 after
            # min-max rescaling).
            opacity_stops=[(0.0, 0.0), (0.25, 0.0), (0.45, 0.25), (0.70, 0.7), (1.0, 0.95)],
            density_scale=2.2,
            max_steps=384,
        )
    ),
    Transform(pos=(0, 0, 0), scale=(2.0, 2.0, 2.0)),
    n=1,
)


# ── Animation system ─────────────────────────────────────────────────────────
# Drift velocities through the source lattice (voxels per second).
# y-component is largest → smoke "rises"; x is a steady breeze; z is a
# slower swirl. The source is fully periodic, so any integer multiple of
# N_SOURCE is invisible — drift can run forever without seam.
_DRIFT = np.array([4.5, 9.0, 2.5], dtype=np.float32)


@engine.system
def evolve_smoke(query: Query[Volume], dt: float):
    """Per-frame: re-upload a fractionally-shifted noise window and orbit the camera."""
    t = engine.elapsed
    fx = _DRIFT[0] * t + 3.0 * math.sin(t * 0.4)
    fy = _DRIFT[1] * t
    fz = _DRIFT[2] * t + 4.0 * math.cos(t * 0.3)

    frame = (extract_window(fx, fy, fz) * ENVELOPE).astype(np.float32)
    engine.update_volume(vol_id, frame)

    # Camera: slow orbit, gentle dolly in/out so depth cues are visible.
    engine.camera.orbit(d_azimuth=dt * 18.0)
    target_dist = 1.8 + 0.35 * math.sin(t * 0.35)
    current_dist = engine.camera.get_distance()
    engine.camera.dolly(current_dist - target_dist)


if __name__ == "__main__":
    engine.cli()
