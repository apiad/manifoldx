"""Image-Based Lighting: EnvironmentMap class, builders, and CPU precomputation."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class EnvironmentMap:
    """Equirectangular float32 radiance data (H, W, 3), linear colour space.

    Build via class methods; pass to engine.set_environment().
    Precomputed GPU resources are cached on the object after first upload.
    """

    def __init__(self, data: np.ndarray, intensity: float = 1.0, show_skybox: bool = False):
        assert data.ndim == 3 and data.shape[2] == 3, "data must be (H, W, 3)"
        self.data = data.astype(np.float32)
        self.intensity = intensity
        self.show_skybox = show_skybox

        # Lazily populated by _precompute()
        self._irradiance: Optional[np.ndarray] = None   # (6, 64, 64, 4) float16
        self._prefiltered: Optional[list] = None         # list of 8 arrays (6, S, S, 4) float16
        self._computed: bool = False

    @classmethod
    def from_color(cls, rgb: Tuple[float, float, float]) -> "EnvironmentMap":
        """Uniform ambient — same radiance from all directions."""
        data = np.full((32, 64, 3), rgb, dtype=np.float32)
        return cls(data=data)

    @classmethod
    def from_sky(
        cls,
        zenith: Tuple[float, float, float],
        horizon: Tuple[float, float, float],
        ground: Tuple[float, float, float],
        height: int = 64,
        width: int = 128,
    ) -> "EnvironmentMap":
        """Procedural sky gradient: zenith (top) → horizon (middle) → ground (bottom)."""
        lats = np.linspace(np.pi / 2, -np.pi / 2, height)
        data = np.zeros((height, width, 3), dtype=np.float32)
        z = np.array(zenith, dtype=np.float32)
        h = np.array(horizon, dtype=np.float32)
        g = np.array(ground, dtype=np.float32)
        for i, lat in enumerate(lats):
            t = float(np.sin(lat))  # 1 at zenith, 0 at horizon, -1 at ground
            if t >= 0:
                data[i] = z * t + h * (1.0 - t)
            else:
                data[i] = h * (1.0 + t) + g * (-t)
        return cls(data=data)

    @classmethod
    def from_image(cls, path: str, exposure: float = 1.0) -> "EnvironmentMap":
        """Load any Pillow-supported image (JPEG/PNG) as equirectangular env map.

        Input is treated as sRGB and linearised before use.
        """
        from PIL import Image

        img = Image.open(path).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        # sRGB → linear
        arr = np.where(arr <= 0.04045, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)
        arr *= exposure
        return cls(data=arr)

    @classmethod
    def from_hdr(cls, path: str) -> "EnvironmentMap":
        """Load a Radiance RGBE (.hdr) file and return linear-float32 data."""
        data = _decode_rgbe(path)
        return cls(data=data)

    def _precompute(self):
        """Compute irradiance cubemap + prefiltered mip chain. Cached after first call."""
        if self._computed:
            return
        cube = _equirect_to_cubemap(self.data, face_size=128)
        self._irradiance = _compute_irradiance(cube, out_size=64, samples=256)
        self._prefiltered = []
        for mip in range(8):
            roughness = mip / 7.0
            size = max(1, 128 >> mip)
            self._prefiltered.append(
                _compute_prefiltered(cube, roughness=roughness, out_size=size, samples=512)
            )
        self._computed = True


# ---------------------------------------------------------------------------
# Built-in presets
# ---------------------------------------------------------------------------

PRESETS = {
    "studio": lambda: EnvironmentMap.from_sky(
        zenith=(0.6, 0.6, 0.65),
        horizon=(0.5, 0.5, 0.55),
        ground=(0.1, 0.1, 0.1),
    ),
    "sky": lambda: EnvironmentMap.from_sky(
        zenith=(0.05, 0.15, 0.6),
        horizon=(0.5, 0.65, 0.9),
        ground=(0.05, 0.05, 0.05),
    ),
    "neutral": lambda: EnvironmentMap.from_color((0.3, 0.3, 0.3)),
    "dark": lambda: EnvironmentMap.from_color((0.02, 0.02, 0.02)),
}


# ---------------------------------------------------------------------------
# RGBE decoder
# ---------------------------------------------------------------------------

def _decode_rgbe(path: str) -> np.ndarray:
    """Decode a Radiance RGBE (.hdr) file to float32 (H, W, 3), linear."""
    with open(path, "rb") as f:
        raw = f.read()

    # Parse header: skip to blank line
    pos = 0
    while True:
        nl = raw.index(b"\n", pos)
        line = raw[pos:nl]
        pos = nl + 1
        if line == b"":
            break

    # Size line: -Y H +X W
    nl = raw.index(b"\n", pos)
    size_line = raw[pos:nl].decode("ascii")
    pos = nl + 1
    parts = size_line.split()
    H, W = int(parts[1]), int(parts[3])

    # Decode scanlines
    result = np.zeros((H, W, 4), dtype=np.uint8)
    for row in range(H):
        marker = raw[pos: pos + 4]
        pos += 4
        if marker[0] == 2 and marker[1] == 2:
            # New RLE: each channel encoded separately
            scanline_w = (marker[2] << 8) | marker[3]
            for ch in range(4):
                x = 0
                while x < scanline_w:
                    code = raw[pos]
                    pos += 1
                    if code > 128:
                        count = code - 128
                        val = raw[pos]
                        pos += 1
                        result[row, x: x + count, ch] = val
                        x += count
                    else:
                        result[row, x: x + code, ch] = list(raw[pos: pos + code])
                        pos += code
                        x += code
        else:
            # Uncompressed / old RLE
            chunk = np.frombuffer(raw[pos - 4: pos - 4 + W * 4], dtype=np.uint8)
            result[row] = chunk.reshape(W, 4)
            pos += (W - 1) * 4

    # RGBE → float32 RGB
    R = result[..., 0].astype(np.float32)
    G = result[..., 1].astype(np.float32)
    B = result[..., 2].astype(np.float32)
    E = result[..., 3].astype(np.float32)
    scale = np.where(E > 0, np.ldexp(1.0, (E.astype(int) - 128 - 8)), 0.0)
    rgb = np.stack([R * scale, G * scale, B * scale], axis=-1)
    return rgb.astype(np.float32)


# ---------------------------------------------------------------------------
# Cubemap utilities
# ---------------------------------------------------------------------------

def _face_directions(face_size: int):
    """Return list of 6 (face_size, face_size, 3) world-direction arrays."""
    coords = np.linspace(-1.0, 1.0, face_size, dtype=np.float32)
    u, v = np.meshgrid(coords, coords)
    return [
        np.stack([ np.ones_like(u), -v, -u], axis=-1),  # +X
        np.stack([-np.ones_like(u), -v,  u], axis=-1),  # -X
        np.stack([ u,  np.ones_like(v),  v], axis=-1),  # +Y
        np.stack([ u, -np.ones_like(v), -v], axis=-1),  # -Y
        np.stack([ u, -v,  np.ones_like(u)], axis=-1),  # +Z
        np.stack([-u, -v, -np.ones_like(u)], axis=-1),  # -Z
    ]


def _equirect_to_cubemap(equirect: np.ndarray, face_size: int) -> np.ndarray:
    """Sample equirectangular image into 6-face cubemap.

    equirect: (H, W, 3) float32
    Returns:  (6, face_size, face_size, 3) float32
    """
    H, W = equirect.shape[:2]
    cube = np.zeros((6, face_size, face_size, 3), dtype=np.float32)
    for fi, dirs in enumerate(_face_directions(face_size)):
        norms = np.linalg.norm(dirs, axis=-1, keepdims=True).clip(min=1e-8)
        d = dirs / norms
        phi = np.arctan2(d[..., 2], d[..., 0])
        theta = np.arcsin(np.clip(d[..., 1], -1.0, 1.0))
        eu = (phi / (2 * np.pi) + 0.5) * (W - 1)
        ev = (0.5 - theta / np.pi) * (H - 1)
        eu0 = np.clip(np.floor(eu).astype(int), 0, W - 1)
        eu1 = np.clip(eu0 + 1, 0, W - 1)
        ev0 = np.clip(np.floor(ev).astype(int), 0, H - 1)
        ev1 = np.clip(ev0 + 1, 0, H - 1)
        wu, wv = (eu - eu0)[..., None], (ev - ev0)[..., None]
        cube[fi] = (
            equirect[ev0, eu0] * (1 - wu) * (1 - wv)
            + equirect[ev0, eu1] * wu * (1 - wv)
            + equirect[ev1, eu0] * (1 - wu) * wv
            + equirect[ev1, eu1] * wu * wv
        )
    return cube


def _sample_cube(cube: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """Sample (6, H, W, C) cubemap at N directions (N, 3). Returns (N, C)."""
    ax = np.abs(dirs)
    dom = np.argmax(ax, axis=-1)
    sign = np.sign(dirs[np.arange(len(dirs)), dom])
    face_map = np.where(
        dom == 0,
        np.where(sign > 0, 0, 1),
        np.where(dom == 1, np.where(sign > 0, 2, 3), np.where(sign > 0, 4, 5)),
    )
    dom_val = ax[np.arange(len(dirs)), dom].clip(min=1e-8)
    u_sel = np.where(
        dom == 0,
        np.where(sign > 0, -dirs[:, 2], dirs[:, 2]),
        np.where(dom == 1, dirs[:, 0], np.where(sign > 0, dirs[:, 0], -dirs[:, 0])),
    ) / dom_val
    v_sel = np.where(
        dom == 0,
        -dirs[:, 1],
        np.where(dom == 1, np.where(sign > 0, dirs[:, 2], -dirs[:, 2]), -dirs[:, 1]),
    ) / dom_val
    S = cube.shape[1]
    pu = np.clip(((u_sel + 1) / 2) * (S - 1), 0, S - 1).astype(int)
    pv = np.clip(((v_sel + 1) / 2) * (S - 1), 0, S - 1).astype(int)
    return cube[face_map, pv, pu]


# ---------------------------------------------------------------------------
# Precomputation
# ---------------------------------------------------------------------------

def _compute_irradiance(cube: np.ndarray, out_size: int = 64, samples: int = 256) -> np.ndarray:
    """Cosine-weighted hemisphere integration for diffuse irradiance.

    cube: (6, S, S, 3) float32
    Returns: (6, out_size, out_size, 4) float16
    """
    irr = np.zeros((6, out_size, out_size, 3), dtype=np.float32)
    rng = np.random.default_rng(0)
    phi_s = rng.uniform(0.0, 2 * np.pi, samples).astype(np.float32)
    cos_t = rng.uniform(0.0, 1.0, samples).astype(np.float32)
    sin_t = np.sqrt(1.0 - cos_t ** 2)
    ts = np.stack([sin_t * np.cos(phi_s), sin_t * np.sin(phi_s), cos_t], axis=-1)

    for fi, dirs in enumerate(_face_directions(out_size)):
        norms = np.linalg.norm(dirs, axis=-1, keepdims=True).clip(min=1e-8)
        N_all = (dirs / norms).reshape(-1, 3)
        is_up = np.abs(N_all[:, 1]) > 0.99
        ref = np.where(is_up[:, None], np.array([[1., 0., 0.]]), np.array([[0., 1., 0.]]))
        right = np.cross(N_all, ref)
        right /= np.linalg.norm(right, axis=-1, keepdims=True).clip(min=1e-8)
        up2 = np.cross(N_all, right)
        world = (
            ts[None, :, 0:1] * right[:, None, :]
            + ts[None, :, 1:2] * up2[:, None, :]
            + ts[None, :, 2:3] * N_all[:, None, :]
        ).reshape(-1, 3)
        colors = _sample_cube(cube, world).reshape(-1, samples, 3)
        irr[fi] = colors.mean(axis=1).reshape(out_size, out_size, 3) * np.pi

    out = np.concatenate([irr, np.ones((*irr.shape[:3], 1), dtype=np.float32)], axis=-1)
    return out.astype(np.float16)


def _ggx_importance_sample(roughness: float, samples: int, rng) -> np.ndarray:
    """GGX-importance-sampled half-vectors in tangent space. Returns (samples, 3)."""
    a = roughness * roughness
    xi1 = rng.random(samples).astype(np.float32)
    xi2 = rng.random(samples).astype(np.float32)
    phi = 2 * np.pi * xi1
    cos_theta = np.sqrt((1 - xi2) / np.maximum(1 - (1 - a ** 2) * xi2, 1e-8))
    sin_theta = np.sqrt(np.maximum(1 - cos_theta ** 2, 0))
    return np.stack([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta], axis=-1).astype(np.float32)


def _compute_prefiltered(cube: np.ndarray, roughness: float, out_size: int, samples: int = 512) -> np.ndarray:
    """GGX-importance-sampled prefiltered env map at given roughness.

    cube: (6, S, S, 3) float32
    Returns: (6, out_size, out_size, 4) float16
    """
    pf = np.zeros((6, out_size, out_size, 3), dtype=np.float32)
    rng = np.random.default_rng(42)
    H_ts = _ggx_importance_sample(max(roughness, 0.01), samples, rng)

    for fi, dirs in enumerate(_face_directions(out_size)):
        norms = np.linalg.norm(dirs, axis=-1, keepdims=True).clip(min=1e-8)
        N_all = (dirs / norms).reshape(-1, 3)
        is_up = np.abs(N_all[:, 1]) > 0.99
        ref = np.where(is_up[:, None], np.array([[1., 0., 0.]]), np.array([[0., 1., 0.]]))
        right = np.cross(N_all, ref)
        right /= np.linalg.norm(right, axis=-1, keepdims=True).clip(min=1e-8)
        up2 = np.cross(N_all, right)
        H_world = (
            H_ts[None, :, 0:1] * right[:, None, :]
            + H_ts[None, :, 1:2] * up2[:, None, :]
            + H_ts[None, :, 2:3] * N_all[:, None, :]
        )
        N_exp = N_all[:, None, :]
        NdotH = np.clip((H_world * N_exp).sum(axis=-1, keepdims=True), 0, 1)
        L = 2 * NdotH * H_world - N_exp
        L_flat = L.reshape(-1, 3)
        colors = _sample_cube(cube, L_flat).reshape(-1, samples, 3)
        NdotL = np.clip((L * N_exp).sum(axis=-1, keepdims=True), 0, 1)
        weighted = (colors * NdotL).sum(axis=1) / NdotL.sum(axis=1).clip(min=1e-4)
        pf[fi] = weighted.reshape(out_size, out_size, 3)

    out = np.concatenate([pf, np.ones((*pf.shape[:3], 1), dtype=np.float32)], axis=-1)
    return out.astype(np.float16)


# ---------------------------------------------------------------------------
# BRDF LUT loader
# ---------------------------------------------------------------------------

def load_brdf_lut() -> np.ndarray:
    """Load pre-baked Smith-GGX BRDF LUT. Returns (512, 512, 2) float32."""
    from pathlib import Path
    path = Path(__file__).parent / "assets" / "ibl" / "brdf_lut.npy"
    return np.load(path).astype(np.float32)
