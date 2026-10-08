"""Texture loading and GPU upload for the mesh-PBR path.

A TextureHandle is the public reference users hand to materials
(e.g. StandardMaterial(albedo_map=handle)). The engine's
TextureRegistry owns the underlying GPU resources.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import wgpu


class TextureSizeError(ValueError):
    """Image is larger than the device's max_texture_dimension_2d limit."""


@dataclass
class TextureHandle:
    id: int
    texture: Any            # wgpu.Texture
    view: Any               # wgpu.TextureView
    sampler: Any            # wgpu.Sampler
    size: tuple[int, int]   # (width, height) in pixels


class TextureRegistry:
    """Owns GPU texture + sampler resources for the engine lifetime."""

    def __init__(self) -> None:
        self._handles: Dict[int, TextureHandle] = {}
        self._next_id = 1

    def add(self, handle: TextureHandle) -> None:
        self._handles[handle.id] = handle

    def alloc_id(self) -> int:
        new_id = self._next_id
        self._next_id += 1
        return new_id


MAX_ANISOTROPY = 8


def _mip_chain(img):
    """The image and its successive halvings down to 1x1, as RGBA uint8 arrays.

    Each level is a box filter of the previous one, done on the sRGB bytes; averaging
    in linear light would be more exact and costs a float pass per level for a
    difference that does not show on ground textures."""
    from PIL import Image

    levels = [np.asarray(img, dtype=np.uint8)]
    w, h = img.size
    while w > 1 or h > 1:
        w, h = max(1, w // 2), max(1, h // 2)
        img = img.resize((w, h), Image.Resampling.BOX)
        levels.append(np.asarray(img, dtype=np.uint8))
    return levels


def load_texture(engine, path: str | Path) -> TextureHandle:
    """Decode an image file with Pillow, upload to GPU as Rgba8UnormSrgb with a full
    mip chain, sampled trilinearly with anisotropic filtering and repeat addressing.

    Args:
        engine: the manifoldx Engine (must have a device initialized).
        path: filesystem path to a PNG / JPEG / any Pillow-supported format.

    Returns:
        A TextureHandle the caller passes to material constructors.
    """
    from PIL import Image

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    img = Image.open(p).convert("RGBA")
    w, h = img.size

    device = getattr(engine, "_device", None)
    if device is None:
        raise RuntimeError(
            "engine has no device yet; initialize the canvas before load_texture(...)"
        )

    limits = getattr(device, "limits", {}) or {}
    max_dim = limits.get("max-texture-dimension-2d", 8192) if isinstance(limits, dict) else 8192
    if w > max_dim or h > max_dim:
        raise TextureSizeError(
            f"image is {w}x{h}, device max_texture_dimension_2d is {max_dim}"
        )

    levels = _mip_chain(img)
    texture = device.create_texture(
        size=(w, h, 1),
        format=wgpu.TextureFormat.rgba8unorm_srgb,
        usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        mip_level_count=len(levels),
        sample_count=1,
    )
    for i, level in enumerate(levels):
        lh, lw = level.shape[:2]
        device.queue.write_texture(
            {"texture": texture, "mip_level": i, "origin": (0, 0, 0)},
            level.tobytes(),
            {"offset": 0, "bytes_per_row": lw * 4, "rows_per_image": lh},
            (lw, lh, 1),
        )
    view = texture.create_view()
    sampler = device.create_sampler(
        mag_filter=wgpu.FilterMode.linear,
        min_filter=wgpu.FilterMode.linear,
        mipmap_filter=wgpu.MipmapFilterMode.linear,
        address_mode_u=wgpu.AddressMode.repeat,
        address_mode_v=wgpu.AddressMode.repeat,
        max_anisotropy=MAX_ANISOTROPY,
    )

    registry = engine._texture_registry
    handle = TextureHandle(
        id=registry.alloc_id(), texture=texture, view=view, sampler=sampler,
        size=(w, h),
    )
    registry.add(handle)
    return handle
