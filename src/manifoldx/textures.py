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


def load_texture(engine, path: str | Path) -> TextureHandle:
    """Decode an image file with Pillow, upload to GPU as Rgba8UnormSrgb.

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
    arr = np.asarray(img, dtype=np.uint8)   # (H, W, 4)
    h, w = arr.shape[:2]

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

    texture = device.create_texture(
        size=(w, h, 1),
        format=wgpu.TextureFormat.rgba8unorm_srgb,
        usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        mip_level_count=1,
        sample_count=1,
    )
    device.queue.write_texture(
        {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
        arr.tobytes(),
        {"offset": 0, "bytes_per_row": w * 4, "rows_per_image": h},
        (w, h, 1),
    )
    view = texture.create_view()
    sampler = device.create_sampler(
        mag_filter=wgpu.FilterMode.linear,
        min_filter=wgpu.FilterMode.linear,
        address_mode_u=wgpu.AddressMode.repeat,
        address_mode_v=wgpu.AddressMode.repeat,
    )

    registry = engine._texture_registry
    handle = TextureHandle(
        id=registry.alloc_id(), texture=texture, view=view, sampler=sampler,
        size=(w, h),
    )
    registry.add(handle)
    return handle
